"""
Skeleton for non-LAL waveform backends in tdinf (e.g. TEOBResumS).

Design rules
------------
1. Heavy imports happen lazily, once per process, via `_load_module`. Nothing
   in tdinf's top-level import chain requires the external package, so LAL-only
   users are unaffected.
2. Backend objects hold ONLY picklable state (strings, floats). emcee's Pool
   pickles `likelihood_manager.get_log_posterior` (and so the whole manager)
   with its tasks, and module objects cannot be pickled -- never store the
   module on `self`.
3. A backend returns gwpy TimeSeries objects with t = 0 at the peak of the
   waveform amplitude, matching what `get_projected_waveform` expects from the
   LAL path (it does `hp.t0 = geocenter_time + hp.t0.value`).
4. On any waveform failure, return (np.nan, np.nan). The existing code already
   turns that into ln L = -inf.

!! The TEOBResumS parameter names below are from memory and MUST be checked
!! against the version you have installed (see the VERIFY comments).
"""
import functools
import importlib

import numpy as np
from gwpy.timeseries import TimeSeries


@functools.lru_cache(maxsize=None)
def _load_module(module_name):
    """Import once per process; later calls are a dict lookup."""
    return importlib.import_module(module_name)


def _check_importable(module_name, what):
    """
    Fail loudly at construction time if an external package is missing.
    Without this, a missing install would only surface inside a pool worker on
    the first likelihood call (or, worse, be swallowed as ln L = -inf).
    """
    try:
        _load_module(module_name)
    except ImportError as e:
        raise ImportError(
            f"{what} requires the python module '{module_name}', which could "
            f"not be imported ({e}). Is it installed in this environment?"
        ) from e


# Names routed to the GWSignal backend. EDIT THIS to the approximants you want
# to run through lalsimulation.gwsignal instead of SimInspiralChooseTDWaveform.
GWSIGNAL_APPROXIMANTS = {'SEOBNRv5HM'}


class TEOBResumSBackend:
    """Aligned-spin TEOBResumS time-domain backend."""

    def __init__(self, approx_name='TEOBResumS', warmup=True):
        # Only picklable state lives on self.
        self.approx_name = approx_name
        self.module_name = 'EOBRun_module'  # VERIFY: name of your install's python module
        _check_importable(self.module_name, 'Approximant TEOBResumS')
        if warmup:
            self._warmup()

    def _warmup(self):
        """
        Do one throwaway call so one-time initialisation happens in the parent
        process, before emcee forks workers (they inherit it).
        If your install misbehaves with fork, set warmup=False.
        """
        try:
            self.generate(30., 30., [0, 0, 0.], [0, 0, 0.], delta_t=1 / 2048.,
                          dist_mpc=100., f22_start=20., inclination=0., phi_ref=0.)
        except Exception as e:
            print('TEOBResumS warmup failed (continuing):', e)

    def generate(self, m1_msun, m2_msun, chi1, chi2, delta_t, dist_mpc=1.,
                 f22_start=20., f_ref=None, inclination=0., phi_ref=0., **unused):
        """
        Return (hp, hc) as gwpy TimeSeries with t = 0 at the amplitude peak,
        or (np.nan, np.nan) on failure.

        `chi1`, `chi2` are [x, y, z] lists as in the LAL path; only z is used.
        `f_ref` is ignored (aligned spins); it is accepted so the call
        signature matches the LAL path.
        """
        EOB = _load_module(self.module_name)

        pars = {
            'M': m1_msun + m2_msun,
            'q': m1_msun / m2_msun,             # VERIFY: TEOB expects q >= 1 (tdinf's q <= 1 is m2/m1; here m1 >= m2 so this is >= 1)
            'chi1': chi1[2],
            'chi2': chi2[2],
            'domain': 0,                        # VERIFY: 0 = time domain
            'use_geometric_units': 'no',        # VERIFY: want seconds / Hz / Mpc
            'initial_frequency': f22_start,     # VERIFY: Hz, start of the 22 mode
            'interp_uniform_grid': 'yes',       # VERIFY: output on uniform grid at the data rate,
            'srate_interp': 1. / delta_t,       #         so no extra resampling is needed
            'distance': dist_mpc,
            'inclination': inclination,
            'coalescence_angle': phi_ref,       # VERIFY: phase convention vs LAL phi_ref
            'output_hpc': 'no',                 # VERIFY: avoid writing files on every call
        }

        try:
            out = EOB.EOBRunPy(pars)
            t, hp, hc = out[0], out[1], out[2]  # VERIFY: return tuple layout
        except Exception as e:
            print('TEOBResumS failure:', e)
            return np.nan, np.nan

        t = np.asarray(t); hp = np.asarray(hp); hc = np.asarray(hc)
        if not (np.all(np.isfinite(hp)) and np.all(np.isfinite(hc))):
            return np.nan, np.nan

        # Put t = 0 at the amplitude peak, to match the LAL convention
        # (do not rely on TEOB's own time origin).
        t_peak = t[np.argmax(hp ** 2 + hc ** 2)]
        t0 = t[0] - t_peak

        # VERIFY sign convention of hc vs LAL by comparing to an aligned-spin
        # LAL approximant (e.g. SEOBNRv4) at the same parameters.
        return TimeSeries(hp, t0=t0, dt=delta_t), TimeSeries(hc, t0=t0, dt=delta_t)


@functools.lru_cache(maxsize=None)
def _get_gwsignal_generator(approx_name):
    """
    Build the gwsignal generator once per process (this is the expensive part:
    it can load model data / python packages). Kept at module level, NOT on
    `self`, because generator objects may not be picklable.
    """
    from lalsimulation.gwsignal.core import waveform as gws_wf   # VERIFY import path
    return gws_wf.gwsignal_get_waveform_generator(approx_name)    # VERIFY function name


class GWSignalBackend:
    """Any approximant run through lalsimulation.gwsignal."""

    def __init__(self, approx_name):
        self.approx_name = approx_name      # only a string lives on self
        # Build the generator now, in the parent process (before emcee forks):
        # this doubles as the availability check and warms the cache.
        try:
            _get_gwsignal_generator(self.approx_name)
        except Exception as e:
            raise RuntimeError(
                f"Could not build a gwsignal generator for '{self.approx_name}': {e}"
            ) from e

    def generate(self, m1_msun, m2_msun, chi1, chi2, delta_t, dist_mpc=1.,
                 f22_start=20., f_ref=20., inclination=0., phi_ref=0., **unused):
        """
        Return (hp, hc) as gwpy TimeSeries, or (np.nan, np.nan) on failure.
        chi1, chi2 are [x, y, z]; all three components are passed through.
        """
        import astropy.units as u
        from lalsimulation.gwsignal.core import waveform as gws_wf   # VERIFY (cheap: cached by Python)

        gen = _get_gwsignal_generator(self.approx_name)

        params = {                                                    # VERIFY key names for your LALSuite version
            'mass1': m1_msun * u.solMass,
            'mass2': m2_msun * u.solMass,
            'spin1x': chi1[0] * u.dimensionless_unscaled,
            'spin1y': chi1[1] * u.dimensionless_unscaled,
            'spin1z': chi1[2] * u.dimensionless_unscaled,
            'spin2x': chi2[0] * u.dimensionless_unscaled,
            'spin2y': chi2[1] * u.dimensionless_unscaled,
            'spin2z': chi2[2] * u.dimensionless_unscaled,
            'distance': dist_mpc * u.Mpc,
            'inclination': inclination * u.rad,
            'phi_ref': phi_ref * u.rad,
            'f22_start': f22_start * u.Hz,
            'f22_ref': f_ref * u.Hz,
            'deltaT': delta_t * u.s,
            'condition': 1,                                           # VERIFY: conditioning (taper/pad) on
        }

        try:
            hp, hc = gws_wf.GenerateTDWaveform(params, gen)
        except Exception as e:
            print('GWSignal failure:', e)
            return np.nan, np.nan

        # Re-wrap as plain gwpy TimeSeries (drops astropy units), keeping the
        # waveform's own time origin. VERIFY that t = 0 is at the peak, as in
        # the LAL path; if not, align to the amplitude peak as TEOBResumSBackend does.
        hp_v = np.asarray(hp.value); hc_v = np.asarray(hc.value)
        if not (np.all(np.isfinite(hp_v)) and np.all(np.isfinite(hc_v))):
            return np.nan, np.nan
        t0 = hp.t0.value
        return TimeSeries(hp_v, t0=t0, dt=delta_t), TimeSeries(hc_v, t0=t0, dt=delta_t)


def _backend_class_for(approx_name):
    """Which backend (if any) handles this approximant name."""
    if approx_name == 'TEOBResumS':
        return TEOBResumSBackend
    if approx_name in GWSIGNAL_APPROXIMANTS:
        return GWSignalBackend
    return None          # -> normal LAL path


def is_external_approximant(approx_name):
    return _backend_class_for(approx_name) is not None


def make_backend(approx_name, **kwargs):
    return _backend_class_for(approx_name)(approx_name=approx_name, **kwargs)
