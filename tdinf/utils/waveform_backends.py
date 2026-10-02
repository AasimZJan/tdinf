import functools
import importlib
import numpy as np
from gwpy.timeseries import TimeSeries


@functools.lru_cache(maxsize=None)
def _load_module(module_name):
    """
    Import a module by name, caching the result for the life of the process.

    The first call performs the real import; later calls return the cached
    module. The module is deliberately NOT stored on backend objects, since
    module objects cannot be pickled.

    Parameters
    ----------
    module_name : str
        Name of the module to import (e.g. 'EOBRun_module').

    Returns
    -------
    module
        The imported module.

    Raises
    ------
    ImportError
        If the module is not installed in the current environment.
    """
    return importlib.import_module(module_name)


# Approximants that are NOT known to LAL.
EXTERNAL_APPROXIMANTS = {'TEOBResumS'}


def is_external_approximant(approx_name):
    """
    Check whether an approximant must be handled by an external backend.

    Parameters
    ----------
    approx_name : str
        Approximant name, as passed to `--approx`.

    Returns
    -------
    bool
        True if the approximant is handled by a backend in this module,
        False if it should go through the normal LAL path.
    """
    return approx_name in EXTERNAL_APPROXIMANTS


class TEOBResumSBackend:
    """
    Time-domain TEOBResumS backend, with precessing spins.

    Wraps `EOBRun_module.EOBRunPy` and returns waveforms in the same form as
    the LAL path of `WaveformManager`, so the rest of TDinf is unaware of the
    difference. The object stores only the module name, so it is safe to
    pickle.

    Parameters
    ----------
    warmup : bool, optional
        If True, make one throwaway waveform call at construction so that
        one-time initialization happens in the parent process, before emcee
        forks its workers. Default is True.
    """
    def __init__(self, warmup=True):
        # Only picklable state lives on self.
        self.module_name = 'EOBRun_module'  # VERIFY: name of your install's python module
        if warmup:
            self._warmup()

    def _warmup(self):
        """
        Do one throwaway call so one-time initialisation happens in the parent
        process, before emcee forks workers (they inherit it).
        If your install misbehaves with fork, set warmup=False.
        """
        try:
            self.get_hphc(30., 30., [0, 0, 0.], [0, 0, 0.], delta_t=1 / 2048.,
                          dist_mpc=100., f22_start=20., inclination=0., phi_ref=0.)
        except Exception as e:
            print('TEOBResumS warmup failed (continuing):', e)

    def get_hphc(self, m1_msun, m2_msun, chi1, chi2, delta_t, dist_mpc=1.,
                 f22_start=20., f_ref=None, inclination=0., phi_ref=0., **unused):
        """
        Generate plus and cross polarizations at geocenter with TEOBResumS.

        Parameters
        ----------
        m1_msun, m2_msun : float
            Component masses in solar masses, with m1_msun >= m2_msun
        chi1, chi2 : array_like
            Dimensionless spin vectors [x, y, z] of each component. All three
            components are passed to the model.
        delta_t : float
            Time spacing of the output in seconds (sets `srate_interp`).
        dist_mpc : float, optional
            Luminosity distance in Mpc. Default is 1.
        f22_start : float, optional
            Starting frequency of the (2,2) mode in Hz. Default is 20.
        f_ref : float, optional
            Accepted so the signature matches the LAL path, but NOT passed to
            TEOBResumS. Spins are interpreted in TEOBResumS's own frame, which
            may differ from the LAL frame when f_ref != f22_start.
        inclination : float, optional
            Inclination angle in radians. Default is 0.
        phi_ref : float, optional
            Reference phase in radians, in the LAL convention. Passed to the
            model as `coalescence_angle = pi/2 - phi_ref`.
        **unused
            Ignored. Lets callers pass LAL-path-only keywords (e.g. `NR_kws`).

        Returns
        -------
        hp, hc : gwpy.timeseries.TimeSeries
        """

        EOB = _load_module(self.module_name)
        pars = {
                'M'                  : m1_msun + m2_msun,
                'q'                  : m1_msun/m2_msun,
                'chi1x'              : chi1[0],
                'chi1y'              : chi1[1],
                'chi1z'              : chi1[2],
                'chi2x'              : chi2[0],
                'chi2y'              : chi2[1],
                'chi2z'              : chi2[2],
                'ecc'                : 0,
                'inclination'        : inclination,
                'coalescence_angle'  : np.pi / 2 - phi_ref,
                'srate_interp'       : 1./delta_t,
                'initial_frequency'  : f22_start,
                'distance'           : dist_mpc,
                'anomaly'            : 0,
                'domain'             : 0,
                'arg_out'            : "yes",
                'interp_uniform_grid': "yes",
                'use_geometric_units': "no",
                # spin_flx can be "EOB" or "PN" prescription. Currently suggested to use EOB as standard (use PN for GIOTTO like prescription)
                'spin_flx'           : "EOB",
                'spin_interp_domain' : 0,
                # This can be "QNMs" or "constant". Currently it looks like without QNMs enabled gives basically only prior for precessing events (even for low mass events)
                'ringdown_eulerangles': "QNMs",
                # Using (2,2), (2,1), (3,3), (4,4)
                'use_mode_lm'        : [1, 0, 4, 8],
                'output_hpc'         : "no"
            }
        
        try:
            t, hp, hc, hlm, dyn = EOB.EOBRunPy(pars)
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

        return TimeSeries(hp, t0=t0, dt=delta_t), TimeSeries(hc, t0=t0, dt=delta_t)


_BACKENDS = {
    'TEOBResumS': TEOBResumSBackend,
}


def make_backend(approx_name, **kwargs):
    """
    Construct the backend object for an external approximant.

    Parameters
    ----------
    approx_name : str
        Name of an approximant for which `is_external_approximant` is True.
    **kwargs
        Passed to the backend constructor (e.g. `warmup=False`).

    Returns
    -------
    TEOBResumSBackend
        The backend for `approx_name`.

    Raises
    ------
    KeyError
        If `approx_name` has no registered backend.
    """
    return _BACKENDS[approx_name](**kwargs)

