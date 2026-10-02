import functools
import importlib
import numpy as np
from gwpy.timeseries import TimeSeries


@functools.lru_cache(maxsize=None)
def _load_module(module_name):
    """Import once per process; later calls are a dict lookup."""
    return importlib.import_module(module_name)


# Approximants that are NOT known to LAL.
EXTERNAL_APPROXIMANTS = {'TEOBResumS'}


def is_external_approximant(approx_name):
    return approx_name in EXTERNAL_APPROXIMANTS


class TEOBResumSBackend:
    """TEOBResumS time-domain backend."""

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
        Return (hp, hc) as gwpy TimeSeries with t = 0 at the amplitude peak,
        or (np.nan, np.nan) on failure.

        `chi1`, `chi2` are [x, y, z] lists as in the LAL path; only z is used.
        `f_ref` is ignored (aligned spins); it is accepted so the call
        signature matches the LAL path.
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

        # VERIFY sign convention of hc vs LAL by comparing to an aligned-spin
        # LAL approximant (e.g. SEOBNRv4) at the same parameters.
        return TimeSeries(hp, t0=t0, dt=delta_t), TimeSeries(hc, t0=t0, dt=delta_t)


_BACKENDS = {
    'TEOBResumS': TEOBResumSBackend,
}


def make_backend(approx_name, **kwargs):
    return _BACKENDS[approx_name](**kwargs)

