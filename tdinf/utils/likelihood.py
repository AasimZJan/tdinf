import numpy as np
import h5py
from scipy.linalg import solve_toeplitz
import lal
import lalsimulation as lalsim
import sys
from gwpy.timeseries import TimeSeries
import os
from . import detector_times_and_antenna_patterns as t_and_AP
from .spins_and_masses import m1m2_from_mtotq
from .misc import *

from .parameter import LogisticParameter, CartesianAngle, TrigLogisticParameter
from .preprocessing import get_ACF
from .whiten import whitenData
import astropy.units as u

from .waveform_backends import is_external_approximant, make_backend

try:
    from bilby.core.prior import Prior as _BilbyPrior
except ImportError:         # bilby is only needed by LnPriorManager.get_bilby_mcmc_prior_dict
    _BilbyPrior = object
from scipy.special import expit

def check_spin_settings_of_approx(approx_name):
    """
    Check the spin configuration supported by a given waveform approximant.

    Parameters:
    - approx_name (str): Name of the waveform approximant.

    Returns:
    - aligned_spins (bool): True if the approximant supports only aligned spins.
    - no_spins (bool): True if the approximant supports no spins.
    """
    
    aligned_spins = False
    no_spins = False
    # for external approximants
    if is_external_approximant(approx_name):
        return False, False  
    
    approx = lalsim.GetApproximantFromString(approx_name)

    if not lalsim.SimInspiralImplementedTDApproximants(approx):
        raise ValueError(f"ERROR: {approx_name} is not available as a time domain waveform")
    
    spin_enum = lalsim.SimInspiralGetSpinSupportFromApproximant(approx)

    if spin_enum == lalsim.SIM_INSPIRAL_PRECESSINGSPIN:
        aligned_spins = False
    elif spin_enum == lalsim.SIM_INSPIRAL_ALIGNEDSPIN:
        aligned_spins = True
    elif spin_enum == lalsim.SIM_INSPIRAL_SPINLESS:
        no_spins = True
    else:
        print("WARNING, UNSURE IF WAVEFORM HAS SPINS")
        
    return aligned_spins, no_spins


class LogisticParameterManager:
    """
    Manager for logistic and Cartesian parameter transformations.

    Handles:
    - Spin, mass, distance, inclination, phase, and sky location parameters
    - Conversion between logistic space and physical space
    - Cartesian angle transformations
    - Spin normalization
    - Fixed and sampled parameters
    """
    
    def __init__(self, vary_time=False, vary_skypos=False, eccentric_analysis=False, aligned_spin_analysis=False, **kwargs):

        # vary time and sky pos?
        self.vary_time = vary_time
        self.vary_skypos = vary_skypos

        #  where we will put parameters which are NOT sampled over
        self.fixed = {}
        
        # reference time
        self.reference_time = kwargs['geocenter_time']

        # get allowed spin settings of waveform model
        self.aligned_spins, self.no_spins = check_spin_settings_of_approx(kwargs['approx'])

        # check if the user has requested aligned-spin analysis; allow aligned-spin analysis for models that support generic spin configurations.
        # For non-LAL approximants, both are False by default
        if self.aligned_spins == False and self.no_spins == False:
            self.aligned_spins = aligned_spin_analysis

        # check if the user has requested eccentric analysis
        self.eccentric_analysis = eccentric_analysis

        # set up the logistic parameters which are always sampled over
        self.logistic_parameters = [
            LogisticParameter('total_mass', kwargs['mtot_lim'], None),
            LogisticParameter('mass_ratio', kwargs['q_lim'], None),
            LogisticParameter('luminosity_distance', kwargs['dist_lim'], None),
            TrigLogisticParameter('inclination', 'cos', [-1, 1], None)                            
        ]
        # set up the cartesian parameters which are always sampled over
        self.cartesian_angles = [CartesianAngle('phase')]
        
        # add spin magnitudes, if supported by waveform model
        if not self.no_spins:
            self.logistic_parameters.extend([
                LogisticParameter('spin1_magnitude', kwargs['chi_lim'], None),
                LogisticParameter('spin2_magnitude', kwargs['chi_lim'], None)
            ])
        # include eccentricity parameters if user has requested an eccentric analysis
        if self.eccentric_analysis:
            self.logistic_parameters.extend([
                            LogisticParameter('eccentricity', kwargs['ecc_lim'], None),
                            ])
            self.cartesian_angles.extend([
                            CartesianAngle('mean_anomaly', phase_offset=np.pi),
                        ])
        else:
            self.fixed['eccentricity'] = 0.0
            self.fixed['mean_anomaly'] = 0.0

        # add sky position, if we want to sample it
        if self.vary_skypos:
            # right ascension
            self.logistic_parameters.append(TrigLogisticParameter('declination', 'sin', [-1, 1], None))
            # declination
            self.cartesian_angles.extend([
                CartesianAngle('right_ascension', phase_offset=np.pi),
                CartesianAngle('polarization')
            ])
        # otherwise, add to fixed parameters
        else:
            self.fixed['right_ascension'] = kwargs['right_ascension']
            self.fixed['declination'] = kwargs['declination']
            self.fixed['polarization'] = kwargs['polarization']

        # add time, if we want to sample it
        if self.vary_time:
            self.sigma_time = kwargs['sigma_time']
        # otherwise, add to fixed parameters 
        else:
            self.fixed['geocenter_time'] = self.reference_time


        # generate list of ALL parameters we are sampling over
        self.sampled_keys = [p.logistic_name for p in self.logistic_parameters]
        for cartesian_angle in self.cartesian_angles:
            self.sampled_keys.extend([cartesian_angle.x_name, cartesian_angle.y_name])
        if not self.no_spins:
            # if aligned spins, only sample z-component
            if self.aligned_spins:
                self.sampled_keys.extend(['c1_z', 'c2_z'])
            # else, sample over all components (x,y,z)
            else:
                self.sampled_keys.extend(['c1_x', 'c1_y', 'c1_z','c2_x', 'c2_y', 'c2_z'])
        if self.vary_time:
            self.sampled_keys.append('geocenter_time')  

        # NUMBER of sampled parameters
        self.num_parameters = len(self.sampled_keys)

    def get_logistic_dict(self, x):
        """
        Convert array `x` of sampled values into a dictionary keyed by sampled 
        parameter names.
        """
        return {self.sampled_keys[i]: x[i] for i in range(self.num_parameters)}

    def get_physical_dict_from_logistic_dict(self, x_dict):
        """
        Convert a dictionary of logistic parameters into their physical values.
        """
        physicalDict = self.fixed.copy()

        physicalDict.update({
            transform.physical_name: transform.logistic_to_physical(
                x_dict[transform.logistic_name]
            )
            for transform in self.logistic_parameters
        })
        physicalDict.update({
            transform.physical_name: transform.cartesian_to_radian(
                x_dict[transform.x_name],x_dict[transform.y_name]
            ) 
            for transform in self.cartesian_angles
        })
        return physicalDict

    def get_physical_spins(self, spin_magnitude, c_x, c_y, c_z):
        """
        Normalize Cartesian spin components to the correct spin magnitude.
        """
        if self.no_spins:
            return 0, 0, 0

        chi_norm = spin_magnitude / np.sqrt(c_x ** 2 + c_y ** 2 + c_z ** 2)

        spin_x = c_x * chi_norm
        spin_y = c_y * chi_norm
        spin_z = c_z * chi_norm
        
        return spin_x, spin_y, spin_z

    def samp_to_phys(self, x):
        """
        Convert a sampled array `x` into a physical dictionary with spins normalized.
        """
        # Convert `x` from an array into a dict
        x_dict = self.get_logistic_dict(x)
        physical_dict = self.get_physical_dict_from_logistic_dict(x_dict)

        # Normalize spins
        for i in ['1', '2']:
            physical_dict[f'spin{i}_x'], physical_dict[f'spin{i}_y'], physical_dict[f'spin{i}_z'] = \
                self.get_physical_spins(
                    physical_dict.get(f'spin{i}_magnitude', 0),
                    x_dict.get(f'c{i}_x', 0),
                    x_dict.get(f'c{i}_y', 0),
                    x_dict.get(f'c{i}_z', 0)
                )
            
        if self.vary_time:
            physical_dict['geocenter_time'] = x_dict['geocenter_time']

        return physical_dict

    @staticmethod
    def physical_dict_to_waveform_dict(physical_dict):
        """
        Take in the physical dictionary and return one that has units for astropy
        """
        # Get component masses from total mass and mass ratio
        m1, m2 = m1m2_from_mtotq(physical_dict['total_mass'], physical_dict['mass_ratio'])
        
        # Add units to all parameters
        param_dict = {
            'mass1': m1 * u.Msun,
            'mass2': m2 * u.Msun,
            'spin1x': physical_dict['spin1_x'] * u.dimensionless_unscaled,
            'spin1y': physical_dict['spin1_y'] * u.dimensionless_unscaled,
            'spin1z': physical_dict['spin1_z'] * u.dimensionless_unscaled,
            'spin2x': physical_dict['spin2_x'] * u.dimensionless_unscaled,
            'spin2y': physical_dict['spin2_y'] * u.dimensionless_unscaled,
            'spin2z': physical_dict['spin2_z'] * u.dimensionless_unscaled,
            'phi_ref': physical_dict['phase'] * u.rad,
            'distance': physical_dict['luminosity_distance'] * u.Mpc,
            'inclination': physical_dict['inclination'] * u.rad,
        }
        
        return param_dict


class _LogisticCoordinatePrior(_BilbyPrior):
    """
    bilby prior of x = logit(u), u = (v - lower) / (upper - lower), for a physical parameter v whose density on [lower, upper] is
    given by the subclass through the density of u on (0, 1). The density of x is p(u) u (1 - u) with u = expit(x).
    """

    def __init__(self, physical_minimum, physical_maximum, name=None, latex_label=None, unit=None, boundary=None):
        super().__init__(name=name, latex_label=latex_label, unit=unit, minimum=-np.inf, maximum=np.inf, boundary=boundary)
        self.physical_minimum = float(physical_minimum)
        self.physical_maximum = float(physical_maximum)

    def _ln_density_u(self, u):       # ln p(u) of the normalised density on (0, 1)
        raise NotImplementedError

    def _cdf_u(self, u):
        raise NotImplementedError

    def _inverse_cdf_u(self, f):
        raise NotImplementedError

    def ln_prob(self, val):
        val = np.asarray(val, dtype=float)
        ln_u_one_minus_u = -np.abs(val) - 2.0 * np.log1p(np.exp(-np.abs(val)))          # ln[u (1 - u)], u = expit(val)
        return self._ln_density_u(expit(val)) + ln_u_one_minus_u

    def prob(self, val):
        return np.exp(self.ln_prob(val))

    def cdf(self, val):
        return self._cdf_u(expit(np.asarray(val, dtype=float)))

    def rescale(self, val):
        u = np.clip(self._inverse_cdf_u(np.asarray(val, dtype=float)), 1e-300, 1.0 - 1e-16)
        return np.log(u) - np.log1p(-u)


class LVKTotalMassCoordinate(_LogisticCoordinatePrior):
    """x_total_mass with the LVK prior: the total mass M in [a, b] has density proportional to M."""

    def _ln_density_u(self, u):
        a, b = self.physical_minimum, self.physical_maximum
        return np.log(a + (b - a) * u) - np.log(0.5 * (a + b))

    def _cdf_u(self, u):
        a, b = self.physical_minimum, self.physical_maximum
        return u * (2.0 * a + (b - a) * u) / (a + b)

    def _inverse_cdf_u(self, f):
        a, b = self.physical_minimum, self.physical_maximum
        return (-a + np.sqrt(a * a + (b - a) * (a + b) * f)) / (b - a)


class LVKMassRatioCoordinate(_LogisticCoordinatePrior):
    """x_mass_ratio with the LVK prior: the mass ratio q in [a, b] has density proportional to (1 + q)^-2."""

    def _ln_density_u(self, u):
        a, b = self.physical_minimum, self.physical_maximum
        return np.log((1.0 + a) * (1.0 + b)) - 2.0 * np.log1p(a + (b - a) * u)

    def _cdf_u(self, u):
        a, b = self.physical_minimum, self.physical_maximum
        q = a + (b - a) * u
        return (1.0 + b) * (q - a) / ((b - a) * (1.0 + q))

    def _inverse_cdf_u(self, f):
        a, b = self.physical_minimum, self.physical_maximum
        q = ((1.0 + b) * a + f * (b - a)) / ((1.0 + b) - f * (b - a))
        return (q - a) / (b - a)


class LVKMassRatioPrior(_BilbyPrior):
    """The mass ratio q itself (physical coordinates) with the LVK prior: density proportional to (1 + q)^-2 on [minimum, maximum]."""

    def __init__(self, minimum, maximum, name=None, latex_label=None, unit=None, boundary=None):
        super().__init__(name=name, latex_label=latex_label, unit=unit, minimum=minimum, maximum=maximum, boundary=boundary)

    def prob(self, val):
        val = np.asarray(val, dtype=float)
        a, b = self.minimum, self.maximum
        return np.where((val >= a) & (val <= b), (1.0 + a) * (1.0 + b) / ((b - a) * (1.0 + val) ** 2), 0.0)

    def cdf(self, val):
        a, b = self.minimum, self.maximum
        q = np.clip(np.asarray(val, dtype=float), a, b)
        return (1.0 + b) * (q - a) / ((b - a) * (1.0 + q))

    def rescale(self, val):
        a, b = self.minimum, self.maximum
        f = np.asarray(val, dtype=float)
        return ((1.0 + b) * a + f * (b - a)) / ((1.0 + b) - f * (b - a))
    
class LnPriorManager(LogisticParameterManager):

    def initialize_walkers(self, nwalkers, injected_parameters, reference_posterior=None, verbose=False, use_lvk_mass_and_time_prior=False):
        '''
        Function to get initial walkers for an emcee run depending on 
        settings provided by the user.
        '''
            
        # get logistic vs cartesian parameters
        logistic_params = self.logistic_parameters
        logistic_names_phys = [x.physical_name for x in logistic_params]
        cartesian_params = self.cartesian_angles
        cartesian_names_phys = [x.physical_name for x in cartesian_params]
        
        # Initialize walkers randomly
        # (code sees unit scale quantities; use logit transformations to take boundaries to +/- infinity)
        p0_arr = np.asarray([[np.random.normal() for j in range(self.num_parameters)] for i in range(nwalkers)])
        
        if reference_posterior is None:
            # we want the initial walkers to be drawn from "balls" tightly around the injected parameters
            # in logistic space
            for param in logistic_params:
                
                # cycle through the logistic parameters
                p = self.sampled_keys.index(param.logistic_name)
                param_kw = param.physical_name
                
                try:
                    param_phys = injected_parameters[param_kw]
                    # If given (e.g. declination, get sin declination like we need)
                    if type(param) == TrigLogisticParameter:
                        param_phys = param.trig_function(param_phys)
                    if verbose:
                        print('injected', param_phys, param_kw)
                except ValueError:
                    if verbose:
                        print(f"{param_kw} not in injected_parameters dict, continuing anyways")
                    continue
                except KeyError:
                    if verbose:
                        print(f"{param_kw} not in injected_parameters dict, continuing anyways")
                    continue
                if param_phys in param.limit:
                    print(f'reference value of {param_kw} is on boundary of acceptable range,'
                          f' drawing random value from within range, '
                          f"We are not going to draw around that value because we'll get infs!")
                    continue
                elif param_phys < param.limit[0] or param_phys > param.limit[1]:
                    print(f"WARNING: Injected value ({param_phys}) for {param_kw} is outside limit {param.limit}."
                          f" We will not set initial values around this value")
                    continue
                
                # transform into logistic space
                param_logit = param.physical_to_logistic(param_phys)

                # draw gaussian ball in logistic space
                p0_arr[:, p] = np.asarray([np.random.normal(loc=param_logit, scale=0.05) for i in range(nwalkers)])

            # if time of coalescence sampled over need to include this separately since it isn't a unit scaled quantity
            if self.vary_time:
                if use_lvk_mass_and_time_prior:
                    initial_t_walkers = np.random.uniform(self.reference_time-0.075, self.reference_time+0.075, size=nwalkers)
                else:
                    initial_t_walkers = np.random.normal(loc=self.reference_time, scale=self.sigma_time, size=nwalkers)
                

                p0_arr[:, self.sampled_keys.index('geocenter_time')] = initial_t_walkers
            
        else: 
            # if instead given a reference posterior, choose initial walkers from it for most parameters.
            # select random set of indices from the given reference posterior
            idxs = np.random.choice(range(len(reference_posterior['total_mass'])), size=nwalkers) 
            
            # cycle through logistic params
            for param in logistic_params:
                # draw walkers
                column = np.asarray(reference_posterior[param.physical_name]) if param.physical_name in reference_posterior else None
                if column is None or np.ptp(column) == 0:
                    print(f"{param.physical_name} is missing or constant in the reference posterior; starting it at random")
                    continue
                
                walkers_phys = column[idxs]
                                 
                # transform into logistic space
                walkers_logit = param.physical_to_logistic(walkers_phys) 
                
                # put in p0 array
                p0_arr[:, self.sampled_keys.index(param.logistic_name)] = walkers_logit
           
            # cycle through cartesian params
            for param in cartesian_params:

                # draw walkers
                param_kw = param.physical_name
                column = np.asarray(reference_posterior[param_kw]) if param_kw in reference_posterior else None
                if column is None or np.ptp(column) == 0:
                    print(f"{param_kw} is missing or constant in the reference posterior; starting it at random")
                    continue
                
                walkers_phys = column[idxs]

                # get x and y components
                initial_x, initial_y = param.radian_to_cartesian(walkers_phys) 
                
                # put in p0 array
                p0_arr[:, self.sampled_keys.index(f'{param_kw}_x')] = -1.0*initial_x
                p0_arr[:, self.sampled_keys.index(f'{param_kw}_y')] = -1.0*initial_y
        
            # finally, if sampling time ... 
            if self.vary_time:
                p0_arr[:, self.sampled_keys.index('geocenter_time')] = np.asarray(reference_posterior['geocenter_time'])[idxs]
        
        p0 = p0_arr.tolist()
        return p0

    def get_lnprior(self, x, phys_dict=None, use_lvk_mass_and_time_prior=False):
        '''
        Get the Bayesian log prior for a parameter vector `x`. 
        
        The priors used in the TDinf code are as follows: 
        - Uniform in total mass, mass ratio, luminosity distance, and 
          spin magnitudes with bounds provided by the user.
        - Uniform in phi_12, theta_jn, phase, polarization, and 
          right_ascension over their allowed angular ranges [radians].
        - Uniform in the cosine of the spin tilt angles and theta_jn.
        - Uniform in the sine of the declination.
        - Gaussian distributed around the geocenter reference time with a
          standard deviation provided by the user. 
        - use_lvk_mass_and_time_prior: sets prior uniform in detector frame 
          masses and time (+/- 0.075 s around reference time).
        - uniform in eccentricity and mean anomaly
        '''

        # Convert `x` from an array to a dictionary
        x_dict = self.get_logistic_dict(x)

        # If phys_dict passed in kws, return it, if not, calculate it with samp_to_phys
        if phys_dict is None:
            phys_dict = self.samp_to_phys(x)

        lnprior = 0
        # Logistic jacobians
        for logistic_param in self.logistic_parameters:
            lnprior += logistic_param.ln_prior_weight(phys_dict[logistic_param.physical_name])
        if use_lvk_mass_and_time_prior:
            # Previous line makes the prior flat in M and q and this term [M / (q + 1)**2] makes it flat in det frame masses
            lnprior += np.log(phys_dict['total_mass']) - 2.0 * np.log1p(phys_dict['mass_ratio'])
        # cartesian angle terms
        for cartesian_param in self.cartesian_angles:
            lnprior += cartesian_param.ln_prior_weight(x_dict[cartesian_param.x_name], x_dict[cartesian_param.y_name])

        # Gaussian
        if self.vary_time:
            if not(use_lvk_mass_and_time_prior):
                lnprior -= 0.5 * ((phys_dict['geocenter_time'] - self.reference_time) ** 2) / (self.sigma_time ** 2)
            else:
                delta_t = 0.15 # 2 times of 75 ms, which is what is typically used
                lnprior += np.log(1./delta_t)
                if abs(phys_dict['geocenter_time'] - self.reference_time) > delta_t/2:
                    return -np.inf

        # Spins
        if not self.no_spins:
            if self.aligned_spins:
                lnprior += -0.5 * (x_dict['c1_z'] ** 2)
                lnprior += -0.5 * (x_dict['c2_z'] ** 2)
            else:
                lnprior += -0.5 * (x_dict['c1_x'] ** 2 + x_dict['c1_y'] ** 2 + x_dict['c1_z'] ** 2)
                lnprior += -0.5 * (x_dict['c2_x'] ** 2 + x_dict['c2_y'] ** 2 + x_dict['c2_z'] ** 2)

        return lnprior


    def get_bilby_mcmc_prior_dict(self, use_lvk_mass_and_time_prior=False, use_physical_coordinates_for_sampling=False):
        '''
        The prior of the analysis as a bilby `PriorDict`, to be passed to bilby_mcmc. Only the SAMPLED parameters get a prior.
        `use_lvk_mass_and_time_prior` must be the value that is passed to get_lnprior. Needs bilby.

        use_physical_coordinates_for_sampling=False (default): the sampling coordinates of this manager, one prior per sampled
        key, equal to get_lnprior up to a constant (tdinf drops normalisations, bilby's priors are normalised):
          logistic coordinates                      -> Logistic(0, 1)
          total mass / mass ratio, LVK flag         -> LVKTotalMassCoordinate / LVKMassRatioCoordinate (the extra factor M / (1 + q)**2)
          both components of each Cartesian angle,
          and the spin-direction numbers c          -> Gaussian(0, 1)

        use_physical_coordinates_for_sampling=True: the physical parameters themselves, with the same distributions, built from the
        limits and names of logistic_parameters and cartesian_angles:
          limits [a, b] of a logistic parameter     -> Uniform(a, b)   (LVK flag: total mass PowerLaw(1), mass ratio LVKMassRatioPrior)
          inclination (cos) / declination (sin)     -> Sine(0, pi) / Cosine(-pi/2, pi/2), flat in the cosine / sine
          Cartesian angles                          -> Uniform over one period around phase_offset, boundary='periodic'
          spins, precessing: spin{1,2}_magnitude as above, plus spin{1,2}_tilt -> Sine(0, pi) and spin{1,2}_azimuth -> Uniform(0, 2 pi),
                             the polar and azimuthal angles of the spin in the frame of spin{1,2}_x, _y, _z, so that
                             spin_x = a sin(tilt) cos(azimuth), spin_y = a sin(tilt) sin(azimuth), spin_z = a cos(tilt)
                             (uniform magnitude, isotropic direction, as in the sampling-coordinate prior)
          spins, aligned: spin{1,2}_z -> Uniform(-chi_max, chi_max) (they replace the magnitudes; needs chi_lim[0] = 0)
        In both cases geocenter_time (if sampled) is Gaussian(reference time, sigma_time), or with the LVK flag
        Uniform(reference time +- 0.075 s).
        '''
        from bilby.core.prior import PriorDict, Logistic, Gaussian, TruncatedGaussian, Uniform, Sine, Cosine, PowerLaw

        priors = PriorDict()
        lvk = use_lvk_mass_and_time_prior

        def time_prior(key):
            if lvk:
                return Uniform(minimum=self.reference_time - 0.075, maximum=self.reference_time + 0.075, name=key, boundary='reflective')
            # A Gaussian cut at +-5 sigma (the cut removes 6e-7 of the prior). bilby's proposals scale their steps by the prior WIDTH and treat an infinite width as 1 (second), so with an
            # uncut Gaussian the millisecond-wide time posterior sets the step size of the adaptive Gaussian proposal for every parameter; a finite width puts time on the same footing.
            return TruncatedGaussian(mu=self.reference_time, sigma=self.sigma_time, minimum=self.reference_time - 5 * self.sigma_time,
                                     maximum=self.reference_time + 5 * self.sigma_time, name=key, boundary='reflective')

        if not use_physical_coordinates_for_sampling:
            # ---- the sampling coordinates: logistic parameters, Cartesian angles, spin-direction numbers
            logistic = {p.logistic_name: p for p in self.logistic_parameters}
            cartesian = {name for angle in self.cartesian_angles for name in (angle.x_name, angle.y_name)}
            for key in self.sampled_keys:
                if key in logistic:
                    lower, upper = logistic[key].limit
                    if lvk and logistic[key].physical_name == 'total_mass':
                        priors[key] = LVKTotalMassCoordinate(lower, upper, name=key)
                    elif lvk and logistic[key].physical_name == 'mass_ratio':
                        priors[key] = LVKMassRatioCoordinate(lower, upper, name=key)
                    else:
                        priors[key] = Logistic(mu=0.0, scale=1.0, name=key)
                elif key == 'geocenter_time':
                    priors[key] = time_prior(key)
                elif key in cartesian or key.startswith(('c1_', 'c2_')):
                    priors[key] = Gaussian(mu=0.0, sigma=1.0, name=key)
                else:
                    raise ValueError(f"no bilby prior is defined for the sampled parameter '{key}': add it to get_bilby_mcmc_prior_dict")
            return priors

        # ---- the physical parameters
        aligned = self.aligned_spins and not self.no_spins
        chi_limit = None
        for parameter in self.logistic_parameters:
            name = parameter.physical_name
            lower, upper = parameter.limit
            if isinstance(parameter, TrigLogisticParameter):
                if parameter.trig_function is np.cos:           # flat in cos(angle): inclination
                    priors[name] = Sine(minimum=0.0, maximum=np.pi, name=name)
                else:                                           # flat in sin(angle): declination
                    priors[name] = Cosine(minimum=-np.pi / 2, maximum=np.pi / 2, name=name)
            elif name in ('spin1_magnitude', 'spin2_magnitude'):
                chi_limit = (lower, upper)
                if not aligned:
                    priors[name] = Uniform(minimum=lower, maximum=upper, name=name)
            elif lvk and name == 'total_mass':
                priors[name] = PowerLaw(alpha=1, minimum=lower, maximum=upper, name=name)
            elif lvk and name == 'mass_ratio':
                priors[name] = LVKMassRatioPrior(lower, upper, name=name)
            else:
                priors[name] = Uniform(minimum=lower, maximum=upper, name=name)

        for angle in self.cartesian_angles:
            priors[angle.physical_name] = Uniform(minimum=angle.phase_offset - np.pi, maximum=angle.phase_offset + np.pi,
                                                  name=angle.physical_name, boundary='periodic')
            if angle.physical_name=="polarization":
                priors[angle.physical_name] = Uniform(minimum=angle.phase_offset, maximum=angle.phase_offset + np.pi,
                                                                  name=angle.physical_name, boundary='periodic')

        if not self.no_spins:
            for i in ('1', '2'):
                if aligned:
                    if chi_limit[0] != 0:
                        raise ValueError(f"aligned spins in physical coordinates need the spin magnitude limit to start at 0, not {chi_limit}")
                    priors[f'spin{i}_z'] = Uniform(minimum=-chi_limit[1], maximum=chi_limit[1], name=f'spin{i}_z')
                else:
                    priors[f'spin{i}_tilt'] = Sine(minimum=0.0, maximum=np.pi, name=f'spin{i}_tilt')
                    priors[f'spin{i}_azimuth'] = Uniform(minimum=0.0, maximum=2 * np.pi, name=f'spin{i}_azimuth', boundary='periodic')

        if self.vary_time:
            priors['geocenter_time'] = time_prior('geocenter_time')
        # Bounded priors: a proposal that leaves the support is reflected back inside instead of being rejected by the prior (a symmetric move, so detailed balance holds).
        # About half of the differential-evolution jumps of a cold chain otherwise land outside the spin, mass-ratio and tilt bounds and are wasted.
        # The mass ratio is left out: its upper wall (q = 1) is only the convention m1 >= m2, and the physical fold across it is q -> 1/q with the two spins exchanged, not q -> 2 - q.
        for key, prior in priors.items():
            if key == 'mass_ratio':
                continue
            if prior.boundary is None and np.isfinite(prior.minimum) and np.isfinite(prior.maximum):
                prior.boundary = 'reflective'
        return priors


class AntennaAndTimeManager(LogisticParameterManager):
    """
    Manager for antenna patterns and time delays for a set of interferometers (IFOs).

    Inherits from LogisticParameterManager and adds methods to handle:
    - fixed or varying sky position
    - fixed or varying geocenter times
    - precomputing antenna patterns and time delays
    """

    def __init__(self, ifos, *args, **kwargs):
        super(AntennaAndTimeManager, self).__init__(*args, **kwargs)

        # Initialize storage for peak times, antenna patterns, and time delays
        self.peak_time_dict = None
        self.antenna_pattern_dict = None
        self.time_delay_dict = None

        if not self.vary_skypos:
            # Antenna pattern is fixed if sky location is fixed
            # (it will vary slightly over different geocenter times)
            self.antenna_pattern_dict = t_and_AP.get_antenna_pattern_dict(
                self.reference_time, ifos,
                self.fixed['right_ascension'],
                self.fixed['declination'],
                self.fixed['polarization']
            )

            if not self.vary_time:
                # Peak time dict is only fixed if both sky location and time are fixed
                self.peak_time_dict = t_and_AP.get_tgps_dict(
                    self.reference_time, ifos,
                    self.fixed['right_ascension'],
                    self.fixed['declination']
                )

                self.time_delay_dict = self.compute_time_delay_dict(
                    self.fixed['right_ascension'],
                    self.fixed['declination'],
                    self.reference_time, ifos
                )
        return

    def get_time_delay_dict(self, x_phys, ifos):
        """
        Return time delay dictionary for each interferometer.
        """
        if self.time_delay_dict is not None:
            return self.time_delay_dict

        return self.compute_time_delay_dict(
            x_phys['right_ascension'],
            x_phys['declination'],
            x_phys['geocenter_time'],
            ifos
        )

    @staticmethod
    def compute_time_delay_dict(right_ascension, declination, geocenter_time, ifos):
        """
        Compute the time delay from geocenter to each interferometer.
        """
        time_delay_dict = {
            ifo: lal.TimeDelayFromEarthCenter(
                lal.cached_detector_by_prefix[ifo].location,
                right_ascension,
                declination,
                geocenter_time
            ) for ifo in ifos
        }
        return time_delay_dict

    def get_tpeak_dict(self, x_phys, ifos):
        """
        Return the peak times for each interferometer.
        """
        if self.peak_time_dict is not None:
            return self.peak_time_dict

        return t_and_AP.get_tgps_dict(
            x_phys['geocenter_time'],
            ifos,
            x_phys['right_ascension'],
            x_phys['declination']
        )

    def get_antenna_pattern_dict(self, x_phys, ifos):
        """
        Return the antenna pattern (Fp, Fc) for each interferometer.
        """
        if self.antenna_pattern_dict is not None:
            return self.antenna_pattern_dict

        return t_and_AP.get_antenna_pattern_dict(
            x_phys['geocenter_time'], ifos,
            x_phys['right_ascension'],
            x_phys['declination'],
            x_phys['polarization']
        )

class WaveformManager(LogisticParameterManager):
    def __init__(self, ifos, *args, **kwargs):
        super(WaveformManager, self).__init__(*args, **kwargs)
        self.approx_name = kwargs['approx']
        if is_external_approximant(self.approx_name):
            self.approximant = kwargs['approx']
            self.backend = make_backend(self.approx_name)
            print(f"Approximant '{self.approx_name}' is not a LAL approximant; "
                    f"initializing external backend ({type(self.backend).__name__}).")
        else:
            self.approximant = lalsim.SimInspiralGetApproximantFromString(self.approx_name)
            self.backend = None
        self.antenna_and_time_manager = AntennaAndTimeManager(ifos, *args, **kwargs)

    def generate_lal_hphc(self, m1_msun, m2_msun, chi1, chi2, delta_t, dist_mpc=1,
                          f22_start=20, f_ref=11, inclination=0, phi_ref=0.,NR_kws=None, eccentricity=0, mean_anomaly=0):
        """
        Generate the plus and cross polarizations for given waveform parameters and approximant
        """
        # Convert masses from solar masses --> kilograms
        m1_kg = m1_msun * lal.MSUN_SI
        m2_kg = m2_msun * lal.MSUN_SI

        # Convert distance from MPc --> meters
        distance = dist_mpc * 1e6 * lal.PC_SI

        param_dict = lal.CreateDict()

        # if using an SXS waveform
        if self.approximant==90:
            assert NR_kws is not None, "If using NR_hdf5 approximate, must pass NR_kws !!"
            nr_path = NR_kws['nr_path']
            # get masses
            mtot_msun = NR_kws["mtot"]
            with h5py.File(nr_path, "r") as f:
                m1 = f.attrs["mass1"]
                m2 = f.attrs["mass2"]
                m1_kg = m1 * mtot_msun * lal.MSUN_SI / (m1 + m2)
                m2_kg = m2 * mtot_msun * lal.MSUN_SI / (m1 + m2)
            # compute spin components in lalsim frame
            s = lalsim.SimInspiralNRWaveformGetSpinsFromHDF5File(
                NR_kws["f_ref"], mtot_msun, nr_path
            )
            chi1 = [s[0], s[1], s[2]]
            chi2 = [s[3], s[4], s[5]]
        
            lalsim.SimInspiralWaveformParamsInsertNumRelData(param_dict, nr_path)
            ascending_node = np.pi / 2
        else: 
            ascending_node = 0

        # generate waveform
        try:
            hp, hc = lalsim.SimInspiralChooseTDWaveform(m1_kg, m2_kg,
                                                    chi1[0], chi1[1], chi1[2],
                                                    chi2[0], chi2[1], chi2[2],
                                                    distance, inclination,
                                                    phi_ref, ascending_node, eccentricity, mean_anomaly,
                                                    delta_t, f22_start, f_ref,
                                                    param_dict,
                                                    self.approximant)
            
        # catch input domain errors (these show up for NRSur waveforms)
        except Exception as e:
            if "Input domain error" in str(e):
                print("Caught Input domain error")
                print("m1_kg =", m1_kg)
                print("m2_kg =", m2_kg)
                print("chi1 =", chi1)
                print("chi2 =", chi2)
                print("eccentricity =", eccentricity)
                print("mean_anomaly =", mean_anomaly)
                print("distance =", distance)
                print("inclination =", inclination)
                print("phi_ref =", phi_ref)
                print("delta_t =", delta_t)
                print("ascending node =", ascending_node)
                print("f22_start =", f22_start)
                print("f_ref =", f_ref)
                print("Exception:", e)
                hp, hc = np.nan, np.nan # return nans
            else:
                raise  # Re-raise other errors
            
        return hp, hc

    def get_hplus_hcross(self, x_phys, delta_t, f22_start=11, f_ref=11, NR_kws=None):
        """
        Get plus and cross polarizations of waveform at geocenter
        """

        # get parameters needed for LAL waveform generation
        m1, m2 = m1m2_from_mtotq(x_phys['total_mass'], x_phys['mass_ratio'])
        chi1 = [x_phys['spin1_x'], x_phys['spin1_y'], x_phys['spin1_z']]
        chi2 = [x_phys['spin2_x'], x_phys['spin2_y'], x_phys['spin2_z']]
        eccentricity, mean_anomaly = x_phys.get('eccentricity', 0.0), x_phys.get('mean_anomaly', 0.0)

        if self.backend is not None:
            hp, hc = self.backend.get_hphc(
                m1, m2, chi1, chi2, delta_t=delta_t,
                dist_mpc=x_phys['luminosity_distance'],
                f22_start=f22_start, f_ref=f_ref,
                inclination=x_phys['inclination'], phi_ref=x_phys['phase'],
                eccentricity=eccentricity, mean_anomaly=mean_anomaly
            )
            # for catching waveform errors
            if isinstance(hp, float) and hp!=hp:
                return np.nan, np.nan
            # already a Timeseries
            return hp, hc
        
        hp, hc = self.generate_lal_hphc(m1, m2, chi1, chi2, delta_t=delta_t,
                                        dist_mpc=x_phys['luminosity_distance'],
                                        f22_start=f22_start, f_ref=f_ref,
                                        inclination=x_phys['inclination'],
                                        phi_ref=x_phys['phase'],NR_kws=NR_kws,
                                        eccentricity=eccentricity, mean_anomaly=mean_anomaly)
        
        # for catching waveform errors
        if isinstance(hp, float) and hp!=hp:
            return np.nan, np.nan

        # wrap as gwpy TimeSeries
        return TimeSeries.from_lal(hp), TimeSeries.from_lal(hc)

    
    def get_projected_waveform(self, x_phys, ifos, time_dict, f22_start=11, f_ref=11, 
                               window=False, NR_kws=None):
        """
        Generate plus and cross polarizations for a waveform and project them 
        onto the detectors.
        """
        
        # get time spacing of data
        delta_t = time_dict[ifos[0]][1] - time_dict[ifos[0]][0]

        # get hplus and hcross
        hp, hc = self.get_hplus_hcross(x_phys, delta_t, f22_start=f22_start, 
                                       f_ref=f_ref,NR_kws=NR_kws)
        
        # for catching waveform errors
        if isinstance(hp, float) and hp!=hp:
            return np.nan

        # apply tukey window if desired 
        if window:
            print
            hp = apply_window(hp)
            hc = apply_window(hc)
        
        # set times in geocenter time
        hp.t0 = x_phys['geocenter_time'] + hp.t0.value
        hc.t0 = x_phys['geocenter_time'] + hc.t0.value

        # get antenna pattern
        AP_dict = self.antenna_and_time_manager.get_antenna_pattern_dict(x_phys, ifos)

        # get delay times
        time_delay_dict = self.antenna_and_time_manager.get_time_delay_dict(x_phys, ifos)

        # cycle through ifos
        projected_waveform_dict = {}
        for ifo in ifos:

            # generate complex timeseries
            h_td = hp.value - 1j * hc.value

            # get antenna patterns for this ifo
            Fp, Fc = AP_dict[ifo]

            # project waveform using antenna patterns
            h_ifo = Fp * h_td.real - Fc * h_td.imag

            # convert h_ifo to detector_time
            time_delay = time_delay_dict[ifo]

            # convert geocenter timeseries to detector timeseries by adding time delay
            # then interpolating that timeseries to the actual, sampled detector times
            h_projected = interpolate_timeseries(hp.times.value + time_delay, h_ifo, time_dict[ifo])
            projected_waveform_dict[ifo] = h_projected

        return projected_waveform_dict


class LnLikelihoodManager(LogisticParameterManager):
    def __init__(self, psd_dict, time_dict, data_dict, f_low, f_ref, f22_start, 
                 f_max=None, only_prior=False, *args, **kwargs):
        self.time_dict = time_dict
        self.data_dict = data_dict
        self.f_low = f_low
        self.f_ref = f_ref
        self.f22_start = f22_start
        self.f_max = f_max
        self.psd_dict = psd_dict
        self.rho_dict, self.conditioned_psd_dict = self._make_autocorrolation_dict()
        self.ifos = list(self.data_dict.keys())
        for ifo, rho in self.rho_dict.items():
            assert len(rho) == len(self.data_dict[ifo]), 'Length for ACF is not the same as for the data'
        self.only_prior = only_prior
        self.whitened_data_dict = self._make_whitened_data_dict()
        self.waveform_manager = WaveformManager(self.ifos, *args, **kwargs)
        self.use_lvk_mass_and_time_prior=kwargs.get('use_lvk_mass_and_time_prior', False)
        self.log_prior = LnPriorManager(*args, **kwargs)
        
        # Express inverse covariance matrix using Gohberg-Semencul theorem
        # Precompute the necessary quantities here for efficiency
        self.Tinv_first_col_dict = {}
        for ifo in self.ifos:
            self.Tinv_first_col_dict[ifo] = gohberg_semencul_prep(self.rho_dict[ifo])
        super().__init__(*args, **kwargs)

    def _make_autocorrolation_dict(self):
        '''
        Auto-generate ACF and additionally return the PSDs used to calculate them
        '''
        acf_dict, cond_psds_dict = get_ACF(self.psd_dict, self.time_dict, f_low=self.f_low, f_max=self.f_max, return_psds=True)
        return acf_dict, cond_psds_dict 

    def _make_whitened_data_dict(self): 
        '''
        Auto-generate whitened data with FREQUENCY DOMAIN whitening
        '''
        return {
            ifo : whitenData(
                self.data_dict[ifo], self.time_dict[ifo], 
                self.conditioned_psd_dict[ifo][:,1], self.conditioned_psd_dict[ifo][:,0]
            ) for ifo in self.ifos
        }

    def waveform_has_error(self, phys_dict, waveform_ifo, residual):
        '''
        Function to check for various waveform generation errors
        '''
        # Make sure waveform falls in desired time window
        if np.all(waveform_ifo == 0):
            print('waveform falls outside allowed window for:')
            print(phys_dict)
            # This is actually fine when calculating the likelihood, but 
            # we want to print out that it is happening to the log file
            return False
        # Check for NaNs in the residual
        if sum(np.isnan(residual)) > 0:
            print('NaNs in residuals for:')
            print(phys_dict)
            return True
        # Check for infinities in the residual
        if sum(np.isinf(residual)) > 0:
            print('infinities in residuals for:')
            print(phys_dict)
            return True
        return False

    def get_log_likelihood(self, x_phys, verbose=False, **kwargs):
        """
        Compute the Bayesian log-likelihood for a given set of physical parameters.
    
        The log-likelihood is computed by generating the model waveform for the 
        specified parameters, subtracting it from the observed data, over-whitening the 
        residuals using the autocorrelation function, and computing the Gaussian 
        likelihood under the assumption of stationary Gaussian noise.
    
        Parameters
        ----------
        x_phys : dict or array-like
            The set of physical waveform parameters (e.g., masses, spins, etc.) 
            in physical units.
        verbose : bool, optional
            If True, prints additional information during computation, e.g., for debugging.
        **kwargs : dict, optional
            Additional keyword arguments passed to `waveform_manager.get_projected_waveform`.
            Supported keys:
            - `f22_start` : float, start frequency for the (2,2) mode. Defaults to `self.f22_start`.
            - `f_ref` : float, reference frequency for spin definitions. Defaults to `self.f_ref`.
            - `window` : bool, whether to apply a window function to the waveform. Defaults to False.
    
        Returns
        -------
        float
            The computed log-likelihood value. Returns `-np.inf` if waveform generation 
            fails or if there are waveform errors.
        """

        # Generate waveform for parameters `x_phys`
        if verbose:
            print('getting wf')
        projected_wf_dict = self.waveform_manager.get_projected_waveform(
            x_phys, self.ifos, self.time_dict,
            f22_start=kwargs.get('f22_start', self.f22_start),
            f_ref=kwargs.get('f_ref', self.f_ref),
            window=kwargs.get('window', False)
        )
        if verbose:
            print('done getting wf')
            
        # Return NaN if "Input domain error" for waveform
        if isinstance(projected_wf_dict, float) and projected_wf_dict!=projected_wf_dict:
            return -np.inf

        ln_posterior = 0

        # Cycle through ifos
        for ifo, data in self.data_dict.items():

            # Truncate and compute residuals
            r = data - projected_wf_dict[ifo]

            # Check for other waveform errors
            if self.waveform_has_error(x_phys, projected_wf_dict[ifo], r):
                return -np.inf

            # "Over whiten" residuals
            # rwt = solve_toeplitz(self.rho_dict[ifo], r)
            rwt = gohberg_semencul_multiply(self.Tinv_first_col_dict[ifo], r)

            # Compute log likelihood for ifo
            ln_posterior -= 0.5 * np.dot(r, rwt)
            
        return ln_posterior

    def get_log_posterior(self, x, verbose=False, **kwargs):
        """
        Compute the Bayesian log-posterior probability for a given parameter
        vector `x`.
    
        The posterior is computed as: log_posterior = log_likelihood + log_prior
    
        If `self.only_prior` is True, the likelihood is skipped and only the prior 
        is evaluated.
    
        Parameters
        ----------
        x : array-like
            Parameter vector in sampling coordinates.
        verbose : bool, optional
            If True, prints additional information during computation, e.g., for debugging.
        **kwargs : dict, optional
            Optional arguments:
            - `x_phys` : dict or array-like, precomputed physical parameters. 
              If not provided, computed via `self.samp_to_phys(x)`.
    
        Returns
        -------
        float
            The log-posterior value. Returns `-np.inf` if the posterior evaluates to NaN.
        """
        
        # get physical parameters
        x_phys = kwargs.pop('x_phys', None)
        if x_phys is None:
            x_phys = self.samp_to_phys(x)
        if verbose:
            print('x_phys', x_phys)

        # Initialize posterior to 0
        lnprob = 0

        # Calculate prior
        lnprob += self.log_prior.get_lnprior(x, phys_dict=x_phys, use_lvk_mass_and_time_prior=self.use_lvk_mass_and_time_prior)
        # right now lnprob is just the prior so check if it is finite
        if not np.isfinite(lnprob):
            return -np.inf
        
        # Calculate posterior by adding lnL
        if not self.only_prior:
            lnprob += self.get_log_likelihood(x_phys, verbose=verbose)


        # Check for NaN
        if lnprob != lnprob:
            print('lnprob = NaN for:')
            print(x_phys)
            return -np.inf

        # Return posterior
        return lnprob
            
    def get_SNRs(self, samples): 
        '''
        Get SNRs (optimal and matched filter, for all ifos and network)
        for a list of samples. 
        '''
                
        # set up arrays
        per_detector_opt_snrs = np.zeros((len(self.ifos), len(samples)))
        per_detector_mf_snrs = np.zeros((len(self.ifos), len(samples)))
        network_opt_snrs = np.zeros(len(samples))
        network_mf_snrs = np.zeros(len(samples))
        
        # cycle through the input samples
        samples_phys = [self.samp_to_phys(x) for x in samples]
        for i,x_phys in enumerate(samples_phys):
            
            # get waveforms for this sample
            projected_wf_dict = self.waveform_manager.get_projected_waveform(
                    x_phys, self.ifos, self.time_dict,
                    f22_start=self.f22_start,
                    f_ref=self.f_ref
                )
            
            # lists for per-detector snrs
            opt_snrs = []
            mf_snrs = []
            
            # cycle through the interferometers
            for ifo in self.ifos:
            
                sig = projected_wf_dict[ifo]
                data = self.data_dict[ifo]
                rho = self.rho_dict[ifo]
                
                opt_snrs.append(calc_opt_SNR(sig, rho))
                mf_snrs.append(calc_mf_SNR(data, sig, rho))
                
            # add to arrays 
            per_detector_opt_snrs[:,i] = opt_snrs
            per_detector_mf_snrs[:,i] = mf_snrs
            
            # calculate network snrs
            network_opt_snrs[i] = calc_network_SNR(opt_snrs)
            network_mf_snrs[i] = calc_network_SNR(mf_snrs)
        
        # create dict with all and return 
        SNRs_dict = {
            'network_optimal_SNR':network_opt_snrs,
            'network_matched_filter_SNR':network_mf_snrs,
            **{f'{ifo}_optimal_SNR':per_detector_opt_snrs[i] for i,ifo in enumerate(self.ifos)}, 
            **{f'{ifo}_matched_filter_SNR':per_detector_mf_snrs[i] for i,ifo in enumerate(self.ifos)}
        }
        return SNRs_dict

try:
    from bilby import Likelihood as _BilbyLikelihood
except ImportError:         # bilby is only needed by TdinfBilbyLikelihood
    _BilbyLikelihood = object


class TdinfBilbyLikelihood(_BilbyLikelihood):
    """
    tdinf's log LIKELIHOOD for bilby (the prior is the PriorDict from LnPriorManager.get_bilby_mcmc_prior_dict). It calls the
    likelihood manager's get_log_likelihood with the physical dictionary, which it builds from the parameters bilby hands over:

      use_physical_coordinates_for_sampling=False: the parameters are the sampling coordinates (the manager's sampled_keys); the
          dictionary is lm.samp_to_phys(x), as in the emcee code.
      use_physical_coordinates_for_sampling=True: the parameters are the physical ones (parameter_keys = the keys of the prior
          dict); the dictionary is the manager's fixed parameters plus those parameters, with spin{1,2}_x, _y, _z made by the manager's
          get_physical_spins from magnitude, tilt and azimuth (precessing), from spin{1,2}_z (aligned) or zero (no spins).

    Anything that is not finite (a failed waveform, NaN) is -inf; with only_prior the likelihood is 0.

    The two conversion methods are only needed in physical coordinates: physical_parameters_from_sampling (a starting point in
    sampling coordinates -> the physical parameters bilby samples, for initial_sample_dict) and sampling_from_physical_parameters
    (bilby's samples -> sampling coordinates, for the usual post-processing and ln prior).

    npool: bilby pickles this object into every worker process, so it holds only plain, picklable state and is defined at module
    level. limit_threads=True (set it when npool > 1) limits BLAS / OpenMP to one thread in every process that evaluates it, so that
    the workers do not oversubscribe the cores; this is decided per process (by pid), so an object that was already used in the
    parent before being pickled still limits the threads in each worker.
    """

    _SPIN_ANGLE_KEYS = ('spin1_tilt', 'spin1_azimuth', 'spin2_tilt', 'spin2_azimuth', 'spin1_z', 'spin2_z')

    def __init__(self, likelihood_manager, parameter_keys=None, use_physical_coordinates_for_sampling=False, limit_threads=False):
        self.lm = likelihood_manager
        self.use_physical = bool(use_physical_coordinates_for_sampling)
        self.limit_threads = bool(limit_threads)
        self._limited_in_pid = None
        if self.use_physical:
            if parameter_keys is None:
                raise ValueError("physical coordinates need parameter_keys: pass list(prior_dict), the keys of the prior dict")
            self.keys = list(parameter_keys)
            self._copied_keys = [key for key in self.keys if key not in self._SPIN_ANGLE_KEYS]
        else:
            self.keys = list(likelihood_manager.sampled_keys)
        if _BilbyLikelihood is not object:
            super().__init__(parameters={key: None for key in self.keys})

    # ---------------------------------------------------------------------------------------------- the likelihood
    def _limit_threads_once(self):
        if self.limit_threads and self._limited_in_pid != os.getpid():
            try:
                from threadpoolctl import threadpool_limits
                threadpool_limits(limits=1)
            except ImportError:
                pass
            self._limited_in_pid = os.getpid()

    def _physical_dict(self):
        lm, p = self.lm, self.parameters
        physical = dict(lm.fixed)                       # eccentricity, mean_anomaly, sky position, time when they are not sampled
        for key in self._copied_keys:
            physical[key] = float(p[key])
        for i in ('1', '2'):
            if lm.no_spins:
                magnitude, direction = 0.0, (0.0, 0.0, 1.0)
            elif lm.aligned_spins:
                z = float(p[f'spin{i}_z'])
                magnitude, direction = abs(z), (0.0, 0.0, 1.0 if z >= 0 else -1.0)
                physical[f'spin{i}_magnitude'] = magnitude
            else:
                magnitude, tilt, azimuth = float(p[f'spin{i}_magnitude']), float(p[f'spin{i}_tilt']), float(p[f'spin{i}_azimuth'])
                direction = (np.sin(tilt) * np.cos(azimuth), np.sin(tilt) * np.sin(azimuth), np.cos(tilt))
            # the manager's own function: magnitude * c / |c| (zeros if the waveform has no spins)
            physical[f'spin{i}_x'], physical[f'spin{i}_y'], physical[f'spin{i}_z'] = lm.get_physical_spins(magnitude, *direction)
        return physical

    def log_likelihood(self):
        if self.lm.only_prior:
            return 0.0
        self._limit_threads_once()
        if self.use_physical:
            x_phys = self._physical_dict()
        else:
            x_phys = self.lm.samp_to_phys(np.array([self.parameters[key] for key in self.keys], dtype=float))
        value = self.lm.get_log_likelihood(x_phys)
        return float(value) if np.isfinite(value) else -np.inf

    # ---------------------------------------------------------------------------------------------- conversions (physical coordinates)
    def physical_parameters_from_sampling(self, x):
        """
        The physical parameters bilby samples in physical coordinates (the keys of get_bilby_mcmc_prior_dict(...,
        use_physical_coordinates_for_sampling=True)) for a point `x` in sampling coordinates. Use it to turn a starting point into
        an initial_sample_dict.
        """
        lm = self.lm
        phys = lm.samp_to_phys(np.asarray(x, dtype=float))
        out = {}
        for parameter in lm.logistic_parameters:
            if lm.aligned_spins and parameter.physical_name in ('spin1_magnitude', 'spin2_magnitude'):
                continue                                    # aligned spins are sampled through spin1_z, spin2_z
            out[parameter.physical_name] = float(phys[parameter.physical_name])
        for angle in lm.cartesian_angles:
            out[angle.physical_name] = float(phys[angle.physical_name])
        if not lm.no_spins:
            for i in ('1', '2'):
                sx, sy, sz = phys[f'spin{i}_x'], phys[f'spin{i}_y'], phys[f'spin{i}_z']
                if lm.aligned_spins:
                    out[f'spin{i}_z'] = float(sz)
                else:
                    magnitude = np.sqrt(sx ** 2 + sy ** 2 + sz ** 2)
                    out[f'spin{i}_tilt'] = float(np.arccos(np.clip(sz / magnitude, -1.0, 1.0))) if magnitude > 0 else np.pi / 2
                    out[f'spin{i}_azimuth'] = float(np.arctan2(sy, sx) % (2 * np.pi))
        if lm.vary_time:
            out['geocenter_time'] = float(phys['geocenter_time'])
        return out

    def sampling_from_physical_parameters(self, table, rng=None):
        """
        Samples in SAMPLING coordinates, shape (n, ndim) in the order of sampled_keys, from a table (DataFrame or dict of columns) of
        the physical parameters that bilby sampled in physical coordinates. Needed for the usual post-processing and ln prior.

        The sampling coordinates contain redundant radial coordinates (the length of each Cartesian pair and of each spin-direction
        vector) that the physical parameters do not have. They are independent of the data, so they are drawn from their priors
        (Rayleigh for a pair, chi with 3 degrees of freedom for a vector, half-normal for an aligned spin): the result is then a
        genuine sample of the sampling-coordinate posterior, and samp_to_phys gives back exactly the physical parameters.
        """
        lm = self.lm
        rng = np.random.default_rng() if rng is None else rng
        column = {key: i for i, key in enumerate(lm.sampled_keys)}
        get = lambda name: np.asarray(table[name], dtype=float)
        n = len(get('geocenter_time' if lm.vary_time else lm.cartesian_angles[0].physical_name))
        x = np.empty((n, lm.num_parameters))
        for parameter in lm.logistic_parameters:
            name = parameter.physical_name
            values = np.abs(get(name.replace('_magnitude', '_z'))) if (lm.aligned_spins and name in ('spin1_magnitude', 'spin2_magnitude')) else get(name)
            x[:, column[parameter.logistic_name]] = parameter.physical_to_logistic(values)
        for angle in lm.cartesian_angles:
            # radian_to_cartesian ignores phase_offset (cartesian_to_radian adds it), so it is given the angle minus the offset
            direction_x, direction_y = angle.radian_to_cartesian(get(angle.physical_name) - angle.phase_offset)
            radius = np.sqrt(-2.0 * np.log(1.0 - rng.random(n)))              # Rayleigh(1): the length of a 2-D standard normal
            x[:, column[angle.x_name]] = radius * direction_x
            x[:, column[angle.y_name]] = radius * direction_y
        if not lm.no_spins:
            for i in ('1', '2'):
                if lm.aligned_spins:
                    x[:, column[f'c{i}_z']] = np.sign(get(f'spin{i}_z')) * np.abs(rng.standard_normal(n))
                else:
                    tilt, azimuth = get(f'spin{i}_tilt'), get(f'spin{i}_azimuth')
                    radius = np.linalg.norm(rng.standard_normal((n, 3)), axis=1)    # chi with 3 degrees of freedom
                    x[:, column[f'c{i}_x']] = radius * np.sin(tilt) * np.cos(azimuth)
                    x[:, column[f'c{i}_y']] = radius * np.sin(tilt) * np.sin(azimuth)
                    x[:, column[f'c{i}_z']] = radius * np.cos(tilt)
        if lm.vary_time:
            x[:, column['geocenter_time']] = get('geocenter_time')
        return x