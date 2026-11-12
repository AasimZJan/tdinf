#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import json
import pandas as pd
from multiprocessing import Pool, cpu_count
from contextlib import closing
import os

try:
    import emcee
except ImportError:          
    emcee = None

import tdinf.utils as utils
from tdinf.utils import sampler_utils

    
def create_run_sampler_arg_parser():
    """
    Parse arguments.  
    """
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Which sampler, and its settings
    p.add_argument('--sampler', default='bilby_mcmc', choices=sorted(SAMPLERS),
                   help='Which sampler to run. Default: bilby_mcmc')
    p.add_argument('--sampler-kwargs', default=None,
                   help='Settings of the sampler as a JSON string, e.g. \'{"nsamples": 2000, "ntemps": 6}\', or the path '
                        'of a JSON file. They are passed to the sampler as they are (true/false/null in JSON).')
    p.add_argument('--use-physical-coordinates-for-sampling', action='store_true', help='Sampler samples the physical parameters')

    # Where to save the output; the .dat file is written next to it
    p.add_argument('-o', '--output', '--output-h5', dest='output', required=True,
                   help='Output path. The label of the run is its file name without extension, the folder is where the '
                        'sampler writes its files, and the posterior is saved as <path without extension>.dat. '
                        '(--output-h5 is accepted as the old name of this option.)')

    # Whether to run pre-Tcut, post-Tcut, or full (Tstart to Tend)?
    p.add_argument('-m', '--mode', required=True,
                   help='Options: "pre", "post", or "full"')

    # Args for cutoff (defined in # of cycles OR seconds from merger; up to user)
    p.add_argument('-t', '--Tcut-cycles', type=float, default=None,
                   help='Cutoff time in cycles from peak.')
    p.add_argument('-ts', '--Tcut-seconds', type=float, default=None,
                   help='Cutoff time in seconds from peak.')

    # Start & end times for segment of data to analyze
    p.add_argument('--Tstart', type=float, required=True,
                   help='Start time for full and pre runs.')
    p.add_argument('--Tend', type=float, required=True,
                   help='End time for full and post runs.')

    # Interferometers
    p.add_argument('--ifos', nargs='+', default=['H1', 'L1', 'V1'],
                   help='Interferometers; some combination of H1, L1, and/or V1.')

    # Place where input data is stored
    p.add_argument('--data', default=None, required=True, action='append',
                   help='path to data formatted as --data {ifo}:path/To/psd. '
                        '!! must include one per ifo.')
    p.add_argument('--psd', default=None, required=True, action='append',
                   help='path to data formatted as --psd {ifo}:path/To/psd. '
                        '!! must include one per ifo.')

    # Option to do an injection instead of use real data;
    p.add_argument('--injected-parameters', default=None,
                   help='...')

    # How to generate reference parameters? Can either be from a reference posterior, or reference
    # point. If injected-parameters is passed, those are used as the reference
    p.add_argument('--reference-parameters', default=None,
                   help='json of parameters that initialize '
                        '1) how 0 is defined in the time cuts '
                        '2) the initialization of prior draw points')
    p.add_argument('--reference-posterior-file', default=None,
                   help='posterior file containing pe samples, '
                        ' used only if injected-parameters==None and reference-parameters=None')
    p.add_argument('--reference-parameter-method', default='tightest_time_posterior',
                   help='how to get reference parameter from reference posterior; '
                        'options = "tightest_time_posterior" or "maxL"')

    # Start the sampler from an existing result
    p.add_argument('--initial-walkers', default=None,
                   help='File or folder to start the sampler from: a tdinf posterior (.dat), the h5 backend of an emcee run '
                        '(its last sample), or a .npy file of positions in sampling coordinates. '
                        'If None, the reference posterior (if any) or the reference point is used.')
    p.add_argument('--initial-walker-type', default='walkers', choices=['posterior', 'backend', 'walkers'],
                   help='What --initial-walkers is. If not given it is inferred: .dat or a folder = posterior, '
                        '.h5 = backend, .npy = walkers.')

    # Optional args for waveform/data settings
    p.add_argument('--approx', default='NRSur7dq4',
                   help='Waveform approximate string. Default: NRSur7dq4')
    p.add_argument('--sampling-rate', type=int, default=2048)
    p.add_argument('--flow', type=float, default=11,
                   help="Lower frequency bound for data conditioning and likelihood function (ACF). "
                        "Default: 11 Hz")
    p.add_argument('--fmax', type=float, default=None,
                   help="Upper frequency bound for data conditioning and likelihood function (ACF). "
                        "Default: None")
    p.add_argument('--f22-start', type=float, default=11,
                   help="frequency at which to start generating 22 mode for waveforms. "
                        "Default: 11 Hz")
    p.add_argument('--fref', type=float, default=11,
                   help='Reference frequency. Default: 11 Hz')

    # Do we want to run with only the prior?
    p.add_argument('--only-prior', action='store_true',
                   help='Include flag if you want to run with only the prior.')

    # Do we want to sample in time and/or sky position?
    p.add_argument('--vary-time', action='store_true',
                   help='Include flag if you want to sample over time parameter.')
    p.add_argument('--vary-skypos', action='store_true',
                   help='Include flag if you want to sample over sky position parameters '
                        '(right ascension, declination, and polarization).')

    # Do we want to sample in eccentric parameters?
    p.add_argument('--eccentric-analysis', action='store_true',
                   help='Include flag if you want to sample over eccentric parameters.')

    # Do we want an aligned spin analysis?
    p.add_argument('--aligned-spin-analysis', action='store_true',
                   help='Include flag if you want to the sampler to assume the analysis is aligned-spin.')

    # Set up prior bounds
    p.add_argument('--total-mass-prior-bounds', type=float, nargs=2, default=[200, 350],
                   help="detector frame total mass bounds (in solar masses) for total_mass prior."
                        "Default: [200,350]")
    p.add_argument('--mass-ratio-prior-bounds', type=float, nargs=2, default=[0.17, 1],
                   help="mass ratio bounds for mass ratio prior."
                        "Default: [0.17, 1]")
    p.add_argument('--luminosity-distance-prior-bounds', type=float, nargs=2, default=[100, 10000],
                   help="luminosity_distance bounds for luminosity_distance prior."
                        "Default: [100, 10000]")
    p.add_argument('--spin-magnitude-prior-bounds', type=float, nargs=2, default=[0, 0.99],
                   help="Spin magnitude bounds for spin magnitude prior."
                        "Default: [0,0.99]")
    p.add_argument('--time-prior-sigma', type=float, default=0.01,
                   help="Standard deviation of time prior [s.]"
                        "Default: 0.01 s")
    p.add_argument('--use-lvk-mass-and-time-prior', default=False, action='store_true',
                   help="Use uniform in detector frame mass and time prior.")
    p.add_argument('--eccentricity-prior-bounds', type=float, nargs=2, default=[0.0, 0.4],
                   help="eccentricity bounds for eccentricity prior."
                        "Default: [0.0, 0.4]")

    # Do we want to resume an old run? (passed on to samplers that can: for bilby_mcmc it sets 'resume')
    p.add_argument('--resume', action='store_true',
                   help='Resume from the sampler\'s own checkpoint files in the output folder, if the sampler supports it. '
                        '!! Recommended to include !!')

    # Debugging
    p.add_argument('--verbose', action='store_true',
                   help='Include flag if you want code to print out additional information as it '
                        'runs. Helpful for debugging.')

    # Options of the emcee-only version of this script. They are NOT used any more (sampler settings go in --sampler-kwargs), but
    # group_postprocess.py and waveform_h5s.py read the command_line.sh of earlier runs with THIS parser, and those lines contain
    # them. They are accepted, hidden from --help and ignored (main() warns if one is set).
    p.add_argument('--nwalkers', type=int, default=512, help=argparse.SUPPRESS)
    p.add_argument('--nsteps', type=int, default=50000, help=argparse.SUPPRESS)
    p.add_argument('--ncpu', type=int, default=4, help=argparse.SUPPRESS)
    p.add_argument('--produce-diagnostics', action='store_true', help=argparse.SUPPRESS)

    return p

###########################################################################################
# Common functions
###########################################################################################

def parse_sampler_kwargs(text):
    """
    --sampler-kwargs as a dictionary. `text` is a JSON string or the path of a JSON file; None or '' gives {}.
    """
    if text is None or str(text).strip() == '':
        return {}
    if os.path.isfile(text):
        with open(text) as handle:
            content = json.load(handle)
    else:
        try:
            content = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError(f"--sampler-kwargs is neither a file nor valid JSON ({error}). "
                             "Use double quotes, and true / false / null: --sampler-kwargs '{\"nsamples\": 2000}'") from error
    if not isinstance(content, dict):
        raise ValueError(f"--sampler-kwargs must be a JSON object (a dictionary), not {type(content).__name__}")
    return content


def modify_parameters(data, args, all_params=True):

    # Figure out which format `data` has, and convert it to a pandas DataFrame
    # if it is not already one.
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    elif isinstance(data, dict):
        df = pd.DataFrame([data])  # Convert the dictionary to a one-row DataFrame
    elif isinstance(data, np.ndarray): 
        df = pd.DataFrame({k:data[k] for k in data.dtype.names})
    else:
        raise ValueError(f"Input to `modify_parameters` is {type(data)}: must be either "\
                           "a DataFrame, dictionary, or array with names.")

    def equivocate_columns(dataframe, wanted_key, maybe_key, verbose=False):
        """
        Often the same parameters will have the different names for different codes. 
        This function checks for and sets both names (`wanted key` and `maybe_key`)
        and equivocates them. This is needed for, e.g., comparing to a `bilby` result.
        """
        if wanted_key not in dataframe.columns:
            if maybe_key in dataframe.columns:
                dataframe[wanted_key] = dataframe[maybe_key]
            else:
                if verbose:
                    print(f'warning! neither {wanted_key} nor {maybe_key} in df.columns!')

    # For the case where we only need reference_parameters, bare minimum needed 
    # are time and skyposition.
    equivocate_columns(df, 'geocenter_time', 'geocent_time')
    equivocate_columns(df, 'right_ascension', 'ra')
    equivocate_columns(df, 'declination', 'dec')
    equivocate_columns(df, 'polarization', 'psi')

    # Otherwise, need to make sure we have all parameters.
    if all_params:
        
        equivocate_columns(df, 'mean_anomaly', 'mean_anomaly_periastron')
        equivocate_columns(df, 'luminosity_distance', 'distance_mpc')
    
        if 'f_ref' not in df.columns:
            df['f_ref'] = args.fref
    
        if 'mass_1' not in df.columns and all(col in df.columns for col in ['total_mass', 'mass_ratio']):
            df['mass_1'], df['mass_2'] = utils.m1m2_from_mtotq(df['total_mass'], df['mass_ratio'])
    
        if 'total_mass' not in df.columns:
            df['total_mass'] = df['mass_1'] + df['mass_2']
    
        if 'mass_ratio' not in df.columns:
            df['mass_ratio'] = df['mass_2'] / df['mass_1']
    
        spin_component_keys = ['inclination', 'spin1_x', 'spin1_y', 'spin1_z', 'spin2_x', 'spin2_y', 'spin2_z']
        if not all(key in df.columns for key in spin_component_keys):
            df[['inclination', 'spin1_x', 'spin1_y', 'spin1_z', 'spin2_x', 'spin2_y', 'spin2_z']] = df.apply(
                lambda row: pd.Series(utils.transform_spins(
                    row['theta_jn'], row['phi_jl'],
                    row['tilt_1'], row['tilt_2'],
                    row['phi_12'], row['a_1'], row['a_2'],
                    row['mass_1'], row['mass_2'],
                    row['f_ref'], row['phase'])
                ), axis=1)
        else:
            df[['theta_jn', 'phi_jl', 'tilt_1', 'tilt_2', 'phi_12', 'a_1', 'a_2']] = df.apply(
                lambda row: pd.Series(utils.transformPrecessingWvf2PE(
                    row['inclination'],
                    row['spin1_x'], row['spin1_y'], row['spin1_z'],
                    row['spin2_x'], row['spin2_y'], row['spin2_z'],
                    row['mass_1'], row['mass_2'],
                    row['f_ref'], row['phase'])
                ), axis=1)
            
        equivocate_columns(df, 'spin1_magnitude', 'a_1')
        equivocate_columns(df, 'spin2_magnitude', 'a_2')

    # return in original format
    if isinstance(data, pd.DataFrame):
        return df
    elif isinstance(data, dict):
        return df.to_dict(orient='records')[0]  # Convert back to dictionary
    elif isinstance(data, np.ndarray):
        return df.to_dict(orient='list')
        
def initialize_kwargs(args, reference_parameters):
    # Check that a cutoff time is given
    assert args.Tcut_cycles is not None or args.Tcut_seconds is not None, "must give a cutoff time"

    # Check that the given mode is allowed
    run_mode = args.mode
    assert run_mode in ['full', 'pre', 'post'], f"mode must be 'full', 'pre', or 'post'. given mode = '{run_mode}'."

    """
    Arguments for the posterior function
    """

    # put all arguments into a dict
    kwargs = {
        'mtot_lim': args.total_mass_prior_bounds,
        'q_lim': args.mass_ratio_prior_bounds,
        'chi_lim': args.spin_magnitude_prior_bounds,
        'dist_lim': args.luminosity_distance_prior_bounds,
        'sigma_time': args.time_prior_sigma,
        'approx': args.approx,
        'f_ref': args.fref,
        'f_low': args.flow,
        'f22_start': args.f22_start,
        'delta_t': 1 / args.sampling_rate,
        'right_ascension': reference_parameters['right_ascension'],
        'declination': reference_parameters['declination'],
        'polarization': reference_parameters['polarization'],
        'geocenter_time': reference_parameters['geocenter_time'],
        'verbose': args.verbose,
    }

    if args.eccentric_analysis:
        kwargs.update({'ecc_lim': args.eccentricity_prior_bounds})

    return kwargs

def get_injected_parameters(args, initial_run_dir='', verbose=False):
    """
    Loads in injection parameters from args.injected_paramters or, if there was no injection,
    loads in a reference sample from args.reference_posterior_file
    """

    if (args.injected_parameters is None) and (args.reference_parameters is None) and (args.reference_posterior_file is None):
        raise ValueError("WARNING: none of --injected-parameters, "
                         "--reference_parameters, or --reference-posterior-file were given. "
                         " These parameters are needed in order to set up time cuts and initialize the sampler."
                         " Please provide one!")

    # if using real data
    if args.injected_parameters is None:
        
        # Use reference parameter from a full PE file...
        if args.reference_parameters is None:
            reference_posterior_file = os.path.join(initial_run_dir, args.reference_posterior_file)
            ref_pe_samples = utils.get_pe_samples(reference_posterior_file)
            reference_parameters = utils.get_reference_parameters_from_posterior(ref_pe_samples, args.reference_parameter_method)
            all_params = True
            
        # Set reference parameters to the passed in reference_parameters
        else:
            reference_parameters = utils.parse_injected_parameters(args.reference_parameters,
                                                                  initial_run_dir=initial_run_dir)
            ref_pe_samples = None
            all_params = False

        if 'f_ref' not in reference_parameters.keys():
            reference_parameters['f_ref'] = args.fref    
            
    # Else, generate an injection (currently, only set up for no noise case)
    else:
        # Load in injected parameters
        reference_parameters = utils.parse_injected_parameters(args.injected_parameters, initial_run_dir=initial_run_dir)

        # Print a warning if that the reference freqs for inj/recovery do not line up
        if reference_parameters['f_ref']!= args.fref:
            print(f"WARNING: fref={reference_parameters['f_ref']} does not equal sampler fref={args.fref}")
        
        ref_pe_samples = None
        all_params = True

    reference_parameters = modify_parameters(reference_parameters, args, all_params=all_params)

    if verbose:
        print('reference_parameters are', reference_parameters)
        
    return reference_parameters, ref_pe_samples

def get_initial_walkers(likelihood_manager, args, nwalkers, ndim,
                        reference_parameters, ref_pe_samples, verbose=True): 
    '''
    Function to get the correct posterior distribution (or None) for walker initialization
    '''
    print('initial walkers folder:', args.initial_walkers)

    # Case 1: no initial_walkers_distribution passed via args, then default to the reference PE samples
    if args.initial_walkers is None: 

        # Case 1A: reference PE samples is also none
        if ref_pe_samples is None: 
            print('Initializing walkers based on an injection or reference point.')
            initial_walker_dist = None

        # Case 1B: reference PE samples used to initialize
        else:
            print('Initializing walkers based on a reference posterior.')
            initial_walker_dist = modify_parameters(ref_pe_samples, args)
            
    # Case 2: initial_walkers_distribution is passed, then we need to load in that file
    else:
        assert args.initial_walker_type is not None, 'if passed args.initial_walkers, must also pass args.initial_walker_type!'

        # path to file or folder for initial walkers
        initial_walker_path =  args.initial_walkers
        
        # Case 2A: the initial walkers dist is a posterior 
        if args.initial_walker_type == "posterior":

            if '.dat' in initial_walker_path:  # if passed a posterior file
                draw_from = initial_walker_path
            else:  # if passsed a folder
                which_run = args.output.split('/')[0]
                draw_from = os.path.join(initial_walker_path, which_run, which_run+'.dat') 
                
            print('Drawing initial walkers from reference posterior:', draw_from)
            initial_walker_dist = utils.get_pe_samples(draw_from)
            
            # Make sure all the parameter labels align with the TD code
            initial_walker_dist = modify_parameters(initial_walker_dist, args)
            
        # Case 2B: initial walkers drawn from a reference backend (different from the one
        # saved for this round of inference). This is useful for picking up an old run without
        # the backend getting unfeasible big.
        elif args.initial_walker_type == "backend": 

            if '.h5' in initial_walker_path: # if passed a backend file 
                draw_from = initial_walker_path
            else: # if passed a folder:
                which_run = args.output.split('/')[0]
                draw_from = os.path.join(initial_walker_path, which_run, which_run+'.h5')
            
            print('Drawing initial walkers from a reference backend:', draw_from)
            ref_backend = emcee.backends.HDFBackend(draw_from, read_only=True)
            p0 = ref_backend.get_last_sample()
            return p0

        # Case 2C: directly pass the distribution of initial walkers
        elif args.initial_walker_type == "walkers": 

            assert '.npy' in initial_walker_path, "Must load in a .npy file for custom intial walkers."

            print('Initial walkers positions given directly in the file', initial_walker_path)
            p0 = np.load(initial_walker_path)

            # check that the shape is correct
            err_msg =  f"shape of initial walkers must be (nwalkers, ndim) = {(nwalkers, ndim)}, but instead was {p0.shape}"
            assert p0.shape==(nwalkers, ndim), err_msg
            
            return p0

    # If not Case 2B or 2C, initialize walkers using the function in the likelihood_manager, and return those
    p0 = likelihood_manager.log_prior.initialize_walkers(
        nwalkers, reference_parameters, reference_posterior=initial_walker_dist, verbose=True, use_lvk_mass_and_time_prior=args.use_lvk_mass_and_time_prior
    )
    return p0

def get_conditioned_time_and_data(args, wf_manager, reference_parameters, initial_run_dir='', verbose=False):
    # Check that a cutoff time is given
    assert args.Tcut_cycles is not None or args.Tcut_seconds is not None, "must give a cutoff time"

    # Check that the given mode is allowed
    run_mode = args.mode
    assert run_mode in ['full', 'pre', 'post'], f"mode must be 'full', 'pre', or 'post'. given mode = '{run_mode}'."

    # Unpack some basic parameters
    ifos = args.ifos
    data_path_dict, psd_path_dict = utils.parse_data_and_psds(args, initial_run_dir)

    """
    Load or generate strain data
    """

    # Either reads injection file or finds maxL parameters from
    tpeak_geocent = reference_parameters['geocenter_time']

    # PSDs
    pe_psds = {}
    for ifo in ifos:
        pe_psds[ifo] = np.genfromtxt(psd_path_dict[ifo], dtype=float)

    # If real data ...
    if args.injected_parameters is None:
        # Load data
        raw_time_dict, raw_data_dict = utils.load_raw_data(ifos=ifos, path_dict=data_path_dict, verbose=verbose)

    # Else, generate an injection (currently, only set up for no noise case)
    else:
        # Times
        raw_time_dict, _ = utils.load_raw_data(ifos=ifos, path_dict=data_path_dict, verbose=verbose)
        if verbose:
            print('reference_injection params: ', reference_parameters)
        raw_data_dict = wf_manager.get_projected_waveform(x_phys=reference_parameters, ifos=args.ifos,
                                                          time_dict=raw_time_dict,
                                                          f_ref=args.fref, f22_start=args.f22_start)

    # tcut = cutoff time in waveform
    if args.Tcut_seconds is not None:
        # option 1: truncation time given in seconds already
        tcut_geocent = tpeak_geocent + args.Tcut_seconds
    else:
        # option 2: find truncation time based off of # number of cycles from peak
        Ncycles = args.Tcut_cycles

        projected_waveform_dict = wf_manager.get_projected_waveform(reference_parameters,
                                                                    ifos=args.ifos,
                                                                    time_dict=raw_time_dict,
                                                                    f_ref=args.fref, f22_start=args.f22_start)

        tcut_geocent = utils.get_Tcut_from_Ncycles(projected_waveform_dict, raw_time_dict,
                                                   ifo="H1", Ncycles=Ncycles,
                                                   ra=reference_parameters['right_ascension'],
                                                   dec=reference_parameters['declination'])

    if verbose:
        print(f'\nCutoff time: {tcut_geocent}')
    tcut_dict, _ = utils.get_tgps_and_ap_dicts(tcut_geocent, ifos,
                                               reference_parameters['right_ascension'],
                                               reference_parameters['declination'],
                                               reference_parameters['polarization'])

    """
    Condition data
    """
    # icut = index corresponding to cutoff time
    time_dict, data_dict, icut_dict = utils.condition(raw_time_dict, raw_data_dict, tcut_dict,
                                                      args.sampling_rate, f_min=args.flow * 3 / 4,
                                                      f_max=args.fmax, verbose=verbose)

    # Time spacing of data
    dt = time_dict[ifos[0]][1] - time_dict[ifos[0]][0]

    # Decide how much data to analyze based of off run mode
    if run_mode == 'full':
        TPre = tcut_geocent - args.Tstart
        TPost = args.Tend - tcut_geocent
    elif run_mode == 'pre':
        TPre = tcut_geocent - args.Tstart
        TPost = 0
    elif run_mode == 'post':
        TPre = 0
        TPost = args.Tend - tcut_geocent
    else:
        raise NotImplementedError(f'Run mode {run_mode} is not defined, please use one of pre post or full')

    if TPre < 0:
        print(f"Warning! Seconds analyzed before cut is less than 0!: {TPre} from {args.Tstart} to {tcut_geocent}")
    if TPost < 0:
        print(f"Warning! Seconds analyzed after cut is less than 0!: {TPost} from {tcut_geocent} to {args.Tend}")

    # Duration --> number of time samples to look at
    # Note Npre cannot be greater than idx or else our times index from [-1: idx],
    # (so itll be empty), to avoid this off by 1 error we use np.floor
    Npre = int(np.floor(TPre / dt))
    Npost = int(
        round(TPost / dt)) + 1  # must add one so that the target time is actually included, even if Tpost = 0,
    # otherwise WF placement gets messed up
    Nanalyze = Npre + Npost
    Tanalyze = Nanalyze * dt
    if verbose:
        print('\nWill analyze {:.3f} s of data at {:.1f} Hz\n'.format(Tanalyze, 1 / dt))
    assert Tanalyze > 0, "Geocenter cut time must be between Tstart and Tend. Please Modify your run settings." \
                         f" Start to end: {args.Tstart}, {args.Tend} with cut at {tcut_geocent}"

    # Crop analysis data to specified duration.
    for ifo, idx in icut_dict.items():
        if Npre > idx:
            print("ERROR! You cannot have more points pre-cutoff time than there are points "
                    "between the start and the cutoff time")
        # idx = sample closest to desired time
        time_dict[ifo] = time_dict[ifo][idx - Npre:idx + Npost]
        data_dict[ifo] = data_dict[ifo][idx - Npre:idx + Npost]

    return time_dict, data_dict, pe_psds


def make_waveform_manager(args, **kwargs):
    '''
    Generate waveform manager (see utils.likelihood.WaveformManger) from args and kwargs
    '''
    return utils.WaveformManager(
        args.ifos,vary_time=args.vary_time,vary_skypos=args.vary_skypos,eccentric_analysis=args.eccentric_analysis,aligned_spin_analysis=args.aligned_spin_analysis, 
        **kwargs
    )

def build_likelihood_manager(args, kwargs, psd_dict, time_dict, data_dict):
    """
    The likelihood manager, shared by every sampler. It holds the data, the ACFs, the prior and the --use-lvk-mass-and-time-prior
    flag (get_log_posterior and the post-processing read the flag from it).
    """
    return utils.LnLikelihoodManager(
        psd_dict=psd_dict, time_dict=time_dict,
        data_dict=data_dict,
        vary_time=args.vary_time, vary_skypos=args.vary_skypos,
        f_max=args.fmax,
        only_prior=args.only_prior,
        use_lvk_mass_and_time_prior=args.use_lvk_mass_and_time_prior,
        eccentric_analysis=args.eccentric_analysis, aligned_spin_analysis=args.aligned_spin_analysis,
        **kwargs
    )

###########################################################################################
# bilby_mcmc function
###########################################################################################

def bilby_number_of_chains(sampler_settings):
    """
    How many chains bilby_mcmc runs for these settings (as documented): nensemble * ntemps if pt_ensemble is True, otherwise
    nensemble + ntemps - 1 (only the first ensemble chain is tempered). An explicit initial_betas ladder sets the number of temperatures.
    """
    ntemps = len(sampler_settings['initial_betas']) if sampler_settings.get('initial_betas') is not None else int(sampler_settings.get('ntemps', 1))
    nensemble = int(sampler_settings.get('nensemble', 1))
    return nensemble * ntemps if sampler_settings.get('pt_ensemble', False) else nensemble + ntemps - 1


def get_bilby_initial_samples(likelihood_manager, args, sampler_settings, reference_parameters, ref_pe_samples, likelihood=None, max_tries=20):
    """
    One starting point for every chain of the run (see bilby_number_of_chains), taken from the posterior, or None.

    * No posterior to start from (no --initial-walkers and no reference posterior): None, so that bilby draws from the prior itself.
    * Otherwise the points come from that posterior / backend / walkers through get_initial_walkers (so --initial-sample-select applies),
      and every one of them has a finite log posterior (impossible points are replaced by random draws; if that does not work after
      max_tries an error says where to look). The FIRST point is drawn on its own, so that --initial-sample-select applies to it exactly
      (with max_posterior it is the best row): it is the one bilby_mcmc uses. The others are drawn together; with a selection other than
      random they are drawn, with repeats possible, from the best rows.

    Returns a list of dictionaries in the coordinates bilby samples: the sampling coordinates (the keys of sampled_keys), or the physical
    parameters if `likelihood` was built with use_physical_coordinates_for_sampling=True (its physical_parameters_from_sampling converts).

    NOTE: bilby_mcmc takes ONE initial_sample_dict, not one per chain (its documentation, and its log, which prints a single initial
    sample even for 16 chains). So only the first point can be handed to it; the others are for samplers that take a whole set
    (emcee, eryn) or for independent runs.
    """
    if args.initial_walkers is None and ref_pe_samples is None:
        return None
    lm = likelihood_manager
    n_chains = bilby_number_of_chains(sampler_settings)
    def draw_finite(count):
        points = get_initial_walkers(lm, args, count, lm.num_parameters, reference_parameters, ref_pe_samples)
        return [x for x in points if np.isfinite(lm.get_log_posterior(x))]

    starts = draw_finite(1)                      # the first start alone, following --initial-sample-select exactly
    if n_chains > 1:
        starts += draw_finite(n_chains - 1)
    tries = 0
    while len(starts) < n_chains and tries < max_tries:
        starts += draw_finite(n_chains - len(starts))           # replacements for impossible points
        tries += 1
    if len(starts) < n_chains:
        raise RuntimeError(
            f"only {len(starts)} of the {n_chains} starting points have a finite log posterior after {max_tries} attempts. The starting "
            "posterior is probably outside the prior bounds (check --total-mass-prior-bounds, --mass-ratio-prior-bounds, "
            "--spin-magnitude-prior-bounds, --luminosity-distance-prior-bounds, --eccentricity-prior-bounds and the time prior).")
    if likelihood is not None and likelihood.use_physical:
        return [likelihood.physical_parameters_from_sampling(x) for x in starts]
    return [{key: float(value) for key, value in zip(lm.sampled_keys, x)} for x in starts]


def save_posterior(samples, ln_posterior, likelihood_manager, output, **kwargs):
    """
    Turn samples (n, ndim) in sampling coordinates and their ln posterior into the usual tdinf table (physical parameters,
    ln_posterior, ln_prior, ln_likelihood, SNRs) and write <output without extension>.dat
    """
    df = utils.postprocessing_get_complete_samples_dict(np.asarray(samples), np.asarray(ln_posterior), likelihood_manager, **kwargs)
    sample_path = os.path.splitext(output)[0] + '.dat'
    df.to_csv(sample_path, sep=' ', index=False)
    print("File saved: %r" % sample_path)
    return df

def make_bilby_swaps_continue():
    """
    Make the temperature swaps of bilby_mcmc carry the exchanged state forward.

    In the bilby_mcmc code this was checked against, an accepted swap does `chain[-1] = other_state`, which rewrites only the STORED last row of the two chains.
    Each chain goes on stepping from `chain.current_sample`, which is not updated, so the exchanged states never become the chains' states: the hot chains do not
    feed the cold chains, and the recorded cold-chain rows at the swap iterations are copies from the neighbouring temperature (the pickled chains of a real run showed
    this directly: after 41-47% of the swap iterations the cold chain continued from its own old state, matching the swap acceptance).
    This wraps `swap_tempered_chains`: bilby's own swap runs first, then every chain continues from the state stored in its last row. For a chain that did not swap, that
    row equals its current sample, so nothing changes; for a bilby that already continues from the swapped state it is a no-op. Only the main process is affected (the
    swaps are done there; the workers only step the chains). Returns True if the wrapper was installed now, False if it already was, None if bilby's class is not available.
    """
    try:
        from bilby.bilby_mcmc.sampler import BilbyPTMCMCSampler
    except ImportError:
        return None
    bilby_swap = BilbyPTMCMCSampler.swap_tempered_chains
    if getattr(bilby_swap, '_tdinf_continues', False):
        return False

    def swap_and_continue(self):
        bilby_swap(self)
        for samplers in self.sampler_dictionary.values():
            for sampler in samplers:
                sampler.chain.current_sample = sampler.chain[-1]

    swap_and_continue._tdinf_continues = True
    BilbyPTMCMCSampler.swap_tempered_chains = swap_and_continue
    return True

def make_bilby_chains_ship_light(headroom=1000):
    """
    Send each bilby_mcmc chain to the worker processes without its unused empty rows.

    With a pool, bilby pickles every sampler, chain included, to a worker and back at EVERY iteration. A chain stores its samples in an array that is allocated in blocks of
    100000 rows, so all of it is sent although only the first `position + 1` rows are filled (a checkpoint of a 16-chain run is 220 MB at 34000 rows). This gives `Chain` a
    __getstate__ that keeps only the filled rows and a __setstate__ that restores `headroom` empty rows (and extends by that much when they run out), so the amount sent is
    that of the samples actually stored. The chain itself is unchanged, and older checkpoints load as before. It has to be installed before the pool is created, so that the
    workers have it too, which holds for the default 'fork' start method; with 'spawn' only the trip to the workers is lighter.
    Returns True if installed now, False if it already was, None if bilby's class is not available.
    """
    try:
        from bilby.bilby_mcmc.chain import Chain
    except ImportError:
        return None
    if getattr(Chain, '_tdinf_light', False):
        return False

    def __getstate__(self):
        state = self.__dict__.copy()
        filled = self.position + 1
        state['_chain_array'] = self._chain_array[:filled]
        state['_chain_array_length'] = filled
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        filled = self.position + 1
        columns = self._chain_array.shape[1]
        self._chain_array = np.concatenate((self._chain_array[:filled], np.zeros((headroom, columns))), axis=0)
        self._chain_array_length = len(self._chain_array)
        self.block_length = headroom

    Chain.__getstate__ = __getstate__
    Chain.__setstate__ = __setstate__
    Chain._tdinf_light = True
    return True

def run_bilby_mcmc(likelihood_manager, args, kwargs, sampler_kwargs, reference_parameters, ref_pe_samples):
    """
    Run bilby_mcmc. 
    """
    try:
        from bilby.bilby_mcmc.sampler import Bilby_MCMC
    except ImportError as error:
        raise ImportError("--sampler bilby_mcmc needs bilby") from error
    import inspect
    import bilby

    # arguments
    outdir = os.path.dirname(args.output) or '.'
    label = os.path.splitext(os.path.basename(args.output))[0]
    os.makedirs(outdir, exist_ok=True)

    # Arguments of the Bilby_MCMC constructor
    CONSTRUCTOR = dict(
        outdir=outdir,                   # where checkpoint files and outputs go
        label=label,                     # prefix of the output files
        use_ratio=False,                 # True if log_likelihood() returns a log likelihood RATIO
        skip_import_verification=True,   # (True)
        check_point_plot=False,          # (True) make progress plots at every checkpoint; slow, so off here
        diagnostic=True,                 # (False) extra deep-diagnostic plots for convergence problems
        resume=bool(args.resume),        # (True) resume from checkpoint files found in outdir
        exit_code=130,                   # (130) exit code used when the run is interrupted
        verbose=True,                    # (True) print progress
        npool=1,                         # processes sharing the chains; choose a divisor of the number of chains
    )

    # Sampler arguments (this is Bilby_MCMC.default_kwargs
    SAMPLER = dict(
        # --- how many samples, burn-in and thinning
        nsamples=1000,                   # (1000) number of samples to draw
        burn_in_nact=10,                 # (10) autocorrelation times (ACTs) discarded as burn-in
        thin_by_nact=1,                  # (1) thinning factor in ACTs; below 1 keeps correlated samples
        fixed_discard=0,                 # (0) steps discarded in addition to the burn-in
        # --- the chains
        nensemble=1,                     # (1) ensemble chains run with periodic communication
        ntemps=1,                        # (1) parallel-tempered chains
        pt_ensemble=False,               # (False) True: a tempered set per ensemble chain (nensemble x ntemps chains)
        Tmax=None,                       # (None) maximum temperature of the ladder
        Tmax_from_SNR=20,                # (20) alternative to Tmax: an SNR to estimate a suitable Tmax from
        initial_betas=None,              # (None) alternative to both: an explicit inverse-temperature ladder
        adapt=True,                      # (True) adapt the temperature ladder
        adapt_t0=100,                    # (100) adaptation time-lag
        adapt_nu=10,                     # (10) adaptation rate
        pt_rejection_sample=False,       # (False) rejection-sample the tempered chains when drawing samples
        # --- steps and convergence
        L1steps=100,                     # (100) internal steps per iteration; the ACT is counted in iterations
        L2steps=3,                       # (3) iterations between swaps of the tempered and ensemble chains
        autocorr_c=5,                    # (5) minimum number of ACTs needed to trust the ACT estimate
        min_tau=1,                       # (1) smallest ACT accepted; can be used to force a larger one
        fixed_tau=None,                  # (None) a fixed ACT instead of an estimated one (see docstring)
        tau_window=None,                 # (None) window used for the ACT estimate (see docstring)
        stop_after_convergence=False,    # (False) with tempering: stop updating the chains once they have converged
        # --- proposals
        proposal_cycle="default",        # ("default") built-in cycle, or a ProposalCycle object
        # --- evidence (only meaningful with ntemps > 1)
        evidence_method="stepping_stone",  # ("stepping_stone") see docstring for the alternatives
        # --- the starting point
        initial_sample_method="prior",   # ("prior") how the starting point is drawn (see docstring for the other methods)
        initial_sample_dict=None,        # (None) a dict of parameter values that overrides the drawn starting point; may be partial
        # --- printing and checkpoints
        printdt=1800,                       # (60) seconds between progress prints AND between convergence checks
        check_point_delta_t=1800,        # (1800) seconds between checkpoints
        # tdinf arguments for bilby_mcmc
        tdinf_fix_swaps=True,
        tdinf_light_chains=True,
    )

    # Apply user-specified sampler kwargs
    for key, value in sampler_kwargs.items():
        if key in CONSTRUCTOR:
            CONSTRUCTOR[key] = value
        elif key in SAMPLER:
            SAMPLER[key] = value

    # the swaps between temperatures must carry the exchanged state forward (see make_bilby_swaps_continue); "tdinf_fix_swaps": false switches this off
    if SAMPLER['ntemps'] > 1 and SAMPLER.get('tdinf_fix_swaps', True):
        installed = make_bilby_swaps_continue()
        print({True: "tempered chains: temperature swaps now carry the exchanged state forward", False: "tempered chains: swap wrapper already installed", None: "WARNING: could not install the swap wrapper (bilby's BilbyPTMCMCSampler not found)"}[installed])

    # with a pool, send the chains to the workers without their unused empty rows (see make_bilby_chains_ship_light); "tdinf_light_chains": false switches this off
    if CONSTRUCTOR['npool'] > 1 and SAMPLER.get('tdinf_light_chains', True):
        installed = make_bilby_chains_ship_light()
        print({True: "pool: chains are now sent to the workers without their unused empty rows", False: "pool: light-chain wrapper already installed", None: "WARNING: could not install the light-chain wrapper (bilby's Chain not found)"}[installed])

    print('Constructor arguments:', CONSTRUCTOR)
    print('Sampler arguments:', SAMPLER)

    physical = bool(args.use_physical_coordinates_for_sampling)                              
    prior_dict = likelihood_manager.log_prior.get_bilby_mcmc_prior_dict(use_lvk_mass_and_time_prior=args.use_lvk_mass_and_time_prior,use_physical_coordinates_for_sampling=physical)
    
    # the proposal cycle "tdinf_gw" ("tdinf_gw_noFM", ...)
    if str(SAMPLER['proposal_cycle']).startswith('tdinf_gw'):
        try:
            from tdinf.utils import sampler_utils
        except ImportError as error:
            raise RuntimeError('proposal_cycle "tdinf_gw" needs tdinf/utils/sampler_utils.py and a bilby with bilby_mcmc') from error
        prior_dict = sampler_utils.TdinfPriorDictForGWProposal(prior_dict)
        sampler_utils.register_tdinf_gw_cycle()
        print(f"proposal cycle {SAMPLER['proposal_cycle']} installed: bilby's gwA for tdinf's parameters; parameters per group "
              f"{ {g: len(getattr(prior_dict, g)) for g in ('intrinsic', 'extrinsic', 'mass', 'spin', 'measured_spin', 'sky')} }")
    
    likelihood = utils.TdinfBilbyLikelihood(likelihood_manager, parameter_keys=list(prior_dict),
                                            use_physical_coordinates_for_sampling=physical, limit_threads=CONSTRUCTOR['npool'] > 1)

    # the starting point, from the posterior if the user gave one (--initial-walkers or a reference posterior); otherwise bilby draws
    # it from the prior. bilby_mcmc takes a single initial_sample_dict, so the first of the starting points is used.
    # if SAMPLER['initial_sample_dict'] is None:
    #     starts = get_bilby_initial_samples(likelihood_manager, args, SAMPLER, reference_parameters, ref_pe_samples, likelihood=likelihood)
    #     if starts is not None:
    #         SAMPLER['initial_sample_dict'] = starts[0]
    #         if len(starts) > 1:
    #             print(f"NOTE: {len(starts)} chains but bilby_mcmc takes a single initial_sample_dict: the first starting point is used.")
    #     else:
    #         print("No posterior to start from: bilby draws the starting point from the prior.")

    print(f"\nbilby_mcmc {bilby.__version__}: {len(prior_dict)} parameters in {'physical' if physical else 'sampling'} coordinates, "
          f"{bilby_number_of_chains(SAMPLER)} chains, outdir={outdir}, label={label}")
    print("use_lvk_mass_and_time_prior:", args.use_lvk_mass_and_time_prior, "| only_prior:", likelihood_manager.only_prior)

    # avoid oversubscribing the cores with BLAS / OpenMP threads in the workers (as the emcee code did)
    if CONSTRUCTOR['npool'] > 1:
        os.environ["OMP_NUM_THREADS"] = "1"

    print("Available cores: %i" % cpu_count())
    print("Running with %i cores" % CONSTRUCTOR['npool'])
    
    # run. save=False: tdinf writes its own .dat file (bilby's result file would also store the custom priors)
    result = bilby.run_sampler(likelihood=likelihood, priors=prior_dict, sampler='bilby_mcmc', save=False, plot=False,
                               **CONSTRUCTOR, **SAMPLER)

    # the samples in SAMPLING coordinates, for the usual post-processing
    posterior = result.posterior
    samples = (likelihood.sampling_from_physical_parameters(posterior) if physical
               else posterior[likelihood_manager.sampled_keys].to_numpy(dtype=float))

    # ln posterior = ln likelihood + tdinf's ln prior at those samples, so ln_prior, ln_likelihood and ln_posterior in the .dat have
    # tdinf's usual convention. The ln likelihood is the one bilby stored, checked against a recomputation of one random sample.
    ln_likelihood = posterior['log_likelihood'].to_numpy(dtype=float) if 'log_likelihood' in posterior else None
    if ln_likelihood is not None and len(samples) and not likelihood_manager.only_prior:
        check = np.random.default_rng().integers(len(samples))
        recomputed = likelihood_manager.get_log_likelihood(likelihood_manager.samp_to_phys(samples[check]))
        if not np.isclose(recomputed, ln_likelihood[check], rtol=0, atol=1e-6):
            print("WARNING: the log likelihood stored by bilby does not match tdinf's: recomputing it")
            ln_likelihood = None
    if ln_likelihood is None:
        print(f"computing the ln likelihood of the {len(samples)} samples")
        ln_likelihood = np.array([0.0 if likelihood_manager.only_prior
                                  else likelihood_manager.get_log_likelihood(likelihood_manager.samp_to_phys(x)) for x in samples])
    ln_prior = np.array([likelihood_manager.log_prior.get_lnprior(x, use_lvk_mass_and_time_prior=args.use_lvk_mass_and_time_prior)
                         for x in samples])
    print(f"bilby_mcmc finished: {len(samples)} samples")
    return samples, ln_likelihood + ln_prior


###########################################################################################
# EMCEE functions
###########################################################################################

EMCEE_MOVES = {'stretch': 'StretchMove', 'de': 'DEMove', 'snooker': 'DESnookerMove', 'walk': 'WalkMove'}

"""
Helpers for emcee multiprocessing.

The likelihood manager holds all of the data, ACFs, PSDs, etc., so pickling it
and sending it along with every emcee task is wasteful. Instead it is handed to
each worker ONCE through the Pool initializer, and stored in a module global.
These must live at module level (not inside main) so they can be pickled by name.
"""

_LM = None        # likelihood manager, set once in each worker process
_VERBOSE = False  # verbose flag forwarded to get_log_posterior


def _init_worker(likelihood_manager, verbose=False):
    """
    Pool initializer: runs once in each worker process.

    Stores the likelihood manager in a module global so it is not re-pickled
    with every task (under fork it is inherited; under spawn it is pickled once
    per worker). Also limits BLAS/OpenMP thread pools to one thread, which, unlike
    setting OMP_NUM_THREADS after numpy is imported, takes effect at runtime.
    """
    global _LM, _VERBOSE
    _LM = likelihood_manager
    _VERBOSE = verbose
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=1)
    except ImportError:
        pass


def _log_prob(x):
    """Log posterior for one walker position; this is what emcee maps over walkers."""
    return _LM.get_log_posterior(x, verbose=_VERBOSE)


class _ChunkOnePool:
    """
    Wrap a multiprocessing Pool so emcee's `pool.map(f, walkers)` hands out one
    walker at a time. A free worker immediately takes the next walker, which
    balances the load when waveform runtimes differ between walkers (the default
    Pool.map gives each worker a chunk of several walkers at once).
    """
    def __init__(self, pool):
        self.pool = pool

    def map(self, func, iterable):
        return self.pool.map(func, iterable, chunksize=1)


def run_emcee(likelihood_manager, args, kwargs, sampler_kwargs, reference_parameters, ref_pe_samples):
    """Run emcee; similar to the version tdinf used it before."""
    try:
        import emcee
    except ImportError as error:
        raise ImportError("--sampler emcee needs emcee (pip install emcee)") from error

    SAMPLER = {'nwalkers':args.nwalkers,
            'nsteps':args.nsteps,
            'ncpu':args.ncpu,
            'check_interval':5000, # can slow down a run as number of steps increase. Keep it high. Exceeding nsteps prevents convergence checking.
            'tau_multiple':100,
            'tau_tolerance':0.01,
            'progress':True,
            'moves':None,
            'produce_diagnostics':bool(args.produce_diagnostics),
            }        

    SAMPLER.update(sampler_kwargs)
    nwalkers, nsteps, ncpu = int(SAMPLER['nwalkers']), int(SAMPLER['nsteps']), int(SAMPLER['ncpu'])
    verbose = args.verbose
    ndim = likelihood_manager.num_parameters
    moves = sampler_utils.emcee_moves(SAMPLER['moves'])

    if args.use_physical_coordinates_for_sampling:
        print("WARNING: --use-physical-coordinates-for-sampling is for bilby_mcmc; emcee samples tdinf's sampling coordinates")
    if verbose:
        print("Sampling %i parameters." % ndim)

    # where emcee saves its output while the sampler runs
    backend_path = args.output if args.output.endswith('.h5') else os.path.splitext(args.output)[0] + '.h5'
    os.makedirs(os.path.dirname(backend_path) or '.', exist_ok=True)
    backend = emcee.backends.HDFBackend(backend_path)

    # Resume if we want
    if args.resume and os.path.isfile(backend_path):
        resumed = False
        try:
            # Load in last sample to use as the new starting walkers
            p0 = backend.get_last_sample()
            # adjust number of steps
            nsteps_already_taken = backend.get_chain().shape[0]
            nsteps = nsteps - nsteps_already_taken
        except AttributeError:
            # This happens when the file has been made but the job crashed before writing any points to it.
            # So we overwrite it
            print('WARNING! Backend was empty, resetting backend')
            try:
                backend.reset(nwalkers, ndim)
            except OSError:
                print("resetting backend did not seem to fix issue, deleting h5 file and making new backend")
                os.remove(backend_path)
                backend = emcee.backends.HDFBackend(backend_path)
                backend.reset(nwalkers, ndim)
            p0 = get_initial_walkers(likelihood_manager, args, nwalkers, ndim, reference_parameters, ref_pe_samples)
    else:
        # Reset the backend or make new backend
        backend.reset(nwalkers, ndim)
        p0 = get_initial_walkers(likelihood_manager, args, nwalkers, ndim, reference_parameters, ref_pe_samples)

    # Avoid oversubscribing cores with BLAS/OpenMP threads. Setting the environment variable here is only a fallback, since numpy is already imported
    # and most libraries read it at load time. The real limit is applied inside each worker by threadpoolctl (see _init_worker).
    if ncpu > 1:
        os.environ["OMP_NUM_THREADS"] = "1"
        try:
            import threadpoolctl  # noqa: F401
        except ImportError:
            print("WARNING: threadpoolctl is not installed, so worker threads are not limited at runtime. Install it, or set OMP_NUM_THREADS=1 in "
                  "the environment before launching python.")
    # for multiprocessing: start the walkers that take the longest first
    def sort_on_runtime(pos):
        p = np.atleast_2d(pos)
        idx = np.argsort(p[:, 0])[::-1]
        return p[idx], idx

    def sample(sampler):
        # If there are still iterations left to run ...
        if nsteps <= 0:
            return
        # This will be useful for testing convergence
        old_tau = np.inf
        # Cycle through remaining iterations
        for _ in sampler.sample(p0, iterations=nsteps, progress=bool(SAMPLER['progress'])):
            # Only check convergence every check_interval steps
            if sampler.iteration % int(SAMPLER['check_interval']):
                continue
            # Compute the autocorrelation time so far. Using tol=0 means that we'll always get an estimate even if it isn't trustworthy
            tau = sampler.get_autocorr_time(tol=0)
            # Check convergence
            converged = np.all(tau * float(SAMPLER['tau_multiple']) < sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < float(SAMPLER['tau_tolerance']))
            if converged:
                print(f"Converged at step {sampler.iteration}")
                break  # if convergence reached before nsteps, stop running
            old_tau = tau

    print("Available cores: %i" % cpu_count())
    print("Running with %i cores" % ncpu)
    
    if ncpu > 1 and nsteps > 0:
        # The likelihood manager is passed to the workers once, via the initializer
        with closing(Pool(processes=ncpu, initializer=_init_worker, initargs=(likelihood_manager, verbose))) as pool:
            # chunksize=1 so free workers pick up the next walker
            sampler = emcee.EnsembleSampler(nwalkers, ndim, _log_prob, backend=backend, pool=_ChunkOnePool(pool), moves=moves,
                                            runtime_sortingfn=sort_on_runtime)
            sample(sampler)
            pool.terminate()
    else:
        # one core (or nothing left to run): no pool, so no worker start-up cost
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lambda x: likelihood_manager.get_log_posterior(x, verbose=verbose), backend=backend, moves=moves)
        sample(sampler)

    # Print dimensions of chain
    print(sampler.get_chain().shape)

   # the burn-in and thinning of the old post-processing; main() then saves the table
    burnin, thin = sampler_utils.emcee_burn_in_and_thin(sampler)
    samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
    ln_posterior = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)

    if SAMPLER['produce_diagnostics']:
        try:
            from tdinf.utils import sampler_diagnostics
            args.output_h5 = backend_path        # the diagnostics read the backend through this name
            sampler_diagnostics.report_from_args(args, kwargs=kwargs, max_steps=20000)
        except Exception as e:
            print(f"WARNING: diagnostics skipped: {e}")
    print(f"emcee finished: {len(samples)} samples")
    return samples, ln_posterior

###########################################################################################
# Eryn functions
###########################################################################################
"""
Eryn and pickling.

With a pool, Eryn sends its log-likelihood function (and the arguments it was given) to the worker processes with every task, so everything it
holds must be picklable, and anything large is re-sent every time. Therefore:
  * the function Eryn is given is the module-level `_eryn_log_like`: it is pickled by NAME (no closure, lambda or bound method);
  * it takes no `args` / `kwargs`: the likelihood manager (data, ACFs, waveform generator) is sent to each worker ONCE by the Pool initializer
    `_init_worker` (the one run_emcee uses), and read from the module global `_LM`; with --ncpu 1 the same global is set in this process;
  * the pool Eryn gets is `_ChunkOnePool`, which is only ever asked for `.map`, so it is never pickled; the prior object, the sampler and the HDF5
    backend stay in the main process.
"""

ERYN_BRANCH = 'tdinf'          # Eryn calls a group of parameters a "branch". tdinf has one branch with one leaf (one signal).
ERYN_BAD_LIKELIHOOD = -1e300   # a failed waveform: very negative but finite, so that beta * lnL is not NaN for a beta = 0 chain


def _eryn_log_like(x):
    """
    ln likelihood of ONE walker for Eryn (Eryn adds the prior itself and tempers only this part). Reads the likelihood manager from the worker
    global. Accepts what Eryn may hand over for one walker with one branch and one leaf: a list with one array per branch, a dict of branches,
    or the array itself, shape (1, ndim) or (ndim,).
    """
    lm = _LM
    if isinstance(x, dict):
        x = next(iter(x.values()))
    elif isinstance(x, (list, tuple)) and len(x) == 1:
        x = x[0]
    x = np.asarray(x, dtype=float)
    ndim = lm.num_parameters
    if x.size != ndim:
        raise ValueError(f"Eryn passed {x.size} values for one walker, expected one leaf of {ndim}: this runner uses one branch with one leaf")
    x = x.reshape(ndim)
    x_phys = lm.samp_to_phys(x)
    # as in get_log_posterior, the prior decides first: outside its support the point is rejected by Eryn's prior and no waveform is made
    if not np.isfinite(lm.log_prior.get_lnprior(x, phys_dict=x_phys, use_lvk_mass_and_time_prior=lm.use_lvk_mass_and_time_prior)):
        return 0.0
    if lm.only_prior:
        return 0.0
    value = lm.get_log_likelihood(x_phys)
    return float(value) if np.isfinite(value) else ERYN_BAD_LIKELIHOOD


def _eryn_prior_class():
    """
    tdinf's prior as an Eryn prior object: logpdf(x) is tdinf's get_lnprior of each row. Built on first use (it needs Eryn) and registered under a
    module-level name so it can be pickled by name. Subclasses ProbDistContainer when Eryn has it (in case Eryn checks the type) without calling its
    __init__, whose signature is not needed: only the documented attributes and the methods Eryn calls are provided.
    """
    if 'TdinfPrior' in globals():
        return globals()['TdinfPrior']
    try:
        from eryn.prior import ProbDistContainer as base
    except ImportError:
        base = object

    class TdinfPrior(base):
        def __init__(self, likelihood_manager):
            self._lm = likelihood_manager
            for name, value in (('ndim', likelihood_manager.num_parameters), ('use_cupy', False), ('return_gpu', False), ('xp', np), ('priors_in', {}), ('priors', [])):
                try:
                    setattr(self, name, value)
                except AttributeError:
                    pass

        def logpdf(self, x):
            lm = self._lm
            x = np.atleast_2d(np.asarray(x, dtype=float))
            return np.array([lm.log_prior.get_lnprior(row, use_lvk_mass_and_time_prior=lm.use_lvk_mass_and_time_prior) for row in x])

        def rvs(self, size=1):
            raise NotImplementedError("tdinf's prior is not sampled through Eryn: the starting points come from get_initial_walkers")

        def ppf(self, x):
            raise NotImplementedError("tdinf's prior has no ppf here")

    TdinfPrior.__qualname__ = 'TdinfPrior'
    TdinfPrior.__module__ = __name__
    globals()['TdinfPrior'] = TdinfPrior
    return TdinfPrior


def _eryn_accepted_kwargs(function, wanted, name, essential=()):
    """
    Only the keyword arguments that the INSTALLED Eryn accepts, so that a difference between Eryn versions is reported clearly. A missing one in
    `essential` is an error that prints the installed signature; any other is dropped with a note.
    """
    import inspect
    signature = inspect.signature(function)
    parameters = signature.parameters
    takes_anything = any(p.kind == p.VAR_KEYWORD for p in parameters.values())
    missing = [k for k in wanted if k not in parameters and not takes_anything]
    lost = [k for k in missing if k in essential]
    if lost:
        raise RuntimeError(f"the installed Eryn's {name} does not accept {lost}, which this runner needs. Its signature is {name}{signature}. "
                           "Please compare it with what run_eryn passes (nwalkers, ndims, log_like_fn, priors, tempering_kwargs, branch_names, ...).")
    if missing:
        print(f"NOTE: the installed Eryn's {name} has no {missing}: not passed")
    return {k: v for k, v in wanted.items() if k not in missing}


def eryn_moves(spec):
    """The `moves` of Eryn's EnsembleSampler from the "moves" setting: None or "stretch" is Eryn's default; "de" is differential evolution (if this Eryn has DEMove)."""
    if spec is None or spec == 'stretch':
        return None
    if spec == 'de':
        import eryn.moves
        if not hasattr(eryn.moves, 'DEMove'):
            raise ValueError("this Eryn has no DEMove")
        return [(eryn.moves.DEMove(), 1.0)]
    raise ValueError(f"the Eryn setting 'moves' must be null, 'stretch' or 'de', not {spec!r}")


def eryn_burn_in_and_thin(cold_chain):
    """The burn-in and thinning of run_emcee (5 x the largest and 0.5 x the smallest autocorrelation time), here of the COLD chain, shape (steps, walkers, ndim)."""
    try:
        tau = emcee.autocorr.integrated_time(cold_chain, c=5, tol=0, quiet=True)
        burnin = max(int(5 * np.max(tau)), 0)
        thin = max(int(0.5 * np.min(tau)), 1)
        print(f'burnin = {burnin}, thin = {thin}')
    except Exception:
        print('WARNING thinning by 1, 0 burnin!')
        burnin = 0
        thin = 1
    return burnin, thin


def run_eryn(likelihood_manager, args, kwargs, sampler_kwargs, reference_parameters, ref_pe_samples):
    """
    Run Eryn, with the structure of run_emcee: the walkers are saved to an HDF5 backend (<output without extension>.h5), --resume continues from its
    last step, and the run stops early once the autocorrelation time of the cold chain has converged. Returns (samples, ln_posterior) in tdinf's
    sampling coordinates after the burn-in and thinning of run_emcee, so main() saves the same .dat file. Only the COLD chain (temperature index 0)
    is returned. Eryn tempers the likelihood only: the likelihood is `_eryn_log_like`, the prior is tdinf's own (TdinfPrior).

    Settings (--sampler-kwargs; the old options --nwalkers, --nsteps and --ncpu, if given, are the defaults):
      nwalkers (512) walkers PER TEMPERATURE, nsteps (50000), ncpu (4: a pool of worker processes; 1 runs in this process),
      ntemps (1: no tempering, like emcee), tempering ({}: more tempering_kwargs of Eryn, for example {"Tmax": 100}),
      check_interval (500), tau_multiple (100), tau_tolerance (0.01): as in run_emcee,
      progress (true), moves (null: Eryn's default; "de").
    """
    global _LM, _VERBOSE
    settings = dict(nwalkers=args.nwalkers, nsteps=args.nsteps, ncpu=args.ncpu, ntemps=1, tempering={}, check_interval=500, tau_multiple=100,
                    tau_tolerance=0.01, progress=True, moves=None)
    unknown = sorted(set(sampler_kwargs) - set(settings))
    if unknown:
        raise ValueError(f"unknown eryn setting(s) {unknown}; valid settings are {sorted(settings)}")
    settings.update(sampler_kwargs)
    nwalkers, nsteps, ncpu, ntemps = int(settings['nwalkers']), int(settings['nsteps']), int(settings['ncpu']), int(settings['ntemps'])
    verbose = args.verbose
    ndim = likelihood_manager.num_parameters
    try:
        import eryn
        from eryn.ensemble import EnsembleSampler
        from eryn.state import State
        from eryn.backends import HDFBackend
    except ImportError as error:
        raise ImportError("--sampler eryn needs eryn (pip install eryn)") from error
    moves = eryn_moves(settings['moves'])

    if args.use_physical_coordinates_for_sampling:
        print("WARNING: --use-physical-coordinates-for-sampling is for bilby_mcmc; eryn samples tdinf's sampling coordinates")
    if args.produce_diagnostics:
        print("NOTE: --produce-diagnostics reads an emcee backend; it is skipped for eryn")
    print(f"eryn {getattr(eryn, '__version__', '?')}: {ntemps} temperature(s) x {nwalkers} walkers, {ndim} parameters")

    # where Eryn saves its output while the sampler runs
    backend_path = args.output if args.output.endswith('.h5') else os.path.splitext(args.output)[0] + '.h5'
    os.makedirs(os.path.dirname(backend_path) or '.', exist_ok=True)

    # Resume if we want
    resumed = False
    if args.resume and os.path.isfile(backend_path):
        backend = HDFBackend(backend_path)
        try:
            state = backend.get_last_sample()        # the last state of the old run is the new starting state
            steps_done = int(backend.iteration)
            if steps_done < 1:
                raise AttributeError('no steps in the backend')
            resumed = True
        except (AttributeError, KeyError, OSError, ValueError):
            # the file was made but the job crashed before writing any steps: start a new one
            print('WARNING! Backend was empty, resetting backend')
            os.remove(backend_path)
    elif os.path.isfile(backend_path):
        os.remove(backend_path)                      # a fresh run replaces the old backend, as backend.reset did for emcee
    if resumed:
        nsteps = nsteps - steps_done
        for name, value in (('nwalkers', nwalkers), ('ntemps', ntemps)):
            in_backend = getattr(backend, name, None)
            if in_backend is not None and int(in_backend) != value:
                print(f"NOTE: the backend has {int(in_backend)} {name}, not {value}: continuing with {int(in_backend)}")
                if name == 'nwalkers':
                    nwalkers = int(in_backend)
                else:
                    ntemps = int(in_backend)
    else:
        backend = HDFBackend(backend_path)
        p0 = get_initial_walkers(likelihood_manager, args, ntemps * nwalkers, ndim, reference_parameters, ref_pe_samples)
        coords = np.asarray(getattr(p0, 'coords', p0), dtype=float).reshape(ntemps, nwalkers, 1, ndim)
        state_arguments = _eryn_accepted_kwargs(State.__init__, dict(coords={ERYN_BRANCH: coords}, inds={ERYN_BRANCH: np.ones((ntemps, nwalkers, 1), dtype=bool)}),
                                                'State.__init__', essential=('coords',))
        state = State(**state_arguments)

    # Avoid oversubscribing cores with BLAS/OpenMP threads (see run_emcee)
    if ncpu > 1:
        os.environ["OMP_NUM_THREADS"] = "1"
        try:
            import threadpoolctl  # noqa: F401
        except ImportError:
            print("WARNING: threadpoolctl is not installed, so worker threads are not limited at runtime. Install it, or set OMP_NUM_THREADS=1 in "
                  "the environment before launching python.")

    # the cold chain of the old steps (a resumed run), and of the new ones as they are made, for the convergence check
    cold_old = np.empty((0, nwalkers, ndim))
    if resumed:
        previous = backend.get_chain()
        previous = np.asarray(previous[ERYN_BRANCH] if isinstance(previous, dict) else previous)
        cold_old = previous.reshape(previous.shape[0], ntemps, nwalkers, ndim)[:, 0]
    cold_new = []

    def make_sampler(pool):
        wanted = dict(nwalkers=nwalkers, ndims=ndim, log_like_fn=_eryn_log_like, priors={ERYN_BRANCH: _eryn_prior_class()(likelihood_manager)},
                      tempering_kwargs=dict(ntemps=ntemps, **settings['tempering']), nbranches=1, branch_names=[ERYN_BRANCH], nleaves_max=1,
                      nleaves_min=1, pool=pool, backend=backend, moves=moves)
        accepted = _eryn_accepted_kwargs(EnsembleSampler.__init__, {k: v for k, v in wanted.items() if v is not None or k in ('pool',)}, 'EnsembleSampler.__init__',
                                         essential=('nwalkers', 'ndims', 'log_like_fn', 'priors', 'branch_names', 'backend', 'tempering_kwargs')
                                         + (('pool',) if pool is not None else ()) + (('moves',) if moves is not None else ()))
        try:
            return EnsembleSampler(**accepted)
        except Exception as error:
            raise RuntimeError(f"Eryn rejected the arguments {sorted(accepted)}: {error!r}. Installed signature: EnsembleSampler{__import__('inspect').signature(EnsembleSampler.__init__)}") from error

    def sample(sampler):
        # If there are still iterations left to run ...
        if nsteps <= 0:
            return
        old_tau = np.inf
        run_arguments = _eryn_accepted_kwargs(sampler.sample, dict(iterations=nsteps, progress=bool(settings['progress'])), 'EnsembleSampler.sample', essential=('iterations',))
        for new_state in sampler.sample(state, **run_arguments):
            cold_new.append(np.array(new_state.branches_coords[ERYN_BRANCH])[0].reshape(nwalkers, ndim))
            iteration = cold_old.shape[0] + len(cold_new)
            # Only check convergence every check_interval steps
            if iteration % int(settings['check_interval']):
                continue
            tau = emcee.autocorr.integrated_time(np.concatenate([cold_old, np.array(cold_new)]), c=5, tol=0, quiet=True)
            converged = np.all(tau * float(settings['tau_multiple']) < iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < float(settings['tau_tolerance']))
            if converged:
                print(f"Converged at step {iteration}")
                break  # if convergence reached before nsteps, stop running
            old_tau = tau

    _LM, _VERBOSE = likelihood_manager, verbose       # for ncpu 1 the function reads them in this process; workers get them from _init_worker
    print("Available cores: %i" % cpu_count())
    print("Running with %i cores" % ncpu)
    if ncpu > 1 and nsteps > 0:
        # The likelihood manager is passed to the workers once, via the initializer
        with closing(Pool(processes=ncpu, initializer=_init_worker, initargs=(likelihood_manager, verbose))) as pool:
            sampler = make_sampler(_ChunkOnePool(pool))
            sample(sampler)
            pool.terminate()
    else:
        sampler = make_sampler(None)
        sample(sampler)

    # What Eryn stored: (steps, temperatures, walkers, leaves, ndim); the layout is checked, because a wrong assumption would silently mix axes
    stored = sampler.get_chain()
    chain = np.asarray(stored[ERYN_BRANCH] if isinstance(stored, dict) else stored)
    ln_like, ln_prior = np.asarray(sampler.get_log_like()), np.asarray(sampler.get_log_prior())
    steps = chain.shape[0]
    if chain.shape != (steps, ntemps, nwalkers, 1, ndim) or ln_like.shape != (steps, ntemps, nwalkers) or ln_prior.shape != (steps, ntemps, nwalkers):
        raise RuntimeError(f"the arrays stored by Eryn do not have the layout run_eryn assumes: chain {chain.shape} (expected {(steps, ntemps, nwalkers, 1, ndim)}), "
                           f"log_like {ln_like.shape} and log_prior {ln_prior.shape} (expected {(steps, ntemps, nwalkers)}). Installed Eryn: {getattr(eryn, '__version__', '?')}")
    if cold_new and not np.allclose(chain[cold_old.shape[0] + len(cold_new) - 1, 0, :, 0, :], cold_new[-1]):
        raise RuntimeError("temperature index 0 of Eryn's chain is not the chain that was followed while sampling: the temperature axis is not as assumed")
    betas = np.asarray(sampler.get_betas()).reshape(-1, ntemps)
    if not np.isclose(betas[-1][0], 1.0):
        raise RuntimeError(f"temperature index 0 is not the cold chain (beta = {betas[-1][0]}): the temperature axis is not as assumed")
    print(chain.shape)

    # the cold chain only, with the burn-in and thinning of run_emcee (and emcee's slice convention)
    cold = chain[:, 0, :, 0, :]
    ln_posterior_cold = ln_like[:, 0] + ln_prior[:, 0]
    burnin, thin = eryn_burn_in_and_thin(cold)
    keep = slice(burnin + thin - 1, None, thin)
    samples = cold[keep].reshape(-1, ndim)
    ln_posterior = ln_posterior_cold[keep].reshape(-1)
    if len(samples) == 0:
        raise RuntimeError(f"no samples are left after a burn-in of {burnin} and thinning by {thin} of a chain of {steps} steps: run more steps")
    print(f"eryn finished: {len(samples)} samples")
    return samples, ln_posterior


SAMPLERS = {
    "bilby_mcmc": run_bilby_mcmc,
    "emcee": run_emcee,
    "eryn": run_eryn,
}

def main():

    # Parse the commandline arguments
    p = create_run_sampler_arg_parser()
    args = p.parse_args()
    verbose = args.verbose
    sampler_kwargs = parse_sampler_kwargs(args.sampler_kwargs)

    # get reference parameters
    reference_parameters, ref_pe_samples = get_injected_parameters(args, verbose=verbose)

    # get kwargs
    kwargs = initialize_kwargs(args, reference_parameters)

    # make waveform manager
    wf_manager = make_waveform_manager(args, **kwargs)

    # get data
    time_dict, data_dict, psd_dict = get_conditioned_time_and_data(
        args, wf_manager=wf_manager, reference_parameters=reference_parameters, verbose=verbose
    )

    if verbose:
        print("kwargs are:")
        print(kwargs)

    # set up likelihood
    likelihood_manager = build_likelihood_manager(args, kwargs, psd_dict, time_dict, data_dict)
    print(f"Sampler: {args.sampler} | "
          f"use_lvk_mass_and_time_prior = {args.use_lvk_mass_and_time_prior} | only_prior = {args.only_prior}")

    # run the chosen sampler: it returns the samples (sampling coordinates) and their ln posterior
    samples, ln_posterior = SAMPLERS[args.sampler](
        likelihood_manager, args, kwargs, sampler_kwargs, reference_parameters, ref_pe_samples)
    print('samples shape:', np.shape(samples))

    # post-processing and saving, the same for every sampler
    save_posterior(samples, ln_posterior, likelihood_manager, args.output, **kwargs)


if __name__ == "__main__":
    main()
