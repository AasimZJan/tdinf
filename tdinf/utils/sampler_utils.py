import numpy as np

try:
    import emcee
except ImportError:                      # the emcee helpers at the end need it; the bilby part does not
    emcee = None

try:
    from bilby.core.prior import PriorDict
    from bilby.bilby_mcmc import proposals as _proposals
    from bilby.bilby_mcmc.proposals import (AdaptiveGaussianProposal, DifferentialEvolutionProposal, FisherMatrixProposal, GMMProposal, KDEProposal, PriorProposal, ProposalCycle)
    _BILBY_ERROR = None
except ImportError as _error:            # an emcee-only environment can still import this file; the GW cycle then says what is missing
    PriorDict, _proposals, _BILBY_ERROR = object, None, _error


###########################################################################################
# bilby_mcmc functions
###########################################################################################

SPIN = ["spin1_magnitude", "spin2_magnitude", "spin1_tilt", "spin2_tilt", "spin1_azimuth", "spin2_azimuth", "spin1_z", "spin2_z", "c1", "c2"]
MASS = ["total_mass", "mass_ratio"]
PHASE = ["phase"]
PARAMETER_SETS = dict(
    spin=SPIN,
    mass=MASS,
    phase=PHASE,
    intrinsic=SPIN + MASS + PHASE + ["eccentricity", "mean_anomaly"],
    extrinsic=["luminosity_distance", "inclination", "cos_inclination", "polarization", "geocenter_time", "right_ascension", "declination", "sin_declination"],
    sky=["right_ascension", "declination", "sin_declination"],
    measured_spin=["spin1_magnitude", "spin2_magnitude", "spin1_z", "spin2_z"],
    primary_spin_and_q=["spin1_magnitude", "spin1_z", "mass_ratio"],
    tidal=[],
)

SMALL_WEIGHT, BIG_WEIGHT, TINY_WEIGHT = 5, 10, 0.5            # bilby's own
LEARNING_KWARGS = dict(first_fit=1000, nsamples_for_density=10000, fit_multiplier=2)


def _base_name(key):
    """total_mass for x_total_mass; phase for phase_x; c1 for c1_z; unchanged for a physical name such as spin1_z"""
    if key.startswith("x_"):
        return key[2:]
    for stem in ("phase", "mean_anomaly", "right_ascension", "polarization", "c1", "c2"):
        if key in (stem + "_x", stem + "_y", stem + "_z"):
            return stem
    return key


def _members(keys, names):
    return [k for k in keys if k in names or _base_name(k) in names]


class TdinfPriorDictForGWProposal(PriorDict):
    """A bilby prior dictionary that also knows the groups of parameters of bilby's gwA cycle, for tdinf's names (as bilby's CBCPriorDict does for bilby's names).
    Each property lists the sampled (non-fixed) parameters of the group; an empty list is false, as bilby's own properties are."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _group(self, name):
        return _members(list(self.non_fixed_keys), PARAMETER_SETS[name])

    intrinsic = property(lambda self: self._group("intrinsic"))
    extrinsic = property(lambda self: self._group("extrinsic"))
    mass = property(lambda self: self._group("mass"))
    spin = property(lambda self: self._group("spin"))
    phase = property(lambda self: self._group("phase"))
    sky = property(lambda self: self._group("sky"))
    measured_spin = property(lambda self: self._group("measured_spin"))
    primary_spin_and_q = property(lambda self: self._group("primary_spin_and_q"))
    tidal = property(lambda self: [])


# ---------------------------------------------------------------------------------------------------------------------------------- the cycle
def build_tdinf_gw_cycle(string, priors, warn=True):
    """bilby's gwA cycle (bilby.bilby_mcmc.proposals.get_proposal_cycle) for tdinf's parameters; `string` is e.g. 'tdinf_gw_noFM'"""
    drop_spin_fisher = "nospinFM" in string
    string = string.replace("nospinFM", "")
    keys = list(priors.non_fixed_keys)

    def group(name):
        attribute = getattr(priors, name, None)
        return list(attribute) if attribute is not None else _members(keys, PARAMETER_SETS[name])

    have_gmm = GMMProposal.check_dependencies(warn=False)
    if not have_gmm and warn:
        print("tdinf_gw: scikit-learn is not installed, so the Gaussian mixture proposals are left out")

    def gmm(subset, weight):
        return [GMMProposal(priors, weight=weight, subset=subset, **LEARNING_KWARGS)] if have_gmm else []

    all_params = [key for key in priors if "recalib" not in key]
    plist = [AdaptiveGaussianProposal(priors, weight=SMALL_WEIGHT, subset=all_params), DifferentialEvolutionProposal(priors, weight=SMALL_WEIGHT, subset=all_params)]

    for name in ("intrinsic", "extrinsic"):
        subset = group(name)
        if subset:
            plist += [AdaptiveGaussianProposal(priors, weight=SMALL_WEIGHT, subset=subset), DifferentialEvolutionProposal(priors, weight=SMALL_WEIGHT, subset=subset),
                      KDEProposal(priors, weight=SMALL_WEIGHT, subset=subset, **LEARNING_KWARGS)] + gmm(subset, SMALL_WEIGHT)

    mass, spin = group("mass"), group("spin")
    if mass:
        plist += [DifferentialEvolutionProposal(priors, weight=SMALL_WEIGHT, subset=mass)] + gmm(mass, SMALL_WEIGHT) + [FisherMatrixProposal(priors, weight=SMALL_WEIGHT, subset=mass)]
    if spin:
        plist += [DifferentialEvolutionProposal(priors, weight=SMALL_WEIGHT, subset=spin)] + gmm(spin, SMALL_WEIGHT)
        if not drop_spin_fisher:
            plist.append(FisherMatrixProposal(priors, weight=BIG_WEIGHT, subset=spin))
    measured_spin = group("measured_spin")
    if measured_spin:
        plist += [AdaptiveGaussianProposal(priors, weight=SMALL_WEIGHT, subset=measured_spin), FisherMatrixProposal(priors, weight=SMALL_WEIGHT, subset=measured_spin)]
    if mass and spin:
        plist.append(DifferentialEvolutionProposal(priors, weight=SMALL_WEIGHT, subset=group("primary_spin_and_q")))
    # bilby's gwA has here, when the phase is sampled, PhaseReversalProposal, CorrelatedPolarisationPhaseJump and PhasePolarisationReversalProposal at the tiny weight: left out on purpose
    sky = group("sky")
    if sky:
        plist += [FisherMatrixProposal(priors, weight=SMALL_WEIGHT, subset=sky)] + gmm(sky, SMALL_WEIGHT)
    for key in ("polarization", "spin2_azimuth", "spin2_tilt"):                        # bilby: psi, phi_12 (= azimuth 2 - azimuth 1, so azimuth 2 here), tilt_2
        if key in keys:
            plist.append(PriorProposal(priors, subset=[key], weight=TINY_WEIGHT))
    plist = _proposals.remove_proposals_using_string(plist, string)
    return ProposalCycle(plist)


_INSTALLED = [False]


def register_tdinf_gw_cycle():
    """
    Make the cycle names "tdinf_gw", "tdinf_gw_noFM", ... available to bilby_mcmc: any name that starts with tdinf_gw (what follows is the usual removal string, "noFM" and so on).
    Every other name still goes to bilby. Returns True if installed now, False if it already was.
    """
    if _INSTALLED[0]:
        return False
    previous = _proposals.get_proposal_cycle

    def get_proposal_cycle(string, priors, L1steps=1, warn=True):
        if not string.startswith("tdinf_gw"):
            return previous(string, priors, L1steps=L1steps, warn=warn)
        return build_tdinf_gw_cycle(string, priors, warn=warn)

    _proposals.get_proposal_cycle = get_proposal_cycle
    _INSTALLED[0] = True
    return True

def bilby_number_of_chains(sampler_settings):
    """
    How many chains bilby_mcmc runs for these settings (as documented): nensemble * ntemps if pt_ensemble is True, otherwise
    nensemble + ntemps - 1 (only the first ensemble chain is tempered). An explicit initial_betas ladder sets the number of temperatures.
    """
    ntemps = len(sampler_settings['initial_betas']) if sampler_settings.get('initial_betas') is not None else int(sampler_settings.get('ntemps', 1))
    nensemble = int(sampler_settings.get('nensemble', 1))
    return nensemble * ntemps if sampler_settings.get('pt_ensemble', False) else nensemble + ntemps - 1


def get_bilby_initial_samples(likelihood_manager, args, sampler_settings, reference_parameters, ref_pe_samples, get_initial_walkers, likelihood=None, max_tries=20):
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

###########################################################################################
# emcee functions
###########################################################################################
EMCEE_MOVES = {'stretch': 'StretchMove', 'de': 'DEMove', 'snooker': 'DESnookerMove', 'walk': 'WalkMove'}

def emcee_moves(spec):
    """
    The `moves` argument of emcee.EnsembleSampler from the "moves" setting: None or "stretch" is emcee's default (the stretch move, as in the old
    script); "de" is differential evolution with a few snooker moves (emcee's recommendation, 0.8 / 0.2); or a list of [name, weight] pairs with names
    from stretch, de, snooker, walk.
    """
    if spec is None or spec == 'stretch':
        return None
    _require_emcee("the 'de' or list-of-moves setting")
    if spec == 'de':
        return [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)]
    if isinstance(spec, (list, tuple)):
        moves = []
        for name, weight in spec:
            if name not in EMCEE_MOVES:
                raise ValueError(f"unknown emcee move '{name}'; use one of {sorted(EMCEE_MOVES)}")
            moves.append((getattr(emcee.moves, EMCEE_MOVES[name])(), float(weight)))
        return moves
    raise ValueError(f"the emcee setting 'moves' must be null, 'stretch', 'de' or a list of [name, weight] pairs, not {spec!r}")


def emcee_burn_in_and_thin(sampler):
    """The burn-in and thinning the old script used (utils.postprocess_samples): 5 x the largest and 0.5 x the smallest autocorrelation time."""
    tau = sampler.get_autocorr_time(quiet=True)
    try:
        burnin = max(int(5 * np.max(tau)), 0)
        thin = max(int(0.5 * np.min(tau)), 1)
        print(f'burnin = {burnin}, thin = {thin}')
    except Exception:
        print('WARNING thinning by 1, 0 burnin!')
        burnin = 0
        thin = 1
    return burnin, thin

###########################################################################################
# eryn functions
###########################################################################################

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