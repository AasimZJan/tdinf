"""
sampler_utils: the gravitational-wave proposal cycle of bilby_mcmc ("gwA" in bilby.bilby_mcmc.proposals.get_proposal_cycle) for tdinf's parameter names.

Put this file in tdinf/utils/ and see run_sampler.py for the hook. Use it with  "proposal_cycle": "tdinf_gw"  (or "tdinf_gw_noFM", "tdinf_gw_noFMnoGM", ... as for bilby's names).

Why it is needed
----------------
bilby's gwA cycle takes the groups of parameters from the prior dictionary (priors.intrinsic, .extrinsic, .mass, .spin, .measured_spin, .phase, .sky) and the members of each
group from PARAMETER_SETS in bilby.gw.source, which lists bilby's CBC names (a_1, tilt_1, phi_12, ra, psi, theta_jn, ...). tdinf's names differ (spin1_magnitude, spin1_tilt,
right_ascension, polarization, inclination, ...), so bilby's own gwA cannot be used. TdinfPriorDict is a prior dictionary that carries the same groups for tdinf's names, and
build_tdinf_gw_cycle builds the same cycle from them.

The sets (bilby.gw.source.PARAMETER_SETS on bilby's master branch) and their tdinf names
    bilby set            bilby names                                                          tdinf names
    spin                 a_1 a_2 tilt_1 tilt_2 phi_12 phi_jl chi_1 chi_2 chi_i_in_plane       spin1/2_magnitude, spin1/2_tilt, spin1/2_azimuth, spin1/2_z; c1, c2 (spin directions)
    mass                 chirp_mass mass_ratio total_mass mass_1 mass_2                       total_mass, mass_ratio
    phase                phase                                                                phase
    intrinsic            spin + mass + phase + tidal                                          the same, plus eccentricity and mean_anomaly when they are sampled
    extrinsic            luminosity_distance psi theta_jn cos_theta_jn geocent_time ra dec    luminosity_distance, polarization, inclination, cos_inclination, geocenter_time,
                         zenith azimuth time_jitter                                           right_ascension, declination, sin_declination
    sky                  ra dec zenith azimuth                                                right_ascension, declination, sin_declination
    measured_spin        chi_1 chi_2 a_1 a_2 chi_1_in_plane                                   spin1/2_magnitude, spin1/2_z
    primary_spin_and_q   a_1 chi_1 mass_ratio                                                 spin1_magnitude, spin1_z, mass_ratio
Note that bilby counts the binary PHASE as intrinsic, not extrinsic (the bilby-MCMC paper does too: 11 intrinsic parameters including the phase, 6 extrinsic ones).
In sampling coordinates a name is first reduced to its base name (x_total_mass -> total_mass, phase_x -> phase, c1_z -> c1, x_cos_inclination -> cos_inclination).

The cycle (the order, subsets and weights of bilby's gwA; small weight 5, big weight 10, tiny weight 0.5)
    all parameters:      adaptive Gaussian (AG), differential evolution (DE)
    intrinsic block:     AG, DE, KDE, Gaussian mixture (GMM)          (the learning proposals with first_fit 1000, 10000 samples, fit_multiplier 2)
    extrinsic block:     AG, DE, KDE, GMM
    mass block:          DE, GMM, Fisher matrix
    spin block:          DE, GMM, Fisher matrix at weight 10
    measured-spin block: AG, Fisher matrix
    primary spin and q:  DE                                          (only if there are mass and spin parameters)
    sky block:           Fisher matrix, GMM
    prior draws (tiny):  polarization (bilby: psi), spin2_azimuth (bilby: phi_12), spin2_tilt (bilby: tilt_2)
The names that remove proposals work as for bilby's own: "noFM" removes the Fisher proposals, "noGM" the mixtures, "noKD" the KDEs, "noPR" the prior draws, "noAG", "noDE", ...

Left out, on purpose
    * bilby's phase proposals (PhaseReversalProposal, CorrelatedPolarisationPhaseJump, PhasePolarisationReversalProposal): no phase jumps for now;
    * any scaling of the steps (the adaptive Gaussian proposals keep bilby's sigma = 1 for every parameter);
    * what does not apply to tdinf: tidal parameters, calibration, time_jitter, the in-plane spin parameterisation.
Different from bilby: if scikit-learn is missing bilby's gwA raises an error; here the mixture proposals are left out with a message.
"""
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


def _require_bilby():
    if _BILBY_ERROR is not None:
        raise ImportError(f"the tdinf_gw proposal cycle needs bilby with bilby_mcmc ({_BILBY_ERROR})")


def _require_emcee(what):
    if emcee is None:
        raise ImportError(f"{what} needs emcee (pip install emcee)")


# ---------------------------------------------------------------------------------------------------------------------------------- the groups, by base name
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
        _require_bilby()
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
    _require_bilby()
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
        plist += [DifferentialEvolutionProposal(priors, weight=SMALL_WEIGHT, subset=spin)] + gmm(spin, SMALL_WEIGHT) + [FisherMatrixProposal(priors, weight=BIG_WEIGHT, subset=spin)]
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
    _require_bilby()
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


# ---------------------------------------------------------------------------------------------------------------------------------- emcee helpers (used by run_emcee)
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
