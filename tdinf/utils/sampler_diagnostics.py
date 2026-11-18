#!/usr/bin/env python3
"""
sampler_diagnostics: the HTML diagnostics of a tdinf run, for whichever sampler made it, in ONE file (it needs only numpy, scipy, matplotlib and h5py).

    from tdinf.utils import sampler_diagnostics
    sampler_diagnostics.run(args.sampler, args, kwargs=kwargs, sampler_kwargs=sampler_kwargs)       # run_sampler.py does exactly this in main(), with --produce-diagnostics

It takes the name of the sampler and what the sampler was given (run_sampler's args, kwargs and sampler_kwargs, from which the files it wrote are found) and writes one
self-contained HTML report next to the output, <--output without extension>_diagnostics.html. What it does depends on the sampler:

  emcee        log-probability, burn-in, walkers, traces, autocorrelation, R-hat, ESS (PART 1 below: the module that used to be emcee_diagnostics.py).
  eryn         the same report for the COLD chain (temperature 0, its walkers being the walkers of the report), and a Tempering section with what only a tempered
               ensemble has: the temperature ladder and whether it has settled, the swap acceptance of every adjacent pair (measured, and predicted from the stored
               log likelihoods), the log likelihood of every temperature and how neighbours overlap, the acceptance of every move at every temperature, the
               autocorrelation time of the POPULATION (see below) and the drift of every temperature. Its checks and verdict are part of the Overview (PART 3).
  bilby_mcmc   the report from the resume pickle: chains, ladder, swaps, proposals, per-chain statistics (PART 2: the module that used to be bilby_diagnostics.py).
Any other sampler gets a message and no report. Nothing here can stop a run: with safe=True (default) any error is printed as a warning.

Why Eryn gets population numbers. Every step swaps the walkers of neighbouring temperatures, so a walker SLOT of the cold chain is refilled by a neighbour within a few steps
and its autocorrelation time is only a few steps. The state of the whole ensemble can still move slowly: the share of walkers in a mode, the mean of a parameter. The Tempering
section therefore also gives the autocorrelation time of the ensemble mean of every parameter at every temperature, and the shift of that mean between the two halves of the
run. The run length divided by that time, not the per-slot time, says how many independent looks the run has at slow collective behaviour such as mode weights.

Layout of this file. PART 1 is the emcee report; PART 2 the bilby_mcmc report (the 31 helpers it shares with PART 1 are the same code and are defined once, in PART 1; the 14 that
differ carry a bilby_ prefix: bilby_analyze, bilby_build_report, bilby_report_from_args, ...); PART 3 the dispatcher run() and the Eryn analysis; the command line is at the end.
The documentation of the two old modules is kept in _EMCEE_DOC and _BILBY_DOC (shown by their --help).

From the shell:
    python sampler_diagnostics.py emcee      -o run.h5 --vary-skypos --vary-time        (the old emcee_diagnostics command line; emcee --help for the rest)
    python sampler_diagnostics.py eryn       run.h5 --vary-skypos --vary-time          (eryn --help)
    python sampler_diagnostics.py bilby_mcmc run_resume.pickle [--posterior-widths] [--swaps-only] ...     (bilby_mcmc --help)
"""

import argparse
import base64
import collections
import copy
import html as _html
import io
import itertools
import json
import os
import pickle
import re
import sys
import time
from contextlib import contextmanager

import numpy as np
from scipy.special import ndtri
from scipy.stats import binom, rankdata

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

_EMCEE_DOC = """
emcee diagnostics (PART 1 of sampler_diagnostics.py): exploration and convergence diagnostics for emcee output,
with a self-contained HTML report. One file: it needs only numpy, scipy,
matplotlib (and h5py to read files).

Defaults: NO thinning and NO burn-in. Everything is computed on the chain exactly
as it is in the file. The estimated burn-in is shown (dashed line, Burn-in section)
and the Burn-in section shows how R-hat, N/tau and drift change as steps are
discarded, but nothing is discarded unless you ask (burn=... / thin=...).

With run_sampler.py
-------------------
    at the end of main():   from tdinf.utils import sampler_diagnostics; sampler_diagnostics.run(args.sampler, args, kwargs=kwargs)
    or from the shell:      python sampler_diagnostics.py emcee -o full/full.h5 --vary-skypos --vary-time   (run_sampler's own flags)

Reading
-------
    run = read_emcee_h5('run.h5', names=get_layout('precessing', True, True))
    an  = build_report(run, 'report.html')
For numpy arrays:  Run.from_arrays(chain, lnp, names=...).
get_layout / candidate_layouts / decode_samples are built in (tdinf column order).

What is checked
---------------
* log-likelihood: ensemble trace, per-walker distributions, per-walker medians,
  a walker x time heat map (stuck or lagging walkers show up as stripes);
* walkers: fraction of steps moved, longest frozen run, low-lnp outliers,
  parameter offsets relative to the other walkers;
* convergence: integrated autocorrelation time (emcee-style) and N/tau, the
  autocorrelation function itself, tau-based ESS, rank-normalised split R-hat,
  half-vs-half drift, acceptance;
* burn-in: estimated plateau start, and the statistics as a function of the
  number of steps discarded.
The sampled log-probability is analysed like a parameter ("ln_posterior").

Large runs
----------
Runs with more than `max_steps` steps (default 100,000) are read with a stride, so that about max_steps steps are used;
the report says so, and autocorrelation times are scaled back by the stride.
Nothing is thinned unless you pass thin= or max_points=. The file is read in
blocks, FFTs are done in walker blocks to bound memory, and plots draw at most
~1200 points per line. A very large run therefore costs time (minutes) rather
than failing; if memory is tight use thin= (tau is scaled back by the stride).

Caveat: walkers of an affine-invariant ensemble are not independent chains, so
R-hat across walkers is a heuristic. N > 50 tau is the primary criterion.
"""

_BILBY_DOC = """
bilby_mcmc diagnostics (PART 2 of sampler_diagnostics.py): exploration and convergence diagnostics for a bilby_mcmc run (parallel tempering), read from its resume pickle <label>_resume.pickle,
as one self-contained HTML report. It is the bilby counterpart of the emcee report (tdinf/utils/sampler_diagnostics.py) and looks and reads the same way:
an Overview with a verdict, cards, a plain-language reading of the run, what to try and a table of checks, then Settings, Log-likelihood, Temperatures, Burn-in,
Chains, Proposals, Traces (with the same interactive explorer), Convergence, Posteriors and Methods. It needs only numpy, scipy and matplotlib: the pickle is read with a
restricted unpickler, WITHOUT bilby or dill, and nothing in the file is run (a normal dill.load runs whatever the file says).

From the shell
--------------
    python sampler_diagnostics.py bilby_mcmc output-trial-v0_resume.pickle                      # writes output-trial-v0_resume_diagnostics.html next to it
    python sampler_diagnostics.py bilby_mcmc run_resume.pickle -o report.html --burn estimated  # --burn: bilby (default), estimated, <rows> or <fraction>
    python sampler_diagnostics.py bilby_mcmc run_resume.pickle --posterior-widths               # just the JSON for "tdinf_posterior_width"
    python sampler_diagnostics.py bilby_mcmc run_resume.pickle --swaps-only                     # just the swap test
    python sampler_diagnostics.py bilby_mcmc old_resume.pickle --drop-rows-mod 4:1              # an old run whose swaps rewrote rows (see Temperatures)

With run_sampler.py (the same as --produce-diagnostics for emcee)
-----------------------------------------------------------------
    at the end of run_bilby_mcmc():   from tdinf.utils import sampler_diagnostics; sampler_diagnostics.run("bilby_mcmc", args, sampler_kwargs=sampler_kwargs)
    (never raises: a problem in the report is printed as a warning and cannot break the run)

What is checked
---------------
* swaps: acceptance of every pair (recorded and predicted from the stored ln L), evenness of the ladder, a more even ladder, the replica round trip by simulation, and whether
  the swaps move the chains (the cold chain's rows that jump, and whether the next row returns);
* convergence of the cold chains: integrated autocorrelation time tau (emcee style, also per chain) and N/tau, the autocorrelation function, ESS, rank-normalised split R-hat,
  half-vs-half drift, tau against chain length, tau of every parameter at every temperature, and the slowest direction of the posterior;
* burn-in: bilby's own cut and the rule that set it, the ln L plateau, and the statistics as a function of the rows removed;
* chains: ln L, tau, offsets of the means in standard errors, bilby's local acceptance;
* proposals: acceptance of every proposal at every temperature, the size of the adaptive Gaussian steps against the posterior widths, and the share of differential-evolution
  jumps that leave the prior (and so are rejected before the likelihood);
* bilby's own count of samples against the target, and a corrected estimate of the time to finish.

Caveat: the cold chains of an ensemble are not independent runs, so R-hat between them is a heuristic. N > 50 tau is the primary criterion.
"""


# ==============================================================================================================================================
# PART 1: emcee (what used to be emcee_diagnostics.py, with the extras hook)
# ==============================================================================================================================================
matplotlib.use("Agg")

__all__ = ["Run", "read_emcee_h5", "analyze", "build_report", "get_layout", "candidate_layouts", "detect_layout", "TDINF_DEFAULT_BOUNDS", "FLAG_RULES_VERSION",
           "decode_samples", "integrated_autocorr_time", "mean_acf", "split_rhat", "rhat_bulk_tail",
           "estimate_burnin", "burn_scan", "report_from_args", "infer_layout_from_file"]

# ============================================================================== style

FLAG_RULES_VERSION = "1.0"      # the walker-flagging rules are fixed; this changes only if the rules ever change

C = dict(blue="#3b6ea5", light="#a9c1de", orange="#e07b39", green="#4c9a6a",
         red="#c8453f", amber="#d9a22b", gray="#8a8f98", ink="#222831", grid="#e6e9ee", burn="#7b5ea7")
STYLE = {
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#b9bec7", "axes.labelcolor": C["ink"], "text.color": C["ink"],
    "xtick.color": C["ink"], "ytick.color": C["ink"],
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.8,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.frameon": False,
    "legend.fontsize": 9, "font.size": 10, "figure.dpi": 100,
    "axes.prop_cycle": matplotlib.cycler(color=[C["blue"], C["orange"], C["green"], C["red"], "#7b5ea7", C["amber"]]),
}


@contextmanager
def _style():
    with plt.rc_context(STYLE):
        yield


# ============================================================================== tdinf layout and decoding

def get_layout(spin_model, vary_skypos, vary_time):
    """
    Names of the columns of a tdinf emcee chain, in order (identical to tdinf's `sampled_keys`).
    Logistic-transformed parameters carry the prefix x_; angles are stored as (x, y) pairs;
    c1_*, c2_* are spin direction vectors.

    spin_model : 'precessing', 'aligned' or 'none';  vary_skypos, vary_time : bool
    """
    if spin_model not in ("precessing", "aligned", "none"):
        raise ValueError("spin_model must be 'precessing', 'aligned' or 'none'")
    keys = ["x_total_mass", "x_mass_ratio", "x_luminosity_distance", "x_cos_inclination"]
    if spin_model != "none":
        keys += ["x_spin1_magnitude", "x_spin2_magnitude"]
    if vary_skypos:
        keys += ["x_sin_declination"]
    keys += ["phase_x", "phase_y"]
    if vary_skypos:
        keys += ["right_ascension_x", "right_ascension_y", "polarization_x", "polarization_y"]
    if spin_model == "precessing":
        keys += ["c1_x", "c1_y", "c1_z", "c2_x", "c2_y", "c2_z"]
    elif spin_model == "aligned":
        keys += ["c1_z", "c2_z"]
    if vary_time:
        keys += ["geocenter_time"]
    return keys


def candidate_layouts(ndim):
    """All (spin_model, vary_skypos, vary_time) configurations that give `ndim` columns."""
    out = []
    for spin_model, sky, time_ in itertools.product(("precessing", "aligned", "none"), (False, True), (False, True)):
        if len(get_layout(spin_model, sky, time_)) == ndim:
            out.append(dict(spin_model=spin_model, vary_skypos=sky, vary_time=time_))
    return out


# Prior bounds that tdinf's run_sampler uses unless the command line changes them. They are NOT stored in the h5 file.
TDINF_DEFAULT_BOUNDS = dict(mtot_lim=(200, 350), q_lim=(0.17, 1), dist_lim=(100, 10000), chi_lim=(0, 0.99))


def detect_layout(names):
    """If `names` is a tdinf column layout, return dict(spin_model, vary_skypos, vary_time); otherwise None."""
    for spin_model, sky, time_ in itertools.product(("precessing", "aligned", "none"), (False, True), (False, True)):
        if list(names) == get_layout(spin_model, sky, time_):
            return dict(spin_model=spin_model, vary_skypos=sky, vary_time=time_)
    return None


def _sigmoid(y):
    return 0.5 * (1.0 + np.tanh(0.5 * y))          # overflow-free exp(y) / (1 + exp(y))


def _inv_logit(y, lo, hi):
    return lo + (hi - lo) * _sigmoid(y)             # equals tdinf's (exp(y)*hi + lo) / (1 + exp(y))


def decode_samples(chain, spin_model, vary_skypos, vary_time, mtot_lim, q_lim, dist_lim, chi_lim,
                   fixed=None, f_ref=None, check_layout=True):
    """
    Convert sampled numbers (last axis = columns of get_layout) into physical parameters.

    mtot_lim, q_lim, dist_lim, chi_lim : (low, high) prior bounds of the run (from its command line).
    fixed : values of parameters that were not sampled: 'right_ascension', 'declination', 'polarization'
            (if not vary_skypos) and 'geocenter_time' (if not vary_time).
    f_ref : if given and lalsimulation is installed, theta_jn, phi_jl, tilt1, tilt2, phi12 come from LAL;
            otherwise tilt1, tilt2, phi12 are computed analytically and theta_jn, phi_jl are omitted.
    Returns a dict of arrays with the shape of chain[..., 0].
    Conventions are those of tdinf: logistic x = lo + (hi - lo) * sigmoid(y); inclination = arccos(...),
    declination = arcsin(...); angles = arctan2(y, x) (+ pi for right_ascension); spin = magnitude * c / |c|.
    """
    fixed = {} if fixed is None else fixed
    chain = np.asarray(chain, dtype=float)
    keys = get_layout(spin_model, vary_skypos, vary_time)
    if chain.shape[-1] != len(keys):
        raise ValueError(f"chain has {chain.shape[-1]} columns but this configuration has {len(keys)}: {keys}. "
                         f"Configurations with {chain.shape[-1]} columns: {candidate_layouts(chain.shape[-1])}")
    col = {k: chain[..., i] for i, k in enumerate(keys)}
    zeros = np.zeros_like(col["x_total_mass"])
    if check_layout:
        if vary_time and not np.all(np.abs(col["geocenter_time"]) > 1e6):
            print("WARNING: vary_time=True but the last column does not look like a GPS time")
        if not vary_time and np.any(np.abs(chain[..., -1]) > 1e6):
            print("WARNING: vary_time=False but the last column looks like a GPS time")

    out = {}
    out["total_mass"] = _inv_logit(col["x_total_mass"], *mtot_lim)
    out["mass_ratio"] = _inv_logit(col["x_mass_ratio"], *q_lim)
    out["luminosity_distance"] = _inv_logit(col["x_luminosity_distance"], *dist_lim)
    out["inclination"] = np.arccos(_inv_logit(col["x_cos_inclination"], -1.0, 1.0))
    out["phase"] = np.arctan2(col["phase_y"], col["phase_x"])
    out["mass_1"] = out["total_mass"] / (1.0 + out["mass_ratio"])
    out["mass_2"] = out["total_mass"] - out["mass_1"]
    out["chirp_mass"] = (out["mass_1"] * out["mass_2"]) ** 0.6 / out["total_mass"] ** 0.2

    for i in ("1", "2"):
        if spin_model == "none":
            out[f"spin{i}_magnitude"] = zeros.copy()
            sx = sy = sz = zeros.copy()
        else:
            mag = _inv_logit(col[f"x_spin{i}_magnitude"], *chi_lim)
            cx, cy, cz = col.get(f"c{i}_x", zeros), col.get(f"c{i}_y", zeros), col.get(f"c{i}_z", zeros)
            with np.errstate(divide="ignore", invalid="ignore"):
                scale = mag / np.sqrt(cx ** 2 + cy ** 2 + cz ** 2)
            out[f"spin{i}_magnitude"] = mag
            sx, sy, sz = cx * scale, cy * scale, cz * scale
        out[f"spin{i}_x"], out[f"spin{i}_y"], out[f"spin{i}_z"] = sx, sy, sz

    if vary_skypos:
        out["declination"] = np.arcsin(_inv_logit(col["x_sin_declination"], -1.0, 1.0))
        out["right_ascension"] = np.arctan2(col["right_ascension_y"], col["right_ascension_x"]) + np.pi
        out["polarization"] = np.arctan2(col["polarization_y"], col["polarization_x"])
    else:
        for k in ("right_ascension", "declination", "polarization"):
            if k not in fixed:
                raise KeyError(f"'{k}' was not sampled, so pass it in `fixed`")
            out[k] = np.full_like(zeros, fixed[k])
    if vary_time:
        out["geocenter_time"] = col["geocenter_time"]
    else:
        if "geocenter_time" not in fixed:
            raise KeyError("'geocenter_time' was not sampled, so pass it in `fixed`")
        out["geocenter_time"] = np.full_like(zeros, fixed["geocenter_time"])

    if spin_model != "none":
        out.update(_spin_angles(out, f_ref))
        out.update(_spin_combinations(out))
    return out


def _spin_combinations(p):
    """Effective spin chi_eff and effective precessing spin chi_p (same formulas as tdinf's spins_and_masses)."""
    m1, m2 = p["mass_1"], p["mass_2"]
    chi_eff = (m1 * p["spin1_z"] + m2 * p["spin2_z"]) / (m1 + m2)
    s1p = np.hypot(p["spin1_x"], p["spin1_y"])             # a * sin(tilt)
    s2p = np.hypot(p["spin2_x"], p["spin2_y"])
    q_inv = m1 / m2
    A1, A2 = 2.0 + 1.5 * q_inv, 2.0 + 1.5 / q_inv
    chi_p = np.maximum(A1 * s2p * m2 * m2, A2 * s1p * m1 * m1) / (A2 * m1 * m1)
    return {"chi_eff": chi_eff, "chi_p": chi_p}


def _spin_angles(p, f_ref):
    """tilt1, tilt2, phi12 (analytic) and, with LAL and f_ref, theta_jn and phi_jl."""
    if f_ref is not None:
        try:
            import lalsimulation as lalsim
        except ImportError:
            print("WARNING: lalsimulation not installed; skipping theta_jn and phi_jl")
        else:
            shape = p["total_mass"].shape
            flat = {k: np.ravel(v) for k, v in p.items()}
            res = np.zeros((7, flat["total_mass"].size))
            for j in range(res.shape[1]):
                res[:, j] = lalsim.SimInspiralTransformPrecessingWvf2PE(
                    flat["inclination"][j], flat["spin1_x"][j], flat["spin1_y"][j], flat["spin1_z"][j],
                    flat["spin2_x"][j], flat["spin2_y"][j], flat["spin2_z"][j],
                    flat["mass_1"][j], flat["mass_2"][j], f_ref, flat["phase"][j])
            return {n_: res[k].reshape(shape) for k, n_ in enumerate(["theta_jn", "phi_jl", "tilt1", "tilt2", "phi12"])}
    with np.errstate(divide="ignore", invalid="ignore"):
        tilt1 = np.arccos(np.clip(p["spin1_z"] / p["spin1_magnitude"], -1, 1))
        tilt2 = np.arccos(np.clip(p["spin2_z"] / p["spin2_magnitude"], -1, 1))
    phi12 = np.mod(np.arctan2(p["spin2_y"], p["spin2_x"]) - np.arctan2(p["spin1_y"], p["spin1_x"]), 2 * np.pi)
    return {"tilt1": tilt1, "tilt2": tilt2, "phi12": phi12}


_PHYSICAL_KEYS = ["total_mass", "mass_ratio", "mass_1", "mass_2", "chirp_mass", "luminosity_distance", "inclination", "phase",
                  "spin1_magnitude", "spin2_magnitude", "chi_eff", "chi_p", "tilt1", "tilt2", "phi12", "right_ascension",
                  "declination", "polarization"]


# ============================================================================== data container

def _next_pow2(n):
    return 1 << (int(n) - 1).bit_length()


def _column_offsets(x, rows=200):
    """
    Reference values for columns that float32 cannot resolve, such as a GPS time (about 1.2e9 s with a spread of
    0.01 s: float32 steps are about 128 s). Such columns are stored as (value - offset); the offset is rounded to
    1 ms and shown in the column name. Normal columns get offset 0.
    """
    sub = np.asarray(x[:rows], dtype=np.float64)
    flat = sub.reshape(-1, sub.shape[-1])
    med = np.nanmedian(flat, axis=0)
    sd = np.nanstd(flat, axis=0)
    big = (np.abs(med) > 1.0) & (sd < 1e-4 * np.abs(med))
    return np.where(big, np.round(med, 3), 0.0)


class Run:
    """
    An emcee run held in memory (all steps unless you asked for thinning).

    Attributes
    ----------
    x : float32 array (n, nwalkers, ndim)  chain (thinned only if thin/max_points were given)
    lnp : float32 array (n, nwalkers)      log-probability (non-finite -> NaN)
    steps : int array (n,)                 step index of each row
    stride : int                           thinning stride (1 = nothing thinned)
    nsteps_total : int                     steps in the file
    tail_x, tail_lnp : contiguous last steps at full resolution (or None)
    accepted : per-walker acceptance fraction from the backend (or None)
    columns : dict name -> (n, nwalkers) array; "ln_posterior" first, then the sampled columns, then derived ones
    """

    def __init__(self, x, lnp, steps, nsteps_total, names=None, tail_x=None, tail_lnp=None,
                 accepted=None, source="", offsets=None, thin_note=""):
        self.thin_note = thin_note
        x = np.asarray(x)
        d = x.shape[2]
        if offsets is None:                       # x is the true chain: find columns that float32 cannot resolve
            offsets = _column_offsets(x)
            if offsets.any():
                x = np.asarray(x, dtype=np.float64) - offsets
                if tail_x is not None:
                    tail_x = np.asarray(tail_x, dtype=np.float64) - offsets
        self.offsets = np.asarray(offsets, dtype=np.float64)
        self.x = np.asarray(x, dtype=np.float32)
        lnp = np.asarray(lnp, dtype=np.float32)
        self.bad = ~np.isfinite(lnp)
        if self.bad.any():
            lnp = lnp.copy()
            lnp[self.bad] = np.nan
        self.lnp = lnp
        self.steps = np.asarray(steps)
        self.nsteps_total = int(nsteps_total)
        self.stride = int(self.steps[1] - self.steps[0]) if len(self.steps) > 1 else 1
        self.tail_x = None if tail_x is None else np.asarray(tail_x, dtype=np.float32)
        self.tail_lnp = None if tail_lnp is None else np.asarray(tail_lnp, dtype=np.float32)
        self.accepted = None if accepted is None else np.asarray(accepted, dtype=float)
        self.source = source
        self.raw_names = list(names) if names is not None else [f"p{i}" for i in range(d)]
        if len(self.raw_names) != d:
            raise ValueError(f"{len(self.raw_names)} names for {d} columns. Run configurations with {d} columns: "
                             f"{candidate_layouts(d)}; pass names=get_layout(...) accordingly.")
        if self.raw_names[-1] == "geocenter_time" and np.nanmedian(np.abs(self.x[..., -1].astype(np.float64) + self.offsets[-1])) < 1e6:
            print("WARNING: the last column is named geocenter_time but does not look like a GPS time; "
                  "check the run configuration (vary_skypos / vary_time).")
        self.col_names = [f"{nm} - {off:.3f}" if off else nm for nm, off in zip(self.raw_names, self.offsets)]
        self.columns = {"ln_posterior": self.lnp}
        self.columns.update({nm: self.x[:, :, i] for i, nm in enumerate(self.col_names)})
        self.derived_names = []
        self.physical_info = None

    @property
    def n(self):
        return self.x.shape[0]

    @property
    def nwalkers(self):
        return self.x.shape[1]

    @property
    def ndim(self):
        return self.x.shape[2]

    @property
    def names(self):
        return list(self.columns.keys())

    def add_derived(self, mapping):
        """Add derived columns: dict name -> array (n, nwalkers). A name that already exists gets the suffix ' (derived)'."""
        for k, v in mapping.items():
            v = np.asarray(v, dtype=np.float32)
            if v.shape != (self.n, self.nwalkers):
                raise ValueError(f"derived column '{k}' has shape {v.shape}, expected {(self.n, self.nwalkers)}")
            if k in self.columns:
                k = f"{k} (derived)"
            self.columns[k] = v
            self.derived_names.append(k)

    def add_physical(self, spin_model, vary_skypos, vary_time, mtot_lim, q_lim, dist_lim, chi_lim,
                     fixed=None, keys=None, block=2000):
        """
        Add physical parameters decoded from the sampled numbers (tdinf layout), computed in blocks of
        `block` steps so that memory stays bounded. `keys` defaults to masses (including the chirp mass), distance,
        inclination, phase, spin magnitudes, chi_eff, chi_p, tilts, phi12 and (if sampled) the sky position.
        Every column needs steps x walkers x 4 bytes of memory; pass `keys` to limit them on very large runs.
        """
        self.physical_info = dict(mtot_lim=tuple(mtot_lim), q_lim=tuple(q_lim), dist_lim=tuple(dist_lim), chi_lim=tuple(chi_lim),
                                  fixed=dict(fixed or {}), auto=False)
        want = list(keys) if keys else [k for k in _PHYSICAL_KEYS
                                         if vary_skypos or k not in ("right_ascension", "declination", "polarization")]
        want = [k for k in want if not (spin_model == "none" and k in ("spin1_magnitude", "spin2_magnitude", "chi_eff", "chi_p", "tilt1", "tilt2", "phi12"))]
        out = {}
        for b0 in range(0, self.n, block):
            blk = self.x[b0:b0 + block].astype(np.float64) + self.offsets      # true values, offsets added back
            dec = decode_samples(blk, spin_model, vary_skypos, vary_time, mtot_lim, q_lim,
                                 dist_lim, chi_lim, fixed=fixed, check_layout=(b0 == 0))
            for k in want:
                if k in dec:
                    out.setdefault(k, np.empty((self.n, self.nwalkers), dtype=np.float32))[b0:b0 + block] = dec[k]
        self.add_derived(out)

    @classmethod
    def from_arrays(cls, chain, lnp, names=None, accepted=None, thin=1, max_points=None, n_tail=1000,
                    source="arrays", max_steps=None):
        """Build a Run from chain (nsteps, nwalkers, ndim) and lnp (nsteps, nwalkers). No thinning by default."""
        chain = np.asarray(chain)
        lnp = np.asarray(lnp)
        nsteps, w, d = chain.shape
        stride = int(max(1, thin))
        why = "requested" if stride > 1 else ""
        if max_steps and nsteps > max_steps and int(np.ceil(nsteps / max_steps)) > stride:
            stride, why = int(np.ceil(nsteps / max_steps)), "limit"
        if max_points and int(np.ceil(nsteps * w * d / max_points)) > stride:
            stride, why = int(np.ceil(nsteps * w * d / max_points)), "points"
        idx = np.arange(0, nsteps, stride)
        t0 = max(0, nsteps - n_tail)
        return cls(chain[idx], lnp[idx], idx, nsteps, names=names, tail_x=chain[t0:], tail_lnp=lnp[t0:],
                   accepted=accepted, source=source, thin_note=_thin_note(stride, why, nsteps, max_steps))


def _ordinal(n):
    n = int(n)
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def _thin_note(stride, why, n_avail, max_steps):
    if stride <= 1:
        return ""
    return {"limit": f"The file has {n_avail:,} steps, more than the limit of {max_steps:,}, so every {_ordinal(stride)} step was read.",
            "requested": f"Every {_ordinal(stride)} step was read, as requested.",
            "points": f"Every {_ordinal(stride)} step was read to stay within the memory budget (max_points)."}[why]


def read_emcee_h5(path, names=None, group="mcmc", thin=1, max_points=None, max_steps=100_000, n_tail=1000, start=0,
                  stop=None, block_steps=4000, warn_gb=2.0):
    """
    Read an emcee HDFBackend file in blocks. Every step is kept unless the run is longer than `max_steps`.

    max_steps : a run with more steps than this (default 100,000) is read with a stride, so that about max_steps steps
    are used; None or 0 turns this off.  thin : keep every `thin`-th step (explicit stride).
    max_points : alternatively cap steps * walkers * dims (raises the stride).
    The last `n_tail` steps are always available at full resolution (used to measure whether walkers move).
    Uses the `iteration` attribute, so unwritten (zero) rows are never read.
    """
    import h5py
    with h5py.File(path, "r") as f:
        g = f[group]
        nsteps = int(g.attrs["iteration"])
        stop = nsteps if stop is None else min(int(stop), nsteps)
        _, w, d = g["chain"].shape
        stride = int(max(1, thin))
        why = "requested" if stride > 1 else ""
        n_avail = stop - start
        if max_steps and n_avail > max_steps and int(np.ceil(n_avail / max_steps)) > stride:
            stride, why = int(np.ceil(n_avail / max_steps)), "limit"
        if max_points and int(np.ceil(n_avail * w * d / max_points)) > stride:
            stride, why = int(np.ceil(n_avail * w * d / max_points)), "points"
        thin_note = _thin_note(stride, why, n_avail, max_steps)
        steps = np.arange(start, stop, stride)
        gb = len(steps) * w * d * 4 / 1e9
        if gb > warn_gb:
            print(f"WARNING: keeping {len(steps):,} steps x {w} walkers x {d} columns needs about {gb:.1f} GB "
                  f"(plus copies during the analysis). Use thin=... or max_points=... if memory is tight.")
        x = np.empty((len(steps), w, d), dtype=np.float32)
        lnp = np.empty((len(steps), w), dtype=np.float32)
        offsets = _column_offsets(g["chain"][start:min(start + 200, stop)])     # e.g. a GPS time column
        blk = max(stride, (block_steps // stride) * stride)       # keep blocks aligned with the stride
        k = 0
        for b0 in range(start, stop, blk):
            b1 = min(b0 + blk, stop)
            ch = np.asarray(g["chain"][b0:b1:stride], dtype=np.float64) - offsets
            lp = g["log_prob"][b0:b1:stride]
            x[k:k + len(ch)] = ch
            lnp[k:k + len(lp)] = lp
            k += len(ch)
        if stride == 1:
            tail_x, tail_lnp = x[-n_tail:], lnp[-n_tail:]
        else:
            t0 = max(start, stop - n_tail)
            tail_x = (np.asarray(g["chain"][t0:stop], dtype=np.float64) - offsets).astype(np.float32)
            tail_lnp = g["log_prob"][t0:stop].astype(np.float32)
        accepted = g["accepted"][:] / max(nsteps, 1) if "accepted" in g else None
    print(f"read {path}: {nsteps} steps x {w} walkers x {d} dims; stride {stride} -> {len(steps)} rows kept")
    return Run(x, lnp, steps, nsteps, names=names, tail_x=tail_x, tail_lnp=tail_lnp,
               accepted=accepted, source=str(path), offsets=offsets, thin_note=thin_note)


# ============================================================================== statistics

def _fill_nan(x):
    """float64 copy in which non-finite entries are replaced by the median (statistics need finite input)."""
    x = np.asarray(x, dtype=np.float64)
    bad = ~np.isfinite(x)
    if bad.any():
        good = x[~bad]
        x = x.copy()
        x[bad] = np.median(good) if good.size else 0.0
    return x


def mean_acf(x, block_elems=1e7):
    """
    Mean normalised autocorrelation function over the walkers of x (n, nwalkers), via FFT.
    Done in blocks of walkers so that memory stays bounded for very long chains.
    Walkers with zero variance (frozen) are skipped. Returns (acf of length n, number of walkers used).
    """
    n, w = x.shape
    N = _next_pow2(n)
    block = max(1, int(block_elems // (2 * N)))
    total = np.zeros(n)
    used = 0
    for j in range(0, w, block):
        xb = np.asarray(x[:, j:j + block], dtype=np.float64)
        xb = xb - xb.mean(axis=0, keepdims=True)
        ok = (xb ** 2).sum(axis=0) > 0
        if not ok.any():
            continue
        xb = xb[:, ok]
        f = np.fft.rfft(xb, n=2 * N, axis=0)
        acf = np.fft.irfft(f * np.conj(f), n=2 * N, axis=0)[:n]
        acf /= acf[0]
        total += acf.sum(axis=1)
        used += acf.shape[1]
    if used == 0:
        return None, 0
    return total / used, used


def auto_window(taus, c):
    """Sokal's automated windowing (as in emcee)."""
    m = np.arange(len(taus)) < c * taus
    if np.any(~m):
        return int(np.argmin(m))
    return len(taus) - 1


def integrated_autocorr_time(x, c=5.0):
    """
    Integrated autocorrelation time of x (n, nwalkers), emcee style: autocorrelation function of every
    walker, averaged over walkers, integrated with Sokal's window. In units of the rows of x.
    """
    if x.shape[0] < 4:
        return np.nan
    acf, used = mean_acf(x)
    if acf is None:
        return np.nan
    taus = 2.0 * np.cumsum(acf) - 1.0
    return float(taus[auto_window(taus, c)])


def _rank_normalize(x):
    r = rankdata(x.ravel(), method="average").reshape(x.shape)
    return ndtri((r - 0.375) / (x.size + 0.25))


def split_rhat(x):
    """Split R-hat for x (n, nchains): every chain is cut in two halves."""
    n = x.shape[0] // 2
    if n < 4 or x.shape[1] < 2:
        return np.nan
    ch = np.concatenate([x[:n], x[n:2 * n]], axis=1)
    W = ch.var(axis=0, ddof=1).mean()
    if W <= 0:
        return np.nan
    B = n * ch.mean(axis=0).var(ddof=1)
    return float(np.sqrt(((n - 1) / n * W + B / n) / W))


def rhat_bulk_tail(x):
    """Rank-normalised split R-hat (Vehtari et al. 2021): returns (bulk, tail)."""
    x = np.asarray(x, dtype=np.float64)
    bulk = split_rhat(_rank_normalize(x))
    tail = split_rhat(_rank_normalize(np.abs(x - np.median(x))))
    return bulk, tail


def drift_z(x, tau):
    """Difference of the means of the two halves, in standard errors (tau gives the effective sample size)."""
    n, w = x.shape
    h = n // 2
    if h < 4 or not np.isfinite(tau):
        return np.nan
    sd = x.std()
    if sd == 0:
        return np.nan
    n_eff = w * h / max(tau, 1.0)
    se = sd / np.sqrt(n_eff)
    return float((x[h:2 * h].mean() - x[:h].mean()) / (np.sqrt(2.0) * se))


def robust_z(v):
    v = np.asarray(v, dtype=float)
    med = np.nanmedian(v)
    mad = 1.4826 * np.nanmedian(np.abs(v - med))
    mad = max(mad, 1e-12 * max(1.0, abs(med)))
    return (v - med) / mad


def robust_sd(a):
    """Standard deviation estimated from the median absolute deviation (insensitive to a few frozen walkers)."""
    a = np.asarray(a, dtype=float)
    return 1.4826 * float(np.nanmedian(np.abs(a - np.nanmedian(a))))


def estimate_burnin(lnp, nblocks=60, tail_frac=0.25, return_info=False):
    """
    Burn-in estimate from the ensemble-median log-probability.

    1. median over walkers at every step (the ensemble median);
    2. cut the run into `nblocks` blocks and take the median of that curve in each block;
    3. plateau = median of the ensemble median over the last `tail_frac` of the run;
       tolerance = max(0.1 x robust scatter of late ln p, 3 x scatter of the late ensemble median);
    4. burn-in ends at the start of the first block after which every block median stays above plateau - tolerance.
    If only the last two blocks qualify no plateau is declared (the second half of the run is suggested).
    Resolution is n / nblocks rows. Returns (row index, plateau_found[, info dict]).
    """
    n = lnp.shape[0]
    med = np.nanmedian(lnp, axis=1)
    nb = int(min(nblocks, max(4, n // 10)))
    edges = np.linspace(0, n, nb + 1).astype(int)
    bm = np.array([np.nanmedian(med[a:b]) for a, b in zip(edges[:-1], edges[1:])])
    t0 = int(n * (1 - tail_frac))
    ref = np.nanmedian(med[t0:])
    late_sd = robust_sd(lnp[t0:])
    med_sd = float(np.nanstd(med[t0:]))
    tol = max(0.1 * late_sd, 3.0 * med_sd, 1e-12)
    bad = bm < ref - tol
    if not bad.any():
        idx, found = 0, True
    else:
        last_bad = int(np.max(np.nonzero(bad)[0]))
        if last_bad >= nb - 2:
            idx, found = n // 2, False
        else:
            idx, found = int(edges[last_bad + 1]), True
    if return_info:
        return idx, found, dict(nblocks=nb, block_rows=n / nb, block_median=bm, ref=float(ref), tol=float(tol),
                                late_sd=float(late_sd), med_sd=med_sd, tail_frac=tail_frac)
    return idx, found


def _longest_true_run(same):
    run = np.zeros(same.shape[1], dtype=int)
    best = np.zeros_like(run)
    for row in same:
        run = (run + 1) * row
        best = np.maximum(best, run)
    return best


def _status(value, ok_thr, warn_thr, higher_is_better=False):
    if value is None or not np.isfinite(value):
        return "na"
    if higher_is_better:
        return "ok" if value >= ok_thr else ("warn" if value >= warn_thr else "bad")
    return "ok" if value < ok_thr else ("warn" if value < warn_thr else "bad")


_RANK = {"na": -1, "ok": 0, "warn": 1, "bad": 2}


def _worst(*s):
    return max(s, key=lambda v: _RANK[v])


def _keep_mask(an):
    """Boolean mask of unflagged walkers (all walkers if fewer than 4 would remain)."""
    keep = np.ones(an["w"], dtype=bool)
    if an.get("ensemble_wide"):
        return keep                      # most walkers are flagged: the "unflagged" group would not be representative
    keep[an["flagged"]] = False
    if keep.sum() < 4:
        keep[:] = True
    return keep


def burn_scan(run, an, fractions=(0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5), max_rows=3000, max_walkers=128,
              tau_c=5.0, seed=0):
    """
    How the convergence statistics change when the first part of the chain is discarded.
    For each fraction, R-hat, N/tau and drift are computed on what is left. For speed this scan uses at most
    `max_rows` evenly spaced steps and `max_walkers` unflagged walkers (N/tau is unaffected by thinning).
    Returns a dict of arrays: step (steps discarded), frac, rhat (max), n_over_tau (min), drift (max |z|), plateau.
    """
    rng = np.random.default_rng(seed)
    keep = np.nonzero(_keep_mask(an))[0]
    walkers = np.sort(rng.choice(keep, size=min(len(keep), max_walkers), replace=False))
    sstride = max(1, int(np.ceil(run.n / max_rows)))
    rows = np.arange(0, run.n, sstride)
    cols = {nm: _fill_nan(run.columns[nm][rows][:, walkers]) for nm in run.names}
    lnp_s = run.lnp[rows][:, walkers]
    out = {k: [] for k in ("step", "frac", "rhat", "n_over_tau", "drift", "plateau")}
    for f in fractions:
        b = int(round(f * len(rows)))
        if len(rows) - b < 60:
            continue
        rh, no, dz = [], [], []
        for nm in run.names:
            x = cols[nm][b:]
            tau = integrated_autocorr_time(x, tau_c)
            rb, rt = rhat_bulk_tail(x)
            rh.append(np.nanmax([rb, rt]) if np.isfinite([rb, rt]).any() else np.nan)
            no.append(x.shape[0] / tau if np.isfinite(tau) and tau > 0 else np.nan)
            dz.append(abs(drift_z(x, tau)))
        out["step"].append(int(run.steps[rows[b]]))
        out["frac"].append(b / len(rows))
        out["rhat"].append(np.nanmax(rh) if np.isfinite(rh).any() else np.nan)
        out["n_over_tau"].append(np.nanmin(no) if np.isfinite(no).any() else np.nan)
        out["drift"].append(np.nanmax(dz) if np.isfinite(dz).any() else np.nan)
        out["plateau"].append(float(np.nanmedian(lnp_s[b:])))
    return {k: np.array(v) for k, v in out.items()}


# ============================================================================== analysis

def _diagnose(an, run):
    """Plain-language reading of the numbers and suggestions of what to try, as two lists of sentences (rule based)."""
    obs, tips = [], []
    w, nf = an["w"], len(an["flagged"])
    tot = run.steps[-1] + 1
    r = an["mixing_ratio"]
    a = an["accept_median"]
    T = an["tail_steps"]
    rows = an["params"]
    if an["ensemble_wide"]:
        obs.append(f"{nf} of {w} walkers ({100 * nf / w:.0f}%) are flagged, so this is behaviour of the whole ensemble, "
                   f"not of a few bad walkers. All walkers are therefore used for bands and axis limits.")
    elif nf:
        obs.append(f"{nf} of {w} walkers are flagged (see the Walkers section); the rest look alike.")
    if r >= 0.3:
        obs.append(f"Walkers do not mix in ln p: the medians of different walkers spread by {an['between_sd']:.2f}, while one walker "
                   f"scatters by {an['within_sd']:.2f} (ratio {r:.2f}). In a mixed ensemble every walker visits the same ln p levels and the "
                   f"ratio is below about 0.3; here each walker keeps its own level for long stretches.")
    if a < 0.1:
        if a * T < 1 and not an["accept_from_backend"]:
            obs.append(f"The typical walker did not move at all in the last {T} steps, so its acceptance is below {1 / T:.4f}; "
                       f"it waits more than {T} steps between accepted moves and the autocorrelation time is at least that long.")
        else:
            obs.append(f"Typical walkers move in only {100 * a:.2f}% of steps ({'acceptance from the backend' if an['accept_from_backend'] else 'fraction of steps moved'}), "
                       f"so they wait about {1 / max(a, 1e-9):,.0f} steps between accepted moves; the autocorrelation time is at least of that order.")
    late = an["est_found"] and an["est_step"] > 0.5 * tot
    if not an["est_found"]:
        obs.append("No ln p plateau was found: the ensemble median is still changing at the end of the run.")
    elif late:
        obs.append(f"The ensemble-median ln p reaches its plateau only at step {an['est_step']:,}, {100 * an['est_step'] / tot:.0f}% of the run, "
                   f"which leaves {tot - an['est_step']:,} steps after the burn-in.")
    n_over = [(r_["n_over_tau"], r_["name"]) for r_ in rows if np.isfinite(r_["n_over_tau"])]
    short = bool(n_over) and min(n_over)[0] < 50
    if n_over and min(n_over)[0] < 20:
        v, nm = min(n_over)
        obs.append(f"The chain is only {v:.1f} autocorrelation times long for {nm} (50 or more is wanted), so the statistics are not reliable yet.")
    elif short:
        v, nm = min(n_over)
        obs.append(f"The chain is {v:.1f} autocorrelation times long for {nm}; 50 or more is wanted before the statistics can be trusted.")
    rh = [r_["rhat"] for r_ in rows if np.isfinite(r_["rhat"])]
    rc = [r_["rhat_clean"] for r_ in rows if np.isfinite(r_["rhat_clean"])]
    if rh and rc and max(rh) > 1.05 and max(rc) < 1.05:
        obs.append(f"R-hat falls from {max(rh):.2f} to {max(rc):.2f} when the flagged walkers are left out, so they account for most of the disagreement between walkers.")

    # ---- what to try
    if short:
        taus = [(r_["tau"], r_["name"]) for r_ in rows if np.isfinite(r_["tau"])]
        if taus:
            tau_max, nm = max(taus)
            needed, have = 50 * tau_max, an["n_post"] * run.stride
            tips.append(f"Run longer. The slowest autocorrelation time is about {tau_max:,.0f} steps ({nm}); 50 autocorrelation times need about {needed:,.0f} steps after the burn-in, "
                        f"roughly {max(needed - have, 0):,.0f} more than you have. The estimate of tau is itself uncertain while the chain is this short, so treat this as a minimum.")
    if r >= 0.3 or a < 0.1 or an["ensemble_wide"]:
        tips.append("Help the walkers move. emcee can mix in differential-evolution moves, for example moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)]. "
                    "These often cope better with many dimensions and with correlated or multimodal posteriors, but they are not guaranteed to fix it.")
        tips.append("Check the likelihood. If many proposals return -inf (for example failed waveforms or points outside a prior bound), most moves are rejected and walkers look stuck.")
        tips.append("Look for separate modes. If the per-walker histograms in the Log-probability section form distinct groups, walkers are sitting in different modes, and a longer run or a different sampler is needed.")
    if a > 0.6:
        tips.append("Acceptance is high, so the moves may be too timid. A larger stretch scale, for example emcee.moves.StretchMove(a=3), proposes bolder moves.")
    if nf and not an["ensemble_wide"]:
        tips.append("Deal with the flagged walkers: restart frozen or lagging walkers from the positions of healthy walkers, or leave them out of the analysis. "
                    "The histograms in the Walkers section show what changes when they are left out.")
    if late:
        tips.append("The burn-in is long. Start the next run from the end of this one (draw the initial positions from its last step) so that the burn-in is not repeated.")
    if not an["est_found"]:
        tips.append("Keep running until the median ln p across walkers stops rising.")
    if not obs:
        obs.append("Nothing stands out in the automated checks.")
    if not tips:
        tips.append("No action needed. As a final check, repeat the run from different starting points and compare the posteriors.")
    return obs, tips


def analyze(run, burn=0, tau_c=5.0, curve_points=10, curve_walkers=128, scan=True,
            stuck_run=None, lnp_z=4.0, lnp_gap_sigma=0.5, offset_z=5.0,
            ess_target=1000, seed=0, verbose=True):
    """
    Run all diagnostics. Returns a dict used by the plotting and report functions.

    burn : what to DISCARD before computing tau, R-hat, ESS and drift.
           0 (default) = nothing; 'auto' = the estimated burn-in; a float in (0, 1) = that fraction of the
           steps; an int = that many steps (original step units).
    The estimated burn-in is always computed and shown, and it is also used to decide which part of the run
    to judge walkers on (frozen / low ln p / offset), because the early transient differs for every walker.
    Thresholds can be changed with the keyword arguments; see the report's Methods section.
    """
    rng = np.random.default_rng(seed)
    n, w = run.n, run.nwalkers
    lnp = run.lnp
    names = run.names

    # ---- burn-in: estimated (always) and discarded (only if asked)
    est_idx, est_found, binfo = estimate_burnin(lnp, return_info=True)
    cap = n - 20 if n > 40 else 0
    est_idx = int(min(est_idx, cap))
    if isinstance(burn, str):
        if burn != "auto":
            raise ValueError("burn must be 0, 'auto', a fraction in (0, 1) or a number of steps")
        bi = est_idx
    elif burn is None or burn == 0:
        bi = 0
    elif isinstance(burn, float) and 0 < burn < 1:
        bi = int(burn * n)
    else:
        bi = int(np.searchsorted(run.steps, burn))
    bi = int(min(max(bi, 0), cap))
    vi = max(bi, est_idx)                       # walkers are judged on this part of the run
    n_post_steps = (n - bi) * run.stride
    an = {"n": n, "w": w, "stride": run.stride, "nsteps_total": run.nsteps_total,
          "burn_idx": bi, "burn_step": int(run.steps[bi]), "est_idx": est_idx, "est_step": int(run.steps[est_idx]),
          "est_found": bool(est_found), "view_idx": vi, "view_step": int(run.steps[vi]),
          "n_post": n - bi, "n_view": n - vi, "burn_info": dict(binfo, block_steps=binfo["block_rows"] * run.stride)}

    # ---- per-walker statistics (after the estimated burn-in)
    lp = lnp[vi:].astype(np.float64)
    wmed = np.nanmedian(lp, axis=0)
    wq16, wq84 = np.nanpercentile(lp, [16, 84], axis=0)
    wmax = np.nanmax(lp, axis=0)
    pooled_sd = robust_sd(lp)
    ens_med = float(np.nanmedian(wmed))
    gap_sigma = (wmed - ens_med) / max(pooled_sd, 1e-12)
    zl = robust_z(wmed)
    nonfinite_frac = run.bad[vi:].mean(axis=0)

    if run.tail_x is not None and len(run.tail_x) > 10:
        tx = run.tail_x
        moved = (np.diff(tx, axis=0) != 0).any(axis=2)
        moved_src = f"last {len(tx)} steps at full resolution"
    else:
        moved = (np.diff(run.x[vi:], axis=0) != 0).any(axis=2)
        moved_src = f"thinned chain (stride {run.stride}); frozen-run length is only a lower bound"
    moved_frac = moved.mean(axis=0)
    longest = _longest_true_run(~moved)
    accept = run.accepted if run.accepted is not None else moved_frac

    wmeans = np.empty((len(names), w))
    psd = np.empty(len(names))
    for j, nm in enumerate(names):
        col = _fill_nan(run.columns[nm][vi:])
        wmeans[j] = col.mean(axis=0)
        psd[j] = col.std()
    if w >= 8:
        zmean = np.array([robust_z(wmeans[j]) for j in range(len(names))])
        dev = (wmeans - np.median(wmeans, axis=1, keepdims=True)) / np.maximum(psd[:, None], 1e-300)
        sidx = [j for j, nm in enumerate(names) if nm in set(run.col_names)]      # sampled columns only (not ln p, not derived)
        offset_flag = ((np.abs(zmean[sidx]) > offset_z) & (np.abs(dev[sidx]) > 0.5)).any(axis=0)
    else:
        zmean = np.zeros_like(wmeans)
        offset_flag = np.zeros(w, dtype=bool)

    # A walker is "frozen" only if it moves less than chance allows given how often walkers move in THIS run.
    # (A fixed threshold would flag almost every walker in a low-acceptance run.)
    T = moved.shape[0]
    k_obs = moved.sum(axis=0)
    a_typ = float(np.median(moved_frac))
    if a_typ > 0:
        # unlikely by chance (Bonferroni over walkers) AND a practically large shortfall (under half the typical rate);
        # moves are not independent, so the binomial test alone would flag the odd healthy walker
        rate_low = (binom.cdf(k_obs, T, a_typ) < 0.01 / w) & (moved_frac < 0.5 * a_typ)
        if stuck_run is None:
            run_thr = max(100, int(np.ceil(np.log(max(w * T * a_typ / 0.01, 2.0)) / -np.log1p(-min(a_typ, 0.999)))))
        else:
            run_thr = int(stuck_run)
        stuck = rate_low | (longest >= run_thr) | ((k_obs == 0) & (T >= 200))
    else:
        run_thr = stuck_run or 100
        stuck = np.zeros(w, dtype=bool)                                         # nobody moves: ensemble-wide, not per walker
    low = (zl < -lnp_z) & (gap_sigma < -lnp_gap_sigma)
    flagged = stuck | low | offset_flag | (nonfinite_frac > 0.5)
    within = np.array([robust_sd(lp[:, i]) for i in range(w)])
    between_sd, within_sd = robust_sd(wmed), float(np.nanmedian(within))
    mixing_ratio = between_sd / max(within_sd, 1e-12)
    an.update(zmean=zmean, wmeans=wmeans, pooled_sd_lnp=pooled_sd, ens_median_lnp=ens_med, moved_src=moved_src,
              between_sd=between_sd, within_sd=within_sd, mixing_ratio=float(mixing_ratio), moved_median_tail=a_typ,
              accept_median=float(np.median(accept)), tail_steps=int(T),
              frozen_run_threshold=int(run_thr), ensemble_wide=bool(flagged.sum() / w > 0.2),
              stuck=np.nonzero(stuck)[0], low=np.nonzero(low)[0], offset=np.nonzero(offset_flag)[0],
              nonfinite=np.nonzero(nonfinite_frac > 0.5)[0], flag_rules_version=FLAG_RULES_VERSION,
              flagged=np.nonzero(flagged)[0], accept=np.asarray(accept), accept_from_backend=run.accepted is not None)

    walkers = []
    for i in range(w):
        flags = [t for t, c in (("stuck", stuck[i]), ("low lnp", low[i]), ("offset", offset_flag[i]),
                                ("non-finite", nonfinite_frac[i] > 0.5)) if c]
        walkers.append(dict(id=i, flags=", ".join(flags) if flags else "", median_lnp=float(wmed[i]),
                            gap_sigma=float(gap_sigma[i]), q16=float(wq16[i]), q84=float(wq84[i]),
                            max_lnp=float(wmax[i]), moved=float(moved_frac[i]), accept=float(accept[i]),
                            longest_frozen=int(longest[i]), nonfinite=float(nonfinite_frac[i]),
                            last_lnp=float(lnp[-1, i]) if np.isfinite(lnp[-1, i]) else float("nan")))
    an["walkers"] = walkers

    # ---- per-parameter convergence statistics (on the chain minus the DISCARDED burn-in only)
    clean = ~flagged
    use_clean = flagged.any() and clean.sum() >= 4
    rows = []
    for j, nm in enumerate(names):
        if verbose:
            print(f"  [{j + 1}/{len(names)}] {nm}", flush=True)
        x = _fill_nan(run.columns[nm][bi:])
        tau_thin = integrated_autocorr_time(x, tau_c)
        tau = tau_thin * run.stride
        rb, rt = rhat_bulk_tail(x)
        rhat = np.nanmax([rb, rt]) if np.isfinite([rb, rt]).any() else np.nan
        if use_clean:
            cb, ct = rhat_bulk_tail(x[:, clean])
            rclean = np.nanmax([cb, ct]) if np.isfinite([cb, ct]).any() else np.nan
        else:
            rclean = np.nan
        ok_tau = np.isfinite(tau) and tau > 0
        n_over_tau = n_post_steps / tau if ok_tau else np.nan
        ess = w * n_post_steps / tau if ok_tau else np.nan
        dz = drift_z(x, tau_thin)
        q05, q50, q95 = np.percentile(x, [5, 50, 95])
        st_tau = _status(n_over_tau, 50, 20, higher_is_better=True)
        st_rhat = _status(rhat, 1.01, 1.05)
        st_ess = _status(ess, ess_target, ess_target / 2.5, higher_is_better=True)
        st_drift = _status(abs(dz) if np.isfinite(dz) else np.nan, 3, 5)
        res = dict(name=nm, mean=float(x.mean()), std=float(x.std()), q05=float(q05), q50=float(q50), q95=float(q95),
                   rhat_bulk=rb, rhat_tail=rt, rhat=rhat, rhat_clean=rclean, tau=tau, n_over_tau=n_over_tau, ess=ess,
                   drift_z=dz, st_tau=st_tau, st_rhat=st_rhat, st_ess=st_ess, st_drift=st_drift,
                   tau_resolved=bool(np.isfinite(tau_thin) and tau_thin >= 3), derived=nm in run.derived_names)
        res["status"] = _worst(st_tau, st_rhat, st_ess, st_drift)
        rows.append(res)
    an["params"] = rows

    # ---- tau as a function of chain length (emcee-tutorial style convergence plot)
    n_use = n - bi
    if curve_points and n_use > 60:
        sub = np.sort(rng.choice(w, size=min(w, curve_walkers), replace=False))
        Ns = np.unique(np.geomspace(max(30, n_use // 40), n_use, curve_points).astype(int))
        curve = np.full((len(Ns), len(names)), np.nan)
        for j, nm in enumerate(names):
            x = _fill_nan(run.columns[nm][bi:][:, sub])
            for k, Nk in enumerate(Ns):
                curve[k, j] = integrated_autocorr_time(x[:Nk], tau_c) * run.stride
        an["tau_curve"] = dict(N=Ns * run.stride, tau=curve, names=names)
    else:
        an["tau_curve"] = None

    # ---- overall checks
    n_over = np.array([r["n_over_tau"] for r in rows], dtype=float)
    rh = np.array([r["rhat"] for r in rows], dtype=float)
    ess_arr = np.array([r["ess"] for r in rows], dtype=float)
    dz_arr = np.abs(np.array([r["drift_z"] for r in rows], dtype=float))
    acc_med = float(np.nanmedian(accept))

    def pick(arr, fn):
        return int(fn(arr)) if np.isfinite(arr).any() else 0

    wn, wr, we, wd = pick(n_over, np.nanargmin), pick(rh, np.nanargmax), pick(ess_arr, np.nanargmin), pick(dz_arr, np.nanargmax)
    pct = 100.0 * an["est_step"] / max(run.steps[-1] + 1, 1)
    plateau_txt = (f"plateau from step {an['est_step']:,} ({pct:.0f}% of the run)" + (f"; the first {an['burn_step']:,} steps are removed" if bi > 0 else "")
                   if est_found else "no plateau found")
    alpha_ = 0.01 / w
    moves_cutoff = min(int(binom.ppf(alpha_, T, a_typ)) - 1, int(np.ceil(0.5 * a_typ * T)) - 1) if a_typ > 0 else -1
    low_units = max(lnp_z * between_sd, lnp_gap_sigma * pooled_sd)
    an["flag_info"] = dict(tail_steps=int(T), typical_move=a_typ, moves_cutoff=moves_cutoff, run_threshold=int(run_thr),
                           low_units=float(low_units), between_sd=float(between_sd), pooled_sd=float(pooled_sd),
                           lnp_z=lnp_z, lnp_gap_sigma=lnp_gap_sigma, offset_z=offset_z)
    checks = [
        ("tau", "Chain length vs autocorrelation time", _status(np.nanmin(n_over), 50, 20, True),
         f"min N/tau = {np.nanmin(n_over):.1f} ({names[wn]})", "Good if the chain is at least 50 autocorrelation times long; check from 20; problem below 20."),
        ("rhat", "Rank-normalised split R-hat", _status(np.nanmax(rh), 1.01, 1.05),
         f"max R-hat = {np.nanmax(rh):.3f} ({names[wr]})", "Good below 1.01; check up to 1.05; problem above 1.05."),
        ("ess", "Effective sample size", _status(np.nanmin(ess_arr), ess_target, ess_target / 2.5, True),
         f"min ESS = {np.nanmin(ess_arr):.0f} ({names[we]})", f"Good at {ess_target} or more independent samples; check from {ess_target / 2.5:.0f}; problem below that."),
        ("drift", "Half-vs-half drift of the means", _status(np.nanmax(dz_arr), 3, 5),
         f"max |z| = {np.nanmax(dz_arr):.1f} ({names[wd]})", "Good if the two halves of the chain agree within 3 standard errors; check up to 5; problem above 5."),
        ("mixing", "Do walkers mix in ln p?", _status(mixing_ratio, 0.3, 0.6),
         f"between/within = {mixing_ratio:.2f} (walker medians spread {between_sd:.2f}, within a walker {within_sd:.2f})",
         "Good below 0.3; check below 0.6; problem above. Walkers of a mixed ensemble sit at the same ln p levels."),
        ("stuck", "Frozen walkers", "ok" if len(an["stuck"]) == 0 else ("warn" if len(an["stuck"]) / w < 0.05 else "bad"),
         f"{len(an['stuck'])} of {w}", f"Frozen: moved far less than other walkers do, or stayed unchanged for {run_thr} steps in a row or more (see the Walkers section)."),
        ("low", "Walkers with low log-probability",
         "ok" if len(an["low"]) == 0 else ("warn" if len(an["low"]) / w < 0.05 else "bad"),
         f"{len(an['low'])} of {w}", f"Typical ln p more than {low_units:.1f} units below the middle walker (see the Walkers section)."),
        ("offset", "Walkers offset in some parameter",
         "ok" if len(an["offset"]) == 0 else ("warn" if len(an["offset"]) / w < 0.05 else "bad"),
         f"{len(an['offset'])} of {w}", "In some parameter the walker's average is far from the other walkers' averages (see the Walkers section)."),
        ("accept", "Acceptance fraction", "ok" if 0.15 <= acc_med <= 0.6 else ("warn" if 0.08 <= acc_med <= 0.75 else "bad"),
         f"median {acc_med:.2f}" + ("" if run.accepted is not None else " (fraction of steps moved)"),
         "emcee recommends roughly 0.2 to 0.5. Good from 0.15 to 0.6; problem outside 0.08 to 0.75."),
        ("plateau", "Burn-in plateau in log-probability", "ok" if est_found else "warn", plateau_txt,
         "The median ln p across walkers must stop rising."),
    ]
    an["checks"] = [dict(key=k, label=l, status=s, value=v, rule=r) for k, l, s, v, r in checks]
    an["verdict"] = _worst(*[c[2] for c in checks])
    an["diagnosis"], an["suggestions"] = _diagnose(an, run)

    an["burn_scan"] = burn_scan(run, an, tau_c=tau_c, seed=seed) if scan else None
    return an


# ============================================================================== plotting helpers

def _plot_idx(n, max_pts=1200):
    return np.unique(np.linspace(0, n - 1, min(n, max_pts)).astype(int))


def _nanq(a, q, axis):
    return np.nanquantile(a, q, axis=axis) if np.isnan(a).any() else np.quantile(a, q, axis=axis)


def _block_reduce(a, nb, func=np.nanmedian):
    n = a.shape[0]
    nb = int(min(nb, n))
    edges = np.linspace(0, n, nb + 1).astype(int)
    return np.array([func(a[s:e], axis=0) for s, e in zip(edges[:-1], edges[1:])]), edges


def _fmt(x):
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{x:.10g}" if abs(x) >= 1e5 else f"{x:.4g}"


def _robust_ylim(a, lo=0.2, hi=99.8, pad=0.08):
    a = a[np.isfinite(a)]
    if a.size == 0:
        return -1, 1
    l, h = np.percentile(a, [lo, hi])
    if h == l:
        h, l = l + 1, l - 1
    return l - pad * (h - l), h + pad * (h - l)


def _plateau_ylim(a, pad=0.12):
    """y-limits around the plateau: the 0.1-99.9% range, but never more than 10 robust sigma below or 8 above the median."""
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return -1.0, 1.0
    med = float(np.median(a))
    sd = max(robust_sd(a), 1e-9 * max(1.0, abs(med)))
    lo = max(np.percentile(a, 0.1), med - 10 * sd)
    hi = min(np.percentile(a, 99.9), med + 8 * sd)
    if hi <= lo:
        lo, hi = med - sd, med + sd
    return lo - pad * (hi - lo), hi + pad * (hi - lo)


def _pick_walkers(an, k_random=8, seed=1):
    rng = np.random.default_rng(seed)
    flagged = list(an["flagged"][:8])
    rest = [i for i in range(an["w"]) if i not in set(an["flagged"])]
    rnd = list(rng.choice(rest, size=min(k_random, len(rest)), replace=False)) if rest else []
    return flagged, rnd


def _mark_burn(ax, an, run=None, offset=0.0):
    """Shade the discarded burn-in (if any) and draw the estimated burn-in as a dashed line."""
    if an["burn_idx"] > 0:
        ax.axvspan(offset, an["burn_step"] + offset, color=C["gray"], alpha=0.15, lw=0)
    if an["est_found"] and an["est_step"] > 0:
        ax.axvline(an["est_step"] + offset, color=C["burn"], ls="--", lw=1.4)


def _density(v, bins):
    """Histogram density normalised by ALL samples, so mass outside the plotted range is not renormalised away."""
    h, _ = np.histogram(v, bins=bins)
    return h / (max(len(v), 1) * np.maximum(np.diff(bins), 1e-300))


def _clean_post(run, an, arr=None):
    """Post burn-in values of the walkers that are NOT flagged (all walkers if too few remain).
    Axis limits come from these, so that one frozen walker cannot squash the plot."""
    a = run.lnp if arr is None else arr
    keep = np.ones(an["w"], dtype=bool)
    keep[an["flagged"]] = False
    if keep.sum() < 4:
        keep[:] = True
    return np.asarray(a[an["view_idx"]:][:, keep], dtype=np.float64)


def _offscale(ax, an, values_by_walker, label_fmt="walker {i}: {v:.4g}"):
    """Annotate flagged walkers whose (median) value lies outside the current y-limits."""
    lo, hi = ax.get_ylim()
    msgs = [label_fmt.format(i=i, v=v) for i, v in values_by_walker.items() if v < lo or v > hi]
    if msgs:
        extra = "" if len(msgs) <= 4 else f"\n(+{len(msgs) - 4} more)"
        ax.text(0.995, 0.03, "off-scale flagged: " + "; ".join(msgs[:4]) + extra, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, color=C["red"],
                bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))


# ============================================================================== plots

def plot_lnp_overview(run, an, n_random=30, n_flagged=20, seed=1):
    """
    Static summary of ln p: individual walkers (a random sample of unflagged ones in grey, flagged ones in red)
    and the median over all walkers. Whole run clipped to the plateau scale, and a post burn-in zoom.
    Axis limits come from unflagged walkers; flagged walkers that fall outside are listed in red.
    The interactive version (all walkers, switchable groups) is in the report.
    """
    rng = np.random.default_rng(seed)
    with _style():
        fig, axes = plt.subplots(2, 1, figsize=(11, 6.6), gridspec_kw=dict(height_ratios=[1.1, 1]))
        idx = _plot_idx(run.n)
        s = run.steps[idx]
        med = np.nanmedian(run.lnp[idx].astype(np.float64), axis=1)
        fl = np.asarray(an["flagged"])
        unfl = np.setdiff1d(np.arange(an["w"]), fl)
        pick_u = rng.choice(unfl, size=min(len(unfl), n_random), replace=False) if len(unfl) else []
        pick_f = fl[np.argsort([-abs(an["walkers"][i]["gap_sigma"]) for i in fl])][:n_flagged] if len(fl) else []
        clean = _clean_post(run, an)
        centre, sd = float(np.nanmedian(clean)), max(robust_sd(clean), 1e-9)
        fl_med = {i: an["walkers"][i]["median_lnp"] for i in fl}
        for ax, title, zoom in ((axes[0], "Individual walkers, whole run", False),
                                (axes[1], "Post burn-in zoom", True)):
            for i in pick_u:
                ax.plot(s, run.lnp[idx, i], color=C["gray"], lw=0.5, alpha=0.5)
            for i in pick_f:
                ax.plot(s, run.lnp[idx, i], color=C["red"], lw=0.8, alpha=0.7)
            ax.plot(s, med, color=C["ink"], lw=1.6)
            _mark_burn(ax, an, run)
            ax.set_title(title, loc="left")
            ax.set_ylabel("ln posterior")
            if zoom:
                ax.set_xlim(an["view_step"], run.steps[-1])
                ax.set_ylim(*_plateau_ylim(clean))
                ax.set_xlabel("step")
                ax.text(0.995, 0.97, "y-axis limited to the plateau; walkers outside it are clipped", transform=ax.transAxes,
                        ha="right", va="top", fontsize=8, color=C["gray"])
            else:
                ax.set_ylim(centre - 25 * sd, centre + 6 * sd)
                ax.text(0.995, 0.97, "y-axis clipped to 25 sigma below the plateau", transform=ax.transAxes,
                        ha="right", va="top", fontsize=8, color=C["gray"])
            _offscale(ax, an, fl_med, "walker {i}: median {v:.0f}")
        axes[0].plot([], [], color=C["gray"], lw=1, label=f"{len(pick_u)} random unflagged walkers")
        if len(pick_f):
            axes[0].plot([], [], color=C["red"], lw=1, label=f"{len(pick_f)} of {len(fl)} flagged walkers")
        axes[0].plot([], [], color=C["ink"], lw=1.6, label="median over all walkers")
        axes[0].legend(ncol=3, loc="lower right", bbox_to_anchor=(1, 1.0))
        fig.tight_layout(h_pad=1.6)
    return fig


def plot_lnp_distributions(run, an, max_walkers=60, seed=2):
    """Left: per-walker histograms of ln p (all walkers should overlap). Right: sorted per-walker medians.
    The axes are set by the unflagged walkers; flagged walkers outside are listed in red."""
    with _style():
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw=dict(width_ratios=[1.15, 1]))
        post = run.lnp[an["view_idx"]:].astype(np.float64)
        ref = an["ens_median_lnp"]
        clean = _clean_post(run, an) - ref
        lo, hi = _plateau_ylim(clean, pad=0.05)
        bins = np.linspace(lo, hi, 70)
        rng = np.random.default_rng(seed)
        fl = set(an["flagged"].tolist())
        gap = {i: abs(an["walkers"][i]["gap_sigma"]) for i in fl}
        fl_show = set(sorted(fl, key=lambda i: -gap[i])[:20])          # the 20 most deviant, never hundreds
        pool = [i for i in range(an["w"]) if i not in fl]
        pick = set(rng.choice(pool, size=min(len(pool), max_walkers), replace=False).tolist()) if pool else set()
        pick |= fl_show
        a_fl = 0.9 if len(fl_show) <= 6 else 0.45
        for i in sorted(pick):
            v = post[:, i] - ref
            v = v[np.isfinite(v)]
            if v.size == 0:
                continue
            h = _density(v, bins)
            a1.step(bins[:-1], h, where="post", color=C["red"] if i in fl_show else C["blue"],
                    alpha=a_fl if i in fl_show else 0.22, lw=1.1 if i in fl_show else 0.8, zorder=3 if i in fl_show else 2)
        allv = clean.ravel()
        allv = allv[np.isfinite(allv)]
        h = _density(allv, bins)
        a1.step(bins[:-1], h, where="post", color=C["ink"], lw=2.0, label="all walkers" if an["ensemble_wide"] else "unflagged walkers", zorder=4)
        a1.plot([], [], color=C["blue"], alpha=0.5, label=f"individual unflagged walkers ({min(len(pool), max_walkers)} shown)")
        if fl:
            a1.plot([], [], color=C["red"], label=f"flagged (the {len(fl_show)} most deviant of {len(fl)})")
        a1.set_xlim(lo, hi)
        a1.set_ylim(0, 1.4 * max(h.max(), 1e-12))     # a frozen walker is a spike; do not let it set the scale
        a1.set_xlabel("ln p - ensemble median")
        a1.set_ylabel("density")
        a1.set_title("Per-walker log-probability distributions", loc="left")
        a1.legend(loc="upper left")
        off = {i: an["walkers"][i]["median_lnp"] - ref for i in range(an["w"])}
        far = [(i, v) for i, v in off.items() if v < lo or v > hi]
        if far:
            far.sort(key=lambda t: -abs(t[1]))
            a1.text(0.995, 0.03, f"{len(far)} walker(s) with their median outside this range, e.g.\n" +
                    "\n".join(f"walker {i}: {v:+.0f}" for i, v in far[:4]), transform=a1.transAxes,
                    ha="right", va="bottom", fontsize=8, color=C["red"],
                    bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))

        wm = np.array([r["median_lnp"] for r in an["walkers"]]) - ref
        q16 = np.array([r["q16"] for r in an["walkers"]]) - ref
        q84 = np.array([r["q84"] for r in an["walkers"]]) - ref
        keep = np.array([(i not in fl) or an["ensemble_wide"] for i in range(an["w"])])
        ylo, yhi = _robust_ylim(np.r_[q16[keep], q84[keep]] if keep.sum() > 3 else np.r_[q16, q84], 0.5, 99.5, pad=0.15)
        order = np.argsort(wm)
        rank = np.arange(an["w"])
        isfl = np.isin(order, [] if an["ensemble_wide"] else list(fl))
        colors = np.where(isfl, C["red"], C["blue"])
        a2.vlines(rank, np.clip(q16[order], ylo, yhi), np.clip(q84[order], ylo, yhi), colors=colors, alpha=0.35, lw=1)
        pos = np.clip(wm[order], ylo, yhi)
        clipped = (wm[order] < ylo) | (wm[order] > yhi)
        a2.scatter(rank[~clipped], pos[~clipped], s=7, c=colors[~clipped], zorder=3)
        a2.scatter(rank[clipped], pos[clipped], s=36, marker="v", c=C["red"], zorder=4)
        a2.axhline(0, color=C["ink"], lw=1)
        a2.set_ylim(ylo, yhi)
        if clipped.any():
            a2.text(0.995, 0.03, f"{int(clipped.sum())} walker(s) off-scale (triangles)", transform=a2.transAxes,
                    ha="right", va="bottom", fontsize=8, color=C["red"])
        a2.set_xlabel("walker (sorted by median)")
        a2.set_ylabel("ln p - ensemble median")
        a2.set_title("Per-walker median and 16-84% range", loc="left")
        fig.tight_layout()
    return fig


def plot_walker_heatmap(run, an, nb=300):
    """Walker x time map of block-median ln p relative to the ensemble median; stuck/lagging walkers are stripes."""
    with _style():
        lnp = run.lnp.astype(np.float64)
        blocks, edges = _block_reduce(lnp, nb)
        med = np.nanmedian(blocks, axis=1, keepdims=True)
        scale = max(an["pooled_sd_lnp"], 1e-9)
        z = (blocks - med) / scale
        order = np.argsort([r["median_lnp"] for r in an["walkers"]])
        z = z[:, order].T
        fig, ax = plt.subplots(figsize=(11, 5.2))
        xs = run.steps[np.minimum(edges, run.n - 1)]
        im = ax.pcolormesh(xs, np.arange(an["w"] + 1), z, cmap="RdBu", vmin=-3, vmax=3, shading="flat",
                           rasterized=True)
        ax.grid(False)
        _mark_burn(ax, an, run)
        pos = {w_: k for k, w_ in enumerate(order)}
        for i in an["flagged"][:40]:
            ax.plot([xs[0]], [pos[i] + 0.5], marker=">", color=C["red"], ms=5, clip_on=False)
        ax.set_xlabel("step")
        ax.set_ylabel("walker (sorted by median ln p, lowest at bottom)")
        ax.set_title("Block-median ln p relative to the ensemble (in units of the pooled sigma)", loc="left")
        cb = fig.colorbar(im, ax=ax, pad=0.01)
        cb.set_label("sigma below (red) / above (blue) the ensemble median")
        fig.tight_layout()
    return fig


def plot_traces(run, an, params=None, per_page=5, max_pts=1200):
    """
    Trace plots with marginal histograms. One figure per `per_page` parameters.
    Bands are across-walker quantiles; flagged walkers are drawn in red.
    """
    names = params or run.names
    idx = _plot_idx(run.n, max_pts)
    s = run.steps[idx]
    flagged, rnd = _pick_walkers(an)
    prm = {r["name"]: r for r in an["params"]}
    bi = an["view_idx"]
    figs = []
    with _style():
        for p0 in range(0, len(names), per_page):
            chunk = names[p0:p0 + per_page]
            fig = plt.figure(figsize=(11, 2.15 * len(chunk) + 0.5))
            gs = GridSpec(len(chunk), 2, width_ratios=[5, 1], hspace=0.38, wspace=0.03, figure=fig)
            axl = None
            for r, nm in enumerate(chunk):
                col = run.columns[nm]
                at = fig.add_subplot(gs[r, 0], sharex=axl)
                ah = fig.add_subplot(gs[r, 1], sharey=at)
                axl = axl or at
                x = col[idx].astype(np.float64)
                q = _nanq(x[:, _keep_mask(an)], [0.05, 0.16, 0.5, 0.84, 0.95], axis=1)
                at.fill_between(s, q[0], q[4], color=C["light"], alpha=0.5, lw=0)
                at.fill_between(s, q[1], q[3], color=C["blue"], alpha=0.35, lw=0)
                for i in rnd:
                    at.plot(s, col[idx, i], color=C["gray"], lw=0.45, alpha=0.55)
                for i in flagged:
                    at.plot(s, col[idx, i], color=C["red"], lw=0.8, alpha=0.9)
                at.plot(s, q[2], color=C["ink"], lw=1.2)
                _mark_burn(at, an, run)
                post = col[bi:].astype(np.float64)
                at.set_ylim(*_robust_ylim(_clean_post(run, an, col), 0.1, 99.9, pad=0.15))
                fl_vals = {i: float(np.nanmedian(post[:, i])) for i in an["flagged"]}
                _offscale(at, an, fl_vals, "walker {i}: {v:.4g}")
                at.set_ylabel(nm, fontsize=10)
                info = prm[nm]
                txt = f"tau={_fmt(info['tau'])}   N/tau={_fmt(info['n_over_tau'])}   R-hat={_fmt(info['rhat'])}"
                if np.isfinite(info["rhat_clean"]):
                    txt += f" (without flagged: {_fmt(info['rhat_clean'])})"
                txt += f"   ESS={_fmt(info['ess'])}"
                at.text(0.01, 0.97, txt, transform=at.transAxes, va="top", fontsize=8,
                        color={"ok": C["green"], "warn": C["amber"], "bad": C["red"], "na": C["gray"]}[info["status"]],
                        bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.5))
                plt.setp(at.get_xticklabels(), visible=(r == len(chunk) - 1))
                if r == len(chunk) - 1:
                    at.set_xlabel("step")
                vals = post.ravel()
                vals = vals[np.isfinite(vals)]
                lo, hi = at.get_ylim()
                bins = np.linspace(lo, hi, 45)
                ah.hist(vals, bins=bins, orientation="horizontal", density=True, color=C["gray"], alpha=0.35)
                half = post.shape[0] // 2
                for part, colr in ((post[:half], C["blue"]), (post[half:], C["orange"])):
                    pv = part.ravel()
                    pv = pv[np.isfinite(pv)]
                    if pv.size:
                        hh = _density(pv, bins)
                        ah.step(hh, bins[:-1], where="post", color=colr, lw=1.2)
                ah.grid(False)
                ah.set_xticks([])
                plt.setp(ah.get_yticklabels(), visible=False)
                ah.spines["left"].set_visible(False)
                if r == 0:
                    ah.set_title("after est. burn-in\n(blue: 1st half, orange: 2nd)", fontsize=8, loc="left")
            figs.append(fig)
    return figs


def plot_autocorr(an):
    """tau estimate vs chain length, with the N/50 line: curves must stay below it."""
    tc = an["tau_curve"]
    if tc is None:
        return None
    with _style():
        fig, ax = plt.subplots(figsize=(8.5, 4.6))
        N, tau, names = tc["N"], tc["tau"], tc["names"]
        final = np.nan_to_num(tau[-1], nan=0.0)
        top = set(np.argsort(final)[::-1][:6].tolist())
        cmap = plt.get_cmap("tab10")
        for j, nm in enumerate(names):
            if j in top:
                ax.plot(N, tau[:, j], marker="o", ms=3, lw=1.5, label=nm, color=cmap(len(ax.lines) % 10))
            else:
                ax.plot(N, tau[:, j], lw=0.8, color=C["gray"], alpha=0.35)
        ax.plot(N, N / 50.0, "--", color=C["ink"], lw=1.3, label="N / 50")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("number of steps used (post burn-in)")
        ax.set_ylabel("tau (steps)")
        ax.set_title("Autocorrelation time estimate vs chain length (six slowest labelled)", loc="left")
        ax.legend(ncol=2)
        fig.tight_layout()
    return fig


def plot_rhat_ess(an):
    """R-hat and ESS for every parameter, coloured by status."""
    rows = an["params"]
    names = [r["name"] for r in rows]
    cols = {"ok": C["green"], "warn": C["amber"], "bad": C["red"], "na": C["gray"]}
    with _style():
        h = max(3.2, 0.28 * len(rows) + 1.2)
        fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(11, h), sharey=True)
        y = np.arange(len(rows))
        rh = np.array([r["rhat"] for r in rows], dtype=float)
        a1.barh(y, np.nan_to_num(rh - 1, nan=0), left=1, color=[cols[r["st_rhat"]] for r in rows])
        a1.axvline(1.01, color=C["ink"], ls=":", lw=1)
        a1.axvline(1.05, color=C["ink"], ls="--", lw=1)
        a1.set_xlim(0.99, max(1.08, np.nanmax(rh) * 1.01 if np.isfinite(rh).any() else 1.08))
        a1.set_title("rank-normalised split R-hat", loc="left", fontsize=10)
        no = np.array([r["n_over_tau"] for r in rows], dtype=float)
        a2.barh(y, np.nan_to_num(no, nan=0), color=[cols[r["st_tau"]] for r in rows])
        a2.axvline(50, color=C["ink"], ls="--", lw=1)
        a2.set_title("N / tau  (want > 50)", loc="left", fontsize=10)
        es = np.array([r["ess"] for r in rows], dtype=float)
        a3.barh(y, np.nan_to_num(es, nan=0), color=[cols[r["st_ess"]] for r in rows])
        a3.set_title("effective sample size", loc="left", fontsize=10)
        a1.set_yticks(y)
        a1.set_yticklabels(names)
        a1.invert_yaxis()
        fig.tight_layout()
    return fig


def plot_acceptance(an):
    with _style():
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        a = np.asarray(an["accept"], dtype=float)
        ax.hist(a, bins=min(40, max(8, an["w"] // 6)), color=C["blue"], alpha=0.85)
        ax.axvline(np.nanmedian(a), color=C["ink"], lw=1.3, label=f"median {np.nanmedian(a):.2f}")
        ax.set_xlabel("acceptance fraction" + ("" if an["accept_from_backend"] else " (fraction of steps moved)"))
        ax.set_ylabel("walkers")
        ax.set_title("Acceptance per walker", loc="left")
        ax.legend()
        fig.tight_layout()
    return fig


def plot_walker_offsets(run, an):
    """Walkers x parameters: robust z-score of each walker's post-burn mean. A red/blue row is an outlier walker."""
    z = an["zmean"]
    with _style():
        order = np.argsort(-np.abs(z).max(axis=0))
        fig, ax = plt.subplots(figsize=(11, 4.6))
        im = ax.imshow(np.clip(z[:, order], -6, 6), aspect="auto", cmap="RdBu_r", vmin=-6, vmax=6,
                       interpolation="nearest")
        ax.grid(False)
        ax.set_yticks(range(len(run.names)))
        ax.set_yticklabels(run.names, fontsize=8)
        ax.set_xlabel("walker (sorted by largest |z|; most deviant on the left)")
        ax.set_title("Offset of each walker's mean from the other walkers (robust z)", loc="left")
        fig.colorbar(im, ax=ax, pad=0.01).set_label("z")
        fig.tight_layout()
    return fig


def plot_flagged(run, an, max_walkers=6):
    """Close-ups of the worst flagged walkers: ln p relative to the ensemble median on a symmetric-log axis,
    so that both the ensemble band (near zero) and a walker that is thousands of units away stay visible."""
    w = an["walkers"]
    stuck = set(an["stuck"].tolist())
    sev = sorted(an["flagged"].tolist(), key=lambda i: (-(i in stuck), w[i]["gap_sigma"]))[:max_walkers]
    if not sev:
        return None
    with _style():
        idx = _plot_idx(run.n)
        s = run.steps[idx]
        ref = an["ens_median_lnp"]
        lin = max(3 * an["pooled_sd_lnp"], 1e-6)
        q = _nanq(run.lnp[idx][:, _keep_mask(an)].astype(np.float64), [0.05, 0.5, 0.95], axis=1) - ref
        ncol = 2
        nrow = int(np.ceil(len(sev) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(11, 2.5 * nrow), squeeze=False)
        for ax in axes.ravel():
            ax.set_visible(False)
        for ax, i in zip(axes.ravel(), sev):
            ax.set_visible(True)
            ax.fill_between(s, q[0], q[2], color=C["light"], alpha=0.55, lw=0)
            ax.plot(s, q[1], color=C["ink"], lw=1)
            ax.plot(s, run.lnp[idx, i] - ref, color=C["red"], lw=1.1)
            ax.set_yscale("symlog", linthresh=lin)
            ax.axhline(0, color=C["gray"], lw=0.6)
            _mark_burn(ax, an, run)
            ax.set_title(f"walker {i}: {w[i]['flags']}  (moved {w[i]['moved']:.1%})", loc="left", fontsize=9)
            ax.set_xlabel("step", fontsize=9)
            ax.set_ylabel("ln p - median (symlog)", fontsize=9)
        fig.tight_layout()
    return fig


def plot_burnin(run, an):
    """
    Top: ln p relative to the late-run plateau against log(step), so the climb is visible whatever its length.
    Bottom: what R-hat, N/tau and drift would be if the first part of the run were discarded.
    Purple dashed = estimated burn-in; the green shading marks the good region of each statistic.
    """
    sc = an.get("burn_scan")
    with _style():
        fig = plt.figure(figsize=(11, 7.4))
        gs = GridSpec(2, 3, height_ratios=[1.3, 1], hspace=0.45, wspace=0.32, figure=fig,
                      left=0.07, right=0.98, top=0.95, bottom=0.08)
        ax = fig.add_subplot(gs[0, :])
        n = run.n
        idx = np.unique(np.concatenate([[0], np.geomspace(1, n, 1000).astype(int) - 1]))
        idx = idx[(idx >= 0) & (idx < n)]
        s = run.steps[idx] + 1.0
        bi_ = an["burn_info"]
        ref = bi_["ref"]                                   # the plateau the estimator compares with
        lin = max(3 * an["pooled_sd_lnp"], 1e-6)
        q = _nanq(run.lnp[idx][:, _keep_mask(an)].astype(np.float64), [0.05, 0.16, 0.5, 0.84, 0.95], axis=1) - ref
        flagged, rnd = _pick_walkers(an)
        ax.axhspan(-bi_["tol"], 0, color=C["green"], alpha=0.18, lw=0, label="tolerance below the plateau")
        ax.fill_between(s, q[0], q[4], color=C["light"], alpha=0.5, lw=0, label="90% of walkers")
        ax.fill_between(s, q[1], q[3], color=C["blue"], alpha=0.35, lw=0, label="68%")
        for i in rnd:
            ax.plot(s, run.lnp[idx, i] - ref, color=C["gray"], lw=0.5, alpha=0.5)
        for i in flagged:
            ax.plot(s, run.lnp[idx, i] - ref, color=C["red"], lw=0.9, alpha=0.9)
        ax.plot(s, q[2], color=C["ink"], lw=1.4, label="median")
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=lin)
        ax.set_ylim(top=4 * lin)
        ax.axhline(0, color=C["gray"], lw=0.6)
        _mark_burn(ax, an, offset=1.0)
        msg = (f"estimated burn-in: step {an['est_step']:,}" if an["est_found"] else "no plateau found")
        if an["burn_idx"] > 0:
            msg += f"   |   removed in this report: first {an['burn_step']:,} steps"
        ax.text(0.99, 0.95, msg, transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(fc="white", ec=C["grid"], pad=3))
        lo_, hi_ = ax.get_ylim()
        ticks = [t for t in (-1e6, -1e5, -1e4, -1e3, -100, -10, 0, 10, 100) if lo_ <= t <= hi_]
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xlabel("step + 1 (log scale)")
        ax.set_ylabel("ln p - plateau median (symlog)")
        ax.set_title("Approach to the plateau", loc="left")
        ax.legend(ncol=2, loc="upper left")
        if sc is not None and len(sc["step"]) > 1:
            panels = [("R-hat, worst parameter", sc["rhat"], 1.01, 1.05, False, False, "1.01: good below", "1.05: problem above"),
                      ("N / tau, worst parameter", sc["n_over_tau"], 50, 20, True, True, "50: good above", "20: problem below"),
                      ("|drift z|, worst parameter", sc["drift"], 3, 5, False, False, "3: good below", "5: problem above")]
            for k, (ttl, y, good, bad, higher, logy, glab, blab) in enumerate(panels):
                a = fig.add_subplot(gs[1, k])
                a.plot(sc["step"], y, marker="o", color=C["blue"], lw=1.4, label="after removing the first steps")
                a.axhline(good, color=C["green"], ls="--", lw=1.3, label=glab)
                a.axhline(bad, color=C["red"], ls=":", lw=1.6, label=blab)
                if logy:
                    a.set_yscale("log")
                lo, hi = a.get_ylim()
                if higher:
                    a.axhspan(good, hi, color=C["green"], alpha=0.08, lw=0)
                else:
                    a.axhspan(lo, good, color=C["green"], alpha=0.08, lw=0)
                a.set_ylim(lo, hi)
                if an["est_step"] > 0:
                    a.axvline(an["est_step"], color=C["burn"], ls="--", lw=1.4, label="estimated burn-in")
                if an["burn_idx"] > 0:
                    a.axvline(an["burn_step"], color=C["orange"], lw=1.4, label="removed in this report")
                a.set_title(ttl, loc="left", fontsize=10)
                a.set_xlabel("steps removed from the start")
                a.legend(fontsize=7, loc="best")
    return fig


def plot_acf(run, an, n_params=6, max_walkers=128, seed=0):
    """
    Autocorrelation function (mean over walkers) of the slowest parameters and of ln p, on the chain minus the
    discarded burn-in. Left: linear lag. Right: log lag, to see the tail. Legend gives tau.
    """
    rng = np.random.default_rng(seed)
    rows = {r["name"]: r for r in an["params"]}
    taus = np.array([rows[nm]["tau"] for nm in run.names], dtype=float)
    order = [run.names[i] for i in np.argsort(np.nan_to_num(taus, nan=-1.0))[::-1]]
    chosen = order[:n_params]
    if "ln_posterior" in run.names and "ln_posterior" not in chosen:
        chosen = ["ln_posterior"] + chosen[:-1]
    keep = np.nonzero(_keep_mask(an))[0]
    sub = np.sort(rng.choice(keep, size=min(len(keep), max_walkers), replace=False))
    bi = an["burn_idx"]
    n_post = run.n - bi
    finite = taus[np.isfinite(taus)]
    tmax = float(finite.max()) / run.stride if finite.size else 10.0
    max_lag = int(min(max(n_post // 3, 10), max(60, 12 * tmax)))
    curves = []
    for nm in chosen:
        acf, used = mean_acf(_fill_nan(run.columns[nm][bi:][:, sub]))
        if acf is not None:
            curves.append((nm, acf[:max_lag]))
    if not curves:
        return None
    lags = np.arange(max_lag) * run.stride
    with _style():
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
        cmap = plt.get_cmap("tab10")
        for k, (nm, acf) in enumerate(curves):
            lab = f"{nm} (tau = {_fmt(rows[nm]['tau'])})"
            a1.plot(lags[:len(acf)], acf, color=cmap(k % 10), lw=1.4, label=lab)
            a2.plot(lags[1:len(acf)], acf[1:], color=cmap(k % 10), lw=1.4)
        for a in (a1, a2):
            a.axhline(0, color=C["ink"], lw=0.8)
            a.set_xlabel("lag (steps)")
        a2.set_xscale("log")
        a1.set_ylabel("autocorrelation")
        a1.set_title("Autocorrelation function (slowest parameters)", loc="left")
        a2.set_title("Same, logarithmic lag", loc="left")
        a1.legend(fontsize=8)
        fig.tight_layout()
    return fig


# ============================================================================== HTML report

def _b64(fig, dpi=110):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _img(fig, alt, dpi=110):
    if fig is None:
        return '<p class="muted">Nothing to show here: no walker was flagged.</p>'
    return f'<img alt="{_html.escape(alt)}" src="data:image/png;base64,{_b64(fig, dpi)}">'


def _sig(arr, n=5):
    return [float(f"{v:.{n}g}") if np.isfinite(v) else None for v in np.asarray(arr, dtype=float)]


def _explorer_payload(run, an, max_pts=500, k_random=8):
    idx = _plot_idx(run.n, max_pts)
    flagged, rnd = _pick_walkers(an, k_random)
    walkers = [int(i) for i in flagged] + [int(i) for i in rnd]
    series = {}
    for nm in run.names:
        col = run.columns[nm]
        x = col[idx][:, _keep_mask(an)].astype(np.float64)
        q = _nanq(x, [0.05, 0.16, 0.5, 0.84, 0.95], axis=1)
        series[nm] = {"q": [_sig(r) for r in q], "w": [_sig(col[idx, i]) for i in walkers]}
    return {"steps": [int(v) for v in run.steps[idx]], "burn": int(an["est_step"]), "walkers": walkers,
            "n_flagged": len(flagged), "series": series}


_EXPLORER_JS = r"""
(function(){
  const data = JSON.parse(document.getElementById('ex-data').textContent);
  const sel = document.getElementById('ex-param'), cv = document.getElementById('ex-canvas'), tip = document.getElementById('ex-tip');
  const bands = document.getElementById('ex-bands'), wk = document.getElementById('ex-walkers'),
        fl = document.getElementById('ex-flagged'), all = document.getElementById('ex-all');
  const ctx = cv.getContext('2d');
  const names = Object.keys(data.series);
  names.forEach(function(n){ const o = document.createElement('option'); o.value = n; o.textContent = n; sel.appendChild(o); });
  const H = 380, M = {l: 70, r: 14, t: 12, b: 34};
  let W = 600, view = null;
  function nice(lo, hi, n){
    const span = (hi - lo) || 1, raw = span / n, p = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / p;
    const step = (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * p, out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9 * span; v += step) out.push(v);
    return out;
  }
  function fmt(v){ return Math.abs(v) >= 1e5 ? v.toPrecision(10).replace(/\.?0+$/, '') : (+v.toPrecision(4)).toString(); }
  function resize(){
    const r = cv.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    W = r.width || 800;
    cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw();
  }
  function draw(){
    const s = data.series[sel.value], steps = data.steps, n = steps.length;
    let yi = 0;
    if (!all.checked) { while (yi < n - 1 && steps[yi] < data.burn) yi++; }
    const i0 = 0;
    let lo = Infinity, hi = -Infinity;
    function upd(a){ for (let i = yi; i < n; i++) { const v = a[i]; if (v !== null && v < lo) lo = v; if (v !== null && v > hi) hi = v; } }
    upd(s.q[0]); upd(s.q[4]);
    s.w.forEach(function(a, k){ if (k >= data.n_flagged) upd(a); });
    if (!(hi > lo)) { lo -= 1; hi += 1; }
    const pad = 0.06 * (hi - lo); lo -= pad; hi += pad;
    const x0 = steps[i0], x1 = steps[n - 1];
    const X = function(v){ return M.l + (v - x0) / ((x1 - x0) || 1) * (W - M.l - M.r); };
    const Y = function(v){ return H - M.b - (v - lo) / (hi - lo) * (H - M.t - M.b); };
    view = {X: X, Y: Y, x0: x0, x1: x1, i0: i0};
    ctx.clearRect(0, 0, W, H);
    ctx.font = '11px system-ui, sans-serif'; ctx.fillStyle = '#555'; ctx.strokeStyle = '#e6e9ee'; ctx.lineWidth = 1;
    nice(lo, hi, 6).forEach(function(v){ const y = Y(v); ctx.beginPath(); ctx.moveTo(M.l, y); ctx.lineTo(W - M.r, y); ctx.stroke(); ctx.textAlign = 'right'; ctx.fillText(fmt(v), M.l - 6, y + 4); });
    nice(x0, x1, 8).forEach(function(v){ const x = X(v); ctx.beginPath(); ctx.moveTo(x, M.t); ctx.lineTo(x, H - M.b); ctx.stroke(); ctx.textAlign = 'center'; ctx.fillText(fmt(v), x, H - M.b + 16); });
    ctx.save(); ctx.beginPath(); ctx.rect(M.l, M.t, W - M.l - M.r, H - M.t - M.b); ctx.clip();
    function band(a, b, color){
      ctx.beginPath(); let started = false;
      for (let i = i0; i < n; i++) { if (a[i] === null) continue; const x = X(steps[i]), y = Y(a[i]); if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y); }
      for (let i = n - 1; i >= i0; i--) { if (b[i] === null) continue; ctx.lineTo(X(steps[i]), Y(b[i])); }
      ctx.closePath(); ctx.fillStyle = color; ctx.fill();
    }
    function line(a, color, lw){
      ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = lw; let started = false;
      for (let i = i0; i < n; i++) { if (a[i] === null) { started = false; continue; } const x = X(steps[i]), y = Y(a[i]); if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y); }
      ctx.stroke();
    }
    if (bands.checked) { band(s.q[0], s.q[4], 'rgba(169,193,222,0.55)'); band(s.q[1], s.q[3], 'rgba(59,110,165,0.35)'); }
    s.w.forEach(function(a, k){
      const isFlag = k < data.n_flagged;
      if (isFlag ? fl.checked : wk.checked) line(a, isFlag ? 'rgba(200,69,63,0.95)' : 'rgba(110,115,125,0.55)', isFlag ? 1.2 : 0.8);
    });
    line(s.q[2], '#222831', 1.6);
    if (data.burn > x0 && data.burn < x1) { ctx.setLineDash([5, 4]); ctx.strokeStyle = '#7b5ea7'; ctx.lineWidth = 1.4; ctx.beginPath(); ctx.moveTo(X(data.burn), M.t); ctx.lineTo(X(data.burn), H - M.b); ctx.stroke(); ctx.setLineDash([]); }
    ctx.restore();
    ctx.fillStyle = '#555'; ctx.textAlign = 'center'; ctx.fillText('step', (W + M.l) / 2, H - 4);
  }
  cv.addEventListener('mousemove', function(e){
    if (!view) return;
    const r = cv.getBoundingClientRect(), mx = e.clientX - r.left;
    if (mx < M.l || mx > W - M.r) { tip.style.display = 'none'; return; }
    const step = view.x0 + (mx - M.l) / (W - M.l - M.r) * (view.x1 - view.x0);
    let best = view.i0, bd = Infinity;
    data.steps.forEach(function(v, i){ if (i >= view.i0) { const d = Math.abs(v - step); if (d < bd) { bd = d; best = i; } } });
    const s = data.series[sel.value].q;
    tip.style.display = 'block'; tip.style.left = Math.min(mx + 12, W - 190) + 'px';
    tip.innerHTML = 'step ' + data.steps[best] + '<br>median ' + (s[2][best] === null ? 'n/a' : fmt(s[2][best])) + '<br>68%: ' + (s[1][best] === null ? 'n/a' : fmt(s[1][best]) + ' to ' + fmt(s[3][best]));
  });
  cv.addEventListener('mouseleave', function(){ tip.style.display = 'none'; });
  [sel, bands, wk, fl, all].forEach(function(el){ el.addEventListener('change', draw); });
  window.addEventListener('resize', resize);
  resize();
})();
"""

_SORT_JS = r"""
document.querySelectorAll('table.sortable').forEach(function(t){
  t.querySelectorAll('th').forEach(function(th, ci){
    th.addEventListener('click', function(){
      const body = t.tBodies[0], rows = Array.from(body.rows), asc = th.dataset.asc !== '1';
      rows.sort(function(a, b){
        const x = a.cells[ci].dataset.v || a.cells[ci].textContent, y = b.cells[ci].dataset.v || b.cells[ci].textContent;
        const nx = parseFloat(x), ny = parseFloat(y);
        const c = (!isNaN(nx) && !isNaN(ny)) ? nx - ny : x.localeCompare(y);
        return asc ? c : -c;
      });
      rows.forEach(function(r){ body.appendChild(r); });
      t.querySelectorAll('th').forEach(function(h){ delete h.dataset.asc; });
      th.dataset.asc = asc ? '1' : '0';
    });
  });
});
"""

_CSS = """
:root{--bg:#f6f7f9;--card:#fff;--ink:#222831;--muted:#6b7280;--line:#e6e9ee;--blue:#3b6ea5;--ok:#4c9a6a;--warn:#d9a22b;--bad:#c8453f;--na:#8a8f98}
@media (prefers-color-scheme: dark){:root{--bg:#14171c;--card:#1d2128;--ink:#e6e8ec;--muted:#9aa1ad;--line:#2b313b}}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header{background:linear-gradient(120deg,#26466d,#3b6ea5);color:#fff;padding:26px 32px}
header h1{margin:0 0 4px;font-size:24px} header p{margin:0;opacity:.85;font-size:14px}
.layout{display:flex;align-items:flex-start;max-width:1380px;margin:0 auto}
nav{position:sticky;top:0;min-width:200px;padding:22px 16px;align-self:flex-start}
nav a{display:block;color:var(--muted);text-decoration:none;padding:6px 10px;border-radius:8px;font-size:14px}
nav a:hover{background:var(--line);color:var(--ink)}
main{flex:1;min-width:0;padding:22px 24px 60px}
section{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:22px 26px;margin-bottom:22px}
section h2{margin:0 0 4px;font-size:19px} section h3{margin:22px 0 6px;font-size:15px}
.muted{color:var(--muted);font-size:13.5px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:14px 0}
.card{border:1px solid var(--line);border-radius:12px;padding:12px 14px}
.card .k{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em} .card .v{font-size:22px;font-weight:650}
.badge{display:inline-block;padding:2px 10px;border-radius:999px;font-size:12px;font-weight:650;color:#fff}
.b-ok{background:var(--ok)} .b-warn{background:var(--warn)} .b-bad{background:var(--bad)} .b-na{background:var(--na)}
.dot{display:inline-block;width:13px;height:13px;border-radius:50%;vertical-align:middle}
.legend{display:flex;gap:20px;flex-wrap:wrap;font-size:13px;color:var(--muted);margin:10px 0 6px} .legend span{display:inline-flex;gap:7px;align-items:center}
.verdict{display:flex;align-items:center;gap:12px;padding:14px 16px;border-radius:12px;margin:12px 0;border:1px solid var(--line);border-left:6px solid var(--na)}
.verdict.ok{border-left-color:var(--ok)} .verdict.warn{border-left-color:var(--warn)} .verdict.bad{border-left-color:var(--bad)}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{padding:6px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left} th{position:sticky;top:0;background:var(--card);cursor:pointer;font-weight:650}
table.sortable th:hover{color:var(--blue)} tr:hover td{background:rgba(59,110,165,.06)}
td.ok{color:var(--ok);font-weight:600} td.warn{color:var(--warn);font-weight:600} td.bad{color:var(--bad);font-weight:650}
.scroll{max-height:520px;overflow:auto;border:1px solid var(--line);border-radius:10px}
img{max-width:100%;height:auto;background:#fff;border-radius:8px;border:1px solid var(--line);margin:6px 0}
.explorer{border:1px solid var(--line);border-radius:12px;padding:12px 14px;position:relative}
.explorer .controls{display:flex;flex-wrap:wrap;gap:14px;align-items:center;margin-bottom:8px;font-size:13.5px}
#ex-canvas,#lp-canvas,#pd-hist,#pd-cat,#hh-canvas,#fw-canvas{width:100%;display:block;background:#fff;border-radius:8px} #fw-canvas{height:360px}
.controls button{padding:3px 12px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--ink);cursor:pointer} #ex-canvas{height:380px} #lp-canvas{height:430px} #pd-hist,#pd-cat{height:380px} #hh-canvas{height:340px}
#hh-stats table{width:auto;font-size:12.5px;margin-top:8px} #hh-stats th{cursor:default}
table.args{table-layout:fixed;width:100%} table.args th,table.args td{white-space:normal;text-align:left;vertical-align:top;overflow-wrap:anywhere;line-height:1.4}
table.args td:first-child{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12.5px}
pre.cmd{white-space:pre-wrap;overflow-wrap:anywhere;background:var(--line);padding:10px 12px;border-radius:8px;font-size:12.5px;margin:8px 0}
table.checks{table-layout:fixed} table.checks th,table.checks td{white-space:normal;text-align:left;vertical-align:top;overflow-wrap:anywhere;line-height:1.4}
@media (max-width:1150px){table.checks colgroup,table.checks thead{display:none} table.checks,table.checks tbody,table.checks tr,table.checks td{display:block;width:100%} table.checks tr{border-bottom:1px solid var(--line);padding:8px 0} table.checks td{border:0;padding:2px 0} table.checks td[data-l]::before{content:attr(data-l) \": \";color:var(--muted);font-size:12px}}
.pair{display:grid;grid-template-columns:1.15fr 1fr;gap:12px}
.tip{display:none;position:absolute;top:70px;background:rgba(34,40,49,.92);color:#fff;font-size:12px;padding:6px 9px;border-radius:6px;pointer-events:none;line-height:1.4}
details{margin:8px 0} summary{cursor:pointer;font-weight:600}
code{background:var(--line);padding:1px 5px;border-radius:4px;font-size:13px}
footer{color:var(--muted);font-size:12.5px;text-align:center;padding:10px 0 30px}
@media (max-width:900px){nav{display:none}}
"""


def _cell(text, cls="", val=None):
    v = f' data-v="{val}"' if val is not None else ""
    return f'<td class="{cls}"{v}>{_html.escape(str(text))}</td>'


def _params_table(an):
    head = ["parameter", "mean", "std", "5%", "median", "95%", "R-hat", "R-hat (clean)", "tau", "N/tau", "ESS", "drift z", "status"]
    out = ['<div class="scroll"><table class="sortable"><thead><tr>' + "".join(f"<th>{h}</th>" for h in head) + "</tr></thead><tbody>"]
    for r in an["params"]:
        nm = r["name"] + (" (derived)" if r["derived"] else "")
        tau_txt = _fmt(r["tau"]) + ("" if r["tau_resolved"] else " *")
        out.append("<tr>" + _cell(nm) + _cell(_fmt(r["mean"]), val=r["mean"]) + _cell(_fmt(r["std"]), val=r["std"]) +
                   _cell(_fmt(r["q05"]), val=r["q05"]) + _cell(_fmt(r["q50"]), val=r["q50"]) + _cell(_fmt(r["q95"]), val=r["q95"]) +
                   _cell(_fmt(r["rhat"]), r["st_rhat"], r["rhat"]) + _cell(_fmt(r["rhat_clean"]), val=r["rhat_clean"]) +
                   _cell(tau_txt, val=r["tau"]) + _cell(_fmt(r["n_over_tau"]), r["st_tau"], r["n_over_tau"]) +
                   _cell(_fmt(r["ess"]), r["st_ess"], r["ess"]) + _cell(_fmt(r["drift_z"]), r["st_drift"], r["drift_z"]) +
                   _cell(r["status"].upper(), r["status"]) + "</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


def _walkers_table(an, limit=600):
    rows = sorted(an["walkers"], key=lambda r: (r["flags"] == "", r["gap_sigma"]))[:limit]
    head = ["walker", "flags", "median ln p", "offset from middle (spreads)", "16%", "84%", "max ln p", "moved (share of recent steps)", "acceptance", "longest unchanged stretch (steps)", "non-finite"]
    out = ['<div class="scroll"><table class="sortable"><thead><tr>' + "".join(f"<th>{h}</th>" for h in head) + "</tr></thead><tbody>"]
    for r in rows:
        cls = "bad" if r["flags"] else ""
        out.append("<tr>" + _cell(r["id"], val=r["id"]) + _cell(r["flags"] or "-", cls) + _cell(_fmt(r["median_lnp"]), val=r["median_lnp"]) +
                   _cell(f"{r['gap_sigma']:+.2f}", val=r["gap_sigma"]) + _cell(_fmt(r["q16"]), val=r["q16"]) + _cell(_fmt(r["q84"]), val=r["q84"]) +
                   _cell(_fmt(r["max_lnp"]), val=r["max_lnp"]) + _cell(f"{r['moved']:.1%}", val=r["moved"]) +
                   _cell(f"{r['accept']:.2f}", val=r["accept"]) + _cell(r["longest_frozen"], val=r["longest_frozen"]) +
                   _cell(f"{r['nonfinite']:.1%}", val=r["nonfinite"]) + "</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


def _lnp_payload(run, an):
    """Every walker's ln p, downsampled to a size that keeps the page light (about 400k numbers)."""
    w = run.nwalkers
    n_pts = int(min(1000, max(200, 4e5 // w)))
    idx = _plot_idx(run.n, n_pts)
    lp = run.lnp[idx].astype(np.float64)
    band = _nanq(lp[:, _keep_mask(an)], [0.16, 0.84], axis=1)
    return {"steps": [int(v) for v in run.steps[idx]], "walkers": [_sig(lp[:, i], 6) for i in range(w)],
            "flagged": [int(i) for i in an["flagged"]], "burn": int(an["est_step"]),
            "median": _sig(np.nanmedian(lp, axis=1), 6), "q16": _sig(band[0], 6), "q84": _sig(band[1], 6),
            "info": {str(i): [an["walkers"][i]["flags"], float(an["walkers"][i]["median_lnp"] - an["ens_median_lnp"]),
                              float(an["walkers"][i]["moved"]), int(an["walkers"][i]["longest_frozen"])] for i in an["flagged"]},
            "tail_steps": int(an["tail_steps"])}


_LNP_JS = r"""
(function(){
  const D = JSON.parse(document.getElementById('lp-data').textContent);
  const el = function(id){ return document.getElementById(id); };
  const cv = el('lp-canvas'), tip = el('lp-tip'), ctx = cv.getContext('2d');
  const cbU = el('lp-unflagged'), cbF = el('lp-flagged'), cbM = el('lp-median'), cbB = el('lp-band'),
        selX = el('lp-xscale'), selY = el('lp-yrange'), op = el('lp-opacity'), hl = el('lp-highlight');
  const steps = D.steps, n = steps.length, nw = D.walkers.length;
  const isFlag = []; for (let w = 0; w < nw; w++) isFlag.push(false);
  D.flagged.forEach(function(i){ isFlag[i] = true; });
  let i0 = 0; while (i0 < n - 1 && steps[i0] < D.burn) i0++;
  const H = 430, M = {l: 72, r: 14, t: 12, b: 38};
  let W = 800, view = null;
  const cache = {};
  function fmt(v){ return Math.abs(v) >= 1e5 ? v.toPrecision(10).replace(/\.?0+$/, '') : (+v.toPrecision(5)).toString(); }
  function nice(lo, hi, k){
    const span = (hi - lo) || 1, raw = span / k, p = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / p;
    const step = (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * p, out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9 * span; v += step) out.push(v);
    return out;
  }
  function robustRange(wantU, wantF, from){
    const key = (wantU ? 'u' : '') + (wantF ? 'f' : '') + from;
    if (cache[key]) return cache[key];
    const vals = [], stride = Math.max(1, Math.floor((n - from) * nw / 60000)); let c = 0;
    for (let w = 0; w < nw; w++) {
      if (isFlag[w] ? !wantF : !wantU) continue;
      const a = D.walkers[w];
      for (let i = from; i < n; i++) { if (a[i] !== null) { if ((c++ % stride) === 0) vals.push(a[i]); } }
    }
    vals.sort(function(x, y){ return x - y; });
    let r = [Infinity, -Infinity];
    if (vals.length) {
      const q = function(p){ return vals[Math.min(vals.length - 1, Math.max(0, Math.round(p * (vals.length - 1))))]; };
      const med = q(0.5), sd = Math.max((q(0.75) - q(0.25)) / 1.349, 1e-12);
      r = [Math.max(q(0.005), med - 10 * sd), Math.min(q(0.995), med + 8 * sd)];
    }
    cache[key] = r; return r;
  }
  function resize(){
    const r = cv.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    W = r.width || 800;
    cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw();
  }
  function draw(){
    const wantU = cbU.checked, wantF = cbF.checked, yfrom = (selY.value === 'all') ? 0 : i0;
    let from = 0;
    if (selY.value !== 'all') {
      const xstart = Math.max(steps[0], D.burn - 0.1 * (steps[n - 1] - D.burn));
      while (from < n - 1 && steps[from] < xstart) from++;
    }
    let lo = Infinity, hi = -Infinity;
    if (wantU || wantF) { const r = robustRange(wantU, wantF, yfrom); lo = r[0]; hi = r[1]; }
    for (let i = yfrom; i < n; i++) { const v = D.median[i]; if (v !== null) { if (v < lo) lo = v; if (v > hi) hi = v; } }
    if (!(hi > lo)) { lo = (isFinite(lo) ? lo : 0) - 1; hi = (isFinite(hi) ? hi : 0) + 1; }
    const pad = 0.06 * (hi - lo); lo -= pad; hi += pad;
    const logx = selX.value === 'log';
    const xf = function(s){ return logx ? Math.log10(s + 1) : s; };
    const xmin = xf(steps[from]), xmax = xf(steps[n - 1]);
    const X = function(s){ return M.l + (xf(s) - xmin) / ((xmax - xmin) || 1) * (W - M.l - M.r); };
    const Y = function(v){ return H - M.b - (v - lo) / (hi - lo) * (H - M.t - M.b); };
    view = {X: X, Y: Y, from: from};
    ctx.clearRect(0, 0, W, H);
    ctx.font = '11px system-ui, sans-serif'; ctx.fillStyle = '#555'; ctx.strokeStyle = '#e6e9ee'; ctx.lineWidth = 1;
    nice(lo, hi, 7).forEach(function(v){ const y = Y(v); ctx.beginPath(); ctx.moveTo(M.l, y); ctx.lineTo(W - M.r, y); ctx.stroke(); ctx.textAlign = 'right'; ctx.fillText(fmt(v), M.l - 6, y + 4); });
    if (logx) {
      const e0 = Math.floor(xmin), e1 = Math.ceil(xmax);
      for (let e = e0; e <= e1; e++) { [1, 2, 5].forEach(function(m){ const s = m * Math.pow(10, e) - 1; if (s < steps[from] || s > steps[n - 1]) return; const x = X(s); ctx.beginPath(); ctx.moveTo(x, M.t); ctx.lineTo(x, H - M.b); ctx.stroke(); ctx.textAlign = 'center'; ctx.fillText(fmt(s + 1), x, H - M.b + 16); }); }
    } else {
      nice(steps[from], steps[n - 1], 8).forEach(function(v){ const x = X(v); ctx.beginPath(); ctx.moveTo(x, M.t); ctx.lineTo(x, H - M.b); ctx.stroke(); ctx.textAlign = 'center'; ctx.fillText(fmt(v), x, H - M.b + 16); });
    }
    ctx.save(); ctx.beginPath(); ctx.rect(M.l, M.t, W - M.l - M.r, H - M.t - M.b); ctx.clip();
    function line(a, color, lw){
      ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = lw; let started = false;
      for (let i = from; i < n; i++) { const v = a[i]; if (v === null) { started = false; continue; } const x = X(steps[i]), y = Y(v); if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y); }
      ctx.stroke();
    }
    if (cbB.checked) {
      ctx.beginPath(); let st = false;
      for (let i = from; i < n; i++) { if (D.q16[i] === null) continue; const x = X(steps[i]), y = Y(D.q16[i]); if (!st) { ctx.moveTo(x, y); st = true; } else ctx.lineTo(x, y); }
      for (let i = n - 1; i >= from; i--) { if (D.q84[i] === null) continue; ctx.lineTo(X(steps[i]), Y(D.q84[i])); }
      ctx.closePath(); ctx.fillStyle = 'rgba(59,110,165,0.25)'; ctx.fill();
    }
    const al = parseFloat(op.value) || 0.35;
    if (wantU) { const c = 'rgba(80,100,135,' + al + ')'; for (let w = 0; w < nw; w++) if (!isFlag[w]) line(D.walkers[w], c, 0.8); }
    if (wantF) { const c = 'rgba(200,69,63,' + Math.min(1, al * 1.8) + ')'; for (let w = 0; w < nw; w++) if (isFlag[w]) line(D.walkers[w], c, 1.0); }
    if (cbM.checked) line(D.median, '#222831', 1.9);
    const h = parseInt(hl.value, 10);
    if (!isNaN(h) && h >= 0 && h < nw) line(D.walkers[h], '#e07b39', 2.4);
    if (D.burn > steps[from] && D.burn < steps[n - 1]) { ctx.setLineDash([5, 4]); ctx.strokeStyle = '#7b5ea7'; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(X(D.burn), M.t); ctx.lineTo(X(D.burn), H - M.b); ctx.stroke(); ctx.setLineDash([]); ctx.fillStyle = '#7b5ea7'; ctx.textAlign = 'left'; ctx.fillText('estimated burn-in', X(D.burn) + 5, M.t + 13); }
    ctx.restore();
    ctx.fillStyle = '#555'; ctx.textAlign = 'center'; ctx.fillText('step' + (logx ? ' (log scale)' : ''), (W + M.l) / 2, H - 4);
    ctx.save(); ctx.translate(14, (H - M.b + M.t) / 2); ctx.rotate(-Math.PI / 2); ctx.fillText('ln posterior', 0, 0); ctx.restore();
  }
  function nearest(e){
    if (!view) return null;
    const r = cv.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
    if (mx < M.l || mx > W - M.r) return null;
    let best = view.from, bd = Infinity;
    for (let i = view.from; i < n; i++) { const d = Math.abs(view.X(steps[i]) - mx); if (d < bd) { bd = d; best = i; } }
    let bw = -1, bdy = 14;
    for (let w = 0; w < nw; w++) {
      if (isFlag[w] ? !cbF.checked : !cbU.checked) continue;
      const v = D.walkers[w][best]; if (v === null) continue;
      const dy = Math.abs(view.Y(v) - my); if (dy < bdy) { bdy = dy; bw = w; }
    }
    return bw < 0 ? null : {w: bw, i: best, mx: mx, my: my};
  }
  cv.addEventListener('mousemove', function(e){
    const h = nearest(e);
    if (!h) { tip.style.display = 'none'; return; }
    tip.style.display = 'block'; tip.style.left = (cv.offsetLeft + Math.min(h.mx + 12, W - 190)) + 'px'; tip.style.top = (cv.offsetTop + h.my + 12) + 'px';
    tip.innerHTML = 'walker ' + h.w + (isFlag[h.w] ? ' (flagged)' : '') + '<br>step ' + steps[h.i] + '<br>ln p ' + fmt(D.walkers[h.w][h.i]);
  });
  cv.addEventListener('click', function(e){ const h = nearest(e); if (h) { hl.value = h.w; draw(); } });
  cv.addEventListener('mouseleave', function(){ tip.style.display = 'none'; });
  [cbU, cbF, cbM, cbB, selX, selY, op, hl].forEach(function(c){ c.addEventListener('change', draw); c.addEventListener('input', draw); });
  window.addEventListener('resize', resize);
  resize();
})();
"""


def _lnp_explorer_html(run, an):
    payload = json.dumps(_lnp_payload(run, an), separators=(",", ":")).replace("</", "<\\/")
    nf = len(an["flagged"])
    band = "68% band of all walkers" if an["ensemble_wide"] else "68% band of unflagged walkers"
    return ('<div class="explorer"><div class="controls">'
            f'<label><input type="checkbox" id="lp-unflagged" checked> unflagged walkers ({an["w"] - nf})</label>'
            f'<label><input type="checkbox" id="lp-flagged" checked> flagged walkers ({nf})</label>'
            '<label><input type="checkbox" id="lp-median" checked> median over all walkers</label>'
            f'<label><input type="checkbox" id="lp-band"> {band}</label>'
            '<label>x axis <select id="lp-xscale"><option value="linear">linear</option><option value="log">log</option></select></label>'
            '<label>view <select id="lp-yrange"><option value="post">from just before the estimated burn-in</option><option value="all">whole run</option></select></label>'
            '<label>opacity <input type="range" id="lp-opacity" min="0.05" max="1" step="0.05" value="0.35"></label>'
            '<label>highlight walker <input type="number" id="lp-highlight" min="0" style="width:72px"></label></div>'
            '<canvas id="lp-canvas" height="430"></canvas><div id="lp-tip" class="tip"></div></div>'
            f'<script id="lp-data" type="application/json">{payload}</script><script>{_LNP_JS}</script>')


def _dist_payload(run, an, nbins=60):
    """Per-walker ln p histograms (after the estimated burn-in) on two bin sets, plus per-walker median and 16-84% range."""
    w = run.nwalkers
    ref = an["ens_median_lnp"]
    rel = run.lnp[an["view_idx"]:].astype(np.float64) - ref
    keep = _keep_mask(an)
    lo, hi = _plateau_ylim(_clean_post(run, an) - ref, pad=0.05)
    fin = rel[np.isfinite(rel)]
    flo, fhi = (float(fin.min()), float(fin.max())) if fin.size else (-1.0, 1.0)
    if fhi <= flo:
        flo, fhi = flo - 1, fhi + 1
    out = {"w": w, "flagged": [int(i) for i in an["flagged"]], "ref": float(ref),
           "pooled_label": "all walkers" if an["ensemble_wide"] else "unflagged walkers"}
    for key, (a, b) in (("zoom", (lo, hi)), ("full", (flo, fhi))):
        edges = np.linspace(a, b, nbins + 1)
        hist = [_sig(_density(rel[:, i][np.isfinite(rel[:, i])], edges), 4) for i in range(w)]
        pooled = rel[:, keep].ravel()
        pooled = pooled[np.isfinite(pooled)]
        out[key] = {"edges": _sig(edges, 6), "h": hist, "pooled": _sig(_density(pooled, edges), 4)}
    med = np.array([r["median_lnp"] for r in an["walkers"]]) - ref
    q16 = np.array([r["q16"] for r in an["walkers"]]) - ref
    q84 = np.array([r["q84"] for r in an["walkers"]]) - ref
    zlo, zhi = _robust_ylim(np.r_[q16[keep], q84[keep]], 0.5, 99.5, pad=0.15)
    allv = np.r_[q16, q84, med]
    allv = allv[np.isfinite(allv)]
    pad = 0.05 * (allv.max() - allv.min() + 1e-9)
    out.update(median=_sig(med, 5), q16=_sig(q16, 5), q84=_sig(q84, 5), cat_zoom=[float(zlo), float(zhi)],
               cat_full=[float(allv.min() - pad), float(allv.max() + pad)])
    return out


_DIST_JS = r"""
(function(){
  const D = JSON.parse(document.getElementById('pd-data').textContent);
  const el = function(id){ return document.getElementById(id); };
  const cvH = el('pd-hist'), cvC = el('pd-cat'), tip = el('pd-tip');
  const ctxH = cvH.getContext('2d'), ctxC = cvC.getContext('2d');
  const cbU = el('pd-unflagged'), cbF = el('pd-flagged'), cbP = el('pd-pooled'), selR = el('pd-range'), op = el('pd-opacity'), hl = el('pd-highlight');
  const nw = D.w, isFlag = [], order = [], rankOf = [];
  for (let i = 0; i < nw; i++) { isFlag.push(false); order.push(i); rankOf.push(0); }
  D.flagged.forEach(function(i){ isFlag[i] = true; });
  order.sort(function(a, b){ return D.median[a] - D.median[b]; });
  order.forEach(function(w, r){ rankOf[w] = r; });
  const H = 380, M = {l: 62, r: 12, t: 12, b: 40};
  const wid = {h: 600, c: 520};
  let vH = null, vC = null;
  function fmt(v){ return Math.abs(v) >= 1e5 ? v.toPrecision(10).replace(/\.?0+$/, '') : (+v.toPrecision(4)).toString(); }
  function nice(lo, hi, k){
    const span = (hi - lo) || 1, raw = span / k, p = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / p;
    const step = (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * p, out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9 * span; v += step) out.push(v);
    return out;
  }
  function size(cv, ctx, key){
    const r = cv.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    wid[key] = r.width || 500;
    cv.width = Math.round(wid[key] * dpr); cv.height = Math.round(H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  function cur(){ return selR.value === 'full' ? D.full : D.zoom; }
  function axes(ctx, W, xlo, xhi, ylo, yhi, X, Y, xlabel, ylabel, xticks){
    ctx.clearRect(0, 0, W, H);
    ctx.font = '11px system-ui, sans-serif'; ctx.fillStyle = '#555'; ctx.strokeStyle = '#e6e9ee'; ctx.lineWidth = 1;
    nice(ylo, yhi, 6).forEach(function(v){ const y = Y(v); ctx.beginPath(); ctx.moveTo(M.l, y); ctx.lineTo(W - M.r, y); ctx.stroke(); ctx.textAlign = 'right'; ctx.fillText(fmt(v), M.l - 6, y + 4); });
    if (xticks) nice(xlo, xhi, 7).forEach(function(v){ const x = X(v); ctx.beginPath(); ctx.moveTo(x, M.t); ctx.lineTo(x, H - M.b); ctx.stroke(); ctx.textAlign = 'center'; ctx.fillText(fmt(v), x, H - M.b + 16); });
    ctx.fillStyle = '#555'; ctx.textAlign = 'center'; ctx.fillText(xlabel, (W + M.l) / 2, H - 4);
    ctx.save(); ctx.translate(13, (H - M.b + M.t) / 2); ctx.rotate(-Math.PI / 2); ctx.fillText(ylabel, 0, 0); ctx.restore();
  }
  function stairs(ctx, e, h, X, Y, color, lw){
    ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = lw;
    for (let k = 0; k < h.length; k++) { const y = Y(h[k] === null ? 0 : h[k]); if (k === 0) ctx.moveTo(X(e[k]), y); else ctx.lineTo(X(e[k]), y); ctx.lineTo(X(e[k + 1]), y); }
    ctx.stroke();
  }
  function drawHist(){
    const S = cur(), e = S.edges, nb = e.length - 1, W = wid.h, ctx = ctxH;
    let top = 0; S.pooled.forEach(function(v){ if (v !== null && v > top) top = v; });
    if (!(top > 0)) S.h.forEach(function(a){ a.forEach(function(v){ if (v !== null && v > top) top = v; }); });
    top = (top > 0 ? top : 1) * 1.4;
    const X = function(v){ return M.l + (v - e[0]) / ((e[nb] - e[0]) || 1) * (W - M.l - M.r); };
    const Y = function(v){ return H - M.b - v / top * (H - M.t - M.b); };
    vH = {X: X, Y: Y, e: e, S: S};
    axes(ctx, W, e[0], e[nb], 0, top, X, Y, 'ln p - ensemble median', 'density', true);
    ctx.save(); ctx.beginPath(); ctx.rect(M.l, M.t, W - M.l - M.r, H - M.t - M.b); ctx.clip();
    const al = parseFloat(op.value) || 0.3;
    if (cbU.checked) { const c = 'rgba(80,100,135,' + al + ')'; for (let w = 0; w < nw; w++) if (!isFlag[w]) stairs(ctx, e, S.h[w], X, Y, c, 0.9); }
    if (cbF.checked) { const c = 'rgba(200,69,63,' + Math.min(1, al * 1.8) + ')'; for (let w = 0; w < nw; w++) if (isFlag[w]) stairs(ctx, e, S.h[w], X, Y, c, 1.0); }
    if (cbP.checked) stairs(ctx, e, S.pooled, X, Y, '#222831', 2.2);
    const h = parseInt(hl.value, 10);
    if (!isNaN(h) && h >= 0 && h < nw) stairs(ctx, e, S.h[h], X, Y, '#e07b39', 2.6);
    ctx.restore();
  }
  function drawCat(){
    const W = wid.c, ctx = ctxC, lim = (selR.value === 'full') ? D.cat_full : D.cat_zoom, lo = lim[0], hi = lim[1];
    const X = function(r){ return M.l + (r + 0.5) / nw * (W - M.l - M.r); };
    const Y = function(v){ return H - M.b - (v - lo) / (hi - lo) * (H - M.t - M.b); };
    const yc = function(v){ return Math.max(lo, Math.min(hi, v)); };
    vC = {X: X, Y: Y};
    axes(ctx, W, 0, nw, lo, hi, X, Y, 'walker (sorted by median)', 'ln p - ensemble median', false);
    ctx.strokeStyle = '#222831'; ctx.lineWidth = 1; ctx.beginPath(); if (lo < 0 && hi > 0) { ctx.moveTo(M.l, Y(0)); ctx.lineTo(W - M.r, Y(0)); } ctx.stroke();
    const al = parseFloat(op.value) || 0.3;
    function one(w, col, bar, lw, rad){
      const r = rankOf[w], x = X(r), m = D.median[w];
      if (bar !== null) { ctx.beginPath(); ctx.strokeStyle = bar; ctx.lineWidth = lw; ctx.moveTo(x, Y(yc(D.q16[w]))); ctx.lineTo(x, Y(yc(D.q84[w]))); ctx.stroke(); }
      ctx.fillStyle = col; ctx.beginPath();
      if (m < lo || m > hi) { const y = m < lo ? H - M.b - 4 : M.t + 4; ctx.moveTo(x - 4, m < lo ? y - 6 : y + 6); ctx.lineTo(x + 4, m < lo ? y - 6 : y + 6); ctx.lineTo(x, y); }
      else ctx.arc(x, Y(m), rad, 0, 6.2832);
      ctx.fill();
    }
    for (let w = 0; w < nw; w++) if (!isFlag[w] && cbU.checked) one(w, 'rgba(59,110,165,0.9)', 'rgba(80,100,135,' + al + ')', 1, 2.4);
    for (let w = 0; w < nw; w++) if (isFlag[w] && cbF.checked) one(w, 'rgba(200,69,63,0.95)', 'rgba(200,69,63,' + Math.min(1, al * 1.8) + ')', 1, 2.6);
    const h = parseInt(hl.value, 10);
    if (!isNaN(h) && h >= 0 && h < nw) one(h, '#e07b39', '#e07b39', 2.4, 4);
  }
  function draw(){ drawHist(); drawCat(); }
  function resize(){ size(cvH, ctxH, 'h'); size(cvC, ctxC, 'c'); draw(); }
  function show(e, cv, html, mx, my){ tip.style.display = 'block'; tip.style.left = (cv.offsetLeft + Math.min(mx + 12, cv.getBoundingClientRect().width - 190)) + 'px'; tip.style.top = (cv.offsetTop + my + 12) + 'px'; tip.innerHTML = html; }
  function histHit(e){
    if (!vH) return null;
    const r = cvH.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top, W = wid.h, ed = vH.e, nb = ed.length - 1;
    if (mx < M.l || mx > W - M.r) return null;
    const v = ed[0] + (mx - M.l) / (W - M.l - M.r) * (ed[nb] - ed[0]);
    let k = 0; while (k < nb - 1 && ed[k + 1] <= v) k++;
    let bw = -1, bd = 14;
    for (let w = 0; w < nw; w++) { if (isFlag[w] ? !cbF.checked : !cbU.checked) continue; const y = vH.S.h[w][k]; if (y === null) continue; const d = Math.abs(vH.Y(y) - my); if (d < bd) { bd = d; bw = w; } }
    return bw < 0 ? null : {w: bw, k: k, mx: mx, my: my};
  }
  function catHit(e){
    if (!vC) return null;
    const r = cvC.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top, W = wid.c;
    if (mx < M.l || mx > W - M.r) return null;
    const rk = Math.round((mx - M.l) / (W - M.l - M.r) * nw - 0.5);
    if (rk < 0 || rk >= nw) return null;
    const w = order[rk];
    if (isFlag[w] ? !cbF.checked : !cbU.checked) return null;
    return {w: w, mx: mx, my: my};
  }
  cvH.addEventListener('mousemove', function(e){
    const h = histHit(e); if (!h) { tip.style.display = 'none'; return; }
    show(e, cvH, 'walker ' + h.w + (isFlag[h.w] ? ' (flagged)' : '') + '<br>median ' + fmt(D.median[h.w]) + '<br>density ' + fmt(vH.S.h[h.w][h.k]), h.mx, h.my);
  });
  cvC.addEventListener('mousemove', function(e){
    const h = catHit(e); if (!h) { tip.style.display = 'none'; return; }
    show(e, cvC, 'walker ' + h.w + (isFlag[h.w] ? ' (flagged)' : '') + '<br>median ' + fmt(D.median[h.w]) + '<br>16-84%: ' + fmt(D.q16[h.w]) + ' to ' + fmt(D.q84[h.w]), h.mx, h.my);
  });
  cvH.addEventListener('click', function(e){ const h = histHit(e); if (h) { hl.value = h.w; draw(); } });
  cvC.addEventListener('click', function(e){ const h = catHit(e); if (h) { hl.value = h.w; draw(); } });
  [cvH, cvC].forEach(function(c){ c.addEventListener('mouseleave', function(){ tip.style.display = 'none'; }); });
  [cbU, cbF, cbP, selR, op, hl].forEach(function(c){ c.addEventListener('change', draw); c.addEventListener('input', draw); });
  window.addEventListener('resize', resize);
  resize();
})();
"""


def _dist_explorer_html(run, an):
    payload = json.dumps(_dist_payload(run, an), separators=(",", ":")).replace("</", "<\\/")
    nf = len(an["flagged"])
    pooled = "pooled distribution of all walkers" if an["ensemble_wide"] else "pooled distribution of unflagged walkers"
    return ('<div class="explorer"><div class="controls">'
            f'<label><input type="checkbox" id="pd-unflagged" checked> unflagged walkers ({an["w"] - nf})</label>'
            f'<label><input type="checkbox" id="pd-flagged" checked> flagged walkers ({nf})</label>'
            f'<label><input type="checkbox" id="pd-pooled" checked> {pooled}</label>'
            '<label>x range <select id="pd-range"><option value="zoom">plateau range</option><option value="full">all values</option></select></label>'
            '<label>opacity <input type="range" id="pd-opacity" min="0.05" max="1" step="0.05" value="0.3"></label>'
            '<label>highlight walker <input type="number" id="pd-highlight" min="0" style="width:72px"></label></div>'
            '<div class="pair"><canvas id="pd-hist"></canvas><canvas id="pd-cat"></canvas></div><div id="pd-tip" class="tip"></div></div>'
            f'<script id="pd-data" type="application/json">{payload}</script><script>{_DIST_JS}</script>')


def _stat_guide_html():
    """What R-hat, N/tau and drift z mean, with the values to expect."""
    rows = [("R-hat", "Do all walkers describe the same distribution? Compares the spread between walkers with the spread inside each walker; it is 1 when they agree.",
             "close to 1, below 1.01", "above 1.05"),
            ("N / tau", "How long the chain is in units of the autocorrelation time tau, the number of steps a walker needs to forget where it was. Roughly the number of independent samples per walker.",
             "50 or more", "below 20"),
            ("drift z", "Has the chain stopped moving? The difference between the averages of the first and second half of the chain, in standard errors.",
             "within about 3 (random scatter around 0)", "above 5")]
    out = ['<table><thead><tr><th>statistic</th><th style="text-align:left">what it tells you</th><th>good</th><th>worrying</th></tr></thead><tbody>']
    for name, what, good, bad in rows:
        out.append(f'<tr><td><b>{name}</b></td><td style="text-align:left;white-space:normal">{what}</td>'
                   f'<td class="ok">{good}</td><td class="bad">{bad}</td></tr>')
    out.append("</tbody></table>")
    return "".join(out)


def _flag_rules_html(an):
    """The criteria that flag a walker, in plain language, with the numbers used in this run."""
    fi = an["flag_info"]
    T, a = fi["tail_steps"], fi["typical_move"]
    if a <= 0:
        moves = (f"In the last {T} steps the typical walker did not move at all, so the rule about rare moves cannot single out any walker.")
    elif fi["moves_cutoff"] >= 0:
        moves = (f"In the last {T} steps the typical walker moved in {a:.0%} of the steps. A walker is frozen if it moved {fi['moves_cutoff']} times or fewer: "
                 f"less than half as often as the typical walker, and so rarely that luck would produce it for fewer than 1 walker in 100 across the whole ensemble.")
    else:
        moves = (f"In the last {T} steps the typical walker moved in {a:.0%} of the steps; with that many walkers, no walker could be unlucky enough to be called frozen from its move count alone.")
    return (f'<details open><summary>How a walker gets flagged (rules version {FLAG_RULES_VERSION}, fixed)</summary>'
            f'<p class="muted">A walker is flagged if it meets any one of the points below. Walkers are compared with each other on the part of the run after the estimated burn-in (from step {an["view_step"]:,} on).</p><ul>'
            f'<li><b>Frozen.</b> It hardly moves. {moves}' + ("" if a <= 0 else
            f' A walker is also frozen if it stayed exactly the same for {fi["run_threshold"]} steps in a row or more, '
            f'which is longer than chance allows at the typical rate (and never less than 100 steps).') + '</li>'
            f'<li><b>Low log-probability.</b> Its typical ln p sits well below the other walkers: more than {fi["low_units"]:.1f} ln p units under the middle walker. '
            f'That is {fi["lnp_z"]:g} times the usual walker-to-walker difference ({fi["between_sd"]:.2f}) and at least {fi["lnp_gap_sigma"]:g} times the ordinary ln p scatter ({fi["pooled_sd"]:.2f}).</li>'
            f'<li><b>Offset in a parameter.</b> In at least one of the sampled parameters its average is far from the other walkers: more than {fi["offset_z"]:g} times the usual walker-to-walker difference '
            f'in that parameter, and more than half of that parameter\'s overall spread. Only the sampled parameters are used for this test (not ln p and not the converted physical parameters), '
            f'so the number of flagged walkers does not depend on what else the report shows.</li>'
            '<li><b>Non-finite ln p.</b> More than half of its ln p values are infinite or not a number.</li></ul>'
            '<p class="muted">If more than 20% of the walkers are flagged, the report treats it as a problem of the whole ensemble and uses all walkers for the bands and axis limits.</p></details>')


def select_parameters(run, an, k=6, flag_impact=False):
    """
    Choose which parameters to show instead of showing all of them: ln_posterior first, then the k-1 most informative others.

    flag_impact=True (and some walkers flagged): the parameters in which the flagged walkers sit furthest from the rest.
    Otherwise: the highest concern score, the largest of (R-hat - 1) / 0.05, 50 / (N/tau) and |drift z| / 5
    (a score of 1 or more means at least one of the three statistics is at its limit).
    Returns (list of names, dict name -> score).
    """
    rows = {r["name"]: r for r in an["params"]}
    names = [nm for nm in run.names if nm != "ln_posterior"]
    fl = np.asarray(an["flagged"])
    if flag_impact and len(fl) > 0:
        idx = {nm: j for j, nm in enumerate(run.names)}
        zm = np.abs(an["zmean"])
        score = {nm: float(np.nanmax(zm[idx[nm], fl])) for nm in names}
    else:
        def concern(r):
            parts = []
            if np.isfinite(r["rhat"]):
                parts.append((r["rhat"] - 1.0) / 0.05)
            if np.isfinite(r["n_over_tau"]) and r["n_over_tau"] > 0:
                parts.append(50.0 / r["n_over_tau"])
            if np.isfinite(r["drift_z"]):
                parts.append(abs(r["drift_z"]) / 5.0)
            return max(parts) if parts else 0.0
        score = {nm: concern(rows[nm]) for nm in names}
    ranked = sorted(names, key=lambda nm: -score[nm])
    chosen = (["ln_posterior"] if "ln_posterior" in run.names else []) + ranked[:max(k - 1, 0)]
    return chosen, score


def plot_param_hist(run, an, params, ncols=2, nbins=50):
    """
    Marginal histograms (after the estimated burn-in) of the chosen parameters for: all walkers (black), without the
    flagged walkers (blue) and the flagged walkers only (red). Every curve is normalised by its own number of samples.
    The x-range comes from the unflagged walkers; flagged walkers outside it are clipped.
    """
    vi = an["view_idx"]
    mask = np.zeros(an["w"], dtype=bool)
    mask[an["flagged"]] = True
    has_fl = bool(mask.any()) and bool((~mask).any())
    nrow = int(np.ceil(len(params) / ncols))
    with _style():
        fig, axes = plt.subplots(nrow, ncols, figsize=(11, 2.7 * nrow), squeeze=False)
        for ax in axes.ravel():
            ax.set_visible(False)
        for ax, nm in zip(axes.ravel(), params):
            ax.set_visible(True)
            col = np.asarray(run.columns[nm][vi:], dtype=np.float64)
            allv = col[np.isfinite(col)]
            unfl = col[:, ~mask][np.isfinite(col[:, ~mask])] if (~mask).any() else allv
            ref = unfl if unfl.size > 10 else allv
            lo, hi = _plateau_ylim(ref, pad=0.05)          # robust: a few straggling walkers must not set the range
            bins = np.linspace(lo, hi, nbins + 1)
            h_all = _density(allv, bins)
            ax.stairs(h_all, bins, color=C["ink"], lw=1.8, label="all walkers")
            top = h_all.max()
            if has_fl:
                h_un = _density(unfl, bins)
                fo = col[:, mask][np.isfinite(col[:, mask])]
                ax.stairs(h_un, bins, color=C["blue"], lw=1.6, fill=False, label="without flagged walkers")
                ax.stairs(_density(fo, bins), bins, color=C["red"], lw=1.2, label="flagged walkers only")
                top = max(top, h_un.max())
                dm = (np.mean(unfl) - np.mean(allv)) / max(np.std(unfl), 1e-300)
                ax.text(0.99, 0.95, f"mean shift when flagged walkers are left out: {dm:+.2f} sd", transform=ax.transAxes,
                        ha="right", va="top", fontsize=8, color=C["gray"])
                out_frac = float(np.mean((fo < bins[0]) | (fo > bins[-1]))) if fo.size else 0.0
                if out_frac > 0.05:
                    ax.text(0.99, 0.06, f"{out_frac:.0%} of the flagged walkers' samples lie outside this range", transform=ax.transAxes,
                            ha="right", va="bottom", fontsize=8, color=C["red"])
            ax.set_ylim(0, 1.2 * top)
            ax.set_yticks([])
            ax.set_title(nm, loc="left", fontsize=10)
        axes.ravel()[0].legend(fontsize=8, loc="upper left")
        fig.tight_layout()
    return fig


def _hist_payload(run, an, names=None, nbins=50):
    """Histograms and summary numbers for every column, for the three groups and two x-ranges."""
    vi = an["view_idx"]
    mask = np.zeros(an["w"], dtype=bool)
    mask[an["flagged"]] = True
    series = {}
    names = list(names) if names else list(run.names)
    for nm in names:
        col = np.asarray(run.columns[nm][vi:], dtype=np.float64)
        step = max(1, int(np.ceil(col.size / 2e6)))
        col = col[::step]
        groups = {"all": col, "unflagged": col[:, ~mask] if (~mask).any() else col, "flagged": col[:, mask]}
        groups = {g: v[np.isfinite(v)] for g, v in groups.items()}
        ref = groups["unflagged"] if groups["unflagged"].size > 10 else groups["all"]
        lo, hi = _plateau_ylim(ref, pad=0.05)
        flo, fhi = float(groups["all"].min()), float(groups["all"].max())
        if fhi <= flo:
            flo, fhi = flo - 1, fhi + 1
        sets = {}
        for key, (a, b) in (("zoom", (lo, hi)), ("full", (flo, fhi))):
            edges = np.linspace(a, b, nbins + 1)
            sets[key] = {"edges": _sig(edges, 6), **{g: _sig(_density(v, edges), 4) for g, v in groups.items()}}
        stats = {}
        for g, v in groups.items():
            if v.size:
                q = np.percentile(v, [5, 50, 95])
                stats[g] = [int(v.size), float(v.mean()), float(v.std()), float(q[0]), float(q[1]), float(q[2])]
        series[nm] = {"sets": sets, "stats": stats}
    return {"names": names, "series": series, "n_flagged": int(mask.sum()), "default": None}


_HIST_JS = r"""
(function(){
  const D = JSON.parse(document.getElementById('hh-data').textContent);
  const el = function(id){ return document.getElementById(id); };
  const cv = el('hh-canvas'), ctx = cv.getContext('2d'), sel = el('hh-param'), st = el('hh-stats'),
        cbA = el('hh-all'), cbU = el('hh-unflagged'), cbF = el('hh-flagged'), selR = el('hh-range');
  D.names.forEach(function(n){ const o = document.createElement('option'); o.value = n; o.textContent = n; sel.appendChild(o); });
  if (D.default) sel.value = D.default;
  if (D.n_flagged === 0) { cbF.checked = false; cbF.disabled = true; }
  const H = 340, M = {l: 20, r: 14, t: 12, b: 38};
  let W = 800;
  function fmt(v){ return Math.abs(v) >= 1e5 ? v.toPrecision(10).replace(/\.?0+$/, '') : (+v.toPrecision(4)).toString(); }
  function nice(lo, hi, k){
    const span = (hi - lo) || 1, raw = span / k, p = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / p;
    const step = (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * p, out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9 * span; v += step) out.push(v);
    return out;
  }
  function resize(){
    const r = cv.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    W = r.width || 800; cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); draw();
  }
  function stairs(e, h, X, Y, color, lw, fill){
    ctx.beginPath(); ctx.moveTo(X(e[0]), Y(0));
    for (let k = 0; k < h.length; k++) { const y = Y(h[k] === null ? 0 : h[k]); ctx.lineTo(X(e[k]), y); ctx.lineTo(X(e[k + 1]), y); }
    ctx.lineTo(X(e[e.length - 1]), Y(0));
    if (fill) { ctx.fillStyle = fill; ctx.fill(); }
    ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.stroke();
  }
  function draw(){
    const S = D.series[sel.value], set = S.sets[selR.value] || S.sets.zoom, e = set.edges, nb = e.length - 1;
    const show = {all: cbA.checked, unflagged: cbU.checked, flagged: cbF.checked};
    let top = 0;
    ['all', 'unflagged'].forEach(function(g){ if (show[g]) set[g].forEach(function(v){ if (v !== null && v > top) top = v; }); });
    if (!(top > 0) && show.flagged) set.flagged.forEach(function(v){ if (v !== null && v > top) top = v; });
    top = (top > 0 ? top : 1) * 1.15;
    const X = function(v){ return M.l + (v - e[0]) / ((e[nb] - e[0]) || 1) * (W - M.l - M.r); };
    const Y = function(v){ return H - M.b - v / top * (H - M.t - M.b); };
    ctx.clearRect(0, 0, W, H);
    ctx.font = '11px system-ui, sans-serif'; ctx.fillStyle = '#555'; ctx.strokeStyle = '#e6e9ee'; ctx.lineWidth = 1;
    nice(e[0], e[nb], 8).forEach(function(v){ const x = X(v); ctx.beginPath(); ctx.moveTo(x, M.t); ctx.lineTo(x, H - M.b); ctx.stroke(); ctx.textAlign = 'center'; ctx.fillText(fmt(v), x, H - M.b + 16); });
    ctx.fillStyle = '#555'; ctx.textAlign = 'center'; ctx.fillText(sel.value, (W + M.l) / 2, H - 4);
    ctx.save(); ctx.beginPath(); ctx.rect(M.l, M.t, W - M.l - M.r, H - M.t - M.b); ctx.clip();
    if (show.unflagged) stairs(e, set.unflagged, X, Y, '#3b6ea5', 2, 'rgba(59,110,165,0.18)');
    if (show.all) stairs(e, set.all, X, Y, '#222831', 2, null);
    if (show.flagged) stairs(e, set.flagged, X, Y, '#c8453f', 1.6, null);
    ctx.restore();
    const names = {all: 'all walkers', unflagged: 'without flagged walkers', flagged: 'flagged walkers only'};
    let rows = '';
    ['all', 'unflagged', 'flagged'].forEach(function(g){
      if (!show[g] || !S.stats[g]) return; const s = S.stats[g];
      rows += '<tr><td>' + names[g] + '</td><td>' + s[0] + '</td><td>' + fmt(s[1]) + '</td><td>' + fmt(s[2]) + '</td><td>' + fmt(s[3]) + '</td><td>' + fmt(s[4]) + '</td><td>' + fmt(s[5]) + '</td></tr>';
    });
    st.innerHTML = '<table><thead><tr><th>group</th><th>samples</th><th>mean</th><th>std</th><th>5%</th><th>median</th><th>95%</th></tr></thead><tbody>' + rows + '</tbody></table>';
  }
  [sel, cbA, cbU, cbF, selR].forEach(function(c){ c.addEventListener('change', draw); });
  window.addEventListener('resize', resize);
  resize();
})();
"""


def _hist_explorer_html(run, an, names=None, default=None):
    pay = _hist_payload(run, an, names=names)
    pay["default"] = default
    payload = json.dumps(pay, separators=(",", ":")).replace("</", "<\\/")
    nf = pay["n_flagged"]
    return ('<div class="explorer"><div class="controls">'
            '<label>parameter <select id="hh-param"></select></label>'
            '<label><input type="checkbox" id="hh-all" checked> all walkers</label>'
            '<label><input type="checkbox" id="hh-unflagged" checked> without flagged walkers</label>'
            f'<label><input type="checkbox" id="hh-flagged" checked> flagged walkers only ({nf})</label>'
            '<label>x range <select id="hh-range"><option value="zoom">range of the unflagged walkers</option><option value="full">all values</option></select></label></div>'
            '<canvas id="hh-canvas" height="340"></canvas><div id="hh-stats"></div></div>'
            f'<script id="hh-data" type="application/json">{payload}</script><script>{_HIST_JS}</script>')


def _flag_breakdown_text(an):
    nf = len(an["flagged"])
    return (f"{nf} of {an['w']} walkers are flagged: {len(an['stuck'])} frozen, {len(an['low'])} with low ln p, "
            f"{len(an['offset'])} offset in a sampled parameter, {len(an['nonfinite'])} with mostly non-finite ln p. "
            "A walker can have more than one reason.")


_FLAGGED_JS = r"""
(function(){
  const D = JSON.parse(document.getElementById('lp-data').textContent);
  const el = function(id){ return document.getElementById(id); };
  const cv = el('fw-canvas'), ctx = cv.getContext('2d'), sel = el('fw-select'), selY = el('fw-yrange'),
        info = el('fw-info'), btnP = el('fw-prev'), btnN = el('fw-next');
  const flagged = D.flagged.slice().sort(function(a, b){ return a - b; });
  const steps = D.steps, n = steps.length;
  flagged.forEach(function(w){
    const o = document.createElement('option'); o.value = String(w);
    o.textContent = 'walker ' + w + ' (' + D.info[String(w)][0] + ')'; sel.appendChild(o);
  });
  let i0 = 0; while (i0 < n - 1 && steps[i0] < D.burn) i0++;
  const H = 360, M = {l: 70, r: 14, t: 12, b: 38};
  let W = 800;
  function fmt(v){ return Math.abs(v) >= 1e5 ? v.toPrecision(10).replace(/\.?0+$/, '') : (+v.toPrecision(5)).toString(); }
  function nice(lo, hi, k){
    const span = (hi - lo) || 1, raw = span / k, p = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / p;
    const step = (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * p, out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9 * span; v += step) out.push(v);
    return out;
  }
  function cur(){ const v = parseInt(sel.value, 10); return isNaN(v) ? flagged[0] : v; }
  function resize(){
    const r = cv.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    W = r.width || 800; cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); draw();
  }
  function draw(){
    const w = cur(), a = D.walkers[w], inf = D.info[String(w)];
    let lo = Infinity, hi = -Infinity;
    for (let i = i0; i < n; i++) {
      if (D.q16[i] !== null && D.q16[i] < lo) lo = D.q16[i];
      if (D.q84[i] !== null && D.q84[i] > hi) hi = D.q84[i];
    }
    if (!(hi > lo)) { lo = (isFinite(lo) ? lo : 0) - 1; hi = (isFinite(hi) ? hi : 0) + 1; }
    const pad = 0.35 * (hi - lo); lo -= pad; hi += pad;
    if (selY.value === 'walker') {
      for (let i = i0; i < n; i++) { const v = a[i]; if (v !== null) { if (v < lo) lo = v; if (v > hi) hi = v; } }
      const p2 = 0.05 * (hi - lo); lo -= p2; hi += p2;
    }
    const x0 = steps[0], x1 = steps[n - 1];
    const X = function(s){ return M.l + (s - x0) / ((x1 - x0) || 1) * (W - M.l - M.r); };
    const Y = function(v){ return H - M.b - (v - lo) / (hi - lo) * (H - M.t - M.b); };
    ctx.clearRect(0, 0, W, H);
    ctx.font = '11px system-ui, sans-serif'; ctx.fillStyle = '#555'; ctx.strokeStyle = '#e6e9ee'; ctx.lineWidth = 1;
    nice(lo, hi, 6).forEach(function(v){ const y = Y(v); ctx.beginPath(); ctx.moveTo(M.l, y); ctx.lineTo(W - M.r, y); ctx.stroke(); ctx.textAlign = 'right'; ctx.fillText(fmt(v), M.l - 6, y + 4); });
    nice(x0, x1, 8).forEach(function(v){ const x = X(v); ctx.beginPath(); ctx.moveTo(x, M.t); ctx.lineTo(x, H - M.b); ctx.stroke(); ctx.textAlign = 'center'; ctx.fillText(fmt(v), x, H - M.b + 16); });
    ctx.fillStyle = '#555'; ctx.textAlign = 'center'; ctx.fillText('step', (W + M.l) / 2, H - 4);
    ctx.save(); ctx.translate(14, (H - M.b + M.t) / 2); ctx.rotate(-Math.PI / 2); ctx.fillText('ln posterior', 0, 0); ctx.restore();
    ctx.save(); ctx.beginPath(); ctx.rect(M.l, M.t, W - M.l - M.r, H - M.t - M.b); ctx.clip();
    ctx.beginPath(); let st = false;
    for (let i = 0; i < n; i++) { if (D.q16[i] === null) continue; const x = X(steps[i]), y = Y(D.q16[i]); if (!st) { ctx.moveTo(x, y); st = true; } else ctx.lineTo(x, y); }
    for (let i = n - 1; i >= 0; i--) { if (D.q84[i] === null) continue; ctx.lineTo(X(steps[i]), Y(D.q84[i])); }
    ctx.closePath(); ctx.fillStyle = 'rgba(59,110,165,0.25)'; ctx.fill();
    function line(arr, color, lw){
      ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = lw; let s = false;
      for (let i = 0; i < n; i++) { const v = arr[i]; if (v === null) { s = false; continue; } const x = X(steps[i]), y = Y(v); if (!s) { ctx.moveTo(x, y); s = true; } else ctx.lineTo(x, y); }
      ctx.stroke();
    }
    line(D.median, '#222831', 1.6);
    line(a, '#c8453f', 2.0);
    if (D.burn > x0 && D.burn < x1) { ctx.setLineDash([5, 4]); ctx.strokeStyle = '#7b5ea7'; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(X(D.burn), M.t); ctx.lineTo(X(D.burn), H - M.b); ctx.stroke(); ctx.setLineDash([]); ctx.fillStyle = '#7b5ea7'; ctx.textAlign = 'left'; ctx.fillText('estimated burn-in', X(D.burn) + 5, M.t + 13); }
    ctx.restore();
    info.textContent = 'Walker ' + w + ': ' + inf[0] + '. Its median ln p is ' + (inf[1] >= 0 ? '+' : '') + fmt(inf[1]) + ' from the ensemble median. It moved in ' +
      (100 * inf[2]).toFixed(1) + '% of the last ' + D.tail_steps + ' steps; the longest unchanged stretch there is ' + inf[3] + ' steps.';
  }
  function step(d){ const k = flagged.indexOf(cur()), m = (k + d + flagged.length) % flagged.length; sel.value = String(flagged[m]); draw(); }
  sel.addEventListener('change', draw); selY.addEventListener('change', draw);
  btnP.addEventListener('click', function(){ step(-1); }); btnN.addEventListener('click', function(){ step(1); });
  window.addEventListener('resize', resize);
  resize();
})();
"""


def _flagged_explorer_html(run, an):
    """Dropdown over ALL flagged walkers, showing only ln p. Re-uses the data of the ln p explorer (id lp-data)."""
    if len(an["flagged"]) == 0:
        return '<p class="muted">No walker is flagged.</p>'
    return ('<div class="explorer"><div class="controls">'
            '<label>flagged walker <select id="fw-select"></select></label>'
            '<button type="button" id="fw-prev">previous</button><button type="button" id="fw-next">next</button>'
            '<label>y range <select id="fw-yrange"><option value="plateau">around the plateau</option><option value="walker">show this walker fully</option></select></label></div>'
            '<div id="fw-info" class="muted" style="margin-bottom:6px"></div>'
            '<canvas id="fw-canvas" height="360"></canvas></div>'
            f'<script>{_FLAGGED_JS}</script>')


# ---- the arguments section ------------------------------------------------------------------------------------------
# (group title, [(argument name as parsed, flag as typed, what it means)])
_ARG_GROUPS = [
    ("Run and output", [
        ("output_h5", "--output-h5 (-o)", "emcee backend file the samples are written to"),
        ("mode", "--mode (-m)", "which stretch of data is analysed relative to the cut: pre, post or full"),
        ("Tcut_cycles", "--Tcut-cycles (-t)", "cutoff time, in cycles from the waveform peak"),
        ("Tcut_seconds", "--Tcut-seconds (-ts)", "cutoff time, in seconds from the peak"),
        ("Tstart", "--Tstart", "start time of the analysed data segment"),
        ("Tend", "--Tend", "end time of the analysed data segment"),
        ("resume", "--resume", "continue an unfinished run from its last saved step"),
        ("verbose", "--verbose", "print extra information while running")]),
    ("Data and starting points", [
        ("ifos", "--ifos", "detectors used"),
        ("data", "--data", "strain data file for each detector"),
        ("psd", "--psd", "noise power spectral density file for each detector"),
        ("injected_parameters", "--injected-parameters", "parameters of an injected signal (an injection run instead of real data)"),
        ("reference_parameters", "--reference-parameters", "json with reference parameters: they fix the cutoff time and start the walkers"),
        ("reference_posterior_file", "--reference-posterior-file", "posterior file the reference parameters are taken from"),
        ("reference_parameter_method", "--reference-parameter-method", "how the reference sample is chosen from that posterior"),
        ("initial_walkers", "--initial-walkers", "folder or file the starting walkers come from"),
        ("initial_walker_type", "--initial-walker-type", "kind of that starting file: posterior, backend or walkers")]),
    ("Waveform and data conditioning", [
        ("approx", "--approx", "waveform model"),
        ("sampling_rate", "--sampling-rate", "sampling rate of the analysed data (Hz)"),
        ("flow", "--flow", "lower frequency bound for the data conditioning and the likelihood (Hz)"),
        ("fmax", "--fmax", "upper frequency bound (Hz)"),
        ("f22_start", "--f22-start", "frequency at which the (2,2) mode of the waveform starts (Hz)"),
        ("fref", "--fref", "reference frequency at which spins and inclination are defined (Hz)")]),
    ("Sampler", [
        ("nwalkers", "--nwalkers", "number of walkers"),
        ("nsteps", "--nsteps", "maximum number of steps"),
        ("ncpu", "--ncpu", "number of CPUs"),
        ("only_prior", "--only-prior", "sample the prior only, without the likelihood"),
        ("vary_time", "--vary-time", "sample over the geocenter time"),
        ("vary_skypos", "--vary-skypos", "sample over right ascension, declination and polarization")]),
    ("Priors", [
        ("total_mass_prior_bounds", "--total-mass-prior-bounds", "detector-frame total mass range (solar masses)"),
        ("mass_ratio_prior_bounds", "--mass-ratio-prior-bounds", "mass ratio range"),
        ("luminosity_distance_prior_bounds", "--luminosity-distance-prior-bounds", "luminosity distance range (Mpc)"),
        ("spin_magnitude_prior_bounds", "--spin-magnitude-prior-bounds", "spin magnitude range"),
        ("time_prior_sigma", "--time-prior-sigma", "width of the Gaussian prior on the geocenter time (s)")]),
]
_ARG_SHORT = {"o": "output_h5", "m": "mode", "t": "Tcut_cycles", "ts": "Tcut_seconds"}
_ARG_MULTI = {"data", "psd", "ifos"}


def _is_number(tok):
    try:
        float(tok)
        return True
    except ValueError:
        return False


def _loose_parse(tokens):
    """Turn a pasted command line into {argument: value(s)} without knowing the parser (used by the command line)."""
    out, key = {}, None
    for tok in tokens:
        if tok.startswith("-") and not _is_number(tok):
            raw = tok.lstrip("-")
            key = _ARG_SHORT.get(raw, raw.replace("-", "_"))
            out.setdefault(key, [])
        elif key is not None:
            out[key].append(float(tok) if _is_number(tok) else tok)
    return {k: (True if not v else (v if (len(v) > 1 or k in _ARG_MULTI) else v[0])) for k, v in out.items()}


def _fmt_arg(v):
    if v is None:
        return "not set"
    if v is True:
        return "yes"
    if v is False:
        return "no"
    if isinstance(v, (list, tuple)):
        return ", ".join(_fmt_arg(x) for x in v)
    if isinstance(v, float):
        return f"{v:.14g}"
    return str(v)


def _reconstruct_command(args_dict):
    parts = []
    for k, v in args_dict.items():
        if v is None or v is False:
            continue
        flag = "--" + k.replace("_", "-")
        if v is True:
            parts.append(flag)
        elif isinstance(v, (list, tuple)) and k in ("data", "psd"):
            parts += [f"{flag} {x}" for x in v]
        elif isinstance(v, (list, tuple)):
            parts.append(flag + " " + " ".join(_fmt_arg(x) for x in v))
        else:
            parts.append(f"{flag} {_fmt_arg(v)}")
    return " ".join(parts)


def _arguments_section(arguments, command_line, reference_values, run, an):
    """The 'Arguments' section: what the run was started with, and what the diagnostics used."""
    d = None
    if arguments is not None:
        d = dict(vars(arguments)) if hasattr(arguments, "__dict__") else dict(arguments)
    out = ['<section id="arguments"><h2>Arguments</h2>']

    def table(rows, head=("argument", "what it is", "value")):
        t = ['<table class="args"><colgroup><col style="width:27%"><col style="width:40%"><col style="width:33%"></colgroup>'
             f'<thead><tr><th>{head[0]}</th><th>{head[1]}</th><th>{head[2]}</th></tr></thead><tbody>']
        for a, m, v in rows:
            t.append(f"<tr><td>{_html.escape(a)}</td><td>{_html.escape(m)}</td><td>{_html.escape(v)}</td></tr>")
        t.append("</tbody></table>")
        return "".join(t)

    if d is not None:
        out.append('<p class="muted">The arguments the run was started with, as parsed (defaults included).</p>')
        used = set()
        for title, entries in _ARG_GROUPS:
            rows = [(flag, meaning, _fmt_arg(d[k])) for k, flag, meaning in entries if k in d]
            used |= {k for k, _, _ in entries}
            if rows:
                out.append(f"<h3>{title}</h3>" + table(rows))
        other = [("--" + k.replace("_", "-"), "", _fmt_arg(v)) for k, v in d.items() if k not in used]
        if other:
            out.append("<h3>Other</h3>" + table(other))
        cmd = command_line if command_line else _reconstruct_command(d)
        label = "Command line" if command_line else "Command line (reconstructed from the parsed arguments)"
        out.append(f'<details><summary>{label}</summary><pre class="cmd">{_html.escape(cmd)}</pre></details>')

    if reference_values:
        sky = bool(d.get("vary_skypos")) if d is not None else None
        tm = bool(d.get("vary_time")) if d is not None else None
        names = (("right_ascension", "right ascension (rad)", sky), ("declination", "declination (rad)", sky),
                 ("polarization", "polarization (rad)", sky), ("geocenter_time", "geocenter time (GPS s)", tm))
        rows = []
        for key, label, sampled in names:
            if key in reference_values:
                role = ("sampled; this is only the reference value" if sampled else "fixed at this value") if sampled is not None else ""
                rows.append((key, label + (f": {role}" if role else ""), _fmt_arg(float(reference_values[key]))))
        if rows:
            out.append("<h3>Reference values used by the run</h3>" + table(rows))

    layout = detect_layout(run.raw_names)
    rows = [("emcee file", "the file these diagnostics were made from", str(run.source)),
            ("steps in the file", "", f"{run.nsteps_total:,}"), ("walkers", "", f"{run.nwalkers}"),
            ("sampled columns", "numbers sampled per walker", f"{run.ndim}" + (f" ({layout['spin_model']} spins; sky position {'sampled' if layout['vary_skypos'] else 'fixed'}; time {'sampled' if layout['vary_time'] else 'fixed'})" if layout else ""))]
    if run.stride > 1:
        rows.append(("steps read", run.thin_note or f"every {_ordinal(run.stride)} step", f"{run.n:,} of {run.nsteps_total:,}"))
    rows.append(("estimated burn-in", "from the ensemble-median ln p", f"step {an['est_step']:,}" if an["est_found"] else "no plateau found"))
    if an["burn_idx"] > 0:
        rows.append(("burn-in removed", "steps removed before the statistics", f"{an['burn_step']:,}"))
    if run.physical_info:
        pi = run.physical_info
        rows.append(("prior bounds for the physical parameters",
                     "used to convert the sampled numbers" + (" (tdinf defaults)" if pi["auto"] else ""),
                     f"total mass {pi['mtot_lim'][0]:g} to {pi['mtot_lim'][1]:g}; mass ratio {pi['q_lim'][0]:g} to {pi['q_lim'][1]:g}; "
                     f"distance {pi['dist_lim'][0]:g} to {pi['dist_lim'][1]:g}; spin magnitude {pi['chi_lim'][0]:g} to {pi['chi_lim'][1]:g}"))
    rows.append(("walker-flag rules", "fixed set of criteria", f"version {FLAG_RULES_VERSION}"))
    out.append("<h3>What the diagnostics used</h3>" + table(rows, head=("setting", "what it is", "value")))
    out.append("</section>")
    return "".join(out)


def _scan_table(an):
    sc = an.get("burn_scan")
    if sc is None or len(sc["step"]) == 0:
        return ""
    head = ["steps removed", "% of run", "max R-hat", "min N/tau", "max |drift z|", "plateau median ln p"]
    out = ['<table><thead><tr>' + "".join(f"<th>{h}</th>" for h in head) + "</tr></thead><tbody>"]
    for k in range(len(sc["step"])):
        out.append("<tr>" + _cell(f"{sc['step'][k]:,}") + _cell(f"{100 * sc['frac'][k]:.0f}%") +
                   _cell(_fmt(sc["rhat"][k]), _status(sc["rhat"][k], 1.01, 1.05)) +
                   _cell(_fmt(sc["n_over_tau"][k]), _status(sc["n_over_tau"][k], 50, 20, True)) +
                   _cell(_fmt(sc["drift"][k]), _status(sc["drift"][k], 3, 5)) + _cell(_fmt(sc["plateau"][k])) + "</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def build_report(run, out_path, title=None, burn=0, trace_pages=True, explorer=True, dpi=110,
                 trace_params=None, physical="auto", arguments=None, command_line=None, reference_values=None,
                 extras=None, kind="emcee", **analysis_kwargs):
    """
    Analyse `run` and write a self-contained HTML report (images embedded, no internet needed).
    burn=0 (default) discards nothing; see `analyze` for the other options.
    physical : "auto" (default), False, or a dict. Physical parameters (masses, spins, angles, ...) are computed from the
    sampled numbers and shown in the histograms with and without flagged walkers. With "auto" the run configuration is
    recognised from the column names (use names=get_layout(...) when reading) and tdinf's default prior bounds are used
    (TDINF_DEFAULT_BOUNDS); a dict overrides any of mtot_lim, q_lim, dist_lim, chi_lim, fixed, keys, spin_model,
    vary_skypos, vary_time. False turns the conversion off. (Or call `run.add_physical(...)` yourself first.)
    trace_params : optional list of column names for the static trace plots; by default all columns are shown.
    arguments : the arguments the run was started with (an argparse Namespace or a dict); shown in the Arguments section.
    command_line : the command line as a string (otherwise it is reconstructed from `arguments`).
    reference_values : dict with right_ascension, declination, polarization, geocenter_time as used by the run.
    extras : optional callable extras(run, an) -> dict, called once the analysis is done, for samplers with more to report than walkers (sampler_diagnostics uses it for the
    tempering of Eryn). The dict may hold: "sections" (list of (id, title, html), shown before Methods and in the navigation), "checks" (list of dicts with key, label, status
    ("ok", "warn", "bad", "na"), value, rule: added to the checks table and to the verdict), "diagnosis" and "suggestions" (lists of sentences), "cards" (list of (label, value)
    or (label, value, tooltip)), "methods" (html added to the Methods section) and "header" (text added to the line under the title).
    kind : the name of the sampler, used in the default title.
    Returns the analysis dict.
    """
    t0 = time.time()
    physical_problem = ""
    if physical is not False and not run.derived_names:
        cfg = detect_layout(run.raw_names)
        if cfg is not None or isinstance(physical, dict):
            kw = dict(cfg or {}, **TDINF_DEFAULT_BOUNDS)
            if isinstance(physical, dict):
                kw.update(physical)
            try:
                run.add_physical(**kw)
                run.physical_info["auto"] = not (isinstance(physical, dict) and any(k in physical for k in TDINF_DEFAULT_BOUNDS))
            except (KeyError, ValueError) as e:
                physical_problem = f"Physical parameters could not be computed: {e.args[0] if e.args else e}"
        else:
            physical_problem = ("The column names are not a tdinf layout, so the run configuration is unknown. "
                                "Read the file with names=get_layout(...), or pass physical=dict(...).")
    an = analyze(run, burn=burn, **analysis_kwargs)
    ex = extras(run, an) if extras is not None else {}
    if ex.get("checks"):
        an["checks"] = list(an["checks"]) + [dict(c) for c in ex["checks"]]
        an["verdict"] = _worst(an["verdict"], *[c["status"] for c in ex["checks"]])
    if ex.get("diagnosis"):
        an["diagnosis"] = list(an["diagnosis"]) + list(ex["diagnosis"])
    if ex.get("suggestions"):
        an["suggestions"] = list(an["suggestions"]) + list(ex["suggestions"])
    title = title or f"{kind} diagnostics: {run.source}"
    verdict_text = {"ok": "No red flags in the automated checks", "warn": "Some checks need attention",
                    "bad": "Problems detected: do not trust these chains yet", "na": "Not enough information"}[an["verdict"]]

    cards = [("steps in file", f"{run.nsteps_total:,}"), ("walkers", f"{run.nwalkers}"),
             ("lnL evaluations", f"{run.nsteps_total * run.nwalkers:,}", "steps x walkers"),
             ("estimated burn-in", f"{an['est_step']:,}" if an["est_found"] else "none found"),
             ("steps analysed", f"{an['n_post'] * run.stride:,}"), ("flagged walkers", f"{len(an['flagged'])}")]
    if an["burn_idx"] > 0:
        cards.insert(4, ("burn-in removed", f"{an['burn_step']:,}"))
    cards.extend(ex.get("cards", []))
    cards_html = "".join(f'<div class="card" title="{c[2] if len(c) > 2 else ""}"><div class="k">{c[0]}</div><div class="v">{c[1]}</div></div>' for c in cards)
    burn_note = (f'<p class="muted">{_html.escape(run.thin_note)} Autocorrelation times are scaled back by the stride, so N/tau is not affected.</p>'
                 if run.stride > 1 else "")

    def badge(s):
        return f'<span class="badge b-{s}">{ {"ok": "OK", "warn": "CHECK", "bad": "PROBLEM", "na": "n/a"}[s] }</span>'

    def dot(s):
        label = {"ok": "good", "warn": "check", "bad": "problem", "na": "not available"}[s]
        return f'<span class="dot b-{s}" title="{label}" aria-label="{label}"></span>'

    legend_html = ('<div class="legend"><span><i class="dot b-ok"></i> good</span><span><i class="dot b-warn"></i> check</span>'
                   '<span><i class="dot b-bad"></i> problem</span><span><i class="dot b-na"></i> not available</span></div>')
    checks_html = legend_html + ('<table class="checks"><colgroup><col style="width:24%"><col style="width:9%"><col style="width:32%"><col style="width:35%"></colgroup>'
                   '<thead><tr><th>check</th><th>status</th><th>result</th><th>what counts as good</th></tr></thead><tbody>' +
                   "".join(f'<tr><td data-l="check"><b>{_html.escape(c["label"])}</b></td><td data-l="status">{dot(c["status"])}</td>'
                           f'<td data-l="result">{_html.escape(c["value"])}</td><td data-l="good" class="muted">{_html.escape(c["rule"])}</td></tr>' for c in an["checks"]) + "</tbody></table>")

    bi_ = an["burn_info"]
    tot = run.steps[-1] + 1
    diag_html = ("<h3>Reading this run</h3><ul>" + "".join(f"<li>{_html.escape(t)}</li>" for t in an["diagnosis"]) + "</ul>"
                 "<h3>What to try</h3><ul>" + "".join(f"<li>{_html.escape(t)}</li>" for t in an["suggestions"]) + "</ul>")
    found_txt = (f"step {an['est_step']:,} ({100 * an['est_step'] / tot:.0f}% of the run)." if an["est_found"]
                 else "no such block (only the last two blocks qualify), so no plateau is declared.")
    burn_explain = (
        '<details><summary>How the burn-in is estimated</summary><ol>'
        '<li>At every saved step take the median ln p over all walkers (the <i>ensemble median</i>).</li>'
        f'<li>Cut the run into {bi_["nblocks"]} blocks of about {bi_["block_steps"]:,.0f} steps and take the median of the ensemble median in each block.</li>'
        f'<li>The <b>plateau</b> is the median of the ensemble median over the last {bi_["tail_frac"]:.0%} of the run: {bi_["ref"]:.2f}. '
        f'The <b>tolerance</b> is the larger of 0.1 x the robust scatter of late ln p ({0.1 * bi_["late_sd"]:.2f}) and 3 x the scatter of the late ensemble median '
        f'({3 * bi_["med_sd"]:.2f}): {bi_["tol"]:.2f}.</li>'
        f'<li>The burn-in ends at the start of the first block after which every block median stays above plateau minus tolerance. Here: {found_txt}</li>'
        '<li>The estimate decides where walkers are judged (frozen, low ln p, offset) and where the purple dashed lines are drawn. '
        'The plots below show R-hat, N/tau and drift as a function of the number of steps removed from the start.</li></ol>'
        '<p class="muted">Limits: the resolution is one block; it only looks at the ensemble-median ln p, so it cannot tell whether individual walkers have forgotten their start '
        'or whether every parameter has equilibrated. Treat it as a lower bound.</p></details>')

    secs = []
    secs.append(f'<section id="overview"><h2>Overview</h2><p class="muted">Generated {time.strftime("%Y-%m-%d %H:%M")} from <code>{_html.escape(run.source)}</code>.</p>'
                f'<div class="verdict {an["verdict"]}">{badge(an["verdict"])}<strong>{verdict_text}</strong></div>'
                f'<div class="cards">{cards_html}</div>{burn_note}{diag_html}{checks_html}'
                f'<p class="muted" style="margin-top:12px">Automated checks are heuristics. They can reveal problems but cannot prove convergence; see <a href="#methods">Methods</a>.</p></section>')

    secs.append(_arguments_section(arguments, command_line, reference_values, run, an))

    secs.append('<section id="lnp"><h2>Log-probability</h2><p class="muted">All walkers should climb to, and then fluctuate around, the same plateau. '
                'Each line below is one walker; flagged walkers (frozen, low ln p, offset) are red.</p>'
                + _lnp_explorer_html(run, an)
                + "<h3>Per-walker distributions</h3><p class=\"muted\">Computed on the part of the run after the estimated burn-in. Left: the ln p histogram of every walker; "
                  "walkers that mix lie on top of each other. Right: each walker's median and 16-84% range, sorted by median; a mixed ensemble gives a flat row of dots.</p>"
                + _dist_explorer_html(run, an) + "</section>")

    secs.append('<section id="burnin"><h2>Burn-in</h2>'
                '<p class="muted">Top: the approach of ln p to its plateau on a log step axis; the purple dashed line is the estimated burn-in and the green band is the tolerance used to find it. '
                'Bottom: three convergence statistics recomputed after removing the first part of the run. If a statistic keeps improving as more steps are removed, the chain has not forgotten its start.</p>'
                + burn_explain
                + _stat_guide_html()
                + '<p class="muted" style="margin-top:8px">In the plots each statistic is the worst value over all parameters: the largest R-hat, the smallest N/tau, the largest |drift z|. '
                  'Green dashed line: the value that separates good from not good; red dotted line: the value beyond which it is a problem; green shading: the good region.</p>'
                + _img(plot_burnin(run, an), "burn-in", dpi) + "<h3>Statistics versus steps removed</h3>" + _scan_table(an)
                + '<p class="muted">The scan uses up to 3000 evenly spaced steps and 128 unflagged walkers for speed.</p></section>')

    phys_names = [nm for nm in run.names if nm in run.derived_names]
    if phys_names:
        hist_names = ["ln_posterior"] + phys_names
        pi = run.physical_info
        hist_intro = ("Physical parameters, converted from the sampled numbers (ln p is included as well). Prior bounds used for the conversion: "
                      f"total mass {pi['mtot_lim'][0]:g} to {pi['mtot_lim'][1]:g}, mass ratio {pi['q_lim'][0]:g} to {pi['q_lim'][1]:g}, "
                      f"distance {pi['dist_lim'][0]:g} to {pi['dist_lim'][1]:g} Mpc, spin magnitude {pi['chi_lim'][0]:g} to {pi['chi_lim'][1]:g}"
                      + (" (the tdinf defaults; the h5 file does not store them, so if your run used other bounds pass "
                         "<code>physical=dict(mtot_lim=(...), q_lim=(...), dist_lim=(...), chi_lim=(...))</code>). " if pi["auto"] else ". "))
    else:
        hist_names = list(run.names)
        hist_intro = ("<b>These are the sampled coordinates, not physical parameters.</b> " + _html.escape(physical_problem) + " "
                      if physical_problem else "These are the sampled coordinates, not physical parameters. ")
    score = select_parameters(run, an, k=len(run.names), flag_impact=True)[1]
    candidates = [nm for nm in hist_names if nm != "ln_posterior"]
    default_hist = max(candidates, key=lambda nm: score.get(nm, 0.0)) if candidates else hist_names[0]
    nf_ = len(an["flagged"])
    if nf_:
        hist_note = (f"Black: all walkers. Blue: without the {nf_} flagged walkers. Red: the flagged walkers only. Each curve is normalised by its own number of samples. "
                     "A blue curve that matches the black one means the flagged walkers hardly change that parameter. "
                     "The parameter shown first is the one in which the flagged walkers sit furthest from the others.")
    else:
        hist_note = "No walker is flagged, so the curves with and without flagged walkers are identical."
    secs.append('<section id="walkers"><h2>Walkers</h2><p class="muted">Movement is measured on ' + _html.escape(an["moved_src"]) + '. Click a column header to sort.</p>'
                + _flag_rules_html(an)
                + _walkers_table(an)
                + "<h3>Flagged walkers: ln p</h3><p class=\"muted\">" + _html.escape(_flag_breakdown_text(an)) + " Pick any of them: the plot shows its ln p against the unflagged walkers (68% band and median).</p>"
                + _flagged_explorer_html(run, an)
                + "<h3>Acceptance fraction: it should be roughly 0.2 to 0.5</h3><p class=\"muted\">The share of proposed moves that are accepted. Far below 0.2 walkers hardly move.</p>"
                + _img(plot_acceptance(an), "acceptance", dpi)
                + "<h3>Parameter histograms with and without flagged walkers</h3><p class=\"muted\">" + hist_intro + _html.escape(hist_note) + "</p>"
                + _hist_explorer_html(run, an, names=hist_names, default=default_hist) + "</section>")

    tr = ['<section id="traces"><h2>Traces</h2>']
    if explorer:
        payload = json.dumps(_explorer_payload(run, an), separators=(",", ":")).replace("</", "<\\/")
        tr.append('<p class="muted">Interactive explorer: bands are the 68% and 90% ranges across the unflagged walkers, black is their median, '
                  'red lines are flagged walkers, grey are random walkers. The dashed line marks the estimated burn-in.</p>'
                  '<div class="explorer"><div class="controls"><label>parameter <select id="ex-param"></select></label>'
                  '<label><input type="checkbox" id="ex-bands" checked> bands</label>'
                  '<label><input type="checkbox" id="ex-walkers" checked> random walkers</label>'
                  '<label><input type="checkbox" id="ex-flagged" checked> flagged walkers</label>'
                  '<label><input type="checkbox" id="ex-all"> include burn-in in the y-range</label></div>'
                  '<canvas id="ex-canvas" height="380"></canvas><div id="ex-tip" class="tip"></div></div>'
                  f'<script id="ex-data" type="application/json">{payload}</script><script>{_EXPLORER_JS}</script>')
    if trace_pages:
        trace_names = list(trace_params) if trace_params else list(run.names)
        tr.append(f"<h3>Static trace plots</h3><p class=\"muted\">All {len(trace_names)} parameters. Shaded: the 68% and 90% ranges across the unflagged walkers; black: their median; "
                  "red lines: flagged walkers; grey lines: random walkers; purple dashed line: the estimated burn-in. "
                  "Right-hand panels compare the first (blue) and second (orange) half of the chain after the estimated burn-in: they should coincide.</p>")
        for k, fig in enumerate(plot_traces(run, an, params=trace_names, per_page=5)):
            tr.append(_img(fig, f"traces page {k + 1}", dpi))
    tr.append("</section>")
    secs.append("".join(tr))

    secs.append('<section id="convergence"><h2>Convergence and autocorrelation</h2><p class="muted">N/tau above 50 is the primary criterion; R-hat across walkers is a supporting check because walkers of an ensemble sampler are not independent.</p>'
                + "<h3>Autocorrelation function</h3>" + _img(plot_acf(run, an), "autocorrelation function", dpi)
                + "<h3>tau against chain length</h3><p class=\"muted\">tau must level off, and stay under the N/50 line.</p>" + _img(plot_autocorr(an), "autocorrelation time", dpi)
                + "<h3>Per-parameter statistics</h3>" + _params_table(an)
                + ('<p class="muted">* tau below 3 rows of the thinned chain: not resolved at this stride. Raise the step limit or thin less.</p>'
                   if run.stride > 1 and any(not r["tau_resolved"] for r in an["params"]) else "") + "</section>")

    for sid, stitle, body in ex.get("sections", []):
        secs.append(f'<section id="{sid}"><h2>{_html.escape(stitle)}</h2>{body}</section>')

    secs.append('<section id="methods"><h2>Methods and caveats</h2>'
                '<details open><summary>What is computed</summary><ul>'
                '<li><b>Reading.</b> ' + ("" if run.stride == 1 else f"Only every {_ordinal(run.stride)} saved step was read; autocorrelation times are multiplied back by the stride. ") +
                'Movement of walkers uses the last steps at full resolution.</li>'
                '<li><b>Burn-in.</b> Estimated as the start of the first block after which the ensemble-median ln p stays within tolerance of its late-run plateau (explained in the Overview). Walker flags look at the part of the run after it. To remove the first steps before computing R-hat, N/tau and drift, pass <code>burn="auto"</code>, a fraction, or a number of steps.</li>'
                '<li><b>tau.</b> emcee-style integrated autocorrelation time: autocorrelation of each walker, averaged over walkers, integrated with Sokal\'s window (c = 5). ESS = walkers x N / tau. The sampled ln p is analysed like a parameter.</li>'
                '<li><b>R-hat.</b> Rank-normalised split R-hat (bulk and folded tail), maximum of the two. The "clean" column excludes flagged walkers.</li>'
                '<li><b>Drift.</b> Difference between the means of the two halves of the analysed chain, in standard errors computed with the ESS.</li>'
                '<li><b>Walker flags.</b> The criteria are written out in plain language at the top of the Walkers section, with the numbers used in this run.</li>'
                '<li><b>Mixing.</b> The spread of the walker medians of ln p divided by the typical spread within one walker. Walkers of a mixed ensemble share the same ln p levels, so this is small (below about 0.3).</li>'
                '</ul></details>'
                '<details><summary>Caveats</summary><ul>'
                '<li>Sampled coordinates may include redundant dimensions (for example radial parts of Cartesian pairs). Their posterior is their prior; they are included in the tables.</li>'
                '<li>A multimodal posterior can look converged if no walker ever visits the other mode. Check the log-probability distributions of the walkers.</li>'
                '<li>The early part of the run, before the plateau, makes R-hat, drift and N/tau look worse; the Burn-in section shows by how much.</li>'
                '</ul></details>' + ex.get("methods", "") + '</section>')

    nav_items = [("overview", "Overview"), ("arguments", "Arguments"), ("lnp", "Log-probability"), ("burnin", "Burn-in"),
                 ("walkers", "Walkers"), ("traces", "Traces"), ("convergence", "Convergence")]
    nav_items += [(sid, stitle) for sid, stitle, _ in ex.get("sections", [])] + [("methods", "Methods")]
    nav = "".join(f'<a href="#{i}">{_html.escape(t)}</a>' for i, t in nav_items)
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{_html.escape(title)}</title><style>{_CSS}</style></head><body>'
            f'<header><h1>{_html.escape(title)}</h1><p>{run.nsteps_total:,} steps, {run.nwalkers} walkers, {run.ndim} sampled numbers{_html.escape(ex.get("header", ""))}</p></header>'
            f'<div class="layout"><nav>{nav}</nav><main>{"".join(secs)}</main></div>'
            f'<footer>report generated in {time.time() - t0:.0f} s by sampler_diagnostics.py</footer><script>{_SORT_JS}</script></body></html>')
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(page)
    print(f"wrote {out_path} ({len(page) / 1e6:.1f} MB) in {time.time() - t0:.0f} s; verdict: {an['verdict']}")
    return an


# ============================================================================== run_sampler integration and command line

_RS_BOUNDS = (("mtot_lim", "total_mass_prior_bounds"), ("q_lim", "mass_ratio_prior_bounds"),
              ("dist_lim", "luminosity_distance_prior_bounds"), ("chi_lim", "spin_magnitude_prior_bounds"))
_FIXED_ALIASES = {"right_ascension": ("right_ascension", "ra"), "declination": ("declination", "dec"),
                  "polarization": ("polarization", "psi"), "geocenter_time": ("geocenter_time", "geocent_time")}


def infer_layout_from_file(path, vary_skypos, vary_time, group="mcmc"):
    """
    Work out which columns an emcee file holds from the number of columns and the --vary-skypos / --vary-time flags.
    The spin model follows from the column count (precessing, aligned and spinless runs differ by 4 columns each).
    Returns (spin_model, names). Raises a clear error if the flags do not match the file.
    """
    import h5py
    sky, tm = bool(vary_skypos), bool(vary_time)
    with h5py.File(path, "r") as f:
        ndim = int(f[group]["chain"].shape[-1])
    ok = [m for m in ("precessing", "aligned", "none") if len(get_layout(m, sky, tm)) == ndim]
    if len(ok) != 1:
        raise ValueError(f"{path} has {ndim} sampled columns, which does not match --vary-skypos={sky} --vary-time={tm}. "
                         f"Run configurations with {ndim} columns: {candidate_layouts(ndim)}")
    return ok[0], get_layout(ok[0], sky, tm)


def _fixed_from_json(path):
    """Right ascension, declination, polarization and geocenter time from a run_sampler --reference-parameters json."""
    with open(path) as f:
        d = json.load(f)
    out = {}
    for key, aliases in _FIXED_ALIASES.items():
        for a in aliases:
            if a in d:
                out[key] = float(d[a])
                break
    return out


def _print_summary(an):
    word = {"ok": "OK", "warn": "CHECK", "bad": "PROBLEM", "na": "n/a"}
    print(f"\nDiagnostics verdict: {word[an['verdict']]}")
    for c in an["checks"]:
        print(f"  [{word[c['status']]:7s}] {c['label']}: {c['value']}")
    print(f"  {len(an['flagged'])} of {an['w']} walkers flagged (rules version {FLAG_RULES_VERSION})")


def _physical_from_args(args, kwargs, spin_model, sky, tm, fixed=None):
    """The `physical` settings of build_report from run_sampler's args (prior bounds, reference values of what was not sampled) and kwargs."""
    phys = dict(spin_model=spin_model, vary_skypos=sky, vary_time=tm)
    for key, attr in _RS_BOUNDS:
        v = getattr(args, attr, None)
        if v is not None:
            phys[key] = tuple(float(x) for x in v)
    fx = {}
    ref = getattr(args, "reference_parameters", None)
    if ref and os.path.exists(str(ref)):
        fx.update(_fixed_from_json(str(ref)))
    if kwargs:
        fx.update({k: float(kwargs[k]) for k in _FIXED_ALIASES if k in kwargs})
    fx.update(fixed or {})
    phys["fixed"] = fx
    return phys


def report_from_args(args, kwargs=None, out_path=None, safe=True, burn=0, thin=1, max_points=None, max_steps=100_000,
                     group="mcmc", title=None, fixed=None, spin_model=None, physical="args", arguments=None,
                     command_line=None, **build_kw):
    """
    Write the diagnostics report for a tdinf run from run_sampler.py's own `args` (and, optionally, its `kwargs` dict).

    Uses args.output_h5 (the emcee backend), args.vary_skypos, args.vary_time and the four prior-bound arguments
    (total_mass_prior_bounds, mass_ratio_prior_bounds, luminosity_distance_prior_bounds, spin_magnitude_prior_bounds).
    The spin model is inferred from the number of columns. Right ascension, declination, polarization and geocenter time
    are only needed when they were not sampled; they are taken from `fixed`, then `kwargs`, then args.reference_parameters.
    The report goes to <output_h5 without .h5>_diagnostics.html unless out_path is given.

    safe=True (default): any error is printed as a warning and None is returned, so a diagnostics problem cannot
    break the run that called this. Returns the analysis dict.

    max_steps : a run with more steps than this (default 100,000) is read with a stride (about max_steps steps are used);
    the report says so. None or 0 turns it off. The Arguments section shows `args` (or `arguments`) and `kwargs`.
    """
    def _go():
        path = args.output_h5
        sky, tm = bool(getattr(args, "vary_skypos", False)), bool(getattr(args, "vary_time", False))
        if spin_model is None:
            sm, names = infer_layout_from_file(path, sky, tm, group)
        else:
            sm, names = spin_model, get_layout(spin_model, sky, tm)
        run = read_emcee_h5(path, names=names, group=group, thin=thin, max_points=max_points, max_steps=max_steps or None)
        phys = _physical_from_args(args, kwargs, sm, sky, tm, fixed) if (isinstance(physical, str) and physical == "args") else physical
        out = out_path or (re.sub(r"\.h5$", "", str(path)) + "_diagnostics.html")
        an = build_report(run, out, title=title or f"emcee diagnostics: {os.path.basename(str(path))}", burn=burn,
                          physical=phys, arguments=args if arguments is None else arguments, command_line=command_line,
                          reference_values=kwargs, **build_kw)
        _print_summary(an)
        return an

    if not safe:
        return _go()
    try:
        return _go()
    except Exception as e:                                     # a diagnostics problem must not break the run
        print(f"WARNING: the diagnostics report could not be made: {type(e).__name__}: {e}")
        return None


def _parse_burn(b):
    if str(b).lower() == "auto":
        return "auto"
    v = float(b)
    return 0 if v == 0 else (v if 0 < v < 1 else int(v))


def emcee_main(argv=None):
    """
    Command line that understands run_sampler.py's arguments. You can paste the run_sampler command line; flags that
    are not about the diagnostics are ignored. Example:
        python sampler_diagnostics.py emcee -o full/full.h5 --vary-skypos --vary-time --total-mass-prior-bounds 150 400
    """
    p = argparse.ArgumentParser(allow_abbrev=False, description="Diagnose a tdinf emcee backend and write an HTML report. "
                                "Takes run_sampler.py's arguments; unrelated ones are ignored.")
    p.add_argument("-o", "--output-h5", "--h5", dest="output_h5", default=None, help="the emcee backend file (as in run_sampler.py)")
    p.add_argument("--vary-time", action="store_true", help="as in run_sampler.py: time was sampled")
    p.add_argument("--vary-skypos", action="store_true", help="as in run_sampler.py: sky position was sampled")
    p.add_argument("--total-mass-prior-bounds", "--mtot-lim", type=float, nargs=2, default=[200, 350])
    p.add_argument("--mass-ratio-prior-bounds", "--q-lim", type=float, nargs=2, default=[0.17, 1])
    p.add_argument("--luminosity-distance-prior-bounds", "--dist-lim", type=float, nargs=2, default=[100, 10000])
    p.add_argument("--spin-magnitude-prior-bounds", "--chi-lim", type=float, nargs=2, default=[0, 0.99])
    p.add_argument("--reference-parameters", default=None,
                   help="as in run_sampler.py (json); supplies the sky position and time if they were not sampled")
    p.add_argument("--fixed-ra", type=float, default=None, help="right ascension, if it was not sampled")
    p.add_argument("--fixed-dec", type=float, default=None, help="declination, if it was not sampled")
    p.add_argument("--fixed-psi", type=float, default=None, help="polarization, if it was not sampled")
    p.add_argument("--fixed-time", type=float, default=None, help="geocenter time, if it was not sampled")
    p.add_argument("--spin-model", choices=["precessing", "aligned", "none"], default=None,
                   help="normally inferred from the number of columns")
    p.add_argument("--report", default=None, help="output html (default: <output_h5>_diagnostics.html)")
    p.add_argument("--group", default="mcmc")
    p.add_argument("--thin", type=int, default=1, help="keep every n-th step (default 1: all steps)")
    p.add_argument("--max-points", type=float, default=None, help="cap on steps x walkers x columns kept (raises the stride)")
    p.add_argument("--max-steps", type=int, default=100000,
                   help="a run with more steps than this is read with a stride (default 100000; 0 turns it off)")
    p.add_argument("--burn", default="0", help="steps to remove before the statistics: 0 (default), 'auto', a fraction in (0,1), or a number")
    p.add_argument("--no-physical", action="store_true", help="do not compute physical parameters")
    p.add_argument("--title", default=None)
    a, ignored = p.parse_known_args(argv)
    if not a.output_h5:
        p.error("give the emcee backend with -o/--output-h5, as in run_sampler.py")
    args = argparse.Namespace(output_h5=a.output_h5, vary_time=a.vary_time, vary_skypos=a.vary_skypos,
                              total_mass_prior_bounds=a.total_mass_prior_bounds, mass_ratio_prior_bounds=a.mass_ratio_prior_bounds,
                              luminosity_distance_prior_bounds=a.luminosity_distance_prior_bounds,
                              spin_magnitude_prior_bounds=a.spin_magnitude_prior_bounds, reference_parameters=a.reference_parameters)
    fixed = {k: v for k, v in (("right_ascension", a.fixed_ra), ("declination", a.fixed_dec),
                               ("polarization", a.fixed_psi), ("geocenter_time", a.fixed_time)) if v is not None}
    tokens = list(argv) if argv is not None else sys.argv[1:]
    report_from_args(args, out_path=a.report, safe=False, burn=_parse_burn(a.burn), thin=a.thin, max_points=a.max_points,
                     max_steps=a.max_steps, group=a.group, title=a.title, fixed=fixed, spin_model=a.spin_model,
                     physical=False if a.no_physical else "args", arguments=_loose_parse(tokens), command_line=" ".join(tokens))


# ==============================================================================================================================================
# PART 2: bilby_mcmc (what used to be bilby_diagnostics.py; the names that differ from the emcee part carry a bilby_ prefix)
# ==============================================================================================================================================


# ============================================================================== reading the pickle without bilby or dill

class Placeholder:
    def __init__(self, *args, **kwargs):          # an unknown global that the pickle calls with arguments (types.MethodType, a function rebuilt by dill ...): the arguments are only kept
        if args:
            self.__dict__['_args'] = args

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        elif isinstance(state, tuple) and len(state) == 2 and isinstance(state[1], dict):
            self.__dict__.update(state[1])
        else:
            self._state = state

    # a class that is not known here may have been a dict, list or set (bilby's PriorDict, which the prior-draw proposals of the tdinf_gw cycle keep a reference to): accept what the
    # pickle then does to it, and keep it in plain attributes
    def __setitem__(self, key, value):
        self.__dict__.setdefault('_items', {})[key] = value

    def __getitem__(self, key):
        return self.__dict__.get('_items', {})[key]

    def append(self, value):
        self.__dict__.setdefault('_list', []).append(value)

    def extend(self, values):
        self.__dict__.setdefault('_list', []).extend(values)

    def add(self, value):
        self.__dict__.setdefault('_list', []).append(value)

    def update(self, *args, **kwargs):
        self.__dict__.setdefault('_items', {}).update(*args, **kwargs)

    def __repr__(self):
        return f"<{type(self).__name__}>"


_made = {}


def _placeholder(module, name):
    if (module, name) not in _made:
        _made[(module, name)] = type(name, (Placeholder,), {'__module__': module})
    return _made[(module, name)]


def _create_array(f, args, state, npdict=None):       # dill's way of pickling a numpy array (f is numpy's own _reconstruct, via the whitelist)
    array = f(*args)
    array.__setstate__(state)
    if npdict is not None:
        array.__dict__ = npdict
    return array


_SAFE = {('collections', n) for n in ('Counter', 'OrderedDict', 'defaultdict', 'deque')} | {('copyreg', '_reconstructor'), ('copyreg', '__newobj__')} | \
        {('builtins', n) for n in ('set', 'frozenset', 'slice', 'complex', 'bytearray', 'range', 'object', 'list', 'dict', 'tuple', 'int', 'float', 'str', 'bytes')}


_NAMEDTUPLES = {
    'ConvergenceInputs': ['autocorr_c', 'burn_in_nact', 'thin_by_nact', 'fixed_discard', 'target_nsamples', 'stop_after_convergence', 'L1steps', 'L2steps', 'min_tau', 'fixed_tau', 'tau_window'],
    'ParallelTemperingInputs': ['ntemps', 'nensemble', 'Tmax', 'Tmax_from_SNR', 'initial_betas', 'adapt', 'adapt_t0', 'adapt_nu', 'pt_ensemble'],
}


class _Safe(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('numpy') and name in ('_reconstruct', 'ndarray', 'dtype', 'scalar', 'float64', 'int64', 'bool_', 'float32', 'int32'):
            import importlib
            return getattr(importlib.import_module(module), name)
        if (module, name) in _SAFE:
            import importlib
            return getattr(importlib.import_module(module), name)
        if module.endswith('bilby_mcmc.utils') and name in _NAMEDTUPLES:      # bilby's two input tuples, when a file was written by plain pickle instead of dill
            return collections.namedtuple(name, _NAMEDTUPLES[name])
        if module == 'dill._dill' and name == '_create_namedtuple':
            return lambda n, fields, mod, defaults=None: collections.namedtuple(n, fields, defaults=defaults)
        if module == 'dill._dill' and name == '_create_array':
            return _create_array
        if module == 'dill._dill' and name == '_load_type':                 # dill's way of pickling a class object (the prior-draw proposals of the tdinf_gw cycle hold some)
            return lambda type_name, *args: _placeholder('dill.types', str(type_name))
        if module == 'dill._dill':                                          # any other dill helper (functions, code objects, cells ...): never run, the arguments are only kept
            def inert(*args, **kwargs):
                held = _placeholder('dill._dill', name)()
                held._args = args
                return held
            return inert
        return _placeholder(module, name)


def load(path):
    with open(path, 'rb') as f:
        return _Safe(f).load()


# ============================================================================== the run

TWO_PI = 2 * np.pi
CIRCULAR_PERIOD = {"phase": TWO_PI, "right_ascension": TWO_PI, "mean_anomaly": TWO_PI, "spin1_azimuth": TWO_PI, "spin2_azimuth": TWO_PI, "polarization": np.pi}
SHORT = {"AdaptiveGaussianProposal": "AG", "DifferentialEvolutionProposal": "DE", "FisherMatrixProposal": "Fisher", "FixedGaussianProposal": "FG", "UniformProposal": "Uniform",
         "PriorProposal": "Prior draw", "KDEProposal": "KDE", "GMMProposal": "GMM", "EnsembleStretch": "Stretch", "TdinfPolarisationPeriodJump": "psi+pi jump"}
PHYSICAL_NEEDED = ("total_mass", "mass_ratio", "spin1_magnitude", "spin2_magnitude", "spin1_tilt", "spin2_tilt")


def _kind(p):
    n = type(p).__name__
    return SHORT.get(n, n.replace("Proposal", ""))


def _acc(p):
    a, r = getattr(p, "_accepted", getattr(p, "accepted", 0)), getattr(p, "_rejected", getattr(p, "rejected", 0))
    return a / (a + r) if (a + r) else float("nan"), int(a + r)


def _first(sampler, kind):
    return next((p for p in sampler.proposal_cycle.proposal_list if type(p).__name__ == kind), None)


class BilbyRun:
    """What the resume pickle of a bilby_mcmc run holds, as arrays: every chain (all temperatures, all ensemble members), the settings and the counters."""

    def __init__(self, pt, path, drop=None):
        self.pt, self.path, self.drop = pt, str(path), drop
        sd = pt.sampler_dictionary
        self.sd = sd
        self.nT = len(sd)
        self.nE = len(sd[0])
        c0 = sd[0][0].chain
        self.keys = list(c0.keys)
        self.names = list(self.keys[:-2])
        self.il = self.keys.index("logl") if "logl" in self.keys else len(self.keys) - 2
        self.arrays = {}
        self.cold_full = {}                       # the cold chains with every row, for the swap test (dropping rows would remove the evidence)
        for T in sd:
            for E in range(len(sd[T])):
                c = sd[T][E].chain
                A = np.asarray(c._chain_array[: c.position + 1])
                if T == 0:
                    self.cold_full[E] = A
                if drop:
                    A = A[(np.arange(len(A)) % drop[0]) != drop[1]]
                self.arrays[(T, E)] = A
        self.rows = min(len(self.arrays[(0, E)]) for E in range(self.nE))
        self.betas = np.array([float(sd[T][0].beta) for T in sorted(sd)])
        ci, pi = getattr(pt, "convergence_inputs", None), getattr(pt, "pt_inputs", None)
        self.conv = dict(ci._asdict()) if hasattr(ci, "_asdict") else {}
        self.ptin = dict(pi._asdict()) if hasattr(pi, "_asdict") else {}
        self.L1 = int(getattr(pt, "L1steps", self.conv.get("L1steps", 1)))
        self.L2 = int(getattr(pt, "L2steps", self.conv.get("L2steps", 1)))
        self.swap_rounds = int(getattr(pt, "swap_counter", {}).get("temperature", 0))
        self.iterations = self.swap_rounds * self.L2
        self.rows_per_iter = (len(self.arrays[(0, 0)]) / self.iterations) if self.iterations else 1.0
        self.sampling_time = float(getattr(pt, "sampling_time", 0.0))
        self.adapting = bool(getattr(pt, "adapt", False))
        self.hist_n = sorted(getattr(pt, "_nsamples_dict", {}).items())
        self.hist_tau = sorted(getattr(c0, "max_tau_dict", {}).items())
        self.bilby_burn = []
        for E in range(self.nE):
            b = getattr(sd[0][E].chain, "_last_minimum_index", None)
            self.bilby_burn.append((int(b[0]), int(b[1]), str(b[2])) if b is not None and len(b) >= 3 else (0, 0, "?"))
        self.has_physical = all(k in self.names for k in PHYSICAL_NEEDED)
        self.target = self.conv.get("target_nsamples")
        self.thin_by = float(self.conv.get("thin_by_nact", 1.0) or 1.0)

    # ---- arrays
    def logl_cold(self, rows=None):
        rows = rows or self.rows
        return np.column_stack([self.arrays[(0, E)][:rows, self.il] for E in range(self.nE)])

    def centres(self, burn):
        """circular mean of every angle over the cold chains after the burn-in (angles are unwrapped around it before any statistic is computed)"""
        pooled = np.vstack([self.arrays[(0, E)][burn:] for E in range(self.nE)])
        return {k: float(np.angle(np.exp(2j * np.pi * pooled[:, self.keys.index(k)] / per).mean()) * per / TWO_PI) for k, per in CIRCULAR_PERIOD.items() if k in self.names}

    def table(self, A, centres, derived=True):
        """one chain array -> {name: values}; angles unwrapped around their circular mean; derived quantities appended"""
        out = {}
        for k in self.names:
            x = A[:, self.keys.index(k)]
            if k in CIRCULAR_PERIOD:
                per = CIRCULAR_PERIOD[k]
                x = (x - centres[k] + per / 2) % per - per / 2
            out[k] = x
        if derived and self.has_physical:
            out.update(derived_quantities(out))
        return out

    def stack(self, T, burn, centres, names=None, derived=True):
        """{name: (n_post, nE_T)} for temperature T"""
        tabs = [self.table(self.arrays[(T, E)][burn:self.rows], centres, derived) for E in range(len([1 for (t, e) in self.arrays if t == T]))]
        names = names or list(tabs[0])
        return {k: np.column_stack([t[k] for t in tabs]) for k in names}


def derived_quantities(X):
    q, a1, a2, t1, t2 = X["mass_ratio"], X["spin1_magnitude"], X["spin2_magnitude"], X["spin1_tilt"], X["spin2_tilt"]
    M = X["total_mass"]
    out = {"chi_eff": (a1 * np.cos(t1) + q * a2 * np.cos(t2)) / (1 + q),
           "chi_p": np.maximum(a1 * np.sin(t1), (4 * q + 3) / (4 + 3 * q) * q * a2 * np.sin(t2)),
           "spin1_z": a1 * np.cos(t1), "spin2_z": a2 * np.cos(t2), "spin1_in_plane": a1 * np.sin(t1), "spin2_in_plane": a2 * np.sin(t2),
           "chirp_mass": M * q ** 0.6 / (1 + q) ** 1.2, "mass_1": M / (1 + q), "mass_2": M * q / (1 + q)}
    return out


def choose_burn(run, burn):
    if isinstance(burn, str) and burn == "bilby":
        idx = [b[1] for b in run.bilby_burn]
        v = int(max(idx)) if idx else 0
        return int(v * (1 - 1.0 / run.drop[0])) if run.drop else v
    if isinstance(burn, str) and burn == "estimated":
        return int(estimate_burnin(run.logl_cold())[0])
    if isinstance(burn, float) and 0 < burn < 1:
        return int(burn * run.rows)
    return int(burn)


# ============================================================================== swaps, ladder, round trips

def _quantile_points(L, betas, beta, m=500):
    """m equally weighted ln L values that represent the distribution at inverse temperature beta, from the chain whose beta is closest (importance reweighting)"""
    k = int(np.argmin(np.abs(np.log(betas) - np.log(beta))))
    x = np.sort(np.asarray(L[k], dtype=float))
    w = np.exp((beta - betas[k]) * (x - x.max()))
    c = np.cumsum(w)
    c /= c[-1]
    return np.interp((np.arange(m) + 0.5) / m, c, x)


def p_swap(L, betas, ba, bb, m=500):
    """predicted acceptance of a swap between chains at ba > bb, from the stored ln L: E[min(1, exp((ba - bb)(L_b - L_a)))]"""
    a, b = _quantile_points(L, betas, ba, m), _quantile_points(L, betas, bb, m)
    return float(np.mean(np.minimum(1.0, np.exp(np.minimum((ba - bb) * (b[None, :] - a[:, None]), 50.0)))))


def equalised_ladder(L, betas, ntemps, beta_min):
    """ladder from beta = 1 to beta_min with the same predicted swap acceptance in every pair"""
    def build(p):
        out = [1.0]
        for _ in range(ntemps - 1):
            lo, hi = 1e-4, out[-1]
            for _ in range(16):
                mid = np.sqrt(lo * hi)
                lo, hi = (lo, mid) if p_swap(L, betas, out[-1], mid, 300) > p else (mid, hi)        # acceptance too high: the next beta can be lower
            out.append(np.sqrt(lo * hi))
        return out
    lo, hi = 0.02, 0.95
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if build(mid)[-1] > beta_min else (mid, hi)                           # the ladder stops above beta_min: the pairs must be farther apart, so a lower acceptance
    p = 0.5 * (lo + hi)
    out = build(p)
    out[-1] = beta_min
    return np.array(out), p


def round_trip(p, rounds=60000, seed=0):
    """mean number of swap rounds for a replica to go hot -> cold -> hot with these swap acceptances (adjacent pairs, tried in order from the cold end)"""
    rng = np.random.default_rng(seed)
    n = len(p) + 1
    where = np.arange(n)
    state, last, trips = "to_cold", None, []
    for r in range(rounds):
        for k in range(n - 1):
            if rng.random() < p[k]:
                where[k], where[k + 1] = where[k + 1], where[k]
        t = int(np.flatnonzero(where == n - 1)[0])
        if state == "to_cold" and t == 0:
            state = "to_hot"
        elif state == "to_hot" and t == n - 1:
            state = "to_cold"
            if last is not None:
                trips.append(r - last)
            last = r
    return float(np.mean(trips)) if trips else float("nan")


def swap_check(run, burn):
    """do the swaps carry the exchanged state forward? Looks at the cold chain's rows that jump far from the previous row and asks whether the next row returns to the old state."""
    out = []
    keys = [k for k in run.names if k not in CIRCULAR_PERIOD] or run.names
    ix = [run.keys.index(k) for k in keys]
    b0 = max([b[1] for b in run.bilby_burn] + [0]) if run.drop else burn                # with dropped rows the cut is in the original row numbers
    for E in range(run.nE):
        Z = run.cold_full[E][b0:][:, ix]
        n = len(Z)
        if n < 400:
            out.append(dict(jumps=0, frac=0.0, share=float("nan"), verdict="na"))
            continue
        Z = Z / (Z.std(axis=0) + 1e-300)
        step = np.linalg.norm(np.diff(Z, axis=0), axis=1)
        typical = np.median(step)
        big = np.r_[False, step > 4 * typical]                      # row k jumped away from row k-1
        t = np.arange(2, n - 1)
        jump = big[t] & ~big[t - 1]
        tj = t[jump]
        if len(tj) < 30:
            out.append(dict(jumps=len(tj), frac=len(tj) / max(len(t), 1), share=float("nan"), verdict="na"))
            continue
        returns = np.linalg.norm(Z[tj + 1] - Z[tj - 1], axis=1) < np.linalg.norm(Z[tj + 1] - Z[tj], axis=1)
        share = float(returns.mean())
        out.append(dict(jumps=len(tj), frac=len(tj) / len(t), share=share, verdict="bad" if share > 0.7 else "ok" if share < 0.3 else "na"))
    return out


# ============================================================================== proposals: step sizes and prior walls

def step_sizes(run, X):
    """the step of the all-parameter adaptive Gaussian proposal of the cold chain for every parameter, as a share of the posterior standard deviation"""
    ag = _first(run.sd[0][0], "AdaptiveGaussianProposal")
    if ag is None:
        return None
    rows = []
    for k in run.names:
        if k not in ag.prior_width_dict:
            continue
        w = float(ag.prior_width_dict[k])
        w = 1.0 if not np.isfinite(w) else w
        sig = float(ag.sigmas.get(k, 1.0)) if hasattr(ag, "sigmas") else 1.0
        step = float(ag.scale) * w * sig
        sd = float(np.std(X[k]))
        rows.append(dict(name=k, width=w, sigma=sig, step=step, post_sd=sd, share=step / sd if sd > 0 else float("nan")))
    return dict(scale=float(ag.scale), accept=_acc(ag)[0], rows=rows, sigmas_all_one=all(abs(r["sigma"] - 1) < 1e-9 for r in rows))


def prior_walls(run, burn, njump=300000, seed=0):
    """how many differential-evolution jumps of the cold chain would leave the prior, per parameter, and which of those parameters have a hard wall (no reflective / periodic boundary)"""
    de = _first(run.sd[0][0], "DifferentialEvolutionProposal")
    if de is None:
        return None
    rng = np.random.default_rng(seed)
    keys = [k for k in run.names if k in de.prior_minimum_dict]
    ix = [run.keys.index(k) for k in keys]
    A = np.vstack([run.arrays[(0, E)][burn:run.rows][:, ix] for E in range(run.nE)])
    if len(A) < 100:
        return None
    lo = np.array([de.prior_minimum_dict[k] for k in keys], float)
    hi = np.array([de.prior_maximum_dict[k] for k in keys], float)
    kind = [str(de.prior_boundary_dict.get(k)) for k in keys]
    d = len(keys)
    frac = float(getattr(de, "mode_hopping_frac", 0.5))
    x = A[rng.integers(len(A), size=njump)]
    a, b = A[rng.integers(len(A), size=njump)], A[rng.integers(len(A), size=njump)]
    gamma = np.where(rng.random(njump) > frac, 1.0, rng.normal(0, 2.38 / np.sqrt(2 * d), njump) * np.exp(np.log(0.1) + np.log(100.0) * rng.random(njump)))
    y = x + gamma[:, None] * (a - b)
    finite = np.isfinite(lo) & np.isfinite(hi) & (np.abs(hi - lo) < 1e8)
    out = (y < lo) | (y > hi)
    out[:, ~finite] = False
    per = np.array([k == "periodic" for k in kind])
    out[:, per] = False
    hard = np.array([k == "None" for k in kind])
    return dict(names=keys, boundary=kind, share=out.mean(axis=0), any_hard=float(out[:, hard].any(axis=1).mean()) if hard.any() else 0.0,
                any_wall=float(out.any(axis=1).mean()), mode_hopping_frac=frac)


# ============================================================================== the analysis

def _tau_per_chain(x, c):
    return np.array([integrated_autocorr_time(_fill_nan(x[:, [e]]), c) for e in range(x.shape[1])])


def bilby_analyze(run, burn="bilby", tau_c=5.0, ess_target=400, verbose=False):
    t0 = time.time()
    an = {"burn_choice": burn}
    n = run.rows
    est_idx, est_found, est_info = estimate_burnin(run.logl_cold(), return_info=True)
    an.update(est_idx=int(est_idx), est_found=bool(est_found), est_info=est_info)
    bi = choose_burn(run, burn)
    an["bi"], an["bilby_idx"] = bi, choose_burn(run, "bilby")
    n_post = n - bi
    an["n_post"] = n_post
    centres = run.centres(bi)
    an["centres"] = centres
    X = run.stack(0, bi, centres)
    an["X"] = X
    names = list(X)
    sampled = list(run.names)
    nE = run.nE
    L = {T: np.concatenate([run.arrays[(T, E)][bi:run.rows][:, run.il] for E in range(len([1 for (t, e) in run.arrays if t == T]))]) for T in range(run.nT)}
    an["lnl_by_T"] = L
    ct = run.conv.get("target_nsamples")

    # ---- per parameter
    rows = []
    for j, nm in enumerate(names):
        if verbose:
            print(f"  [{j + 1}/{len(names)}] {nm}", flush=True)
        x = _fill_nan(X[nm])
        tau = integrated_autocorr_time(x, tau_c)
        tc = _tau_per_chain(x, tau_c)
        rb, rt = rhat_bulk_tail(x) if nE >= 2 else (np.nan, np.nan)
        rhat = np.nanmax([rb, rt]) if np.isfinite([rb, rt]).any() else np.nan
        ok = np.isfinite(tau) and tau > 0
        n_over = n_post / tau if ok else np.nan
        ess = nE * n_post / tau if ok else np.nan
        dz = drift_z(x, tau)
        q05, q50, q95 = np.percentile(x, [5, 50, 95])
        st = dict(tau=_status(n_over, 50, 20, True), rhat=_status(rhat, 1.01, 1.05), ess=_status(ess, ess_target, ess_target / 2.5, True), drift=_status(abs(dz) if np.isfinite(dz) else np.nan, 3, 5))
        rows.append(dict(name=nm, derived=nm not in sampled, mean=float(x.mean()), std=float(x.std()), q05=float(q05), q50=float(q50), q95=float(q95), rhat=rhat, tau=tau,
                         tau_min=float(np.nanmin(tc)), tau_max=float(np.nanmax(tc)), n_over_tau=n_over, ess=ess, drift_z=dz, st_tau=st["tau"], st_rhat=st["rhat"], st_ess=st["ess"],
                         st_drift=st["drift"], status=_worst(*st.values())))
    an["params"] = rows
    srow = [r for r in rows if not r["derived"]]

    # ---- tau as a function of chain length (computed here) for the slowest parameters
    slow_names = [r["name"] for r in sorted(srow, key=lambda r: -(r["tau"] if np.isfinite(r["tau"]) else 0))[:6]]
    Ns = np.unique(np.geomspace(max(40, n_post // 30), n_post, 10).astype(int))
    an["tau_curve"] = dict(N=Ns, names=slow_names, tau={nm: [integrated_autocorr_time(_fill_nan(X[nm][:N]), tau_c) for N in Ns] for nm in slow_names}) if n_post > 120 else None

    # ---- per temperature
    tauT = {}
    for T in range(run.nT):
        XT = run.stack(T, bi, centres, names=sampled, derived=False)
        tauT[T] = {nm: integrated_autocorr_time(_fill_nan(XT[nm]), tau_c) for nm in sampled}
    an["tau_by_T"] = tauT

    # ---- slowest direction of the posterior
    Zc = np.column_stack([X[k].reshape(-1) for k in sampled])
    mu, sg = Zc.mean(0), Zc.std(0) + 1e-300
    evals, evecs = np.linalg.eigh(np.corrcoef(Zc.T))
    pc_tau = []
    for j in range(len(sampled)):
        proj = (((np.column_stack([X[k] for k in sampled]).reshape(n_post, nE, len(sampled)) - mu) / sg) @ evecs[:, j])
        pc_tau.append(integrated_autocorr_time(proj, tau_c))
    order = np.argsort(np.nan_to_num(pc_tau, nan=0.0))[::-1]
    an["slow_dirs"] = [dict(tau=float(pc_tau[j]), eigenvalue=float(evals[j]), loadings=dict(zip(sampled, evecs[:, j]))) for j in order[:3]]
    v = evecs[:, order[0]]
    tau_dir_T = {}
    for T in range(run.nT):
        XT = run.stack(T, bi, centres, names=sampled, derived=False)
        proj = ((np.stack([XT[k] for k in sampled], axis=-1) - mu) / sg) @ v
        tau_dir_T[T] = integrated_autocorr_time(proj, tau_c)
    an["tau_dir_by_T"] = tau_dir_T

    # ---- swaps, ladder, round trip
    betas = run.betas
    acc_pair = []
    for T in range(run.nT - 1):
        s = run.sd[T][0]
        acc_pair.append(s.pt_accepted / max(s.pt_accepted + s.pt_rejected, 1))
    pred = [p_swap(L, betas, betas[T], betas[T + 1]) for T in range(run.nT - 1)]
    rt_rounds = round_trip(acc_pair)
    rows_per_round = run.rows / max(run.swap_rounds, 1)
    an["swaps"] = dict(recorded=acc_pair, predicted=pred, round_trip_rounds=rt_rounds, rows_per_round=rows_per_round, round_trip_rows=rt_rounds * rows_per_round,
                       round_trip_iterations=rt_rounds * run.L2)
    try:
        beta_eq, p_eq = equalised_ladder(L, betas, run.nT, float(betas[-1]))
        pred_eq = [p_swap(L, betas, beta_eq[T], beta_eq[T + 1]) for T in range(run.nT - 1)]
        rt_eq = round_trip(pred_eq)
        an["ladder_design"] = dict(betas=beta_eq, target=p_eq, predicted=pred_eq, round_trip_rounds=rt_eq)
    except Exception as e:                                     # the design is an extra; it must not stop the report
        an["ladder_design"] = None
        an["ladder_design_error"] = f"{type(e).__name__}: {e}"
    try:                                         # when every parameter decorrelates within 12 rows at a temperature well below the hottest, a cooler ladder gives a shorter round trip
        tmax_by_T = [max(v for v in tauT[T].values() if np.isfinite(v)) for T in range(run.nT)]
        first = next((T for T in range(run.nT) if tmax_by_T[T] <= 12), run.nT - 1)
        margin = min(first + 1, run.nT - 1)                          # one temperature hotter than the first one that decorrelates
        beta_need = float(betas[margin])
        best = None
        if beta_need > 1.3 * float(betas[-1]) and first >= 1:
            for nt in range(4, run.nT + 1):
                lad, p_eq = equalised_ladder(L, betas, nt, beta_need)
                pred_c = [p_swap(L, betas, lad[T], lad[T + 1]) for T in range(nt - 1)]
                if min(pred_c) < 0.25:
                    continue
                rt_c = round_trip(pred_c, rounds=60000)
                if best is None or rt_c < best["round_trip_rounds"]:
                    best = dict(betas=lad, ntemps=nt, predicted=pred_c, round_trip_rounds=rt_c, beta_min=beta_need, first_decorrelating=first, tau_by_T=tmax_by_T)
        an["ladder_cool"] = best
    except Exception as e:
        an["ladder_cool"] = None
        an["ladder_cool_error"] = f"{type(e).__name__}: {e}"
    an["swap_check"] = swap_check(run, bi)

    # ---- proposals
    props = []
    cyc = run.sd[0][0].proposal_cycle.proposal_list
    for i, p in enumerate(cyc):
        row = dict(index=i, kind=_kind(p), n=len(getattr(p, "parameters", [])), weight=float(getattr(p, "weight", float("nan"))), by_T=[], scale=[])
        for T in range(run.nT):
            accs, scales = [], []
            for E in range(len(run.sd[T])):
                q = run.sd[T][E].proposal_cycle.proposal_list
                if i < len(q) and type(q[i]).__name__ == type(p).__name__:
                    a_, k_ = _acc(q[i])
                    accs.append(a_)
                    if hasattr(q[i], "scale"):
                        scales.append(float(q[i].scale))
            row["by_T"].append(float(np.nanmedian(accs)) if accs else float("nan"))
            row["scale"].append(float(np.median(scales)) if scales else float("nan"))
        row["count"] = _acc(p)[1]
        props.append(row)
    an["proposals"] = props
    stretch = getattr(run.pt, "ensemble_proposal_cycle", None)
    an["stretch"] = None
    if stretch is not None and getattr(stretch, "proposal_list", None):
        s_ = stretch.proposal_list[0]
        an["stretch"] = dict(acc=_acc(s_)[0], count=_acc(s_)[1])
    an["steps"] = step_sizes(run, X)
    an["walls"] = prior_walls(run, bi)

    # ---- chains (the cold chains of the ensemble)
    lnl = run.logl_cold()[bi:]
    med = np.median(lnl, axis=0)
    gap = (med - np.median(med)) / max(robust_sd(lnl.reshape(-1)), 1e-12)
    chains = []
    tau_by_chain = np.column_stack([_tau_per_chain(_fill_nan(X[k]), tau_c) for k in sampled])        # (nE, nparams)
    pooled_sd = np.array([np.std(X[k]) for k in sampled])
    pooled_mean = np.array([np.mean(X[k]) for k in sampled])
    offsets = np.zeros((nE, len(sampled)))
    for E in range(nE):
        for j, k in enumerate(sampled):
            tj = tau_by_chain[E, j] if np.isfinite(tau_by_chain[E, j]) else 1.0
            se = pooled_sd[j] * np.sqrt(max(tj, 1.0) / max(n_post, 1))
            offsets[E, j] = (np.mean(X[k][:, E]) - pooled_mean[j]) / max(se, 1e-300)
    taus_max = np.nanmax(tau_by_chain, axis=1)
    med_tau = float(np.nanmedian(taus_max))
    for E in range(nE):
        s = run.sd[0][E]
        flags = []
        if gap[E] < -3 and (med[np.argmax(med)] - med[E]) > 1.0:
            flags.append("low ln L")
        if taus_max[E] > 3 * med_tau and taus_max[E] > 100:
            flags.append("slow")
        if np.nanmax(np.abs(offsets[E])) > 5:
            flags.append("offset")
        b = run.bilby_burn[E]
        a_, _ = _acc(s)
        chains.append(dict(id=E, flags=", ".join(flags), median_lnl=float(med[E]), gap=float(gap[E]), max_lnl=float(np.max(lnl[:, E])), tau_max=float(taus_max[E]),
                           tau_param=sampled[int(np.nanargmax(tau_by_chain[E]))], n_eff=float(n_post / (run.thin_by * taus_max[E])) if np.isfinite(taus_max[E]) and taus_max[E] > 0 else float("nan"),
                           burn=b[1], burn_method=b[2], local_accept=s.accepted / max(s.accepted + s.rejected, 1), offset_max=float(np.nanmax(np.abs(offsets[E]))),
                           offset_param=sampled[int(np.nanargmax(np.abs(offsets[E])))]))
    an["chains"] = chains
    an["offsets"] = offsets
    an["tau_by_chain"] = tau_by_chain
    an["flagged"] = [c["id"] for c in chains if c["flags"]]

    # ---- bilby's own counting and a corrected estimate of the time to finish
    n_now = run.hist_n[-1][1] if run.hist_n else float("nan")
    rate = float(np.nansum([1.0 / (run.thin_by * t) for t in taus_max if np.isfinite(t) and t > 0]))      # independent samples per row, summed over the cold chains
    n_est = n_post * rate
    an["counts"] = dict(n_now=n_now, n_peak=max([v for k, v in run.hist_n], default=float("nan")), target=ct, n_est=n_est, rate_per_row=rate)
    time_per_iter = run.sampling_time / run.iterations if run.iterations else float("nan")
    an["time"] = dict(per_iteration=time_per_iter, per_step_ms=1e3 * time_per_iter / run.L1 if run.L1 else float("nan"), hours=run.sampling_time / 3600.0)
    if ct and rate > 0 and np.isfinite(time_per_iter):
        remaining = max(ct - n_est, 0.0)
        rows_needed = remaining / rate
        iters_needed = rows_needed / run.rows_per_iter
        an["etf"] = dict(remaining=remaining, rows=rows_needed, iterations=iters_needed, hours=iters_needed * time_per_iter / 3600.0)
        # what bilby's printed line would say: one chain makes all the remaining samples at the primary chain's tau
        t_prim = taus_max[0]
        an["etf"]["bilby_hours"] = max(ct - n_est, 0.0) * run.thin_by * t_prim * time_per_iter / 3600.0 if np.isfinite(t_prim) else float("nan")      # bilby: samples x thin x tau x L1 x time per step
    else:
        an["etf"] = None

    # ---- burn-in scan: the statistics after discarding more and more of the run
    fracs = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)
    scan = []
    cen0 = run.centres(0)
    full = {k: run.stack(0, 0, cen0, names=[k], derived=False)[k] for k in sampled}
    for f in fracs:
        b = int(f * n)
        if n - b < 120:
            continue
        rh, nt, dr = [], [], []
        for k in sampled:
            x = _fill_nan(full[k][b:])
            tau_ = integrated_autocorr_time(x, tau_c)
            if nE >= 2:
                bb, tt = rhat_bulk_tail(x)
                rh.append(np.nanmax([bb, tt]))
            nt.append((n - b) / tau_ if np.isfinite(tau_) and tau_ > 0 else np.nan)
            dr.append(abs(drift_z(x, tau_)) if np.isfinite(tau_) else np.nan)
        scan.append(dict(frac=f, rows=b, rhat=float(np.nanmax(rh)) if rh else np.nan, n_over_tau=float(np.nanmin(nt)), drift=float(np.nanmax(dr))))
    an["burn_scan"] = scan

    # ---- checks
    sw = an["swaps"]
    rec = np.array(sw["recorded"], dtype=float)
    n_over = np.array([r["n_over_tau"] for r in srow], dtype=float)
    rh = np.array([r["rhat"] for r in srow], dtype=float)
    ess_arr = np.array([r["ess"] for r in srow], dtype=float)
    dz_arr = np.abs(np.array([r["drift_z"] for r in srow], dtype=float))
    wn, wr, we, wd = (int(np.nanargmin(n_over)) if np.isfinite(n_over).any() else 0, int(np.nanargmax(rh)) if np.isfinite(rh).any() else 0,
                      int(np.nanargmin(ess_arr)) if np.isfinite(ess_arr).any() else 0, int(np.nanargmax(dz_arr)) if np.isfinite(dz_arr).any() else 0)
    checks = []
    if len(rec):
        lo_i, hi_i = int(np.argmin(rec)), int(np.argmax(rec))
        st = "ok" if rec.min() >= 0.2 else ("warn" if rec.min() >= 0.1 else "bad")
        if st == "ok" and rec.max() > 0.8:
            st = "warn"
        checks.append(("swap", "Swap acceptance between neighbouring temperatures", st, f"lowest {rec.min():.2f} (T{lo_i} to T{lo_i + 1}), highest {rec.max():.2f} (T{hi_i} to T{hi_i + 1})",
                       "Good from 0.2 up to 0.8 in every pair; check from 0.1; problem below 0.1. A pair that rarely swaps is a bottleneck for the whole ladder."))
        sp = float(rec.max() - rec.min())
        checks.append(("spread", "Evenness of the ladder", "ok" if sp < 0.25 else ("warn" if sp < 0.45 else "bad"), f"spread (highest minus lowest) {sp:.2f}",
                       "Good below 0.25: a ladder with equal acceptances loses the least time. The Temperatures section shows a more even ladder."))
    sc = an["swap_check"]
    concl = [s_ for s_ in sc if s_["verdict"] != "na"]
    if sc:
        nbad, nok = sum(s_["verdict"] == "bad" for s_ in concl), sum(s_["verdict"] == "ok" for s_ in concl)
        med_share = float(np.nanmedian([s_["share"] for s_ in concl])) if concl else float("nan")
        if concl and nbad > len(concl) / 2:
            checks.append(("carry", "Swaps carry the exchanged state forward", "bad", f"the next row returns to the old state in {100 * med_share:.0f}% of jumps (median over {len(concl)} conclusive chains)",
                           "In a correct run it is below about 30%. A high value means the swaps only rewrote the stored rows (the bug fixed by make_bilby_swaps_continue)."))
        elif concl and nok > len(concl) / 2:
            checks.append(("carry", "Swaps carry the exchanged state forward", "ok", f"the next row returns to the old state in {100 * med_share:.0f}% of jumps (median over {len(concl)} conclusive chains)",
                           "Good below about 30%."))
        else:
            checks.append(("carry", "Swaps carry the exchanged state forward", "na", "not conclusive (too few jumps, mixed results between the chains, or other large moves such as KDE / mixture proposals confuse the test)",
                           "The test looks at jumps of the stored state and works best for slowly moving chains."))
    checks.append(("tau", "Chain length vs autocorrelation time", _status(np.nanmin(n_over), 50, 20, True) if np.isfinite(n_over).any() else "na",
                   f"min N/tau = {np.nanmin(n_over):.1f} ({srow[wn]['name']})" if np.isfinite(n_over).any() else "n/a",
                   "Good if the cold chains are at least 50 autocorrelation times long; check from 20; problem below 20."))
    checks.append(("rhat", "Rank-normalised split R-hat between the cold chains", _status(np.nanmax(rh), 1.01, 1.05) if np.isfinite(rh).any() else "na",
                   f"max R-hat = {np.nanmax(rh):.3f} ({srow[wr]['name']})" if np.isfinite(rh).any() else "needs two or more cold chains",
                   "Good below 1.01; check up to 1.05; problem above 1.05."))
    checks.append(("ess", "Effective sample size", _status(np.nanmin(ess_arr), ess_target, ess_target / 2.5, True) if np.isfinite(ess_arr).any() else "na",
                   f"min ESS = {np.nanmin(ess_arr):.0f} ({srow[we]['name']})" if np.isfinite(ess_arr).any() else "n/a",
                   f"Good at {ess_target} or more independent samples; check from {ess_target / 2.5:.0f}; problem below that."))
    checks.append(("drift", "Half-vs-half drift of the means", _status(np.nanmax(dz_arr), 3, 5) if np.isfinite(dz_arr).any() else "na",
                   f"max |z| = {np.nanmax(dz_arr):.1f} ({srow[wd]['name']})" if np.isfinite(dz_arr).any() else "n/a",
                   "Good if the two halves agree within 3 standard errors; check up to 5; problem above 5."))
    nf = len(an["flagged"])
    checks.append(("chains", "Cold chains that disagree or lag", "ok" if nf == 0 else ("warn" if nf / nE < 0.3 else "bad"), f"{nf} of {nE} flagged",
                   "A chain is flagged if its ln L is far below the others, its tau is more than 3 times the median, or its mean differs from the others by more than 5 standard errors."))
    if an["steps"] is not None:
        worst = min(an["steps"]["rows"], key=lambda r: r["share"] if np.isfinite(r["share"]) else 9)
        mx = max((r["share"] for r in an["steps"]["rows"] if np.isfinite(r["share"])), default=np.nan)
        checks.append(("steps", "Size of the adaptive Gaussian steps", _status(min(r["share"] for r in an["steps"]["rows"] if np.isfinite(r["share"])), 0.05, 0.01, True),
                       f"smallest step is {100 * worst['share']:.1f}% of the posterior sd ({worst['name']}); scale {an['steps']['scale']:.4g}",
                       "Good if every parameter moves at least 5% of its posterior width per accepted step; check from 1%. One shared scale is adapted for all parameters, so the parameter whose "
                       "posterior is narrowest compared with its prior sets the steps for all of them."))
    if an["walls"] is not None:
        wl = an["walls"]
        hard_names = ", ".join(nm for nm, k, s_ in zip(wl["names"], wl["boundary"], wl["share"]) if k == "None" and s_ > 0.01) or "none"
        walls_value = (f"{100 * wl['any_hard']:.0f}% leave a prior with no boundary rule ({hard_names}) and are rejected; "
                       f"{100 * (wl['any_wall'] - wl['any_hard']):.0f}% more leave only reflective ranges and are reflected back inside")
        checks.append(("walls", "Differential-evolution jumps rejected at a prior wall", _status(wl["any_hard"], 0.10, 0.30), walls_value,
                       "Good below 10%. A jump that leaves a prior without a boundary rule is rejected before the likelihood is evaluated (a wasted step). "
                       "A jump that leaves a reflective prior is reflected back inside and evaluated; a periodic one is wrapped (exact for any symmetric jump). Reflection is exact only for diagonal steps such as the adaptive Gaussian: "
                       "for differential-evolution and Fisher jumps, whose components are correlated, it can bias the posterior near the walls (see the check on reflective boundaries)."))
    ag0 = _first(run.sd[0][0], "AdaptiveGaussianProposal")
    refl = [k for k in run.names if ag0 is not None and str(ag0.prior_boundary_dict.get(k)) == "reflective"]
    corr_kinds = sorted({p["kind"] for p in props if p["kind"] in ("DE", "Fisher")})
    if refl and corr_kinds:
        near = sorted(zip(wl["share"], wl["names"]), reverse=True)[:3] if an["walls"] is not None else []
        checks.append(("reflect", "Reflective boundaries with correlated jump proposals", "warn",
                       f"{len(refl)} reflective parameters ({', '.join(refl[:5])}{', ...' if len(refl) > 5 else ''}) and the cycle has {' and '.join(corr_kinds)} proposals",
                       "Reflection is exact only if the jump distribution is symmetric under flipping each coordinate separately (a diagonal Gaussian). Differential-evolution and Fisher jumps have correlated components, "
                       "and then reflecting at a wall biases the posterior near it (on a bounded Gaussian: 0.04-0.08 on a mean with sd 0.2, against 0.002 for a hard wall). Hard walls (boundary None) are exact for every proposal."))
    de_row = next((p for p in props if p["kind"] == "DE" and p["n"] == len(run.names)), None)
    ag_row = next((p for p in props if p["kind"] == "AG" and p["n"] == len(run.names)), None)
    if ag_row is not None:
        a0 = ag_row["by_T"][0]
        checks.append(("ag", "Adaptive Gaussian acceptance (cold chain)", "ok" if 0.15 <= a0 <= 0.35 else ("warn" if 0.08 <= a0 <= 0.5 else "bad"), f"{a0:.2f}", "The scale is adapted to 0.234; good from 0.15 to 0.35."))
    if de_row is not None:
        a0 = de_row["by_T"][0]
        checks.append(("de", "Differential-evolution acceptance (cold chain)", _status(a0, 0.05, 0.02, True), f"{a0:.3f}", "Good from 5%; check from 2%. With many parameters moving at once DE is accepted rarely; block moves help."))
    wasted = [p for p in props if p["kind"] in ("KDE", "GMM") and np.isfinite(p["by_T"][0]) and p["count"] > 2000 and p["by_T"][0] < 0.005]
    if any(p["kind"] in ("KDE", "GMM") for p in props):
        checks.append(("learn", "Learning proposals (KDE, mixture) in the cold chain", "warn" if wasted else "ok",
                       (f"{len(wasted)} accepted below 0.5%: " + ", ".join(f"{p['kind']}({p['n']})" for p in wasted)) if wasted else "all accepted at least 0.5%",
                       "A proposal that is almost never accepted costs time (and is shipped to the workers every iteration). 'noKD' / 'noGM' remove them."))
    checks.append(("burn", "Burn-in", "ok" if bi < 0.5 * n else "warn", f"bilby discards {an['bilby_idx']:,} of {n:,} rows ({100 * an['bilby_idx'] / n:.0f}%); method {_methods_text(run)}; analysed from row {bi:,}",
                   "Good if less than half of the run is burn-in."))
    if run.adapting:
        checks.append(("adapt", "Temperature adaptation", "warn", "the ladder is still adapting", "The ladder is adapted during the first adapt_t0 x 10 rows and then fixed."))
    an["checks"] = [dict(key=k, label=l, status=s, value=v, rule=r) for k, l, s, v, r in checks]
    an["verdict"] = _worst(*[c[2] for c in checks])
    an["diagnosis"], an["suggestions"] = _bilby_diagnose(run, an)
    an["seconds"] = time.time() - t0
    return an


def _methods_text(run):
    ms = collections.Counter(b[2] for b in run.bilby_burn)
    return ", ".join(f"{m} ({c} chain{'s' if c > 1 else ''})" for m, c in ms.most_common())


def _bilby_diagnose(run, an):
    """Plain-language reading of the numbers and suggestions of what to try, as two lists of sentences (rule based)."""
    obs, tips = [], []
    n, bi, nE = run.rows, an["bi"], run.nE
    sw, srow = an["swaps"], [r for r in an["params"] if not r["derived"]]
    rec = np.array(sw["recorded"], dtype=float)
    ck = {c["key"]: c for c in an["checks"]}
    if "carry" in ck and ck["carry"]["status"] == "bad":
        obs.append("The temperature swaps do not carry the exchanged state forward: after most jumps of the stored cold chain the next row returns to the chain's own old state. "
                   "The hot chains then do not feed the cold one, and only the stored rows were rewritten.")
        tips.append("Install the swap fix (make_bilby_swaps_continue in run_sampler.py) and start a new run; this checkpoint was made while the swaps did not move the chains.")
    if len(rec) and (rec.min() < 0.2 or rec.max() - rec.min() > 0.3):
        i = int(np.argmin(rec))
        obs.append(f"The swap acceptances range from {rec.min():.2f} (T{i} to T{i + 1}) to {rec.max():.2f}; a replica needs about {sw['round_trip_rounds']:.0f} swap rounds, "
                   f"{sw['round_trip_rows']:,.0f} rows or {sw['round_trip_iterations']:,.0f} iterations, for a hot-to-cold-to-hot round trip.")
        ld = an.get("ladder_design")
        if ld is not None:
            tips.append(f"A ladder with the same hottest temperature and an equal predicted acceptance of {ld['target']:.2f} in every pair would need about {ld['round_trip_rounds']:.0f} swap rounds per round trip "
                        f"(betas: {', '.join(f'{b:.4f}' for b in ld['betas'])}). Pass them as \"initial_betas\" with \"adapt\": false: bilby's adaptation uses the local acceptance of each chain, not the swap acceptance.")
    sp = an["steps"]
    if sp is not None and sp["sigmas_all_one"]:
        worst = min(sp["rows"], key=lambda r: r["share"] if np.isfinite(r["share"]) else 9)
        if worst["share"] < 0.05:
            obs.append(f"The adaptive Gaussian scale is {sp['scale']:.4g}; at that scale {worst['name']} moves {100 * worst['share']:.1f}% of its posterior width per accepted step "
                       f"(prior width {worst['width']:.4g}, posterior sd {worst['post_sd']:.4g}). One shared scale is set by the parameter whose posterior is narrowest compared with its prior.")
            tips.append("Give the proposal per-parameter step factors (sigma_i = posterior width / prior width): the cycles tdinf_scaled, tdinf_blocks or tdinf_gw with \"tdinf_posterior_width\" "
                        "(this report prints the widths in the Proposals section).")
    wl = an["walls"]
    if wl is not None and wl["any_hard"] > 0.1:
        hard = sorted([(s_, nm) for s_, nm, k in zip(wl["share"], wl["names"], wl["boundary"]) if k == "None" and s_ > 0.01], reverse=True)
        hard_txt = ", ".join(f"{nm} {100 * s_:.0f}%" for s_, nm in hard)
        reflected = 100 * (wl["any_wall"] - wl["any_hard"])
        obs.append(f"{100 * wl['any_hard']:.0f}% of the differential-evolution jumps of the cold chain leave a prior that has no boundary rule ({hard_txt}) and are rejected before the likelihood is evaluated. "
                   f"A further {reflected:.0f}% leave only the range of parameters with a reflective boundary; those are reflected back inside and evaluated, which is exact for diagonal steps but can bias the posterior near the walls for "
                   "the correlated jumps of differential evolution and Fisher proposals (see the check on reflective boundaries).")
        if [nm for _, nm in hard] == ["mass_ratio"]:
            tips.append("The only parameter without a boundary rule is mass_ratio. A hard wall is exact for every proposal; the price is the rejected jumps. (Reflecting at q = 1 would recover them for the diagonal "
                        "adaptive Gaussian steps, but not for differential evolution or Fisher jumps.)")
        else:
            tips.append(f"Parameters without a boundary rule ({', '.join(nm for _, nm in hard)}) have a hard wall, which is exact for every proposal; the price is the rejected jumps. Reflective boundaries recover them "
                        "but are exact only for diagonal steps (the adaptive Gaussian); for differential-evolution and Fisher jumps they bias the posterior near the walls. bilby's own default priors use hard walls (boundary None) "
                        "for everything except the periodic angles.")
    ch = an["chains"]
    cold_tau = max((r["tau"] for r in srow if np.isfinite(r["tau"])), default=float("nan"))
    rt_rows = sw["round_trip_rows"]
    if np.isfinite(cold_tau) and np.isfinite(rt_rows) and rt_rows > 0 and cold_tau >= 0.4 * rt_rows:
        tm = an.get("ladder_cool", None)
        decor = [T for T in range(run.nT) if max(v for v in an["tau_by_T"][T].values() if np.isfinite(v)) <= 12]
        obs.append(f"The slowest autocorrelation time of the cold chains ({cold_tau:.0f} rows) is {100 * cold_tau / rt_rows:.0f}% of the replica round trip ({rt_rows:,.0f} rows): a state has to travel up the ladder and back to "
                   "decorrelate, so the tempering sets the pace of the cold chain. Better local moves would show up in the warm chains, not in the cold one."
                   + (f" Every parameter decorrelates within 12 rows from T{decor[0]} (beta {run.betas[decor[0]]:.2f}) upwards, so the hottest temperatures are not needed to decorrelate." if decor else ""))
        if tm is not None:
            tips.append(f"A cooler ladder shortens the round trip: {tm['ntemps']} temperatures down to beta {tm['beta_min']:.3f} (T {1 / tm['beta_min']:.1f}) with an equal predicted swap acceptance of {np.mean(tm['predicted']):.2f} would need about "
                        f"{tm['round_trip_rounds']:.0f} swap rounds per round trip instead of {sw['round_trip_rounds']:.0f} (betas: {', '.join(f'{b:.3f}' for b in tm['betas'])}). Pass them as \"initial_betas\" with \"adapt\": false. "
                        "Check first that the posterior has no mode that only the hottest chains can reach: compare how often the cold and the warm chains switch between the modes you know of (for example the sky or the inclination modes).")
        tips.append(f"Swapping more often also shortens it: the round trip is {sw['round_trip_iterations']:,.0f} iterations at L2steps = {run.L2}. \"L2steps\": 1 gives up to {run.L2} times more swap attempts per iteration, but the ensemble step that shares the counter "
                    f"evaluates the likelihood once per chain, one after another in the main process: with {run.nT * run.nE} chains that adds up to {100 * run.nT * run.nE / (run.L1 * 1):.0f}% of an iteration at L2steps = 1 "
                    f"(when all chains run in parallel; less with fewer workers than chains). Compare seconds per iteration on a short run.")
    elif run.L2 > 1:
        tips.append(f"Swaps happen every {run.L2} iterations (L2steps = {run.L2}). \"L2steps\": 1 swaps every iteration, at the price of an ensemble step (one serial likelihood evaluation per chain) every iteration.")
    sdr = an["slow_dirs"][0]
    top = sorted(sdr["loadings"].items(), key=lambda kv: -abs(kv[1]))[:4]
    obs.append(f"The slowest direction in parameter space (tau about {sdr['tau']:.0f} rows) is a combination: " + ", ".join(f"{k} {c:+.2f}" for k, c in top) + ".")
    n_over = [(r["n_over_tau"], r["name"]) for r in srow if np.isfinite(r["n_over_tau"])]
    if n_over and min(n_over)[0] < 50:
        v, nm = min(n_over)
        obs.append(f"The cold chains are only {v:.1f} autocorrelation times long for {nm} (50 or more is wanted).")
        taus = [(r["tau"], r["name"]) for r in srow if np.isfinite(r["tau"])]
        tau_max, nm2 = max(taus)
        need = 50 * tau_max
        tips.append(f"Run longer: the slowest autocorrelation time is about {tau_max:,.0f} rows ({nm2}); 50 of them need about {need:,.0f} rows after the burn-in, roughly {max(need - an['n_post'], 0):,.0f} more than you have. "
                    "tau is itself uncertain while the chains are short, so treat this as a minimum.")
    tc = an["tau_curve"]
    if tc is not None:
        nm = tc["names"][0]
        t_ = np.array(tc["tau"][nm], dtype=float)
        if len(t_) >= 4 and np.isfinite(t_[-1]) and t_[-1] > 1.3 * np.nanmedian(t_[len(t_) // 2:len(t_) // 2 + 2]):
            obs.append(f"tau of {nm} is still growing with the length of the chain ({t_[len(t_) // 2]:.0f} at half the post-burn-in length, {t_[-1]:.0f} at the end), so it has not levelled off.")
    if an["flagged"]:
        obs.append(f"{len(an['flagged'])} of {nE} cold chains are flagged ({', '.join('E' + str(i) + ' ' + ch[i]['flags'] for i in an['flagged'])}); see the Chains section.")
    rh = [r["rhat"] for r in srow if np.isfinite(r["rhat"])]
    if rh and max(rh) > 1.05:
        tips.append("R-hat is above 1.05: the cold chains disagree. Look at the histograms per chain; a posterior with separate modes can look like this when chains sit in different ones.")
    wasted = [c for c in an["checks"] if c["key"] == "learn" and c["status"] == "warn"]
    if wasted:
        tips.append("Some learning proposals are almost never accepted in the cold chain; try the cycle name with 'noKD' (or 'noGM') to drop them and save the time they cost per iteration.")
    etf = an.get("etf")
    if etf is not None and etf["remaining"] > 0:
        obs.append(f"By the cold chains' own tau the run has about {an['counts']['n_est']:,.0f} of the {an['counts']['target']:,} samples asked for (bilby counts {an['counts']['n_now']:,.0f}); "
                   f"finishing needs about {etf['rows']:,.0f} more rows, {etf['hours']:.1f} h at the present speed. bilby's own printed estimate assumes one chain makes all the remaining samples "
                   f"({etf['bilby_hours']:.1f} h here), so with {nE} cold chains it reads too high.")
    if not obs:
        obs.append("Nothing stands out in the automated checks.")
    if not tips:
        tips.append("No action needed. As a final check, repeat the run with a different seed and compare the posteriors.")
    return obs, tips


# ============================================================================== plots

def _col(E):
    return plt.cm.tab10(E % 10)


def _tname(T):
    return f"T{T}"


def plot_logl(run, an):
    """ln L of the cold chains against the row, and the distribution of each chain after the burn-in"""
    with _style():
        fig, axs = plt.subplots(1, 2, figsize=(12.8, 4.3), gridspec_kw=dict(width_ratios=[2.3, 1]))
        L = run.logl_cold()
        n = L.shape[0]
        idx = _plot_idx(n)
        ax = axs[0]
        for E in range(run.nE):
            ax.plot(idx, L[idx, E], lw=0.7, alpha=0.7, color=_col(E), label=f"E{E}" if run.nE <= 8 else None)
        ax.plot(idx, np.median(L, axis=1)[idx], color=C["ink"], lw=1.5, label="median of the cold chains")
        ax.axvline(an["bilby_idx"], ls="--", color=C["burn"], lw=1.3, label=f"bilby's burn-in ({an['bilby_idx']:,})")
        if an["est_found"] and an["est_idx"] > 0:
            ax.axvline(an["est_idx"], ls=":", color=C["green"], lw=1.5, label=f"ln L plateau estimate ({an['est_idx']:,})")
        elif an["est_found"]:
            ax.plot([], [], ls=":", color=C["green"], label="ln L plateau reached within the first block")
        lo, hi = _plateau_ylim(L[max(an["bi"], an["est_idx"]):])
        ax.set_ylim(lo, hi)
        ax.set_xlabel("row")
        ax.set_ylabel("ln L")
        ax.set_title("Log-likelihood of the cold chains")
        ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.17), fontsize=8)
        ax = axs[1]
        post = L[an["bi"]:]
        lo, hi = _plateau_ylim(post)
        bins = np.linspace(lo, hi, 50)
        for E in range(run.nE):
            ax.hist(post[:, E], bins=bins, histtype="step", density=True, color=_col(E), lw=1.0, alpha=0.8)
        ax.hist(post.reshape(-1), bins=bins, histtype="step", density=True, color=C["ink"], lw=1.8)
        ax.set_xlabel("ln L (after the burn-in)")
        ax.set_title("Per chain (colour) and pooled (black)")
        fig.tight_layout()
    return fig


def plot_ladder(run, an):
    """swap acceptance per pair (recorded and predicted), ln L distribution per temperature, and the ladder with its more even alternative"""
    with _style():
        fig, axs = plt.subplots(1, 3, figsize=(14.5, 4.4))
        sw, nT = an["swaps"], run.nT
        x = np.arange(nT - 1)
        ax = axs[0]
        ax.axhspan(0.2, 0.8, color=C["green"], alpha=0.10, lw=0)
        ax.bar(x - 0.18, sw["recorded"], 0.36, color=C["blue"], label="recorded")
        ax.bar(x + 0.18, sw["predicted"], 0.36, color=C["light"], label="predicted from the stored ln L")
        if an.get("ladder_design") is not None:
            ax.plot(x, an["ladder_design"]["predicted"], "o--", color=C["orange"], ms=4, lw=1.2, label="more even ladder (predicted)")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{i}-{i + 1}" for i in x])
        ax.set_ylim(0, 1)
        ax.set_xlabel("pair of temperatures")
        ax.set_ylabel("swap acceptance")
        ax.set_title("Swap acceptance (green: 0.2 to 0.8)")
        ax.legend(fontsize=8, loc="upper left")
        ax = axs[1]
        L = an["lnl_by_T"]
        for T in range(nT):
            v = np.percentile(L[T], [5, 16, 50, 84, 95])
            ax.plot([T, T], [v[0], v[4]], color=C["light"], lw=3)
            ax.plot([T, T], [v[1], v[3]], color=C["blue"], lw=6, solid_capstyle="butt")
            ax.plot([T], [v[2]], "o", color="white", ms=4, mec=C["ink"])
        ax.set_xticks(range(nT))
        ax.set_xticklabels([f"T{T}\n{run.betas[T]:.2f}" for T in range(nT)], fontsize=8)
        ax.set_ylabel("ln L")
        ax.set_title("ln L per temperature (5-95%, 16-84%, median)")
        ax = axs[2]
        ax.plot(range(nT), run.betas, "o-", color=C["blue"], label="this run")
        if an.get("ladder_design") is not None:
            ax.plot(range(nT), an["ladder_design"]["betas"], "o--", color=C["orange"], label="equal predicted acceptance")
        ax.set_yscale("log")
        ax.set_xlabel("temperature index")
        ax.set_ylabel("beta = 1 / T")
        ax.set_title("The ladder")
        ax.legend(fontsize=8)
        fig.tight_layout()
    return fig


def bilby_plot_burnin(run, an):
    with _style():
        fig = plt.figure(figsize=(12.5, 7.4))
        gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1])
        ax = fig.add_subplot(gs[0, :])
        L = run.logl_cold()
        n = L.shape[0]
        med = np.median(L, axis=1)
        idx = _plot_idx(n, 1500)
        ax.plot(idx + 1, med[idx], color=C["blue"], lw=1.0, label="median ln L of the cold chains")
        info = an["est_info"]
        ax.axhline(info["ref"], color=C["green"], lw=1.0, ls="--", label="plateau")
        ax.axhspan(info["ref"] - info["tol"], info["ref"] + info["tol"], color=C["green"], alpha=0.12, lw=0, label="tolerance")
        ax.axvline(max(an["bilby_idx"], 1), color=C["burn"], ls="--", lw=1.3, label=f"bilby's burn-in ({an['bilby_idx']:,})")
        if an["est_found"] and an["est_idx"] > 0:
            ax.axvline(an["est_idx"], color=C["green"], ls=":", lw=1.5, label=f"plateau estimate ({an['est_idx']:,})")
        elif an["est_found"]:
            ax.plot([], [], color=C["green"], ls=":", label="plateau reached within the first block")
        ax.set_xscale("log")
        late = med[int(0.75 * n):]
        sd = max(robust_sd(late), 1e-6)
        ax.set_ylim(info["ref"] - 12 * max(sd, info["tol"]), info["ref"] + 5 * max(sd, info["tol"]))
        ax.set_xlabel("row (log axis)")
        ax.set_ylabel("ln L")
        ax.set_title("Approach to the plateau")
        ax.legend(ncol=3, fontsize=8, loc="lower right")
        scan = an["burn_scan"]
        if scan:
            xs = [s["rows"] for s in scan]
            specs = [("rhat", "worst R-hat (largest)", 1.01, 1.05, False), ("n_over_tau", "worst N / tau (smallest)", 50, 20, True), ("drift", "worst |drift z| (largest)", 3, 5, False)]
            for k, (key, label, good, badv, hib) in enumerate(specs):
                a2 = fig.add_subplot(gs[1, k])
                ys = [s[key] for s in scan]
                a2.plot(xs, ys, "o-", color=C["blue"])
                a2.axhline(good, color=C["green"], ls="--", lw=1.1)
                a2.axhline(badv, color=C["red"], ls=":", lw=1.3)
                a2.set_xlabel("rows removed from the start")
                a2.set_title(label, fontsize=10)
                if key == "n_over_tau":
                    a2.set_yscale("log")
        fig.tight_layout()
    return fig


def plot_proposals(run, an):
    """acceptance and scale of every proposal against the temperature"""
    with _style():
        fig, axs = plt.subplots(1, 2, figsize=(13, 4.4))
        beta = run.betas
        props = an["proposals"]
        ax = axs[0]
        for k, p in enumerate(props):
            y = np.array(p["by_T"], dtype=float)
            if np.isfinite(y).any():
                ax.plot(beta, y, "o-", ms=3.5, lw=1.2, color=plt.cm.tab20(k % 20), label=f"{p['kind']} ({p['n']})")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.invert_xaxis()
        ax.set_xlabel("beta (cold chain on the left)")
        ax.set_ylabel("acceptance")
        ax.set_title("Acceptance of each proposal")
        ax.legend(fontsize=7, ncol=2)
        ax = axs[1]
        for k, p in enumerate(props):
            y = np.array(p["scale"], dtype=float)
            if np.isfinite(y).any():
                ax.plot(beta, y, "o-", ms=3.5, color=plt.cm.tab20(k % 20), label=f"{p['kind']} ({p['n']})")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.invert_xaxis()
        ax.set_xlabel("beta (cold chain on the left)")
        ax.set_ylabel("scale")
        ax.set_title("Scale of the adaptive Gaussian proposals")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
        fig.tight_layout()
    return fig


def plot_steps(run, an):
    sp = an["steps"]
    if sp is None:
        return None
    with _style():
        rows = sorted(sp["rows"], key=lambda r: r["share"] if np.isfinite(r["share"]) else 9)
        fig, ax = plt.subplots(figsize=(8.5, 0.34 * len(rows) + 1.4))
        y = np.arange(len(rows))
        v = [100 * r["share"] for r in rows]
        ax.barh(y, v, color=[C["red"] if s < 1 else C["amber"] if s < 5 else C["green"] for s in v])
        ax.set_yticks(y)
        ax.set_yticklabels([r["name"] for r in rows], fontsize=9)
        ax.set_xscale("log")
        ax.axvline(1, color=C["red"], ls=":", lw=1.3)
        ax.axvline(5, color=C["green"], ls="--", lw=1.1)
        ax.set_xlabel("one accepted adaptive-Gaussian step as a share of the posterior sd (%)")
        ax.set_title(f"Step sizes of the cold chain (scale {sp['scale']:.4g})")
        fig.tight_layout()
    return fig


def plot_walls(run, an):
    wl = an["walls"]
    if wl is None:
        return None
    with _style():
        order = np.argsort(wl["share"])[::-1]
        fig, ax = plt.subplots(figsize=(8.5, 0.34 * len(order) + 1.4))
        colours = {"None": C["red"], "reflective": C["green"], "periodic": C["gray"]}
        ax.barh(np.arange(len(order)), [100 * wl["share"][i] for i in order], color=[colours.get(wl["boundary"][i], C["amber"]) for i in order])
        ax.set_yticks(np.arange(len(order)))
        ax.set_yticklabels([f"{wl['names'][i]} ({wl['boundary'][i]})" for i in order], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("share of differential-evolution jumps that leave this parameter's prior (%)")
        ax.set_title("Prior walls: red = no boundary (the jump is rejected), green = reflective (the jump is reflected back)")
        fig.tight_layout()
    return fig


def plot_offsets(run, an):
    with _style():
        off = an["offsets"]
        names = [r["name"] for r in an["params"] if not r["derived"]]
        fig, ax = plt.subplots(figsize=(max(7, 0.55 * len(names) + 2), 0.5 * run.nE + 2.2))
        lim = max(6.0, float(np.nanmax(np.abs(off))))
        im = ax.imshow(off, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(run.nE))
        ax.set_yticklabels([f"E{E}" for E in range(run.nE)])
        ax.grid(False)
        for E in range(run.nE):
            for j in range(len(names)):
                if abs(off[E, j]) > 3:
                    ax.text(j, E, f"{off[E, j]:.0f}", ha="center", va="center", fontsize=7, color="white" if abs(off[E, j]) > 0.6 * lim else "black")
        fig.colorbar(im, ax=ax, label="mean of the chain minus the pooled mean, in standard errors")
        ax.set_title("Do the cold chains agree? (values above 5 are flagged)")
        fig.tight_layout()
    return fig


def bilby_plot_traces(run, an, names, per_page=5):
    pages = []
    bi = an["bi"]
    n = run.rows
    cen = an["centres"]
    X0 = run.stack(0, 0, run.centres(0), names=None, derived=True)
    names = [k for k in names if k in X0]
    idx = _plot_idx(n)
    for s in range(0, len(names), per_page):
        chunk = names[s:s + per_page]
        with _style():
            fig, axs = plt.subplots(len(chunk), 2, figsize=(13, 2.1 * len(chunk) + 0.5), gridspec_kw=dict(width_ratios=[3.4, 1]), squeeze=False)
            for r, nm in enumerate(chunk):
                x = X0[nm]
                label = nm
                if nm == "geocenter_time":
                    x, label = (x - np.mean(x)) * 1e3, "geocenter_time - mean (ms)"
                ax = axs[r, 0]
                for E in range(run.nE):
                    ax.plot(idx, x[idx, E], lw=0.6, alpha=0.7, color=_col(E))
                ax.plot(idx, np.median(x, axis=1)[idx], color=C["ink"], lw=1.3)
                ax.axvline(bi, color=C["burn"], ls="--", lw=1.2)
                lo, hi = _robust_ylim(x[bi:])
                ax.set_ylim(lo, hi)
                ax.set_ylabel(label, fontsize=9)
                if r == len(chunk) - 1:
                    ax.set_xlabel("row")
                post = x[bi:]
                h = len(post) // 2
                bins = np.linspace(lo, hi, 36)
                ax = axs[r, 1]
                ax.hist(post[:h].reshape(-1), bins=bins, density=True, histtype="step", color=C["blue"], lw=1.4)
                ax.hist(post[h:].reshape(-1), bins=bins, density=True, histtype="step", color=C["orange"], lw=1.4)
                ax.set_yticks([])
            fig.tight_layout()
        pages.append(fig)
    return pages


def bilby_plot_acf(run, an, nparam=6):
    names = (an["tau_curve"] or {}).get("names") or [r["name"] for r in an["params"] if not r["derived"]][:nparam]
    names = names[:nparam]
    X = an["X"]
    with _style():
        fig, axs = plt.subplots(2, 3, figsize=(13, 6.2))
        for ax, nm in zip(axs.ravel(), names):
            x = _fill_nan(X[nm])
            tau = integrated_autocorr_time(x)
            K = int(min(len(x) - 1, max(60, 6 * (tau if np.isfinite(tau) else 50))))
            for E in range(run.nE):
                a, _ = mean_acf(x[:, [E]])
                if a is not None:
                    ax.plot(a[:K], lw=0.8, alpha=0.6, color=_col(E))
            a, _ = mean_acf(x)
            if a is not None:
                ax.plot(a[:K], color=C["ink"], lw=1.7)
            ax.axhline(0, color=C["gray"], lw=0.8)
            if np.isfinite(tau) and tau < K:
                ax.axvline(tau, color=C["burn"], ls="--", lw=1.1)
            ax.set_title(f"{nm}: tau {tau:.0f}", fontsize=10)
            ax.set_xlabel("lag (rows)")
        for ax in axs.ravel()[len(names):]:
            ax.axis("off")
        fig.tight_layout()
    return fig


def plot_history(run, an):
    """how n (bilby's count) and tau changed during the run, and tau against chain length as computed here"""
    with _style():
        fig, axs = plt.subplots(1, 3, figsize=(14.5, 4.1))
        ax = axs[0]
        if run.hist_n:
            xs, ys = zip(*run.hist_n)
            ax.plot(xs, ys, color=C["blue"])
            if run.target:
                ax.axhline(run.target, color=C["green"], ls="--", lw=1.1, label=f"target {run.target:,}")
                ax.legend(fontsize=8)
        ax.set_xlabel("row")
        ax.set_ylabel("n (bilby's count over the cold chains)")
        ax.set_title("Samples counted by bilby")
        ax = axs[1]
        tau_hist = [(p, t) for p, t in run.hist_tau if np.isfinite(t)]
        if tau_hist:
            xs, ys = zip(*tau_hist)
            ax.plot(xs, ys, color=C["orange"])
            ax.set_yscale("log")
        ax.set_xlabel("row")
        ax.set_ylabel("max tau over the parameters (rows)")
        ax.set_title("tau as bilby saw it (primary chain)")
        ax = axs[2]
        tc = an["tau_curve"]
        if tc is not None:
            for k, nm in enumerate(tc["names"]):
                ax.plot(tc["N"], tc["tau"][nm], "o-", ms=3, lw=1.2, label=nm)
            ax.plot(tc["N"], tc["N"] / 50.0, "k--", lw=1.0, label="N / 50")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.legend(fontsize=7, ncol=2)
        ax.set_xlabel("rows of the chain used")
        ax.set_ylabel("tau (rows)")
        ax.set_title("tau against chain length (after the burn-in)")
        fig.tight_layout()
    return fig


def plot_tau_by_T(run, an):
    sampled = [r["name"] for r in an["params"] if not r["derived"]]
    M = np.array([[an["tau_by_T"][T][k] for T in range(run.nT)] for k in sampled], dtype=float)
    with _style():
        fig, axs = plt.subplots(1, 2, figsize=(13, 0.34 * len(sampled) + 2.2), gridspec_kw=dict(width_ratios=[1.6, 1]))
        ax = axs[0]
        im = ax.imshow(np.log10(np.maximum(M, 1.0)), aspect="auto", cmap="viridis_r")
        ax.set_xticks(range(run.nT))
        ax.set_xticklabels([f"T{T}\n{run.betas[T]:.2f}" for T in range(run.nT)], fontsize=8)
        ax.set_yticks(range(len(sampled)))
        ax.set_yticklabels(sampled, fontsize=8)
        ax.grid(False)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=7, color="white" if np.log10(max(M[i, j], 1)) < 0.5 * np.nanmax(np.log10(np.maximum(M, 1))) else "black")
        fig.colorbar(im, ax=ax, label="log10 tau (rows)")
        ax.set_title("tau of every parameter at every temperature")
        ax = axs[1]
        d = an["tau_dir_by_T"]
        ax.plot(range(run.nT), [d[T] for T in range(run.nT)], "o-", color=C["orange"], label="slowest direction")
        ax.plot(range(run.nT), np.nanmax(M, axis=0), "s--", color=C["blue"], label="slowest parameter")
        ax.set_yscale("log")
        ax.set_xticks(range(run.nT))
        ax.set_xticklabels([f"T{T}" for T in range(run.nT)])
        ax.set_ylabel("tau (rows)")
        ax.set_title("tau against temperature")
        ax.legend(fontsize=8)
        fig.tight_layout()
    return fig


def plot_slow_direction(run, an):
    sd = an["slow_dirs"]
    with _style():
        fig, axs = plt.subplots(1, len(sd), figsize=(5.2 * len(sd), 0.3 * len(sd[0]["loadings"]) + 1.8), squeeze=False)
        for ax, d in zip(axs[0], sd):
            items = sorted(d["loadings"].items(), key=lambda kv: abs(kv[1]))
            ax.barh([k for k, _ in items], [v for _, v in items], color=[C["blue"] if v > 0 else C["orange"] for _, v in items])
            ax.axvline(0, color=C["gray"], lw=0.8)
            ax.set_title(f"tau {d['tau']:.0f} rows", fontsize=10)
            ax.tick_params(axis="y", labelsize=8)
        fig.suptitle("Slowest directions of the posterior (loadings on the standardised parameters)", fontsize=11, fontweight="bold")
        fig.tight_layout()
    return fig


def plot_hist(run, an, names=None, ncols=4):
    X = an["X"]
    names = names or [r["name"] for r in an["params"] if not r["derived"]]
    names = [k for k in names if k in X]
    nr = int(np.ceil(len(names) / ncols))
    with _style():
        fig, axs = plt.subplots(nr, ncols, figsize=(3.2 * ncols, 2.3 * nr), squeeze=False)
        for ax, nm in zip(axs.ravel(), names):
            x = X[nm]
            label = nm
            if nm == "geocenter_time":
                x, label = (x - np.mean(x)) * 1e3, "geocenter_time - mean (ms)"
            lo, hi = _robust_ylim(x.reshape(-1), 0.2, 99.8, 0.02)
            bins = np.linspace(lo, hi, 40)
            for E in range(run.nE):
                ax.hist(x[:, E], bins=bins, density=True, histtype="step", color=_col(E), lw=0.8, alpha=0.8)
            ax.hist(x.reshape(-1), bins=bins, density=True, histtype="step", color=C["ink"], lw=1.6)
            ax.set_title(label, fontsize=9)
            ax.set_yticks([])
            ax.tick_params(axis="x", labelsize=8)
        for ax in axs.ravel()[len(names):]:
            ax.axis("off")
        fig.tight_layout()
    return fig


# ============================================================================== HTML report

def _status_cls(v, ok, warn, higher=False):
    return _status(v, ok, warn, higher)


def _bilby_explorer_payload(run, an, max_pts=500):
    """the data of the interactive trace explorer (the same layout as in the emcee report: the cold chains play the part of the walkers)"""
    n = run.rows
    idx = _plot_idx(n, max_pts)
    X = run.stack(0, 0, run.centres(0), derived=True)
    X["ln_likelihood"] = np.column_stack([run.arrays[(0, E)][:n, run.il] for E in range(run.nE)])
    flagged = [int(i) for i in an["flagged"]]
    walkers = flagged + [E for E in range(run.nE) if E not in flagged]
    series = {}
    for nm, x in X.items():
        x = np.asarray(x, dtype=np.float64)
        q = _nanq(x[idx], [0.05, 0.16, 0.5, 0.84, 0.95], axis=1)
        series[nm] = {"q": [_sig(r) for r in q], "w": [_sig(x[idx, E]) for E in walkers]}
    return {"steps": [int(v) for v in idx], "burn": int(an["bi"]), "walkers": walkers, "n_flagged": len(flagged), "series": series}


def _fnum(x, spec=".4g"):
    if x is None or not np.isfinite(x):
        return "n/a"
    return format(x, spec)


def _bilby_params_table(an):
    head = ["parameter", "mean", "std", "5%", "median", "95%", "R-hat", "tau", "tau per chain (min - max)", "N/tau", "ESS", "drift z", "status"]
    out = ['<div class="scroll"><table class="sortable"><thead><tr>' + "".join(f"<th>{h}</th>" for h in head) + "</tr></thead><tbody>"]
    for r in an["params"]:
        nm = r["name"] + (" (derived)" if r["derived"] else "")
        out.append("<tr>" + _cell(nm) + _cell(_fmt(r["mean"]), val=r["mean"]) + _cell(_fmt(r["std"]), val=r["std"]) + _cell(_fmt(r["q05"]), val=r["q05"]) + _cell(_fmt(r["q50"]), val=r["q50"]) +
                   _cell(_fmt(r["q95"]), val=r["q95"]) + _cell(_fmt(r["rhat"]), r["st_rhat"], r["rhat"]) + _cell(_fmt(r["tau"]), val=r["tau"]) +
                   _cell(f"{_fmt(r['tau_min'])} - {_fmt(r['tau_max'])}", val=r["tau_max"]) + _cell(_fmt(r["n_over_tau"]), r["st_tau"], r["n_over_tau"]) +
                   _cell(_fmt(r["ess"]), r["st_ess"], r["ess"]) + _cell(_fmt(r["drift_z"]), r["st_drift"], r["drift_z"]) + _cell(r["status"].upper(), r["status"]) + "</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


def _chains_table(an):
    head = ["chain", "flags", "median ln L", "offset from the others (spreads)", "max ln L", "slowest tau (rows)", "of", "n (tau-based, bilby's counting)", "largest offset of a mean (std. errors)", "of",
            "bilby's burn-in (row)", "by", "local acceptance"]
    out = ['<div class="scroll"><table class="sortable"><thead><tr>' + "".join(f"<th>{h}</th>" for h in head) + "</tr></thead><tbody>"]
    for c in an["chains"]:
        cls = "bad" if c["flags"] else ""
        out.append("<tr>" + _cell(f"E{c['id']}", val=c["id"]) + _cell(c["flags"] or "-", cls) + _cell(_fmt(c["median_lnl"]), val=c["median_lnl"]) + _cell(f"{c['gap']:+.2f}", val=c["gap"]) +
                   _cell(_fmt(c["max_lnl"]), val=c["max_lnl"]) + _cell(_fmt(c["tau_max"]), val=c["tau_max"]) + _cell(c["tau_param"]) + _cell(_fmt(c["n_eff"]), val=c["n_eff"]) +
                   _cell(_fmt(c["offset_max"]), "bad" if c["offset_max"] > 5 else "", c["offset_max"]) + _cell(c["offset_param"]) + _cell(f"{c['burn']:,}", val=c["burn"]) + _cell(c["burn_method"]) +
                   _cell(f"{c['local_accept']:.3f}", val=c["local_accept"]) + "</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


def _temps_table(run, an):
    head = ["T", "beta", "1 / beta", "swap to the next (recorded)", "predicted", "ln L median", "ln L sd", "slowest parameter tau", "slowest direction tau", "AG scale", "AG acceptance", "DE acceptance"]
    out = ['<div class="scroll"><table class="sortable"><thead><tr>' + "".join(f"<th>{h}</th>" for h in head) + "</tr></thead><tbody>"]
    ag = next((p for p in an["proposals"] if p["kind"] == "AG" and p["n"] == len(run.names)), None)
    de = next((p for p in an["proposals"] if p["kind"] == "DE" and p["n"] == len(run.names)), None)
    rec, pred = an["swaps"]["recorded"], an["swaps"]["predicted"]
    for T in range(run.nT):
        L = an["lnl_by_T"][T]
        tmax = max((v for v in an["tau_by_T"][T].values() if np.isfinite(v)), default=float("nan"))
        r_ = rec[T] if T < len(rec) else float("nan")
        cls = ("ok" if 0.2 <= r_ <= 0.8 else "warn" if r_ >= 0.1 else "bad") if np.isfinite(r_) else ""
        out.append("<tr>" + _cell(f"T{T}", val=T) + _cell(f"{run.betas[T]:.4f}", val=run.betas[T]) + _cell(f"{1 / run.betas[T]:.3f}", val=1 / run.betas[T]) + _cell(_fnum(r_, ".3f"), cls, r_) +
                   _cell(_fnum(pred[T] if T < len(pred) else float("nan"), ".3f")) + _cell(_fmt(float(np.median(L))), val=float(np.median(L))) + _cell(_fmt(float(np.std(L))), val=float(np.std(L))) +
                   _cell(_fmt(tmax), val=tmax) + _cell(_fmt(an["tau_dir_by_T"][T]), val=an["tau_dir_by_T"][T]) + _cell(_fnum(ag["scale"][T] if ag else float("nan")), val=ag["scale"][T] if ag else None) +
                   _cell(_fnum(ag["by_T"][T] if ag else float("nan"), ".3f")) + _cell(_fnum(de["by_T"][T] if de else float("nan"), ".3f")) + "</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


def _proposals_table(run, an):
    head = ["#", "proposal", "parameters", "weight", "proposals made (cold chain)"] + [f"T{T} ({run.betas[T]:.2f})" for T in range(run.nT)]
    out = ['<div class="scroll"><table class="sortable"><thead><tr>' + "".join(f"<th>{h}</th>" for h in head) + "</tr></thead><tbody>"]
    for p in an["proposals"]:
        cells = []
        for a in p["by_T"]:
            cls = ""
            if p["kind"] in ("KDE", "GMM") and np.isfinite(a) and a < 0.005 and p["count"] > 2000:
                cls = "warn"
            cells.append(_cell(_fnum(a, ".3f") if np.isfinite(a) else "n/a", cls, a))
        out.append("<tr>" + _cell(p["index"], val=p["index"]) + _cell(p["kind"]) + _cell(p["n"], val=p["n"]) + _cell(_fnum(p["weight"], "g"), val=p["weight"]) + _cell(f"{p['count']:,}", val=p["count"]) + "".join(cells) + "</tr>")
    out.append("</tbody></table></div>")
    if an["stretch"] is not None:
        out.append(f'<p class="muted">The ensemble stretch move (between the chains of one temperature, every {run.L2} iterations): accepted {100 * an["stretch"]["acc"]:.1f}% of {an["stretch"]["count"]:,} proposals, '
                   "counted over all chains.</p>")
    return "".join(out)


def _steps_table(an):
    sp = an["steps"]
    if sp is None:
        return ""
    head = ["parameter", "prior width used", "sigma_i", "step of one accepted move", "posterior sd", "step / sd", "status"]
    out = ['<div class="scroll"><table class="sortable"><thead><tr>' + "".join(f"<th>{h}</th>" for h in head) + "</tr></thead><tbody>"]
    for r in sorted(sp["rows"], key=lambda r: r["share"] if np.isfinite(r["share"]) else 9):
        st = _status(r["share"], 0.05, 0.01, True)
        out.append("<tr>" + _cell(r["name"]) + _cell(_fmt(r["width"]), val=r["width"]) + _cell(_fmt(r["sigma"]), val=r["sigma"]) + _cell(_fmt(r["step"]), val=r["step"]) + _cell(_fmt(r["post_sd"]), val=r["post_sd"]) +
                   _cell(f"{100 * r['share']:.2f}%", st, r["share"]) + _cell({"ok": "OK", "warn": "CHECK", "bad": "SMALL", "na": "n/a"}[st], st) + "</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


def _widths_json(an):
    sampled = [r for r in an["params"] if not r["derived"]]
    return json.dumps({r["name"]: float(f"{r['std']:.3g}") for r in sampled})


def _bilby_stat_guide_html():
    rows = [("R-hat", "Do the cold chains describe the same distribution? Compares the spread between chains with the spread inside each chain; it is 1 when they agree.", "close to 1, below 1.01", "above 1.05"),
            ("N / tau", "How long a cold chain is in units of the autocorrelation time tau, the number of rows it needs to forget where it was. Roughly the number of independent samples per chain.", "50 or more", "below 20"),
            ("drift z", "Has the chain stopped moving? The difference between the averages of the first and second half of the chain, in standard errors.", "within about 3 (random scatter around 0)", "above 5"),
            ("swap acceptance", "How often two neighbouring temperatures exchange their states. Too low: the hot chains cannot feed the cold one. Too high: the temperatures are closer than they need to be.", "0.2 to 0.8", "below 0.1")]
    out = ['<table><thead><tr><th>statistic</th><th style="text-align:left">what it tells you</th><th>good</th><th>worrying</th></tr></thead><tbody>']
    for name, what, good, bad in rows:
        out.append(f'<tr><td><b>{name}</b></td><td style="text-align:left;white-space:normal">{what}</td><td class="ok">{good}</td><td class="bad">{bad}</td></tr>')
    out.append("</tbody></table>")
    return "".join(out)


def _arg_table(rows, head=("setting", "what it is", "value")):
    t = ['<table class="args"><colgroup><col style="width:27%"><col style="width:40%"><col style="width:33%"></colgroup>' f"<thead><tr><th>{head[0]}</th><th>{head[1]}</th><th>{head[2]}</th></tr></thead><tbody>"]
    for a, m, v in rows:
        t.append(f"<tr><td>{_html.escape(str(a))}</td><td>{_html.escape(str(m))}</td><td>{_html.escape(str(v))}</td></tr>")
    t.append("</tbody></table>")
    return "".join(t)


def _bilby_fmt_arg(v):
    if isinstance(v, (list, tuple)):
        return " ".join(_bilby_fmt_arg(x) for x in v)
    if isinstance(v, bool) or v is None:
        return str(v)
    if isinstance(v, float):
        return f"{v:.10g}"
    return str(v)


def _settings_section(run, an, arguments=None, command_line=None, sampler_kwargs=None):
    out = ['<section id="settings"><h2>Settings</h2><p class="muted">Read from the resume file: what bilby_mcmc was set up with, and what the run did so far.</p>']
    cv, pi = run.conv, run.ptin
    ladder = [("ntemps", "temperatures in the ladder", pi.get("ntemps", run.nT)), ("nensemble", "chains per temperature (the cold ones give the samples)", pi.get("nensemble", run.nE)),
              ("pt_ensemble", "an ensemble at every temperature (False: only at the cold one)", pi.get("pt_ensemble")), ("Tmax", "hottest temperature asked for", pi.get("Tmax")),
              ("Tmax_from_SNR", "hottest temperature chosen as SNR^2 / ndim for this SNR", pi.get("Tmax_from_SNR")), ("initial_betas", "starting betas, if given", pi.get("initial_betas")),
              ("adapt", "adapt the ladder (at the start of the run; it stops after adapt_t0 x 10 rows)", pi.get("adapt")), ("adapt_t0", "adaptation time scale", pi.get("adapt_t0")),
              ("adapt_nu", "adaptation rate", pi.get("adapt_nu")), ("adapting now", "is the ladder still being adapted", run.adapting)]
    out.append("<h3>Parallel tempering</h3>" + _arg_table([(a, m, _bilby_fmt_arg(v)) for a, m, v in ladder]))
    steps = [("L1steps", "internal steps of a chain per stored row (one iteration)", run.L1), ("L2steps", "iterations between temperature swaps (and ensemble moves)", run.L2),
             ("target_nsamples", "samples asked for", cv.get("target_nsamples")), ("burn_in_nact", "burn-in in autocorrelation times", cv.get("burn_in_nact")),
             ("thin_by_nact", "samples are one per thin_by_nact x tau rows", cv.get("thin_by_nact")), ("autocorr_c", "Sokal window factor for tau", cv.get("autocorr_c")),
             ("fixed_discard", "rows always discarded", cv.get("fixed_discard")), ("min_tau", "smallest tau used", cv.get("min_tau")), ("fixed_tau", "tau fixed by the user", cv.get("fixed_tau")),
             ("tau_window", "tau from the last tau_window x tau rows", cv.get("tau_window")), ("stop_after_convergence", "stop stepping a chain once it has enough samples", cv.get("stop_after_convergence")),
             ("evidence_method", "", getattr(run.pt, "evidence_method", None)), ("initial_sample_method", "", getattr(run.pt, "initial_sample_method", None))]
    out.append("<h3>Counting and stopping</h3>" + _arg_table([(a, m, _bilby_fmt_arg(v)) for a, m, v in steps]))
    t = an["time"]
    prog = [("rows per chain", "stored rows (steps of one chain are L1steps per row, plus an extra row per ensemble move)", f"{run.rows:,}"), ("iterations", "swap rounds x L2steps", f"{run.iterations:,}"),
            ("rows per iteration", "above 1 because of the ensemble rows", f"{run.rows_per_iter:.3f}"), ("swap rounds", "", f"{run.swap_rounds:,}"),
            ("sampling time", "", f"{t['hours']:.2f} h"), ("time per iteration", "wall time, all chains in parallel", f"{t['per_iteration']:.2f} s"),
            ("time per internal step", "", f"{t['per_step_ms']:.1f} ms"), ("sampled parameters", "", f"{len(run.names)}: " + ", ".join(run.names)),
            ("burn-in", "bilby's last cut for each cold chain", _methods_text(run) + f"; the largest cut is row {an['bilby_idx']:,}")]
    out.append("<h3>Progress</h3>" + _arg_table(prog))
    ag = _first(run.sd[0][0], "AdaptiveGaussianProposal")
    if ag is not None:
        rows = []
        for k in run.names:
            if k in ag.prior_minimum_dict:
                rows.append((k, f"boundary: {ag.prior_boundary_dict.get(k)}", f"{_bilby_fmt_arg(float(ag.prior_minimum_dict[k]))} to {_bilby_fmt_arg(float(ag.prior_maximum_dict[k]))} (width {_bilby_fmt_arg(float(ag.prior_width_dict[k]))})"))
        out.append("<h3>Priors as the proposals see them</h3><p class=\"muted\">The range, the width that scales the adaptive Gaussian steps (an infinite prior counts as 1) and the boundary rule of every sampled parameter.</p>"
                   + _arg_table(rows, head=("parameter", "boundary", "range")))
    cyc = [f"{_kind(p)} ({len(getattr(p, 'parameters', []))} parameters, weight {getattr(p, 'weight', '?')})" for p in run.sd[0][0].proposal_cycle.proposal_list]
    out.append("<h3>Proposal cycle of every chain</h3><p class=\"muted\">" + _html.escape("; ".join(cyc)) + "</p>")
    d = None
    if arguments is not None:
        d = dict(vars(arguments)) if hasattr(arguments, "__dict__") else dict(arguments)
    if d:
        out.append('<h3>Arguments of run_sampler.py</h3><p class="muted">The arguments the run was started with, as parsed (defaults included).</p>' + _arg_table([("--" + k.replace("_", "-"), "", _bilby_fmt_arg(v)) for k, v in sorted(d.items())]))
        cmd = command_line or "python run_sampler.py " + " ".join(f"--{k.replace('_', '-')} {_bilby_fmt_arg(v)}" if not isinstance(v, bool) else (f"--{k.replace('_', '-')}" if v else "") for k, v in d.items() if v not in (None, False))
        out.append(f'<details><summary>Command line {"" if command_line else "(reconstructed from the parsed arguments)"}</summary><pre class="cmd">{_html.escape(cmd)}</pre></details>')
    if sampler_kwargs:
        out.append('<h3>Sampler keyword arguments</h3><pre class="cmd">' + _html.escape(json.dumps(sampler_kwargs, indent=2, default=str)) + "</pre>")
    out.append("</section>")
    return "".join(out)


def _swap_rows(sc):
    names = {"bad": "NOT carried forward", "ok": "carried forward", "na": "not conclusive"}
    cls = {"bad": "bad", "ok": "ok", "na": ""}
    out = []
    for i, s in enumerate(sc):
        share = "n/a" if not np.isfinite(s["share"]) else f"{100 * s['share']:.0f}%"
        out.append(f'<tr><td>E{i}</td><td>{s["jumps"]}</td><td>{100 * s["frac"]:.1f}%</td><td>{share}</td><td class="{cls[s["verdict"]]}">{names[s["verdict"]]}</td></tr>')
    return "".join(out)


def bilby_build_report(run, out_path, an=None, title=None, burn="bilby", trace_pages=True, explorer=True, dpi=110, arguments=None, command_line=None, sampler_kwargs=None, verbose=False, **analysis_kwargs):
    """Analyse `run` and write a self-contained HTML report (images embedded, no internet needed). Returns the analysis dict."""
    t0 = time.time()
    an = an or bilby_analyze(run, burn=burn, verbose=verbose, **analysis_kwargs)
    title = title or f"bilby_mcmc diagnostics: {os.path.basename(run.path)}"
    verdict_text = {"ok": "No red flags in the automated checks", "warn": "Some checks need attention", "bad": "Problems detected: do not trust these chains yet", "na": "Not enough information"}[an["verdict"]]
    sampled = [r for r in an["params"] if not r["derived"]]
    cts = an["counts"]
    taus = [r["tau"] for r in sampled if np.isfinite(r["tau"])]
    rec = an["swaps"]["recorded"]
    cards = [("iterations", f"{run.iterations:,}"), ("rows per chain", f"{run.rows:,}"), ("chains", f"{run.nT} x {run.nE}", "temperatures x chains per temperature"),
             ("sampling time", f"{an['time']['hours']:.1f} h"), ("samples n (bilby)", f"{_fnum(cts['n_now'], ',.0f')}" + (f" / {cts['target']:,}" if cts["target"] else ""), "bilby's own count over the cold chains and the target"),
             ("analysed from row", f"{an['bi']:,}"), ("slowest tau", f"{max(taus):,.0f} rows" if taus else "n/a"), ("min N / tau", _fnum(min((r["n_over_tau"] for r in sampled if np.isfinite(r["n_over_tau"])), default=np.nan), ".0f")),
             ("swap acceptance", f"{min(rec):.2f} - {max(rec):.2f}" if rec else "n/a"), ("round trip", f"{an['swaps']['round_trip_iterations']:,.0f} it.", "iterations for a replica to go hot -> cold -> hot"),
             ("flagged chains", f"{len(an['flagged'])}")]
    cards_html = "".join(f'<div class="card" title="{_html.escape(c[2]) if len(c) > 2 else ""}"><div class="k">{c[0]}</div><div class="v">{c[1]}</div></div>' for c in cards)

    def badge(s):
        return f'<span class="badge b-{s}">{ {"ok": "OK", "warn": "CHECK", "bad": "PROBLEM", "na": "n/a"}[s] }</span>'

    def dot(s):
        label = {"ok": "good", "warn": "check", "bad": "problem", "na": "not available"}[s]
        return f'<span class="dot b-{s}" title="{label}" aria-label="{label}"></span>'

    legend_html = ('<div class="legend"><span><i class="dot b-ok"></i> good</span><span><i class="dot b-warn"></i> check</span><span><i class="dot b-bad"></i> problem</span><span><i class="dot b-na"></i> not available</span></div>')
    checks_html = legend_html + ('<table class="checks"><colgroup><col style="width:24%"><col style="width:9%"><col style="width:32%"><col style="width:35%"></colgroup>'
                                 '<thead><tr><th>check</th><th>status</th><th>result</th><th>what counts as good</th></tr></thead><tbody>' +
                                 "".join(f'<tr><td data-l="check"><b>{_html.escape(c["label"])}</b></td><td data-l="status">{dot(c["status"])}</td><td data-l="result">{_html.escape(c["value"])}</td>'
                                         f'<td data-l="good" class="muted">{_html.escape(c["rule"])}</td></tr>' for c in an["checks"]) + "</tbody></table>")
    diag_html = ("<h3>Reading this run</h3><ul>" + "".join(f"<li>{_html.escape(t)}</li>" for t in an["diagnosis"]) + "</ul><h3>What to try</h3><ul>" + "".join(f"<li>{_html.escape(t)}</li>" for t in an["suggestions"]) + "</ul>")
    secs = []
    secs.append(f'<section id="overview"><h2>Overview</h2><p class="muted">Generated {time.strftime("%Y-%m-%d %H:%M")} from <code>{_html.escape(run.path)}</code>.</p>'
                f'<div class="verdict {an["verdict"]}">{badge(an["verdict"])}<strong>{verdict_text}</strong></div><div class="cards">{cards_html}</div>{diag_html}{checks_html}'
                '<p class="muted" style="margin-top:12px">Automated checks are heuristics. They can reveal problems but cannot prove convergence; see <a href="#methods">Methods</a>. '
                "One row of a chain is one iteration of L1steps internal steps, so tau in rows times L1steps is tau in steps of the sampler.</p></section>")
    secs.append(_settings_section(run, an, arguments, command_line, sampler_kwargs))
    secs.append('<section id="lnl"><h2>Log-likelihood</h2><p class="muted">The cold chains should climb to, and then fluctuate around, the same plateau. Dashed purple: the burn-in bilby used; dotted green: where the median ln L reaches its plateau.</p>'
                + _img(plot_logl(run, an), "log-likelihood", dpi) + "</section>")
    ld = an.get("ladder_design")
    swaps_txt = (f"A replica needs about {an['swaps']['round_trip_rounds']:.0f} swap rounds for a round trip from the hottest to the coldest chain and back "
                 f"({an['swaps']['round_trip_rows']:,.0f} rows, {an['swaps']['round_trip_iterations']:,.0f} iterations), estimated by simulating swaps with the recorded acceptances.")
    design = ""
    if ld is not None:
        design = ('<h3>A more even ladder</h3><p class="muted">Same number of temperatures and same hottest temperature, spaced so that the predicted swap acceptance is the same in every pair '
                  f'({ld["target"]:.2f}); the prediction reweights the stored ln L of the nearest chain and was checked against the recorded acceptances above. Round trip with it: about {ld["round_trip_rounds"]:.0f} swap rounds.</p>'
                  f'<pre class="cmd">"initial_betas": [{", ".join(f"{b:.4f}" for b in ld["betas"])}], "adapt": false</pre>')
    elif an.get("ladder_design_error"):
        design = f'<p class="muted">The ladder design could not be made: {_html.escape(an["ladder_design_error"])}</p>'
    sc = an["swap_check"]
    sc_rows = _swap_rows(sc)
    secs.append('<section id="temperatures"><h2>Temperatures and swaps</h2><p class="muted">Parallel tempering: hot chains cross barriers easily and hand their states down to the cold chain through swaps. '
                "The swap acceptance of every pair should be roughly even; the predicted values come from the stored ln L and should match the recorded ones.</p>"
                + _img(plot_ladder(run, an), "ladder and swaps", dpi) + f'<p class="muted">{swaps_txt}</p><h3>Per temperature</h3>' + _temps_table(run, an) + design
                + '<h3>Do the swaps move the chains?</h3><p class="muted">After a swap the cold chain\'s stored row jumps to the state it received. If the chain then continues from that state the next row stays near it; '
                  "if the swap only rewrote the stored row, the next row returns to the chain's own old state. The share of such returns among the jumps is about 100% when the swaps do nothing and low when they work. "
                  "Large moves made by the proposals themselves (KDE, mixture) can blur the test.</p>"
                + '<table><thead><tr><th>chain</th><th>jumps</th><th>share of rows</th><th>next row returns</th><th>reading</th></tr></thead><tbody>' + sc_rows + "</tbody></table></section>")
    bi_ = an["est_info"]
    scan_rows = "".join(f"<tr><td>{100 * s['frac']:.0f}%</td><td>{s['rows']:,}</td>{_cell(_fnum(s['rhat'], '.3f'), _status(s['rhat'], 1.01, 1.05), s['rhat'])}{_cell(_fnum(s['n_over_tau'], '.1f'), _status(s['n_over_tau'], 50, 20, True), s['n_over_tau'])}"
                        f"{_cell(_fnum(s['drift'], '.1f'), _status(s['drift'], 3, 5), s['drift'])}</tr>" for s in an["burn_scan"])
    burn_src = "bilby's cut" if an["burn_choice"] == "bilby" else str(an["burn_choice"])
    burn_para = ('<p class="muted">bilby removes rows by the largest of several rules: <b>FD</b> a fixed number, <b>10tau</b> burn_in_nact x tau, <b>z1</b> until the running ln posterior is within ndim/2 of its maximum, '
                 "<b>AD</b> the last temperature adaptation, <b>PR</b> a learning proposal. Here: " + _html.escape(_methods_text(run)) + f"; the cut is row {an['bilby_idx']:,} ({100 * an['bilby_idx'] / run.rows:.0f}% of the run). "
                 f"The statistics in this report use the part from row {an['bi']:,} ({_html.escape(burn_src)}).</p>")
    secs.append('<section id="burnin"><h2>Burn-in</h2><p class="muted">Top: the approach of ln L to its plateau on a log axis. Bottom: the three convergence statistics recomputed after removing the first part of the run. '
                "If a statistic keeps improving as more rows are removed, the chains have not forgotten their start.</p>"
                + burn_para
                + _bilby_stat_guide_html() + _img(bilby_plot_burnin(run, an), "burn-in", dpi)
                + '<h3>Statistics versus rows removed</h3><table><thead><tr><th>removed</th><th>rows</th><th>worst R-hat</th><th>worst N / tau</th><th>worst |drift z|</th></tr></thead><tbody>' + scan_rows + "</tbody></table></section>")
    secs.append('<section id="chains"><h2>Chains</h2><p class="muted">The cold chains (temperature 0) are the ensemble that gives the posterior. Click a column header to sort. A chain is flagged if its ln L is more than 3 robust spreads below the others '
                "and at least 1 unit lower than the best, if its slowest tau is more than 3 times the median and over 100 rows, or if some parameter's mean is more than 5 standard errors from the pooled mean.</p>"
                + _chains_table(an) + "<h3>Do the cold chains agree?</h3>" + _img(plot_offsets(run, an), "chain offsets", dpi) + "</section>")
    walls_img = _img(plot_walls(run, an), "prior walls", dpi) if an["walls"] is not None else ""
    steps_img = _img(plot_steps(run, an), "step sizes", dpi) if an["steps"] is not None else ""
    secs.append('<section id="proposals"><h2>Proposals</h2><p class="muted">What every chain proposes during its L1steps internal steps, and how well the proposals work at each temperature. Acceptance of a learning proposal (KDE, mixture) below 0.5% is shown in amber.</p>'
                + _proposals_table(run, an) + _img(plot_proposals(run, an), "proposals", dpi)
                + '<h3>Step sizes</h3><p class="muted">bilby\'s adaptive Gaussian proposal moves parameter i by scale x (prior width) x sigma_i x N(0,1) (equation 7 of the bilby-MCMC paper); the bilby cycles leave every sigma_i at 1, '
                  "so one shared scale, adapted to a 23% acceptance, serves all parameters. The table compares the step with the posterior width of each parameter in the cold chains.</p>" + steps_img + _steps_table(an)
                + '<p class="muted">Posterior widths from these chains, for <code>"tdinf_posterior_width"</code> of the tdinf_scaled, tdinf_blocks and tdinf_gw cycles (circular parameters are measured after unwrapping):</p>'
                  f'<pre class="cmd">"tdinf_posterior_width": {_html.escape(_widths_json(an))}</pre>'
                + '<h3>Prior walls</h3><p class="muted">Differential-evolution jumps x + gamma (a - b), with a and b random samples of the cold chains and bilby\'s distribution of gamma, tested against the prior ranges. A jump that leaves a prior with no boundary rule is '
                  "rejected before the likelihood is evaluated; a reflective boundary reflects it back (exact for diagonal steps such as the adaptive Gaussian, but it can bias the posterior near the walls for the correlated jumps of differential evolution and Fisher proposals), a periodic one wraps it (exact for any symmetric jump).</p>" + walls_img + "</section>")
    tr = ['<section id="traces"><h2>Traces</h2>']
    if explorer:
        payload = json.dumps(_bilby_explorer_payload(run, an), separators=(",", ":")).replace("</", "<\\/")
        tr.append('<p class="muted">Interactive explorer: bands are the 68% and 90% ranges across the cold chains, black is their median, grey lines are the chains, red lines flagged chains. The dashed line marks the burn-in.</p>'
                  '<div class="explorer"><div class="controls"><label>parameter <select id="ex-param"></select></label><label><input type="checkbox" id="ex-bands" checked> bands</label>'
                  '<label><input type="checkbox" id="ex-walkers" checked> chains</label><label><input type="checkbox" id="ex-flagged" checked> flagged chains</label>'
                  '<label><input type="checkbox" id="ex-all"> include burn-in in the y-range</label></div><canvas id="ex-canvas" height="380"></canvas><div id="ex-tip" class="tip"></div></div>'
                  f'<script id="ex-data" type="application/json">{payload}</script><script>{_EXPLORER_JS}</script>')
    if trace_pages:
        names = [r["name"] for r in an["params"]]
        tr.append(f"<h3>Static trace plots</h3><p class=\"muted\">All {len(names)} parameters (derived ones included). Coloured lines: the cold chains; black: their median; purple dashed: the burn-in. "
                  "Right-hand panels compare the first (blue) and second (orange) half of the chains after the burn-in: they should coincide.</p>")
        for k, fig in enumerate(bilby_plot_traces(run, an, names, per_page=5)):
            tr.append(_img(fig, f"traces page {k + 1}", dpi))
    tr.append("</section>")
    secs.append("".join(tr))
    sd0 = an["slow_dirs"][0]
    secs.append('<section id="convergence"><h2>Convergence and autocorrelation</h2><p class="muted">N/tau above 50 is the primary criterion; R-hat between the cold chains is a supporting check. tau is in rows.</p>'
                "<h3>How n and tau developed</h3>" + _img(plot_history(run, an), "history", dpi) + "<h3>Autocorrelation function</h3>" + _img(bilby_plot_acf(run, an), "autocorrelation function", dpi)
                + "<h3>Per-parameter statistics</h3>" + _bilby_params_table(an)
                + '<h3>tau at every temperature</h3><p class="muted">A tau that is much larger in the cold chain than one temperature up points to the local moves of the cold chain; a similar tau everywhere points to the swaps.</p>' + _img(plot_tau_by_T(run, an), "tau by temperature", dpi)
                + '<h3>Slowest directions</h3><p class="muted">Principal directions of the standardised cold-chain samples, ranked by the tau of their projection. A slow combination of parameters is what limits the run, not any one of them.</p>'
                + _img(plot_slow_direction(run, an), "slow directions", dpi) + "</section>")
    secs.append('<section id="posteriors"><h2>Posteriors</h2><p class="muted">Histograms after the burn-in: black pools the cold chains, colours are the individual chains. Chains that sit in different places or in different modes show up here before R-hat flags them.</p>'
                + _img(plot_hist(run, an), "histograms", dpi) + "</section>")
    secs.append('<section id="methods"><h2>Methods and caveats</h2><details open><summary>What is computed</summary><ul>'
                "<li><b>Chains.</b> Every chain in the resume file; the cold ones (beta = 1, one per ensemble member) are analysed as the ensemble. The pickle is read with a restricted loader: nothing in it is run.</li>"
                "<li><b>Burn-in.</b> By default bilby's own last cut (the largest over the cold chains), because the early rows would spoil tau and R-hat; the Burn-in section shows what the statistics look like for other cuts, and the ln L plateau estimate (as in the emcee report).</li>"
                "<li><b>tau.</b> emcee-style integrated autocorrelation time: the autocorrelation of each cold chain, averaged over the chains and integrated with Sokal's window (c = 5), in rows. ESS = chains x N / tau. "
                "bilby itself computes tau per chain and takes the maximum over parameters; the table also gives the range over the chains.</li>"
                "<li><b>R-hat.</b> Rank-normalised split R-hat (bulk and folded tail), the larger of the two, between the cold chains.</li>"
                "<li><b>Angles.</b> Phase, right ascension, polarization (folded onto a period of pi: psi and psi + pi give the same signal) and the spin azimuths are unwrapped around their circular mean before any statistic.</li>"
                "<li><b>Swap prediction.</b> The ln L of the chain with the closest beta is importance-reweighted to the beta wanted and the swap acceptance E[min(1, exp((b1 - b2)(L2 - L1)))] is averaged over 500 x 500 quantile points.</li>"
                "<li><b>Round trip.</b> A replica is followed through swaps drawn with the recorded acceptances, pairs tried in order from the cold end.</li>"
                "<li><b>Time to finish.</b> The remaining samples divided by the rate at which the cold chains make independent samples (one per thin_by_nact x tau rows each), at the present time per iteration. "
                "bilby's printed estimate multiplies by the primary chain's tau only, so with several cold chains it reads too high.</li></ul></details>"
                "<details><summary>Caveats</summary><ul><li>The cold chains of an ensemble are not independent runs, so R-hat between them is a heuristic; N / tau is the primary criterion.</li>"
                "<li>A posterior with separate modes can look converged if no chain visits the other mode. Parallel tempering helps, but only if the hot chains reach the mode; compare the temperatures in the tables above.</li>"
                "<li>The swap test looks for jumps of the stored state; with large moves from learning proposals it can be inconclusive.</li>"
                "<li>Where the file lacks a number (an old pickle, a run with a different layout) the corresponding check shows n/a.</li></ul></details></section>")
    nav = "".join(f'<a href="#{i}">{t}</a>' for i, t in (("overview", "Overview"), ("settings", "Settings"), ("lnl", "Log-likelihood"), ("temperatures", "Temperatures"), ("burnin", "Burn-in"), ("chains", "Chains"),
                                                          ("proposals", "Proposals"), ("traces", "Traces"), ("convergence", "Convergence"), ("posteriors", "Posteriors"), ("methods", "Methods")))
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{_html.escape(title)}</title><style>{_CSS}</style></head><body>'
            f'<header><h1>{_html.escape(title)}</h1><p>{run.iterations:,} iterations, {run.nT} temperatures x {run.nE} chains, {len(run.names)} sampled parameters</p></header>'
            f'<div class="layout"><nav>{nav}</nav><main>{"".join(secs)}</main></div><footer>report generated in {time.time() - t0 + an["seconds"]:.0f} s by sampler_diagnostics.py</footer><script>{_SORT_JS}</script></body></html>')
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(page)
    print(f"wrote {out_path} ({len(page) / 1e6:.1f} MB) in {time.time() - t0 + an['seconds']:.0f} s; verdict: {an['verdict']}")
    return an


# ============================================================================== run_sampler integration and command line

def _bilby_print_summary(an):
    print(f"verdict: {an['verdict']}")
    for c in an["checks"]:
        print(f"  [{c['status']:4s}] {c['label']}: {c['value']}")


def _bilby_parse_burn(b):
    if b in ("bilby", "estimated"):
        return b
    try:
        v = float(b)
    except ValueError:
        raise SystemExit("--burn must be bilby, estimated, a number of rows or a fraction")
    return v if "." in str(b) and 0 < v < 1 else int(v)


def report_from_pickle(path, out_path=None, burn="bilby", drop_rows_mod=None, arguments=None, command_line=None, sampler_kwargs=None, title=None, safe=False, **build_kw):
    """Write the HTML report for a bilby_mcmc resume pickle. The report goes to <pickle without .pickle>_diagnostics.html unless out_path is given."""
    def _go():
        pt = load(path)
        run = BilbyRun(pt, path, drop=drop_rows_mod)
        out = out_path or re.sub(r"\.pickle$", "", str(path)) + "_diagnostics.html"
        an = bilby_build_report(run, out, burn=burn, arguments=arguments, command_line=command_line, sampler_kwargs=sampler_kwargs, title=title, **build_kw)
        _bilby_print_summary(an)
        return an
    if not safe:
        return _go()
    try:
        return _go()
    except Exception as e:                                     # a diagnostics problem must not break the run that called this
        print(f"WARNING: the diagnostics report could not be made: {type(e).__name__}: {e}")
        return None


def bilby_report_from_args(args, kwargs=None, sampler_kwargs=None, pickle_path=None, out_path=None, safe=True, **build_kw):
    """
    Write the diagnostics report for a tdinf bilby_mcmc run from run_sampler.py's own `args` (the same call as for emcee runs).
    The resume file is <outdir>/<label>_resume.pickle with outdir and label taken from args.output unless pickle_path is given; the report goes to
    <output without .h5/.dat>_diagnostics.html unless out_path is given. safe=True (default): any error is printed as a warning and None is returned.
    """
    out = str(args.output)
    base = os.path.splitext(out)[0]
    path = pickle_path or os.path.join(os.path.dirname(out) or ".", os.path.basename(base) + "_resume.pickle")
    if not os.path.isfile(path):
        print(f"NOTE: no resume file {path}; bilby may have removed it when the run finished, so no diagnostics report was made")
        return None
    return report_from_pickle(path, out_path or base + "_diagnostics.html", arguments=args, sampler_kwargs=sampler_kwargs or kwargs, safe=safe, **build_kw)


def bilby_main(argv=None):
    ap = argparse.ArgumentParser(description="Diagnostics of a bilby_mcmc run from its resume pickle, as one self-contained HTML report.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=_BILBY_DOC)
    ap.add_argument("pickle", help="<label>_resume.pickle")
    ap.add_argument("-o", "--out", default=None, help="report to write (default: <pickle>_diagnostics.html)")
    ap.add_argument("--burn", default="bilby", help="where the statistics start: bilby (its own cut), estimated (ln L plateau), a number of rows, or a fraction of the run")
    ap.add_argument("--drop-rows-mod", default=None, help="M:R, leave out the rows whose index is R mod M (an old run whose swaps did not carry the state forward: 4:1)")
    ap.add_argument("--no-explorer", action="store_true", help="leave out the interactive trace explorer")
    ap.add_argument("--no-trace-pages", action="store_true", help="leave out the static trace plots (smaller file)")
    ap.add_argument("--dpi", type=int, default=110)
    ap.add_argument("--posterior-widths", action="store_true", help="only print the posterior widths as JSON for the tdinf_scaled / tdinf_blocks / tdinf_gw cycles")
    ap.add_argument("--swaps-only", action="store_true", help="only print the swap check")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)
    drop = tuple(int(x) for x in a.drop_rows_mod.split(":")) if a.drop_rows_mod else None
    burn = _bilby_parse_burn(a.burn)
    if a.posterior_widths or a.swaps_only:
        run = BilbyRun(load(a.pickle), a.pickle, drop=drop)
        bi = choose_burn(run, burn)
        if a.posterior_widths:
            X = run.stack(0, bi, run.centres(bi), derived=False)
            print('"tdinf_posterior_width": ' + json.dumps({k: float(f"{np.std(v):.3g}") for k, v in X.items()}))
        if a.swaps_only:
            for i, s in enumerate(swap_check(run, bi)):
                print(f"ensemble {i}: {s['jumps']} jump rows ({100 * s['frac']:.1f}% of rows); the next row returns to the pre-jump state in "
                      f"{'n/a' if not np.isfinite(s['share']) else format(100 * s['share'], '.0f') + '%'} -> {s['verdict']}")
        return
    report_from_pickle(a.pickle, a.out, burn=burn, drop_rows_mod=drop, explorer=not a.no_explorer, trace_pages=not a.no_trace_pages, dpi=a.dpi, verbose=a.verbose)


# ==============================================================================================================================================
# PART 3: the dispatcher and Eryn (the cold chain through the emcee report, plus the tempering analysis)
# ==============================================================================================================================================
SUPPORTED = ("emcee", "eryn", "bilby_mcmc")


# ============================================================================== the dispatcher
def backend_path(args):
    """The HDF5 backend of emcee and Eryn runs: --output if it ends in .h5, else --output with .h5 (as run_emcee and run_eryn name it)."""
    out = str(args.output)
    return out if out.endswith(".h5") else os.path.splitext(out)[0] + ".h5"


def default_out_path(args):
    return os.path.splitext(str(args.output))[0] + "_diagnostics.html"


def run(sampler_name, args, kwargs=None, sampler_kwargs=None, samples=None, ln_posterior=None, out_path=None, safe=True, **options):
    """
    Write the diagnostics report of the sampler called `sampler_name` ("emcee", "eryn" or "bilby_mcmc").

    args, kwargs, sampler_kwargs : what run_sampler gave the sampler (the files it wrote are found from args.output); samples and ln_posterior (what the sampler returned) are
    accepted for the samplers that may use them and are not needed by the three above. out_path : the report (default <output without extension>_diagnostics.html).
    options go to the sampler's report: for emcee and eryn burn, thin, max_points, max_steps (default 20000), title, spin_model, physical; for bilby_mcmc burn, trace_pages,
    explorer, dpi, drop_rows_mod. Returns the analysis dictionary, or None when there is no report. safe=True: errors are printed, not raised.
    """
    name = str(sampler_name).lower()
    handlers = dict(emcee=emcee_report, eryn=eryn_report, bilby_mcmc=bilby_mcmc_report)
    if name not in handlers:
        print(f"NOTE: there are no sampler diagnostics for {sampler_name!r} yet (available: {', '.join(SUPPORTED)})")
        return None
    t0 = time.time()
    print(f"writing the {name} diagnostics report ...", flush=True)
    result = handlers[name](args, kwargs=kwargs, sampler_kwargs=sampler_kwargs, out_path=out_path or default_out_path(args), safe=safe, **options)
    if result is not None:
        print(f"diagnostics done in {time.time() - t0:.0f} s")
    return result


def emcee_report(args, kwargs=None, sampler_kwargs=None, out_path=None, safe=True, max_steps=20000, **options):
    """The emcee report (the module's own report_from_args)."""
    args = copy.copy(args)
    args.output_h5 = getattr(args, "output_h5", None) or backend_path(args)
    return report_from_args(args, kwargs=kwargs, out_path=out_path or default_out_path(args), safe=safe, max_steps=max_steps, **options)


def bilby_mcmc_report(args, kwargs=None, sampler_kwargs=None, out_path=None, safe=True, **options):
    """The bilby_mcmc report, from <outdir>/<label>_resume.pickle."""
    return bilby_report_from_args(args, kwargs=kwargs, sampler_kwargs=sampler_kwargs, out_path=out_path or default_out_path(args), safe=safe, **options)
# ============================================================================== reading an Eryn backend
def _h5_attr(obj, key):
    try:
        return obj.attrs[key]
    except Exception:                                   # a file copied or killed while it was written can have attributes that cannot be read
        return None


def _last_filled(ds, n):
    """Number of rows that hold data (rows never written are all zeros: Eryn grows its datasets ahead of the run)."""
    if n < 1:
        return 0
    filled = lambda i: bool(np.any(np.asarray(ds[i]) != 0))
    if filled(n - 1):
        return int(n)
    if not filled(0):
        return 0
    lo, hi = 0, n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if filled(mid) else (lo, mid)
    return int(lo + 1)


def _find_chain(f, h5, branch=None):
    names = []

    def visit(name, obj):
        try:
            if isinstance(obj, h5.Dataset) and "chain" in name.split("/") and len(obj.shape) == 5:
                names.append(name)
        except Exception:
            pass

    f.visititems(visit)
    names = sorted(names)
    if branch is not None:
        names = [n for n in names if n.endswith("/" + branch)]
    if not names:
        raise ValueError("no chain dataset (steps, temperatures, walkers, leaves, parameters) in the file: is it an Eryn backend?")
    cname = names[0]
    return cname, (cname.split("/chain/")[0] if "/chain/" in cname else "")


def eryn_file_info(path, branch=None):
    """(ndim, ntemps, nwalkers, steps written) of an Eryn backend, cheaply."""
    import h5py
    with h5py.File(path, "r") as f:
        cname, group = _find_chain(f, h5py, branch)
        ch = f[cname]
        n_alloc, T, W, L, D = ch.shape
        it = _h5_attr(f[group] if group else f, "iteration")
        n = min(int(it), n_alloc) if it is not None else n_alloc
        n = _last_filled(ch, n)
    return dict(ndim=int(D), ntemps=int(T), nwalkers=int(W), n_steps=int(n))


def read_eryn_backend(path, names=None, max_steps=20000, max_temp_rows=3000, max_pop_rows=1500, max_pop_bytes=4e8, n_tail=1000, block_steps=2000, branch=None):
    """
    Read an Eryn HDF5 backend for the report. Returns a dict with
      run               a Run (PART 1) of the COLD chain (temperature 0, ln p = ln likelihood + ln prior), read with a stride when the run has more than max_steps steps
      ll_T, betas_T     the log likelihood of every temperature (rows, T, W) and the betas (rows, T), at steps_T (a stride keeping at most max_temp_rows rows)
      chain_P           the chain of every temperature (rows, T, W, D, float32, the Run's column offsets removed) at steps_P (at most max_pop_rows rows and max_pop_bytes)
      swaps_accepted, accepted, moves    the counters of the backend (or None), moves: name -> acceptance fraction (T, W)
      and the sizes, the Eryn version, and what was dropped. Rows that were never written, or that cannot be read, are not used.
    """
    import h5py
    notes = []
    with h5py.File(path, "r") as f:
        cname, group = _find_chain(f, h5py, branch)
        join = lambda k: f"{group}/{k}" if group else k
        ch = f[cname]
        n_alloc, T, W, L, D = ch.shape
        if L != 1:
            raise ValueError(f"{cname} has {L} leaves per walker: tdinf runs have one")
        gobj = f[group] if group else f
        it = _h5_attr(gobj, "iteration")
        n = min(int(it), n_alloc) if it is not None else n_alloc
        n_written = _last_filled(ch, n)
        if n_written < n:
            notes.append(f"the last {n - n_written:,} step(s) were never written (all zeros; a copy made while the run was writing): they are not used")
            n = n_written
        if n < 8:
            raise ValueError(f"only {n} steps in the backend")
        ll_ds, lp_ds = f[join("log_like")], f[join("log_prior")]
        stride = max(int(np.ceil(n / float(max_steps))), 1) if max_steps else 1
        steps = np.arange(0, n, stride)
        offsets = _column_offsets(ch[0:min(200, n), 0, :, 0, :])
        x = np.empty((len(steps), W, D), dtype=np.float32)
        lnp = np.empty((len(steps), W), dtype=np.float32)
        blk = max(stride, (block_steps // stride) * stride)
        k = 0
        for b0 in range(0, n, blk):
            b1 = min(b0 + blk, n)
            c = np.asarray(ch[b0:b1:stride, 0, :, 0, :], dtype=np.float64) - offsets
            lp = np.asarray(ll_ds[b0:b1:stride, 0], dtype=np.float64) + np.asarray(lp_ds[b0:b1:stride, 0], dtype=np.float64)
            x[k:k + len(c)], lnp[k:k + len(c)] = c, lp
            k += len(c)
        t0 = max(0, n - n_tail)
        tail_x = (np.asarray(ch[t0:n, 0, :, 0, :], dtype=np.float64) - offsets).astype(np.float32)
        tail_lnp = (np.asarray(ll_ds[t0:n, 0], dtype=np.float64) + np.asarray(lp_ds[t0:n, 0], dtype=np.float64)).astype(np.float32)
        # counters
        def get(key):
            try:
                return np.asarray(f[join(key)][()])
            except Exception:
                return None
        swaps, accepted = get("swaps_accepted"), get("accepted")
        moves = {}
        try:
            for mname in f[join("moves")].keys():
                try:
                    moves[mname] = np.asarray(f[join(f"moves/{mname}/acceptance_fraction")][()], dtype=float)
                except Exception:
                    pass
        except Exception:
            pass
        if moves:
            first = moves[sorted(moves)[0]]
            acc0 = first[0]
        elif accepted is not None and accepted.ndim == 2:
            acc0 = accepted[0] / max(n, 1)
        else:
            acc0 = None
        # all temperatures: log likelihood and ladder at a stride, the chain at a coarser one
        s2 = max(int(np.ceil(n / float(max_temp_rows))), 1)
        steps_T = np.arange(0, n, s2)
        ll_T = np.asarray(ll_ds[0:n:s2], dtype=np.float32)
        betas_T = None
        try:
            betas_T = np.asarray(f[join("betas")][0:n:s2], dtype=float)
        except Exception:
            notes.append("the backend has no betas: the ladder is not shown")
        s3 = max(int(np.ceil(n / float(max_pop_rows))), int(np.ceil(n * T * W * D * 4 / float(max_pop_bytes))), 1)
        steps_P = np.arange(0, n, s3)
        chain_P = np.empty((len(steps_P), T, W, D), dtype=np.float32)
        kk, step_blk = 0, max(s3, (max(block_steps // 4, 1) // s3) * s3)
        for b0 in range(0, n, step_blk):
            b1 = min(b0 + step_blk, n)
            c = np.asarray(ch[b0:b1:s3, :, :, 0, :], dtype=np.float64) - offsets
            chain_P[kk:kk + len(c)] = c
            kk += len(c)
        version = _h5_attr(gobj, "version")
        version = version.decode() if isinstance(version, bytes) else (str(version) if version is not None else "?")
    note = _thin_note(stride, "limit" if stride > 1 else "", n, max_steps)
    run_ = Run(x, lnp, steps, n, names=names, tail_x=tail_x, tail_lnp=tail_lnp, accepted=acc0, source=str(path), offsets=offsets, thin_note=note)
    print(f"read {path}: {n} steps x {T} temperatures x {W} walkers x {D} parameters; cold chain stride {stride} -> {len(steps)} rows; "
          f"all temperatures: ln L at stride {s2} ({len(steps_T)} rows), chain at stride {s3} ({len(steps_P)} rows)")
    return dict(run=run_, path=str(path), n_steps=int(n), ntemps=int(T), nwalkers=int(W), ndim=int(D), version=version, stride=stride, notes=notes, offsets=offsets,
                steps_T=steps_T, ll_T=ll_T, betas_T=betas_T, stride_T=s2, steps_P=steps_P, chain_P=chain_P, stride_P=s3,
                swaps_accepted=None if swaps is None else np.asarray(swaps, dtype=float).reshape(-1), accepted=accepted, moves=moves)


# ============================================================================== the tempering analysis
def _eryn_fmt(v, spec=".3g"):
    return "n/a" if v is None or not np.isfinite(v) else format(v, spec)


def _table(head, rows, cls=""):
    """An HTML table: head is a list of titles, rows a list of lists of (text) or (text, css class)."""
    out = [f'<div class="scroll"><table class="{cls}"><thead><tr>' + "".join(f"<th>{_html.escape(h)}</th>" for h in head) + "</tr></thead><tbody>"]
    for r in rows:
        out.append("<tr>" + "".join(_cell(c[0], c[1]) if isinstance(c, tuple) else _cell(c) for c in r) + "</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


def _st_cls(status):
    return status if status in ("ok", "warn", "bad") else ""


def predicted_swap_acceptance(ll_T, betas_T, rows, rng=None):
    """
    The swap acceptance that the stored log likelihoods predict for each adjacent pair: the mean over the given rows of min(1, exp((beta_k - beta_k+1) (lnL_k+1 - lnL_k)))
    with the walkers of the two temperatures paired at random. This needs no counter, so it also shows what a ladder WOULD do. Returns an array of T - 1 values.
    """
    rng = rng or np.random.default_rng(0)
    L, B = np.asarray(ll_T[rows], dtype=float), np.asarray(betas_T[rows], dtype=float)
    R, T, W = L.shape
    out = np.empty(T - 1)
    for k in range(T - 1):
        perm = rng.random((R, W)).argsort(axis=1)
        a = np.take_along_axis(L[:, k + 1, :], perm, axis=1) - L[:, k, :]
        d = (B[:, k] - B[:, k + 1])[:, None]
        out[k] = float(np.mean(np.minimum(1.0, np.exp(np.clip(d * a, -50, 50)))))
    return out


def population_tau(chain_P, rows, stride_P):
    """
    For every temperature and parameter: the autocorrelation time, in steps, of the ensemble mean (the mean over walkers at each step); the half-vs-half drift of that mean in
    standard errors (z); and the shift of the ensemble mean between the two halves in units of the parameter's own spread over all walkers and steps (shift).
    The shift does not depend on the autocorrelation time. z does, and a series with a trend has a long autocorrelation time because of that trend, which makes z small
    however large the drift: z alone can reassure falsely. chain_P (rows, T, W, D); rows selects the steps. Returns tau, z, shift, each (T, D); nan where the series is too short.
    """
    X = np.asarray(chain_P[rows], dtype=np.float64)
    R, T, W, D = X.shape
    tau, z, shift = (np.full((T, D), np.nan) for _ in range(3))
    if R < 8:
        return tau, z, shift
    h = R // 2
    for t in range(T):
        M = X[:, t, :, :].mean(axis=1)
        spread = X[:, t, :, :].reshape(-1, D).std(axis=0)
        for j in range(D):
            s = M[:, j:j + 1]
            tr = integrated_autocorr_time(s, 5.0)
            if np.isfinite(tr):
                tau[t, j] = tr * stride_P
                z[t, j] = drift_z(s, tr)
            if spread[j] > 0:
                shift[t, j] = (M[h:2 * h, j].mean() - M[:h, j].mean()) / spread[j]
    return tau, z, shift


def _fig_ladder(E, pred, measured):
    T = E["ntemps"]
    with _style():
        fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.9), gridspec_kw=dict(width_ratios=[1.35, 1]))
        if E["betas_T"] is not None:
            temps = 1.0 / np.maximum(E["betas_T"], 1e-12)
            for k in range(T):
                ax[0].plot(E["steps_T"], temps[:, k], lw=1.4, label=f"T{k}")
            ax[0].set_yscale("log")
            ax[0].legend(ncol=min(T, 5), loc="upper center", bbox_to_anchor=(0.5, -0.2))
        else:
            ax[0].text(0.5, 0.5, "no betas in the backend", ha="center", va="center", transform=ax[0].transAxes)
        ax[0].set_xlabel("step")
        ax[0].set_ylabel("temperature 1 / beta")
        ax[0].set_title("Temperature ladder")
        x = np.arange(T - 1)
        w = 0.38
        ax[1].axhspan(0.2, 0.5, color=C["green"], alpha=0.10, lw=0)
        if measured is not None:
            ax[1].bar(x - w / 2, measured, w, color=C["blue"], label="measured (counters)")
        if pred is not None:
            ax[1].bar(x + (w / 2 if measured is not None else 0), pred, w if measured is not None else 0.6, color=C["orange"], label="predicted from ln L")
        ax[1].set_xticks(x)
        ax[1].set_xticklabels([f"T{k}-T{k + 1}" for k in range(T - 1)])
        ax[1].set_ylim(0, 1)
        ax[1].set_ylabel("swap acceptance")
        ax[1].set_title("Swap acceptance per pair")
        ax[1].legend(loc="upper right")
        fig.tight_layout()
    return fig


def _fig_lnl(E, start, nb=60):
    T, steps, L = E["ntemps"], E["steps_T"], E["ll_T"]
    post = steps >= start
    with _style():
        fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.9), gridspec_kw=dict(width_ratios=[1.35, 1]))
        edges = np.linspace(0, len(steps), min(nb, max(len(steps) // 4, 2)) + 1).astype(int)
        cen = np.array([steps[edges[i]:max(edges[i + 1], edges[i] + 1)].mean() for i in range(len(edges) - 1)])
        meds = []
        for k in range(T):
            m = np.array([np.nanmedian(L[edges[i]:max(edges[i + 1], edges[i] + 1), k]) for i in range(len(edges) - 1)])
            lo = np.array([np.nanpercentile(L[edges[i]:max(edges[i + 1], edges[i] + 1), k], 16) for i in range(len(edges) - 1)])
            hi = np.array([np.nanpercentile(L[edges[i]:max(edges[i + 1], edges[i] + 1), k], 84) for i in range(len(edges) - 1)])
            ax[0].plot(cen, m, lw=1.5, label=f"T{k}")
            ax[0].fill_between(cen, lo, hi, alpha=0.15, lw=0)
            meds.append(m)
        floor = np.nanpercentile(np.concatenate(meds), 2) if meds else 0
        top = np.nanmax(np.concatenate(meds)) if meds else 1
        ax[0].set_ylim(floor - 0.1 * (top - floor), top + 0.1 * (top - floor))
        if start > 0:
            ax[0].axvline(start, color=C["burn"], ls="--", lw=1.3)
        ax[0].set_xlabel("step")
        ax[0].set_ylabel("ln likelihood (median, 16-84%)")
        ax[0].set_title("ln L of every temperature")
        ax[0].legend(ncol=min(T, 5), loc="lower right")
        if post.sum() > 4:
            allv = L[post].reshape(-1)
            bins = np.linspace(np.nanpercentile(allv, 0.5), np.nanpercentile(allv, 99.5), 70)
            for k in range(T):
                v = L[post][:, k].reshape(-1)
                ax[1].hist(v[np.isfinite(v)], bins=bins, density=True, histtype="step", lw=1.6, label=f"T{k}")
        ax[1].set_xlabel("ln likelihood after the burn-in")
        ax[1].set_ylabel("density")
        ax[1].set_title("Overlap of neighbouring temperatures")
        fig.tight_layout()
    return fig


def _fig_moves(E):
    moves, T = E["moves"], E["ntemps"]
    if not moves:
        return None
    with _style():
        fig, ax = plt.subplots(figsize=(8.5, 3.6))
        names = sorted(moves)
        w = 0.8 / len(names)
        for i, nm in enumerate(names):
            ax.bar(np.arange(T) + (i - (len(names) - 1) / 2) * w, np.nanmean(moves[nm], axis=1), w, label=nm)
        ax.axhspan(0.2, 0.5, color=C["green"], alpha=0.10, lw=0)
        ax.set_xticks(np.arange(T))
        ax.set_xticklabels([f"T{k}" for k in range(T)])
        ax.set_ylabel("acceptance fraction (mean over walkers)")
        ax.set_title("Move acceptance at every temperature")
        ax.legend()
        fig.tight_layout()
    return fig


def _fig_population(E, params, labels, start):
    steps, X, T = E["steps_P"], E["chain_P"], E["ntemps"]
    k = len(params)
    ncols = 3 if k > 4 else 2
    nrows = int(np.ceil(k / ncols))
    with _style():
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 2.7 * nrows), squeeze=False)
        for a, j, lab in zip(axes.ravel(), params, labels):
            for t in range(T):
                a.plot(steps, X[:, t, :, j].astype(float).mean(axis=1), lw=1.1, label=f"T{t}")
            if start > 0:
                a.axvline(start, color=C["burn"], ls="--", lw=1.1)
            a.set_title(lab, fontsize=10)
            a.set_xlabel("step")
        for a in axes.ravel()[k:]:
            a.set_visible(False)
        axes.ravel()[0].legend(ncol=min(T, 5), fontsize=8)
        fig.tight_layout()
    return fig


def make_eryn_extras(E, dpi=110):
    """The `extras` callable of build_report (PART 1) for an Eryn backend read by read_eryn_backend."""
    def extras(run, an):
        T, W, n = E["ntemps"], E["nwalkers"], E["n_steps"]
        start = int(an["est_step"]) if an["est_found"] else int(0.5 * n)
        start_note = ("the estimated burn-in of the cold chain" if an["est_found"] else "half of the run (no ln p plateau was found)")
        betas, stepsT, LL = E["betas_T"], E["steps_T"], E["ll_T"]
        post_T = np.where(stepsT >= start)[0]
        if len(post_T) < 8:
            post_T = np.arange(len(stepsT) // 2, len(stepsT))
        checks, obs, tips, sections_html = [], [], [], []
        # ---- ladder
        final_T = None
        ladder_html = ""
        if betas is not None:
            final_T = 1.0 / np.maximum(betas[-1], 1e-12)
        if betas is not None and T > 1:
            fr = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
            idx = [min(int(round(f * (len(stepsT) - 1))), len(stepsT) - 1) for f in fr]
            rows = [[f"{int(stepsT[i]):,}"] + [_eryn_fmt(1.0 / max(betas[i, k], 1e-12), ".3f") for k in range(T)] for i in idx]
            ladder_html = _table(["step"] + [f"T{k}" for k in range(T)], rows)
            q = int(0.75 * (len(stepsT) - 1))
            rel = np.abs(betas[-1] - betas[q]) / np.maximum(np.abs(betas[q]), 1e-12)
            rel[0] = 0.0
            worst = int(np.argmax(rel))
            checks.append(dict(key="ladder", label="Temperature ladder settled", status=_status(float(rel.max()), 0.02, 0.05),
                               value=f"largest change of a beta over the last quarter of the run: {100 * rel.max():.1f}% (T{worst})",
                               rule="Good if no beta changes by more than 2% over the last quarter of the run; check up to 5%. The adaptive ladder keeps moving until adaptation stops."))
            if rel.max() >= 0.02:
                tips.append("The temperature ladder is still adapting at the end of the run. Stop the adaptation once it has settled (the stop_adaptation entry of tempering_kwargs), "
                            "so that the last part of the chain is sampled with a fixed ladder.")
        # ---- swap acceptance
        measured = None
        if E["swaps_accepted"] is not None and len(E["swaps_accepted"]) == T - 1:
            measured = E["swaps_accepted"] / float(n * W)
        pred = predicted_swap_acceptance(LL, betas, post_T) if (betas is not None and T > 1 and len(post_T) >= 4) else None
        both = measured if measured is not None else pred
        pairs_rows = []
        if both is not None:
            for k in range(T - 1):
                m_, p_ = (measured[k] if measured is not None else None), (pred[k] if pred is not None else None)
                st = _status(both[k], 0.15, 0.05, True)
                pairs_rows.append([f"T{k} - T{k + 1}", _eryn_fmt(final_T[k], ".3f") if final_T is not None else "n/a", _eryn_fmt(final_T[k + 1], ".3f") if final_T is not None else "n/a",
                                   (_eryn_fmt(m_, ".3f"), _st_cls(st) if measured is not None else ""), (_eryn_fmt(p_, ".3f"), _st_cls(st) if measured is None else ""),
                                   _eryn_fmt(p_ / m_, ".2f") if (m_ and p_ is not None) else "n/a"])
            lo_, hi_ = float(np.min(both)), float(np.max(both))
            src = "measured" if measured is not None else "predicted from ln L"
            checks.append(dict(key="swap", label="Swap acceptance of every pair", status=_status(lo_, 0.15, 0.05, True),
                               value=f"{lo_:.2f} to {hi_:.2f} ({src}; lowest pair T{int(np.argmin(both))}-T{int(np.argmin(both)) + 1})",
                               rule="Good if every adjacent pair accepts at least 15% of its swaps (20% to 50% is the usual target); check from 5%; problem below 5%: the ladder has a gap."))
            ratio = hi_ / max(lo_, 1e-12)
            checks.append(dict(key="swap_equal", label="Swap acceptance equalised across the ladder", status=_status(ratio, 1.5, 2.5),
                               value=f"highest / lowest = {ratio:.2f}",
                               rule="Good if the pairs differ by less than a factor 1.5; check up to 2.5. The adaptive ladder aims to equalise them; a large spread means it has not."))
            if lo_ < 0.1:
                obs.append(f"The swap acceptance of T{int(np.argmin(both))}-T{int(np.argmin(both)) + 1} is only {lo_:.2f}: replicas hardly cross that gap, so the hot chains help the cold chain very little.")
                tips.append("Add temperatures, or lower the highest temperature, so that every pair accepts more swaps.")
            elif lo_ >= 0.4:
                obs.append(f"Every pair accepts {lo_:.2f} to {hi_:.2f} of its swaps: the temperatures are closer together than they need to be.")
                tips.append("The same number of evaluations can reach a hotter top temperature (a larger Tmax at the same number of temperatures) while keeping the swap acceptance near 0.2 to 0.3. "
                            "Whether a hotter top is needed is shown by the drift of the hottest chain below.")
            if measured is not None and pred is not None and float(np.nanmax(np.abs(pred / np.maximum(measured, 1e-9) - 1))) > 0.4:
                obs.append("The measured swap acceptance (whole run, including the burn-in) and the one predicted from the stored log likelihoods (after the burn-in) differ by more than 40% for some pair.")
        if T == 1:
            swap_html = "<p class=\"muted\">There is one temperature (no tempering), so there are no swaps.</p>"
        elif both is not None:
            swap_html = (_table(["pair", "T low", "T high", "measured (whole run)", "predicted (after burn-in)", "predicted / measured"], pairs_rows)
                         + _img(_fig_ladder(E, pred, measured), "ladder and swaps", dpi))
        else:
            swap_html = "<p class=\"muted\">The backend has neither swap counters nor betas, so the swap acceptance cannot be shown.</p>"
        # ---- ln L by temperature
        meds = [float(np.nanmedian(LL[post_T][:, k])) for k in range(T)]
        lnl_rows = [[f"T{k}", _eryn_fmt(1.0 / betas[-1, k], ".3f") if betas is not None else "n/a", _eryn_fmt(meds[k], ".2f"),
                     _eryn_fmt(float(np.nanpercentile(LL[post_T][:, k], 16)), ".2f"), _eryn_fmt(float(np.nanpercentile(LL[post_T][:, k], 84)), ".2f")] for k in range(T)]
        ordered = all(meds[k] > meds[k + 1] for k in range(T - 1))
        if T > 1:
            checks.append(dict(key="lnl_order", label="ln likelihood falls with temperature", status="ok" if ordered else "bad",
                           value="medians " + " > ".join(f"{m:.1f}" for m in meds) if ordered else "medians " + ", ".join(f"{m:.1f}" for m in meds),
                           rule="Good if the median ln likelihood of every temperature is below that of the colder one. Otherwise the temperatures are not ordered as the ladder says (a wrong temperature axis, or a run that has not equilibrated)."))
        lnl_html = _table(["temperature", "T", "median ln L", "16%", "84%"], lnl_rows) + _img(_fig_lnl(E, start), "ln L by temperature", dpi)
        # ---- acceptance of the moves
        moves_html, acc0 = "", None
        if E["moves"]:
            mrows = []
            for nm in sorted(E["moves"]):
                a = E["moves"][nm]
                mrows.append([nm] + [_eryn_fmt(float(np.nanmean(a[k])), ".3f") for k in range(T)] + [f"{_eryn_fmt(float(np.nanmin(a[0])), '.3f')} to {_eryn_fmt(float(np.nanmax(a[0])), '.3f')}"])
            moves_html = _table(["move"] + [f"T{k}" for k in range(T)] + ["T0 walkers (min to max)"], mrows) + _img(_fig_moves(E), "move acceptance", dpi)
            first = E["moves"][sorted(E["moves"])[0]]
            acc0 = float(np.nanmean(first[0]))      # (the report's own acceptance check above judges it)
            if acc0 < 0.1:
                tips.append(f"The acceptance of the move is low at the cold temperature ({acc0:.3f}). A different or additional move (for example a differential-evolution move) is worth a trial, "
                            "judged by the population autocorrelation time and the number of evaluations, not by the per-walker time.")
        else:
            moves_html = "<p class=\"muted\">The backend has no per-move acceptance (track_moves was off).</p>"
        # ---- population autocorrelation and drift
        steps_P = E["steps_P"]
        rowsP = np.where(steps_P >= start)[0]
        if len(rowsP) < 16:
            rowsP = np.arange(len(steps_P) // 2, len(steps_P))
        tau_p, z_p, shift_p = population_tau(E["chain_P"], rowsP, E["stride_P"])
        names = run.col_names
        slot = {r["name"]: r["tau"] for r in an["params"]}
        post_steps = float(n - start)
        pop_html, drift_html = "", ""
        if np.isfinite(tau_p[0]).any():
            order = np.argsort(-np.nan_to_num(tau_p[0], nan=-1))[:10]
            prow = [[names[j], _eryn_fmt(slot.get(names[j]), ".1f"), _eryn_fmt(tau_p[0, j], ".0f"), _eryn_fmt(tau_p[0, j] / slot[names[j]], ".0f") if np.isfinite(slot.get(names[j], np.nan)) and slot[names[j]] > 0 else "n/a",
                     _eryn_fmt(post_steps / tau_p[0, j], ".1f") if np.isfinite(tau_p[0, j]) else "n/a"] for j in order]
            pop_html = (f"<p class=\"muted\">Autocorrelation time of the ensemble mean of each parameter in the cold chain, on the steps after {start_note} (step {start:,}). "
                        "Compared with the per-walker time of the Convergence section, which a swap refreshes within a few steps.</p>"
                        + _table(["parameter", "per-walker tau (steps)", "population tau (steps)", "ratio", "steps / population tau"], prow))
            jmax = int(np.nanargmax(tau_p[0]))
            checks.append(dict(key="pop_tau", label="Run length vs the population autocorrelation time", status=_status(post_steps / tau_p[0, jmax], 50, 20, True),
                               value=f"{post_steps / tau_p[0, jmax]:.1f} (population tau {tau_p[0, jmax]:.0f} steps, {names[jmax]})",
                               rule="Good if the steps after the burn-in are at least 50 population autocorrelation times; check from 20; problem below 20. This is the time that governs slow collective behaviour such as mode weights."))
            slot_max = np.nanmax([v for v in slot.values() if np.isfinite(v)]) if slot else np.nan
            if np.isfinite(slot_max) and tau_p[0, jmax] > 5 * max(slot_max, 1):
                obs.append(f"The per-walker autocorrelation time is at most {slot_max:.0f} steps, but the ensemble mean of {names[jmax]} has {tau_p[0, jmax]:.0f}: swaps refresh the walker slots quickly "
                           "while the population as a whole moves slowly, so the per-walker time says the run is much better converged than it is for mode weights and other collective quantities.")
                tips.append("Judge the length of the run by the population autocorrelation time; shares of walkers in each mode, followed by temperature, show whether they have stopped changing.")
            # drift by temperature: the shift of the ensemble mean between the two halves, in posterior standard deviations (and z, which needs the autocorrelation time)
            drows = []
            worst = (0.0, 0, 0)
            for t in range(T):
                sh = np.abs(shift_p[t])
                if np.isfinite(sh).any():
                    jj = int(np.nanargmax(sh))
                    drows.append([f"T{t}", _eryn_fmt(final_T[t], ".2f") if final_T is not None else "n/a", (_eryn_fmt(sh[jj], ".2f"), _st_cls(_status(float(sh[jj]), 0.1, 0.25))), names[jj],
                                  _eryn_fmt(abs(z_p[t, jj]), ".1f"), _eryn_fmt(np.nanmax(tau_p[t]), ".0f")])
                    if sh[jj] >= worst[0]:
                        worst = (float(sh[jj]), t, jj)
            drift_html = ("<p class=\"muted\">For every temperature: how far the ensemble mean of each parameter moved between the first and second half of the run after the burn-in, in units of that parameter's spread over "
                          "all walkers and steps. The largest over the parameters is shown, with the half-vs-half z in standard errors for comparison: z uses the population autocorrelation time, which is long "
                          "precisely when the mean is drifting, so a small z does not clear a large shift. A hot chain that is still drifting has not equilibrated, and the cold chain inherits that through the swaps.</p>"
                          + _table(["temperature", "T", "largest shift (posterior sd)", "parameter", "z", "largest population tau (steps)"], drows))
            checks.append(dict(key="drift_T", label="Shift of the ensemble mean at every temperature", status=_status(worst[0], 0.1, 0.25),
                               value=f"largest shift between the halves = {worst[0]:.2f} posterior sd (T{worst[1]}, {names[worst[2]]})",
                               rule="Good if no parameter's ensemble mean, at any temperature, moves by more than 0.1 of its spread between the first and second half after the burn-in; check up to 0.25; problem above."))
            if worst[0] >= 0.25 and worst[1] == T - 1:
                obs.append(f"The ensemble mean of {names[worst[2]]} at the hottest temperature moved by {worst[0]:.2f} posterior sd between the halves of the run: the hot end has not equilibrated, and the colder chains follow it through the swaps.")
                tips.append("A hotter top temperature (or more temperatures), or walkers started from a posterior that already contains every mode, shortens the equilibration of the whole ladder.")
            elif worst[0] >= 0.25:
                obs.append(f"The ensemble mean of {names[worst[2]]} at T{worst[1]} moved by {worst[0]:.2f} posterior sd between the halves of the run after the burn-in: it is still drifting.")
            pj = [int(j) for j in np.argsort(-np.nan_to_num(tau_p[0], nan=-1))[:6]]
            pop_fig = _img(_fig_population(E, pj, [names[j] for j in pj], start), "population means by temperature", dpi)
        else:
            pop_fig = ""
            pop_html = "<p class=\"muted\">Too few rows after the burn-in to estimate a population autocorrelation time.</p>"
        sections_html = (
            "<p class=\"muted\">A tempered ensemble runs the same walkers at several temperatures T = 1 / beta and swaps them between neighbours; the posterior is the temperature-1 chain (the cold chain analysed in the rest of this report). "
            f"The statistics below use {start_note}.</p>"
            + (f"<p class=\"muted\">{_html.escape(' '.join(E['notes']))}</p>" if E["notes"] else "")
            + "<h3>Temperature ladder</h3>" + (ladder_html or "<p class=\"muted\">no betas</p>")
            + "<h3>Swaps</h3><p class=\"muted\">Measured: swaps accepted over steps x walkers, from the backend counters (whole run, burn-in included). Predicted: from the stored ln likelihoods, "
              "pairing walkers of neighbouring temperatures at random after the burn-in; it needs no counter and shows what the ladder does now. Green band: 0.2 to 0.5.</p>" + swap_html
            + "<h3>ln likelihood by temperature</h3><p class=\"muted\">Temperatures must be ordered, and neighbours must overlap for swaps to be accepted.</p>" + lnl_html
            + "<h3>Move acceptance</h3>" + moves_html
            + "<h3>Population autocorrelation</h3>" + pop_html
            + ("<h3>Drift of every temperature</h3>" + drift_html + "<h3>Ensemble means by temperature</h3><p class=\"muted\">The six parameters with the longest population autocorrelation time. "
               "Every temperature should fluctuate around a flat level after the burn-in (purple dashed line).</p>" + pop_fig if drift_html else ""))
        cards = [("temperatures", f"{T}"), ("highest temperature", _eryn_fmt(final_T[-1], ".2f") if final_T is not None else "n/a"),
                 ("swap acceptance", f"{np.min(both):.2f} to {np.max(both):.2f}" if both is not None else "n/a", "lowest to highest pair"),
                 ("lnL evaluations, all temperatures", f"{n * W * T:,}", "steps x walkers x temperatures")]
        methods = ("<details><summary>Tempering diagnostics</summary><ul>"
                   "<li><b>Cold chain.</b> The report above is for temperature index 0 (beta = 1), the walkers being the walkers of that temperature. Its ln p is the ln likelihood plus the ln prior.</li>"
                   "<li><b>Ladder.</b> betas at every saved step (at a stride for long runs). Settled means that none changes by more than 2% over the last quarter of the run.</li>"
                   "<li><b>Swap acceptance.</b> Measured is the backend counter divided by steps x walkers (one swap attempt per walker pair per step); predicted uses min(1, exp((beta_k - beta_k+1)(lnL_k+1 - lnL_k))) with random pairing.</li>"
                   "<li><b>Population autocorrelation.</b> The ensemble mean of each parameter at each temperature is a time series with one value per step; its integrated autocorrelation time uses the same estimator as the rest of the report. "
                   "Drift is the shift of that mean between the two halves in units of the parameter's spread (the z in standard errors is also shown; it can be misleadingly small when the mean has a trend).</li>"
                   "<li><b>Limits.</b> The thresholds are rules of thumb. Replica round trips cannot be counted because the backend does not record which replica is where. "
                   "All statistics after the burn-in use the estimated burn-in of the cold chain (or half of the run when none was found).</li></ul></details>")
        header = f"; Eryn {E['version']}, {T} temperatures, cold chain shown" + (f", stride {E['stride']}" if E["stride"] > 1 else "")
        return dict(sections=[("tempering", "Tempering", sections_html)], checks=checks, diagnosis=obs, suggestions=tips, cards=cards, methods=methods, header=header)
    return extras


def eryn_report(args, kwargs=None, sampler_kwargs=None, out_path=None, safe=True, burn=0, max_steps=20000, title=None, spin_model=None, physical="args", branch=None, fixed=None,
                dpi=110, **build_kw):
    """The Eryn report: the emcee report (PART 1) on the cold chain, plus the Tempering section, checks and verdict."""
    def _go():
        path = getattr(args, "output_h5", None) or backend_path(args)
        sky, tm = bool(getattr(args, "vary_skypos", False)), bool(getattr(args, "vary_time", False))
        info = eryn_file_info(path, branch)
        ok = [m for m in ("precessing", "aligned", "none") if len(get_layout(m, sky, tm)) == info["ndim"]]
        if spin_model is None:
            if len(ok) != 1:
                raise ValueError(f"{path} has {info['ndim']} sampled columns, which does not match --vary-skypos={sky} --vary-time={tm}. "
                                 f"Run configurations with {info['ndim']} columns: {candidate_layouts(info['ndim'])}")
            sm = ok[0]
        else:
            sm = spin_model
        names = get_layout(sm, sky, tm)
        E = read_eryn_backend(path, names=names, max_steps=max_steps, branch=branch)
        phys = _physical_from_args(args, kwargs, sm, sky, tm, fixed) if (isinstance(physical, str) and physical == "args") else physical
        out = out_path or default_out_path(args)
        an = build_report(E["run"], out, title=title or f"eryn diagnostics: {os.path.basename(path)}", burn=burn, physical=phys, arguments=args, reference_values=kwargs,
                             extras=make_eryn_extras(E, dpi), kind="eryn", dpi=dpi, **build_kw)
        _print_summary(an)
        return an

    if not safe:
        return _go()
    try:
        return _go()
    except Exception as error:                                   # a diagnostics problem must not break the run
        print(f"WARNING: the eryn diagnostics report could not be made: {type(error).__name__}: {error}")
        return None


# ============================================================================================================================================
# command line
# ============================================================================================================================================
def _eryn_cli(argv):
    ap = argparse.ArgumentParser(prog="sampler_diagnostics.py eryn", description="HTML diagnostics of a finished Eryn run from its backend (.h5): the cold chain through the emcee report, "
                                 "and the Tempering section.", epilog=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="the Eryn backend .h5")
    ap.add_argument("-o", "--out", default=None, help="report to write (default: <path without .h5>_diagnostics.html)")
    ap.add_argument("--vary-skypos", action="store_true", help="the run sampled the sky position (tells the layout of the columns)")
    ap.add_argument("--vary-time", action="store_true", help="the run sampled the geocenter time")
    ap.add_argument("--max-steps", type=int, default=20000, help="read the cold chain with a stride when there are more steps than this (default 20000)")
    ap.add_argument("--burn", default="0", help="what to discard before the statistics: a number of steps, a fraction of the run (below 1), or auto (the estimated burn-in). "
                                                "Default: nothing; the estimated burn-in is shown")
    a = ap.parse_args(argv)
    ns = argparse.Namespace(output=a.path, output_h5=a.path, vary_skypos=a.vary_skypos, vary_time=a.vary_time, reference_parameters=None)
    out = a.out or (re.sub(r"\.h5$", "", a.path) + "_diagnostics.html")
    run("eryn", ns, out_path=out, safe=False, max_steps=a.max_steps, burn=_parse_burn(a.burn))


def main(argv=None):
    """python sampler_diagnostics.py {emcee|eryn|bilby_mcmc} ...; without a sampler name the old emcee command line is understood."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "emcee":
        return emcee_main(argv[1:])
    if argv and argv[0] == "bilby_mcmc":
        return bilby_main(argv[1:])
    if argv and argv[0] == "eryn":
        return _eryn_cli(argv[1:])
    if argv and argv[0] in ("-h", "--help") or not argv:
        print(__doc__)
        return None
    return emcee_main(argv)


if __name__ == "__main__":
    main()
