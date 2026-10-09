"""
emcee_diagnostics: exploration and convergence diagnostics for emcee output,
with a self-contained HTML report. One file: it needs only numpy, scipy,
matplotlib (and h5py to read files).

Defaults: NO thinning and NO burn-in. Everything is computed on the chain exactly
as it is in the file. The estimated burn-in is shown (dashed line, Burn-in section)
and the Burn-in section shows how R-hat, N/tau and drift change as steps are
discarded, but nothing is discarded unless you ask (burn=... / thin=...).

With run_sampler.py
-------------------
    at the end of main():   from tdinf import emcee_diagnostics as ed; ed.report_from_args(args, kwargs=kwargs)
    or from the shell:      python emcee_diagnostics.py -o full/full.h5 --vary-skypos --vary-time   (run_sampler's own flags)

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
import argparse
import base64
import html as _html
import io
import itertools
import json
import os
import re
import sys
import time
from contextlib import contextmanager

import numpy as np
from scipy.special import ndtri
from scipy.stats import binom, rankdata

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
from matplotlib.gridspec import GridSpec               # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator   # noqa: E402

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
                 **analysis_kwargs):
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
    title = title or f"emcee diagnostics: {run.source}"
    verdict_text = {"ok": "No red flags in the automated checks", "warn": "Some checks need attention",
                    "bad": "Problems detected: do not trust these chains yet", "na": "Not enough information"}[an["verdict"]]

    cards = [("steps in file", f"{run.nsteps_total:,}"), ("walkers", f"{run.nwalkers}"),
             ("lnL evaluations", f"{run.nsteps_total * run.nwalkers:,}", "steps x walkers"),
             ("estimated burn-in", f"{an['est_step']:,}" if an["est_found"] else "none found"),
             ("steps analysed", f"{an['n_post'] * run.stride:,}"), ("flagged walkers", f"{len(an['flagged'])}")]
    if an["burn_idx"] > 0:
        cards.insert(4, ("burn-in removed", f"{an['burn_step']:,}"))
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
                '</ul></details></section>')

    nav = "".join(f'<a href="#{i}">{t}</a>' for i, t in (("overview", "Overview"), ("arguments", "Arguments"), ("lnp", "Log-probability"), ("burnin", "Burn-in"),
                                                          ("walkers", "Walkers"), ("traces", "Traces"),
                                                          ("convergence", "Convergence"), ("methods", "Methods")))
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{_html.escape(title)}</title><style>{_CSS}</style></head><body>'
            f'<header><h1>{_html.escape(title)}</h1><p>{run.nsteps_total:,} steps, {run.nwalkers} walkers, {run.ndim} sampled numbers</p></header>'
            f'<div class="layout"><nav>{nav}</nav><main>{"".join(secs)}</main></div>'
            f'<footer>report generated in {time.time() - t0:.0f} s by emcee_diagnostics.py</footer><script>{_SORT_JS}</script></body></html>')
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
        phys = physical
        if isinstance(physical, str) and physical == "args":
            phys = dict(spin_model=sm, vary_skypos=sky, vary_time=tm)
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


def main(argv=None):
    """
    Command line that understands run_sampler.py's arguments. You can paste the run_sampler command line; flags that
    are not about the diagnostics are ignored. Example:
        python emcee_diagnostics.py -o full/full.h5 --vary-skypos --vary-time --total-mass-prior-bounds 150 400
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


if __name__ == "__main__":
    main()
