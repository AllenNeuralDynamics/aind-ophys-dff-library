"""Configuration for the triexp dF/F pipeline: DffConfig dataclass and set_dff_config factory."""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from aind_ophys_utils.signal_utils import noise_std as _noise_std


@dataclass(frozen=True)
class DffConfig:
    """Pre-computed per-ROI fit setup returned by ``set_dff_config``.

    Attributes
    ----------
    params : dict
        Optional, unverified snapshot of the scalar inputs to
        ``set_dff_config`` plus a few derived scalars (n_skip, t_max).
        Intended as a JSON-loggable reproducibility record; not read by
        ``dff()``.  Empty by default.  Keyword-only.
    n_skip : int
        Frames removed from the start of the trace before fitting.
    t_rel : (T_fit,) np.ndarray
        Timestamps of the fit window relative to session start (s).
    x0_all : (N, 7) np.ndarray
        Per-ROI initial parameter vectors
        [b_inf, b_slow, t_slow, b_fast, t_fast, b_bright, t_bright].
        Note: b_bright is encoded as a NEGATIVE amplitude under the
        sum_of_exps convention.
    bounds_all : list[list[tuple]]
        Per-ROI bounds — outer list length N, inner list of 7 (lb, ub) tuples.
    sigma_all : (N,) np.ndarray
        Per-ROI noise std (MAD).
    min_frac_below_f0 : float
        Threshold for the low-F0 check in Pass 1.
    tukey_param_combos : tuple of (c_pos, c_neg) pairs
        Asymmetric Tukey biweight thresholds swept during fitting.
    """

    params: dict = field(default_factory=dict, kw_only=True)
    n_skip: int
    t_rel: np.ndarray
    x0_all: np.ndarray
    bounds_all: list
    sigma_all: np.ndarray
    min_frac_below_f0: float
    tukey_param_combos: Tuple[Tuple[int, int], ...]


def set_dff_config(
    F: np.ndarray,
    fs: float = 30.0,
    ts: Optional[np.ndarray] = None,
    # initial values
    b_inf_n_frames: int = 1000,
    t_fast_init_s: float = 60.0,
    # amplitude bound multipliers
    b_slow_max_factor: float = 2.0,
    b_bright_max_factor: float = 1.0,
    b_fast_ptp_window_s: float = 300.0,
    b_inf_lb_factor: float = 1.0,
    # time-constant bounds in seconds
    t_fast_min_s: float = 30.0,
    t_fast_max_s: float = 300.0,
    t_slow_min_s: float = 600.0,
    t_bright_min_s: float = 600.0,
    t_slow_min_tmax_factor: float = 0.25,
    t_bright_min_tmax_factor: float = 0.25,
    t_high_factor: float = 5.0,
    # trim
    skip_initial_s: float = 5.0,
    # low-F0 detection
    min_frac_below_f0: float = 0.05,
    # fitting combos
    tukey_param_combos: Tuple[Tuple[int, int], ...] = (
        (2, 3), (2, 4), (2, 5), (3, 4), (3, 5)
    ),
) -> DffConfig:
    """Pre-compute per-ROI initial values, bounds, and noise estimates.

    All parameters are plain numbers (no arrays). The returned DffConfig is
    passed directly to dff().

    Parameters
    ----------
    F : (N, T) or (T,) array
        Neuropil-corrected fluorescence.
    fs : float
        Sampling frequency in Hz.  Used when ts is None (the typical case):
        n_skip = int(skip_initial_s * fs) and t_rel is built from uniform
        spacing starting at ~skip_initial_s.
    ts : (T,) array or None
        Actual timestamps in seconds (relative to session start), same length
        as F.  Pass ts instead of relying on fs when timestamps are irregular
        (dropped frames, variable rate): n_skip is then found via searchsorted
        and t_rel = ts[n_skip:] - ts[0].  fs and ts should not be passed
        together; if ts is provided, fs is ignored.
    b_inf_n_frames : int
        Number of frames from the end of the trace used to initialise b_inf.
        The tail mean approximates the asymptotic fluorescence floor.
    t_fast_init_s : float
        Initial value for t_fast in seconds.
    b_slow_max_factor : float
        b_slow upper bound = factor × (P99-P1) of the fit window.
    b_bright_max_factor : float
        b_bright upper bound = factor × (P99-P1) of the fit window.
    b_fast_ptp_window_s : float
        b_fast upper bound = (P99-P1) of the first b_fast_ptp_window_s seconds
        of the fit window.  Suppresses convex arcs driven by the early transient.
    b_inf_lb_factor : float
        b_inf lower bound = factor × P1(F_fit).
    t_fast_min_s, t_fast_max_s : float
        Hard bounds for t_fast in seconds.
    t_slow_min_s, t_bright_min_s : float
        Absolute minima for t_slow and t_bright in seconds.
    t_slow_min_tmax_factor, t_bright_min_tmax_factor : float
        t_slow / t_bright lower bound = max(absolute_min, factor × T_total).
        Short t_bright is the primary driver of low_f0 fits — keep >= 0.25*T.
    t_high_factor : float
        t_slow / t_bright upper bound = factor × T_total.
    skip_initial_s : float
        Seconds to discard from the start before fitting.
    min_frac_below_f0 : float
        Minimum fraction of frames that must lie below the analytic F0 for a
        Pass 1 winner to be accepted without triggering Pass 2.  A "low F0"
        fit is one where ``mean(F < F0_analytic) < min_frac_below_f0``:
        almost no frames fall below the estimated baseline, meaning F0 is
        suppressed below the bulk of the raw fluorescence and the signal
        floor is not captured.  Typically caused by the b_bright term
        pulling F0 down throughout the session.  Default 0.05 (5%).
        Must be in [0, 1); values below 0.001 or above 0.5 trigger a
        warning.
    tukey_param_combos : tuple of (c_pos, c_neg) pairs
        Asymmetric Tukey biweight thresholds swept during fitting.
        c_pos / c_neg are the rejection widths (in units of sigma) above /
        below the current F0.  Default: (2,3) (2,4) (2,5) (3,4) (3,5).

    Returns
    -------
    DffConfig
        Frozen dataclass with fields:
            params             : dict      — JSON-serializable snapshot of scalar
                                              inputs + derived scalars (n_skip,
                                              t_max).  For reproducibility logging;
                                              not read by dff().
            n_skip             : int        — frames removed from the start
            t_rel              : (T_fit,)  — timestamps of the fit window relative
                                              to session start (s); starts at
                                              ~skip_initial_s
            x0_all             : (N, 7)    — per-ROI initial parameter vectors
            bounds_all         : list[N]   — per-ROI bounds (list of 7 (lb, ub) tuples)
            sigma_all          : (N,)      — per-ROI noise std (MAD)
            min_frac_below_f0  : float     — passed through to dff()
            tukey_param_combos : tuple     — passed through to dff()
    """
    if not (0 <= min_frac_below_f0 < 1):
        raise ValueError(
            f"min_frac_below_f0 must be in [0, 1), got {min_frac_below_f0}"
        )
    if min_frac_below_f0 < 0.001:
        warnings.warn(
            f"min_frac_below_f0={min_frac_below_f0} is below 0.001. At this"
            " threshold the low_f0 check is effectively disabled — fits will"
            " almost never be flagged regardless of how low F0 sits.",
            UserWarning, stacklevel=2,
        )
    if min_frac_below_f0 > 0.5:
        warnings.warn(
            f"min_frac_below_f0={min_frac_below_f0} is above 0.5. A fit is"
            " flagged as low_f0 when fewer than this fraction of frames lie"
            " below F0, so a threshold above 0.5 means the majority of fits"
            " will be flagged.",
            UserWarning, stacklevel=2,
        )
    F2d = np.atleast_2d(np.asarray(F, dtype=np.float64))
    N = F2d.shape[0]

    if ts is not None and len(ts) != F2d.shape[1]:
        raise ValueError(
            f"ts length ({len(ts)}) does not match F time axis ({F2d.shape[1]})"
        )

    ts_provided = ts is not None
    if ts_provided:
        ts_arr = np.asarray(ts, dtype=np.float64)
        t0 = float(ts_arr[0])
        n_skip = int(np.searchsorted(ts_arr - t0, skip_initial_s))
        F_fit = F2d[:, n_skip:]
        T_fit = F_fit.shape[1]
        t_rel = ts_arr[n_skip:] - t0
    else:
        n_skip = int(skip_initial_s * fs)
        F_fit = F2d[:, n_skip:]
        T_fit = F_fit.shape[1]
        # start at ~skip_initial_s, not 0, so the model is evaluated at the
        # correct physical time (bleaching decay already partially completed)
        t_rel = (np.arange(T_fit) + n_skip) / fs

    if T_fit == 0:
        raise ValueError(
            f"No frames remain after skipping {skip_initial_s} s "
            f"(n_skip={n_skip}, total frames={F2d.shape[1]}). "
            "Reduce skip_initial_s or check the input length."
        )

    t_max = float(t_rel[-1])

    # Robust ptp per ROI
    p1 = np.percentile(F_fit, 1, axis=1)
    p99 = np.percentile(F_fit, 99, axis=1)
    rptp = p99 - p1

    # b_fast upper bound: robust ptp of first b_fast_ptp_window_s seconds
    # Use t_rel (not fs) so this is correct whether ts or fs was provided.
    n_fast_win = max(1, min(int(np.searchsorted(t_rel - t_rel[0], b_fast_ptp_window_s)), T_fit))
    b_fast_ubs = (np.percentile(F_fit[:, :n_fast_win], 99, axis=1)
                - np.percentile(F_fit[:, :n_fast_win], 1, axis=1)).astype(float)

    b_slow_ubs = (b_slow_max_factor * rptp).astype(float)
    b_bright_ubs = (b_bright_max_factor * rptp).astype(float)
    b_inf_lbs = (b_inf_lb_factor * p1).astype(float)

    t_slow_lb = max(t_slow_min_s, t_slow_min_tmax_factor * t_max)
    t_bright_lb = max(t_bright_min_s, t_bright_min_tmax_factor * t_max)
    t_high = max(t_high_factor * t_max, t_slow_lb, t_bright_lb)

    # Initial values: all bleach/bright amplitudes start at 0 (binit0)
    n_tail = min(b_inf_n_frames, T_fit)
    b_inf_init = F_fit[:, -n_tail:].mean(axis=1)
    t_init = 0.5 * t_max

    x0_all = np.column_stack([
        b_inf_init,                  # 0: b_inf
        np.zeros(N),                 # 1: b_slow
        np.full(N, t_init),          # 2: t_slow
        np.zeros(N),                 # 3: b_fast
        np.full(N, t_fast_init_s),   # 4: t_fast
        np.zeros(N),                 # 5: b_bright (≤ 0; starts at 0 — no bright term initially)
        np.full(N, t_init),          # 6: t_bright
    ])  # (N, 7), interleaved: [b_inf, b_slow, t_slow, b_fast, t_fast, b_bright, t_bright]

    bounds_all = [
        [
            (float(b_inf_lbs[i]), None),         # 0: b_inf  >= b_inf_lb_factor * P1(F)
            (0.0, float(b_slow_ubs[i])),            # 1: b_slow ∈ [0, b_slow_max_factor * rptp]
            (t_slow_lb, t_high),                 # 2: t_slow
            (0.0, float(b_fast_ubs[i])),           # 3: b_fast ∈ [0, rptp of first 300 s]
            (t_fast_min_s, t_fast_max_s),        # 4: t_fast
            (-float(b_bright_ubs[i]), 0.0),        # 5: b_bright ∈ [-b_bright_max_factor*rptp, 0]
            (t_bright_lb, t_high),               # 6: t_bright
        ]
        for i in range(N)
    ]

    sigma_all = np.asarray(
        _noise_std(F_fit, method="mad", device="cpu"), dtype=np.float64
    )

    return DffConfig(
        params={
            "ts_provided":              bool(ts_provided),
            "fs":                       float(fs) if not ts_provided else None,
            "skip_initial_s":           float(skip_initial_s),
            "n_skip":                   int(n_skip),
            "t_max":                    float(t_max),
            "b_inf_n_frames":           int(b_inf_n_frames),
            "t_fast_init_s":            float(t_fast_init_s),
            "b_slow_max_factor":        float(b_slow_max_factor),
            "b_bright_max_factor":      float(b_bright_max_factor),
            "b_fast_ptp_window_s":      float(b_fast_ptp_window_s),
            "b_inf_lb_factor":          float(b_inf_lb_factor),
            "t_fast_min_s":             float(t_fast_min_s),
            "t_fast_max_s":             float(t_fast_max_s),
            "t_slow_min_s":             float(t_slow_min_s),
            "t_bright_min_s":           float(t_bright_min_s),
            "t_slow_min_tmax_factor":   float(t_slow_min_tmax_factor),
            "t_bright_min_tmax_factor": float(t_bright_min_tmax_factor),
            "t_high_factor":            float(t_high_factor),
            "min_frac_below_f0":        float(min_frac_below_f0),
            "tukey_param_combos":       [list(c) for c in tukey_param_combos],
        },
        n_skip=n_skip,
        t_rel=t_rel,
        x0_all=x0_all,
        bounds_all=bounds_all,
        sigma_all=sigma_all,
        min_frac_below_f0=min_frac_below_f0,
        tukey_param_combos=tukey_param_combos,
    )
