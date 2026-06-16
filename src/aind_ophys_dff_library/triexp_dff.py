"""dF/F using a triexponential baseline fit.

Typical usage
-------------
    config = set_dff_config(F, fs=30.0)
    dFF, F0, noise_sd, params, logs = dff(F, config)

If timestamps are irregular (dropped frames, variable rate), pass ts instead:
    config = set_dff_config(F, ts=ts)
    dFF, F0, noise_sd, params, logs = dff(F, config)

Or with custom parameters:
    config = set_dff_config(F, fs=30.0,
                            b_inf_n_frames=500, b_slow_max_factor=3.0,
                            tukey_param_combos=((2, 3), (2, 5), (3, 5)))
    dFF, F0, noise_sd, params, logs = dff(F, config)

Model
-----
F0(t) = b_inf
      + b_slow   * exp(-t / t_slow)
      + b_fast   * exp(-t / t_fast)
      + b_bright * exp(-t / t_bright)   # b_bright ≤ 0 (encodes brightening-suppression)

Three-pass fitting (at ROI level, not per-combo):
  Pass 1 — all combos, standard IRLS with fixed_sigma = noise_std.
            → winner selection; done if winner exists and is not low_f0.
  Pass 2 — all combos, sigma relaxation to escape zero-gradient traps.
            → winner selection; done if any valid winner exists
              (accept even if low_f0 — Pass 3 cannot help).
  Pass 3 — all combos, per-combo b_bright upper-bound constraint
            derived from Pass 2 params; dual x0 (recipe default and
            Pass 2 params with b_bright clipped).
            → accept best available winner.

Winner selection:
  Apply three eligibility filters per combo (analytic F0 min ≥ 1,
  finite med_neg, no extrapolation error), then pick the combo whose
  median |negative residual| is closest to 0.674 * noise_std.

Nomenclature:
  "analytic F0" — raw triexponential model output (unclamped).
  "F0"          — analytic F0 clamped to the per-ROI noise floor:
                    F0 = max(analytic F0, noise_sd)
                  This is what dff() returns and what dF/F is computed from.

Notes
-----
Imports the generic primitives from aind_ophys_utils.baseline_fitting
(sum_of_exps, nonlinear_fit, AsymmetricTukeyBiweight) and
aind_ophys_utils.signal_utils.noise_std (via dff_config.py).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import brentq as _brentq

from aind_ophys_utils.baseline_fitting import (
    AsymmetricTukeyBiweight,
    nonlinear_fit,
    sum_of_exps,
)
import jax.numpy as jnp

from .dff_config import DffConfig


# ---------------------------------------------------------------------------
# Boundary fit error filter (eligibility filter 3)
# ---------------------------------------------------------------------------

def _boundary_fit_error(
    params: np.ndarray,
    F_roi: np.ndarray,
    t_rel: np.ndarray,
    dt: float,
    f0: Optional[np.ndarray] = None,
) -> bool:
    """Return True if the fit has a Pattern A or B boundary artifact.

    Pattern A — b_bright overshoot at the start:
        F0 dips > 50% below b_inf in the first 600 s, while raw F stays
        entirely above F0 in the beginning window.

    Pattern B — bleach extrapolation at the end:
        F0 is still > 50% above b_inf at session end and still falling,
        while raw F stays entirely below F0 in the end window.

    Entry guard: skip if b_inf <= 0 or t_bright / t_max >= 1.5.

    Parameters
    ----------
    dt : median frame interval of t_rel in seconds.  Pre-computed once per
        ROI and passed in so np.diff(t_rel) is not recomputed per combo.
    f0 : pre-computed analytic F0 array, or None to compute from params.
    """
    b_inf, b_slow, t_slow, b_fast, t_fast, b_bright, t_bright = [float(p) for p in params]
    t_max = float(t_rel[-1])

    if b_inf <= 0 or (t_bright / t_max) >= 1.5:
        return False

    if f0 is None:
        f0 = sum_of_exps(params, t_rel)
    n_win = min(int(max(300.0, 0.10 * t_max) / dt), len(t_rel) // 3)
    n_win_600 = min(n_win, int(600.0 / dt))

    # Pattern A
    if ((b_inf - float(np.min(f0[:n_win_600]))) / b_inf > 0.5
            and np.all(F_roi[:n_win] - f0[:n_win] >= 0)):
        return True

    # Pattern B — derivative of F0 at t_end
    d1_last = (- b_slow / t_slow * np.exp(-t_rel[-1] / t_slow)
               - b_fast / t_fast * np.exp(-t_rel[-1] / t_fast)
               - b_bright / t_bright * np.exp(-t_rel[-1] / t_bright))
    if ((float(f0[-1]) - b_inf) / b_inf > 0.5
            and d1_last < 0
            and np.all(F_roi[-n_win:] - f0[-n_win:] <= 0)):
        return True

    return False


# ---------------------------------------------------------------------------
# Per-pass helpers (module-level for joblib pickling)
# ---------------------------------------------------------------------------

def _compute_sigma_relax(M, sigma: float) -> float:
    """Relaxed sigma for Pass 2: sigma * 2.0 / z_half.

    z_half is the z in (0, 2) where min(M.weights(z), M.weights(-z)) = 0.5.
    Falls back to sigma unchanged if M weights don't drop below 0.5 in [0, 2].
    """
    if float(min(M.weights(2.0), M.weights(-2.0))) >= 0.5:
        return sigma
    z_half = _brentq(
        lambda z: float(min(M.weights(z), M.weights(-z))) - 0.5,
        0.0, 2.0,
    )
    return sigma * 2.0 / z_half


def _fit_one(
    F_row: np.ndarray,
    t_rel: np.ndarray,
    x0: np.ndarray,
    bounds: list,
    M: AsymmetricTukeyBiweight,
    fixed_sigma: float,
) -> Tuple[np.ndarray, np.ndarray, int, bool]:
    """Single-pass IRLS for one ROI, one combo.

    Returns (F0_analytic, params, irls_nit, converged).
    irls_nit: number of IRLS outer reweighting iterations.
    converged: whether the scipy optimizer succeeded in the final IRLS step.
    """
    # float64 is required: float32 causes premature L-BFGS-B convergence
    # (ftol=1e-12 triggers float32 numerical noise at ~1e-7 relative precision,
    # causing the OLS pre-pass to stop before escaping flat local minima).
    F0_analytic, res = nonlinear_fit(
        F_row, t_rel, sum_of_exps, x0,
        bounds=bounds,
        M=M,
        fixed_sigma=fixed_sigma,
        maxiter=5,
        tol=1e-3,
        sigma_relax_threshold=0.0,    # disable internal sigma relax
        backend="jax",
        dtype=jnp.float64,
    )
    return (np.asarray(F0_analytic, dtype=np.float32),
            np.asarray(res.x, dtype=np.float64),
            int(getattr(res, "irls_nit", -1)),
            bool(getattr(res, "success", False)))


def _select_winner(
    F_row: np.ndarray,
    F0_analytics: dict,
    params_store: dict,
    target: float,
    t_rel: np.ndarray,
    dt: float,
    combos: tuple,
) -> Tuple[Optional[tuple], dict]:
    """Apply eligibility filters and return ``(winner_combo, per_combo_stats)``.

    ``winner_combo`` is the eligible combo whose ``score = abs(med_neg - target)``
    is minimized, or ``None`` if no combo passed all three eligibility filters.

    ``per_combo_stats[combo]`` is a dict with three fields, populated for
    every combo (including ineligible ones):

    - ``med_neg``         : float | None — median absolute negative residual.
                            ``None`` only if Filter 1 (≥10 negative residuals
                            with finite median) rejected the combo.
    - ``score``           : float | None — ``abs(med_neg - target)``. ``None``
                            iff ``med_neg`` is ``None``. Defined for combos
                            rejected by Filter 2 / Filter 3 so callers can
                            rank ineligible combos by structural distance to
                            the target if they wish.
    - ``filter_failure``  : str | None — eligibility tag. ``None`` means the
                            combo passed all filters and is winner-eligible.
                            Possible failure tags:
                              * ``"few_neg_residuals"`` — fewer than 10
                                negative residuals (Filter 1).
                              * ``"med_neg_not_finite"`` — Filter 1 passed
                                count check but ``med_neg`` was non-finite.
                              * ``"f0_lt_1"``  — analytic F0 minimum below
                                1.0 (Filter 2).
                              * ``"boundary_extrap"`` — extrapolation /
                                boundary error (Filter 3).
    """
    stats: dict = {}
    best_combo: Optional[tuple] = None
    best_dist = np.inf
    for combo in combos:
        params = params_store[combo]
        F0_analytic = F0_analytics[combo]

        # Filter 1 (doc §7): finite med_neg (need > 10 negative residuals)
        resid = F_row - F0_analytic.astype(np.float64)
        neg = resid[resid < 0]
        if len(neg) <= 10:
            stats[combo] = {"med_neg": None, "score": None,
                            "filter_failure": "few_neg_residuals"}
            continue
        med_neg = float(np.median(np.abs(neg)))
        if not np.isfinite(med_neg):
            stats[combo] = {"med_neg": None, "score": None,
                            "filter_failure": "med_neg_not_finite"}
            continue
        score = abs(med_neg - target)

        # Filter 2 (doc §7): analytic (unclamped) F0 min >= 1.0
        f0_analytic = sum_of_exps(params, t_rel)
        if float(f0_analytic.min()) < 1.0:
            stats[combo] = {"med_neg": med_neg, "score": score,
                            "filter_failure": "f0_lt_1"}
            continue

        # Filter 3 (doc §7): no extrapolation error (reuse already-computed f0_analytic)
        if _boundary_fit_error(params, F_row, t_rel, dt, f0=f0_analytic):
            stats[combo] = {"med_neg": med_neg, "score": score,
                            "filter_failure": "boundary_extrap"}
            continue

        stats[combo] = {"med_neg": med_neg, "score": score, "filter_failure": None}
        if score < best_dist:
            best_dist = score
            best_combo = combo

    return best_combo, stats


def _is_low_f0(
    F_row: np.ndarray,
    params: np.ndarray,
    t_rel: np.ndarray,
    min_frac_below_f0: float = 0.05,
) -> bool:
    """True if the fitted F0 is too low (baseline underestimated).

    Fewer than ``min_frac_below_f0`` fraction of frames have F < F0, meaning
    F0 sits below the bulk of the raw fluorescence.  This indicates the
    baseline is underestimated — almost no data falls below the fitted F0,
    so the signal floor is not being captured.  Typically caused by the
    b_bright term suppressing F0 throughout the session.

    Parameters
    ----------
    min_frac_below_f0 : float
        Minimum acceptable fraction of frames below F0.  Default 0.05 (5%).
    """
    f0 = sum_of_exps(params, t_rel)
    return float(np.mean(F_row < f0)) < min_frac_below_f0


def _process_roi(  # noqa: C901
    i: int,
    F_fit: np.ndarray,
    t_rel: np.ndarray,
    x0_all: np.ndarray,
    bounds_all: list,
    sigma_all: np.ndarray,
    tukey_param_combos: tuple,
    noise_floor: np.ndarray,
    min_frac_below_f0: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Run all 3 passes for ROI i.

    Returns
    -------
    F0     : (T_fit,) float32  — analytic F0 clamped to noise floor
    params  : (7,) float64      — winner model parameters
    log     : dict              — pass diagnostics (see below)

    Log keys
    --------
    n_passes          : int — number of passes run (1, 2, or 3)
    winner_combo      : (c_pos, c_neg) — final accepted winner
    winner_params     : (7,) float64 — model parameters of the winner. Convenience
                        shortcut for ``log[f"pass{winner_pass}"][winner_combo]["params"]``.
    winner_pass       : 1 | 2 | 3 — which pass produced the winning fit.
    pass1_trigger     : "low_f0" | "no_winner" — why Pass 2 was entered (None if Pass 1 won)
    pass1_winner      : (c_pos, c_neg) | None — best Pass 1 combo before low_f0 check;
                        None if no combo passed eligibility filters
    pass2_winner      : (c_pos, c_neg) | None — best Pass 2 combo;
                        None means all combos failed eligibility → Pass 3 was entered
    pass1             : {combo: {<per-combo entry>}, ...}
    pass2             : same shape as pass1, or None if Pass 2 was not entered
    pass2_sigma_relax : {combo: float, ...} or None
    pass3             : same shape as pass1 plus an extra ``used_x0_B`` bool,
                        or None if Pass 3 was not run

    Per-combo entry shape (in ``pass1`` / ``pass2`` / ``pass3``)
    ------------------------------------------------------------
    Every combo gets an entry — the winner is stored alongside the losers with
    the same fields; identify the winner via ``log["winner_combo"]`` and
    ``log["winner_pass"]``.

    - ``irls_nit``       : int     — IRLS iterations to convergence.
    - ``converged``      : bool    — IRLS converged (True) or hit max-iters (False).
    - ``params``         : (7,) float64 — fitted parameter vector
                           ``[b_inf, b_slow, t_slow, b_fast, t_fast, b_bright, t_bright]``.
    - ``med_neg``        : float | None — median absolute negative residual.
                           ``None`` only if Filter 1 rejected the combo (fewer
                           than 10 negative residuals, or non-finite median).
    - ``score``          : float | None — ``abs(med_neg - target)``; the value
                           the winner selection minimizes. ``None`` iff
                           ``med_neg`` is ``None``. Populated for combos
                           rejected by Filter 2 / Filter 3 so callers can rank
                           ineligible combos by structural distance to target.
    - ``filter_failure`` : str | None — eligibility tag, ``None`` for combos
                           that passed all 3 filters. Tags:
                             * ``"few_neg_residuals"`` — Filter 1 count check.
                             * ``"med_neg_not_finite"`` — Filter 1 finiteness check.
                             * ``"f0_lt_1"`` — Filter 2 (analytic F0 minimum < 1.0).
                             * ``"boundary_extrap"`` — Filter 3 (extrapolation error).
    - ``used_x0_B``      : bool — Pass 3 only — True if the alt-x0 path with
                           the b_bright lower-bound clamp won the per-combo
                           A vs B loss comparison.

    Pass-3 absolute-fallback caveat
    -------------------------------
    When all combos still fail eligibility after Pass 3, the chosen
    ``winner_combo`` is ``tukey_param_combos[0]`` (absolute fallback). In that
    case the winner's own ``pass3`` entry will have ``filter_failure != None`` —
    the asymmetry is real (no eligible combo was found) and is preserved
    honestly rather than masked.
    """
    F_row = F_fit[i]
    sigma = float(sigma_all[i])
    x0 = x0_all[i].astype(np.float64)
    bounds = bounds_all[i]
    tgt = 0.674 * sigma
    dt = float(np.median(np.diff(t_rel)))  # shared across all combos and passes

    log: dict = {
        "n_passes": 0,
        "winner_combo": None,
        "winner_params": None,
        "winner_pass": None,
        "pass1_trigger": None,
        "pass1_winner": None,
        "pass2_winner": None,
        "pass1": {},
        "pass2": None,
        "pass2_sigma_relax": None,
        "pass3": None,
    }

    # ── Pass 1: standard IRLS ─────────────────────────────────────────────
    p1_analytic, p1_params = {}, {}
    for combo in tukey_param_combos:
        M = AsymmetricTukeyBiweight(c_pos=combo[0], c_neg=combo[1])
        f0, px, irls_nit, converged = _fit_one(F_row, t_rel, x0, bounds, M, sigma)
        p1_analytic[combo], p1_params[combo] = f0, px
        log["pass1"][combo] = {"irls_nit": irls_nit, "converged": converged}

    winner, p1_stats = _select_winner(
        F_row, p1_analytic, p1_params, tgt, t_rel, dt, tukey_param_combos
    )
    log["pass1_winner"] = winner
    for combo in tukey_param_combos:
        log["pass1"][combo].update(p1_stats[combo])
        log["pass1"][combo]["params"] = p1_params[combo]
    if winner is not None:
        if not _is_low_f0(F_row, p1_params[winner], t_rel, min_frac_below_f0):
            log["n_passes"] = 1
            log["winner_pass"] = 1
            log["winner_combo"] = winner
            log["winner_params"] = p1_params[winner]
            return np.maximum(p1_analytic[winner], noise_floor[i]), p1_params[winner], log
        log["pass1_trigger"] = "low_f0"
    else:
        log["pass1_trigger"] = "no_winner"

    # ── Pass 2: sigma relaxation ──────────────────────────────────────────
    p2_analytic, p2_params, sigma_relax = {}, {}, {}
    log["pass2"] = {}
    log["pass2_sigma_relax"] = {}
    for combo in tukey_param_combos:
        M = AsymmetricTukeyBiweight(c_pos=combo[0], c_neg=combo[1])
        sr = _compute_sigma_relax(M, sigma)
        sigma_relax[combo] = sr
        log["pass2_sigma_relax"][combo] = sr
        f0, px, irls_nit, converged = _fit_one(F_row, t_rel, x0, bounds, M, sr)
        p2_analytic[combo], p2_params[combo] = f0, px
        log["pass2"][combo] = {"irls_nit": irls_nit, "converged": converged}

    winner, p2_stats = _select_winner(
        F_row, p2_analytic, p2_params, tgt, t_rel, dt, tukey_param_combos
    )
    log["pass2_winner"] = winner
    for combo in tukey_param_combos:
        log["pass2"][combo].update(p2_stats[combo])
        log["pass2"][combo]["params"] = p2_params[combo]
    # Accept Pass 2 winner even if low_f0 — Pass 3 cannot fix a low_f0 winner
    # that already passes the analytic F0 >= 1.0 filter.
    if winner is not None:
        log["n_passes"] = 2
        log["winner_pass"] = 2
        log["winner_combo"] = winner
        log["winner_params"] = p2_params[winner]
        return np.maximum(p2_analytic[winner], noise_floor[i]), p2_params[winner], log

    # ── Pass 3: per-combo b_bright lower-bound + dual x0 ──────────────────
    # Only reached when no combo survives all eligibility filters after Pass 2.
    p3_analytic, p3_params = {}, {}
    log["pass3"] = {}
    for combo in tukey_param_combos:
        M = AsymmetricTukeyBiweight(c_pos=combo[0], c_neg=combo[1])
        M_np = M.with_xp(np)
        sr = sigma_relax[combo]
        px2 = p2_params[combo]

        b_bright_mag = max(0.0, float(px2[0] + px2[1] + px2[3]))

        bounds_p3 = list(bounds)
        up_bb = bounds_p3[5][1] if bounds_p3[5][1] is not None else 0.0
        bounds_p3[5] = (min(-b_bright_mag, up_bb), up_bb)

        x0_B = px2.astype(np.float64).copy()
        x0_B[5] = max(float(x0_B[5]), -b_bright_mag)

        f0_A, px_A, nit_A, conv_A = _fit_one(F_row, t_rel, x0,   bounds_p3, M, sr)
        f0_B, px_B, nit_B, conv_B = _fit_one(F_row, t_rel, x0_B, bounds_p3, M, sr)

        loss_A = float(np.sum(M_np.rho((F_row - f0_A.astype(np.float64)) / sr)))
        loss_B = float(np.sum(M_np.rho((F_row - f0_B.astype(np.float64)) / sr)))
        used_x0_B = loss_B < loss_A
        if used_x0_B:
            p3_analytic[combo], p3_params[combo] = f0_B, px_B
        else:
            p3_analytic[combo], p3_params[combo] = f0_A, px_A
        log["pass3"][combo] = {
            "irls_nit": nit_B if used_x0_B else nit_A,
            "converged": conv_B if used_x0_B else conv_A,
            "used_x0_B": bool(used_x0_B),
        }

    winner, p3_stats = _select_winner(
        F_row, p3_analytic, p3_params, tgt, t_rel, dt, tukey_param_combos
    )
    for combo in tukey_param_combos:
        log["pass3"][combo].update(p3_stats[combo])
        log["pass3"][combo]["params"] = p3_params[combo]
    if winner is None:
        # All combos still fail eligibility after the b_bright constraint — typically
        # because F0 dips analytically below 1.0 at the start of the fit window even
        # with b_bright clipped (short t_bright + small b_inf).  Degenerate trace.
        # NOTE: the absolute-fallback combo's pass3 stats entry will have a non-None
        # filter_failure tag — that asymmetry is real (no eligible combo was found)
        # and is preserved honestly in the log rather than masked.
        winner = tukey_param_combos[0]  # absolute fallback: first combo

    log["n_passes"] = 3
    log["winner_pass"] = 3
    log["winner_combo"] = winner
    log["winner_params"] = p3_params[winner]
    return np.maximum(p3_analytic[winner], noise_floor[i]), p3_params[winner], log


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def log_to_jsonable(log: dict) -> dict:
    """Normalize a per-ROI triexp log dict into a JSON-serializable shape.

    The per-ROI logs emitted by ``dff`` (one entry per ROI in the returned
    ``logs`` list) contain tuple keys (e.g. ``(c_pos, c_neg)`` combo IDs),
    NumPy arrays, and nested dicts. JSON does not accept tuple keys or
    ndarray values, so callers that want to serialize a log to JSON must
    normalize it first. This helper handles the conversion:

    - Tuple keys become underscore-joined strings (``(2, 3) -> "2_3"``).
    - ndarray values become Python lists (via ``.tolist()``).
    - Tuple values become Python lists.
    - Dict values are recursed into.
    - All other values pass through unchanged.

    Parameters
    ----------
    log : dict
        A single per-ROI log dict, as produced by ``dff`` (one element of
        the returned ``logs`` list).

    Returns
    -------
    dict
        JSON-serializable shallow copy of ``log``. The original is not
        mutated.

    Notes
    -----
    This helper intentionally returns the log body only — it does not add a
    ``"roi"`` index or any other caller-specific identity. Callers that want
    a per-row identity should wrap the return value (e.g.
    ``{"roi": i, **log_to_jsonable(log)}``).
    """
    def _key(k):
        """Flatten 2-tuple keys to underscore-joined strings."""
        return f"{k[0]}_{k[1]}" if isinstance(k, tuple) else k

    def _val(v):
        """Recursively normalize a log value into a JSON-serializable shape."""
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, dict):
            return {_key(k): _val(vv) for k, vv in v.items()}
        if isinstance(v, (tuple, list)):
            return [_val(x) for x in v]
        return v

    return {_key(k): _val(v) for k, v in log.items()}


def dff(
    F: np.ndarray,
    config: "DffConfig | dict",
    n_jobs: int = -1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """Fit the triexp baseline and return dFF, F0, noise_sd, params, and logs.

    Parameters
    ----------
    F : (N, T) or (T,) array
        Neuropil-corrected fluorescence (same array passed to set_dff_config).
    config : DffConfig or dict
        Output of set_dff_config().  A dict is also accepted, in which case
        it must contain keys matching the DffConfig field names exactly
        (it is splatted via ``DffConfig(**config)``).  All model and fitting
        parameters are read from config, including tukey_param_combos.
    n_jobs : int
        Parallel workers (joblib). -1 = all CPUs, 1 = sequential.

    Returns
    -------
    dFF : (N, T) float32
    F0 : (N, T) float32        — analytic F0 clamped to noise floor (max(model_output, noise_sd))
    noise_sd : (N,) float32    — per-ROI MAD noise estimate (from set_dff_config),
                                  used as the noise floor for F0 clamping
    params : (N, 7) float64    — winner model parameters per ROI, interleaved
                                  [b_inf, b_slow, t_slow, b_fast, t_fast, b_bright, t_bright]
                                  (b_bright ≤ 0)
    logs : list[dict]          — per-ROI pass diagnostics (length N);
                                  for 1D input F, a single dict is returned instead
    """
    if isinstance(config, dict):
        config = DffConfig(**config)

    is_1d = F.ndim == 1
    F2d = np.atleast_2d(np.asarray(F, dtype=np.float64))
    N, T = F2d.shape

    n_skip = config.n_skip
    t_rel = config.t_rel
    x0_all = config.x0_all
    bounds_all = config.bounds_all
    sigma_all = config.sigma_all
    min_frac_below_f0 = config.min_frac_below_f0
    tukey_param_combos = config.tukey_param_combos

    F_fit = F2d[:, n_skip:]

    noise_floor = sigma_all.astype(np.float32)

    args = (F_fit, t_rel, x0_all, bounds_all,
            sigma_all, tukey_param_combos, noise_floor, min_frac_below_f0)

    if n_jobs == 1:
        rows = [_process_roi(i, *args) for i in range(N)]
    else:
        rows = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_process_roi)(i, *args) for i in range(N)
        )

    winner_F0_fit = np.stack([r[0] for r in rows])   # (N, T_fit)
    params_all = np.stack([r[1] for r in rows])   # (N, 7)
    logs = [r[2] for r in rows]

    F0_full = np.empty((N, T), dtype=np.float32)
    if n_skip > 0:
        F0_full[:, :n_skip] = winner_F0_fit[:, :1]  # hold first fitted value
    F0_full[:, n_skip:] = winner_F0_fit

    with np.errstate(divide="ignore", invalid="ignore"):
        safe_F0 = np.where(F0_full > 0, F0_full, np.float32(1.0))
        dff_out = (F2d.astype(np.float32) - F0_full) / safe_F0
        dff_out = np.where(F0_full > 0, dff_out, np.float32(0.0))

    if is_1d:
        return dff_out[0], F0_full[0], noise_floor, params_all[0], logs[0]

    return dff_out, F0_full, noise_floor, params_all, logs
