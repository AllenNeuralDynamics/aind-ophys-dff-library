"""dF/F orchestration for one plane; :func:`run` is the library entry point.

Reads the metadata and the upstream extraction traces, computes dF/F with
either the triexponential baseline (:mod:`aind_ophys_dff_library.triexp_dff`)
or the rolling-percentile baseline (``aind_ophys_utils.dff``), and writes the
traces HDF5, per-ROI fit logs and QC figures, ``processing.json`` and
``quality_control.json``.
"""

import inspect
import json
import logging
import os
from multiprocessing import Pool
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np
from aind_ophys_utils import dff as dff_percentile
from aind_pophys_metadata import find_one, relative_to_root
from aind_pophys_metadata.processing import collect_upstream_process_names
from aind_pophys_metadata.runner import StageContext, stage_guard
from log_schema.main import setup_logging
from scipy.stats import skew

from . import qc
from .dff_config import set_dff_config
from .settings import DFFSettings
from .triexp_dff import dff as triexp_dff
from .triexp_dff import log_to_jsonable
from .utils import metadata_utils as mu

logger = logging.getLogger("Ophys dF/F")

PROCESS_NAME = "aind-ophys-dff"
EXTRACTION_GLOB = "*/extraction"
EXTRACTION_H5_GLOB = "*extraction.h5"
DFF_DIR = "dff"
PLOTS_DIR = "plots"


def parse_triexp_overrides(raw: str) -> dict:
    """Parse and validate the --triexp-config-overrides JSON string.

    Returns a dict of validated overrides to splat into ``set_dff_config``.
    Fails fast on invalid JSON, non-object root, or unknown keys.
    """
    parsed = json.loads(raw or "{}")
    # Exclude positional args + fields exposed as dedicated DFFSettings knobs
    # (dedicated fields are authoritative; JSON blob must not shadow them).
    allowed = set(inspect.signature(set_dff_config).parameters) - {
        "F",
        "fs",
        "ts",
        "sigma_anneal_steps",
    }
    unknown = set(parsed) - allowed
    if unknown:
        raise ValueError(
            f"triexp_config_overrides contains unknown keys: "
            f"{sorted(unknown)}. Allowed: {sorted(allowed)}"
        )
    return parsed


def compute_dff(
    traces: np.ndarray,
    settings: "DFFSettings",
    frame_rate: float,
    ts: Optional[np.ndarray],
    n_jobs: int,
) -> tuple:
    """Dispatch to the selected dF/F algorithm; return a uniform 5-tuple.

    Parameters
    ----------
    traces : (N, T) ndarray
        Neuropil-corrected fluorescence traces.
    settings : DFFSettings
        Parsed capsule settings; ``settings.method`` selects the algorithm.
    frame_rate : float
        Sampling frequency (Hz).
    ts : (T,) ndarray or None
        Per-frame timestamps (s). Used only by triexp; ignored by percentile.
    n_jobs : int
        Worker count. ``-1`` uses all CPUs (joblib semantics). The percentile
        path translates ``<= 0`` to ``None`` for ``multiprocessing.Pool``.

    Returns
    -------
    dff_traces : (N, T) ndarray
    baseline : (N, T) ndarray
    noise : (N,) ndarray or scalar
    logs : list[dict] (triexp) or None (percentile)
    config_snapshot : dict (triexp) or None (percentile)
    """
    if settings.method == "triexp":
        overrides = parse_triexp_overrides(settings.triexp_config_overrides)
        config = set_dff_config(
            traces,
            fs=frame_rate,
            ts=ts,
            sigma_anneal_steps=settings.sigma_anneal_steps,
            **overrides,
        )
        dff_traces, baseline, noise, _params, logs = triexp_dff(
            traces, config, n_jobs=n_jobs
        )
        return dff_traces, baseline, noise, logs, config.params

    percentile_n_jobs = n_jobs if n_jobs and n_jobs > 0 else None
    dff_traces, baseline, noise = dff_percentile.dff(
        traces,
        long_window=settings.long_window,
        short_window=settings.short_window,
        fs=frame_rate,
        inactive_percentile=settings.inactive_percentile,
        noise_method=settings.noise_method,
        n_jobs=percentile_n_jobs,
    )
    return dff_traces, baseline, noise, None, None


def _locate_extraction(input_dir: Path) -> Tuple[Path, str]:
    """Locate the upstream extraction output and its plane id.

    Parameters
    ----------
    input_dir : Path
        The input directory.

    Returns
    -------
    tuple
        ``(extraction_h5, unique_id)``.

    Raises
    ------
    FileNotFoundError
        If no ``*/extraction`` directory or no ``*extraction.h5`` is present.
        The message names both the pattern and the directory searched.
    """
    extraction_dir = find_one(input_dir, EXTRACTION_GLOB)
    extraction_h5 = find_one(
        extraction_dir, EXTRACTION_H5_GLOB, recursive=False
    )
    return extraction_h5, extraction_dir.parent.name


def _make_output_dir(results_root: Path, unique_id: str) -> Path:
    """Create and return ``<results_root>/<unique_id>/dff``.

    Parameters
    ----------
    results_root : Path
        The results root.
    unique_id : str
        Plane/FOV identifier.

    Returns
    -------
    Path
        The created output directory.
    """
    output_dir = Path(results_root) / unique_id / DFF_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _n_jobs() -> int:
    """Return the worker count from ``CO_CPUS``, or -1 for all CPUs."""
    return int(os.environ.get("CO_CPUS") or -1)


def _write_traces_h5(
    output_dir: Path,
    unique_id: str,
    dff_traces: np.ndarray,
    baseline: np.ndarray,
    noise: np.ndarray,
    skewness: np.ndarray,
) -> Path:
    """Write the dF/F traces, baseline, noise and skewness to HDF5.

    Parameters
    ----------
    output_dir : Path
        Directory the file is written to.
    unique_id : str
        Plane/FOV identifier used in the filename.
    dff_traces : np.ndarray
        The dF/F traces.
    baseline : np.ndarray
        The fitted baselines.
    noise : np.ndarray
        Per-ROI noise estimates.
    skewness : np.ndarray
        Per-ROI dF/F skewness.

    Returns
    -------
    Path
        Path to the written HDF5 file.
    """
    path = output_dir / f"{unique_id}_dff.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=dff_traces)
        f.create_dataset("baseline", data=baseline)
        f.create_dataset("noise", data=noise)
        f.create_dataset("skewness", data=skewness)
    return path


def _write_fit_logs(
    output_dir: Path, unique_id: str, logs: Optional[list]
) -> Path:
    """Write the per-ROI triexp fit logs as JSON.

    Parameters
    ----------
    output_dir : Path
        Directory the file is written to.
    unique_id : str
        Plane/FOV identifier used in the filename.
    logs : list or None
        Per-ROI fit logs; ``None`` on the percentile path, which writes ``[]``.

    Returns
    -------
    Path
        Path to the written JSON file.
    """
    payload = (
        [{"roi": i, **log_to_jsonable(log)} for i, log in enumerate(logs)]
        if logs is not None
        else []
    )
    path = output_dir / f"{unique_id}_dff_logs.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def _write_roi_plots(
    settings: DFFSettings,
    output_dir: Path,
    unique_id: str,
    traces: np.ndarray,
    baseline: np.ndarray,
    dff_traces: np.ndarray,
    frame_rate: float,
    logs: Optional[list],
) -> List[str]:
    """Render one diagnostic figure per ROI and return their references.

    Parameters
    ----------
    settings : DFFSettings
        Parsed settings, providing ``zoom_duration`` and the results root.
    output_dir : Path
        This plane's output directory.
    unique_id : str
        Plane/FOV identifier.
    traces : np.ndarray
        Neuropil-corrected traces.
    baseline : np.ndarray
        Fitted baselines.
    dff_traces : np.ndarray
        Computed dF/F traces.
    frame_rate : float
        Frame rate [Hz].
    logs : list or None
        Per-ROI triexp fit logs, annotated onto each figure's title.

    Returns
    -------
    list of str
        Per-ROI figure paths relative to the results root, in ROI order.
    """
    n_rois = traces.shape[0]
    if not n_rois:
        return []

    results_root = Path(settings.output_dir).resolve()
    fig_path = output_dir / PLOTS_DIR
    fig_path.mkdir(parents=True, exist_ok=True)
    # Zero-padded so the figures (and the curation list) sort numerically.
    roi_ids = [f"{n:0{len(str(n_rois))}d}" for n in range(n_rois)]
    plot_logs = logs if logs is not None else [None] * n_rois

    with Pool(_n_jobs() if _n_jobs() > 0 else None) as pool:
        pool.starmap(
            qc.plot_dff,
            zip(
                traces,
                baseline,
                dff_traces,
                [frame_rate] * n_rois,
                roi_ids,
                [fig_path] * n_rois,
                [unique_id] * n_rois,
                plot_logs,
                [settings.zoom_duration] * n_rois,
            ),
        )
    return [
        relative_to_root(
            results_root, fig_path / f"{unique_id}_{roi_id}_dff.png"
        )
        for roi_id in roi_ids
    ]


def _median(values: np.ndarray) -> Optional[float]:
    """Return the median of ``values``, or ``None`` when it is empty.

    Parameters
    ----------
    values : np.ndarray
        Per-ROI values.

    Returns
    -------
    float or None
        The median, or ``None`` for an empty array.
    """
    return float(np.median(values)) if np.size(values) else None


def _run_dff(
    settings: DFFSettings,
    metadata: mu.DFFMetadata,
    extraction_h5: Path,
    unique_id: str,
    output_dir: Path,
    ctx: StageContext,
) -> None:
    """Compute dF/F for the staged plane and write every output but processing.

    ``ctx`` is populated as the run progresses so the ``processing.json`` the
    stage guard writes reports partial provenance even when this raises.

    Parameters
    ----------
    settings : DFFSettings
        Parsed settings.
    metadata : mu.DFFMetadata
        Metadata read at the start of the run.
    extraction_h5 : Path
        The upstream extraction output this plane's traces are read from.
    unique_id : str
        Plane/FOV identifier.
    output_dir : Path
        This plane's dF/F output directory.
    ctx : StageContext
        Stage context the guard reads when it writes ``processing.json``.
    """
    with h5py.File(extraction_h5, "r") as f:
        traces = f["traces/corrected"][()]
    frame_rate = metadata.frame_rate

    if len(traces):
        dff_traces, baseline, noise, logs, config_snapshot = compute_dff(
            traces, settings, frame_rate, None, _n_jobs()
        )
    else:
        logger.warning(
            "No ROIs in %s; writing empty dF/F outputs.", extraction_h5.name
        )
        dff_traces, baseline, noise = traces, traces, np.asarray([])
        logs, config_snapshot = None, None
    skewness = skew(dff_traces, axis=1)

    # Only resolvable once the traces are loaded, so it cannot go into the
    # ``Code`` the guard took at entry; the guard merges these over the
    # up-front parameters when it writes the document.
    ctx.parameters = {
        "resolved_frame_rate": frame_rate,
        "triexp_config": config_snapshot,
    }
    # Scalar summary statistics describing this step's OUTPUTS, chosen so a
    # reviewer can triage the plane without opening the traces. Input
    # parameters belong in ``code.parameters``, never here.
    ctx.output_parameters = {
        "num_rois": int(traces.shape[0]),
        "num_frames": int(traces.shape[1]) if traces.ndim > 1 else 0,
        "median_noise": _median(noise),
        "median_dff_skewness": _median(skewness),
    }

    _write_traces_h5(
        output_dir, unique_id, dff_traces, baseline, noise, skewness
    )
    _write_fit_logs(output_dir, unique_id, logs)
    plot_references = _write_roi_plots(
        settings,
        output_dir,
        unique_id,
        traces,
        baseline,
        dff_traces,
        frame_rate,
        logs,
    )
    metric = qc.build_dff_curation_metric(unique_id, plot_references)
    qc.write_quality_control_json(
        qc.build_quality_control([metric]), output_dir
    )


def run(settings: Optional[DFFSettings] = None) -> None:
    """Run dF/F estimation end to end.

    Parameters
    ----------
    settings : DFFSettings, optional
        Parsed settings; constructed from CLI/env when omitted.
    """
    settings = settings or DFFSettings()
    input_dir = Path(settings.input_dir).resolve()
    results_root = Path(settings.output_dir).resolve()

    metadata = mu.load_dff_metadata(
        input_dir, cli_frame_rate=settings.frame_rate
    )
    setup_logging(
        use_cloudwatch=True,
        model={
            "acquisition_name": metadata.dataset_name or metadata.subject_id,
            "process_name": PROCESS_NAME,
        },
    )

    extraction_h5, unique_id = _locate_extraction(input_dir)
    logger.info("Calculating dF/F for ExperimentID %s", unique_id)

    output_dir = _make_output_dir(results_root, unique_id)
    code = mu.build_dff_code(
        parameters=settings.model_dump(mode="json"),
        input_data=[relative_to_root(input_dir, extraction_h5)],
    )
    with stage_guard(
        output_dir,
        mu.PROCESS_TYPE,
        code,
        plane_id=unique_id,
        logger=logger,
        output_path=relative_to_root(results_root, output_dir),
    ) as ctx:
        ctx.upstream_names = collect_upstream_process_names(
            input_dir, exclude=mu.process_name(unique_id)
        )
        _run_dff(
            settings,
            metadata,
            extraction_h5,
            unique_id,
            output_dir,
            ctx,
        )
