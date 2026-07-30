"""Top-level dF/F orchestration.

``run()`` is the library entry point called by the thin capsule wrapper: read
metadata (v1 or v2) -> configure logging -> locate the extraction output ->
compute dF/F (triexponential or rolling-percentile baseline) -> write the
traces HDF5, the per-ROI fit logs and the per-ROI QC figures -> emit a full
``processing.json`` and ``quality_control.json``.

The dF/F algorithms themselves stay where they already live: the
triexponential fit in :mod:`aind_ophys_dff_library.triexp_dff` /
:mod:`aind_ophys_dff_library.dff_config`, and the rolling-percentile baseline
in ``aind_ophys_utils.dff``. Only orchestration, IO and QC moved here.

``processing.json`` is written from a ``finally`` block so a failed run still
leaves a document describing what was attempted, with the error recorded in
``notes``.
"""

import inspect
import json
import logging
import os
from datetime import datetime as dt
from multiprocessing import Pool
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np
from aind_ophys_utils import dff as dff_percentile
from log_schema.main import setup_logging
from scipy.stats import skew

from . import qc
from .dff_config import set_dff_config
from .settings import DFFSettings
from .triexp_dff import dff as triexp_dff
from .triexp_dff import log_to_jsonable
from .utils import metadata_utils as mu
from .utils.paths import relative_to_root

logger = logging.getLogger("Ophys dF/F")

PROCESS_NAME = "aind-ophys-dff"
EXTRACTION_GLOB = "*/extraction"
EXTRACTION_H5_GLOB = "*extraction.h5"
DFF_DIR = "dff"
PLOTS_DIR = "plots"


def parse_triexp_overrides(raw: str) -> dict:
    """Parse and validate the ``--triexp_config_overrides`` JSON string.

    JSON validity is already enforced by
    :class:`aind_ophys_dff_library.settings.DFFSettings`; this adds the key
    check, which needs ``set_dff_config``'s signature.

    Parameters
    ----------
    raw : str
        JSON object string of overrides.

    Returns
    -------
    dict
        Validated overrides, ready to splat into ``set_dff_config``.

    Raises
    ------
    ValueError
        If a key is not a ``set_dff_config`` keyword argument.
    """
    parsed = json.loads(raw or "{}")
    # Exclude positional args and fields with dedicated DFFSettings knobs:
    # the dedicated field is authoritative, the JSON blob must not shadow it.
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
    settings: DFFSettings,
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
    tuple
        ``(dff_traces, baseline, noise, logs, config_snapshot)``. ``logs`` and
        ``config_snapshot`` are ``None`` on the percentile path.
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
        The capsule input directory.

    Returns
    -------
    tuple
        ``(extraction_h5, unique_id)``.

    Raises
    ------
    FileNotFoundError
        If no ``*/extraction`` directory or no ``*extraction.h5`` is present.
    """
    extraction_dir = next(Path(input_dir).rglob(EXTRACTION_GLOB), None)
    if extraction_dir is None:
        raise FileNotFoundError(
            f"No '{EXTRACTION_GLOB}' directory found under {input_dir}; "
            f"dF/F needs the extraction capsule's output."
        )
    extraction_h5 = next(extraction_dir.glob(EXTRACTION_H5_GLOB), None)
    if extraction_h5 is None:
        raise FileNotFoundError(
            f"No '{EXTRACTION_H5_GLOB}' found in {extraction_dir}"
        )
    return extraction_h5, extraction_dir.parent.name


def _make_output_dir(results_root: Path, unique_id: str) -> Path:
    """Create and return ``<results_root>/<unique_id>/dff``.

    Parameters
    ----------
    results_root : Path
        The capsule results root.
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


def _resolve_frame_rate(
    metadata: mu.DFFMetadata, settings: DFFSettings
) -> float:
    """Resolve the frame rate from metadata, falling back to the CLI value.

    Parameters
    ----------
    metadata : mu.DFFMetadata
        Metadata read at the start of the run.
    settings : DFFSettings
        Parsed settings, providing the CLI fallback.

    Returns
    -------
    float
        The resolved frame rate [Hz].

    Raises
    ------
    ValueError
        If neither the metadata nor the CLI supplies one.
    """
    frame_rate = metadata.frame_rate
    if frame_rate is None:
        frame_rate = settings.frame_rate
        logger.warning(
            "Using CLI fallback frame rate: %s", settings.frame_rate
        )
    if frame_rate is None:
        raise ValueError(
            "frame_rate could not be resolved from metadata or --frame_rate"
        )
    return float(frame_rate)


def _n_jobs() -> int:
    """Return the worker count from ``CO_CPUS``, or -1 for all CPUs."""
    return int(os.environ.get("CO_CPUS") or -1)


def _write_traces_h5(
    output_dir: Path,
    unique_id: str,
    dff_traces: np.ndarray,
    baseline: np.ndarray,
    noise: np.ndarray,
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
        f.create_dataset("skewness", data=skew(dff_traces, axis=1))
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


def _run_dff(
    settings: DFFSettings, metadata: mu.DFFMetadata, ctx: dict
) -> None:
    """Compute dF/F for the staged plane and write every output but processing.

    ``ctx`` is populated as the run progresses so the ``processing.json``
    writer in ``run()``'s ``finally`` can report partial provenance even when
    this raises.

    Parameters
    ----------
    settings : DFFSettings
        Parsed settings.
    metadata : mu.DFFMetadata
        Metadata read at the start of the run.
    ctx : dict
        Run context accumulated for the metadata writer.
    """
    input_dir = Path(settings.input_dir).resolve()
    results_root = Path(settings.output_dir).resolve()

    extraction_h5, unique_id = _locate_extraction(input_dir)
    ctx["unique_id"] = unique_id
    ctx["input_data"] = [relative_to_root(input_dir, extraction_h5)]
    logger.info("Calculating dF/F for ExperimentID %s", unique_id)

    output_dir = _make_output_dir(results_root, unique_id)
    ctx["output_dir"] = output_dir

    with h5py.File(extraction_h5, "r") as f:
        traces = f["traces/corrected"][()]
    frame_rate = _resolve_frame_rate(metadata, settings)

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

    ctx["parameters"] = {
        **settings.model_dump(mode="json"),
        "triexp_config": config_snapshot,
    }
    ctx["output_parameters"] = {
        "num_rois": int(traces.shape[0]),
        "frame_rate": frame_rate,
    }

    _write_traces_h5(output_dir, unique_id, dff_traces, baseline, noise)
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


def _write_processing_metadata(
    settings: DFFSettings,
    ctx: dict,
    start_time: dt,
    error: Optional[BaseException],
) -> None:
    """Best-effort writer for processing.json. Never raises.

    Failures here must not mask a dF/F failure already propagating out of
    ``run()``. Falls back to the results root when the run failed before the
    per-plane output directory was created.

    Parameters
    ----------
    settings : DFFSettings
        Parsed settings.
    ctx : dict
        Run context accumulated by :func:`_run_dff`.
    start_time : dt
        Run start time.
    error : Exception or None
        The exception that terminated the run, if any.
    """
    results_root = Path(settings.output_dir).resolve()
    output_dir = Path(ctx.get("output_dir") or results_root)
    try:
        data_process = mu.build_data_process(
            parameters=ctx.get("parameters")
            or settings.model_dump(mode="json"),
            start_time=start_time,
            end_time=dt.now(),
            input_data=ctx.get("input_data"),
            output_path=relative_to_root(results_root, output_dir),
            output_parameters=ctx.get("output_parameters"),
            error=error,
        )
        mu.write_processing_json(
            mu.build_processing([data_process]), output_dir
        )
    except Exception as proc_exc:
        logger.error(f"processing.json write failed (non-fatal): {proc_exc}")


def _verify_outputs(output_dir: Path) -> None:
    """Round-trip the emitted metadata through the core v2 schema objects.

    Parameters
    ----------
    output_dir : Path
        Directory holding the emitted ``processing.json`` and
        ``quality_control.json``.
    """
    from aind_data_schema.core.processing import Processing
    from aind_data_schema.core.quality_control import QualityControl

    Processing.model_validate_json(
        (Path(output_dir) / "processing.json").read_text()
    )
    QualityControl.model_validate_json(
        (Path(output_dir) / "quality_control.json").read_text()
    )


def run(settings: Optional[DFFSettings] = None) -> None:
    """Run dF/F estimation end to end.

    Parameters
    ----------
    settings : DFFSettings, optional
        Parsed settings; constructed from CLI/env when omitted. When
        ``settings.verify`` is True, the emitted metadata is round-tripped
        through the core v2 schema objects after writing.
    """
    settings = settings or DFFSettings()
    start_time = dt.now()

    metadata = mu.load_dff_metadata(settings.input_dir)
    setup_logging(
        use_cloudwatch=True,
        model={
            "acquisition_name": metadata.dataset_name or metadata.subject_id,
            "process_name": PROCESS_NAME,
        },
    )
    logger.info("Starting dF/F", extra={"event_type": "stage_start"})

    ctx: dict = {}
    run_error: Optional[BaseException] = None
    try:
        _run_dff(settings, metadata, ctx)
        logger.info("dF/F complete", extra={"event_type": "stage_complete"})
    except BaseException as exc:
        run_error = exc
        logger.error("dF/F failed", extra={"event_type": "stage_error"})
        logger.exception("Unhandled exception")
        raise
    finally:
        _write_processing_metadata(settings, ctx, start_time, run_error)

    if settings.verify:
        _verify_outputs(ctx["output_dir"])
