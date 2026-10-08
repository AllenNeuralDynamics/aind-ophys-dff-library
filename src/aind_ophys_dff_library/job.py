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

from .dff_config import set_dff_config
from .settings import DFFSettings
from .triexp_dff import dff as triexp_dff
from .triexp_dff import log_to_jsonable
from .utils import metadata_utils as mu
from datetime import datetime as dt
from typing import Union
from aind_data_schema.core.quality_control import QCEvaluation, QCMetric, QCStatus, Stage, Status
from aind_data_schema_models.modalities import Modality
from aind_qcportal_schema.metric_value import CurationMetric
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from numpy.typing import NDArray
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

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
            plot_dff,
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
    _write_roi_plots(
        settings,
        output_dir,
        unique_id,
        traces,
        baseline,
        dff_traces,
        frame_rate,
        logs,
    )
    write_qc_evalutation(output_dir, unique_id, traces.shape[0])


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


def plot_dff(
    trace: NDArray,
    baseline: NDArray,
    dff_trace: NDArray,
    frame_rate: float,
    roi_id: int,
    fig_path: Union[str, Path],
    unique_id: str,
    log: Optional[dict] = None,
    zoom_duration: float = 60.0,
) -> None:
    """Plot raw and dF/F traces with optional zoomed insets.

    Creates a multi-panel plot showing raw fluorescence signal and baseline-corrected
    dF/F trace. When zoom_duration is specified, includes three zoomed inset plots
    showing detailed views of the beginning, middle, and end of the recording session.

    Parameters
    ----------
    trace : NDArray
        Neuropil-corrected raw fluorescence trace
    baseline : NDArray
        Fitted baseline (F0) values
    dff_trace : NDArray
        Baseline-corrected dF/F trace (fractional units)
    frame_rate : float
        Frame rate in Hz
    roi_id : int
        Region of interest identifier for labeling the plot
    fig_path : str or Path
        Directory path where the output PNG file will be saved
    unique_id: str
        unique identifier
    zoom_duration : float, optional
        Duration in seconds for each zoomed inset window. If None or 0,
        creates a simpler 2-row layout without insets. If positive, creates
        a 3-row layout with inset zoom plots. Default is 60.0.

    Returns
    -------
    None
        Saves the plot as a PNG file to the specified path and closes the figure.

    Notes
    -----
    The function creates different layouts based on zoom_duration:

    - If zoom_duration is None or 0: 2 rows per channel (raw + dF/F)
    - If zoom_duration > 0: 3 rows per channel (raw + insets + dF/F)

    Inset windows show:
    - First: Beginning of recording (0 to zoom_duration seconds)
    - Middle: Center portion of recording
    - Last: End of recording (last zoom_duration seconds)

    The saved filename follows the pattern: '{unique_id}_{roi_id}_dff.png'
    """
    t = np.arange(len(trace)) / frame_rate
    show_insets = zoom_duration is not None and zoom_duration > 0

    # Create figure and axes
    if show_insets:
        fig, ax = plt.subplots(3, 1, figsize=(12, 3.5))
    else:
        fig, ax = plt.subplots(2, 1, figsize=(12, 2.3), sharex=True)

    # Calculate row indices based on layout
    raw_idx = 0
    dff_idx = 2 if show_insets else 1
    inset_idx = 1 if show_insets else None

    # Plot raw signal
    ax[raw_idx].plot(t, trace, label="neuropil-corrected trace", c="C0", lw=0.5)
    ax[raw_idx].plot(t, baseline, label=r"fitted F$_0$", c="#F0E442")
    ax[raw_idx].legend(
        loc=(0.692, 0.78), ncol=2, borderpad=0.05
    ).get_frame().set_linewidth(0.0)
    ax[raw_idx].set_ylabel("F [a.u.]")

    # Plot dF/F signal
    ax[dff_idx].plot(t, dff_trace * 100, label=r"$\Delta$F/F", c="C2", zorder=-1, lw=0.5)
    ax[dff_idx].axhline(0, c="k", ls="--")
    ax[dff_idx].legend(
        loc=(0.923, 0.78), ncol=1, borderpad=0.05
    ).get_frame().set_linewidth(0.0)
    ax[dff_idx].set_ylabel(r"$\Delta$F/F [%]", y=1 if show_insets else 0.5)

    # Add insets if requested
    if show_insets:
        t_total = t[-1] - t[0]
        zoom_windows = [
            (t[0], t[0] + zoom_duration),
            (
                t[0] + (t_total - zoom_duration) / 2,
                t[0] + (t_total + zoom_duration) / 2,
            ),
            (max(t[-1] - zoom_duration, t[0]), t[-1]),
        ]

        ax[inset_idx].axis("off")

        for j, (start_time, end_time) in enumerate(zoom_windows):
            inset_ax = inset_axes(
                ax[inset_idx],
                width="100%",
                height="100%",
                loc="center",
                bbox_to_anchor=([0.01, 0.34, 0.67][j], 0.0, 0.32, 0.8),
                bbox_transform=ax[inset_idx].transAxes,
            )

            mask = (t >= start_time) & (t <= end_time)
            if np.any(mask):
                t_zoom, dff_zoom = t[mask], dff_trace[mask] * 100

                inset_ax.plot(t_zoom, dff_zoom, c="C2", lw=0.5)
                inset_ax.axhline(0, c="k", ls="--")
                inset_ax.grid(True, alpha=0.8)
                inset_ax.set_xlim(start_time, end_time)

                if len(dff_zoom) > 0:
                    mi, ma = np.nanmin(dff_zoom), np.nanmax(dff_zoom)
                    y_margin = 0.1 * (ma - mi)
                    if ~np.isnan(y_margin):
                        inset_ax.set_ylim(mi - y_margin, ma + y_margin)

                inset_ax.set_title(
                    f"{['First', 'Middle', 'Last'][j]} {zoom_duration:.0f}s",
                    fontsize=10,
                    y=0.94,
                )
                inset_ax.set_xticks([])
                inset_ax.set_yticks([])

                mark_inset(
                    ax[dff_idx],
                    inset_ax,
                    loc1=1,
                    loc2=3,
                    fc="none",
                    ec="#333333",
                    alpha=0.8,
                    linestyle="--",
                    linewidth=1,
                )

    # Set x-limits and labels
    tmin, tmax = np.nanmin(t), np.nanmax(t)
    margin = (tmax - tmin) / 100
    ax[raw_idx].set_xlim(tmin - margin, tmax + margin)
    ax[dff_idx].set_xlim(tmin - margin, tmax + margin)
    plt.setp(ax[raw_idx].get_xticklabels(), visible=False)
    ax[dff_idx].set_xlabel("Time [s]")
    title = f"cell_roi_id: {int(roi_id)}"
    if log is not None:
        title += (
            f"  (n_passes={log.get('n_passes')}, "
            f"winner={log.get('winner_combo')}, "
            f"trigger={log.get('pass1_trigger')})"
        )
    ax[0].set_title(title)
    plt.subplots_adjust(hspace=0.1, top=0.935, bottom=0.13, left=0.06, right=0.995)
    fig.savefig(
        fig_path / f"{unique_id}_{roi_id}_dff.png",
        dpi=200,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)


def write_qc_evalutation(output_dir: Path, unique_id: str, N: int) -> None:
    """Writes QC metrics to json file.

    Parameters
    ----------
    output_dir: Path
        output directory
    unique_id: str
        unique identifier
    N: int
        number of ROIs detected
    """
    cell_plots = dict()
    for roi_id in range(N):
        cell_plots[str(roi_id)] = {
            "reference": f"{unique_id}/dff/plots/{unique_id}_{roi_id:0{len(str(N))}d}_dff.png"
        }
    curation = CurationMetric(curations=[json.dumps(cell_plots)])
    metric = QCMetric(
        name=f"{unique_id} dF/F",
        description="dF/F baseline correction",
        reference="",
        status_history=[
            QCStatus(evaluator="Automated", timestamp=dt.now(), status=Status.PASS)
        ],
        value=curation,
    )

    evaluation = QCEvaluation(
        modality=Modality.from_abbreviation("pophys"),
        stage=Stage.PROCESSING,
        name="dF/F",
        description="dF/F baseline correction for each ROI across all FOVs",
        allow_failed_metrics=False,
        metrics=[metric],
        tags=["dff"],
    )

    with open(output_dir / f"{unique_id}_dff_evaluation.json", "w") as f:
        json.dump(json.loads(evaluation.model_dump_json()), f, indent=4)
