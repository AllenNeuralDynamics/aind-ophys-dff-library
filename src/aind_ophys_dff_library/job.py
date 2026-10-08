import json
import logging
import os
from datetime import datetime as dt
from multiprocessing import Pool
from pathlib import Path
from typing import Literal, Optional, Union

import h5py
from aind_ophys_utils import dff as dff_percentile
from aind_ophys_dff_library.dff_config import set_dff_config
from aind_ophys_dff_library.triexp_dff import dff as triexp_dff, log_to_jsonable
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from aind_data_schema.core.processing import DataProcess, ProcessName
from aind_data_schema.core.quality_control import (
    QCEvaluation,
    QCMetric,
    QCStatus,
    Stage,
    Status,
)
from aind_data_schema_models.modalities import Modality
from aind_log_utils.log import setup_logging
from aind_qcportal_schema.metric_value import CurationMetric
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from numpy.typing import NDArray
from pydantic import Field
from pydantic_settings import BaseSettings
from scipy.stats import skew


class DFFSettings(BaseSettings, cli_parse_args=True):
    """Settings for dF/F calculation."""

    input_dir: Path = Field(
        default="/data",
        description="Input directory containing extraction.h5 and metadata files",
    )
    output_dir: Path = Field(
        default="/results",
        description="Output directory where results are saved",
    )
    method: Literal["triexp", "percentile"] = Field(
        default="triexp",
        description=(
            "dF/F algorithm: 'triexp' (parametric 3-pass triexponential baseline fit) or "
            "'percentile' (rolling-percentile baseline)."
        ),
    )

    # Percentile parameters — read only when method == 'percentile'.
    long_window: float = Field(
        default=60.0,
        description="Percentile baseline window (s). Used only when method='percentile'.",
    )
    short_window: float = Field(
        default=3.333,
        description="Short detrending window (s). Used only when method='percentile'.",
    )
    inactive_percentile: int = Field(
        default=10,
        description="Inactive percentile for F0. Used only when method='percentile'.",
    )
    noise_method: str = Field(
        default="mad",
        description="Noise estimator ('mad'|'fft'|'welch'). Used only when method='percentile'.",
    )

    # Triexp parameters — read only when method == 'triexp'.
    sigma_anneal_steps: int = Field(
        default=1,
        description=(
            "IRLS sigma-annealing steps (aind_ophys_utils.baseline_fitting."
            "nonlinear_fit). 1 (default) or 2 disable annealing "
            "(single-jump); 3 enables one intermediate step; 4 uses two; "
            "higher graduates further but exceeds the internal maxiter=5. "
            "Used only when method='triexp'."
        ),
    )
    triexp_config_overrides: str = Field(
        default="{}",
        description=(
            "JSON object of overrides for set_dff_config. Keys must be a subset of "
            "set_dff_config's keyword arguments (except sigma_anneal_steps, which "
            "has its own dedicated field above); any key omitted falls back to the "
            "function's default. Used only when method='triexp'."
        ),
    )

    class Config:
        env_prefix = "DFF_"


def _jsonify_log(roi_idx: int, log: dict) -> dict:
    """Wrap library.log_to_jsonable with this capsule's per-ROI identity prefix."""
    return {"roi": roi_idx, **log_to_jsonable(log)}


def _parse_triexp_overrides(raw: str) -> dict:
    """Parse and validate the --triexp-config-overrides JSON string.

    Returns a dict of validated overrides to splat into ``set_dff_config``.
    Fails fast on invalid JSON, non-object root, or unknown keys.
    """
    import inspect

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"triexp_config_overrides is not valid JSON: {e}") from e
    if not isinstance(parsed, dict):
        raise ValueError(
            f"triexp_config_overrides must be a JSON object, got {type(parsed).__name__}"
        )
    # Exclude positional args + fields exposed as dedicated DFFSettings knobs
    # (dedicated fields are authoritative; JSON blob must not shadow them).
    allowed = set(inspect.signature(set_dff_config).parameters) - {
        "F", "fs", "ts", "sigma_anneal_steps",
    }
    unknown = set(parsed) - allowed
    if unknown:
        raise ValueError(
            f"triexp_config_overrides contains unknown keys: {sorted(unknown)}. "
            f"Allowed: {sorted(allowed)}"
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
        overrides = _parse_triexp_overrides(settings.triexp_config_overrides)
        config = set_dff_config(
            traces, fs=frame_rate, ts=ts,
            sigma_anneal_steps=settings.sigma_anneal_steps,
            **overrides,
        )
        dff_traces, baseline, noise, _params, logs = triexp_dff(
            traces, config, n_jobs=n_jobs,
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


def write_data_process(
    metadata: dict,
    input_fp: Union[str, Path],
    output_fp: Union[str, Path],
    unique_id: str,
    start_time: dt,
    end_time: dt,
) -> None:
    """Writes output metadata to plane processing.json

    Parameters
    ----------
    metadata: dict
        parameters from suite2p motion correction
    input_fp: str
        path to raw movies
    output_fp: str
        path to motion corrected movies
    unique_id: str
        unique identifier
    start_time: dt
        start time of processing
    end_time: dt
        end time of processing
    """
    data_proc = DataProcess(
        name=ProcessName.DF_F_ESTIMATION,
        software_version=os.getenv("VERSION", ""),
        start_date_time=start_time.isoformat(),
        end_date_time=end_time.isoformat(),
        input_location=str(input_fp),
        output_location=str(output_fp),
        code_url=(os.getenv("REPO_URL", "")),
        parameters=metadata,
    )
    if isinstance(output_fp, str):
        output_dir = Path(output_fp).parent
    else:
        output_dir = output_fp.parent
    with open(output_dir / f"{unique_id}_df_f_data_process.json", "w") as f:
        json.dump(json.loads(data_proc.model_dump_json()), f, indent=4)


def get_metadata(input_dir: Path, meta_type: str) -> dict:
    """Extracts metadata from processing and subject json files

    Parameters
    ----------
    input_dir: Path
        input directory
    meta_type: str
        type of metadata to extract

    Returns
    -------
    metadata: dict
        metadata
    """
    input_fp = next(input_dir.rglob(f"{meta_type}"), "")
    if not input_fp:
        raise FileNotFoundError(f"No {meta_type} file found in {input_dir}")
    with open(input_fp, "r") as f:
        metadata = json.load(f)
    return metadata


def make_output_directory(output_dir: Path, unique_id: str) -> str:
    """Creates the output directory if it does not exist

    Parameters
    ----------
    output_dir: Path
        output directory
    unique_id: str
        unique identifier

    Returns
    -------
    output_dir: str
        output directory
    """
    output_dir = output_dir / unique_id
    output_dir.mkdir(exist_ok=True)
    output_dir = output_dir / "dff"
    output_dir.mkdir(exist_ok=True)

    return output_dir


def get_frame_rate(session: dict) -> float:
    """Attempt to pull frame rate from session.json
    Raises ValueError if frame rate not in session.json

    Parameters
    ----------
    session: dict
        session metadata

    Returns
    -------
    frame_rate: float
        frame rate in Hz
    """
    frame_rate_hz = None
    for i in session.get("data_streams", ""):
        if i.get("ophys_fovs", ""):
            frame_rate_hz = i["ophys_fovs"][0]["frame_rate"]
            break
    if frame_rate_hz is None:
        raise ValueError("No frame rate found in session.json")
    if isinstance(frame_rate_hz, str):
        frame_rate_hz = float(frame_rate_hz)
    return frame_rate_hz


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


if __name__ == "__main__":
    start_time = dt.now()
    args = DFFSettings()
    input_dir = args.input_dir
    output_dir = args.output_dir
    data_description_data = get_metadata(input_dir, "data_description.json")
    name = data_description_data.get("name", "")
    subject_data = get_metadata(input_dir, "subject.json")
    subject_id = subject_data.get("subject_id", "")
    session_data = get_metadata(input_dir, "session.json")
    frame_rate = get_frame_rate(session_data)
    setup_logging("aind-ophys-dff", subject_id=subject_id, asset_name=name)
    extraction_dir = next(input_dir.rglob("*/extraction"))
    unique_id = extraction_dir.parent.name
    logging.info(f"Calculating dF/F for ExperimentID {unique_id}")
    extraction_fp = next(extraction_dir.glob("*extraction.h5"))
    output_dir = make_output_directory(output_dir, unique_id)
    with h5py.File(extraction_fp, "r") as f:
        traces = f["traces/corrected"][()]
    if len(traces):
        n_jobs = int(os.environ.get("CO_CPUS") or -1)
        dff_traces, baseline, noise, logs, config_snapshot = compute_dff(
            traces, args, frame_rate, None, n_jobs,
        )
    else:  # no ROIs detected
        dff_traces, baseline, noise = traces, traces, np.asarray([])
        logs, config_snapshot = None, None

    skewness = skew(dff_traces, axis=1)
    with h5py.File(output_dir / f"{unique_id}_dff.h5", "w") as f:
        f.create_dataset("data", data=dff_traces)
        f.create_dataset("baseline", data=baseline)
        f.create_dataset("noise", data=noise)
        f.create_dataset("skewness", data=skewness)

    logs_json = (
        [_jsonify_log(i, lg) for i, lg in enumerate(logs)] if logs is not None else []
    )
    with open(output_dir / f"{unique_id}_dff_logs.json", "w") as f:
        json.dump(logs_json, f, indent=2)

    # Include settings + triexp config snapshot in metadata
    input_params = {**vars(args), "triexp_config": config_snapshot}
    write_data_process(
        input_params,
        extraction_fp,
        output_dir / "dff.h5",
        unique_id,
        start_time,
        dt.now(),
    )

    # QC plots
    N = traces.shape[0]
    if N:
        fig_path = output_dir / "plots"
        os.makedirs(fig_path, exist_ok=True)
        plot_logs = logs if logs is not None else [None] * N
        with Pool(int(tmp) if (tmp := os.environ.get("CO_CPUS")) else tmp) as pool:
            pool.starmap(
                plot_dff,
                zip(
                    traces,
                    baseline,
                    dff_traces,
                    [frame_rate] * N,
                    [f"{n:0{len(str(N))}d}" for n in range(N)],
                    [fig_path] * N,
                    [unique_id] * N,
                    plot_logs,
                ),
            )

    write_qc_evalutation(output_dir, unique_id, N)
