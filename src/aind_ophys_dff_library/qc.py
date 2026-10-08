"""Per-ROI QC figures and the QC evaluation for dF/F estimation."""

import json
from datetime import datetime as dt
from pathlib import Path
from typing import Optional, Union
import numpy as np
from aind_data_schema.core.quality_control import QCEvaluation, QCMetric, QCStatus, Stage, Status
from aind_data_schema_models.modalities import Modality
from aind_qcportal_schema.metric_value import CurationMetric
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from numpy.typing import NDArray
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


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
