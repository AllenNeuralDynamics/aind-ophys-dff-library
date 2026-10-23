"""quality_control.json generation for dF/F estimation.

Owns both halves of this processing step's QC: the per-ROI diagnostic
figures and the ``CurationMetric`` that points a reviewer at them. ``job.run``
accumulates the metrics built here and emits ONE full ``quality_control.json``
(never per-metric files).

``CurationMetric`` here is the core v2 class from
``aind_data_schema.core.quality_control`` (a ``QCMetric`` subclass adding
``value``/``type``/``curation_history``), and it goes straight into
``QualityControl.metrics``. Do not reach for
``aind_qcportal_schema.metric_value.CurationMetric`` — it no longer exists,
and without ``QCEvaluation`` there is no wrapper to put a metric in.
"""

import logging
from datetime import datetime as dt
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib as mpl
import numpy as np
from aind_data_schema.core.quality_control import (
    CurationMetric,
    QCMetric,
    QCStatus,
    QualityControl,
    Stage,
    Status,
)
from aind_data_schema_models.modalities import Modality
from aind_pophys_metadata import quality_control as shared_qc
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from numpy.typing import NDArray

mpl.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402

logger = logging.getLogger(__name__)

CURATION_TYPE = "ROI dF/F trace review"
EVALUATION = "dF/F"
METRIC_TYPE = "Informational"


def build_dff_curation_metric(
    unique_id: str,
    plot_references: Dict[str, str],
) -> CurationMetric:
    """Construct the per-ROI dF/F curation metric for one plane.

    Parameters
    ----------
    unique_id : str
        Plane/FOV identifier used in the metric name.
    plot_references : dict of str to str
        Per-ROI diagnostic figure path, relative to the asset root, keyed by
        ROI id in ROI order. Empty when no ROIs were detected.

    Returns
    -------
    CurationMetric
        The dF/F curation metric, with an automated PASS status. Its value is
        a one-element list holding ``{roi_id: {"reference": path}}``.
    """
    # The QC portal reads the last list element as the curation mapping.
    curation = {
        roi_id: {"reference": reference}
        for roi_id, reference in plot_references.items()
    }
    return CurationMetric(
        name=f"{unique_id} dF/F",
        modality=Modality.POPHYS,
        stage=Stage.PROCESSING,
        type=CURATION_TYPE,
        value=[curation],
        tags={"evaluation": EVALUATION, "type": METRIC_TYPE},
        description=(
            "dF/F for each ROI: neuropil-corrected trace, fitted baseline "
            "and the resulting dF/F."
        ),
        status_history=[
            QCStatus(
                evaluator="Automated",
                status=Status.PASS,
                timestamp=dt.now(shared_qc.SEATTLE_TZ).isoformat(),
            )
        ],
    )


def build_quality_control(metrics: List[QCMetric]) -> QualityControl:
    """Wrap metrics into a full QualityControl with the standard grouping."""
    return shared_qc.build_quality_control(metrics)


def write_quality_control_json(
    quality_control: QualityControl,
    output_dir: Path,
) -> Path:
    """Serialize a QualityControl object to ``quality_control.json``.

    Parameters
    ----------
    quality_control : QualityControl
        The QC document to write.
    output_dir : Path
        Directory into which ``quality_control.json`` is written.

    Returns
    -------
    Path
        The path written.
    """
    return shared_qc.write_quality_control_json(quality_control, output_dir)


def _zoom_windows(t: NDArray, zoom_duration: float) -> list:
    """Return (start, end) second-ranges for the first/middle/last insets.

    Parameters
    ----------
    t : NDArray
        Per-frame time axis [s].
    zoom_duration : float
        Width [s] of each inset window.

    Returns
    -------
    list
        Three ``(start, end)`` tuples.
    """
    total = t[-1] - t[0]
    return [
        (t[0], t[0] + zoom_duration),
        (
            t[0] + (total - zoom_duration) / 2,
            t[0] + (total + zoom_duration) / 2,
        ),
        (max(t[-1] - zoom_duration, t[0]), t[-1]),
    ]


def _draw_inset(
    host_ax: Axes,
    dff_ax: Axes,
    t: NDArray,
    dff_trace: NDArray,
    window: tuple,
    position: int,
    zoom_duration: float,
) -> None:
    """Draw one zoomed dF/F inset into ``host_ax``.

    Parameters
    ----------
    host_ax : matplotlib.axes.Axes
        Axes the inset is placed inside.
    dff_ax : matplotlib.axes.Axes
        Full-trace dF/F axes the inset is marked against.
    t : NDArray
        Per-frame time axis [s].
    dff_trace : NDArray
        dF/F trace (fractional units).
    window : tuple
        ``(start, end)`` seconds to show.
    position : int
        0, 1 or 2 for first/middle/last, selecting the horizontal slot.
    zoom_duration : float
        Inset width [s], used in the title.
    """
    start_time, end_time = window
    mask = (t >= start_time) & (t <= end_time)
    if not np.any(mask):
        return

    inset_ax = inset_axes(
        host_ax,
        width="100%",
        height="100%",
        loc="center",
        bbox_to_anchor=([0.01, 0.34, 0.67][position], 0.0, 0.32, 0.8),
        bbox_transform=host_ax.transAxes,
    )
    t_zoom, dff_zoom = t[mask], dff_trace[mask] * 100
    inset_ax.plot(t_zoom, dff_zoom, c="C2", lw=0.5)
    inset_ax.axhline(0, c="k", ls="--")
    inset_ax.grid(True, alpha=0.8)
    inset_ax.set_xlim(start_time, end_time)

    if len(dff_zoom) > 0:
        lo, hi = np.nanmin(dff_zoom), np.nanmax(dff_zoom)
        margin = 0.1 * (hi - lo)
        if ~np.isnan(margin):
            inset_ax.set_ylim(lo - margin, hi + margin)

    inset_ax.set_title(
        f"{['First', 'Middle', 'Last'][position]} {zoom_duration:.0f}s",
        fontsize=10,
        y=0.94,
    )
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    mark_inset(
        dff_ax,
        inset_ax,
        loc1=1,
        loc2=3,
        fc="none",
        ec="#333333",
        alpha=0.8,
        linestyle="--",
        linewidth=1,
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
    show_insets = bool(zoom_duration)

    if show_insets:
        fig, ax = plt.subplots(3, 1, figsize=(12, 3.5))
        raw_idx, inset_idx, dff_idx = 0, 1, 2
    else:
        fig, ax = plt.subplots(2, 1, figsize=(12, 2.3), sharex=True)
        raw_idx, inset_idx, dff_idx = 0, None, 1

    ax[raw_idx].plot(
        t, trace, label="neuropil-corrected trace", c="C0", lw=0.5
    )
    ax[raw_idx].plot(t, baseline, label=r"fitted F$_0$", c="#F0E442")
    ax[raw_idx].legend(
        loc=(0.692, 0.78), ncol=2, borderpad=0.05
    ).get_frame().set_linewidth(0.0)
    ax[raw_idx].set_ylabel("F [a.u.]")

    ax[dff_idx].plot(
        t, dff_trace * 100, label=r"$\Delta$F/F", c="C2", zorder=-1, lw=0.5
    )
    ax[dff_idx].axhline(0, c="k", ls="--")
    ax[dff_idx].legend(
        loc=(0.923, 0.78), ncol=1, borderpad=0.05
    ).get_frame().set_linewidth(0.0)
    ax[dff_idx].set_ylabel(r"$\Delta$F/F [%]", y=1 if show_insets else 0.5)

    if show_insets:
        ax[inset_idx].axis("off")
        for position, window in enumerate(_zoom_windows(t, zoom_duration)):
            _draw_inset(
                ax[inset_idx],
                ax[dff_idx],
                t,
                dff_trace,
                window,
                position,
                zoom_duration,
            )

    tmin, tmax = np.nanmin(t), np.nanmax(t)
    margin = (tmax - tmin) / 100
    ax[raw_idx].set_xlim(tmin - margin, tmax + margin)
    ax[dff_idx].set_xlim(tmin - margin, tmax + margin)
    plt.setp(ax[raw_idx].get_xticklabels(), visible=False)
    ax[dff_idx].set_xlabel("Time [s]")

    title = f"cell_roi_id: {roi_id}"
    if log is not None:
        title += (
            f"  (n_passes={log.get('n_passes')}, "
            f"winner={log.get('winner_combo')}, "
            f"trigger={log.get('pass1_trigger')})"
        )
    ax[0].set_title(title)
    plt.subplots_adjust(
        hspace=0.1, top=0.935, bottom=0.13, left=0.06, right=0.995
    )
    fig.savefig(
        Path(fig_path) / f"{unique_id}_{roi_id}_dff.png",
        dpi=200,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)
