"""Tests for dF/F quality_control.json generation and the QC figures."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from aind_data_schema.core.quality_control import (
    CurationMetric,
    QualityControl,
    Stage,
    Status,
)
from aind_data_schema_models.modalities import Modality

from aind_ophys_dff_library import qc


class TestDFFCurationMetric(unittest.TestCase):
    """The per-plane dF/F curation metric.

    ``CurationMetric`` is the core aind-data-schema v2 class, and the metric
    sits directly in ``QualityControl.metrics`` — there is no
    ``QCEvaluation`` wrapper in v2.
    """

    def setUp(self) -> None:
        """Build the metric once for the assertions below."""
        self.references = [
            "VISp_0/dff/plots/VISp_0_00_dff.png",
            "VISp_0/dff/plots/VISp_0_01_dff.png",
        ]
        self.metric = qc.build_dff_curation_metric("VISp_0", self.references)

    def test_is_a_core_curation_metric(self) -> None:
        """The metric is the v2 core CurationMetric, not a QCMetric wrapper."""
        self.assertIsInstance(self.metric, CurationMetric)
        self.assertEqual(self.metric.object_type, "Curation metric")

    def test_required_v2_fields(self) -> None:
        """name / modality / stage / type / value are all populated."""
        self.assertEqual(self.metric.name, "VISp_0 dF/F")
        self.assertEqual(self.metric.modality, Modality.POPHYS)
        self.assertEqual(self.metric.stage, Stage.PROCESSING)
        self.assertEqual(self.metric.type, qc.CURATION_TYPE)
        self.assertEqual(self.metric.value, self.references)

    def test_carries_the_grouping_tags(self) -> None:
        """The evaluation tag is what default_grouping groups on."""
        self.assertEqual(self.metric.tags["evaluation"], qc.EVALUATION)
        self.assertEqual(self.metric.tags["type"], "Operational QC")

    def test_awaits_review(self) -> None:
        """The metric lands in the reviewer queue rather than auto-passing."""
        self.assertEqual(len(self.metric.status_history), 1)
        self.assertEqual(self.metric.status_history[0].status, Status.PENDING)

    def test_no_rois_is_an_empty_value_list(self) -> None:
        """A plane with no ROIs still emits a well-formed metric."""
        metric = qc.build_dff_curation_metric("VISp_0", [])
        self.assertEqual(metric.value, [])


class TestQualityControlDocument(unittest.TestCase):
    """The full QualityControl wrapper and its writer."""

    def test_default_grouping(self) -> None:
        """The project-standard grouping is applied."""
        metric = qc.build_dff_curation_metric("VISp_0", ["a.png"])
        document = qc.build_quality_control([metric])
        self.assertEqual(document.default_grouping, ["evaluation"])
        self.assertEqual(len(document.metrics), 1)

    def test_write_round_trips(self) -> None:
        """The written document validates back as a full v2 QualityControl."""
        metric = qc.build_dff_curation_metric("VISp_0", ["a.png"])
        with TemporaryDirectory() as tmp:
            written = qc.write_quality_control_json(
                qc.build_quality_control([metric]), Path(tmp) / "nested"
            )
            self.assertTrue(written.exists())
            QualityControl.model_validate_json(written.read_text())


class TestZoomWindows(unittest.TestCase):
    """Inset window selection."""

    def test_three_windows_spanning_the_trace(self) -> None:
        """First, middle and last windows are returned in order."""
        t = np.arange(0, 300, 1.0)
        windows = qc._zoom_windows(t, 60.0)
        self.assertEqual(len(windows), 3)
        self.assertAlmostEqual(windows[0][0], 0.0)
        self.assertAlmostEqual(windows[0][1], 60.0)
        self.assertAlmostEqual(windows[2][1], t[-1])
        self.assertLess(windows[0][0], windows[1][0])
        self.assertLess(windows[1][0], windows[2][0])

    def test_window_shorter_than_zoom_is_clamped(self) -> None:
        """A trace shorter than the zoom still yields in-range windows."""
        t = np.arange(0, 10, 1.0)
        windows = qc._zoom_windows(t, 60.0)
        self.assertEqual(windows[2][0], t[0])


class TestPlotDff(unittest.TestCase):
    """The per-ROI diagnostic figure."""

    def setUp(self) -> None:
        """Synthesise a short trace with a decaying baseline."""
        self.frame_rate = 10.0
        n = 1200
        t = np.arange(n) / self.frame_rate
        self.baseline = 100.0 + 20.0 * np.exp(-t / 40.0)
        self.trace = self.baseline + np.sin(t) * 2.0
        self.dff = (self.trace - self.baseline) / self.baseline

    def _plot(self, tmp: str, **kwargs: object) -> Path:
        """Render one figure into ``tmp`` and return its path."""
        qc.plot_dff(
            self.trace,
            self.baseline,
            self.dff,
            self.frame_rate,
            "00",
            tmp,
            "VISp_0",
            **kwargs,
        )
        return Path(tmp) / "VISp_0_00_dff.png"

    def test_writes_png_with_insets(self) -> None:
        """The default three-row layout renders."""
        with TemporaryDirectory() as tmp:
            path = self._plot(tmp)
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 0)

    def test_writes_png_without_insets(self) -> None:
        """zoom_duration=0 renders the plain two-row layout."""
        with TemporaryDirectory() as tmp:
            path = self._plot(tmp, zoom_duration=0)
            self.assertTrue(path.exists())

    def test_title_includes_fit_log(self) -> None:
        """A triexp fit log is annotated onto the figure without error."""
        log = {
            "n_passes": 3,
            "winner_combo": (2, 4),
            "pass1_trigger": "low_f0",
        }
        with TemporaryDirectory() as tmp:
            self.assertTrue(self._plot(tmp, log=log).exists())

    def test_flat_trace_does_not_raise(self) -> None:
        """A constant dF/F gives a zero-width inset y-range; must not raise."""
        self.dff = np.zeros_like(self.dff)
        with TemporaryDirectory() as tmp:
            self.assertTrue(self._plot(tmp).exists())

    def test_inset_skipped_when_window_is_empty(self) -> None:
        """An out-of-range window draws nothing rather than failing."""
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        fig, ax = plt.subplots(2, 1)
        t = np.arange(0, 10, 1.0)
        qc._draw_inset(
            ax[0], ax[1], t, np.zeros_like(t), (100.0, 200.0), 0, 60.0
        )
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
