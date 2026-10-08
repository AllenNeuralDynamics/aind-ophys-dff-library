"""Tests for the dF/F orchestration against real compute-result types.

Every fixture standing in for a compute result is a numpy array at a
production shape, because that is the only shape that exercises what the
consumers really do: ``h5py`` dataset writes, ``scipy.stats.skew`` over an
axis, ``np.median`` on a possibly-empty array, and the per-ROI plot fan-out.
A list-shaped fixture satisfies all of these while hiding array truthiness
and axis errors from the suite.
"""

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import h5py
import numpy as np
from aind_pophys_metadata.runner import StageContext

from aind_ophys_dff_library import job
from aind_ophys_dff_library.settings import DFFSettings
from aind_ophys_dff_library.utils.metadata_utils import DFFMetadata

FRAME_RATE = 10.0
N_ROIS = 2
N_FRAMES = 1200
UNIQUE_ID = "VISp_0"


def synthetic_traces(n_rois: int = N_ROIS) -> np.ndarray:
    """Return neuropil-corrected traces of shape ``(n_rois, N_FRAMES)``.

    Parameters
    ----------
    n_rois : int, optional
        ROI count; ``0`` yields the empty-plane matrix.

    Returns
    -------
    numpy.ndarray
        Float traces with a decaying baseline and periodic activity.
    """
    t = np.arange(N_FRAMES) / FRAME_RATE
    baseline = 100.0 + 20.0 * np.exp(-t / 40.0)
    traces = np.empty((n_rois, N_FRAMES))
    for n in range(n_rois):
        traces[n] = baseline + 5.0 * (n + 1) * np.sin(t / (n + 2))
    return traces


def write_extraction_h5(directory: Path, traces: np.ndarray) -> Path:
    """Write ``traces`` as an upstream extraction file and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{UNIQUE_ID}_extraction.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("traces/corrected", data=traces)
    return path


class TestMedian(unittest.TestCase):
    """The per-ROI summary statistic written to output_parameters."""

    def test_median_of_a_numpy_array(self) -> None:
        """A populated per-ROI array reduces to a plain float."""
        median = job._median(np.array([1.0, 3.0, 5.0]))
        self.assertIsInstance(median, float)
        self.assertEqual(median, 3.0)

    def test_empty_array_is_none(self) -> None:
        """A zero-ROI array reports None rather than a nan median."""
        self.assertIsNone(job._median(np.empty((0,))))


class TestTracesHdf5(unittest.TestCase):
    """The traces HDF5 the downstream event-detection stage reads back."""

    def test_round_trips_every_dataset_at_shape(self) -> None:
        """dF/F, baseline, noise and skewness keep their compute shapes."""
        dff_traces = np.zeros((N_ROIS, N_FRAMES), dtype="f4")
        baseline = np.ones((N_ROIS, N_FRAMES), dtype="f4")
        noise = np.zeros(N_ROIS, dtype="f4")
        skewness = np.zeros(N_ROIS, dtype="f4")
        with TemporaryDirectory() as tmp:
            path = job._write_traces_h5(
                Path(tmp), UNIQUE_ID, dff_traces, baseline, noise, skewness
            )
            with h5py.File(path, "r") as f:
                self.assertEqual(f["data"].shape, (N_ROIS, N_FRAMES))
                self.assertEqual(f["baseline"].shape, (N_ROIS, N_FRAMES))
                self.assertEqual(f["noise"].shape, (N_ROIS,))
                self.assertEqual(f["skewness"].shape, (N_ROIS,))
        self.assertEqual(path.name, f"{UNIQUE_ID}_dff.h5")


class TestFitLogs(unittest.TestCase):
    """The per-ROI triexp fit logs, which arrive holding numpy arrays."""

    def test_numpy_and_tuple_keyed_logs_serialize(self) -> None:
        """A log as triexp emits it - ndarray values, tuple combo keys."""
        logs = [
            {
                "n_passes": 2,
                "winner_combo": (2, 4),
                "pass1": {(2, 4): np.arange(3, dtype="f4")},
                "f0": np.ones(N_FRAMES, dtype="f4"),
            }
        ]
        with TemporaryDirectory() as tmp:
            path = job._write_fit_logs(Path(tmp), UNIQUE_ID, logs)
            payload = json.loads(path.read_text())
        self.assertEqual(payload[0]["roi"], 0)
        self.assertEqual(payload[0]["winner_combo"], [2, 4])
        self.assertEqual(payload[0]["pass1"]["2_4"], [0.0, 1.0, 2.0])
        self.assertEqual(len(payload[0]["f0"]), N_FRAMES)

    def test_percentile_path_writes_an_empty_list(self) -> None:
        """The percentile baseline produces no logs, and that is not an error."""
        with TemporaryDirectory() as tmp:
            path = job._write_fit_logs(Path(tmp), UNIQUE_ID, None)
            self.assertEqual(json.loads(path.read_text()), [])


class TestRunDff(unittest.TestCase):
    """The full stage body, from the upstream HDF5 read to the QC document."""

    def _run(self, traces: np.ndarray, tmp: str) -> StageContext:
        """Run the stage body over ``traces`` and return its context."""
        root = Path(tmp)
        extraction_h5 = write_extraction_h5(root / "input", traces)
        output_dir = root / "results" / UNIQUE_ID / "dff"
        output_dir.mkdir(parents=True)
        # cli_parse_args=True would otherwise consume the test runner's argv.
        with patch.object(sys, "argv", ["dff.py"]):
            settings = DFFSettings(
                method="percentile",
                output_dir=root / "results",
                zoom_duration=0,
            )
        ctx = StageContext()
        job._run_dff(
            settings,
            DFFMetadata(frame_rate=FRAME_RATE),
            extraction_h5,
            UNIQUE_ID,
            output_dir,
            ctx,
        )
        return ctx

    def test_populated_plane(self) -> None:
        """Real traces flow through compute, HDF5, plots and QC."""
        with TemporaryDirectory() as tmp:
            ctx = self._run(synthetic_traces(), tmp)
            output_dir = Path(tmp) / "results" / UNIQUE_ID / "dff"
            with h5py.File(output_dir / f"{UNIQUE_ID}_dff.h5", "r") as f:
                self.assertEqual(f["data"].shape, (N_ROIS, N_FRAMES))
                self.assertEqual(f["skewness"].shape, (N_ROIS,))
            plots = sorted((output_dir / job.PLOTS_DIR).glob("*.png"))
            self.assertEqual(len(plots), N_ROIS)
            self.assertTrue((output_dir / "quality_control.json").exists())
        self.assertEqual(ctx.output_parameters["num_rois"], N_ROIS)
        self.assertEqual(ctx.output_parameters["num_frames"], N_FRAMES)
        self.assertIsInstance(ctx.output_parameters["median_noise"], float)
        self.assertIsInstance(
            ctx.output_parameters["median_dff_skewness"], float
        )

    def test_zero_roi_plane(self) -> None:
        """An empty (0, T) matrix writes honest empty outputs, not an error.

        The skewness reduction runs over axis 1 and the medians over an
        empty array, so the traces must arrive as a real 2-D matrix.
        """
        with TemporaryDirectory() as tmp:
            ctx = self._run(synthetic_traces(0), tmp)
            output_dir = Path(tmp) / "results" / UNIQUE_ID / "dff"
            with h5py.File(output_dir / f"{UNIQUE_ID}_dff.h5", "r") as f:
                self.assertEqual(f["data"].shape, (0, N_FRAMES))
            self.assertFalse((output_dir / job.PLOTS_DIR).exists())
            self.assertTrue((output_dir / "quality_control.json").exists())
        self.assertEqual(ctx.output_parameters["num_rois"], 0)
        self.assertEqual(ctx.output_parameters["num_frames"], N_FRAMES)
        self.assertIsNone(ctx.output_parameters["median_noise"])
        self.assertIsNone(ctx.output_parameters["median_dff_skewness"])


if __name__ == "__main__":
    unittest.main()
