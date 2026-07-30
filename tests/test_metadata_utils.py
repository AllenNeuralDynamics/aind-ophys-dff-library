"""Tests for the dF/F metadata read layer and Processing builders."""

import json
import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from aind_data_schema.core.processing import Processing

from aind_ophys_dff_library.utils import metadata_utils as mu

V1_SESSION = {
    "subject_id": "999999",
    "rig_id": "MESO.1",
    "data_streams": [{"ophys_fovs": [{"index": 0, "frame_rate": "9.48"}]}],
}

V2_ACQUISITION = {
    "subject_id": "999999",
    "instrument_id": "MESO.1",
    "data_streams": [
        {
            "configurations": [
                {
                    "object_type": "Imaging config",
                    "sampling_strategy": {"frame_rate": 9.48},
                }
            ]
        }
    ],
}

PLATFORM = {"imaging_plane_groups": [{"acquisition_framerate_Hz": 11.0}]}
DATA_DESCRIPTION = {"name": "multiplane-ophys_999999_2026-07-29_00-00-00"}


def _write(directory: Path, name: str, payload: dict) -> Path:
    """Write ``payload`` as JSON into ``directory`` and return the path."""
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


class TestLoadDFFMetadata(unittest.TestCase):
    """v1/v2 dispatch for the fields dF/F consumes."""

    def test_reads_v1_session(self):
        """A v1 session.json supplies frame rate and subject."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "session.json", V1_SESSION)
            _write(root, "data_description.json", DATA_DESCRIPTION)
            metadata = mu.load_dff_metadata(root)
        self.assertEqual(metadata.frame_rate, 9.48)
        self.assertEqual(metadata.subject_id, "999999")
        self.assertEqual(metadata.dataset_name, DATA_DESCRIPTION["name"])

    def test_reads_v2_acquisition(self):
        """A v2 acquisition.json supplies the same fields."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "acquisition.json", V2_ACQUISITION)
            metadata = mu.load_dff_metadata(root)
        self.assertEqual(metadata.frame_rate, 9.48)
        self.assertIsNone(metadata.dataset_name)

    def test_platform_json_frame_rate_fallback(self):
        """A core file without a frame rate falls back to platform.json."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "session.json", {"subject_id": "1"})
            _write(root, "platform.json", PLATFORM)
            metadata = mu.load_dff_metadata(root)
        self.assertEqual(metadata.frame_rate, 11.0)

    def test_subject_json_fallback(self):
        """subject_id falls back to subject.json when the core file omits it."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "acquisition.json", {"instrument_id": "MESO.1"})
            _write(root, "subject.json", {"subject_id": "NY129"})
            metadata = mu.load_dff_metadata(root)
        self.assertEqual(metadata.subject_id, "NY129")

    def test_missing_core_file_raises(self):
        """Neither core file present is a fail-fast FileNotFoundError."""
        with TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                mu.load_dff_metadata(Path(tmp))

    def test_both_core_files_raise(self):
        """Both core files present is ambiguous and rejected."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "session.json", V1_SESSION)
            _write(root, "acquisition.json", V2_ACQUISITION)
            with self.assertRaises(ValueError):
                mu.load_dff_metadata(root)

    def test_metadata_is_frozen(self):
        """The metadata view is immutable."""
        metadata = mu.DFFMetadata(frame_rate=9.48)
        with self.assertRaises(Exception):
            metadata.frame_rate = 1.0


class TestProcessingBuilders(unittest.TestCase):
    """The v2 DataProcess / Processing builders."""

    def setUp(self):
        """Fix the timestamps used by every builder assertion."""
        self.start = datetime(2026, 7, 29, 12, 0, tzinfo=timezone.utc)
        self.end = datetime(2026, 7, 29, 12, 5, tzinfo=timezone.utc)

    def test_build_data_process(self):
        """A successful run yields a dF/F DataProcess, no notes."""
        data_process = mu.build_data_process(
            parameters={"method": "triexp"},
            start_time=self.start,
            end_time=self.end,
            input_data=["VISp_0/extraction/VISp_0_extraction.h5"],
            output_path="VISp_0/dff",
            output_parameters={"num_rois": 12},
        )
        self.assertEqual(data_process.process_type, "dF/F estimation")
        self.assertEqual(str(data_process.output_path), "VISp_0/dff")
        self.assertIsNone(data_process.notes)
        self.assertEqual(data_process.experimenters, [])
        self.assertEqual(
            data_process.code.input_data[0].name,
            "VISp_0/extraction/VISp_0_extraction.h5",
        )
        self.assertIsNotNone(data_process.resources)

    def test_build_data_process_records_error(self):
        """A failed run records the exception in notes."""
        data_process = mu.build_data_process(
            parameters={},
            start_time=self.start,
            end_time=self.end,
            error=ValueError("no frame rate"),
        )
        self.assertIn("ValueError", data_process.notes)
        self.assertIn("no frame rate", data_process.notes)

    @patch.dict(
        "os.environ",
        {
            "PIPELINE_NAME": "aind-pophys-pipeline",
            "PIPELINE_URL": "https://example.invalid/p",
            "PIPELINE_VERSION": "2.0.0",
        },
        clear=False,
    )
    def test_build_processing_populates_pipeline_from_env(self):
        """pipelines and pipeline_name come from the same env read."""
        data_process = mu.build_data_process(
            parameters={}, start_time=self.start, end_time=self.end
        )
        processing = mu.build_processing([data_process])
        self.assertEqual(len(processing.pipelines), 1)
        self.assertEqual(
            processing.data_processes[0].pipeline_name,
            processing.pipelines[0].name,
        )

    def test_write_processing_json_round_trips(self):
        """The written document validates back as a full v2 Processing."""
        data_process = mu.build_data_process(
            parameters={}, start_time=self.start, end_time=self.end
        )
        with TemporaryDirectory() as tmp:
            written = mu.write_processing_json(
                mu.build_processing([data_process]), Path(tmp) / "nested"
            )
            self.assertTrue(written.exists())
            Processing.model_validate_json(written.read_text())


if __name__ == "__main__":
    unittest.main()
