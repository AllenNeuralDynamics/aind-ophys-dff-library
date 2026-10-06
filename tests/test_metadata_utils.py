"""Tests for the dF/F metadata read layer and the Code builder."""

import json
import unittest
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

from aind_data_schema.core.processing import Processing
from aind_pophys_metadata.runner import stage_guard

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

V2_NO_SUBJECT = {
    key: value
    for key, value in V2_ACQUISITION.items()
    if key != "subject_id"
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

    def test_reads_v1_session(self) -> None:
        """A v1 session.json supplies frame rate, subject and plane ids."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "session.json", V1_SESSION)
            _write(root, "data_description.json", DATA_DESCRIPTION)
            metadata = mu.load_dff_metadata(root)
        self.assertEqual(metadata.frame_rate, 9.48)
        self.assertEqual(metadata.subject_id, "999999")
        self.assertEqual(metadata.dataset_name, DATA_DESCRIPTION["name"])
        self.assertEqual(metadata.fov_ids, ("plane_0",))

    def test_reads_v2_acquisition(self) -> None:
        """A v2 acquisition.json supplies the same fields."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "acquisition.json", V2_ACQUISITION)
            metadata = mu.load_dff_metadata(root)
        self.assertEqual(metadata.frame_rate, 9.48)
        self.assertIsNone(metadata.dataset_name)

    def test_platform_json_frame_rate_fallback(self) -> None:
        """A core file without a frame rate falls back to platform.json."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "session.json", {"subject_id": "1"})
            _write(root, "platform.json", PLATFORM)
            metadata = mu.load_dff_metadata(root)
        self.assertEqual(metadata.frame_rate, 11.0)

    def test_cli_frame_rate_fallback(self) -> None:
        """The CLI override supplies the rate when the metadata cannot."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "session.json", {"subject_id": "1"})
            metadata = mu.load_dff_metadata(root, cli_frame_rate=7.5)
        self.assertEqual(metadata.frame_rate, 7.5)

    def test_missing_frame_rate_raises_in_the_loader(self) -> None:
        """An unresolvable frame rate fails at load, naming the core file."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "session.json", {"subject_id": "1"})
            with self.assertRaises(ValueError) as caught:
                mu.load_dff_metadata(root)
        self.assertIn("session.json", str(caught.exception))

    def test_subject_json_fallback(self) -> None:
        """subject_id falls back to subject.json when the core file omits it."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "acquisition.json", V2_NO_SUBJECT)
            _write(root, "subject.json", {"subject_id": "NY129"})
            metadata = mu.load_dff_metadata(root)
        self.assertEqual(metadata.subject_id, "NY129")

    def test_missing_core_file_raises(self) -> None:
        """Neither core file present is a fail-fast FileNotFoundError."""
        with TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                mu.load_dff_metadata(Path(tmp))

    def test_both_core_files_raise(self) -> None:
        """Both core files present is ambiguous and rejected."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "session.json", V1_SESSION)
            _write(root, "acquisition.json", V2_ACQUISITION)
            with self.assertRaises(ValueError):
                mu.load_dff_metadata(root)

    def test_metadata_is_frozen(self) -> None:
        """The metadata view is immutable."""
        metadata = mu.DFFMetadata(frame_rate=9.48)
        with self.assertRaises(Exception):
            metadata.frame_rate = 1.0


class TestCodeBuilder(unittest.TestCase):
    """The v2 Code block identifying dF/F runs."""

    def test_version_and_url_identify_the_backing_library(self) -> None:
        """code.version is never empty, and the url names the same artifact."""
        code = mu.build_dff_code(
            parameters={"method": "triexp"},
            input_data=["VISp_0/extraction/VISp_0_extraction.h5"],
        )
        self.assertEqual(code.version, version(mu.LIBRARY_NAME))
        self.assertTrue(code.version)
        self.assertEqual(
            code.url,
            f"https://github.com/AllenNeuralDynamics/{mu.LIBRARY_NAME}",
        )
        self.assertNotIn("/tree/main/code", code.url)
        self.assertEqual(
            code.input_data[0].name,
            "VISp_0/extraction/VISp_0_extraction.h5",
        )

    def test_mid_run_parameters_merge_into_the_written_code(self) -> None:
        """Data-derived values join the up-front ones, and still write."""
        code = mu.build_dff_code(parameters={"method": "triexp"})
        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            with stage_guard(output_dir, mu.PROCESS_TYPE, code) as ctx:
                ctx.parameters = {"resolved_frame_rate": 9.48}
            written = json.loads((output_dir / "processing.json").read_text())
        self.assertEqual(
            written["data_processes"][0]["code"]["parameters"],
            {"method": "triexp", "resolved_frame_rate": 9.48},
        )

    def test_process_name_matches_the_emitted_document(self) -> None:
        """The excluded self-name equals the name the guard writes."""
        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            code = mu.build_dff_code(parameters={})
            with stage_guard(
                output_dir, mu.PROCESS_TYPE, code, plane_id="VISp_0"
            ):
                pass
            written = Processing.model_validate_json(
                (output_dir / "processing.json").read_text()
            )
        self.assertEqual(
            written.data_processes[0].name, mu.process_name("VISp_0")
        )


if __name__ == "__main__":
    unittest.main()
