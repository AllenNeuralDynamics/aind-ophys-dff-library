"""Tests for DFFSettings."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from aind_ophys_dff_library.settings import DFFSettings


def _make_settings(**kwargs: object) -> DFFSettings:
    """Construct DFFSettings with a clean ``sys.argv``.

    ``cli_parse_args=True`` would otherwise consume the test runner's own
    arguments and abort with SystemExit.
    """
    with patch.object(sys, "argv", ["dff.py"]):
        return DFFSettings(**kwargs)


class TestDFFSettings(unittest.TestCase):
    """Field defaults, validators and the app-panel contract."""

    def test_defaults(self) -> None:
        """Field defaults are the values the capsule ships with."""
        settings = _make_settings()
        self.assertEqual(settings.input_dir, Path("/data"))
        self.assertEqual(settings.output_dir, Path("/results"))
        self.assertEqual(settings.method, "triexp")
        self.assertEqual(settings.long_window, 60.0)
        self.assertEqual(settings.short_window, 3.333)
        self.assertEqual(settings.inactive_percentile, 10)
        self.assertEqual(settings.noise_method, "mad")
        self.assertEqual(settings.sigma_anneal_steps, 1)
        self.assertEqual(settings.triexp_config_overrides, "{}")
        self.assertEqual(settings.zoom_duration, 60.0)
        self.assertIsNone(settings.frame_rate)
        self.assertFalse(settings.verify)

    def test_method_is_constrained(self) -> None:
        """Only the two implemented baselines are accepted."""
        self.assertEqual(
            _make_settings(method="percentile").method, "percentile"
        )
        with self.assertRaises((ValidationError, SystemExit)):
            _make_settings(method="rolling-median")

    def test_overrides_accept_valid_json_object(self) -> None:
        """A JSON object passes through unchanged."""
        raw = '{"t_fast_min_s": 10.0}'
        self.assertEqual(
            _make_settings(
                triexp_config_overrides=raw
            ).triexp_config_overrides,
            raw,
        )

    def test_overrides_empty_normalises(self) -> None:
        """An empty value becomes an empty JSON object."""
        self.assertEqual(
            _make_settings(triexp_config_overrides="").triexp_config_overrides,
            "{}",
        )

    def test_overrides_reject_malformed_json(self) -> None:
        """A malformed app-panel value fails before any compute runs."""
        with self.assertRaises((ValidationError, SystemExit)):
            _make_settings(triexp_config_overrides="{not json")

    def test_overrides_reject_non_object_json(self) -> None:
        """A JSON scalar or array is not a valid override mapping."""
        for raw in ("[1, 2]", '"text"', "5"):
            with self.subTest(raw=raw):
                with self.assertRaises((ValidationError, SystemExit)):
                    _make_settings(triexp_config_overrides=raw)

    def test_no_single_character_field_names(self) -> None:
        """No field may be 1 character: it renders as an unusable short flag.

        pydantic-settings turns a one-character field into ``-x``, which a Code
        Ocean app panel (``--name=value`` only) can never emit.
        """
        short = [f for f in DFFSettings.model_fields if len(f) < 2]
        self.assertEqual(short, [])

    def test_every_field_is_cli_addressable(self) -> None:
        """Every field parses as ``--name=value``, the only form CO emits."""
        samples = {
            "input_dir": "/data",
            "output_dir": "/results",
            "method": "triexp",
            "frame_rate": "9.48",
            "long_window": "60.0",
            "short_window": "3.333",
            "inactive_percentile": "10",
            "noise_method": "mad",
            "sigma_anneal_steps": "1",
            "triexp_config_overrides": "{}",
            "zoom_duration": "60.0",
            "verify": "1",
        }
        self.assertEqual(sorted(samples), sorted(DFFSettings.model_fields))
        argv = ["dff.py"] + [
            f"--{name}={value}" for name, value in samples.items()
        ]
        with patch.object(sys, "argv", argv):
            settings = DFFSettings()
        self.assertTrue(settings.verify)
        self.assertEqual(settings.frame_rate, 9.48)


if __name__ == "__main__":
    unittest.main()
