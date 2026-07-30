"""Tests for the provenance path helper.

The symlink case is the one that matters: inside the pipeline ``/results`` and
``/data`` are symlinks into the task work directory, and comparing a resolved
root against an unresolved path silently leaks absolute container paths such as
``/results/VISp_0/dff`` into ``processing.json``.
"""

import os
import tempfile
import unittest
from pathlib import Path

from aind_ophys_dff_library.utils.paths import relative_to_root


class TestRelativeToRoot(unittest.TestCase):
    """relative_to_root resolves both sides before comparing."""

    def test_plain_relative(self):
        """A path under the root becomes relative."""
        root = Path(tempfile.mkdtemp())
        target = root / "VISp_0" / "dff"
        target.mkdir(parents=True)
        self.assertEqual(relative_to_root(root, target), "VISp_0/dff")

    def test_symlinked_root(self):
        """An unresolved symlink path still yields a relative result."""
        base = Path(tempfile.mkdtemp())
        real = base / "capsule" / "results"
        real.mkdir(parents=True)
        link = base / "results_link"
        os.symlink(real, link)

        target = real / "VISp_0" / "dff"
        target.mkdir(parents=True)

        self.assertEqual(
            relative_to_root(real, link / "VISp_0" / "dff"), "VISp_0/dff"
        )
        self.assertEqual(relative_to_root(link, target), "VISp_0/dff")

    def test_outside_root_falls_back_to_absolute(self):
        """A path genuinely outside the root keeps its absolute form."""
        root = Path(tempfile.mkdtemp())
        other = Path(tempfile.mkdtemp()) / "elsewhere.h5"
        self.assertEqual(relative_to_root(root, other), str(other))

    def test_nonexistent_paths_do_not_raise(self):
        """Resolution of not-yet-created paths still works."""
        root = Path(tempfile.mkdtemp())
        self.assertEqual(
            relative_to_root(root, root / "plane_0" / "dff"), "plane_0/dff"
        )

    def test_plot_reference_relative(self):
        """The per-ROI plot reference recorded in the curation metric."""
        root = Path(tempfile.mkdtemp())
        plot = root / "VISp_0" / "dff" / "plots" / "VISp_0_00_dff.png"
        plot.parent.mkdir(parents=True)
        plot.touch()
        self.assertEqual(
            relative_to_root(root, plot),
            "VISp_0/dff/plots/VISp_0_00_dff.png",
        )


if __name__ == "__main__":
    unittest.main()
