"""Tests for the path-helper re-export.

The implementation and its behavioural tests (including the symlinked-root
case) live in ``aind_pophys_metadata``; all this module guards is that the
local name still resolves to that one implementation rather than drifting
back into a private copy.
"""

import unittest

from aind_pophys_metadata.paths import relative_to_root as shared
from aind_ophys_dff_library.utils import paths


class TestRelativeToRootReExport(unittest.TestCase):
    """The local name is the shared implementation."""

    def test_is_shared_implementation(self):
        """No private copy of the helper is reintroduced."""
        self.assertIs(paths.relative_to_root, shared)


if __name__ == "__main__":
    unittest.main()
