"""Re-export of the shared path helper.

``relative_to_root`` now lives in ``aind_pophys_metadata.paths`` alongside
``find_one``; this module is a temporary alias for callers that still import
it from here and can be removed once none do.
"""

from aind_pophys_metadata.paths import relative_to_root  # noqa: F401
