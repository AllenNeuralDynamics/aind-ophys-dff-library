"""CLI/environment settings for dF/F estimation.

Kept import-light (pydantic + stdlib only) so it stays cheap to import and
free of numpy / matplotlib / aind-data-schema side effects. The JSON *shape*
of ``triexp_config_overrides`` is validated here so a malformed app-panel
value fails before any compute; validating its *keys* needs
``set_dff_config``'s signature and therefore happens in :mod:`job`.
"""

import json
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DFFSettings(BaseSettings, cli_parse_args=True):
    """Settings for dF/F calculation."""

    input_dir: Path = Field(
        default=Path("/data"),
        description=(
            "Input directory containing extraction.h5 and metadata files"
        ),
    )
    output_dir: Path = Field(
        default=Path("/results"),
        description="Output directory where results are saved",
    )
    method: Literal["triexp", "percentile"] = Field(
        default="triexp",
        description=(
            "dF/F algorithm: 'triexp' (parametric 3-pass triexponential "
            "baseline fit) or 'percentile' (rolling-percentile baseline)."
        ),
    )
    frame_rate: Optional[float] = Field(
        default=None,
        description=(
            "Frame rate [Hz] fallback used only when it cannot be read from "
            "acquisition/session/platform metadata."
        ),
    )

    # Percentile parameters — read only when method == 'percentile'.
    long_window: float = Field(
        default=60.0,
        description=(
            "Percentile baseline window (s). Used only when "
            "method='percentile'."
        ),
    )
    short_window: float = Field(
        default=3.333,
        description=(
            "Short detrending window (s). Used only when method='percentile'."
        ),
    )
    inactive_percentile: int = Field(
        default=10,
        description=(
            "Inactive percentile for F0. Used only when method='percentile'."
        ),
    )
    noise_method: str = Field(
        default="mad",
        description=(
            "Noise estimator ('mad'|'fft'|'welch'). Used only when "
            "method='percentile'."
        ),
    )

    # Triexp parameters — read only when method == 'triexp'.
    sigma_anneal_steps: int = Field(
        default=1,
        description=(
            "IRLS sigma-annealing steps (aind_ophys_utils.baseline_fitting."
            "nonlinear_fit). 1 (default) or 2 disable annealing "
            "(single-jump); 3 enables one intermediate step; 4 uses two; "
            "higher graduates further but exceeds the internal maxiter=5. "
            "Used only when method='triexp'."
        ),
    )
    triexp_config_overrides: str = Field(
        default="{}",
        description=(
            "JSON object of overrides for set_dff_config. Keys must be a "
            "subset of set_dff_config's keyword arguments (except "
            "sigma_anneal_steps, which has its own dedicated field above); "
            "any key omitted falls back to the function's default. Used only "
            "when method='triexp'."
        ),
    )

    # Output options
    zoom_duration: float = Field(
        default=60.0,
        description=(
            "Duration [s] of each zoomed inset in the per-ROI QC plot. Set to "
            "0 for a plain two-panel plot with no insets."
        ),
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="DFF_",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("triexp_config_overrides")
    @classmethod
    def _validate_overrides_json(cls, value: str) -> str:
        """Reject a triexp_config_overrides that is not a JSON object.

        Only the shape is checked here; the allowed key set is enforced in
        :func:`aind_ophys_dff_library.job.parse_triexp_overrides`, which can
        see ``set_dff_config``'s signature.

        Raises
        ------
        ValueError
            If ``value`` is not valid JSON, or parses to something other than
            a JSON object.
        """
        if not value:
            return "{}"
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"triexp_config_overrides is not valid JSON: {exc}"
            ) from exc
        if not isinstance(parsed, dict):
            raise ValueError(
                f"triexp_config_overrides must be a JSON object, got "
                f"{type(parsed).__name__}"
            )
        return value
