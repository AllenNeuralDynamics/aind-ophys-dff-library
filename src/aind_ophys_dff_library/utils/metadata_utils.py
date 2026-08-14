"""aind-data-schema v1/v2 metadata reads for dF/F estimation.

A thin, capsule-specific layer over the shared ``aind_pophys_metadata``
library. The v1/v2 dict paths, schema-version dispatch, the frame-rate
resolution ladder and the ``Code``/``Processing`` builders all live in the
shared library; this module only assembles the ``DFFMetadata`` view and this
capsule's ``Code`` block.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from aind_data_schema.components.identifiers import Code
from aind_data_schema_models.process_names import ProcessName
from aind_pophys_metadata import CoreMetadata
from aind_pophys_metadata import processing as shared_processing

CODE_NAME = "aind-ophys-dff"
# The shared builder derives ``code.version`` from this distribution name, so
# the version is the released backing library's, which bumps automatically on
# merge.
LIBRARY_NAME = "aind-ophys-dff-library"
# Only this repo knows where its own code lives, so the url is supplied here
# rather than derived by the shared builder.
CODE_URL = "https://github.com/AllenNeuralDynamics/aind-ophys-dff-library"

PROCESS_TYPE = ProcessName.DF_F_ESTIMATION


@dataclass(frozen=True)
class DFFMetadata:
    """Minimal metadata the dF/F capsule consumes.

    Attributes
    ----------
    frame_rate : float
        Sampling frequency [Hz]. Sets the baseline-window lengths and the plot
        time axis, so the computation cannot run without it; it is resolved
        and validated in :func:`load_dff_metadata` rather than deep inside the
        run, so a rate-less asset fails with a message naming the file and the
        schema version.
    subject_id : str or None
        Subject identifier, used for logging only.
    dataset_name : str or None
        Derived asset name, used for logging only.
    fov_ids : tuple of str
        Canonical plane/FOV ids from the acquisition metadata, used for
        logging and reporting.
    """

    frame_rate: float
    subject_id: Optional[str] = None
    dataset_name: Optional[str] = None
    fov_ids: Tuple[str, ...] = ()


def load_dff_metadata(
    input_dir: Path, cli_frame_rate: Optional[float] = None
) -> DFFMetadata:
    """Read v1/v2 metadata for dF/F from ``input_dir``.

    Dispatches on which core file is present (``acquisition.json`` -> v2,
    ``session.json`` -> v1). ``frame_rate`` falls back to ``platform.json``
    and then to ``cli_frame_rate``, and is fatal if still unresolved.

    Parameters
    ----------
    input_dir : Path
        Directory searched recursively for the metadata files.
    cli_frame_rate : float, optional
        Operator-supplied frame rate [Hz], used only when the metadata
        supplies none.

    Returns
    -------
    DFFMetadata
        The minimal fields this capsule consumes.

    Raises
    ------
    FileNotFoundError
        If no core metadata file is found.
    ValueError
        If both v1 and v2 core files are present (ambiguous input), or if no source
        supplies a frame rate.
    """
    core = CoreMetadata.load(Path(input_dir))
    return DFFMetadata(
        frame_rate=core.get_frame_rate(
            cli_override=cli_frame_rate, required=True
        ),
        subject_id=core.get_subject_id(),
        dataset_name=core.get_dataset_name(),
        fov_ids=core.get_fov_ids(),
    )


def build_dff_code(
    parameters: dict,
    input_data: Optional[List[str]] = None,
) -> Code:
    """Construct the ``Code`` block identifying this dF/F run.

    Parameters
    ----------
    parameters : dict
        Input parameters this run was configured with.
    input_data : list of str, optional
        Input data path(s)/asset name(s) for provenance.

    Returns
    -------
    Code
        The populated code block. ``version`` is the installed backing
        library's released version, so it is never an empty string in a run
        where that library is installed - which is every real run.
    """
    return shared_processing.build_code(
        CODE_NAME,
        library_name=LIBRARY_NAME,
        url=CODE_URL,
        parameters=parameters,
        input_data=input_data,
    )


def process_name(plane_id: str) -> str:
    """Return the ``DataProcess.name`` this capsule emits for a plane.

    Mirrors the composition
    :func:`aind_pophys_metadata.processing.build_data_process` performs, so
    the name can be excluded from its own upstream dependency list.

    Parameters
    ----------
    plane_id : str
        Plane/FOV identifier.

    Returns
    -------
    str
        The composed process name.
    """
    return f"{plane_id}: {PROCESS_TYPE.value}"
