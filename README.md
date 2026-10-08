# aind-ophys-dff-library

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Code Style](https://img.shields.io/badge/code%20style-black-black)
[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)
![Interrogate](https://img.shields.io/badge/interrogate-100.0%25-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)
![Python](https://img.shields.io/badge/python->=3.10-blue?logo=python)

dF/F estimation for AIND planar optical physiology. This is the
backing library for the `aind-ophys-dff` Code Ocean capsule: the capsule is a
thin wrapper that calls `job.run()`, and all logic, settings and tests live
here.

## Usage

The library entry point is `run()`, which processes one plane per invocation:

```python
from aind_ophys_dff_library.job import run

run()  # settings parsed from CLI arguments / DFF_* environment variables
```

`run()` reads the plane's upstream extraction output, computes dF/F, and writes
its outputs and metadata. Settings may also be constructed explicitly:

```python
from aind_ophys_dff_library.job import run
from aind_ophys_dff_library.settings import DFFSettings

run(DFFSettings(method="percentile", zoom_duration=0))
```

### Settings

`DFFSettings` (see `src/aind_ophys_dff_library/settings.py`) is a
`pydantic-settings` model. Every field is addressable as `--name=value` on the
command line or as a `DFF_`-prefixed environment variable; the Code Ocean app
panel is generated from this class. Key fields:

| field | default | meaning |
|---|---|---|
| `input_dir` | `/data` | directory holding the extraction output and metadata |
| `output_dir` | `/results` | results root |
| `method` | `triexp` | `triexp` (triexponential baseline fit) or `percentile` (rolling-percentile baseline) |
| `frame_rate` | `None` | fallback [Hz], used only when the metadata supplies none |
| `sigma_anneal_steps` | `1` | IRLS sigma-annealing steps (`triexp` only) |
| `triexp_config_overrides` | `{}` | JSON object of `set_dff_config` overrides (`triexp` only) |
| `long_window`, `short_window`, `inactive_percentile`, `noise_method` | | `percentile` parameters |
| `zoom_duration` | `60.0` | width [s] of each zoomed inset in the QC figure; `0` disables the insets |

## Modules

### `job.py`

Orchestration. `run()` reads the metadata, locates the upstream
`*/extraction/*extraction.h5`, dispatches to the selected algorithm via
`compute_dff()`, writes every output, and emits the metadata documents.
Orchestration and IO only; the QC figures and metric live in `qc.py`.

### `settings.py`

`DFFSettings`. Kept import-light (pydantic plus stdlib) so it is cheap to
import and free of numpy / matplotlib / aind-data-schema side effects.

### `triexp_dff.py`

The triexponential baseline fit. `dff()` returns
`(dFF, F0, noise_sd, params, logs)`; `log_to_jsonable()` normalises a per-ROI
fit log into a JSON-serialisable shape.

### `dff_config.py`

`set_dff_config()` pre-computes the per-ROI initial values, bounds and noise
estimates consumed by `triexp_dff.dff()`, returning a frozen `DffConfig`.

### `qc.py`

The per-ROI diagnostic figures (`plot_dff()`) and the `CurationMetric` that
points a reviewer at them (`build_dff_curation_metric()`), plus the
`build_quality_control()` / `write_quality_control_json()` wrappers over the
shared `aind-pophys-metadata` helpers.

### `utils/metadata_utils.py`

The aind-data-schema v1/v2 read layer: `load_dff_metadata()` assembles the
frozen `DFFMetadata` view, `build_dff_code()` builds the dF/F `Code`
block, and `process_name()` reproduces the `DataProcess.name` the run emits.

## Outputs

Everything is written under `<output_dir>/<plane_id>/dff/`:

- `<plane_id>_dff.h5` — datasets `data` (dF/F), `baseline`, `noise`, `skewness`
- `<plane_id>_dff_logs.json` — per-ROI triexp fit logs (`[]` on the percentile path)
- `plots/<plane_id>_<roi_id>_dff.png` — one diagnostic figure per ROI
- `processing.json` — full v2 `Processing`, written by the shared `stage_guard`
- `quality_control.json` — full v2 `QualityControl` holding the dF/F curation metric

`processing.json` and `quality_control.json` are emitted through the shared
[`aind-pophys-metadata`](https://github.com/AllenNeuralDynamics/aind-pophys-metadata)
package, so their shape matches every other processing step in the pipeline.

## Level of Support
Please indicate a level of support:
 - [ ] Supported: We are releasing this code to the public as a tool we expect others to use. Issues are welcomed, and we expect to address them promptly; pull requests will be vetted by our staff before inclusion.
 - [ ] Occasional updates: We are planning on occasional updating this tool with no fixed schedule. Community involvement is encouraged through both issues and pull requests.
 - [ ] Unsupported: We are not currently supporting this code, but simply releasing it to the community AS IS but are not able to provide any guarantees of support. The community is welcome to submit issues, but you should not expect an active response.

## Installation
To use the software, in the root directory, run
```bash
pip install -e .
```

To develop the code, run
```bash
pip install -e . --group dev
```
Note: --group flag is available only in pip versions >=25.1

Alternatively, if using [uv](https://docs.astral.sh/uv/), run
```bash
uv sync
```

## Tests and linting

Tests are `unittest`. The CI gate is:

```bash
flake8 .
interrogate --verbose .
coverage run -m unittest discover
coverage report
```

`flake8` allows 100-character lines; `coverage` and `interrogate` are both
gated at 100%. `triexp_dff.py`, `dff_config.py` and `job.py` are omitted from
the coverage measurement because exercising them fully needs real extraction
output; they are covered by the Code Ocean data run.
