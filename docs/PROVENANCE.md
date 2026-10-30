# Provenance

The code and results in this repository were taken from the authors'
development repository `luke-mccarthy/spike-tokenizer-snn` at the commits
below.

## Source commits

| Artifact | Source commit |
|---|---|
| Front-end runtime code | `95f148159b9b340c2f679c0d10eedf1e35b936f9` |
| Front-end raw test archive | `1c126ff45c504736c41800d5184de110306b0218` |
| Temporal-decoder runtime code | `e20afbd166099210cca18be089e6aae447f6e56a` |
| Temporal-decoder publication archive | `e0e4b66faf8f47e19a62cbc43ee98e8066665418` |
| Bunya environment evidence | `714d90b7f62e637c6f4682ded1ed55a2ab796fc5` |

The front-end files used at the front-end commit have the same Git blob
identities at the temporal-decoder commit, so `paper_code/` is a single tree
containing both sets of files. Files carry descriptive names; their original
development paths and Git blob identities are in `source_manifest.json` and
summarised in `SOURCE_MAPPING.md`. The entry point is `reproduce.py`.

Entry scripts use project-relative default paths, which can be overridden
with the `SPIKE_TOKENIZER_CANONICAL_ROOT` and `SPIKE_TOKENIZER_FIT_ROOT`
environment variables. Source-corpus paths are command-line arguments. These
path changes are listed in `source_manifest.json`, and the original versions
of the two changed files are in `provenance/original_source/`. Model,
training, metric, split and accounting code is unchanged.

The software environment was exported from the original Bunya Conda
environment at development commit `e0e4b66faf8f47e19a62cbc43ee98e8066665418`.
The full export is `environment-lock.yml`; its checksums and source location
are in `docs/bunya_environment.json`.

## Included raw archives

| File | Rows | SHA-256 |
|---|---:|---|
| `results/front_end/all_rows.csv` | 90 | `5973bc8d1175fa403858606a10b39651cdfe8f138b9142495a4c3ddc8e1d60a2` |
| `results/front_end/audit.json` | — | `f2d717c16cabe3d330fc4d12620d3d6f34620e44f444870bdcb0d6123424ac29` |
| `results/tokenizer_fits/audit.json` | 9 | `278a49c7de36a53464a89469171f10b5f46c3e08732ab1dba268b3bacf6dc9f0` |
| `results/temporal_decoders/all_rows.csv` | 162 | `70e64ea88cd49856572121cb7d8e67879a8e7de7c37d84c51523ab4abe5366c2` |
| `results/temporal_decoders/audit.json` | — | `82495d4405905edeef5137b7aa4469f8ecabaac3f1e2c978eb68fb195e82eec6` |
| `results/temporal_decoders/grid_manifest.json` | — | `40c4c5c171fbfa36a8996b533cb654e2f6bab573c3f0761d4b2931cf2f94dabb` |

The CSVs are the row-level outputs of the runs, unedited.

## Contents

The repository contains the files required to:

1. construct and verify canonical splits;
2. fit required tokenizer artifacts;
3. execute the two paper experiment grids;
4. calculate the recorded metrics and analytical accounting fields; and
5. validate the raw archives

Exploratory runs, superseded prototypes, scheduler scripts and manuscript
files are not included.
