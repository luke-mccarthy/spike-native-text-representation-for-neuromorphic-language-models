# Provenance

This repository is a curated reproduction artifact derived from the private
experimental repository `luke-mccarthy/spike-tokenizer-snn`.

## Frozen evidence

| Artifact | Source commit |
|---|---|
| Front-end runtime code | `95f148159b9b340c2f679c0d10eedf1e35b936f9` |
| Front-end raw test archive | `1c126ff45c504736c41800d5184de110306b0218` |
| Temporal-decoder runtime code | `e20afbd166099210cca18be089e6aae447f6e56a` |
| Temporal-decoder publication archive | `e0e4b66faf8f47e19a62cbc43ee98e8066665418` |
| Bunya environment evidence | `714d90b7f62e637c6f4682ded1ed55a2ab796fc5` |

The front-end files required at the earlier source commit have the same Git
blob identities at the temporal-decoder commit. The artifact therefore uses
one deduplicated `paper_code/` tree, with the later decoder-only files added.
The public runtime uses descriptive filenames and identifiers. Original
development paths and exact Git blob identities are retained in
`source_manifest.json`, with a human-readable summary in `SOURCE_MAPPING.md`.
The public entry point is `reproduce.py`.

Public entry scripts use project-relative defaults and
`SPIKE_TOKENIZER_CANONICAL_ROOT`/`SPIKE_TOKENIZER_FIT_ROOT` overrides instead
of private Bunya paths. Source-corpus paths are explicit command-line
arguments. The exact portability changes are enumerated in
`source_manifest.json`; unmodified copies of the two initially patched sources
are retained under `provenance/original_source/`. No model, training, metric,
split, or accounting operation was changed.

The software evidence was captured from the original Bunya Conda environment
at development commit `e0e4b66faf8f47e19a62cbc43ee98e8066665418`. The
sanitised full export is `environment-lock.yml`; exact evidence checksums and
the development-repository location are recorded in
`docs/bunya_environment.json`.

## Included raw archives

| File | Rows | SHA-256 |
|---|---:|---|
| `results/front_end/all_rows.csv` | 90 | `5973bc8d1175fa403858606a10b39651cdfe8f138b9142495a4c3ddc8e1d60a2` |
| `results/front_end/audit.json` | — | `f2d717c16cabe3d330fc4d12620d3d6f34620e44f444870bdcb0d6123424ac29` |
| `results/temporal_decoders/all_rows.csv` | 162 | `70e64ea88cd49856572121cb7d8e67879a8e7de7c37d84c51523ab4abe5366c2` |
| `results/temporal_decoders/audit.json` | — | `82495d4405905edeef5137b7aa4469f8ecabaac3f1e2c978eb68fb195e82eec6` |
| `results/temporal_decoders/grid_manifest.json` | — | `40c4c5c171fbfa36a8996b533cb654e2f6bab573c3f0761d4b2931cf2f94dabb` |

The CSVs are unchanged row-level outputs. No values are hand-copied into this
repository and no result interpretation is included.

## Inclusion rule

Only files required to:

1. construct and verify canonical splits;
2. fit required tokenizer artifacts;
3. execute the two paper experiment grids;
4. calculate the recorded metrics and analytical accounting fields; and
5. validate the raw archives

are included. Scheduler troubleshooting, superseded prototypes, exploratory
results, manuscript files, and later unreported experiments are excluded.
