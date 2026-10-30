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
training, metric and split code is unchanged.

## Decoder accounting correction

In the output-head runs, layer 0 of the local decoder counted input synaptic
accumulations as (active inputs) × (mean fan-in per destination neuron). Each
input channel has exactly 8 outgoing synapses, so the count should be
(active inputs) × 8. The dense decoder also truncated the mean fan-in to an
integer. Layer 1 of the two-layer decoder (128 → 128) has equal mean fan-in
and fan-out and was counted correctly. The counter is detached from the
computation graph, so training, BPB and every event count are unaffected.

| File | Change | Original copy |
|---|---|---|
| `paper_code/src/models/local_spiking_patch_decoder.py` | Layer-0 input accumulations use the per-input fan-out (`input_mask.sum(dim=0)`) | `provenance/pre_correction_runtime/src/models/local_spiking_patch_decoder.py` |
| `paper_code/src/models/event_accounted_wta_ttfs_decoder.py` | Same change in the TTFS decoder | `provenance/pre_correction_runtime/src/models/event_accounted_wta_ttfs_decoder.py` |

`scripts/correct_decoder_fanout.py` applies the same correction to the
recorded rows and recomputes the energy projection with
`paper_code/src/energy/wta_event_energy_projection.py`. For one-layer
decoders the corrected count is the recorded count × 8 / m₀, where m₀ is the
recorded per-input multiplier. For two-layer decoders the layer-0 active-input
total A₀ is rebuilt from the recorded context, feedback and position event
counts, and (8 − m₀) × A₀ is added; A₀ is exact for fixed patches and uses the
mean patch length of the shared segmentation for adaptive patches. The
original TTFS counter truncated each per-step product to an integer, so where
m₀ is 6.5 or 12.5 the corrected count can be lower than a rerun by up to 0.5
per decoder step. The
front-end comparison uses a different decoder whose count already used the
configured fan-out, so its archive is unchanged.

The recorded archive `results/temporal_decoders/` is kept unchanged. The
corrected archive adds the column `decoder_input_fanout_correction_version`.
`validate_release.py` checks the hashes of both archives and of the original
copies, and checks that the corrected archive has the same 162 runs in the same
order with identical `test_bpb`.

### Rerun check

Six Text8, seed-1 rows were rerun with the corrected counters on Bunya
(grid rows 73, 76, 82, 85, 100 and 103: fixed and adaptive eight-byte patches,
one- and two-layer decoders, dense and direct-TTFS heads). Test BPB and every
spike-event count match the recorded archive exactly. Decoder input
accumulations match `results/temporal_decoders_corrected/` as follows:

| Row | Decoder | Validation relative difference | Train relative difference |
|---|---|---|---|
| 73, 76, 82 | One-layer dense and TTFS; two-layer dense, fixed patches | 0 | 0 |
| 85 | Two-layer TTFS, fixed patches | 1.1 × 10⁻⁹ | 4.7 × 10⁻⁷ |
| 100 | Two-layer dense, adaptive patches | 5.3 × 10⁻⁶ | 4.1 × 10⁻⁷ |
| 103 | Two-layer TTFS, adaptive patches | 1.1 × 10⁻⁵ | 3.6 × 10⁻⁶ |

The largest relative difference in projected central validation energy is
2.2 × 10⁻⁷. The row-85 difference comes from the integer truncation in the
original counter, the row-100 difference from the mean-patch-length step, and
the row-103 difference from both. The column-by-column comparison and the software
versions of the rerun are in `provenance/decoder_correction_rerun/`.

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
| `results/temporal_decoders_corrected/all_rows.csv` | 162 | `1b06148f9e8d39ad95916c496e96e28330fcb21c7e0b2cfbd14546c505b740e2` |
| `results/temporal_decoders_corrected/correction_audit.csv` | 324 | `b9ecb308c755906c6986665c41f268afa483993d9794b97877a0482ae622dc1b` |

The CSVs in `results/front_end/`, `results/tokenizer_fits/` and
`results/temporal_decoders/` are the row-level outputs of the runs, unedited.
`results/temporal_decoders_corrected/` is generated from
`results/temporal_decoders/all_rows.csv` by `scripts/correct_decoder_fanout.py`.

## Contents

The repository contains the files required to:

1. construct and verify canonical splits;
2. fit required tokenizer artifacts;
3. execute the two paper experiment grids;
4. calculate the recorded metrics and analytical accounting fields; and
5. validate the raw archives

Exploratory runs, superseded prototypes, scheduler scripts and manuscript
files are not included.
