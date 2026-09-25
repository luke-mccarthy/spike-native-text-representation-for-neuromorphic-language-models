# Spike-Native Text Representation for Neuromorphic Language Models

Reproduction code, frozen configurations, and raw result records for the paper
*Spike-Native Text Representation for Neuromorphic Language Models*.

This repository is a compact paper artifact. It contains only:

- the model, text-representation, training, evaluation, and accounting code
  needed by the reported experiments;
- the immutable experiment grids used to launch the reported rows;
- the raw row-level CSV outputs and machine-readable audit records;
- the canonical dataset-split builder and exact file hashes;
- a pinned software environment and lightweight integrity tests.

It is not the development repository and is not intended for new experiment
design. New trials belong in a separate experimental repository. This artifact
should remain a frozen, reviewable implementation of the paper experiments.

## Repository layout

```text
configs/                  Frozen paper experiment grids
docs/                     Dataset, environment, and provenance documentation
results/
  front_end/              Raw 90-row front-end comparison
  tokenizer_fits/         Raw 9-row tokenizer-fit records and audit
  temporal_decoders/      Raw 162-row decoder factorial
scripts/                  Release-integrity utilities and figure generation
paper_code/               Curated paper runtime with descriptive public names
provenance/               Unmodified originals of portability-patched files
tests/                    Release-integrity tests
environment.yml           Portable pinned runtime environment
environment-lock.yml      Archived full environment export
reproduce.py              Friendly single-row reproduction interface
```

The files under `results/` are raw experimental records. This repository does
not provide a narrative interpretation of those values.

## Code map: networks, front-ends, and decoders

The paper decomposes the pipeline into four stages (raw bytes -> front-end
units/spikes -> SpikeGPT-like global core -> local/terminal decoder ->
p(byte)). The table below points at where each stage actually lives, so you
do not have to guess from file names alone. Exact original (private-repo)
paths for every file are in [docs/SOURCE_MAPPING.md](docs/SOURCE_MAPPING.md).

| Paper concept | Primary file(s) |
|---|---|
| SpikeGPT-like global backbone (RWKV time-mix/channel-mix, WKV recurrence, sequence LIF neuron, ATan surrogate) | `paper_code/src/models/spikegpt_reference.py` |
| Conventional BPE/Unigram subword front-end (token ID -> learned binary embedding) | `paper_code/src/models/spike_input_interfaces.py`, `paper_code/scripts/fit_sentencepiece_tokenizer.py` |
| Predictive spike-boundary + digital-codec hybrid front-end | `paper_code/src/tokenizers/spike_boundary_native_levels.py`, `spike_boundary_refined.py` |
| Codec-free byte spike codes (1-of-256, 2-of-32, recurrent-LIF reservoir + injective suffix) | `paper_code/src/models/spike_input_interfaces.py` |
| Injective/constant-weight codes shared across byte and patch front-ends | `paper_code/src/tokenizers/lossless_codecs.py` |
| Codec-free positional patch front-ends (fixed patch-4/8, adaptive patch-8, homeostatic boundary rule) + local LIF decoder with dense 256-way readout | `paper_code/src/models/spike_native_patch_lm.py`, `local_spiking_patch_decoder.py`, `spike_count_patch_decoder.py`, `paper_code/src/tokenizers/spike_native_patch_frontends.py` |
| Strict TTFS output heads (direct 256-neuron race, factorised 2x16-neuron nibble race) | `paper_code/src/models/hierarchical_ttfs_patch_decoder.py`, `strict_ttfs_patch_decoder.py`, `continuous_wta_ttfs_decoder.py`, `event_accounted_wta_ttfs_decoder.py`, `audited_ttfs_patch_decoder.py` (later files in this list are additive refinements of the earlier ones; `audited_ttfs_patch_decoder.py` is what produced the reported paired-decoder results) |
| Operation/energy ledgers (dense MACs, SynOps, state updates, routed spikes, low/central/high coefficients) | `paper_code/src/energy/analytical_compute.py`, `frontend_energy_projection.py`, `frontend_memory.py`, `scalable_patch_projection.py`, `wta_event_energy_projection.py`, plus `paper_code/src/models/spikegpt_operation_accounting.py` |
| Canonical dataset splitting and the fail-closed data contract | `paper_code/src/canonical_data_contract.py`, `paper_code/scripts/prepare_canonical_splits.py` |
| Per-row training/evaluation entry points | `paper_code/scripts/train_frontend_language_model.py` (subword/boundary/byte rows), `train_patch_language_model.py` (patch rows, the front-end quality/energy results), `train_ttfs_decoder.py` + `run_temporal_decoder_comparison.py` (the paired output-decoder results) |

`paper_code/src/tokenizers/blt_patcher.py`, `spiking_blt_frontend.py`, and
`gpt_neox_20b.py` are supporting infrastructure exercised during development
(BLT-style and GPT-NeoX-tokenizer comparisons) rather than one of the
front-end families reported in the paper; they are included because the
frozen front-end commit imports them.

Table/figure numbers are intentionally omitted from this README, since the
manuscript is still under revision and numbering moves between drafts. See
the current manuscript itself for exact table/figure numbers, and match
results by content (front-end quality, energy summary, paired decoder
comparison, etc.) rather than by number.

## Setup and full reproduction walkthrough

### 1. Install the environment

```bash
conda env create -f environment.yml
conda activate spike-native-text
python reproduce.py validate
```

`validate` checks the committed raw archives, hashes, and provenance records
only; it does not require any dataset and always works on a fresh checkout.

### 2. Obtain the three source corpora

This repository does not redistribute dataset payloads (see
[docs/DATASETS.md](docs/DATASETS.md) for exact requirements and licensing).
You need, locally:

- **Text8**: the standard 100,000,000-byte `text8` corpus.
- **WikiText-2**: the official `wiki.train.tokens` / `wiki.valid.tokens` /
  `wiki.test.tokens` files.
- **TinyStories-5000**: a 5,000-story, `<|endoftext|>`-delimited file. Build
  it with:

  ```bash
  pip install datasets
  python scripts/fetch_tinystories_5000.py --output data/sources/tinystories_5000.txt
  ```

  This is confirmed byte-for-byte identical to the original source file used
  by the study (see [docs/DATASETS.md](docs/DATASETS.md) for the reference
  SHA-256 and how to check it).

### 3. Build the canonical, hash-verified splits

```bash
python reproduce.py prepare-data \
  --text8-full /path/to/text8 \
  --tinystories-5000 data/sources/tinystories_5000.txt \
  --wikitext2-train /path/to/wiki.train.tokens \
  --wikitext2-validation /path/to/wiki.valid.tokens \
  --wikitext2-test /path/to/wiki.test.tokens \
  --output-root data/canonical

python reproduce.py validate
```

The second `validate` call now also compares your local
`data/canonical/manifest.json` against the paper's recorded
`docs/canonical_file_manifest.json` (byte counts and SHA-256 per file) and
fails loudly if anything differs, so you know immediately whether your local
data matches the paper's before spending compute on training.

### 4. Fit the conventional tokenizers

Nine rows (BPE-8k, BPE-16k, Unigram-4k x 3 datasets):

```bash
for row in $(seq 1 9); do
  python reproduce.py fit-tokenizer \
    --row "$row" \
    --data-root data/canonical \
    --tokenizer-root tokenizers/fitted
done
```

### 5. Run the front-end comparison (predictive quality and projected energy)

Inspect one row without running it:

```bash
python reproduce.py front-end --row 1 --dry-run
```

Run one row:

```bash
python reproduce.py front-end \
  --row 1 \
  --data-root data/canonical \
  --tokenizer-root tokenizers/fitted \
  --output-root outputs/front_end
```

Run the full 90-row factorial (10 front-ends x 3 datasets x 3 seeds; this
trains a small model per row and was originally run as 90 independent
one-CPU-thread HPC tasks, so budget CPU time accordingly and prefer
distributing rows across a scheduler over a serial loop):

```bash
for row in $(seq 1 90); do
  python reproduce.py front-end \
    --row "$row" \
    --data-root data/canonical \
    --tokenizer-root tokenizers/fitted \
    --output-root outputs/front_end
done
```

### 6. Run the temporal-decoder (TTFS) comparison (paired output-decoder results)

```bash
python reproduce.py temporal-decoder --row 1 --dry-run

for row in $(seq 1 162); do
  python reproduce.py temporal-decoder \
    --row "$row" \
    --data-root data/canonical \
    --output-root outputs/temporal_decoders
done
```

162 rows = 3 patch architectures x 3 output heads (dense reference, direct
TTFS, nibble TTFS) x 2 decoder capacities (matched, enlarged) x 3 datasets x
3 seeds.

### 7. Regenerate the paper's figures

No new experiment runs required; this reads directly from the committed raw
CSVs (`results/front_end/all_rows.csv`, `results/temporal_decoders/all_rows.csv`):

```bash
python scripts/make_figures.py --output-dir outputs/figures
```

This writes six PNG files to the output directory. The filenames are
generated by this script for its own bookkeeping and do not track the
manuscript's own figure numbers (which move between drafts); match figures to
the manuscript by content, not by filename:

- front-end predictive quality across methods and datasets
- predictive quality versus temporal compression (bytes per global step)
- predictive quality versus projected energy (the quality-energy frontier)
- TTFS output-head operating points (quality versus energy)
- TTFS output-head paired quality/energy trade-offs versus the dense readout
- a supplementary sensitivity chart, not in the manuscript, showing how the
  headline energy-reduction percentages move under the low/central/high
  energy-coefficient scenarios described in the Methods (the manuscript's own
  tables and figures only ever plot the central scenario)

### 8. Compare your own runs against the reported results

Your freshly generated rows land in `outputs/front_end` and
`outputs/temporal_decoders`, separate from the committed `results/` archives
(which are frozen and never overwritten). Compare a run's `test_bpb` /
`bits_per_raw_byte` and energy columns against the matching row in
`results/front_end/all_rows.csv` or `results/temporal_decoders/all_rows.csv`
to check reproduction. Exact bit-for-bit reproduction of every decimal is not
expected across hardware/BLAS versions; large deviations are the signal to
look for.

Rows are one-indexed and map directly to the frozen CSV grids in `configs/`.

## Reproducibility boundary

The experiments were executed on the Bunya HPC system with Python 3.10.20,
NumPy 2.2.6, SentencePiece 0.2.1, snnTorch 0.9.4, and PyTorch
2.11.0+cu126. The models themselves were explicitly executed on CPU with one
Torch, OpenMP, and MKL thread per task. The compact portable environment is in
`environment.yml`; the path-sanitised Linux-64 Bunya export is retained in
`environment-lock.yml`.

The source commits, result commits, environment evidence, expected row counts,
and archived hashes are listed in
[docs/PROVENANCE.md](docs/PROVENANCE.md). Descriptive public filenames are
mapped to their original development paths in
[docs/SOURCE_MAPPING.md](docs/SOURCE_MAPPING.md). The integrity validator
checks those declarations without calculating or interpreting paper findings.

## Licence

This repository is released under the [MIT Licence](LICENSE). The three
source corpora (TinyStories, Text8, WikiText-2) are not redistributed here
and remain governed by their own original licences (see
[docs/DATASETS.md](docs/DATASETS.md)).

## Access and contributions

This repository is currently private, shared with reviewers on request. It
will be made public, with GitHub granting read/clone/fork access to the
public while direct push access remains limited to the authors, as part of
the paper's release. This project is not accepting direct external
contributions; independent work should use a fork or separate clone.

## Citation

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). The
preferred-citation entry will be updated with the final publication venue and
DOI once the paper is accepted.
