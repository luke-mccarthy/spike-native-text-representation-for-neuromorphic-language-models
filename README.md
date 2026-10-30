# Spike-Native Text Representation for Neuromorphic Language Models

Code, experiment grids and raw results for the paper *Spike-Native Text
Representation for Neuromorphic Language Models*. The repository contains:

- the model, front-end, training, evaluation and energy-accounting code used
  by the reported experiments;
- the experiment grids that define every reported run;
- the row-level CSV outputs and audit records of those runs;
- the dataset-split builder and the SHA-256 hashes of every split file;
- a pinned software environment, an integrity validator and the figure script.

## Repository layout

```text
configs/                  Experiment grids (one row per run)
docs/                     Dataset, environment and provenance documentation
results/
  front_end/              Front-end comparison: 90 rows
  tokenizer_fits/         SentencePiece fits: 9 rows
  temporal_decoders/      Output-head comparison: 162 rows, as recorded
  temporal_decoders_corrected/
                          Output-head comparison with corrected decoder
                          input accumulations and energy: 162 rows
scripts/                  Integrity validator, TinyStories tools, decoder
                          accounting correction, figure script
paper_code/               Model, training, evaluation and accounting code
provenance/               Original copies of the files changed after the runs
tests/                    Integrity tests
environment.yml           Pinned software environment
environment-lock.yml      Full export of the original HPC environment (Linux-64)
reproduce.py              Command-line entry point
```

## Code map

The pipeline runs raw bytes → front-end units or spikes → SpikeGPT-style
global core → local or terminal decoder → p(byte). Original development paths
for every file are listed in [docs/SOURCE_MAPPING.md](docs/SOURCE_MAPPING.md).

| Component | File(s) |
|---|---|
| SpikeGPT-style global core (RWKV time and channel mixing, WKV recurrence, LIF neurons, arctangent surrogate) | `paper_code/src/models/spikegpt_reference.py` |
| Subword front-ends (BPE/Unigram token ID → learned binary embedding) | `paper_code/src/models/spike_input_interfaces.py`, `paper_code/scripts/fit_sentencepiece_tokenizer.py` |
| Spiking-boundary/digital-codec hybrid front-end | `paper_code/src/tokenizers/spike_boundary_native_levels.py`, `paper_code/src/tokenizers/spike_boundary_refined.py` |
| Byte spike codes (1-of-256, 2-of-32, recurrent-LIF state with injective suffix) | `paper_code/src/models/spike_input_interfaces.py` |
| Injective constant-weight codes used by the byte and patch front-ends | `paper_code/src/tokenizers/lossless_codecs.py` |
| Patch front-ends (fixed 4- and 8-byte, adaptive 8-byte with homeostatic boundary) and local LIF decoder with dense 256-way readout | `paper_code/src/models/spike_native_patch_lm.py`, `paper_code/src/models/local_spiking_patch_decoder.py`, `paper_code/src/models/spike_count_patch_decoder.py`, `paper_code/src/tokenizers/spike_native_patch_frontends.py` |
| TTFS output heads (direct 256-neuron race, 2 × 16-neuron nibble race) | `paper_code/src/models/event_accounted_wta_ttfs_decoder.py` (`EventAccountedWTATTFSPatchLanguageModel`), which subclasses `continuous_wta_ttfs_decoder.py`, `audited_ttfs_patch_decoder.py` and `strict_ttfs_patch_decoder.py` in the same folder; `paper_code/scripts/run_temporal_decoder_comparison.py` substitutes it for the model class in `train_ttfs_decoder.py` |
| Operation counts and energy projection (dense MACs, synaptic operations, state updates, routed spikes; low, central and high coefficients) | `paper_code/src/energy/analytical_compute.py`, `frontend_energy_projection.py`, `frontend_memory.py`, `scalable_patch_projection.py`, `wta_event_energy_projection.py`, `paper_code/src/models/spikegpt_operation_accounting.py` |
| Dataset splitting and split-hash checks | `paper_code/scripts/prepare_canonical_splits.py`, `paper_code/src/canonical_data_contract.py` |
| Decoder input-accumulation correction applied to the output-head archive | `scripts/correct_decoder_fanout.py` |
| Per-run entry points | `paper_code/scripts/train_frontend_language_model.py` (subword, hybrid and byte runs), `paper_code/scripts/train_patch_language_model.py` (patch runs), `paper_code/scripts/run_temporal_decoder_comparison.py` with `train_ttfs_decoder.py` (output-head runs) |

`paper_code/src/tokenizers/blt_patcher.py`, `spiking_blt_frontend.py` and
`gpt_neox_20b.py` are imported by the front-end code; the reported runs do not
use them.

## Reproducing the results

### 1. Install the environment

```bash
conda env create -f environment.yml
conda activate spike-native-text
python reproduce.py validate
```

`validate` checks the experiment grids, raw result archives, audit records and
code checksums. It needs no datasets.

### 2. Obtain the three corpora

The datasets are not redistributed here. Formats and split rules are in
[docs/DATASETS.md](docs/DATASETS.md). Three inputs are needed:

- **Text8:** the 100,000,000-byte `text8` file.
- **WikiText-2:** the official `wiki.train.tokens`, `wiki.valid.tokens` and
  `wiki.test.tokens` files.
- **TinyStories-5000:** the first 5,000 stories of the `roneneldan/TinyStories`
  training split, built with:

  ```bash
  pip install datasets
  python scripts/fetch_tinystories_5000.py --output data/sources/tinystories_5000.txt
  ```

  The script compares its output with the SHA-256 of the file used in the
  paper and prints PASS or FAIL.

### 3. Build the dataset splits

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

Once `data/canonical/manifest.json` exists, `validate` also compares the byte
count and SHA-256 of every split file with
`docs/canonical_file_manifest.json` and stops with an error on any mismatch.

### 4. Fit the SentencePiece tokenisers

Nine rows: BPE-8k, BPE-16k and Unigram-4k for each of the three datasets.

```bash
for row in $(seq 1 9); do
  python reproduce.py fit-tokenizer \
    --row "$row" \
    --data-root data/canonical \
    --tokenizer-root tokenizers/fitted
done
```

### 5. Run the front-end comparison

90 rows: 10 front-ends × 3 datasets × 3 seeds. Show the command for one row
without running it:

```bash
python reproduce.py front-end --row 1 --dry-run
```

Run all rows:

```bash
for row in $(seq 1 90); do
  python reproduce.py front-end \
    --row "$row" \
    --data-root data/canonical \
    --tokenizer-root tokenizers/fitted \
    --output-root outputs/front_end
done
```

Each row trains one model on one CPU thread. The reported rows were run as 90
independent single-thread tasks on an HPC scheduler.

### 6. Run the output-head comparison

162 rows: 3 patch architectures × 3 output heads (dense, direct TTFS, nibble
TTFS) × 2 decoder capacities × 3 datasets × 3 seeds.

```bash
python reproduce.py temporal-decoder --row 1 --dry-run

for row in $(seq 1 162); do
  python reproduce.py temporal-decoder \
    --row "$row" \
    --data-root data/canonical \
    --output-root outputs/temporal_decoders
done
```

Rows are one-indexed and correspond to the rows of the CSV grids in
`configs/`.

### 7. Apply the decoder accounting correction

The recorded output-head rows counted layer-0 decoder input accumulations
with the mean fan-in per destination neuron instead of the fan-out of 8 per
active input. The counters in `paper_code/` now count the fan-out, so new
runs produce corrected values directly. For the recorded rows:

```bash
python reproduce.py validate          # checks results/temporal_decoders_corrected/
python scripts/correct_decoder_fanout.py   # rebuilds it from results/temporal_decoders/
```

The script changes only the decoder input-accumulation fields and the energy,
traffic and synaptic-operation fields derived from them; BPB is unchanged.
`correction_audit.csv` lists the recorded and corrected count for every row
and phase. Details are in [docs/PROVENANCE.md](docs/PROVENANCE.md).

### 8. Generate the figures

The figure script reads the committed CSVs in `results/` (the output-head
figure uses `results/temporal_decoders_corrected/`), so it needs no new
runs:

```bash
pip install matplotlib==3.11.2
python scripts/make_figures.py --output-dir outputs/figures
```

It writes three PDFs:

| File | Content |
|---|---|
| `rate_quality_tradeoff.pdf` | Test BPB against raw bytes per global recurrent step, per dataset |
| `quality_energy_frontier.pdf` | Test BPB against projected central training energy per raw byte, per dataset |
| `ttfs_paired_tradeoffs.pdf` | Paired change in test BPB and projected energy for the TTFS heads against the dense head, per dataset |

### 9. Compare new runs with the reported results

New rows are written to `outputs/front_end` and `outputs/temporal_decoders`;
the files in `results/` are left untouched. Match each new row to the row with
the same `run_name` in `results/front_end/all_rows.csv` or
`results/temporal_decoders_corrected/all_rows.csv` and compare `test_bpb` and
the energy columns. Floating-point results can differ in the last decimal places across
CPU types and BLAS libraries; timing fields depend on the host.

## Software environment

The reported runs used Python 3.10.20, NumPy 2.2.6, SentencePiece 0.2.1,
snnTorch 0.9.4 and PyTorch 2.11.0+cu126 on the Bunya HPC system, with all
models on CPU and one Torch, OpenMP and MKL thread per task. `environment.yml`
pins these versions; `environment-lock.yml` is the full Linux-64 export of the
original environment. Details are in
[docs/EXECUTION_ENVIRONMENT.md](docs/EXECUTION_ENVIRONMENT.md); source
commits, result commits and archive hashes are in
[docs/PROVENANCE.md](docs/PROVENANCE.md).

## Licence

MIT; see [LICENSE](LICENSE). TinyStories, Text8 and WikiText-2 are not
redistributed and remain under their original licences.

## Citation

Citation metadata is in [CITATION.cff](CITATION.cff).
