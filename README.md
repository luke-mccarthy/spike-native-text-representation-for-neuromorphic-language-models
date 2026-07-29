# Toward Spike-Native Text Representation

Reproduction code, frozen configurations, and raw result records for the paper
*Toward Spike-Native Text Representation for Spiking Language Models*.

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
scripts/                  Release-integrity utilities
paper_code/               Curated paper runtime with descriptive public names
provenance/               Unmodified originals of portability-patched files
tests/                    Release-integrity tests
environment.yml           Portable pinned runtime environment
environment-lock.yml      Archived full environment export
reproduce.py              Friendly single-row reproduction interface
```

The files under `results/` are raw experimental records. This repository does
not provide a narrative interpretation of those values.

## Quick start

Create the recorded Conda environment:

```bash
conda env create -f environment.yml
conda activate spike-native-text
python reproduce.py validate
```

Prepare the canonical data layout from local source files:

```bash
python reproduce.py prepare-data \
  --text8-full /path/to/text8 \
  --tinystories-5000 /path/to/tinystories_5000.txt \
  --wikitext2-train /path/to/wiki.train.tokens \
  --wikitext2-validation /path/to/wiki.valid.tokens \
  --wikitext2-test /path/to/wiki.test.tokens \
  --output-root data/canonical
```

Fit one of the nine dataset-specific conventional tokenizers:

```bash
python reproduce.py fit-tokenizer \
  --row 1 \
  --data-root data/canonical \
  --tokenizer-root tokenizers/fitted
```

The generated dataset `manifest.json` must match the hashes recorded in
`docs/canonical_file_manifest.json`. See [docs/DATASETS.md](docs/DATASETS.md)
before running an experiment.

Print, without executing, the exact command for one front-end row:

```bash
python reproduce.py front-end --row 1 --dry-run
```

Run that row after preparing data and fitted SentencePiece models:

```bash
python reproduce.py front-end \
  --row 1 \
  --data-root data/canonical \
  --tokenizer-root tokenizers/fitted \
  --output-root outputs/front_end
```

Print or run one temporal-decoder row:

```bash
python reproduce.py temporal-decoder --row 1 --dry-run

python reproduce.py temporal-decoder \
  --row 1 \
  --data-root data/canonical \
  --output-root outputs/temporal_decoders
```

Rows are one-indexed and map directly to the frozen CSV grids. Full factorials
can be distributed with a scheduler by assigning one grid row to each task.
The original runs used one CPU thread per task.

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

## Access and contributions

This repository is currently a private review artifact. When made public,
GitHub grants read/clone/fork access to the public but direct push access only
to explicitly authorised collaborators. This project is not accepting direct
external contributions; independent work should use a fork or separate clone.

No software licence has yet been selected. A licence should be chosen before
public release.

## Citation

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). Update the
paper DOI and publication metadata there after acceptance.
