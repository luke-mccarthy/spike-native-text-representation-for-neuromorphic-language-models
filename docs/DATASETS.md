# Datasets

The three source corpora are not redistributed. `python reproduce.py
prepare-data` builds the splits used by every experiment from local copies,
before any tokeniser, boundary rule, threshold, vocabulary or model is fitted.

## Source files and split rules

| Dataset | Required input | Split |
|---|---|---|
| Text8 | The 100,000,000-byte `text8` file | Contiguous 90,000,000 / 5,000,000 / 5,000,000-byte train/validation/test regions |
| WikiText-2 | Official `wiki.train.tokens`, `wiki.valid.tokens`, `wiki.test.tokens` | Official partitions, unchanged |
| TinyStories-5000 | The 5,000-story file built by `scripts/fetch_tinystories_5000.py` | Whole-document SHA-256 deduplication, then a deterministic 80/10/10 document split with seed `20260716` |

The split code is `paper_code/scripts/prepare_canonical_splits.py`. Rerunning
it over an existing output directory checks that the bytes are identical and
stops with an error if they differ.

## Output layout

```text
data/canonical/
  manifest.json
  text8/            train.txt  validation.txt  test.txt
  tinystories_5000/ train.txt  validation.txt  test.txt
  wikitext2/        train.txt  validation.txt  test.txt
```

The split builder also writes the shorter development views used during
method development. The reported runs use the full splits (data tier
`full_train_validation`).

## Hash checks

`docs/canonical_file_manifest.json` lists the path, role, byte count and
SHA-256 of every train, validation and test file used by the experiments. The
training scripts check each file against this manifest through
`paper_code/src/canonical_data_contract.py` and stop on any difference in
path, role, size or hash.

## TinyStories-5000

The source file contains 5,000 stories separated by the marker
`<|endoftext|>` on its own line, with no empty stories.
`prepare_canonical_splits.py` requires exactly 5,000 documents.

`scripts/fetch_tinystories_5000.py` builds the file from the first 5,000
examples of the
[`roneneldan/TinyStories`](https://huggingface.co/datasets/roneneldan/TinyStories)
training split, in the dataset's own order:

```bash
pip install datasets
python scripts/fetch_tinystories_5000.py --output data/sources/tinystories_5000.txt
```

The output is identical to the file used in the paper:

- **Bytes:** 4,262,315
- **SHA-256:** `afff3659756a01acd8c95ed8b16c67b0ad829cd8e3d7fc6161a1bc9982c04fb4`

The script compares its output with these values and prints PASS or FAIL. A
revision of the upstream dataset would change the hash.

`scripts/verify_tinystories_5000.py` applies the deduplication and 80/10/10
split to a TinyStories source file and compares the three resulting files
with the hashes in `docs/canonical_file_manifest.json`, without needing the
Text8 or WikiText-2 files:

```bash
python scripts/verify_tinystories_5000.py --source data/sources/tinystories_5000.txt
```

## SentencePiece models

BPE-8k, BPE-16k and Unigram-4k are fitted separately for each dataset on the
same 1,000,000-byte UTF-8-complete training selection used by the language
models. The fitted models are not stored in the repository;
`python reproduce.py fit-tokenizer --row N` (rows 1–9 of
`configs/tokenizer_fits.csv`) writes them to:

```text
tokenizers/fitted/models/<dataset>_<method>.model
tokenizers/fitted/models/<dataset>_<method>.train.json
tokenizers/fitted/parts/<dataset>_<method>.csv
```

Subword runs check the fit record and model hash before training.
