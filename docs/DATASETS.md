# Canonical datasets

The repository does not redistribute dataset payloads. Reproduction requires
local copies of the three source corpora and constructs the paper splits before
any tokenizer, boundary rule, threshold, vocabulary, or language model is
fitted.

## Required source files

| Dataset | Required input | Split treatment |
|---|---|---|
| Text8 | The complete 100,000,000-byte `text8` corpus | Contiguous 90,000,000 / 5,000,000 / 5,000,000-byte train/validation/test split |
| WikiText-2 | Official train, validation, and test text files | Official split files are retained; development views are UTF-8-safe prefixes |
| TinyStories-5000 | The exact 5,000-story source used by the study | Whole-document SHA-256 deduplication followed by deterministic 80/10/10 assignment with seed `20260716` |

The exact construction is implemented in the commit-pinned paper code and is
exposed as `python reproduce.py prepare-data`. Existing outputs are never
silently overwritten: a rerun verifies identical bytes or fails.

## Expected layout

```text
data/canonical/
  manifest.json
  text8/
    train.txt
    validation.txt
    test.txt
  tinystories_5000/
    train.txt
    validation.txt
    test.txt
  wikitext2/
    train.txt
    validation.txt
    test.txt
```

The split builder also creates the bounded development views used during
method development. The reported paper test grids use the full
`full_train_validation` tier.

## Identity checks

`docs/canonical_file_manifest.json` records the relative path, role, byte
count, and SHA-256 value for every full train/validation/test file accessed by
the paper experiments. The training scripts call the fail-closed checks in
`src/canonical_data_contract.py`; execution stops if a path, role, size, or
hash differs.

The TinyStories source is a study-specific 5,000-story file. A differently
sampled 5,000-story subset will not reproduce the recorded manifest and should
not be silently substituted.

**TinyStories-5000 source format.** The file passed to `--tinystories-5000`
must contain exactly 5,000 documents separated by the literal marker
`<|endoftext|>` on its own line, with no blank documents.
`prepare_canonical_splits.py` asserts the document count is exactly 5,000 and
fails otherwise.

**Reconstructing the source file.** `scripts/fetch_tinystories_5000.py`
rebuilds this file: it takes the first 5,000 examples of the
[`roneneldan/TinyStories`](https://huggingface.co/datasets/roneneldan/TinyStories)
train split, in the dataset's own unshuffled order, and joins them with the
`<|endoftext|>` marker.

```bash
pip install datasets  # not part of the pinned environment; only needed once
python scripts/fetch_tinystories_5000.py --output data/sources/tinystories_5000.txt
```

**This procedure is confirmed byte-for-byte identical to the original
source file.** The original `tinystories_5000.txt` still exists on the
authors' Bunya HPC storage and is recorded in the 2026-07-16 canonical
manifest. The author independently re-ran the procedure above against the
live `roneneldan/TinyStories` dataset and diffed the result against that
surviving original file. Both are:

- **Bytes:** 4,262,315
- **SHA-256:** `afff3659756a01acd8c95ed8b16c67b0ad829cd8e3d7fc6161a1bc9982c04fb4`

`scripts/fetch_tinystories_5000.py` checks its own output against this exact
hash and prints PASS/FAIL immediately, so you can confirm your download
matches before doing anything else with it. This hash could stop matching in
the future only if the upstream `roneneldan/TinyStories` dataset itself is
ever revised; if that happens, this file (and the original) would need to be
substituted for a differently-sourced but pinned copy.

**Checking the downstream splits.** `scripts/verify_tinystories_5000.py`
additionally checks a candidate `tinystories_5000.txt` against the paper's
recorded *post-split* hashes (train/validation/test after deduplication and
the 80/10/10 assignment), without needing the Text8 or WikiText-2 source
files:

```bash
python scripts/verify_tinystories_5000.py --source data/sources/tinystories_5000.txt
```

(Once all three source corpora are available, `python reproduce.py
prepare-data ...` followed by `python reproduce.py validate` performs the
same check as part of building the complete canonical dataset.)

## SentencePiece artifacts

The BPE-8k, BPE-16k, and Unigram-4k models are fitted independently for each
dataset on the exact 1,000,000-byte UTF-8-complete training selection used by
the corresponding language-model rows. Fitted binaries are not stored here.
The expected location is:

```text
tokenizers/fitted/models/<dataset>_<method>.model
tokenizers/fitted/models/<dataset>_<method>.train.json
tokenizers/fitted/parts/<dataset>_<method>.csv
```

Use `python reproduce.py fit-tokenizer --row N` with rows 1–9 of
`configs/tokenizer_fits.csv`. The training provenance JSON and model hash are
checked before a conventional tokenizer row can run.
