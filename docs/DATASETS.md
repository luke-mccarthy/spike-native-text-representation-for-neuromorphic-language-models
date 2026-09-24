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
reconstructs this file: it takes the first 5,000 examples of the
[`roneneldan/TinyStories`](https://huggingface.co/datasets/roneneldan/TinyStories)
train split, in the dataset's own unshuffled order, and joins them with the
`<|endoftext|>` marker.

```bash
pip install datasets  # not part of the pinned environment; only needed once
python scripts/fetch_tinystories_5000.py --output data/sources/tinystories_5000.txt
```

This procedure has been reconstructed from the private development history
(an earlier, differently-formatted prototype script and a later, format-
matching script were both found; the later one's output format and file
naming line up exactly with what `prepare_canonical_splits.py` expects). It
has **not** been confirmed byte-for-byte against the original source file: no
committed job log records the literal command that produced it, and this
repository's sandboxed audit environment could not reach `huggingface.co` to
check the resulting hash directly.

**Checking the reconstruction.** `scripts/verify_tinystories_5000.py` checks
a candidate `tinystories_5000.txt` against the paper's recorded hashes
without needing the Text8 or WikiText-2 source files (it only needs the
TinyStories file, unlike the full `prepare-data` command):

```bash
python scripts/verify_tinystories_5000.py --source data/sources/tinystories_5000.txt
```

It prints PASS/FAIL per split (train/validation/test) against
`docs/canonical_file_manifest.json`. If it fails, this reconstruction is
wrong and should be corrected rather than worked around -- please report it.
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
