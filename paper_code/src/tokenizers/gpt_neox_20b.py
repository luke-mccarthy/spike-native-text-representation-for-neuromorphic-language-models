"""Dependency-free encoder for SpikeGPT's vendored GPT-NeoX 20B tokenizer.

The official SpikeGPT repository vendors a Hugging Face Tokenizers JSON file
whose model is byte-level BPE with NFC normalization.  Bunya's frozen runtime
does not require the optional ``tokenizers`` package for the research pipeline,
so this module implements only the declared features present in that file.

The implementation is deliberately fail-closed: unsupported tokenizer JSON
features, normalization that would change the canonical raw bytes, missing
merge products, or a failed byte round-trip all raise rather than silently
changing the evaluated corpus.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import unicodedata
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


OFFICIAL_SPIKEGPT_TOKENIZER_SOURCE = (
    "https://github.com/ridgerchu/SpikeGPT/blob/"
    "9a0dab0c925f4188aa70fc48dfc8bd9846ecd5ef/20B_tokenizer.json"
)
OFFICIAL_SPIKEGPT_TOKENIZER_BLOB_SHA = (
    "9a0dab0c925f4188aa70fc48dfc8bd9846ecd5ef"
)


def _bytes_to_unicode() -> Tuple[Dict[int, str], Dict[str, int]]:
    """Return the reversible GPT-2/NeoX byte-to-Unicode alphabet."""

    byte_values = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    codepoints = list(byte_values)
    extra = 0
    for value in range(256):
        if value not in byte_values:
            byte_values.append(value)
            codepoints.append(256 + extra)
            extra += 1
    encoder = {
        value: chr(codepoint)
        for value, codepoint in zip(byte_values, codepoints)
    }
    return encoder, {symbol: value for value, symbol in encoder.items()}


def _unicode_class(character: str) -> str:
    category = unicodedata.category(character)
    if category.startswith("L"):
        return "letter"
    if category.startswith("N"):
        return "number"
    if character.isspace():
        return "space"
    return "other"


def _gpt2_pretokenize(text: str) -> List[str]:
    """Match GPT-2's Unicode-aware pre-tokenization expression.

    This is the standard expression used by the ByteLevel pre-tokenizer:
    contractions, optional ASCII-space-prefixed letter/number/punctuation
    groups, and whitespace runs.  ``unicodedata`` supplies the Unicode L/N
    classes so no third-party regex package is needed.
    """

    contractions = ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d")
    pieces: List[str] = []
    index = 0
    while index < len(text):
        contraction = next(
            (value for value in contractions if text.startswith(value, index)),
            None,
        )
        if contraction is not None:
            pieces.append(contraction)
            index += len(contraction)
            continue

        start = index
        prefix_space = text[index] == " " and index + 1 < len(text)
        if prefix_space and _unicode_class(text[index + 1]) != "space":
            index += 1
            group = _unicode_class(text[index])
            while index < len(text) and _unicode_class(text[index]) == group:
                index += 1
            pieces.append(text[start:index])
            continue

        group = _unicode_class(text[index])
        if group != "space":
            index += 1
            while index < len(text) and _unicode_class(text[index]) == group:
                index += 1
            pieces.append(text[start:index])
            continue

        end = index + 1
        while end < len(text) and _unicode_class(text[end]) == "space":
            end += 1
        if end < len(text) and end - index > 1:
            # ``\s+(?!\S)`` backtracks one whitespace before non-whitespace.
            end -= 1
        pieces.append(text[index:end])
        index = end
    if "".join(pieces) != text:
        raise AssertionError("ByteLevel pre-tokenizer did not preserve text")
    return pieces


@dataclass(frozen=True)
class NeoXEncoding:
    ids: Tuple[int, ...]
    token_bytes: Tuple[bytes, ...]
    statistics: Mapping[str, int]


class OfficialSpikeGPT20BTokenizer:
    """Exact byte-level BPE model declared by SpikeGPT's tokenizer JSON."""

    def __init__(self, configuration: Mapping[str, object], *, sha256: str):
        if configuration.get("version") != "1.0":
            raise ValueError("Unsupported tokenizer JSON version")
        if configuration.get("truncation") is not None:
            raise ValueError("Tokenizer truncation must be disabled")
        if configuration.get("padding") is not None:
            raise ValueError("Tokenizer padding must be disabled")
        if configuration.get("normalizer") != {"type": "NFC"}:
            raise ValueError("Expected the official NFC normalizer")

        pre = configuration.get("pre_tokenizer")
        decoder = configuration.get("decoder")
        post = configuration.get("post_processor")
        if not isinstance(pre, Mapping) or pre.get("type") != "ByteLevel":
            raise ValueError("Expected a ByteLevel pre-tokenizer")
        if not isinstance(decoder, Mapping) or decoder.get("type") != "ByteLevel":
            raise ValueError("Expected a ByteLevel decoder")
        if not isinstance(post, Mapping) or post.get("type") != "ByteLevel":
            raise ValueError("Expected a ByteLevel post-processor")
        if bool(pre.get("add_prefix_space", False)):
            raise ValueError("Official tokenizer must not add a prefix space")
        if bool(post.get("add_prefix_space", False)):
            raise ValueError("Official post-processor must not add a prefix space")

        model = configuration.get("model")
        if not isinstance(model, Mapping) or model.get("type") != "BPE":
            raise ValueError("Expected a BPE tokenizer model")
        if model.get("dropout") is not None or model.get("unk_token") is not None:
            raise ValueError("Official BPE must have no dropout or unknown token")
        if model.get("continuing_subword_prefix") not in {None, ""}:
            raise ValueError("Unsupported continuing-subword prefix")
        if model.get("end_of_word_suffix") not in {None, ""}:
            raise ValueError("Unsupported end-of-word suffix")

        raw_vocab = model.get("vocab")
        raw_merges = model.get("merges")
        if not isinstance(raw_vocab, Mapping) or not isinstance(raw_merges, list):
            raise ValueError("Malformed BPE vocabulary or merges")
        self.vocab = {str(token): int(identifier) for token, identifier in raw_vocab.items()}
        if len(set(self.vocab.values())) != len(self.vocab):
            raise ValueError("Tokenizer vocabulary IDs are not unique")
        self.merge_ranks: Dict[Tuple[str, str], int] = {}
        for rank, merge in enumerate(raw_merges):
            left, separator, right = str(merge).partition(" ")
            if not separator or not left or not right:
                raise ValueError(f"Malformed BPE merge: {merge!r}")
            self.merge_ranks[(left, right)] = rank

        added_tokens = configuration.get("added_tokens", [])
        if not isinstance(added_tokens, list):
            raise ValueError("Malformed added-token table")
        parsed_added: List[Tuple[str, int]] = []
        for record in added_tokens:
            if not isinstance(record, Mapping):
                raise ValueError("Malformed added-token record")
            if record.get("lstrip") or record.get("rstrip") or record.get("single_word"):
                raise ValueError("Unsupported contextual added-token rule")
            parsed_added.append((str(record["content"]), int(record["id"])))
        self.added_tokens = tuple(sorted(parsed_added, key=lambda item: (-len(item[0]), item[1])))

        all_ids = [*self.vocab.values(), *(identifier for _, identifier in self.added_tokens)]
        self.vocab_size = max(all_ids) + 1
        if set(all_ids) != set(range(self.vocab_size)):
            raise ValueError("Official tokenizer IDs must form a dense range")
        self.sha256 = sha256
        self.byte_encoder, self.byte_decoder = _bytes_to_unicode()
        self._cache: Dict[str, Tuple[Tuple[str, ...], int, int]] = {}

    @classmethod
    def from_path(cls, path: Path) -> "OfficialSpikeGPT20BTokenizer":
        payload = path.read_bytes()
        configuration = json.loads(payload.decode("utf-8"))
        return cls(configuration, sha256=hashlib.sha256(payload).hexdigest())

    def _bpe(self, bytelevel_word: str) -> Tuple[Tuple[str, ...], int, int]:
        cached = self._cache.get(bytelevel_word)
        if cached is not None:
            return cached
        word = tuple(bytelevel_word)
        pair_evaluations = 0
        merge_applications = 0
        while len(word) > 1:
            pairs = tuple(zip(word, word[1:]))
            pair_evaluations += len(pairs)
            ranked = [
                (self.merge_ranks[pair], pair)
                for pair in pairs
                if pair in self.merge_ranks
            ]
            if not ranked:
                break
            _, selected = min(ranked, key=lambda item: item[0])
            merged: List[str] = []
            index = 0
            while index < len(word):
                if index + 1 < len(word) and (word[index], word[index + 1]) == selected:
                    merged.append(word[index] + word[index + 1])
                    merge_applications += 1
                    index += 2
                else:
                    merged.append(word[index])
                    index += 1
            word = tuple(merged)
        if any(token not in self.vocab for token in word):
            missing = next(token for token in word if token not in self.vocab)
            raise AssertionError(f"BPE produced token absent from vocabulary: {missing!r}")
        result = (word, pair_evaluations, merge_applications)
        self._cache[bytelevel_word] = result
        return result

    def encode(self, text: str) -> NeoXEncoding:
        normalized = unicodedata.normalize("NFC", text)
        if normalized.encode("utf-8", errors="strict") != text.encode("utf-8", errors="strict"):
            raise ValueError(
                "Official tokenizer NFC normalization would change canonical raw bytes"
            )

        ids: List[int] = []
        payloads: List[bytes] = []
        pretoken_count = 0
        pair_evaluations = 0
        merge_applications = 0
        added_token_matches = 0

        def encode_plain(span: str) -> None:
            nonlocal pretoken_count, pair_evaluations, merge_applications
            for pretoken in _gpt2_pretokenize(span):
                pretoken_count += 1
                bytelevel = "".join(
                    self.byte_encoder[value]
                    for value in pretoken.encode("utf-8", errors="strict")
                )
                tokens, pair_count, merge_count = self._bpe(bytelevel)
                pair_evaluations += pair_count
                merge_applications += merge_count
                for token in tokens:
                    ids.append(self.vocab[token])
                    payloads.append(bytes(self.byte_decoder[symbol] for symbol in token))

        plain_start = 0
        index = 0
        while index < len(normalized):
            match = next(
                (
                    (content, identifier)
                    for content, identifier in self.added_tokens
                    if normalized.startswith(content, index)
                ),
                None,
            )
            if match is None:
                index += 1
                continue
            if plain_start < index:
                encode_plain(normalized[plain_start:index])
            content, identifier = match
            ids.append(identifier)
            payloads.append(content.encode("utf-8", errors="strict"))
            added_token_matches += 1
            index += len(content)
            plain_start = index
        if plain_start < len(normalized):
            encode_plain(normalized[plain_start:])

        raw = normalized.encode("utf-8", errors="strict")
        if b"".join(payloads) != raw:
            raise AssertionError("Official tokenizer failed exact raw-byte round-trip")
        return NeoXEncoding(
            ids=tuple(ids),
            token_bytes=tuple(payloads),
            statistics={
                "input_unicode_codepoints": len(normalized),
                "input_raw_bytes": len(raw),
                "pretoken_count": pretoken_count,
                "bpe_pair_rank_evaluations_uncached_logical": pair_evaluations,
                "bpe_merge_applications_uncached_logical": merge_applications,
                "added_token_matches": added_token_matches,
                "output_token_count": len(ids),
            },
        )
