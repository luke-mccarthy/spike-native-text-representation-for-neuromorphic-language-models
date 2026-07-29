from scripts.validate_release import (
    validate_front_end,
    validate_hashes,
    validate_source_manifest,
    validate_temporal_decoders,
)


def test_archived_hashes() -> None:
    validate_hashes()


def test_front_end_archive() -> None:
    validate_front_end()


def test_source_manifest() -> None:
    validate_source_manifest()


def test_temporal_decoder_archive() -> None:
    validate_temporal_decoders()
