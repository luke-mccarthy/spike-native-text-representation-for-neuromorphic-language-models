from scripts.validate_release import (
    validate_accounting_corrections,
    validate_documentation_changes,
    validate_environment_evidence,
    validate_front_end,
    validate_hashes,
    validate_source_manifest,
    validate_temporal_decoders,
    validate_tokenizer_fits,
)


def test_archived_hashes() -> None:
    validate_hashes()


def test_environment_evidence() -> None:
    validate_environment_evidence()


def test_front_end_archive() -> None:
    validate_front_end()


def test_source_manifest() -> None:
    validate_source_manifest()


def test_temporal_decoder_archive() -> None:
    validate_temporal_decoders()


def test_tokenizer_fit_archive() -> None:
    validate_tokenizer_fits()


def test_accounting_corrections() -> None:
    validate_accounting_corrections()


def test_documentation_changes() -> None:
    validate_documentation_changes()
