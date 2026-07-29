"""Phase-separated frontend storage and compulsory-memory lower bounds.

These counts are architecture-neutral byte ledgers, not energy estimates. They
make explicit which data structures must exist or cross a frontend boundary at
least once. Cache refills, allocator traffic, Python objects, routing, static
power and hardware-specific SRAM/DRAM placement are not inferred here.
"""

from __future__ import annotations

import math
from typing import Dict, Mapping


LEDGER_VERSION = "frontend_compulsory_memory_lower_bound_v1"


def _integer(row: Mapping[str, object], key: str, default: int = 0) -> int:
    value = row.get(key, default)
    if value in {"", None, "not_applicable", "not_measured"}:
        return int(default)
    result = int(float(value))
    if result < 0:
        raise ValueError(f"{key} must be non-negative")
    return result


def _phase_memory(
    row: Mapping[str, object],
    *,
    phase: str,
    raw_bytes: int,
    modelling_units: int,
) -> Dict[str, int | str | float]:
    if phase not in {"training", "validation"}:
        raise ValueError("phase must be training or validation")
    if raw_bytes <= 0 or modelling_units <= 0:
        raise ValueError("raw-byte and modelling-unit denominators must be positive")
    dimension = _integer(row, "interface_dimension")
    if dimension <= 0:
        raise ValueError("interface dimension must be positive")
    token_ids = modelling_units * 4
    raw_length_metadata = modelling_units * 8
    packed_spikes = math.ceil(modelling_units * dimension / 8)
    reference_fp32_spikes = modelling_units * dimension * 4
    tokenizer_artifact = _integer(
        row,
        "training_sequence_tokenizer_model_file_bytes",
        _integer(row, "tokenizer_model_file_bytes_current_run"),
    )
    codec_payload = _integer(row, "patch_codec_payload_storage_bytes")
    interface_parameters = _integer(row, "interface_params") * 4
    compulsory = (
        raw_bytes
        + token_ids
        + raw_length_metadata
        + packed_spikes
        + tokenizer_artifact
        + codec_payload
        + interface_parameters
    )
    prefix = f"{phase}_frontend_memory_"
    return {
        f"{prefix}raw_corpus_bytes_read_lower_bound": raw_bytes,
        f"{prefix}unit_id_bytes_written_int32": token_ids,
        f"{prefix}raw_length_metadata_bytes_written_int32": raw_length_metadata,
        f"{prefix}binary_interface_bytes_packed_lower_bound": packed_spikes,
        f"{prefix}binary_interface_bytes_reference_fp32": reference_fp32_spikes,
        f"{prefix}tokenizer_model_artifact_bytes_once": tokenizer_artifact,
        f"{prefix}patch_codec_payload_bytes_once": codec_payload,
        f"{prefix}interface_parameter_bytes_fp32_once": interface_parameters,
        f"{prefix}compulsory_bytes_lower_bound": compulsory,
        f"{prefix}compulsory_bytes_lower_bound_per_raw_byte": (
            compulsory / raw_bytes
        ),
        f"{prefix}memory_access_ledger_complete": 0,
        f"{prefix}cache_routing_static_and_allocator_traffic_included": 0,
    }


def experiment_frontend_memory_ledger(
    row: Mapping[str, object],
    *,
    training_raw_bytes: int,
    validation_raw_bytes: int,
) -> Dict[str, int | str | float]:
    """Return storage facts and phase-separated compulsory byte lower bounds."""

    training_units = _integer(row, "training_num_units")
    validation_units = _integer(row, "validation_num_units")
    result: Dict[str, int | str | float] = {
        "frontend_memory_ledger_version": LEDGER_VERSION,
        "frontend_memory_energy_estimate_ready": 0,
        "frontend_memory_ledger_is_compulsory_lower_bound": 1,
        "frontend_memory_missing_work_never_encoded_as_zero_energy": 1,
        "frontend_active_parameter_storage_bytes_fp32": _integer(
            row, "active_parameter_storage_bytes_fp32"
        ),
        "frontend_tokenizer_model_storage_bytes": _integer(
            row,
            "training_sequence_tokenizer_model_file_bytes",
            _integer(row, "tokenizer_model_file_bytes_current_run"),
        ),
        "frontend_patch_codec_payload_storage_bytes": _integer(
            row, "patch_codec_payload_storage_bytes"
        ),
    }
    result.update(_phase_memory(
        row,
        phase="training",
        raw_bytes=int(training_raw_bytes),
        modelling_units=training_units,
    ))
    result.update(_phase_memory(
        row,
        phase="validation",
        raw_bytes=int(validation_raw_bytes),
        modelling_units=validation_units,
    ))
    result.update({
        "frontend_operation_and_memory_ledger_complete": 0,
        "frontend_system_energy_estimate_ready": 0,
        "frontend_hardware_energy_claim_permitted": 0,
    })
    return result
