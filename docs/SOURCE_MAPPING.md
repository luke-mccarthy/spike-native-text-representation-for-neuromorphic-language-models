# Public source names and development provenance

The public runtime uses descriptive filenames. Dates and internal iteration
suffixes are retained only in provenance metadata because they describe the
development history rather than a reader-facing function. The exact source
blob and runtime SHA-256 for every included file are recorded in
`source_manifest.json`.

| Public runtime path | Original development path |
|---|---|
| `paper_code/scripts/audit_frontend_representations.py` | `scripts/audit_2026_07_16_canonical_frontend_representations.py` |
| `paper_code/scripts/build_frontend_grids.py` | `scripts/make_2026_07_22_decisive_frontend_grids.py` |
| `paper_code/scripts/fit_sentencepiece_tokenizer.py` | `scripts/measure_2026_07_22_sentencepiece_fit.py` |
| `paper_code/scripts/prepare_canonical_splits.py` | `scripts/prepare_2026_07_16_canonical_splits.py` |
| `paper_code/scripts/run_frontend_row.py` | `scripts/run_2026_07_22_decisive_frontend_row.py` |
| `paper_code/scripts/run_frontend_comparison.py` | `scripts/run_2026_07_23_scalable_patch_frontend_row.py` |
| `paper_code/scripts/train_frontend_language_model.py` | `scripts/train_2026_07_17_canonical_lm.py` |
| `paper_code/scripts/train_patch_language_model.py` | `scripts/train_2026_07_23_scalable_patch_lm.py` |
| `paper_code/scripts/train_spike_count_decoder.py` | `scripts/train_2026_07_28_spike_output_decoder_lm.py` |
| `paper_code/scripts/train_ttfs_decoder.py` | `scripts/train_2026_07_28_strict_ttfs_output_lm.py` |
| `paper_code/scripts/run_temporal_decoder_comparison.py` | `scripts/train_2026_07_28_strict_ttfs_output_lm_r4.py` |
| `paper_code/src/energy/frontend_energy_projection.py` | `src/energy/decisive_projection.py` |
| `paper_code/src/energy/wta_event_energy_projection.py` | `src/energy/scalable_patch_projection_r4.py` |
| `paper_code/src/models/local_spiking_patch_decoder.py` | `src/models/spike_native_patch_lm_v2.py` |
| `paper_code/src/models/spike_count_patch_decoder.py` | `src/models/spike_native_patch_lm_v3.py` |
| `paper_code/src/models/hierarchical_ttfs_patch_decoder.py` | `src/models/spike_native_patch_lm_v5.py` |
| `paper_code/src/models/strict_ttfs_patch_decoder.py` | `src/models/spike_native_patch_lm_v6.py` |
| `paper_code/src/models/audited_ttfs_patch_decoder.py` | `src/models/spike_native_patch_lm_v7.py` |
| `paper_code/src/models/continuous_wta_ttfs_decoder.py` | `src/models/spike_native_patch_lm_v8.py` |
| `paper_code/src/models/event_accounted_wta_ttfs_decoder.py` | `src/models/spike_native_patch_lm_v9.py` |
| `paper_code/src/models/strict_spiking_backbones.py` | `src/models/strict_spiking_backbones_v2.py` |

The remaining runtime files already had descriptive names and retain their
original paths. Renaming does not alter raw configuration fields, protocol
identifiers, run names, or result records; those values remain byte-for-byte
identical to the frozen evidence.
