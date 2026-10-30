# Execution environment

The reported runs were independent CPU tasks on the Bunya HPC system
(University of Queensland). The experiment code:

- selects `torch.device("cpu")`;
- sets `torch.set_num_threads(1)`;
- uses one OpenMP thread and one MKL thread per task in the output-head runs;
- uses seeds 1, 2 and 3, as listed in the grids.

| Component | Version |
|---|---|
| Python | 3.10.20 |
| NumPy | 2.2.6 |
| PyTorch | 2.11.0+cu126 |
| SentencePiece | 0.2.1 |
| snnTorch | 0.9.4 |
| Torch threads | 1 |
| OpenMP threads | 1 |
| MKL threads | 1 |

The installed PyTorch build included CUDA support; all models ran on CPU, so
a CPU-only PyTorch 2.11.0 build is sufficient. `environment.yml` pins the
package versions above.

The environment was exported from the original Bunya Conda environment after
the runs. Direct imports confirmed the versions in the table, and `pip check`
reported no broken requirements. `environment-lock.yml` is the full Linux-64
export with local paths removed; it keeps Conda build strings and CUDA
packages and is specific to Linux-64. `docs/bunya_environment.json` records
the export's source revision and checksums.

`pytest` is an optional test dependency (`pip install .[test]`). The archived
records do not include the CPU model. Wall-clock and CPU-time fields depend on
the host.
