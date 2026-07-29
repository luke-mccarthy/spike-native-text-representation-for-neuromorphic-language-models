# Execution environment

The reported factorials were run as independent CPU tasks on the Bunya HPC
system. The experiment code explicitly:

- selected `torch.device("cpu")`;
- set `torch.set_num_threads(1)`;
- used one OpenMP thread and one MKL thread per task for the temporal-decoder
  factorial;
- used seeds 1, 2, and 3 as declared in the frozen grids.

The row ledgers and the post-experiment Bunya environment capture record:

| Component | Recorded value |
|---|---|
| Python | 3.10.20 |
| NumPy | 2.2.6 |
| PyTorch | 2.11.0+cu126 |
| SentencePiece | 0.2.1 |
| snnTorch | 0.9.4 |
| Torch threads | 1 |
| OpenMP threads | 1 |
| MKL threads | 1 |

The CUDA-tagged PyTorch build was installed in the HPC environment, but these
reference implementations executed on CPU. `environment.yml` pins the
cross-platform package versions needed by the code; a CUDA build is not
required for the recorded CPU execution path.

The environment was captured from the original Bunya Conda environment after
the reported experiments. Direct imports confirmed NumPy 2.2.6,
SentencePiece 0.2.1, snnTorch 0.9.4, and PyTorch 2.11.0+cu126. `pip check`
reported no broken requirements. Pytest was not installed in the experimental
environment; it is available only through the optional `test` dependency in
`pyproject.toml`.

`environment-lock.yml` is a sanitised, path-independent copy of the full
Linux-64 Bunya export. It retains Conda build strings and CUDA package pins, so
it is archival rather than cross-platform. Use `environment.yml` for a compact
portable setup. The exact evidence location, source revision, and checksums are
recorded in `bunya_environment.json`.

The archived records do not identify a CPU microarchitecture, so this
repository does not claim one. Wall-clock and process-time fields can vary by
host even when deterministic model and metric values reproduce.
