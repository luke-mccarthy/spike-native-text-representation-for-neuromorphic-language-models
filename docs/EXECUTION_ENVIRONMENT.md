# Execution environment

The reported factorials were run as independent CPU tasks on the Bunya HPC
system. The experiment code explicitly:

- selected `torch.device("cpu")`;
- set `torch.set_num_threads(1)`;
- used one OpenMP thread and one MKL thread per task for the temporal-decoder
  factorial;
- used seeds 1, 2, and 3 as declared in the frozen grids.

The temporal-decoder row ledgers record:

| Component | Recorded value |
|---|---|
| Python | 3.10.20 |
| NumPy | 2.2.6 |
| PyTorch | 2.11.0+cu126 |
| Torch threads | 1 |
| OpenMP threads | 1 |
| MKL threads | 1 |

The CUDA-tagged PyTorch build was installed in the HPC environment, but these
reference implementations executed on CPU. `environment.yml` pins the
cross-platform package versions needed by the code; a CUDA build is not
required for the recorded CPU execution path.

The archived environment export omitted the version of `snntorch` even though
one imported compatibility module requires it at import time. The portable
environment pins `snntorch==0.9.4`; this is an explicit packaging repair, not a
claim that the omitted original version was recoverable from the run ledgers.

The archived records do not identify a CPU microarchitecture, so this
repository does not claim one. Wall-clock and process-time fields can vary by
host even when deterministic model and metric values reproduce.
