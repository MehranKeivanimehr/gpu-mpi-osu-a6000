# Pre-registration: Item C (the MVAPICH 4.1 allreduce cliff)

Written 2026-11-12 23:20 EST and committed to git **before** the sweep, algorithm and knob runs. The only runs done before this
were two smoke runs that checked the tool works (`raw/smoke/`, 3 timed iterations each). Their numbers are not used for any
conclusion.

## Setup (fixed for every run)
* GPUs 2 and 3 (NVLink NV4 pair), one MPI rank per GPU, launched with MVAPICH 4.1 `mpiexec` (gforker).
* Binding with `scripts/gpu_bind.sh`, and the same environment as the first study's `mv` mode: `UCX_TLS=self,sm,cuda`.
  No NCCL, no UCX transport changes, no cross-pair GPUs.
* Program `scripts/allreduce_sizes.c`: `MPI_Allreduce` with `MPI_FLOAT` and `MPI_SUM`, out of place.
  * Device buffers (`cudaMalloc`) or host buffers (`malloc`), allocated once at the largest size, as OMB does.
  * 5 untimed warmup calls, then 20 timed calls, each preceded by `MPI_Barrier`.
  * Correctness is checked on a sample of elements against a host-computed sum. Any `n_bad > 0` marks the run FAILED.
* **Metric:** for each iteration, the maximum time across ranks. Per run, the median of the 20 iterations. Per configuration, the
  median of the 5 run medians, with the min–max of the run medians as spread. Bus bandwidth is S/t × 2(n−1)/n with n = 2, so it equals S/t.
* Each configuration runs 5 times. All (configuration, run) pairs of a phase are shuffled (seed 20261113), and the size order inside
  each run is shuffled too.
* GPU-time budget: under 1 GPU-hour (wall time × 2 GPUs). The driver stops at 3300 GPU-s.

## Definitions
* **Cliff boundary:** the first pair of adjacent tested sizes a < b where median busbw(b) / median busbw(a) < 0.6, i.e. a drop of more
  than 40 % in bus bandwidth from one size to the next.
* **Cliff removed:** no such pair exists among the sizes tested for that configuration.

## Hypotheses and decision rules
* **C1, location.** On the default settings with device buffers, the cliff boundary lies inside 32 → 64 MiB. The original OMB data put
  it between 32 and 64 MiB. Sizes: 24, 28, …, 64 in 4 MiB steps, plus 96 and 128 MiB. Reported as "starts at b MiB".
* **C2, GPU-specific?** Same sizes with host buffers.
  * If host buffers have no cliff boundary in 32 → 64 MiB, the cliff is GPU-specific.
  * If they have one at the same place, it is not GPU-specific.
* **C3, algorithm switch?** With `MPIR_CVAR_DEVICE_COLLECTIVES=none`, MPICH's generic collectives run directly. Each documented
  `MPIR_CVAR_ALLREDUCE_INTRA_ALGORITHM` value is then forced in turn:
  `recursive_doubling`, `recursive_multiplying`, `reduce_scatter_allgather`, `tree`, `recexch`, `ring`, `k_reduce_scatter_allgather`, `nb`, `smp`.
  Three more configurations try the CH4 level: `MPIR_CVAR_ALLREDUCE_POSIX_INTRA_ALGORITHM=mpir` or `release_gather`, and
  `MPIR_CVAR_ALLREDUCE_COMPOSITION=1`, `2` or `3`.
  * Sizes around the cliff: 28, 32, 36, 48 and 64 MiB. They are refined to bracket the boundary found in C1, and any change is written
    in `ALLREDUCE_CLIFF.md`.
  * **If the cliff follows an algorithm switch:** at least one forced algorithm has no cliff, and the default's two sides match two
    different forced algorithms.
  * **Otherwise:** the cliff appears in every forced algorithm that runs correctly, so it is not an algorithm switch.
* **C4, temporary-buffer pool (hypothesis from reading the 4.1 source before any measurement).**
  * Background: `MPIR_CVAR_LMEM_POOL_SIZE` defaults to 32 MiB (`raw/cvars/mpivars.txt`). In `src/include/mpir_mvp_lmem.h`, temporary
    buffers that fit in the pool come from it; bigger ones are allocated with `MPL_aligned_alloc` on every call.
  * **Prediction:** the cliff boundary moves with the pool size. With 64 MiB and 128 MiB pools, sizes up to the pool size stay on the fast
    side; with a 16 MiB pool, the boundary moves below 24 MiB.
  * **Supported** if the boundary moves as predicted for at least 2 of the 3 pool sizes. **Rejected** if the boundary stays inside
    32 → 64 MiB for all three.
  * One host-buffer configuration with a 128 MiB pool checks whether any host-buffer cliff moves too.
* **C5, profile (descriptive, no pass/fail).** Nsight Systems, at 32 MiB and 64 MiB, 10 timed iterations each: CUDA memcpy volume by
  direction (D2H, H2D, D2D, P2P), kernel launches, `cudaHostRegister` calls, and OS-runtime `mmap`/`munmap` calls per iteration.
  These are used only to describe the mechanism (INFERRED).

Anything not listed here and run later is labelled **exploratory** in `ALLREDUCE_CLIFF.md`.
