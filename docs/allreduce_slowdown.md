# The allreduce slowdown between 32 and 36 MiB

In the first part of the study, `osu_allreduce` on two NVLink-bridged GPUs took 4,354.92 microseconds at 32 MiB and
29,377.73 microseconds at 64 MiB. I wrote a small program to find where that slowdown starts and what moves it.

Summary of what I found:

* With device buffers, the allreduce slowed sharply between 32 and 36 MiB. Host buffers slowed at the same size.
* All 9 allreduce algorithms that I could force showed a slowdown, so it was not a switch between algorithms.
* The point moved with `MPIR_CVAR_LMEM_POOL_SIZE`. Raising it from 32 to 128 MiB made a 128 MiB allreduce 3.7 times faster.
* This variable is defined by MVAPICH, not by MPICH. It sizes a pool that MVAPICH 4.1 uses for the temporary buffers of the
  MPICH-derived allreduce code.

This is the open-source MVAPICH 4.1 release. Its changelog lists "Basic support for major GPUs (from NVIDIA, AMD, and Intel)"
([MVP-4.1-CHANGELOG.txt](https://mvapich.cse.ohio-state.edu/static/media/mvapich/MVP-4.1-CHANGELOG.txt), section "MVAPICH 4.0").
I did not test MVAPICH-Plus. Everything below describes what I measured with 4.1 on this node, and ends with questions.

## Setup

* GPUs 2 and 3 (an NVLink pair), one MPI rank per GPU, MVAPICH 4.1 with UCX 1.22.0, the same environment as the `mv` mode in
  `scripts/configs.py`.
* Program: [`src/allreduce_sizes.c`](../src/allreduce_sizes.c), `MPI_Allreduce` of `MPI_FLOAT` with `MPI_SUM`, out of place.
  Buffers were allocated once at the largest size.
* Each run did 5 warmup calls and then 20 timed calls, each after an `MPI_Barrier`. The time of a call was the maximum over
  the two ranks, and a run reported the median of its 20 calls.
* Each configuration ran 5 times. All runs of a phase were shuffled with a fixed seed, and the size order inside each run was
  shuffled too. A table entry is the median of the 5 run medians, with the minimum and maximum of the run medians.
* Bus bandwidth is S/t, because the factor 2(n-1)/n is 1 for two ranks.
* Every run passed the correctness check on a sample of elements: 147 measurement runs, 0 wrong elements.
* I wrote the hypotheses and pass or fail rules down and committed them before the first measurement
  ([method.md](method.md#the-allreduce-follow-up-was-preregistered)).

Run logs are in `raw/allreduce/<phase>/`. [`scripts/analyze_allreduce.py`](../scripts/analyze_allreduce.py) turns them into
`data/allreduce_<phase>.csv` and `data/allreduce_cliff_<phase>.csv`. The slowdown point of a configuration is the first pair of
neighbouring sizes where bus bandwidth drops by more than 40 percent.

## Where it starts

Device buffers, default settings (`data/allreduce_sweep.csv`, logs `raw/allreduce/sweep/dev_default__rep*.log`):

| size (MiB) | 24 | 28 | 32 | 36 | 40 | 48 | 64 | 96 | 128 |
|---|---|---|---|---|---|---|---|---|---|
| time (microseconds) | 3,071.37 | 3,534.82 | 4,079.89 | 15,784.26 | 18,156.03 | 20,108.37 | 28,100.53 | 42,619.11 | 57,107.19 |
| bus bandwidth (GB/s) | 8.194 | 8.306 | 8.224 | 2.392 | 2.310 | 2.503 | 2.388 | 2.362 | 2.350 |

* The run medians at 32 MiB ranged from 3,989.65 to 4,214.01 microseconds, and at 36 MiB from 14,536.07 to 16,656.04.
* Bus bandwidth at 36 MiB was 0.291 of its value at 32 MiB. From 36 to 128 MiB it stayed between 2.29 and 2.50 GB/s.
* My program agreed with the first part's OMB runs within 7 percent: 4,079.89 against 4,354.92 microseconds at 32 MiB, and
  28,100.53 against 29,377.73 at 64 MiB.

Host buffers from `malloc` (`raw/allreduce/sweep/host_default__rep*.log`):

| size (MiB) | 28 | 32 | 36 | 64 | 128 |
|---|---|---|---|---|---|
| time (microseconds) | 9,521.67 | 10,811.59 | 23,023.76 | 43,549.17 | 90,069.74 |
| bus bandwidth (GB/s) | 3.084 | 3.104 | 1.640 | 1.541 | 1.490 |

Host buffers slowed at the same place, 32 to 36 MiB, but less: to 0.528 of the bandwidth instead of 0.291. So the slowdown is
not specific to GPU buffers. My reading is that the extra time per call was about the same for both. At 64 MiB, compared with
a 128 MiB pool (next section), it was 20.4 ms for device buffers (28,100.53 minus 7,741.44 microseconds) and 22.7 ms for host
buffers (43,549.17 minus 20,886.55).

## What moves it: the temporary-buffer pool

`MPIR_CVAR_LMEM_POOL_SIZE` "Sets the amount of space allocated at init time to be used by internal functions for local temp
buffers", with a default of 32 MiB. This is the description in the
[MVAPICH 4.1 user guide](https://mvapich-docs.readthedocs.io/en/open-v4.1/cvar.html#mpir-cvar-lmem-pool-size), page "MVAPICH
Environment Variables", and in the library's own `mpivars` output (`raw/verify/mvapich_mpivars.txt`).

Device buffers (`data/allreduce_knobs.csv`, logs `raw/allreduce/knobs/`):

| pool size | slowdown point | last fast size | first slow size |
|---|---|---|---|
| 16 MiB | 28 to 32 MiB | 28 MiB: 3,600.98 microseconds, 8.153 GB/s | 32 MiB: 13,552.92 microseconds, 2.476 GB/s |
| 32 MiB (default) | 32 to 36 MiB | 32 MiB: 4,093.54 microseconds, 8.197 GB/s | 36 MiB: 15,809.33 microseconds, 2.388 GB/s |
| 64 MiB | 64 to 96 MiB | 64 MiB: 7,570.05 microseconds, 8.865 GB/s | 96 MiB: 42,013.52 microseconds, 2.396 GB/s |
| 128 MiB | none up to 128 MiB | 128 MiB: 15,026.16 microseconds, 8.932 GB/s | |

* With a 128 MiB pool, the 128 MiB allreduce took 15,026.16 microseconds instead of 55,879.78, which is 3.7 times faster.
* Host buffers with a 128 MiB pool had no slowdown up to 128 MiB either (64 MiB: 20,886.55 microseconds; 128 MiB: 45,816.84).
* I had predicted that the point would move with the pool size, and it did for all three non-default pools. One part of the
  prediction was wrong: I expected a 16 MiB pool to move the point below 24 MiB, but it moved only to 28 to 32 MiB. The glibc
  limit described below explains this.

## It is not a switch between algorithms

I forced each documented value of `MPIR_CVAR_ALLREDUCE_INTRA_ALGORITHM`
([user guide entry](https://mvapich-docs.readthedocs.io/en/open-v4.1/cvar.html#mpir-cvar-allreduce-intra-algorithm)) with
`MPIR_CVAR_DEVICE_COLLECTIVES=none`, so that the MPICH-level code runs the forced algorithm. I also tried the CH4-level settings.
I did not run `mvapich_tuned`, `osu_mcast` and `osu_sharp`: the 4.1 source has no case for them in `MPIR_Allreduce_impl`
(`src/mpi/coll/mpir_coll.c`), and the last two need InfiniBand multicast or SHARP hardware.

All 9 forced algorithms and all 6 CH4-level settings showed a slowdown. The default matched `reduce_scatter_allgather` on
both sides of the point. Where the point sat depended on the algorithm. My reading is that each algorithm asks for a different
amount of temporary memory, so each crosses the 32 MiB pool at a different message size. I did not instrument this.

<details>
<summary>Bus bandwidth per configuration (GB/s, median of 5 runs; data/allreduce_algo.csv)</summary>

| configuration | 28 MiB | 32 MiB | 36 MiB | 48 MiB | 64 MiB | slowdown point |
|---|---|---|---|---|---|---|
| default | 8.170 | 8.128 | 2.330 | 2.434 | 2.402 | 32 to 36 |
| `DEVICE_COLLECTIVES=none`, auto | 8.112 | 8.061 | 2.363 | 2.437 | 2.421 | 32 to 36 |
| `reduce_scatter_allgather` | 8.162 | 8.277 | 2.458 | 2.536 | 2.465 | 32 to 36 |
| `recursive_doubling` | 4.715 | 4.797 | 1.235 | 1.253 | 1.245 | 32 to 36 |
| `recexch` | 3.225 | 3.195 | 3.194 | 1.075 | 1.082 | 36 to 48 |
| `k_reduce_scatter_allgather` | 8.217 | 2.350 | 2.406 | 2.329 | 2.371 | 28 to 32 |
| `nb` | 8.060 | 2.363 | 2.429 | 2.347 | 2.435 | 28 to 32 |
| `ring` | 4.282 | 1.946 | 1.870 | 1.878 | 1.965 | 28 to 32 |
| `tree` | 6.463 | 1.258 | 1.259 | 1.281 | 1.266 | 28 to 32 |
| `recursive_multiplying` | 3.327 | 1.135 | 1.154 | 1.111 | 1.083 | 28 to 32 |
| `smp` | 2.662 | 1.261 | 0.726 | 0.729 | 0.725 | 28 to 32 |
| `ALLREDUCE_POSIX_INTRA_ALGORITHM=mpir` | 8.075 | 8.155 | 2.461 | 2.397 | 2.519 | 32 to 36 |
| `ALLREDUCE_POSIX_INTRA_ALGORITHM=release_gather` | 8.289 | 8.209 | 2.384 | 2.499 | 2.528 | 32 to 36 |
| `ALLREDUCE_COMPOSITION=1` | 2.668 | 1.225 | 0.724 | 0.708 | 0.724 | 28 to 32 |
| `ALLREDUCE_COMPOSITION=2` | 8.164 | 8.078 | 2.468 | 2.394 | 2.556 | 32 to 36 |
| `ALLREDUCE_COMPOSITION=3` | 8.186 | 8.210 | 2.381 | 2.432 | 2.456 | 32 to 36 |

The forced algorithms ran with `MPIR_CVAR_DEVICE_COLLECTIVES=none`. The `recexch` runs were bimodal (up to 3.1 times between
runs), so its point is less certain.
</details>

## glibc malloc settings (exploratory)

I added these runs after the pool and profile results; they were not in the preregistration. The settings are glibc tunables
([glibc 2.39 manual, "Memory Allocation Tunables"](https://sourceware.org/glibc/manual/2.39/html_node/Memory-Allocation-Tunables.html)).
The node runs glibc 2.39 (`raw/env/glibc_version.txt`). Device buffers, default pool, `data/allreduce_glibc.csv`:

| `GLIBC_TUNABLES` | 32 MiB | 36 MiB | 64 MiB | 128 MiB | slowdown point |
|---|---|---|---|---|---|
| unset | 4,188.70 | 16,031.46 | 28,075.07 | 56,096.62 | 32 to 36 |
| `glibc.malloc.mmap_max=0` | 4,140.03 | 4,638.15 | 25,857.02 | 53,608.23 | 36 to 64, fast and slow runs mixed |
| `mmap_max=0:top_pad=33554432` | 4,197.13 | 4,803.33 | 25,143.96 | 55,255.04 | 36 to 64, mixed |
| `mmap_max=0:trim_threshold=1073741824` | 4,082.30 | 4,562.34 | 7,938.88 | 15,206.72 | none |

Times are in microseconds. With mmap turned off and the heap never trimmed, the slowdown disappeared, and the 128 MiB time
matched the 128 MiB pool result (15,206.72 against 15,026.16). Turning off mmap alone fixed 36 MiB but left 64 and 128 MiB slow
in most runs. For host buffers, `mmap_max=0` removed the drop but made the whole curve slower, and I did not find out why. These
settings change the allocator for the whole process. I used them to diagnose, not as a recommendation.

## Profile of one call

I ran Nsight Systems on each rank (`--trace=cuda,mpi,osrt`), once at 32 MiB and once at 64 MiB, each with 13 allreduce calls.
Summaries: `data/allreduce_nsys.csv` and `raw/allreduce/nsys/stats_*`. Rank 0:

| | 32 MiB | 64 MiB |
|---|---|---|
| GPU reduction kernels (`yaksuri_cudai_kernel_pack_SUM_float`) | 208 | 416 |
| copies from device to pageable host memory, 4.194 MB each: median time and throughput | 302.3 microseconds, 13.87 GB/s | 1,303.9 microseconds, 3.22 GB/s |
| copies from pinned host memory to device, 1.049 MB each: median throughput | 14.38 GB/s | 21.23 GB/s |
| NVLink peer copies, 4.194 MB each: median throughput | 48.38 GB/s | 48.45 GB/s |
| `cuMemcpyAsync` host time, total and calls | 33.04 ms, 52 | 254.20 ms, 104 |
| `cudaHostRegister` or `cuMemHostRegister` calls | none | none |

After subtracting the program's own copies, one two-rank call did this: one device-to-device copy of S, NVLink peer copies
adding up to S, S/2 copied from the device to pageable host memory in 4 MiB pieces, S/2 copied back from pinned host buffers in
1 MiB pieces, and one reduction kernel per 1 MiB. The sum ran on the GPU, but half of the data went through host memory on every
call. The only step that slowed down at 64 MiB was the copy into pageable host memory.

## Where this comes from in the source

I read the MVAPICH 4.1 source tarball and compared it with MPICH 4.3.0, the version it is based on. The commands and their
output are in [`raw/verify/lmem_pool_source.txt`](../raw/verify/lmem_pool_source.txt).

* `MPIR_CVAR_LMEM_POOL_SIZE` is defined in `mvapich-4.1/src/include/mpir_mvp_lmem.h`, a file with an Ohio State University
  copyright header. MPICH 4.3.0 has no such file (`src/include/mpir_mvp_lmem.h` returned HTTP 404 on GitHub). So the variable
  only looks like an MPICH variable; it is MVAPICH's own.
* The allreduce algorithm file `src/mpi/coll/allreduce/allreduce_intra_reduce_scatter_allgather.c` is identical in MVAPICH 4.1
  and MPICH 4.3.0. It gets its temporary buffer with `MPIR_CHKLMEM_MALLOC` (line 62).
* In MPICH 4.3.0 that macro calls `MPL_malloc` (`src/include/mpir_mem.h`, lines 113 to 115). In MVAPICH 4.1 it calls
  `MPIR_MVP_LALLOC` (`src/include/mpir_mem.h`, lines 116 to 118). `MPIR_MVP_LALLOC` takes the buffer from the pool when it fits
  and otherwise calls `MPL_aligned_alloc`, on every call.

So the slowdown sits on the MPICH-derived allreduce path in MVAPICH 4.1, at the point where MVAPICH routes temporary buffers
through its own pool.

The 4.1 source also has `MPIR_CVAR_MALLOC_ENABLE_MMAP` and `MPIR_CVAR_MALLOC_HEAP_STEP`. The only code that applies them with
`mallopt()` is in the OFI netmod start-up (`src/mpid/ch4/netmod/ofi/ofi_init.c`, lines 727 to 731). My build uses ch4:ucx, so I
think they had no effect here.

## My reading of the mechanism

1. Half of the data goes through a host temporary buffer on every call (measured in the profile).
2. Temporary buffers come from the 32 MiB pool while they fit, and larger ones are allocated on every call (source above).
3. glibc serves allocations above 32 MiB with fresh `mmap` regions and returns them on `free`. From the
   [mallopt(3) manual page](https://man7.org/linux/man-pages/man3/mallopt.3.html), section `M_MMAP_THRESHOLD`: the dynamic mmap
   threshold has an upper limit of `4*1024*1024*sizeof(long)`, which is 32 MiB on 64-bit systems; mmap'd space "is not placed
   on the free list for reuse"; and "the kernel must perform the expensive task of zeroing out memory allocated via mmap(2)".
4. The copy from the device into those fresh pages is slow (13.87 against 3.22 GB/s in the profile).

This is consistent with every measurement above: the pool size moves the point, the algorithms' points differ, host buffers slow
at the same size by about the same time, the glibc settings remove it, and a 16 MiB pool keeps 24 and 28 MiB fast (those
allocations are below glibc's 32 MiB limit, so they come from the reusable heap). I did not record buffer addresses or page
faults, so this is my reading of the evidence, not a direct observation.

Even without the slowdown, the device-buffer allreduce reached about 8.9 GB/s on this NVLink pair. NCCL 2.21.5 reached 43.70 GB/s
at 256 MiB on the same pair (`data/mpi_vs_nccl_table.csv`). My reading is that the host round trip in step 1 remains.

## Questions for the lab

1. Is this how MVAPICH 4.1 is meant to stage GPU reductions: half of the data through a pageable host buffer, with temporary
   buffers above `MPIR_CVAR_LMEM_POOL_SIZE` allocated on every call? Does MVAPICH-Plus keep this staging on the device, or in a
   pinned buffer that is reused?
2. Is raising `MPIR_CVAR_LMEM_POOL_SIZE` a reasonable setting for users, or does the 32 MiB default reflect a memory trade-off
   per rank that I am not seeing?
3. Are `MPIR_CVAR_MALLOC_ENABLE_MMAP` and `MPIR_CVAR_MALLOC_HEAP_STEP` meant to apply only to OFI builds?

## Limits

* One node, one NVLink pair, 2 ranks, `MPI_FLOAT` with `MPI_SUM` only. The source shows that `MPI_DOUBLE` takes a different path
  (`MPIR_CVAR_GPU_DOUBLE_SUPPORT=0`). I did not test more ranks, other data types, other nodes or MVAPICH-Plus.
* The machine was shared. Another user's jobs ran on GPUs 0 and 1, on the same CPU socket as GPUs 2 and 3, at 30 to 100 percent
  utilisation in the 149 pre-run snapshots.
* For 45 of the 201 (configuration, size) points, the slowest run median was more than 1.2 times the fastest. Most of these
  spreads are much smaller than the slowdown itself. The exploratory `mmap_max=0` settings and the forced `recexch` algorithm
  were bimodal.
* I profiled one run per size. The mechanism steps 1 to 3 come from the profile, the documentation and the source.
