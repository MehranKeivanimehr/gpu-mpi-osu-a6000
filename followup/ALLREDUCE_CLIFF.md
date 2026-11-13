# The MVAPICH 4.1 device-buffer allreduce cliff (Item C)

**Short answer.**
* The cliff starts between **32 and 36 MiB**, not just "between 32 and 64 MiB". Host buffers drop at the same place.
* It is **not an algorithm switch**.
* It moves exactly with **`MPIR_CVAR_LMEM_POOL_SIZE`**, MVAPICH 4.1's pre-allocated pool for temporary buffers, which defaults to
  32 MiB.
* My reading: above the pool size, each call stages half of the data through a newly allocated, never-touched pageable host buffer.
  The device-to-host copy into that buffer drops from 13.87 to 3.22 GB/s.

This is open-source MVAPICH 4.1, whose changelog lists "Basic support for major GPUs (from NVIDIA, AMD, and Intel)"
(`env/mvapich_4.1_CHANGELOG_excerpt.txt`, line 28 of the tarball's `CHANGELOG`). It is not MVAPICH-Plus/GDR, and I did not try those.
So everything below describes how 4.1 stages GPU reductions on this node. It is not a verdict on MVAPICH.

Setup (all runs):
* GPUs 2 and 3 (NVLink NV4 pair), one rank per GPU, MVAPICH 4.1 + UCX 1.22.0.
* The same environment as the first study's `mv` mode.
* `scripts/allreduce_sizes.c`: `MPI_FLOAT` sum, out of place.
* Each run: 5 warmups, then 20 timed iterations, each preceded by `MPI_Barrier`. Each iteration's time is the max across ranks; a run
  reports the median of its iterations.
* Each configuration ran 5 times in randomized order, and the size order inside each run was randomized too.
* Each point below is the median of the 5 run medians, with the [min–max] of the run medians.
* Bus bandwidth = S/t for 2 ranks.
* Every run passed the correctness check (`n_bad = 0`; 0 failed runs out of 147 measurement runs).
* The hypotheses and pass/fail rules were committed before measuring: `PREREGISTRATION.md`, commit `e09b4d0`, 23:18:16 EST, six
  seconds before the first sweep run.
* GPU time: 2,443.0 GPU-s = **0.679 GPU-hours** (`results/gpu_time_summary.csv`), plus one untimed `mpivars` start-up on GPU 2.

## 1. Where the cliff starts (MEASURED)

`results/summary_sweep.csv`, `results/cliff_sweep.csv`, logs `raw/sweep/dev_default__rep{1..5}.log`.

| S (MiB) | 24 | 28 | 32 | **36** | 40 | 48 | 64 | 96 | 128 |
|---|---|---|---|---|---|---|---|---|---|
| device, time µs | 3,071.37 | 3,534.82 | **4,079.89** [3,989.65–4,214.01] | **15,784.26** [14,536.07–16,656.04] | 18,156.03 | 20,108.37 | 28,100.53 | 42,619.11 | 57,107.19 |
| device, busbw GB/s | 8.194 | 8.306 | **8.224** | **2.392** | 2.310 | 2.503 | 2.388 | 2.362 | 2.350 |

* **The cliff boundary is 32 → 36 MiB.** Bus bandwidth falls to 0.291× its 32 MiB value: 4 MiB more data takes 3.87× the time. From
  36 MiB up to 128 MiB it stays flat at 2.29–2.50 GB/s.
* **Cross-check against the first study's OMB numbers** (`raw/coll/osu_allreduce__g2_nv23__mv__L__rep*.log`): OMB measured 4,354.92 µs
  at 32 MiB and 29,377.73 µs at 64 MiB; this program measures 4,079.89 and 28,100.53 µs, within 7 %.

## 2. Host vs device buffers (MEASURED)

Same sizes and settings, `malloc`ed buffers. Logs `raw/sweep/host_default__rep*.log`.

| S (MiB) | 28 | 32 | **36** | 64 | 128 |
|---|---|---|---|---|---|
| host, time µs | 9,521.67 | **10,811.59** | **23,023.76** | 43,549.17 | 90,069.74 |
| host, busbw GB/s | 3.084 | **3.104** | **1.640** | 1.541 | 1.490 |

* **Host buffers have a cliff at the same place, 32 → 36 MiB**, but it is smaller: 0.528× the bandwidth vs 0.291× for device buffers.
  So the cliff is **not GPU-specific**. It is bigger for device buffers because their fast side is faster (8.2 vs 3.1 GB/s).
* **INFERRED:** the extra time per call past the cliff is about the same for both buffer types. At 64 MiB, compared with the same size
  with a 128 MiB pool (§3), it is 20.4 ms for device (28,100.53 − 7,741.44 µs) and 22.7 ms for host (43,549.17 − 20,886.55 µs). That points to a cost that doesn't depend on where the user's
  buffers live.

## 3. Which settings move or remove it (MEASURED)

### 3a. The temporary-buffer pool (pre-registered hypothesis C4)

`MPIR_CVAR_LMEM_POOL_SIZE` is documented in `raw/cvars/mpivars.txt` line 680 and in `env/mvapich_4.1_README.envvar_excerpt.txt`:
"Sets the amount of space allocated at init time to be used by internal functions for local temp buffers", default `32 * 1024 * 1024`.

Logs: `raw/knobs/*.log`; summary: `results/summary_knobs.csv`.

| device buffers | cliff boundary | last fast size | first slow size |
|---|---|---|---|
| pool 16 MiB | **28 → 32 MiB** | 28: 3,600.98 µs, 8.153 GB/s | 32: 13,552.92 µs, 2.476 GB/s |
| pool 32 MiB (default) | **32 → 36 MiB** | 32: 4,093.54 µs, 8.197 GB/s | 36: 15,809.33 µs, 2.388 GB/s |
| pool 64 MiB | **64 → 96 MiB** | 64: 7,570.05 µs, 8.865 GB/s | 96: 42,013.52 µs, 2.396 GB/s |
| pool 128 MiB | **none up to 128 MiB** | 128: **15,026.16 µs, 8.932 GB/s** (default: 55,879.78 µs, 2.402) | — |
| host buffers, pool 128 MiB | **none up to 128 MiB** | 64: 20,886.55 µs, 3.213 GB/s; 128: 45,816.84 µs, 2.929 GB/s | — |

* **Verdict on C4:** **supported.** The boundary moved for all 3 non-default pool sizes, and with 64 and 128 MiB pools the sizes up to
  the pool size stayed fast, as predicted.
* **One part of my prediction was wrong.** I predicted that a 16 MiB pool would move the boundary below 24 MiB. It only moved to
  28 → 32 MiB. §4 explains this with a second, glibc-level threshold at 32 MiB.
* With a 128 MiB pool, the 128 MiB device allreduce is **3.72× faster** (55,879.78 → 15,026.16 µs).

### 3b. Allreduce algorithm and CH4 path (pre-registered hypothesis C3)

**The algorithms.** Documented choices, from `raw/cvars/mpivars.txt` lines 335–348 and the README.envvar excerpt.
* Forced algorithms were run with `MPIR_CVAR_DEVICE_COLLECTIVES=none`, so MPICH's generic `MPIR_Allreduce_impl` runs the forced
  algorithm. With the default (`percoll`), the CH4 device-level code would choose instead.
* `mvapich_tuned`, `osu_mcast` and `osu_sharp` were **not run**. In the 4.1 source, `MPIR_Allreduce_impl`'s switch
  (`src/mpi/coll/mpir_coll.c`, around line 5269) has no case for them and ends in `default: MPIR_Assert(0)`. `osu_mcast` and `osu_sharp`
  also need InfiniBand multicast or SHARP hardware, and this node's IB ports are down.
* Sizes: 28, 32, 36, 48 and 64 MiB, bracketing the 32 → 36 boundary from §1, so no refinement was needed.
* Logs: `raw/algo/*.log`; summary: `results/summary_algo.csv`, `results/cliff_algo.csv`.

| configuration | 28 | 32 | 36 | 48 | 64 MiB (median busbw GB/s) | cliff boundary |
|---|---|---|---|---|---|---|
| default (auto, `percoll`) | 8.170 | 8.128 | 2.330 | 2.434 | 2.402 | 32 → 36 |
| `DEVICE_COLLECTIVES=none`, auto | 8.112 | 8.061 | 2.363 | 2.437 | 2.421 | 32 → 36 |
| … `reduce_scatter_allgather` | 8.162 | 8.277 | 2.458 | 2.536 | 2.465 | 32 → 36 |
| … `recursive_doubling` | 4.715 | 4.797 | 1.235 | 1.253 | 1.245 | 32 → 36 |
| … `recexch` | 3.225 | 3.195 | 3.194 | 1.075 | 1.082 | 36 → 48 |
| … `k_reduce_scatter_allgather` | 8.217 | 2.350 | 2.406 | 2.329 | 2.371 | 28 → 32 |
| … `nb` | 8.060 | 2.363 | 2.429 | 2.347 | 2.435 | 28 → 32 |
| … `ring` | 4.282 | 1.946 | 1.870 | 1.878 | 1.965 | 28 → 32 |
| … `tree` | 6.463 | 1.258 | 1.259 | 1.281 | 1.266 | 28 → 32 |
| … `recursive_multiplying` | 3.327 | 1.135 | 1.154 | 1.111 | 1.083 | 28 → 32 |
| … `smp` | 2.662 | 1.261 | 0.726 | 0.729 | 0.725 | 28 → 32 |
| `ALLREDUCE_POSIX_INTRA_ALGORITHM=mpir` | 8.075 | 8.155 | 2.461 | 2.397 | 2.519 | 32 → 36 |
| `ALLREDUCE_POSIX_INTRA_ALGORITHM=release_gather` | 8.289 | 8.209 | 2.384 | 2.499 | 2.528 | 32 → 36 |
| `ALLREDUCE_COMPOSITION=1` | 2.668 | 1.225 | 0.724 | 0.708 | 0.724 | 28 → 32 |
| `ALLREDUCE_COMPOSITION=2` | 8.164 | 8.078 | 2.468 | 2.394 | 2.556 | 32 → 36 |
| `ALLREDUCE_COMPOSITION=3` | 8.186 | 8.210 | 2.381 | 2.432 | 2.456 | 32 → 36 |

* **Verdict on C3: not an algorithm switch** (pre-registered rule).
  * **All 9 forced algorithms have a cliff**, and so do all 6 CH4-level settings.
  * The default's numbers on **both** sides match one algorithm, `reduce_scatter_allgather`, within run-to-run spread.
* **Where the cliff sits depends on the algorithm** (28 → 32, 32 → 36 or 36 → 48 MiB).
  * **INFERRED:** each algorithm asks for a different amount of temporary memory per call, so each crosses the 32 MiB pool at a
    different message size.
  * I did not instrument per-algorithm buffer sizes.

### 3c. glibc malloc settings (EXPLORATORY, added after 3a and the profile; not pre-registered)

The settings are glibc's documented tunables: https://sourceware.org/glibc/manual/2.39/html_node/Memory-Allocation-Tunables.html
(local copy in `env/docs_fetch/`). This node runs glibc 2.39 (`env/glibc_version.txt`). Logs: `raw/glibc/*.log`; summary:
`results/summary_glibc.csv`. The default pool was used throughout.

| device buffers, `GLIBC_TUNABLES=` | 32 MiB | 36 MiB | 64 MiB | 128 MiB | cliff |
|---|---|---|---|---|---|
| (unset, control) | 4,188.70 µs | 16,031.46 | 28,075.07 | 56,096.62 | 32 → 36 |
| `glibc.malloc.mmap_max=0` | 4,140.03 | 4,638.15 | 25,857.02 [7,655.86–27,862.47] | 53,608.23 [15,137.82–56,508.48] | 36 → 64, with fast and slow runs mixed |
| `mmap_max=0:top_pad=33554432` | 4,197.13 | 4,803.33 | 25,143.96 [8,169.67–27,718.22] | 55,255.04 | 36 → 64, mixed |
| `mmap_max=0:trim_threshold=1073741824` | 4,082.30 | 4,562.34 | **7,938.88** | **15,206.72** | **none** |

* With mmap disabled **and** the heap never trimmed, the cliff disappears. At 128 MiB this matches the 128 MiB-pool result
  (15,206.72 vs 15,026.16 µs). Disabling mmap alone fixes 36 MiB but leaves 64–128 MiB slow in most runs.
* For host buffers, `mmap_max=0` (with or without the trim setting) removes the drop but makes the whole host curve slower. With the
  trim setting it runs at about 2.0 GB/s from 28 to 128 MiB, vs 3.1 GB/s below the cliff by default. I did not investigate why, and
  `host_lmem128M` (§3a) is the cleaner host result.
* **Note (INFERRED from source):** MVAPICH 4.1 itself has `MPIR_CVAR_MALLOC_ENABLE_MMAP=0` and `MPIR_CVAR_MALLOC_HEAP_STEP=32 MiB`.
  Its README.envvar says these disable mmap "for cache performance". But the only code that applies them with `mallopt()` is in the OFI
  netmod init (`src/mpid/ch4/netmod/ofi/ofi_init.c` lines 727–731). This build uses ch4:ucx, so they probably have no effect here. My
  `top_pad=33554432` configuration imitates them, and it does not remove the cliff.

## 4. Profile: what happens inside one call (MEASURED with Nsight Systems; mechanism INFERRED)

`nsys profile --sample=none --cpuctxsw=none --trace=cuda,mpi,osrt` ran per rank, without root. It left no daemon behind and opened no
listener (`raw/nsys/listeners_during_nsys.txt`). One run at 32 MiB and one at 64 MiB, each with 3 warmup + 10 timed calls, so 13
allreduce calls per profile. Reports: `raw/nsys/*.nsys-rep`, `raw/nsys/stats_*`; per-operation summary: `results/nsys_trace_summary.csv`.

| per rank (rank 0; rank 1 is the same within a few %) | 32 MiB | 64 MiB |
|---|---|---|
| yaksa reduction kernels (`yaksuri_cudai_kernel_pack_SUM_float`), total / per call | 208 / **16** | 416 / **32** |
| device → **pageable host** copies, 4.194 MB each: median time, throughput | 302.3 µs, **13.87 GB/s** | 1,303.9 µs, **3.22 GB/s** |
| pinned host → device copies, 1.049 MB each: median throughput | 14.38 GB/s | 21.23 GB/s |
| NVLink peer-to-peer copies, 4.194 MB: median throughput | 48.38 GB/s | 48.45 GB/s |
| device → device copy (one of S per call) | 339.45 GB/s | 341.33 GB/s |
| `cuMemcpyAsync` host API time, total / calls | 33.04 ms / 52 | **254.20 ms** / 104 |
| `cudaHostRegister` / `cuMemHostRegister` calls | none | none |

**What one 2-rank call does (MEASURED, by subtracting the program's own one-off copies from the totals):**
* one device→device copy of S;
* NVLink peer copies totalling S;
* **S/2 copied device → pageable host memory** (4 MiB pieces);
* S/2 copied back from pinned host buffers to the device (1 MiB pieces);
* one GPU reduction kernel per 1 MiB.

So **the sum is computed on the GPU, but half of the data makes a round trip through host memory** on every call.

**The only step that slows down at 64 MiB** is the copy into pageable host memory: 4.3× lower throughput. The host-side `cuMemcpyAsync`
time per call grows from 2.54 ms to 19.55 ms. Everything else stays the same or gets faster per byte.

The OS-runtime trace shows the same number of `mmap`/`munmap` calls at both sizes (107 and 24). **INFERRED:** glibc's malloc calls
`mmap` internally, which this kind of tracing does not intercept, so the trace cannot confirm or rule out per-call mapping.

### Likely mechanism (INFERRED, with the evidence for each step)
1. **Half of the data goes through a host temporary buffer.**
   * MEASURED: §4, S/2 to pageable host and back per call.
   * Source: `MPIR_Allreduce` → `MPID_Allreduce` → the `reduce_scatter_allgather` family; the temporary buffer is allocated with
     `MPIR_CHKLMEM_MALLOC`.
2. **Temporary buffers come from a 32 MiB pre-allocated pool while they fit; larger ones are allocated on every call.**
   * DOCUMENTED: the `MPIR_CVAR_LMEM_POOL_SIZE` description.
   * Source: `src/include/mpir_mvp_lmem.h` `MPIR_MVP_LALLOC` falls back to `MPL_aligned_alloc` when
     `mvp_lmem_free_head + bytes > pool end`, and `MPIR_MVP_LFREE` frees it after the call.
3. **glibc hands out allocations above 32 MiB as fresh `mmap` regions and returns them on free, so the next call gets new pages that
   the kernel has to fault in and zero again.**
   * DOCUMENTED, `mallopt(3)` (https://man7.org/linux/man-pages/man3/mallopt.3.html; local copy `env/docs_fetch/mallopt.3.txt`, lines
     121–152): the dynamic mmap threshold has an upper limit of `4*1024*1024*sizeof(long)`, which is 32 MiB on 64-bit. mmap'd space "is
     not placed on the free list for reuse", and "the kernel must perform the expensive task of zeroing out memory allocated via mmap(2)".
4. **The device→host copy into those fresh pages is slow** (MEASURED: 13.87 → 3.22 GB/s), which makes the whole call about 3.4× slower
   per byte. Steps 1–3 are consistent with every measurement:
   * pool size moves the cliff (§3a);
   * the algorithms' cliffs sit at different sizes (§3b);
   * host buffers show the same boundary and about the same extra time (§2);
   * disabling glibc's mmap and trimming removes it (§3c);
   * a 16 MiB pool keeps 24–28 MiB fast. Those per-call allocations are below glibc's 32 MiB limit, so they come from the reusable heap.

   Not shown directly: the buffer address on each call, and page-fault counts. Neither was instrumented.

Even without the cliff, the 2-GPU NVLink allreduce with device buffers reaches about 8.9 GB/s. NCCL 2.21.5 measured 43.70 GB/s at
256 MiB on the same pair in the first study. **INFERRED:** the host round trip in step 1 remains even when the cliff is gone.

## 5. Questions for the lab

1. Is this how MVAPICH 4.1 is meant to stage GPU reductions? That is: half the data through a pageable host temporary buffer, with
   temporary buffers above `MPIR_CVAR_LMEM_POOL_SIZE` allocated on every call. And does MVAPICH-Plus keep this staging on the device,
   or in a pinned, reused pool, which would remove both the cliff and the host round trip?
2. Is raising `MPIR_CVAR_LMEM_POOL_SIZE` a reasonable user-level workaround? Or does the 32 MiB default reflect a per-rank memory
   trade-off I'm not seeing?
3. Are `MPIR_CVAR_MALLOC_ENABLE_MMAP` and `MPIR_CVAR_MALLOC_HEAP_STEP` intended to apply only to OFI builds, or also to ch4:ucx builds
   like this one?

## 6. Limitations
* One node, one NVLink pair (GPUs 2–3), 2 ranks, `MPI_FLOAT` sum only. Nothing here covers more ranks, other datatypes (the source
  shows `MPI_DOUBLE` takes a different path, `MPIR_CVAR_GPU_DOUBLE_SUPPORT=0`), inter-node runs or MVAPICH-Plus.
* **Shared machine.** Another user's jobs ran on GPUs 0–1, on the same socket as GPUs 2–3, the whole time: 30–100 % utilisation in
  the 149 pre-run snapshots.
* **Run-to-run spread.** For 45 of the 201 (configuration, size) points, the slowest run median is more than 1.2× the fastest
  (`results/summary_*.csv`).
  * Most configurations vary by much less than the cliff itself. For example, the default device runs at 32 MiB range 3,989.65–4,214.01 µs.
  * Two groups are **bimodal**, with fast and slow runs mixed: the exploratory `mmap_max=0` configurations (up to 3.7× between runs)
    and the forced `recexch` algorithm (up to 3.1×). Their cliff boundaries are therefore less certain than the others.
* **Only one profile per size.** The profile has 13 calls per size, and the mechanism steps 1–3 are inferred from the profile, the
  documentation and the source. Buffer addresses and page faults were not traced.
* **The glibc runs (§3c) are exploratory and change the allocator for the whole process.** They are a diagnosis, not a recommended
  setting.
* `mvapich_tuned`, `osu_mcast` and `osu_sharp` were not run (§3b).
