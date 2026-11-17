# Method and limits

This page describes how I ran every measurement, why I think the results can be trusted, and every limit I know of. The full
tables of the first part of the study, generated from the logs, are in [`data/tables.md`](../data/tables.md).

## The machine

| item | value | file |
|---|---|---|
| CPUs | 2 x AMD EPYC 7713, 64 cores each, 2 NUMA nodes | `raw/env/lscpu.txt`, `raw/env/numactl_H.txt` |
| GPUs | 8 x NVIDIA RTX A6000, 49140 MiB each, PCIe Gen4 x16 | `raw/env/nvidia-smi_q.txt` |
| NVLink | bridged pairs 0-1, 2-3, 4-5 and 6-7, 4 links of 14.062 GB/s per GPU | `raw/env/nvidia-smi_topo.txt`, `raw/env/nvidia-smi_nvlink_s.txt` |
| NUMA | GPUs 0 to 3 on node 0, GPUs 4 to 7 on node 1 | `raw/env/nvidia-smi_topo.txt` |
| OS and driver | Ubuntu 24.04.5 LTS, kernel 7.0.0-38-generic, NVIDIA driver 580.178.04 | `raw/env/os_driver.txt`, `raw/env/nvidia-smi.txt` |
| network | a dual-port ConnectX-6 InfiniBand card with both ports down | `raw/env/ib_nic.txt` |

`nvidia-smi topo` marks the bridged pairs `NV4`, GPUs on the same socket but different PCIe host bridges `NODE`, and GPUs on
different sockets `SYS`. The names used in the data files are:

| name | GPUs | link |
|---|---|---|
| `nv23` | 2, 3 | NVLink pair, NUMA node 0 |
| `node56` | 5, 6 | same socket, no NVLink |
| `sys36` | 3, 6 | different sockets |
| `g2_nv23` | 2, 3 | 2-GPU collectives |
| `g4_nv23_5_6` | 2, 3, 5, 6 | 4-GPU collectives: one NVLink pair plus two GPUs that are not bridged to each other |

GPUs 0 and 1 ran another user's jobs during all my measurements, and GPU 7 had an older claim by someone else, so I used GPUs 2
to 6. I planned a 4-GPU set made of two NVLink pairs, but with GPU 4 out of use no second free pair was left.

## How I ran the benchmarks

* Binding: [`scripts/gpu_bind.sh`](../scripts/gpu_bind.sh) gives each rank one GPU through `CUDA_VISIBLE_DEVICES` and pins the
  rank with `numactl` to that GPU's NUMA node. During a 2-rank run, `nvidia-smi` showed one benchmark process on each of the two
  intended GPUs (`raw/verify/gpu_mapping_check_compute_apps.csv`).
* Driver: [`scripts/campaign.py`](../scripts/campaign.py) with the configurations in [`scripts/configs.py`](../scripts/configs.py).
  For each phase, all (configuration, run) pairs were shuffled with seed 20261112. Before each run the target GPUs had to be idle
  (at most 5 percent utilisation and 200 MiB used). Each run wrote its own log with the command, the environment, an `nvidia-smi`
  snapshot, the start and end time, the exit code and the wall time.
* OMB options: `-d cuda` with device buffers on both sides. Sizes from 1 B to 4 MiB used OMB's default iterations; sizes from
  8 MiB to 1 GiB used `-i 10 -x 3` and, for `osu_bw` and `osu_bibw`, `-W 8`. Collectives added `-f`.
* Transport modes, set through MVAPICH and UCX variables (documented in the library's `mpivars` output,
  `raw/verify/mvapich_mpivars.txt`):
  * `mv`: the MVAPICH default path inside a node, CH4 shared memory with GPU IPC. It switches to GPU IPC at
    `MPIR_CVAR_CH4_IPC_GPU_P2P_THRESHOLD=1048576`, and `MPIR_CVAR_CH4_IPC_MAP_REPEAT_ADDR=1` maps a reused buffer earlier.
  * `ucxipc`: `MPIR_CVAR_NOLOCAL=1` sends traffic inside the node through the ch4:ucx netmod, with
    `UCX_TLS=self,sm,cuda_copy,cuda_ipc`.
  * `ucxnoipc`: the same with `UCX_TLS=self,sm,cuda_copy`, so device data is staged through host memory.
* NCCL: `all_reduce_perf`, `all_gather_perf` and `reduce_scatter_perf` with `-b 8 -e 1G -f 2 -g 1 -n 20 -w 5 -c 1`, one MPI
  rank per GPU, `NCCL_DEBUG=INFO` and `NCCL_DEBUG_SUBSYS=INIT,TUNING`. The 4-GPU set used `NCCL_P2P_LEVEL=NVL`. MPI only started
  the processes in these runs (`MPIR_CVAR_ENABLE_GPU=0`).
* Bus bandwidth follows [nccl-tests PERFORMANCE.md](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md):
  algbw = S/t, and busbw = algbw x 2(n-1)/n for allreduce and x (n-1)/n for allgather and reduce_scatter. I read the OMB source
  to convert sizes: for `osu_allreduce` the size is S, for `osu_allgather` it is the bytes sent per rank (S = size x n), and for
  `osu_reduce_scatter` it is S.
* Each performance point is the median of 5 runs, with the minimum and maximum of the runs. [`scripts/analyze.py`](../scripts/analyze.py)
  computes every CSV in `data/` from the logs.
* Piecewise alpha-beta fits (t = alpha + n beta, up to 4 segments chosen by BIC) are in `data/fits_*.csv`. I matched their
  breakpoints against the protocol switches logged by NCCL and UCX (`data/nccl_algo_proto_segments.csv`,
  `data/ucx_protocol_selection.md`).

The allreduce follow-up used its own program and driver; its method is in [allreduce_slowdown.md](allreduce_slowdown.md).

## Why I think the results can be trusted

* Correctness before speed. Before the timed runs, each of the 21 OMB configurations ran one full validation sweep with OMB's
  `-c` check, and every size passed (642 sizes, 0 failures, `data/validation_summary.csv`). One 4-GPU allgather sweep aborted
  after its 512 MiB size had passed, so the 1 GiB point of that collective is measured but not validated. All 30 nccl-tests runs
  reported 0 in the `#wrong` columns (`data/nccl_wrong_summary.csv`). All 147 allreduce follow-up runs passed their own sample
  check.
* Repetition. Every point is the median of 5 runs in shuffled order, with the minimum and maximum shown or stored
  (`data/p2p_summary.csv`, `data/coll_mpi_summary.csv`, `data/coll_nccl_summary.csv`, `data/allreduce_*.csv`).
* The transport was confirmed, not assumed. One extra run per mode with `UCX_LOG_LEVEL=info UCX_PROTO_INFO=y` showed the
  `cuda_ipc` transport for the NVLink runs with CUDA IPC, and `frag host` staging without it (`data/ucx_protocol_selection.md`,
  `raw/ucxlog/`). The NCCL logs showed `P2P/CUMEM` on the NVLink pair and `SHM/direct/direct` between pairs (`raw/nccl/`).
* The slowdown was tested against other explanations. All 9 forced allreduce algorithms show it
  (`data/allreduce_cliff_algo.csv`), host buffers show it at the same size (`data/allreduce_cliff_sweep.csv`), and it moves with
  the pool size (`data/allreduce_cliff_knobs.csv`). The hypotheses were written before the data (next section).
* Profile evidence. The Nsight Systems traces show where the data goes during one allreduce (`data/allreduce_nsys.csv`).
* Independent sources, all opened on 2026-11-17:
  * [CUDA Programming Guide, section 3.4.2.5](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html#host-iommu-hardware-pci-access-control-services-and-vms)
    and the [NCCL troubleshooting page, section "Linux bare-metal IOMMU and PCIe peer-to-peer"](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html),
    on IOMMU and PCIe peer-to-peer.
  * The [MVAPICH 4.1 user guide, page "MVAPICH Environment Variables"](https://mvapich-docs.readthedocs.io/en/open-v4.1/cvar.html#mpir-cvar-lmem-pool-size),
    entry `MPIR_CVAR_LMEM_POOL_SIZE`, and the [MVAPICH 4.1 changelog](https://mvapich.cse.ohio-state.edu/static/media/mvapich/MVP-4.1-CHANGELOG.txt),
    sections "MVAPICH 4.1" ("Based on MPICH 4.3.0") and "MVAPICH 4.0" ("Basic support for major GPUs").
  * The [MPICH 4.3.0 source, `src/include/mpir_mem.h`](https://github.com/pmodels/mpich/blob/v4.3.0/src/include/mpir_mem.h),
    lines 113 to 115, compared with the same macro in MVAPICH 4.1 (`raw/verify/lmem_pool_source.txt`).
  * The [mallopt(3) manual page](https://man7.org/linux/man-pages/man3/mallopt.3.html), section `M_MMAP_THRESHOLD`, and the
    [Linux kernel documentation of the IOMMU group type](https://www.kernel.org/doc/Documentation/ABI/testing/sysfs-kernel-iommu_groups).

  These sources support my reading or are consistent with it; none of them proves it.
* What I could not check. The kernel log and the PCIe ACS bits need root, so the cause of the peer-to-peer corruption is not
  proven. The allreduce slowdown was tested only with `MPI_FLOAT` on 2 ranks.

## The allreduce follow-up was preregistered

Before running the allreduce follow-up, I wrote down five hypotheses and their pass or fail rules and committed them to git.
The public history keeps every commit, its order and its dates:

| event | commit | date (EST) |
|---|---|---|
| hypotheses committed (`followup/PREREGISTRATION.md`) | `e09b4d0` | 2026-11-12 23:18:16 |
| first sweep run started (`raw/allreduce/sweep/runs_index.csv`) | | 2026-11-12 23:18:22 |
| first commit with follow-up measurement data | `659a631` | 2026-11-12 23:49:19 |

The only runs before the preregistration commit were two smoke runs that checked the tool (`raw/allreduce/smoke/`); I did not use
their numbers. In the first version of the file I wrote the time as "23:20 EST", which was a guess. In the next commit I replaced
it with the commit time; the hypotheses did not change (`git diff e09b4d0 659a631 -- followup/PREREGISTRATION.md`). The commit
dates come from my own machine's clock, not from an outside timestamp.

The hypotheses and how they came out (the full text is in `git show e09b4d0:followup/PREREGISTRATION.md`):

1. Location: the slowdown lies between 32 and 64 MiB. It started between 32 and 36 MiB.
2. GPU-specific or not: test host buffers at the same sizes. Host buffers slowed at the same place, so it is not GPU-specific.
3. Algorithm switch: force each algorithm. All of them slowed down, so it is not an algorithm switch.
4. Temporary-buffer pool: the point moves with `MPIR_CVAR_LMEM_POOL_SIZE`, supported if it moves as predicted for at least 2 of 3
   pool sizes. It moved for all 3, so the hypothesis was supported. My prediction for the 16 MiB pool was only partly right.
5. Profile: describe the copies and kernels with Nsight Systems, with no pass or fail rule.

The glibc runs were added afterwards and are labelled exploratory.

For publication I rewrote the history to remove machine identifiers, a file with notes that were not part of the study, and
tool references from commit messages. The content of the measurements and the dates did not change, but the commit hashes differ
from my private copy.

## Limits

* Single node. The InfiniBand ports were down, so there are no GPUDirect RDMA or inter-node results, and nothing here says
  anything about MVAPICH's InfiniBand designs or NCCL's network transports. UCX was built without verbs.
* Open-source MVAPICH 4.1 only. Its changelog lists basic GPU support. I did not test MVAPICH-Plus, which ships as RPM packages
  for RHEL.
* The peer-to-peer cause is strongly supported but not proven: I could not read the kernel log or the ACS bits, and I never
  tested GPUs 0 and 2 ([p2p_corruption.md](p2p_corruption.md)).
* The allreduce mechanism is my reading of the evidence. I did not record buffer addresses or page faults, I profiled one run
  per size, and I tested only `MPI_FLOAT` sums on 2 ranks ([allreduce_slowdown.md](allreduce_slowdown.md)).
* Shared machine. GPUs 0 and 1 ran another user's jobs during every measurement, on the same socket and memory as GPUs 2 and 3.
  Results that use host memory may be lower than on an idle node.
* Paths I could not test: peer-to-peer over PCIe between non-NVLink GPUs (unsafe here), MVAPICH's default GPU path and NCCL's
  default peer-to-peer between pairs, gdrcopy (needs a kernel module), UCX `tcp` (it would listen on the network), and Open MPI
  (built but not used, see [build.md](build.md)).
* Statistics. For sizes of 8 MiB and above, OMB ran 10 timed iterations after 3 warmups, and `osu_bw` and `osu_bibw` used a
  window of 8 instead of 64. Validation was one sweep per configuration, not five. One earlier, slower validation attempt is kept
  in `raw/validate_attempt1/`.
* The peer-to-peer failure numbers come from single diagnostic runs. I did not repeat them because they could break more GPUs.
* With the gforker launcher, OMB printed "could not identify the local rank". Each rank saw only one GPU, so this did not change
  the GPU it used, but OMB's own device selection was not exercised.
* NCCL 2.21.5 was chosen on purpose. Newer releases may choose other algorithms and protocols.
* GPU time. The first part used 113.7 minutes of wall time, or 4.86 GPU-hours counted as wall time times the number of GPUs
  (`data/gpu_time_summary.csv`). My budget was 2 hours; counted in GPU-hours, I went over it. The follow-up used 0.679 GPU-hours
  (`data/allreduce_gpu_time.csv`).

<details>
<summary>Validation sweeps (OMB -c, one sweep per configuration; data/validation_summary.csv)</summary>

| benchmark | GPUs | mode | sizes | passed | failed | ranks aborted after the last size |
|---|---|---|---|---|---|---|
| osu_allgather | g2_nv23 | mv | 1 B to 1 GiB | 31 | 0 | no |
| osu_allgather | g4_nv23_5_6 | ucxnoipc | 1 B to 512 MiB | 30 | 0 | yes |
| osu_allreduce | g2_nv23 | mv | 4 B to 1 GiB | 29 | 0 | no |
| osu_allreduce | g4_nv23_5_6 | ucxnoipc | 4 B to 1 GiB | 29 | 0 | no |
| osu_bibw | node56 | ucxnoipc | 1 B to 1 GiB | 31 | 0 | no |
| osu_bibw | nv23 | mv | 1 B to 1 GiB | 31 | 0 | no |
| osu_bibw | nv23 | ucxipc | 1 B to 1 GiB | 31 | 0 | no |
| osu_bibw | nv23 | ucxnoipc | 1 B to 1 GiB | 31 | 0 | no |
| osu_bibw | sys36 | ucxnoipc | 1 B to 1 GiB | 31 | 0 | no |
| osu_bw | node56 | ucxnoipc | 1 B to 1 GiB | 31 | 0 | no |
| osu_bw | nv23 | mv | 1 B to 1 GiB | 31 | 0 | no |
| osu_bw | nv23 | ucxipc | 1 B to 1 GiB | 31 | 0 | no |
| osu_bw | nv23 | ucxnoipc | 1 B to 1 GiB | 31 | 0 | no |
| osu_bw | sys36 | ucxnoipc | 1 B to 1 GiB | 31 | 0 | no |
| osu_latency | node56 | ucxnoipc | 1 B to 1 GiB | 31 | 0 | no |
| osu_latency | nv23 | mv | 1 B to 1 GiB | 31 | 0 | no |
| osu_latency | nv23 | ucxipc | 1 B to 1 GiB | 31 | 0 | no |
| osu_latency | nv23 | ucxnoipc | 1 B to 1 GiB | 31 | 0 | no |
| osu_latency | sys36 | ucxnoipc | 1 B to 1 GiB | 31 | 0 | no |
| osu_reduce_scatter | g2_nv23 | mv | 4 B to 1 GiB | 29 | 0 | no |
| osu_reduce_scatter | g4_nv23_5_6 | ucxnoipc | 4 B to 1 GiB | 29 | 0 | no |

The aborted 4-GPU allgather run ended with `munmap_chunk(): invalid pointer` on all 4 ranks after its last size had passed;
gforker's `mpiexec` still returned 0. The matching measurement runs without `-c` completed normally.
</details>
