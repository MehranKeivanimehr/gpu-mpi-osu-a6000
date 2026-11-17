# PCIe peer-to-peer corruption between GPUs that are not NVLink partners

> **Warning.** On this machine the IOMMU translated GPU DMA. Enabling CUDA peer access between two GPUs that are not NVLink
> partners corrupted every word that was copied and left one GPU in a state that needed a reset. Do not run
> [`src/p2p_probe.cu`](../src/p2p_probe.cu), [`scripts/run_p2p_probe.sh`](../scripts/run_p2p_probe.sh) or
> [`scripts/diag_hang.sh`](../scripts/diag_hang.sh) on a shared machine. On any machine, do not run them across non-NVLink GPUs
> unless the IOMMU is off or in passthrough mode for those GPUs.

## What happened

This study started from wrong allreduce results that NCCL produced between GPUs 0 and 2 in an earlier PyTorch job on this
machine. Those results are not in this repository, and I did not reproduce them. GPU 0 was busy with another user's job during
my measurements, so I never tested GPUs 0 and 2.

During the first part of the study, MVAPICH's default GPU path between GPUs 3 and 4 (different CPU sockets, no NVLink) behaved
badly. These were single diagnostic runs (`raw/diag/`), not performance measurements:

* A 1-byte `osu_latency` took 93,791.80 microseconds. It took 93,773.82 with MVAPICH's GPU IPC threshold raised, and 93,668.13
  through UCX `cuda_ipc`.
* Messages of 256 KiB and larger did not finish within the 40 s timeout.
* An 8 MiB `osu_bw` between GPUs 5 and 6 (same socket, no NVLink) also timed out (`raw/diag/bw_node56_8M.log`). The same
  command on the NVLink pair 2 and 3 finished in 4.18 s at 51,633.77 MB/s (`raw/diag/bw_nv23_8M.log`).

To take MPI, UCX and NCCL out of the picture, I wrote a direct CUDA probe, [`src/p2p_probe.cu`](../src/p2p_probe.cu), and ran it
once per pair:

| GPUs | peer access | 64 MiB copy | wrong words | log |
|---|---|---|---|---|
| 2 and 3 (NVLink partners) | on | 51.16 GB/s | 0 of 16,777,216 | `raw/p2p_probe/probe_2_3_dma.log` |
| 3 and 4 (not partners) | off (the driver staged through host memory) | 8.56 GB/s | 0 of 16,777,216 | `raw/p2p_probe/probe_3_4_dma.log` |
| 3 and 4 (not partners) | on (PCIe peer-to-peer) | 1.27 GB/s | 16,777,216 of 16,777,216 | `raw/p2p_probe/probe_3_4_dma.log` |

With peer access on between GPUs 3 and 4, the 4 KiB and 1 MiB copies were also completely wrong (1,024 of 1,024 and 262,144 of
262,144 words), each copy took about 49 to 53 ms, and a kernel on GPU 4 that read GPU 3's memory directly also read wrong values.
The probe then failed with `unspecified launch failure`.

Afterwards GPU 4 showed 35 percent utilisation with no process, a fan reading of `ERR!` and 97 W in state P2
(`raw/p2p_probe/nvidia-smi_after_probe_3_4.txt`). About 20 hours later it was in the same state
(`raw/p2p_probe/gpu_status/nvidia-smi_query_all.csv`), and `nvidia-smi -q` reported `GPU Recovery Action: Reset` for GPU 4 and
`None` for the other seven GPUs (`raw/p2p_probe/gpu_status/recovery_action_all_gpus.txt`).
I cannot reset a GPU without root. I did not run this probe again, and I did not use GPU 4 or its partner GPU 5 after that.

## The IOMMU evidence

I collected this without root and without any further peer-to-peer runs.

| what | value | file |
|---|---|---|
| IOMMU group type of each GPU | `DMA-FQ` for all 8 GPUs | `raw/env/iommu_groups.csv` |
| kernel command line | no `iommu=`, `amd_iommu=` or `iommu.passthrough=` parameter | `raw/env/iommu_cmdline.txt` |
| kernel build default | `CONFIG_IOMMU_DEFAULT_DMA_LAZY=y`; `CONFIG_IOMMU_DEFAULT_PASSTHROUGH` not set | `raw/env/iommu_kernel_config.txt` |
| `nvidia-smi topo -p2p` | `OK` for every GPU pair | `raw/env/nvidia-smi_topo_p2p_*.txt` |
| PCIe AER error counters, GPU 4 | all 0 | `raw/p2p_probe/gpu_status/aer_and_driver_info.txt` |

From the Linux kernel documentation of
[`/sys/kernel/iommu_groups/<grp_id>/type`](https://www.kernel.org/doc/Documentation/ABI/testing/sysfs-kernel-iommu_groups):
`DMA` means "All the DMA transactions from the device in this group are translated by the iommu", and `DMA-FQ` is the same
"but using batched invalidation to lazily remove translations after use". `identity` would mean the transactions "are not
translated by the iommu". So every GPU on this node was behind a translating IOMMU, and pass-through was not set.

From NVIDIA's documentation:

* [CUDA Programming Guide, section 3.4.2.5, "Host IOMMU Hardware, PCI Access Control Services, and VMs"](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html#host-iommu-hardware-pci-access-control-services-and-vms):
  "On Linux specifically, CUDA and the display driver do not support IOMMU-enabled bare-metal PCIe peer-to-peer memory
  transfer." and "The IOMMU must be disabled when running Linux on a bare metal system to prevent silent device memory
  corruption."
* [NCCL User Guide, "GPU troubleshooting", section "Linux bare-metal IOMMU and PCIe peer-to-peer"](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html):
  "On affected bare-metal systems, avoid translated IOMMU mode for GPU PCIe P2P." The same section says that a positive
  `nvidia-smi topo -p2p p` result "is not a complete system sanity check for this issue".
* [GPUDirect RDMA guide, section 3.4 "Supported Systems", "IOMMUs"](https://docs.nvidia.com/cuda/gpudirect-rdma/index.html):
  GPUDirect RDMA is "incompatible with IOMMUs performing any form of translation other than 1:1".

I opened each of these pages on 2026-11-17 and checked that the quoted sentences and section numbers are on them.

## What this supports and what it does not prove

The configuration that NVIDIA documents as unsupported (bare-metal Linux, IOMMU translating GPU DMA, PCIe peer-to-peer) is the
configuration of this node, and the probe failed in the way the documentation describes. This supports the IOMMU as the cause,
but it does not prove it. I could not read the kernel log (`dmesg` and the journal need root on this node,
`raw/p2p_probe/gpu_status/kernel_log_access.txt`), so I have no IOMMU fault message that ties the failure to the IOMMU. I also
could not read the PCIe ACS settings, which need root.

GPUs 0 and 2 are in the same situation as GPUs 3 and 4: both are behind `DMA-FQ` groups, they sit on different PCIe root
complexes, and they are not NVLink partners. That is consistent with the original NCCL problem having the same cause, but I
did not test that pair.

Two checks would settle it, and both need an administrator: the kernel log around the probe for AMD-Vi `IO_PAGE_FAULT` and NVIDIA
`Xid` messages, and one short correctness test (for example `nccl-tests -c 1` between GPUs 0 and 2) with the IOMMU in passthrough
mode.

## How I worked around it

* NCCL with `NCCL_P2P_LEVEL=NVL` restricts peer-to-peer to NVLink. All 30 nccl-tests runs, including the 4-GPU runs with this
  setting, reported 0 wrong results (`data/nccl_wrong_summary.csv`).
* For MVAPICH between GPUs that are not NVLink partners, I used UCX without `cuda_ipc`, so that data was staged through host
  memory. Every validated size passed (`data/validation_summary.csv`).
* `check_safe()` in [`scripts/configs.py`](../scripts/configs.py) and [`scripts/run_allreduce.py`](../scripts/run_allreduce.py)
  refused any configuration that could put traffic on this path.
