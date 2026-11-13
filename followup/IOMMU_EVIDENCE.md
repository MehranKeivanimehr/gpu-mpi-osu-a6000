# IOMMU evidence for the PCIe P2P corruption (read-only, no new P2P probes)

**Verdict (one line):**
* Every GPU on this node is behind an AMD IOMMU in translated mode (`DMA-FQ`), with no passthrough on the kernel command line.
* NVIDIA documents that this exact configuration is unsupported for PCIe P2P on bare-metal Linux and can silently corrupt device memory.
* So the IOMMU is the most likely cause of the corruption: strongly supported, but **not proven**. The proof would be the kernel log or a
  run with the IOMMU in passthrough, and both need an administrator.

Collected on 2026-11-12 between 23:08 and 23:10 EST as a normal user. Commands are in `followup/commands.log`, outputs in
`followup/env/iommu/`, and page copies in `followup/env/docs_fetch/`. Labels: **MEASURED** (file in this repo), **DOCUMENTED** (official
page I opened, URL given, local copy kept), **INFERRED** (my interpretation).

## 1. Kernel and IOMMU mode (MEASURED)

| what | value | file |
|---|---|---|
| Kernel command line | `<kernel command line removed; it has no iommu=, amd_iommu= or iommu.passthrough= parameter>`. **No `iommu=`, `amd_iommu=` or `iommu.passthrough=`.** | `env/iommu/proc_cmdline.txt` |
| IOMMU units | 8 AMD IOMMUs, `ivhd0`–`ivhd7`, one per PCIe root complex (`pci<root>`, `20`, `40`, `60`, `80`, `a0`, `c0`, `e0`) | `env/iommu/sys_class_iommu.txt`, `env/iommu/ivhd_units.txt` |
| Kernel build default | `CONFIG_IOMMU_DEFAULT_DMA_LAZY=y`; `CONFIG_IOMMU_DEFAULT_PASSTHROUGH` and `CONFIG_IOMMU_DEFAULT_DMA_STRICT` not set; `CONFIG_AMD_IOMMU=y` | `env/iommu/kernel_config_iommu.txt` (from `/boot/config-7.0.0-38-generic`) |
| Per-GPU IOMMU group type | `DMA-FQ` for **all 8 GPUs** (table below) | `env/iommu/gpu_iommu_groups.csv` |

**What `DMA-FQ` means (DOCUMENTED):** "All the DMA transactions from the device in this group are translated by the iommu", "but using
batched invalidation to lazily remove translations after use". `identity` would mean "not translated by the iommu".
Source: Linux kernel ABI documentation, `/sys/kernel/iommu_groups/<grp_id>/type`,
https://www.kernel.org/doc/Documentation/ABI/testing/sysfs-kernel-iommu_groups (local copy `env/docs_fetch/kernel.org_sysfs-kernel-iommu_groups.txt`, lines 33–50).

**INFERRED:** the lazy-DMA kernel default and the absence of `iommu=pt` on the command line together explain why every group is `DMA-FQ`.

### Per-GPU table (MEASURED, `env/iommu/gpu_iommu_groups.csv`, `env/iommu/lspci_tv.txt`)

| GPU | PCI address | IOMMU group | group type | group members | upstream root port | root complex / IOMMU unit | NUMA |
|---|---|---|---|---|---|---|---|
| 0 | `<pci:gpu0>` | 68 | **DMA-FQ** | GPU + its audio fn | `<pci>` | `pci<root>` / `ivhd3` | 0 |
| 1 | `<pci:gpu1>` | 47 | DMA-FQ | GPU + audio | `<pci>` | `pci<root>` / `ivhd2` | 0 |
| 2 | `<pci:gpu2>` | 27 | **DMA-FQ** | GPU + audio | `<pci>` | `pci<root>` / `ivhd1` | 0 |
| 3 | `<pci:gpu3>` | 11 | **DMA-FQ** | GPU + audio | `<pci>` | `pci<root>` / `ivhd0` | 0 |
| 4 | `<pci:gpu4>` | 136 | **DMA-FQ** | GPU + audio | `<pci>` | `pci<root>` / `ivhd7` | 1 |
| 5 | `<pci:gpu5>` | 120 | DMA-FQ | GPU + audio | `<pci>` | `pci<root>` / `ivhd6` | 1 |
| 6 | `<pci:gpu6>` | 103 | DMA-FQ | GPU + audio | `<pci>` | `pci<root>` / `ivhd5` | 1 |
| 7 | `<pci:gpu7>` | 85 | DMA-FQ | GPU + audio | `<pci>` | `pci<root>` / `ivhd4` | 1 |

The NUMA column is from `env/nvidia-smi_topo.txt`. Root complex and IOMMU unit come from the sysfs path in `gpu_iommu_groups.csv`
and the device lists in `ivhd_units.txt`.

**INFERRED from the topology:**
* No two GPUs share a PCIe switch or root complex. Each GPU hangs directly off its own root port, on its own root complex with its own IOMMU unit.
* So any PCIe peer-to-peer transfer between two GPUs that aren't NVLink partners has to leave one root complex, cross the CPU's internal
  fabric and enter another. Both ends are under IOMMU translation.

## 2. PCIe ACS and error counters

* **ACS: BLOCKED.** `lspci -vvv` as a normal user prints `Capabilities: <access denied>` for every GPU and every upstream root port
  (`env/iommu/lspci_vvv_*.txt`, 16 files). PCIe extended config space, where ACS lives, needs root. So I can't report the ACS
  capability or control bits. NVIDIA's own check is `sudo lspci -vvv | grep ACSCtl`, which is also root-only.
* **AER error counters (MEASURED):** all zero for GPUs 2, 3 and 4 (`followup/raw/gpu4/aer_and_driver_info.txt`). **INFERRED:** that
  fits an IOMMU translation problem, which the IOMMU reports in its own event log (kernel `AMD-Vi ... IO_PAGE_FAULT` messages), not as a
  PCIe link error. But the kernel log is not readable (`followup/raw/gpu4/kernel_log_access.txt`).
* **`nvidia-smi topo -p2p` reports `OK` for every pair** (MEASURED, `env/nvidia-smi_topo_p2p_*.txt`, from the first study), so
  CUDA, UCX, MPICH and NCCL all believe PCIe P2P works.

## 3. NVIDIA's documentation (DOCUMENTED, pages opened 2026-11-12 23:09–23:10 EST)

**(a) CUDA Programming Guide, §3.4.2.5 "Host IOMMU Hardware, PCI Access Control Services, and VMs"** (in chapter 3.4 "Programming
Systems with Multiple GPUs").
URL: https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html#host-iommu-hardware-pci-access-control-services-and-vms
(local copy `env/docs_fetch/docs.nvidia.com_cuda_cuda-programming-guide_03-advanced_multi-gpu-systems.html`; text extract
`env/docs_fetch/cuda_programming_guide_multi_gpu_systems.txt`, lines 250–258). The first paragraph of that section says:

> On Linux specifically, CUDA and the display driver do not support IOMMU-enabled bare-metal PCIe peer-to-peer memory transfer.
> However, CUDA and the display driver do support IOMMU via virtual machine pass through.
> The IOMMU must be disabled when running Linux on a bare metal system to prevent silent device memory corruption.

The same section, second paragraph:

> The PCI ACS feature redirects all PCI point-to-point traffic through the CPU root complex, which can cause significant performance
> loss due to the reduction in overall bisection bandwidth.

**(b) NCCL User Guide, "GPU troubleshooting" page, section "Linux bare-metal IOMMU and PCIe peer-to-peer".**
URL: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html
(local copy in `env/docs_fetch/`, text extract `env/docs_fetch/nccl_gpu_troubleshooting.txt`, lines 640–662). The section's first
paragraph says:

> On Linux bare-metal systems, however, CUDA and the NVIDIA driver stack do not support IOMMU-enabled PCIe peer-to-peer memory transfer.

The second paragraph says symptoms "can look like application or framework bugs rather than a clean NCCL error". The section ends:

> On affected bare-metal systems, avoid translated IOMMU mode for GPU PCIe P2P. Disable IOMMU translation for the GPU path.

The same section says a positive `nvidia-smi topo -p2p p` "is not a complete system sanity check for this issue". Its suggested check is
`dmesg`, which I can't read, but the sysfs group `type` in §1 shows the same thing: translated mode. The page's "PCI Access Control
Services (ACS)" section adds that IO virtualization "can interfere with GPU Direct by redirecting all PCI point-to-point traffic to the CPU
root complex, causing a significant performance reduction or even a hang."

**(c)** The NVIDIA GPUDirect RDMA guide, section "IOMMUs", says the same applies to NIC↔GPU RDMA. IOMMUs doing translation other than
1:1 "must be disabled or configured for pass-through translation for GPUDirect RDMA to work". URL:
https://docs.nvidia.com/cuda/gpudirect-rdma/index.html (opened 2026-11-12, quote in `docs/doc_quotes.md`).

## 4. GPUs 0 and 2 (the original NCCL bug) vs GPUs 3 and 4 (the probe)

| | GPUs 0 ↔ 2 (original NCCL wrong-allreduce bug) | GPUs 3 ↔ 4 (my probe: 100 % corrupt) |
|---|---|---|
| IOMMU group type, both ends | **DMA-FQ / DMA-FQ** | **DMA-FQ / DMA-FQ** |
| Root complexes / IOMMU units | `pci<root>`/`ivhd3` ↔ `pci<root>`/`ivhd1` (different) | `pci<root>`/`ivhd0` ↔ `pci<root>`/`ivhd7` (different) |
| `nvidia-smi topo` path | NODE (same socket, NUMA 0) | SYS (across sockets) |
| NVLink partners? | no | no |
| Measured result | not tested (GPU 0 busy with another user; deliberately not run) | 16,777,216 / 16,777,216 words wrong (`raw/p2p_probe/probe_3_4_dma.log`) |

**Answer:** yes, GPUs 0 and 2 sit behind the same IOMMU translation type (`DMA-FQ`) as GPUs 3 and 4 (MEASURED). In both pairs the two
GPUs are on different root complexes, and neither pair is NVLink-bridged.

**One difference matters:** 0↔2 is a same-socket (NODE) path, while 3↔4 crosses sockets (SYS). Here is the closest evidence I have for a
same-socket path, without running a new P2P test (MEASURED, first study):
* MVAPICH's default GPU path between GPUs 5 and 6 (NODE, same socket, both `DMA-FQ`) produced **no result for one 8 MiB `osu_bw`
  measurement in 60 s** (`raw/diag/bw_node56_8M.log`, `timeout` exit 124).
* The same command on the NVLink pair 2↔3 finished in 4.18 s at 51,633.77 MB/s (`raw/diag/bw_nv23_8M.log`).

**What this does and does not show:** this **supports, but does not prove**, that the same mechanism caused the original GPU 0↔2 NCCL
bug.
* Same configuration, a documented failure mode for exactly that configuration, and a measured failure on another same-socket pair.
* But GPU 0↔2 itself was never tested. The original bug is not backed by a log in this repo. And I have no kernel IOMMU
  fault log tying any failure to the IOMMU.

## 5. What would prove it (needs an administrator; not attempted)

1. Read the kernel log around 2026-11-12 03:20:15 EST, when `/var/log/kern.log` was last written
   (`followup/raw/gpu4/kern_log_stat.txt`), for `AMD-Vi ... IO_PAGE_FAULT` events naming `<pci:gpu4>` or `<pci:gpu3>`, and NVIDIA `Xid` lines.
2. In a maintenance window, put the GPU groups in passthrough (`iommu=pt` on the kernel command line, or `identity` group type with the
   driver unbound). Then rerun one short correctness check, such as `nccl-tests -c 1` between GPUs 0 and 2. If it passes there and fails
   in `DMA-FQ` mode, the cause is proven.

## 6. Limits of this evidence

* No root: ACS bits, `dmesg` and the kernel log were unreadable (§2, `followup/raw/gpu4/kernel_log_access.txt`).
* No new P2P runs (by design): the only direct corruption evidence is still the single 3↔4 probe run.
* The GPU 0↔2 bug was not reproduced here.
* I fetched the NVIDIA and kernel pages with `curl`. The pages
  are saved under `env/docs_fetch/` with HTTP status and time in `fetch_times.txt`.
