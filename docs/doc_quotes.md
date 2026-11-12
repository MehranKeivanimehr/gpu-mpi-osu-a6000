# DOCUMENTED sources (verbatim quotes; fetched 2026-11-12)

## NVIDIA NCCL user guide, "GPU troubleshooting"
URL: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html
(section "Linux bare-metal IOMMU and PCIe peer-to-peer", and "PCI Access Control Services (ACS)")
Raw text extract made with curl + html-strip: docs/_nccl_extract.txt

> On Linux bare-metal systems, however, CUDA and the NVIDIA driver stack do not support IOMMU-enabled PCIe
> peer-to-peer memory transfer. The CUDA Programming Guide states that the IOMMU must be disabled on Linux
> bare-metal systems to prevent silent device memory corruption.

> a positive nvidia-smi topo -p2p p result is useful, but it is not a complete system sanity check for this issue.
> On Linux bare-metal PCIe systems, also check the active kernel IOMMU mode.

> On affected bare-metal systems, avoid translated IOMMU mode for GPU PCIe P2P. Disable IOMMU translation for the GPU path.

> IO virtualization (also known as VT-d or IOMMU) can interfere with GPU Direct by redirecting all PCI
> point-to-point traffic to the CPU root complex, causing a significant performance reduction or even a hang.

## This machine (MEASURED, env/iommu_pcie.txt)
* /sys/class/iommu: ivhd0..ivhd7 (AMD IOMMU present); /proc/cmdline has no iommu=pt / amd_iommu=off.
* Each GPU's IOMMU group domain type: DMA-FQ (translated DMA with flush queue), i.e. translated mode.

## MVAPICH (mvapich.cse.ohio-state.edu/downloads/, read 2026-11-12)
* MVAPICH 4.1 (09/18/25), "integrated and ABI compatible with MPICH-4.3.0", source tarball
  https://mvapich.cse.ohio-state.edu/download/mvapich/mv2/mvapich-4.1.tar.gz . Changelog (MVP-4.1-CHANGELOG.txt) for 4.0:
  "Basic support for major GPUs (from NVIDIA, AMD, and Intel)".
* MVAPICH-Plus 5.0.1 (07/24/2026): distributed as RHEL RPMs ("These RPMs contain the MVAPICH-Plus software on the
  corresponding distro"); for Ubuntu the README points to rpm2cpio extraction. Not used here (binary build against
  RHEL9 + DOCA 3.3; this node is Ubuntu 24.04 without DOCA/IB).
* MVAPICH 5.0.0b (11/03/26) is a beta; not used.

## Open MPI (docs.open-mpi.org/en/v5.0.x/tuning-apps/networking/cuda.html)
* Verify: `ompi_info --parsable --all | grep mpi_built_with_cuda_support:value` -> `...:value:true`.

## OSU Micro-Benchmarks (mvapich.cse.ohio-state.edu/benchmarks/)
* OMB 7.5.3 (11/03/26) latest stable; 8.0rc is a release candidate. Configure: `--enable-cuda --with-cuda-include=... --with-cuda-libpath=...`.
* Flags used, from OMB 7.5.3 source c/util/osu_util_options.h (copy: raw/verify/omb_7.5.3_osu_util_options.h):
  `-d cuda` (device buffers), `-c` ("Enable validation"), `-W` window, `-i`/`-x` iterations/warmup, `-f` full stats.

## nccl-tests (github.com/NVIDIA/nccl-tests, doc/PERFORMANCE.md)
* busbw factors: AllReduce 2(n-1)/n, ReduceScatter (n-1)/n, AllGather (n-1)/n; algbw = S/t.
* `-c, --check <n>`: "perform count iterations, checking correctness of results on each iteration".
