#!/bin/bash
# gpu_bind.sh -- per-rank GPU + NUMA binding wrapper.
#
# Usage (one GPU per rank):
#   GPU_LIST=2,3 mpirun -np 2 scripts/gpu_bind.sh ./osu_latency -d cuda D D
#
# The rank's local index (from whichever variable the launcher exports) picks
# GPU_LIST[local_rank]. Then:
#   GPU_BIND_MODE=single (default): CUDA_VISIBLE_DEVICES=<that one GPU>
#   GPU_BIND_MODE=all              : CUDA_VISIBLE_DEVICES=<whole GPU_LIST>; the app
#                                    picks device[local_rank] itself (nccl-tests does this).
# The process is pinned with numactl to the NUMA node that owns the GPU's PCIe
# root (read from /sys/bus/pci/devices/<bus id>/numa_node), CPUs and memory.
# Set GPU_BIND_VERBOSE=1 to print the mapping to stderr.
set -euo pipefail

LR="${OMPI_COMM_WORLD_LOCAL_RANK:-${MPI_LOCALRANKID:-${MV2_COMM_WORLD_LOCAL_RANK:-${PMI_LOCAL_RANK:-${SLURM_LOCALID:-}}}}}"
LR_SRC=local
if [ -z "$LR" ] && [ -n "${PMI_RANK:-}" ]; then
  # MVAPICH/MPICH 'gforker' launcher (used here because it opens no TCP listener) exports only
  # PMI_RANK/PMI_SIZE. gforker is single-node only, so global rank == local rank.
  LR="$PMI_RANK"; LR_SRC=PMI_RANK
fi
if [ -z "$LR" ]; then
  echo "gpu_bind.sh: no local-rank variable found (OMPI_COMM_WORLD_LOCAL_RANK / MPI_LOCALRANKID / PMI_RANK ...)" >&2
  exit 2
fi
: "${GPU_LIST:?gpu_bind.sh: set GPU_LIST, e.g. GPU_LIST=2,3}"
IFS=, read -r -a G <<< "$GPU_LIST"
if [ "$LR" -ge "${#G[@]}" ]; then
  echo "gpu_bind.sh: local rank $LR but GPU_LIST has only ${#G[@]} entries" >&2
  exit 2
fi
GPU="${G[$LR]}"

# GPU index -> PCI bus id (fixed on this machine; see env/nvidia-smi_query.csv).
BUS=(<pci:gpu0> <pci:gpu1> <pci:gpu2> <pci:gpu3> <pci:gpu4> <pci:gpu5> <pci:gpu6> <pci:gpu7>)
NODE=$(cat "/sys/bus/pci/devices/${BUS[$GPU]}/numa_node")

if [ "${GPU_BIND_MODE:-single}" = "all" ]; then
  export CUDA_VISIBLE_DEVICES="$GPU_LIST"
else
  export CUDA_VISIBLE_DEVICES="$GPU"
fi
export GPU_BIND_GPU="$GPU" GPU_BIND_NUMA="$NODE"

if [ "${GPU_BIND_VERBOSE:-0}" = "1" ]; then
  echo "gpu_bind: host=$(hostname) local_rank=$LR(from $LR_SRC) gpu=$GPU bus=${BUS[$GPU]} numa=$NODE CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
fi
exec numactl --cpunodebind="$NODE" --membind="$NODE" "$@"
