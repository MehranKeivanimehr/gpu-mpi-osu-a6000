#!/bin/bash
# Per-rank wrapper: pick GPU_LIST[local rank], set CUDA_VISIBLE_DEVICES and bind CPU and memory to that GPU's NUMA node.
# Usage: GPU_LIST=2,3 mpiexec -n 2 scripts/gpu_bind.sh <program> [args]
# GPU_BIND_MODE=all keeps the whole list visible (nccl-tests picks device[local rank] itself).
set -euo pipefail

LR="${OMPI_COMM_WORLD_LOCAL_RANK:-${MPI_LOCALRANKID:-${MV2_COMM_WORLD_LOCAL_RANK:-${PMI_LOCAL_RANK:-${SLURM_LOCALID:-}}}}}"
LR_SRC=local
if [ -z "$LR" ] && [ -n "${PMI_RANK:-}" ]; then
  # gforker exports only PMI_RANK; it is single-node, so the global rank is the local rank
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

# nvidia-smi prints <pci:gpu2>; sysfs names the device <pci:gpu2>
BUS=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i "$GPU" | tr 'A-F' 'a-f')
BUS="${BUS#0000}"
NODE=$(cat "/sys/bus/pci/devices/$BUS/numa_node")

if [ "${GPU_BIND_MODE:-single}" = "all" ]; then
  export CUDA_VISIBLE_DEVICES="$GPU_LIST"
else
  export CUDA_VISIBLE_DEVICES="$GPU"
fi
export GPU_BIND_GPU="$GPU" GPU_BIND_NUMA="$NODE"

if [ "${GPU_BIND_VERBOSE:-0}" = "1" ]; then
  echo "gpu_bind: local_rank=$LR(from $LR_SRC) gpu=$GPU bus=$BUS numa=$NODE CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
fi
exec numactl --cpunodebind="$NODE" --membind="$NODE" "$@"
