#!/bin/bash
# WARNING: these cases put device-buffer traffic on PCIe peer-to-peer between non-NVLink GPUs, which corrupted data
# on this machine (see docs/p2p_corruption.md). Do not run them on a shared machine.
# Usage: diag_hang.sh run <name> <gpus> <mode> <bench> <size> [extra OMB args]; output in raw/diag/<name>.log
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"; source scripts/log.sh
export LD_LIBRARY_PATH=$HOME/opt/cuda-driver-link:/usr/local/cuda-12.4/lib64 UCX_WARN_UNUSED_ENV_VARS=n
MPX=$HOME/opt/mvapich-4.1-cuda/bin/mpiexec
OMBD=$HOME/opt/omb-7.5.3-mvapich-4.1/libexec/osu-micro-benchmarks/mpi/pt2pt
mkdir -p raw/diag
run() {  # name gpus mode bench size extra...
  local name=$1 gpus=$2 mode=$3 bench=$4 size=$5; shift 5
  local envs="UCX_TLS=self,sm,cuda"
  [ "$mode" = ucxipc ] && envs="MPIR_CVAR_NOLOCAL=1 UCX_TLS=self,sm,cuda_copy,cuda_ipc"
  [ "$mode" = ucxnoipc ] && envs="MPIR_CVAR_NOLOCAL=1 UCX_TLS=self,sm,cuda_copy"
  [ "$mode" = mvnoipc ] && envs="UCX_TLS=self,sm,cuda MPIR_CVAR_CH4_IPC_GPU_P2P_THRESHOLD=2147483647 MPIR_CVAR_CH4_IPC_MAP_REPEAT_ADDR=0"
  local cmd="env $envs GPU_LIST=$gpus timeout 40 $MPX -n 2 scripts/gpu_bind.sh $OMBD/$bench -d cuda -m $size:$size $* D D"
  local t0=$(date +%s.%N)
  { echo "# case: $name"; echo "# start: $(date '+%F %T %Z')"; echo "# cmd: $cmd"; } > raw/diag/$name.log
  L bash -c "$cmd >> raw/diag/$name.log 2>&1"; local rc=$?
  local dt=$(echo "$(date +%s.%N) - $t0" | bc)
  echo "# exit: $rc  wall_s: $dt" >> raw/diag/$name.log
  pkill -u $USER -f "$OMBD/$bench" 2>/dev/null; sleep 1
  printf '%-34s rc=%-4s %6.1fs  %s\n' "$name" "$rc" "$dt" "$(grep -v '^#\|Warning\|multiple\|get_local' raw/diag/$name.log | tail -1)"
}
"$@"
