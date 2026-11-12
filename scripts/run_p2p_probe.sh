#!/bin/bash
# Run src/p2p_probe on GPU pairs given as "a,b" args. Only those two GPUs are made visible.
cd $HOME/mpi-osu-a6000; source scripts/log.sh; mkdir -p raw/p2p_probe
for pr in "$@"; do
  for part in dma ipc; do
    f=raw/p2p_probe/probe_${pr/,/_}_${part}.log
    arg=""; [ $part = ipc ] && arg=ipc
    { echo "# start: $(date '+%F %T %Z')"; echo "# CUDA_VISIBLE_DEVICES=$pr ./src/p2p_probe 0 1 $arg"; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | sed 's/^/#   /'; } > $f
    t0=$(date +%s.%N)
    L bash -c "CUDA_VISIBLE_DEVICES=$pr timeout 90 ./src/p2p_probe 0 1 $arg >> $f 2>&1"; rc=$?
    echo "# exit: $rc wall_s: $(echo "$(date +%s.%N) - $t0" | bc)" >> $f
    echo "== $f (rc=$rc)"; grep -v "^#" $f
  done
done
