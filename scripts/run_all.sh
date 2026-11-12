#!/bin/bash
# Order in which the campaign phases were run (see commands.log for exact timestamps).
# Every phase shuffles its (config, rep) list with a fixed seed and checks GPU idleness before each run.
# Phases and their configs are defined in scripts/configs.py; check_safe() refuses unsafe configs.
set -e
cd $HOME/mpi-osu-a6000
PY=python3
$PY scripts/campaign.py validate --reps 2 --any-rep   # OMB -c correctness, one complete run per config (budget; see README)
$PY scripts/campaign.py p2p --reps 5          # osu_latency/bw/bibw, NVLink pair (MVAPICH default) + cross pairs (host-staged)
$PY scripts/campaign.py ucx --reps 5          # UCX transport contrast on the NVLink pair: cuda_ipc vs cuda_copy only
$PY scripts/campaign.py ucxlog --reps 1       # UCX_LOG_LEVEL=info + UCX_PROTO_INFO=y: which transport carried the data
$PY scripts/campaign.py coll --reps 5         # osu_allreduce/allgather/reduce_scatter on 2 and 4 GPUs
$PY scripts/campaign.py nccl --reps 5         # nccl-tests on the same GPU sets, NCCL_DEBUG=INFO, -c 1
~/miniconda3/bin/python -I scripts/analyze.py
