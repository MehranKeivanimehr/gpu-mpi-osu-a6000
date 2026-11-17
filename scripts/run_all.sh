#!/bin/bash
# The order in which I ran the first study's phases. Each phase shuffles its (config, rep) list and waits for idle GPUs.
set -e
cd "$(dirname "$0")/.."
PY=python3
$PY scripts/campaign.py validate --reps 2 --any-rep   # OMB -c on every config, one complete run each
$PY scripts/campaign.py p2p --reps 5
$PY scripts/campaign.py ucx --reps 5
$PY scripts/campaign.py ucxlog --reps 1                # UCX transport logging
$PY scripts/campaign.py coll --reps 5
$PY scripts/campaign.py nccl --reps 5
python3 -I scripts/analyze.py
