#!/bin/bash
# Measurement chain (after validation passed). Phases run back-to-back; each one checks GPU idleness per run.
cd $HOME/mpi-osu-a6000
for ph in p2p ucxlog coll nccl ucx; do
  reps=5; [ $ph = ucxlog ] && reps=1
  echo "=== $(date '+%F %T %Z') phase $ph reps $reps"
  python3 scripts/campaign.py $ph --reps $reps
done
echo "=== $(date '+%F %T %Z') chain done"
