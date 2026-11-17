#!/bin/bash
# Every 15 s while campaign.py runs, record any TCP/UDP listening socket owned by a benchmark process.
out="$(cd "$(dirname "$0")/.." && pwd)/raw/verify/listen_during_campaign.txt"
echo "# started $(date '+%F %T %Z'); format: timestamp | ss -ltnup lines for osu_/nccl-tests/mpiexec/numactl processes (empty = none)" >> $out
while pgrep -f "[c]ampaign.py" >/dev/null || [ "${1:-}" = "--once" ]; do
  l=$(ss -ltnup 2>/dev/null | grep -E 'osu_|_perf"|mpiexec|numactl')
  b=$(pgrep -fc "[o]mb-7.5.3-mvapich|[n]ccl-tests-2.21.1")
  echo "$(date '+%T') bench_procs=$b | ${l:-none}" >> $out
  [ "${1:-}" = "--once" ] && break
  sleep 15
done
echo "# stopped $(date '+%F %T %Z')" >> $out
