# Source this file, then prefix commands with L to log them to commands.log.
# Usage: source ~/mpi-osu-a6000/scripts/log.sh ; L <cmd> [args...]
# For pipelines/redirections: L bash -c '<pipeline>'
REPO="${REPO:-$HOME/mpi-osu-a6000}"
L() {
  printf '[%s] (cwd=%s) %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$PWD" "$*" >> "$REPO/commands.log"
  "$@"
}
note() { printf '[%s] NOTE: %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*" >> "$REPO/commands.log"; }
