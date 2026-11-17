# Source this file, then prefix commands with L to append them to commands.log (not tracked in git).
# For pipelines or redirections: L bash -c '<pipeline>'
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
L() {
  printf '[%s] (cwd=%s) %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$PWD" "$*" >> "$REPO/commands.log"
  "$@"
}
note() { printf '[%s] NOTE: %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*" >> "$REPO/commands.log"; }
