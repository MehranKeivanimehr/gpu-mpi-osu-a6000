# Source this, then prefix commands with L to log them (with date and cwd) to followup/commands.log.
# For pipelines/redirections: L bash -c '<pipeline>'.  note "<text>" appends a NOTE line.
FREPO="${FREPO:-$HOME/mpi-osu-a6000/followup}"
L() {
  printf '[%s] (cwd=%s) %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$PWD" "$*" >> "$FREPO/commands.log"
  "$@"
}
note() { printf '[%s] NOTE: %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*" >> "$FREPO/commands.log"; }
