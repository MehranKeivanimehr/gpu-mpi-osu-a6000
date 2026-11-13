"""Copy the README.envvar blocks of selected control variables (MVAPICH 4.1 source tree) into one file.
Usage: envvar_excerpt.py <README.envvar> <out> VAR [VAR ...]"""
import sys
src, out, names = sys.argv[1], sys.argv[2], sys.argv[3:]
lines = open(src, errors="replace").read().splitlines()
blocks = {}
for i, l in enumerate(lines):
    if l.strip() in names and not l.startswith(" "):
        j = i + 1
        while j < len(lines) and (lines[j].startswith(" ") or lines[j].strip() == "") and not (
                lines[j].strip() == "" and j + 1 < len(lines) and lines[j + 1].startswith("MPIR_CVAR")):
            j += 1
        blocks[l.strip()] = "\n".join(lines[i:j]).rstrip()
with open(out, "w") as f:
    f.write(f"# Excerpt of {src} (MVAPICH 4.1 source tree, the installed version's own documentation)\n\n")
    for n in names:
        f.write((blocks.get(n) or f"{n}\n    NOT FOUND in README.envvar") + "\n\n")
print(len(blocks), "of", len(names), "found")
