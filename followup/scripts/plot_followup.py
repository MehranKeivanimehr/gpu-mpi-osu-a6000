#!/usr/bin/env python3
"""Plots for ALLREDUCE_CLIFF.md, from results/summary_<phase>.csv (written by analyze_followup.py).
Palette: reference categorical slots 1-5 (validated with the dataviz validator, light mode: all checks pass;
contrast WARN for slots 3-5 -> every line is direct-labelled and every number is also in a table)."""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FREPO = os.path.expanduser("~/mpi-osu-a6000/followup")
OUT = f"{FREPO}/results/plots"
os.makedirs(OUT, exist_ok=True)
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
MARKERS = ["o", "s", "^", "D", "v"]


def load(phase, config):
    rows = [r for r in csv.DictReader(open(f"{FREPO}/results/summary_{phase}.csv")) if r["config"] == config]
    rows.sort(key=lambda r: int(r["size_mib"]))
    x = [int(r["size_mib"]) for r in rows]
    y = [float(r["busbw_GBps_median"]) for r in rows]
    lo = [y[i] - float(r["busbw_GBps_min"]) for i, r in enumerate(rows)]
    hi = [float(r["busbw_GBps_max"]) - y[i] for i, r in enumerate(rows)]
    return x, y, lo, hi


def style(ax, title, subtitle):
    ax.set_facecolor(SURF)
    ax.figure.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("allreduce message size S (MiB), MPI_FLOAT sum, 2 ranks on NVLink pair 2-3", color=INK2, fontsize=9)
    ax.set_ylabel("bus bandwidth (GB/s), median of 5 runs", color=INK2, fontsize=9)
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=22)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=INK2, fontsize=8.5)


def series(ax, i, x, y, lo, hi, label, dy=0.0, direct=True):
    ax.errorbar(x, y, yerr=[lo, hi], color=SLOTS[i], marker=MARKERS[i], markersize=6, linewidth=2, capsize=2,
                elinewidth=1, label=label, markeredgecolor=SURF, markeredgewidth=1)
    if direct:
        ax.annotate(label, (x[-1], y[-1]), xytext=(6, dy), textcoords="offset points", color=INK, fontsize=8.5,
                    va="center")


# 1. device vs host buffers, default settings
fig, ax = plt.subplots(figsize=(8.4, 4.6))
style(ax, "MVAPICH 4.1 allreduce: the cliff sits between 32 and 36 MiB, for device and host buffers",
      "min-max of the 5 run medians as error bars; raw/sweep/*.log")
for i, (cfgid, lab) in enumerate([("dev_default", "device buffers"), ("host_default", "host buffers")]):
    series(ax, i, *load("sweep", cfgid), lab)
ax.axvline(32, color=INK2, linestyle="--", linewidth=1)
ax.text(32.8, 10.6, "MPIR_CVAR_LMEM_POOL_SIZE\ndefault = 32 MiB", color=INK2, fontsize=8)
ax.set_xlim(20, 150)
ax.set_ylim(0, 12)
ax.legend(frameon=False, fontsize=8.5, loc="upper right")
fig.tight_layout()
fig.savefig(f"{OUT}/cliff_device_vs_host.png", dpi=150)

# 2. cliff position follows the temp-buffer pool size
fig, ax = plt.subplots(figsize=(8.4, 4.8))
style(ax, "The cliff moves with MPIR_CVAR_LMEM_POOL_SIZE (device buffers)",
      "16/32/64/128 MiB pools: raw/knobs/*.log; glibc mmap_max=0 + trim_threshold=1 GiB (exploratory): raw/glibc/*.log")
sets = [("knobs", "dev_lmem16M", "pool 16 MiB"), ("knobs", "dev_default", "pool 32 MiB (default)"),
        ("knobs", "dev_lmem64M", "pool 64 MiB"), ("knobs", "dev_lmem128M", "pool 128 MiB"),
        ("glibc", "dev_glibc_mmapmax0_trim1G", "default pool + glibc no-mmap/no-trim")]
offs = [0, 0, 0, 7, -7]
for i, (ph, cid, lab) in enumerate(sets):
    series(ax, i, *load(ph, cid), lab, dy=offs[i], direct=i >= 3)
ax.annotate("pools 16, 32, 64 MiB", (128, 2.42), xytext=(6, 0), textcoords="offset points", color=INK,
            fontsize=8.5, va="center")
ax.set_xlim(20, 175)
ax.set_ylim(0, 12)
ax.legend(frameon=False, fontsize=8, loc="center right", bbox_to_anchor=(1.0, 0.45))
fig.tight_layout()
fig.savefig(f"{OUT}/cliff_vs_pool_size.png", dpi=150)
print("wrote", OUT)
