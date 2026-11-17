#!/usr/bin/env python3
"""Draw figures/*.png and figures/allreduce_pool.gif from the CSVs in data/.

Usage: plot.py [--out DIR]
"""
import io
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = f"{REPO}/data"
OUT = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else f"{REPO}/figures"

BLUE, ORANGE = "#1f4e8c", "#e0702f"
POOL_COLORS = {16: "#a3c9ee", 32: "#5592d8", 64: "#24589f", 128: "#0f2a55"}
POOL_MARKERS = {16: "o", 32: "s", 64: "^", 128: "D"}
POOL_CONFIG = {16: "dev_lmem16M", 32: "dev_default", 64: "dev_lmem64M", 128: "dev_lmem128M"}
INK, MUTED, GHOST = "#222222", "#555555", "#d4d4d4"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.size": 10, "axes.labelsize": 10, "axes.edgecolor": "#888888", "axes.labelcolor": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e8e8e8", "grid.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED,
    "legend.frameon": False, "legend.fontsize": 9, "lines.linewidth": 1.8, "lines.markersize": 4.5,
    "lines.markeredgecolor": "white", "lines.markeredgewidth": 0.8,
})


def size_label(b, _=None):
    for unit, n in (("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10)):
        if b >= n:
            return f"{b / n:g} {unit}"
    return f"{b:g} B"


def log2_size_axis(ax, lo, hi):
    ax.set_xscale("log", base=2)
    ax.set_xlim(lo, hi)
    ax.xaxis.set_major_locator(LogLocator(base=2, numticks=40))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: size_label(v) if v in TICKS else ""))


TICKS = [1, 1 << 5, 1 << 10, 1 << 15, 1 << 20, 1 << 25, 1 << 30]


def osu_bw():
    p = pd.read_csv(f"{DATA}/p2p_summary.csv")
    p = p[p.bench == "osu_bw"]
    series = [  # (path, mode, label, colour, linestyle, marker)
        ("nv23", "mv", "NVLink pair, MVAPICH default (CUDA IPC)", BLUE, "-", "o"),
        ("nv23", "ucxnoipc", "NVLink pair, UCX staged through host", ORANGE, "-", "s"),
        ("node56", "ucxnoipc", "same socket, UCX staged through host", ORANGE, "--", "^"),
        ("sys36", "ucxnoipc", "across sockets, UCX staged through host", ORANGE, ":", "D"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for path, mode, label, col, ls, mk in series:
        g = p[(p.path == path) & (p["mode"] == mode)].sort_values("size")
        ax.fill_between(g["size"], g.vmin, g.vmax, color=col, alpha=0.15, linewidth=0)
        ax.plot(g["size"], g["median"], color=col, ls=ls, marker=mk, label=label)
    log2_size_axis(ax, 1, 1 << 30)
    ax.set_xlabel("Message size")
    ax.set_ylabel("Bandwidth (MB/s, 1 MB = 10$^6$ bytes)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_ylim(0, 60000)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/osu_bw.png", dpi=150)
    plt.close(fig)


def pool_frame(ax, k, highlight=None):
    for pool, cid in POOL_CONFIG.items():
        g = k[k.config == cid].sort_values("size_mib")
        if highlight is None:
            col, z, lab = POOL_COLORS[pool], 3, f"pool {pool} MiB" + (" (default)" if pool == 32 else "")
        elif pool == highlight:
            col, z, lab = BLUE, 3, None
        else:
            col, z, lab = GHOST, 1, None
        yerr = [(g.t_median_us - g.t_min_us) / 1e3, (g.t_max_us - g.t_median_us) / 1e3]
        ax.errorbar(g.size_mib, g.t_median_us / 1e3, yerr=yerr, color=col, marker=POOL_MARKERS[pool],
                    capsize=0, elinewidth=1, zorder=z, label=lab)
    ax.set_xlim(20, 132)
    ax.set_ylim(0, 62)
    ax.set_xticks([24, 32, 48, 64, 96, 128])
    ax.set_xlabel("Allreduce size (MiB, MPI_FLOAT, device buffers)")
    ax.set_ylabel("Time per call (ms)")


def allreduce_pool():
    k = pd.read_csv(f"{DATA}/allreduce_knobs.csv")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    pool_frame(ax, k)
    ax.legend(loc="upper left", title="MPIR_CVAR_LMEM_POOL_SIZE", title_fontsize=9, alignment="left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/allreduce_pool.png", dpi=150)
    plt.close(fig)


def allreduce_pool_gif():
    k = pd.read_csv(f"{DATA}/allreduce_knobs.csv")
    frames = []
    for pool in POOL_CONFIG:
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        pool_frame(ax, k, highlight=pool)
        ax.axvline(pool, color=MUTED, ls="--", lw=1, zorder=0)
        ax.text(pool + 1.5, 59, f"pool = {pool} MiB" + (" (default)" if pool == 32 else ""), color=INK,
                va="top", fontsize=10)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, dpi=100, format="png")
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert("RGB").quantize(colors=64))
    frames[0].save(f"{OUT}/allreduce_pool.gif", save_all=True, append_images=frames[1:], duration=1500, loop=0,
                   optimize=True)


def mpi_vs_nccl():
    m = pd.read_csv(f"{DATA}/coll_mpi_summary.csv")
    n = pd.read_csv(f"{DATA}/coll_nccl_summary.csv")
    m = m[(m.path == "g2_nv23") & (m["mode"] == "mv")]
    n = n[(n.path == "g2_nv23") & (n.p2p_level == "default")]
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.9), sharey=True)
    for ax, kind in zip(axs, ["allreduce", "allgather"]):
        for df, label, col, ls, mk in ((m, "MVAPICH 4.1 (OMB)", BLUE, "-", "o"), (n, "NCCL 2.21.5 (nccl-tests)", ORANGE, "--", "s")):
            g = df[df.kind == kind].sort_values("S")
            ax.fill_between(g.S, g.busbw_vmin, g.busbw_vmax, color=col, alpha=0.15, linewidth=0)
            ax.plot(g.S, g.busbw_median, color=col, ls=ls, marker=mk, label=label)
        log2_size_axis(ax, 2, 1 << 30)
        ax.set_yscale("log")
        ax.set_ylim(3e-5, 100)
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xlabel("Size S (bytes in the nccl-tests convention)")
        ax.set_ylabel(f"{kind.capitalize()} bus bandwidth (GB/s)")
        ax.yaxis.set_tick_params(labelleft=True)
    axs[0].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{OUT}/mpi_vs_nccl.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    osu_bw()
    allreduce_pool()
    allreduce_pool_gif()
    mpi_vs_nccl()
    print("figures written to", OUT)
