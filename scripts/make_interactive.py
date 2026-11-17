#!/usr/bin/env python3
"""Write docs/interactive.html: one self-contained Plotly chart of the data/ CSVs (plotly.js and data inlined).

Usage: make_interactive.py [--out FILE]
"""
import json
import os
import sys

import pandas as pd
import plotly

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = f"{REPO}/data"
OUT = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else f"{REPO}/docs/interactive.html"
PLOTLY_JS = os.path.join(os.path.dirname(plotly.__file__), "package_data", "plotly.min.js")

P2P_PATHS = {
    ("nv23", "mv"): "GPUs 2-3 (NVLink), MVAPICH default path",
    ("nv23", "ucxipc"): "GPUs 2-3 (NVLink), UCX with CUDA IPC",
    ("nv23", "ucxnoipc"): "GPUs 2-3 (NVLink), UCX staged through host",
    ("node56", "ucxnoipc"): "GPUs 5-6 (same socket), UCX staged through host",
    ("sys36", "ucxnoipc"): "GPUs 3-6 (across sockets), UCX staged through host",
}
COLL_PATHS = {"g2_nv23": "2 GPUs: 2-3 (NVLink)", "g4_nv23_5_6": "4 GPUs: 2, 3, 5, 6 (host staged across pairs)"}
POOLS = {"dev_lmem16M": "pool 16 MiB", "dev_default": "pool 32 MiB (default)", "dev_lmem64M": "pool 64 MiB",
         "dev_lmem128M": "pool 128 MiB", "host_lmem128M": "host buffers, pool 128 MiB"}


def triple(g, med, lo, hi, scale=1.0):
    return {"med": list(g[med] * scale), "min": list(g[lo] * scale), "max": list(g[hi] * scale)}


def build():
    bench = {}
    p = pd.read_csv(f"{DATA}/p2p_summary.csv")
    for b, metric in (("osu_latency", "time"), ("osu_bw", "bw"), ("osu_bibw", "bw")):
        paths = {}
        for (path, mode), label in P2P_PATHS.items():
            g = p[(p.bench == b) & (p.path == path) & (p["mode"] == mode)].sort_values("size")
            if len(g):
                s = {"name": "MVAPICH 4.1 (OMB)", "x": list(g["size"])}
                s[metric] = triple(g, "median", "vmin", "vmax", 1.0 if metric == "time" else 1e-3)
                paths[label] = [s]
        bench[b] = {"paths": paths, "xlog": True, "xunit": "bytes",
                    "time_label": "One-way latency (µs)", "bw_label": "Bandwidth (GB/s, 10^9 bytes/s)"}
    m = pd.read_csv(f"{DATA}/coll_mpi_summary.csv")
    n = pd.read_csv(f"{DATA}/coll_nccl_summary.csv")
    for kind in ("allreduce", "allgather", "reduce_scatter"):
        paths = {}
        for path, label in COLL_PATHS.items():
            series = []
            for df, name in ((m, "MVAPICH 4.1 (OMB)"), (n, "NCCL 2.21.5 (nccl-tests)")):
                g = df[(df.kind == kind) & (df.path == path)].sort_values("S")
                series.append({"name": name, "x": list(g.S), "time": triple(g, "t_median", "t_vmin", "t_vmax"),
                               "bw": triple(g, "busbw_median", "busbw_vmin", "busbw_vmax")})
            paths[label] = series
        bench[kind] = {"paths": paths, "xlog": True, "xunit": "bytes (nccl-tests convention)",
                       "time_label": "Time per call (µs)", "bw_label": "Bus bandwidth (GB/s)"}
    k = pd.read_csv(f"{DATA}/allreduce_knobs.csv")
    series = []
    for cid, name in POOLS.items():
        g = k[k.config == cid].sort_values("size_mib")
        series.append({"name": name, "x": list(g.size_mib),
                       "time": triple(g, "t_median_us", "t_min_us", "t_max_us", 1e-3),
                       "bw": triple(g, "busbw_GBps_median", "busbw_GBps_min", "busbw_GBps_max")})
    bench["allreduce, 24 to 128 MiB"] = {"paths": {"GPUs 2-3 (NVLink), MPIR_CVAR_LMEM_POOL_SIZE varied": series},
                                         "xlog": False, "xunit": "MiB, MPI_FLOAT",
                                         "time_label": "Time per call (ms)", "bw_label": "Bus bandwidth (GB/s)"}
    return bench


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GPU MPI benchmarks</title>
<style>
  :root { --ink: #222; --muted: #555; --line: #ddd; --bg: #fff; }
  body { margin: 0; background: var(--bg); color: var(--ink); font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
  main { max-width: 980px; margin: 0 auto; padding: 20px 16px 40px; }
  h1 { font-size: 20px; font-weight: 600; margin: 0 0 4px; }
  p { margin: 4px 0 12px; color: var(--muted); }
  .controls { display: flex; flex-wrap: wrap; gap: 12px 20px; align-items: end; margin: 12px 0; }
  label { display: flex; flex-direction: column; font-size: 13px; color: var(--muted); gap: 4px; }
  select { font: inherit; padding: 4px 6px; border: 1px solid var(--line); border-radius: 4px; background: #fff; color: var(--ink); max-width: 100%; }
  .metric { display: flex; gap: 12px; font-size: 14px; color: var(--ink); }
  .metric label { flex-direction: row; align-items: center; color: var(--ink); font-size: 14px; }
  #chart { width: 100%; height: 480px; }
  .note { font-size: 13px; }
</style>
</head>
<body>
<main>
<h1>MVAPICH 4.1 + OSU Micro-Benchmarks and NCCL on RTX A6000 GPUs</h1>
<p>Pick a benchmark and a GPU path. Lines are medians of 5 runs; the shaded band is the minimum to maximum of the run medians.</p>
<div class="controls">
  <label>Benchmark <select id="bench"></select></label>
  <label>GPUs and path <select id="path"></select></label>
  <div class="metric" id="metric">
    <label><input type="radio" name="metric" value="time"> latency or time</label>
    <label><input type="radio" name="metric" value="bw"> bandwidth</label>
  </div>
</div>
<div id="chart"></div>
<p class="note">Data: the CSV files in data/ of this repository. Sizes for collectives follow the nccl-tests convention.</p>
</main>
<script>__PLOTLY__</script>
<script>
const DATA = __DATA__;
const COLORS = ["#1f4e8c", "#e0702f", "#5592d8", "#0f2a55", "#8a8a8a"];
const DASH = ["solid", "dash", "dot", "dashdot", "longdash"];
const benchSel = document.getElementById("bench"), pathSel = document.getElementById("path");
const radios = Array.from(document.querySelectorAll('input[name="metric"]'));
Object.keys(DATA).forEach(b => benchSel.add(new Option(b, b)));

function sizeLabel(v, unit) {
  if (!unit.startsWith("bytes")) return v + " MiB";
  const u = [["GiB", 2 ** 30], ["MiB", 2 ** 20], ["KiB", 2 ** 10]];
  for (const [n, s] of u) if (v >= s) return (v / s) + " " + n;
  return v + " B";
}
function fillPaths() {
  const b = DATA[benchSel.value];
  pathSel.innerHTML = "";
  Object.keys(b.paths).forEach(p => pathSel.add(new Option(p, p)));
}
function availableMetrics() {
  const s = DATA[benchSel.value].paths[pathSel.value];
  return ["time", "bw"].filter(m => s.some(x => x[m]));
}
function syncRadios() {
  const avail = availableMetrics();
  radios.forEach(r => { r.disabled = !avail.includes(r.value); });
  if (!radios.some(r => r.checked && !r.disabled)) radios.find(r => !r.disabled).checked = true;
}
function draw() {
  const b = DATA[benchSel.value], series = b.paths[pathSel.value];
  const metric = radios.find(r => r.checked).value;
  const traces = [];
  series.forEach((s, i) => {
    const d = s[metric];
    if (!d) return;
    const c = COLORS[i % COLORS.length];
    traces.push({x: s.x.concat(s.x.slice().reverse()), y: d.max.concat(d.min.slice().reverse()), fill: "toself",
      fillcolor: c + "26", line: {width: 0}, hoverinfo: "skip", showlegend: false, type: "scatter"});
    traces.push({x: s.x, y: d.med, name: s.name, type: "scatter", mode: "lines+markers",
      line: {color: c, width: 2, dash: DASH[i % DASH.length]}, marker: {size: 6, color: c, line: {color: "#fff", width: 1}},
      customdata: s.x.map((x, j) => [sizeLabel(x, b.xunit), d.min[j], d.max[j]]),
      hovertemplate: "%{customdata[0]}<br>median %{y:.4g}<br>min %{customdata[1]:.4g}, max %{customdata[2]:.4g}<extra>" + s.name + "</extra>"});
  });
  const layout = {
    margin: {l: 70, r: 20, t: 10, b: 60}, paper_bgcolor: "#fff", plot_bgcolor: "#fff",
    font: {family: "system-ui, sans-serif", size: 13, color: "#222"},
    xaxis: {title: {text: "Size (" + b.xunit + ")"}, type: b.xlog ? "log" : "linear", gridcolor: "#eee", zeroline: false},
    yaxis: {title: {text: metric === "time" ? b.time_label : b.bw_label}, type: b.xlog ? "log" : "linear", gridcolor: "#eee", zeroline: false},
    legend: {orientation: "h", y: -0.2}, hovermode: "closest"
  };
  if (b.xlog) {
    const ticks = [1, 32, 1024, 32768, 1048576, 33554432, 1073741824];
    layout.xaxis.tickvals = ticks;
    layout.xaxis.ticktext = ticks.map(t => sizeLabel(t, "bytes"));
  }
  Plotly.react("chart", traces, layout, {displaylogo: false, responsive: true});
}
benchSel.addEventListener("change", () => { fillPaths(); syncRadios(); draw(); });
pathSel.addEventListener("change", () => { syncRadios(); draw(); });
radios.forEach(r => r.addEventListener("change", draw));
benchSel.value = "allreduce";
fillPaths(); syncRadios(); draw();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    js = open(PLOTLY_JS, encoding="utf-8").read()
    data = json.dumps(build(), separators=(",", ":"))
    html = PAGE.replace("__DATA__", data).replace("__PLOTLY__", js.replace("</script", "<\\/script"))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w", encoding="utf-8").write(html)
    print(f"wrote {OUT} ({len(html) / 1e6:.1f} MB)")
