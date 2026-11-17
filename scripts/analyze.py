#!/usr/bin/env python3
"""Parse raw/ logs into the CSVs and markdown tables in data/.

Usage: analyze.py [--out DIR]
"""
import glob, json, os, re, sys
import numpy as np
import pandas as pd
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = f"{REPO}/raw"
RES = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else f"{REPO}/data"
os.makedirs(RES, exist_ok=True)

KiB, MiB, GiB = 1024, 1024 ** 2, 1024 ** 3


def read_header(path):
    h = {}
    with open(path) as f:
        for line in f:
            m = re.match(r"# ([\w.]+): (.*)", line)
            if m:
                h[m.group(1)] = m.group(2).strip()
    return h


def parse_omb(path):
    """Rows: size + numeric columns (+ validation)."""
    h = read_header(path)
    cols, rows = None, []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s.startswith("# Size"):
                cols = re.split(r"\s{2,}", s[2:].strip())
                continue
            if not s or s.startswith("#") or not re.match(r"^\d+\s", s):
                continue
            tok = s.split()
            try:
                size = int(tok[0])
            except ValueError:
                continue
            val = None
            nums = []
            for t in tok[1:]:
                if t in ("Pass", "Fail"):
                    val = t
                else:
                    try:
                        nums.append(float(t))
                    except ValueError:
                        pass
            rows.append((size, nums, val))
    return h, cols, rows


INFO_SEG = re.compile(r"[A-Za-z][\w.-]*:\d+:\d+ \[\d+\] NCCL INFO[^\n]*\n")  # host:pid:tid [rank] NCCL INFO ...
ROW_START = re.compile(r"^\s+(\d+)\s+(\d+)\s+\w+\s+\w+\s+-?\d+")
ALGO = {0: "Tree", 1: "Ring", 2: "CollNetDirect", 3: "CollNetChain", 4: "NVLS", 5: "NVLSTree"}  # NCCL 2.21
PROTO = {0: "LL", 1: "LL128", 2: "Simple"}


def parse_nccl(path):
    """nccl-tests rows (out-of-place / in-place time, algbw, busbw, #wrong) + NCCL INFO facts.
    NCCL's per-call TUNING INFO lines are written to the same stream and can split a result row in two;
    they are cut out (INFO_SEG) before the rows are parsed."""
    h = read_header(path)
    raw = open(path, errors="replace").read()
    info = {"transports": set(), "version": None, "avg_bus_bw": None, "bootstrap": None, "tuning": []}
    m = re.search(r"NCCL version ([\w.+]+)", raw)
    info["version"] = m.group(1) if m else None
    info["transports"] = set(re.findall(r"Channel \d+(?:/\d+)? : \S+ -> \S+ via (\S+)", raw))
    m = re.search(r"NCCL INFO Bootstrap : Using (\S+)", raw)
    info["bootstrap"] = m.group(1) if m else None
    m = re.search(r"# Avg bus bandwidth\s*:\s*([\d.]+)", raw)
    info["avg_bus_bw"] = float(m.group(1)) if m else None
    # algo/proto per size: TUNING lines from rank [0] that follow a row start
    cur = None
    for line in raw.splitlines():
        r0 = ROW_START.match(line)
        if r0:
            cur = int(r0.group(1))
        t = re.search(r"\[0\] NCCL INFO (\d+) Bytes -> Algo (\d+) proto (\d+) time ([\d.]+)", line)
        if t and cur is not None:
            info["tuning"].append((cur, int(t.group(1)), int(t.group(2)), int(t.group(3)), float(t.group(4))))
    clean = INFO_SEG.sub("", raw)
    rows = []
    for line in clean.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        tok = s.split()
        if len(tok) >= 13 and tok[0].isdigit() and tok[1].isdigit():
            try:
                rows.append({"size": int(tok[0]), "count": int(tok[1]), "dtype": tok[2], "redop": tok[3],
                             "oop_time_us": float(tok[5]), "oop_algbw": float(tok[6]),
                             "oop_busbw": float(tok[7]), "oop_wrong": tok[8],
                             "ip_time_us": float(tok[9]), "ip_algbw": float(tok[10]),
                             "ip_busbw": float(tok[11]), "ip_wrong": tok[12]})
            except ValueError:
                pass
    return h, rows, info


def meta_of(h):
    return {k[5:]: v for k, v in h.items() if k.startswith("meta.")}


def load_omb_phase(phase):
    out = []
    for p in sorted(glob.glob(f"{RAW}/{phase}/*.log")):
        h, cols, rows = parse_omb(p)
        m = meta_of(h)
        abnormal = "signaled with" in open(p, errors="replace").read()   # a rank died (gforker may still exit 0)
        if h.get("status") != "OK":
            print(f"  [skip non-OK] {p} status={h.get('status')}")
            continue
        for size, nums, val in rows:
            if not nums:
                continue
            r = {"phase": phase, "config_id": h["config_id"], "rep": int(h["rep"]), "bench": m["bench"],
                 "path": m["path"], "mode": m["mode"], "range": m["range"], "n": int(m["n"]),
                 "size": size, "value": nums[0], "validation": val, "abnormal_exit": abnormal,
                 "log": os.path.relpath(p, REPO)}
            if len(nums) >= 4:  # -f: avg, min, max, iterations
                r.update({"t_min_iter": nums[1], "t_max_iter": nums[2], "iters": nums[3]})
            out.append(r)
    return pd.DataFrame(out)


def load_nccl_phase(phase):
    out, infos = [], []
    for p in sorted(glob.glob(f"{RAW}/{phase}/*.log")):
        h, rows, info = parse_nccl(p)
        m = meta_of(h)
        infos.append({"config_id": h["config_id"], "rep": int(h["rep"]), "status": h.get("status"),
                      "nccl_version": info["version"], "transports": ";".join(sorted(info["transports"])),
                      "bootstrap": info["bootstrap"], "n_tuning_lines": len(info["tuning"]),
                      "avg_bus_bw_reported": info["avg_bus_bw"],
                      "log": os.path.relpath(p, REPO)})
        if h.get("status") != "OK":
            continue
        for r in rows:
            r.update({"phase": phase, "config_id": h["config_id"], "rep": int(h["rep"]), "bench": m["bench"],
                      "set": m["set"], "p2p_level": m["p2p_level"], "n": int(m["n"]),
                      "log": os.path.relpath(p, REPO)})
            out.append(r)
    return pd.DataFrame(out), pd.DataFrame(infos)


def agg(df, keys, col):
    g = df.groupby(keys)[col]
    a = g.agg(median="median", vmin="min", vmax="max", nreps="count",
              q1=lambda x: np.percentile(x, 25), q3=lambda x: np.percentile(x, 75)).reset_index()
    return a


# nccl-tests convention (doc/PERFORMANCE.md): S = total bytes as defined there; busbw = S/t * factor
FACTOR = {"allreduce": lambda n: 2 * (n - 1) / n, "allgather": lambda n: (n - 1) / n,
          "reduce_scatter": lambda n: (n - 1) / n}


def coll_kind(bench):
    b = bench.replace("osu_", "").replace("_perf", "")
    return {"allreduce": "allreduce", "all_reduce": "allreduce", "allgather": "allgather",
            "all_gather": "allgather", "reduce_scatter": "reduce_scatter"}[b]


def omb_S(kind, size, n):
    """Convert an OMB 'Size' to the nccl-tests S.
    osu_allreduce: size = per-rank vector bytes = S.
    osu_allgather: size = per-rank send bytes; nccl-tests S = sendcount*sizeof*n = size*n.
    osu_reduce_scatter: size = per-rank send bytes (recvcounts = size/n); nccl-tests S = recvcount*sizeof*n = size.
    (from OMB 7.5.3 c/mpi/collective/blocking/*.c and nccl-tests doc/PERFORMANCE.md)"""
    return size * n if kind == "allgather" else size


def fit_segment(x, y):
    """Weighted LS fit y = a + b x with weights 1/y^2 (relative error). Returns a, b, se_a, se_b, rel_rms."""
    w = 1.0 / np.maximum(y, 1e-12) ** 2
    W = np.sqrt(w)
    A = np.vstack([np.ones_like(x), x]).T * W[:, None]
    yb = y * W
    coef, res, rk, sv = np.linalg.lstsq(A, yb, rcond=None)
    r = yb - A @ coef
    dof = max(len(x) - 2, 1)
    s2 = float(r @ r) / dof
    try:
        cov = s2 * np.linalg.inv(A.T @ A)
        se = np.sqrt(np.maximum(np.diag(cov), 0))
    except np.linalg.LinAlgError:
        se = np.array([np.nan, np.nan])
    pred = coef[0] + coef[1] * x
    rel = np.sqrt(np.mean(((pred - y) / y) ** 2))
    return coef[0], coef[1], se[0], se[1], rel


def piecewise_fit(sizes, times, max_seg=4, min_pts=4):
    """Exact DP segmentation of the (sorted) curve into K contiguous segments (K <= max_seg, >= min_pts points
    each), each fit with t = a + b*n by weighted LS on relative residuals. K is chosen by BIC:
    N*ln(RSS/N) + (3K-1)*ln(N)  (2 parameters per segment + K-1 breakpoints). Returns list of segments."""
    x = np.asarray(sizes, float)
    y = np.asarray(times, float)
    N = len(x)
    cost = np.full((N, N + 1), np.inf)
    for i in range(N):
        for j in range(i + min_pts, N + 1):
            a, b, _, _, rel = fit_segment(x[i:j], y[i:j])
            cost[i, j] = rel ** 2 * (j - i)
    best = None
    for K in range(1, max_seg + 1):
        D = np.full((K + 1, N + 1), np.inf)
        P = np.zeros((K + 1, N + 1), int)
        D[0, 0] = 0
        for k in range(1, K + 1):
            for j in range(N + 1):
                for i in range(j):
                    c = D[k - 1, i] + cost[i, j]
                    if c < D[k, j]:
                        D[k, j], P[k, j] = c, i
        rss = max(D[K, N], 1e-12)
        total = N * np.log(rss / N) + (3 * K - 1) * np.log(N)
        if best is None or total < best[0] - 1e-12:
            bounds, j = [], N
            for k in range(K, 0, -1):
                i = P[k, j]
                bounds.append((i, j))
                j = i
            best = (total, bounds[::-1])
    segs = []
    for i, j in best[1]:
        a, b, sa, sb, rel = fit_segment(x[i:j], y[i:j])
        segs.append({"n_from": int(x[i]), "n_to": int(x[j - 1]), "npts": j - i, "alpha_us": a, "alpha_se_us": sa,
                     "beta_us_per_byte": b, "beta_se": sb, "inv_beta_GBps": (1e-3 / b) if b > 0 else np.nan,
                     "rel_rms_err": rel})
    return segs


def human(n):
    for u, d in (("GiB", GiB), ("MiB", MiB), ("KiB", KiB)):
        if n >= d and n % d == 0:
            return f"{n // d} {u}"
    return f"{n} B"


def main():
    report = {}
    val = load_omb_phase("validate")
    if len(val):
        v = val.groupby(["bench", "path", "mode", "rep"]).agg(
            sizes=("size", "count"), passed=("validation", lambda s: int((s == "Pass").sum())),
            failed=("validation", lambda s: int((s == "Fail").sum())),
            min_size=("size", "min"), max_size=("size", "max"), abnormal_exit=("abnormal_exit", "any"),
            log=("log", "first")).reset_index()
        v.to_csv(f"{RES}/validation_summary.csv", index=False)
        report["validation"] = v
    # also include non-OK validation logs
    st = []
    for ph in ["validate"]:
        f = f"{RAW}/{ph}/runs_index.csv"
        if os.path.exists(f):
            st.append(pd.read_csv(f))
    if st:
        pd.concat(st).to_csv(f"{RES}/validation_runs_index.csv", index=False)

    p2p = pd.concat([d for d in (load_omb_phase("p2p"), load_omb_phase("ucx")) if len(d)], ignore_index=True) \
        if any(os.path.isdir(f"{RAW}/{x}") for x in ("p2p", "ucx")) else pd.DataFrame()
    if len(p2p):
        assert not p2p["abnormal_exit"].any(), "a p2p measurement log has an aborted rank"
        p2p.to_csv(f"{RES}/p2p_raw.csv", index=False)
        pa = agg(p2p, ["bench", "path", "mode", "n", "size"], "value")
        pa.to_csv(f"{RES}/p2p_summary.csv", index=False)
        report["p2p"] = pa
        # alpha-beta fits on osu_latency medians
        fits = []
        for (path, mode), g in pa[pa.bench == "osu_latency"].groupby(["path", "mode"]):
            g = g.sort_values("size")
            for s in piecewise_fit(g["size"].values, g["median"].values):
                s.update({"curve": f"osu_latency {path} {mode}"})
                fits.append(s)
        pd.DataFrame(fits).to_csv(f"{RES}/fits_p2p_latency.csv", index=False)
        report["fits_p2p"] = pd.DataFrame(fits)

    coll = load_omb_phase("coll")
    if len(coll):
        assert not coll["abnormal_exit"].any(), "a collective measurement log has an aborted rank"
        coll["kind"] = coll["bench"].map(coll_kind)
        coll["S"] = [omb_S(k, s, n) for k, s, n in zip(coll["kind"], coll["size"], coll["n"])]
        coll["busbw_GBps"] = [S / (t * 1e-6) / 1e9 * FACTOR[k](n) for S, t, k, n in
                              zip(coll["S"], coll["value"], coll["kind"], coll["n"])]
        coll.to_csv(f"{RES}/coll_mpi_raw.csv", index=False)
        ca = agg(coll, ["kind", "path", "mode", "n", "S"], "value").rename(columns=lambda c: c if c in ("kind", "path", "mode", "n", "S") else "t_" + c)
        cb = agg(coll, ["kind", "path", "mode", "n", "S"], "busbw_GBps").rename(columns=lambda c: c if c in ("kind", "path", "mode", "n", "S") else "busbw_" + c)
        cm = ca.merge(cb, on=["kind", "path", "mode", "n", "S"])
        cm["lib"] = "MVAPICH 4.1"
        cm.to_csv(f"{RES}/coll_mpi_summary.csv", index=False)
        report["coll_mpi"] = cm

    nccl, ninfo = load_nccl_phase("nccl") if os.path.isdir(f"{RAW}/nccl") else (pd.DataFrame(), pd.DataFrame())
    if len(ninfo):
        ninfo.to_csv(f"{RES}/nccl_run_info.csv", index=False)
    if len(nccl):
        # nccl-tests prints 0-byte rows when S/n rounds to 0 elements per rank; those are not measurements
        nccl = nccl[(nccl["size"] > 0) & (nccl["count"] > 0)].copy()
        nccl["kind"] = nccl["bench"].map(coll_kind)
        nccl["S"] = nccl["size"]
        nccl["busbw_GBps"] = [S / (t * 1e-6) / 1e9 * FACTOR[k](n) for S, t, k, n in
                              zip(nccl["S"], nccl["oop_time_us"], nccl["kind"], nccl["n"])]
        # sanity check: our busbw (from out-of-place time) vs the busbw column nccl-tests printed
        nccl["busbw_check_rel_diff"] = (nccl["busbw_GBps"] - nccl["oop_busbw"]).abs() / nccl["oop_busbw"].where(nccl["oop_busbw"] > 0.05)
        nccl.to_csv(f"{RES}/coll_nccl_raw.csv", index=False)
        na = agg(nccl, ["kind", "set", "p2p_level", "n", "S"], "oop_time_us").rename(columns=lambda c: c if c in ("kind", "set", "p2p_level", "n", "S") else "t_" + c)
        nb = agg(nccl, ["kind", "set", "p2p_level", "n", "S"], "busbw_GBps").rename(columns=lambda c: c if c in ("kind", "set", "p2p_level", "n", "S") else "busbw_" + c)
        nm = na.merge(nb, on=["kind", "set", "p2p_level", "n", "S"])
        nm = nm.rename(columns={"set": "path"})
        nm["lib"] = "NCCL 2.21.5"
        nm.to_csv(f"{RES}/coll_nccl_summary.csv", index=False)
        report["coll_nccl"] = nm
        wrong = nccl.assign(oop_wrong=pd.to_numeric(nccl.oop_wrong, errors="coerce"),
                            ip_wrong=pd.to_numeric(nccl.ip_wrong, errors="coerce"))
        wsum = wrong.groupby(["bench", "set", "p2p_level", "rep"]).agg(
            sizes=("size", "count"), oop_wrong_total=("oop_wrong", "sum"), ip_wrong_total=("ip_wrong", "sum"),
            log=("log", "first")).reset_index()
        wsum.to_csv(f"{RES}/nccl_wrong_summary.csv", index=False)
        report["nccl_wrong"] = wsum
        nccl_proto = nccl_protocols()
        report["nccl_proto"] = nccl_proto

    if "coll_mpi" in report and "coll_nccl" in report:
        compare_table(report["coll_mpi"], report["coll_nccl"])
        cfits = []
        for name, df, tcol in (("MVAPICH", report["coll_mpi"], "t_median"), ("NCCL", report["coll_nccl"], "t_median")):
            for keys, g in df.groupby(["kind", "path"]):
                g = g.sort_values("S")
                for s in piecewise_fit(g["S"].values, g[tcol].values):
                    s.update({"curve": f"{name} {keys[0]} {keys[1]}"})
                    cfits.append(s)
        pd.DataFrame(cfits).to_csv(f"{RES}/fits_collectives.csv", index=False)

    ucx_protocols()
    gpu_time()
    write_tables(report)
    print("analysis done; results in", RES)


def nccl_protocols():
    """Algorithm/protocol NCCL chose per size (rank 0 TUNING lines, rep1 log of each config)."""
    rows = []
    for p in sorted(glob.glob(f"{RAW}/nccl/*__rep1.log")):
        h, _, info = parse_nccl(p)
        for size, nbytes, algo, proto, t in info["tuning"]:
            rows.append({"config_id": h["config_id"], "size": size, "tuning_bytes": nbytes,
                         "algo": ALGO.get(algo, algo), "proto": PROTO.get(proto, proto), "pred_time_us": t})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.groupby(["config_id", "size", "tuning_bytes", "algo", "proto"]).agg(
        calls=("pred_time_us", "count"), pred_time_us=("pred_time_us", "first")).reset_index()
    df.to_csv(f"{RES}/nccl_algo_proto.csv", index=False)
    # compact: contiguous size ranges with the same choice
    seg = []
    for cid, g in df.groupby("config_id"):
        g = g.sort_values("size")
        dom = g.loc[g.groupby("size")["calls"].idxmax()]
        cur = None
        for _, r in dom.iterrows():
            key = (r.algo, r.proto)
            if cur and cur["key"] == key:
                cur["to"] = r["size"]
            else:
                if cur:
                    seg.append(cur)
                cur = {"config_id": cid, "key": key, "from": r["size"], "to": r["size"]}
        if cur:
            seg.append(cur)
    sg = pd.DataFrame([{"config_id": x["config_id"], "algo": x["key"][0], "proto": x["key"][1],
                        "size_from": x["from"], "size_to": x["to"]} for x in seg])
    sg.to_csv(f"{RES}/nccl_algo_proto_segments.csv", index=False)
    return sg


def compare_table(cm, nm):
    rows = []
    for S in (8 * KiB, 1 * MiB, 256 * MiB):
        for kind in ("allreduce", "allgather", "reduce_scatter"):
            for path in sorted(set(cm.path) | set(nm.path)):
                a = cm[(cm.kind == kind) & (cm.path == path) & (cm.S == S)]
                b = nm[(nm.kind == kind) & (nm.path == path) & (nm.S == S)]
                if not len(a) and not len(b):
                    continue
                r = {"S": human(S), "collective": kind, "gpu_set": path}
                if len(a):
                    a = a.iloc[0]
                    r.update({"mpi_mode": a["mode"], "mpi_t_med_us": a.t_median, "mpi_t_min_us": a.t_vmin,
                              "mpi_t_max_us": a.t_vmax, "mpi_busbw_med_GBps": a.busbw_median, "mpi_nreps": a.t_nreps})
                if len(b):
                    b = b.iloc[0]
                    r.update({"nccl_p2p_level": b["p2p_level"], "nccl_t_med_us": b.t_median, "nccl_t_min_us": b.t_vmin,
                              "nccl_t_max_us": b.t_vmax, "nccl_busbw_med_GBps": b.busbw_median, "nccl_nreps": b.t_nreps})
                if len(a) and len(b):
                    r["nccl_speedup_x"] = r["mpi_t_med_us"] / r["nccl_t_med_us"]
                rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(f"{RES}/mpi_vs_nccl_table.csv", index=False)
    return t


def ucx_protocols():
    """From raw/ucxlog (UCX_LOG_LEVEL=info, UCX_PROTO_INFO=y): the intra-node endpoint transports and the
    protocol table for tag sends from GPU memory, for one process (lines are de-interleaved by pid)."""
    out = ["<!-- generated by scripts/analyze.py from raw/ucxlog/*.log -->"]
    for f in sorted(glob.glob(f"{RAW}/ucxlog/*.log")):
        txt = open(f, errors="replace").read()
        h = read_header(f)
        cfgs = sorted(set(m.group(0) for m in re.finditer(r"intra-node cfg#\d+ tag\([^)]*\)", txt)))
        pids = re.findall(r"\[[^:\]]+:(\d+):0\]", txt)
        pid = pids[0] if pids else None
        lines = [re.sub(r"^\[[\d.]+\] \[[^]]+\]\s*", "", l) for l in txt.splitlines() if pid and f":{pid}:0]" in l]
        tab, grab = [], False
        for l in lines:
            if "tagged message by ucp_tag_send" in l and "from cuda" in l:
                grab = True
                tab.append(l.strip())
                continue
            if grab:
                if re.match(r"^\|\s*[\dKMG]+\.\.", l.strip()):
                    tab.append(l.strip())
                elif l.strip().startswith("+") and len(tab) > 1:
                    break
        out.append(f"\n### {h.get('config_id')}  (`{os.path.relpath(f, REPO)}`)\n")
        out.append("UCX endpoint config: " + "; ".join(f"`{c}`" for c in cfgs) + "\n")
        out.append("```\n" + "\n".join(tab) + "\n```")
    open(f"{RES}/ucx_protocol_selection.md", "w").write("\n".join(out) + "\n")


def gpu_time():
    f = f"{RAW}/gpu_time.csv"
    if not os.path.exists(f):
        return
    g = pd.read_csv(f)
    s = g.groupby("phase").agg(runs=("wall_s", "count"), wall_s=("wall_s", "sum"), gpu_s=("gpu_s", "sum")).reset_index()
    # diagnostics / probes outside campaign.py: take wall_s from their logs
    extra = []
    for p in glob.glob(f"{RAW}/diag/*.log"):
        m = re.search(r"wall_s: ([\d.]+)", open(p).read())
        if m:
            extra.append(("diag", float(m.group(1)), 2))
    for p in glob.glob(f"{RAW}/p2p_probe/probe_*.log"):
        m = re.search(r"wall_s: ([\d.]+)", open(p).read())
        if m:
            extra.append(("p2p_probe", float(m.group(1)), 2))
    if extra:
        e = pd.DataFrame(extra, columns=["phase", "wall_s", "ngpu"])
        e["gpu_s"] = e.wall_s * e.ngpu
        s = pd.concat([s, e.groupby("phase").agg(runs=("wall_s", "count"), wall_s=("wall_s", "sum"),
                                                gpu_s=("gpu_s", "sum")).reset_index()], ignore_index=True)
    # a manual run that hung, outside campaign.py; its start and kill times are in raw/manual_runs.csv
    for _, m in pd.read_csv(f"{RAW}/manual_runs.csv").iterrows():
        d = (pd.Timestamp(m.end) - pd.Timestamp(m.start)).total_seconds()
        s = pd.concat([s, pd.DataFrame([{"phase": f"{m.run} (from commands.log)", "runs": 1,
                                         "wall_s": d, "gpu_s": m.n_gpus * d}])], ignore_index=True)
    tot = pd.DataFrame([{"phase": "TOTAL (logged runs)", "runs": s.runs.sum(), "wall_s": s.wall_s.sum(),
                         "gpu_s": s.gpu_s.sum()}])
    s = pd.concat([s, tot], ignore_index=True)
    s["wall_min"] = s.wall_s / 60
    s["gpu_hours"] = s.gpu_s / 3600
    s.to_csv(f"{RES}/gpu_time_summary.csv", index=False)


def fmt(x, nd=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    if abs(x) >= 100:
        return f"{x:.1f}"
    return f"{x:.{nd}f}"


def md_table(df, cols, headers):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(c(r)) if callable(c) else str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def write_tables(report):
    md = ["<!-- generated by scripts/analyze.py from raw/ logs; do not edit by hand -->"]
    if "validation" in report:
        v = report["validation"]
        md.append("\n### Validation (OMB `-c`, one 1 B–1 GiB sweep per config)\n")
        md.append(md_table(v.sort_values(["bench", "path", "mode"]),
                           ["bench", "path", "mode", lambda r: f"{human(r.min_size)}–{human(r.max_size)}",
                            "sizes", "passed", "failed",
                            lambda r: "**yes: ranks aborted after last size**" if r.abnormal_exit else "no",
                            lambda r: f"`{r.log}`"],
                           ["benchmark", "GPUs", "mode", "sizes validated", "#sizes", "Pass", "Fail",
                            "abnormal exit", "log"]))
    if "p2p" in report:
        pa = report["p2p"]
        rows = []
        pick = {"osu_latency": [1, 8 * KiB, 1 * MiB, 64 * MiB], "osu_bw": [8 * KiB, 1 * MiB, 64 * MiB, 1 * GiB],
                "osu_bibw": [1 * MiB, 64 * MiB, 1 * GiB]}
        for b, sizes in pick.items():
            for (path, mode), g in pa[pa.bench == b].groupby(["path", "mode"]):
                for S in sizes:
                    r = g[g["size"] == S]
                    if len(r):
                        r = r.iloc[0]
                        rows.append({"bench": b, "path": path, "mode": mode, "size": human(S), "median": r["median"],
                                     "min": r.vmin, "max": r.vmax, "n": int(r.nreps)})
        t = pd.DataFrame(rows)
        md.append("\n### Point-to-point, device buffers both sides (median [min–max] over n runs)\n")
        md.append("Latency in µs (one-way, as OMB reports it); bandwidth in MB/s (OMB's MB = 10^6 bytes).\n")
        md.append(md_table(t, ["bench", "path", "mode", "size", lambda r: fmt(r["median"]),
                               lambda r: f"{fmt(r['min'])}–{fmt(r['max'])}", "n"],
                           ["benchmark", "GPUs", "mode", "size", "median", "min–max", "n"]))
    if "fits_p2p" in report and len(report["fits_p2p"]):
        f = report["fits_p2p"]
        md.append("\n### Piecewise α–β fits of osu_latency medians, t(n) = α + n·β\n")
        md.append(md_table(f, ["curve", lambda r: f"{human(r.n_from)}–{human(r.n_to)}", "npts",
                               lambda r: f"{fmt(r.alpha_us)} ± {fmt(r.alpha_se_us)}",
                               lambda r: f"{r.beta_us_per_byte:.3e} ± {r.beta_se:.1e}",
                               lambda r: fmt(r.inv_beta_GBps), lambda r: f"{100 * r.rel_rms_err:.1f}%"],
                           ["curve", "segment", "pts", "α (µs) ± SE", "β (µs/B) ± SE", "1/β (GB/s)", "rel. RMS err"]))
    p = f"{RES}/fits_collectives.csv"
    if os.path.exists(p):
        f = pd.read_csv(p)
        md.append("\n### Piecewise α–β fits of collective time medians (x = S in nccl-tests convention)\n")
        md.append(md_table(f, ["curve", lambda r: f"{human(int(r.n_from))}–{human(int(r.n_to))}", "npts",
                               lambda r: f"{fmt(r.alpha_us)} ± {fmt(r.alpha_se_us)}",
                               lambda r: f"{r.beta_us_per_byte:.3e} ± {r.beta_se:.1e}",
                               lambda r: fmt(r.inv_beta_GBps), lambda r: f"{100 * r.rel_rms_err:.1f}%"],
                           ["curve", "segment", "pts", "α (µs) ± SE", "β (µs/B) ± SE", "1/β (GB/s)", "rel. RMS err"]))
    p = f"{RES}/mpi_vs_nccl_table.csv"
    if os.path.exists(p):
        t = pd.read_csv(p)
        md.append("\n### MPI (MVAPICH 4.1 + OMB) vs NCCL 2.21.5 (nccl-tests), median [min–max] of 5 runs\n")
        md.append("S = bytes in the nccl-tests convention. busbw = S/t × (2(n−1)/n for allreduce, (n−1)/n otherwise).\n")
        md.append(md_table(t, ["S", "collective", "gpu_set", "mpi_mode",
                               lambda r: f"{fmt(r.mpi_t_med_us)} [{fmt(r.mpi_t_min_us)}–{fmt(r.mpi_t_max_us)}]",
                               lambda r: fmt(r.mpi_busbw_med_GBps),
                               lambda r: f"{fmt(r.nccl_t_med_us)} [{fmt(r.nccl_t_min_us)}–{fmt(r.nccl_t_max_us)}]",
                               lambda r: fmt(r.nccl_busbw_med_GBps), lambda r: fmt(r.get("nccl_speedup_x"))],
                           ["S", "collective", "GPUs", "MPI mode", "MPI time µs", "MPI busbw GB/s",
                            "NCCL time µs", "NCCL busbw GB/s", "MPI time / NCCL time"]))
    if "nccl_wrong" in report:
        w = report["nccl_wrong"]
        md.append("\n### nccl-tests correctness (`-c 1`): total #wrong over all sizes, per run\n")
        md.append(md_table(w, ["bench", "set", "p2p_level", "rep", "sizes", lambda r: fmt(r.oop_wrong_total, 0),
                               lambda r: fmt(r.ip_wrong_total, 0)],
                           ["benchmark", "GPUs", "NCCL_P2P_LEVEL", "rep", "#sizes", "#wrong out-of-place", "#wrong in-place"]))
    p = f"{RES}/gpu_time_summary.csv"
    if os.path.exists(p):
        g = pd.read_csv(p)
        md.append("\n### GPU time used (from per-run wall times in the logs)\n")
        md.append(md_table(g, ["phase", "runs", lambda r: fmt(r.wall_min, 1), lambda r: fmt(r.gpu_hours, 2)],
                           ["phase", "runs", "wall minutes", "GPU-hours (wall × #GPUs)"]))
    open(f"{RES}/tables.md", "w").write("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
