#!/usr/bin/env python3
"""Summarise follow-up allreduce runs: parse RESULT lines from raw/<phase>/*.log.

Per run: the program's median over its timed iterations (each iteration = max over ranks).
Per (phase, config, size): median of the run medians, min and max of the run medians, n runs, busbw at that median
(n = 2 ranks, so busbw = S/t), and the total n_bad. Cliff boundary per config (PREREGISTRATION.md): first adjacent pair
a < b of tested sizes with busbw(b)/busbw(a) < 0.6.
Writes results/summary_<phase>.csv, results/cliff_<phase>.csv and prints a table.
Usage: analyze_followup.py <phase> [<phase> ...]
"""
import csv, glob, os, re, statistics, sys

FREPO = os.path.expanduser("~/mpi-osu-a6000/followup")
RX = re.compile(r"^RESULT size_mib=(\d+) bytes=(\d+) mem=(\w+) iters=(\d+) median_us=([\d.]+) min_us=([\d.]+) "
                r"max_us=([\d.]+) mean_us=([\d.]+) busbw_GBps_at_median=([\d.]+) n_checked=(\d+) n_bad=(\d+)")


def load(phase):
    rows = []
    for f in sorted(glob.glob(f"{FREPO}/raw/{phase}/*__rep*.log")):
        cid, rep = re.match(r"(.+)__rep(\d+)\.log", os.path.basename(f)).groups()
        text = open(f, errors="replace").read()
        status = re.search(r"^# status: (\S+)", text, re.M)
        status = status.group(1) if status else "NO_STATUS"
        for line in text.splitlines():
            m = RX.match(line)
            if m:
                rows.append({"config": cid, "rep": int(rep), "size_mib": int(m.group(1)), "bytes": int(m.group(2)),
                             "mem": m.group(3), "run_median_us": float(m.group(5)), "n_bad": int(m.group(11)),
                             "n_checked": int(m.group(10)), "status": status, "log": os.path.relpath(f, FREPO)})
    return rows


def summarise(phase):
    rows = load(phase)
    out, cliffs = [], []
    for cid in sorted({r["config"] for r in rows}):
        rs = [r for r in rows if r["config"] == cid]
        sizes = sorted({r["size_mib"] for r in rs})
        per = {}
        for s in sizes:
            v = [r["run_median_us"] for r in rs if r["size_mib"] == s and r["status"] == "OK"]
            bad = sum(r["n_bad"] for r in rs if r["size_mib"] == s)
            nonok = sorted({r["status"] for r in rs if r["size_mib"] == s and r["status"] != "OK"})
            if not v:
                continue
            med = statistics.median(v)
            S = s << 20
            per[s] = S / (med * 1e-6) / 1e9
            out.append({"phase": phase, "config": cid, "mem": rs[0]["mem"], "size_mib": s, "n_runs": len(v),
                        "t_median_us": round(med, 2), "t_min_us": round(min(v), 2), "t_max_us": round(max(v), 2),
                        "busbw_GBps_median": round(per[s], 3), "busbw_GBps_min": round(S / (max(v) * 1e-6) / 1e9, 3),
                        "busbw_GBps_max": round(S / (min(v) * 1e-6) / 1e9, 3), "n_bad_total": bad,
                        "non_ok_status": ";".join(nonok)})
        ss = sorted(per)
        b = next(((a, c) for a, c in zip(ss, ss[1:]) if per[c] / per[a] < 0.6), None)
        cliffs.append({"phase": phase, "config": cid, "sizes_tested": " ".join(map(str, ss)),
                       "cliff_from_mib": b[0] if b else "", "cliff_to_mib": b[1] if b else "",
                       "busbw_ratio": round(per[b[1]] / per[b[0]], 3) if b else "",
                       "verdict": f"cliff between {b[0]} and {b[1]} MiB" if b else "no cliff (no >40% drop)"})
    os.makedirs(f"{FREPO}/results", exist_ok=True)
    for name, data in (("summary", out), ("cliff", cliffs)):
        if data:
            with open(f"{FREPO}/results/{name}_{phase}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(data[0]))
                w.writeheader()
                w.writerows(data)
    return out, cliffs


if __name__ == "__main__":
    for ph in sys.argv[1:]:
        out, cliffs = summarise(ph)
        print(f"== phase {ph}")
        cur = None
        for r in out:
            if r["config"] != cur:
                cur = r["config"]
                print(f"-- {cur}")
            print(f"  {r['size_mib']:>4} MiB  t={r['t_median_us']:>10.2f} us [{r['t_min_us']:.2f}-{r['t_max_us']:.2f}]  "
                  f"busbw={r['busbw_GBps_median']:6.3f} GB/s  n={r['n_runs']} bad={r['n_bad_total']} {r['non_ok_status']}")
        for c in cliffs:
            print(f"CLIFF {c['config']}: {c['verdict']} (ratio {c['busbw_ratio']})")
