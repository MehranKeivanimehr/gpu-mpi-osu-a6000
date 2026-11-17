"""Summarise nsys cuda_gpu_trace CSVs: count, bytes and median throughput per operation and memory kind.

Usage: nsys_summary.py <trace.csv> ...   (prints one table per file)
"""
import csv, statistics, sys, collections
for path in sys.argv[1:]:
    rows = list(csv.DictReader(open(path)))
    g = collections.defaultdict(list)
    for r in rows:
        name = r["Name"]
        if not name.startswith("[CUDA memcpy") and not name.startswith("[CUDA memset"):
            key = ("kernel", name.split("(")[0], "", "")
            g[key].append((0.0, float(r["Duration (ns)"])))
            continue
        key = ("copy", name, r["SrcMemKd"], r["DstMemKd"])
        g[key].append((float(r["Bytes (MB)"] or 0), float(r["Duration (ns)"])))
    print(f"== {path}")
    print("kind,name,src,dst,count,total_MB,median_op_MB,median_op_us,median_GBps")
    for k, v in sorted(g.items()):
        mb = [x[0] for x in v]; ns = [x[1] for x in v]
        gbps = [x[0] / (x[1] * 1e-9) / 1e3 for x in v if x[1] > 0 and x[0] > 0]
        print(f"{k[0]},{k[1]},{k[2]},{k[3]},{len(v)},{sum(mb):.3f},{statistics.median(mb):.3f},"
              f"{statistics.median(ns)/1e3:.1f},{statistics.median(gbps) if gbps else 0:.2f}")
