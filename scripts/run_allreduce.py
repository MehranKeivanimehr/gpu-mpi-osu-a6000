#!/usr/bin/env python3
"""Run the allreduce size-sweep phases on one NVLink pair.

Usage: run_allreduce.py <phase> [--reps N] [--seed S] [--dry]

All (config, rep) pairs of a phase are shuffled with a fixed seed, and the size order inside each run is shuffled
too. Before each run the config must pass check_safe(), the two GPUs must be idle, and the GPU-time budget must not
be used up. Each run writes raw/allreduce/<phase>/<config>__rep<k>.log with a header and a trailer.
"""
import argparse, csv, datetime as dt, os, random, shlex, subprocess, sys, time

H = os.path.expanduser("~")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{REPO}/raw/allreduce"
MV = f"{H}/opt/mvapich-4.1-cuda"
MPIEXEC = f"{MV}/bin/mpiexec"                      # gforker launcher: opens no TCP listener
WRAP = f"{REPO}/scripts/gpu_bind.sh"
BIN = f"{REPO}/build/allreduce_sizes"
NSYS = "/usr/local/cuda-12.4/nsight-systems-2023.4.4/bin/nsys"
# same as the "mv" mode in scripts/configs.py
BASE_ENV = {
    "LD_LIBRARY_PATH": f"{H}/opt/cuda-driver-link:/usr/local/cuda-12.4/lib64",
    "UCX_TLS": "self,sm,cuda",
    "UCX_WARN_UNUSED_ENV_VARS": "n",
}
PAIR = [2, 3]                                       # NVLink pair (NV4); 6,7 is the fallback pair
ALLOWED_PAIRS = [{2, 3}, {6, 7}]
FORBIDDEN_GPUS = {4, 5}                             # GPU 4 stuck; 5 is its NVLink partner
BUDGET_GPU_S = 3300                                 # stop before 1 GPU-hour (3600 GPU-s)
IDLE_UTIL_MAX, IDLE_MEM_MAX, WAIT_MAX_S = 5, 200, 900
GPU_TIME_CSV = f"{OUT}/gpu_time.csv"

ALL_SIZES = [24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 96, 128]
NEAR_SIZES = [28, 32, 36, 48, 64]                   # around the 32 to 36 MiB slowdown

ALGOS = ["recursive_doubling", "recursive_multiplying", "reduce_scatter_allgather", "tree", "recexch", "ring",
         "k_reduce_scatter_allgather", "nb", "smp"]


def cfg(cid, mem, sizes, env=None, warm=5, iters=20, gpus=PAIR, nsys=None, timeout=300):
    return {"id": cid, "mem": mem, "sizes": sizes, "env": env or {}, "warm": warm, "iters": iters,
            "gpus": gpus, "nsys": nsys, "timeout": timeout}


def phase_smoke():
    return [cfg("smoke_dev", "dev", [32, 64], warm=2, iters=3), cfg("smoke_host", "host", [32, 64], warm=2, iters=3)]


def phase_sweep():
    return [cfg("dev_default", "dev", ALL_SIZES), cfg("host_default", "host", ALL_SIZES)]


def phase_algo():
    c = [cfg("dev_default", "dev", NEAR_SIZES),
         cfg("dev_dcnone_auto", "dev", NEAR_SIZES, {"MPIR_CVAR_DEVICE_COLLECTIVES": "none"})]
    for a in ALGOS:
        c.append(cfg(f"dev_dcnone_{a}", "dev", NEAR_SIZES,
                     {"MPIR_CVAR_DEVICE_COLLECTIVES": "none", "MPIR_CVAR_ALLREDUCE_INTRA_ALGORITHM": a}))
    for p in ["mpir", "release_gather"]:
        c.append(cfg(f"dev_posix_{p}", "dev", NEAR_SIZES, {"MPIR_CVAR_ALLREDUCE_POSIX_INTRA_ALGORITHM": p}))
    for k in ["1", "2", "3"]:
        c.append(cfg(f"dev_comp{k}", "dev", NEAR_SIZES, {"MPIR_CVAR_ALLREDUCE_COMPOSITION": k}))
    return c


def phase_knobs():
    MiB = 1 << 20
    return [
        cfg("dev_default", "dev", ALL_SIZES),
        cfg("dev_lmem64M", "dev", ALL_SIZES, {"MPIR_CVAR_LMEM_POOL_SIZE": str(64 * MiB)}),
        cfg("dev_lmem128M", "dev", ALL_SIZES, {"MPIR_CVAR_LMEM_POOL_SIZE": str(128 * MiB)}),
        cfg("dev_lmem16M", "dev", ALL_SIZES, {"MPIR_CVAR_LMEM_POOL_SIZE": str(16 * MiB)}),
        cfg("host_lmem128M", "host", ALL_SIZES, {"MPIR_CVAR_LMEM_POOL_SIZE": str(128 * MiB)}),
    ]


def phase_glibc():
    """Exploratory, added after the preregistered phases. glibc malloc tunables:
    https://sourceware.org/glibc/manual/2.39/html_node/Memory-Allocation-Tunables.html
    top_pad=32 MiB copies what MVAPICH 4.1 sets with mallopt() in its OFI netmod init, which a ch4:ucx build skips."""
    s = [28, 32, 36, 64, 128]
    return [cfg("dev_default", "dev", s),
            cfg("dev_glibc_mmapmax0", "dev", s, {"GLIBC_TUNABLES": "glibc.malloc.mmap_max=0"}),
            cfg("dev_glibc_mmapmax0_toppad32M", "dev", s,
                {"GLIBC_TUNABLES": "glibc.malloc.mmap_max=0:glibc.malloc.top_pad=33554432"}),
            cfg("host_glibc_mmapmax0", "host", s, {"GLIBC_TUNABLES": "glibc.malloc.mmap_max=0"}),
            # run after the four configs above
            cfg("dev_glibc_mmapmax0_trim1G", "dev", s,
                {"GLIBC_TUNABLES": "glibc.malloc.mmap_max=0:glibc.malloc.trim_threshold=1073741824"}),
            cfg("host_glibc_mmapmax0_trim1G", "host", s,
                {"GLIBC_TUNABLES": "glibc.malloc.mmap_max=0:glibc.malloc.trim_threshold=1073741824"})]


def phase_nsys():
    out = f"{OUT}/nsys"
    return [cfg(f"nsys_dev_{s}MiB", "dev", [s], warm=3, iters=10, nsys=f"{out}/dev_{s}MiB_rank%q{{PMI_RANK}}")
            for s in (32, 64)]


PHASES = {"smoke": phase_smoke, "sweep": phase_sweep, "algo": phase_algo, "knobs": phase_knobs,
          "nsys": phase_nsys, "glibc": phase_glibc}


def check_safe(c):
    g = set(c["gpus"])
    if g & FORBIDDEN_GPUS:
        return f"uses forbidden GPU(s) {sorted(g & FORBIDDEN_GPUS)} (GPU 4 stuck, GPU 5 is its partner)"
    if len(c["gpus"]) != 2 or g not in ALLOWED_PAIRS:
        return f"GPU set {c['gpus']} is not exactly one NVLink pair (2-3 or 6-7)"
    for k in c["env"]:
        if k.startswith("NCCL_") or k.startswith("UCX_TLS"):
            return f"env {k} not allowed here"
    return None


def now():
    return dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def log_cmd(s):
    with open(f"{REPO}/commands.log", "a") as f:
        f.write(f"[{now()}] (cwd={REPO}) {s}\n")


def gpu_state():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    st = {}
    for line in out.strip().splitlines():
        i, u, m = [x.strip() for x in line.split(",")]
        st[int(i)] = (int(u), int(m))
    return st, out


def wait_idle(gpus):
    t0 = time.time()
    while True:
        st, raw = gpu_state()
        busy = [g for g in gpus if st[g][0] > IDLE_UTIL_MAX or st[g][1] > IDLE_MEM_MAX]
        if not busy:
            return True, raw
        if time.time() - t0 > WAIT_MAX_S:
            return False, raw
        print(f"  [{now()}] GPUs busy {busy}: {[st[g] for g in busy]}; waiting", flush=True)
        time.sleep(20)


def gpu_s_used():
    if not os.path.exists(GPU_TIME_CSV):
        return 0.0
    with open(GPU_TIME_CSV) as f:
        return sum(float(r["gpu_s"]) for r in csv.DictReader(f))


def build_cmd(c, sizes):
    prog = [BIN, "-m", c["mem"], "-s", ",".join(map(str, sizes)), "-w", str(c["warm"]), "-n", str(c["iters"])]
    if c["nsys"]:
        prog = [NSYS, "profile", "--sample=none", "--cpuctxsw=none", "--trace=cuda,mpi,osrt", "--mpi-impl=mpich",
                "--force-overwrite=true", "-o", c["nsys"]] + prog
    return [MPIEXEC, "-n", "2", WRAP] + prog


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20261113)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    cfgs = PHASES[a.phase]()
    bad = [(c["id"], check_safe(c)) for c in cfgs if check_safe(c)]
    if bad:
        for cid, why in bad:
            print(f"REFUSING {cid}: {why}")
        sys.exit("unsafe config(s); nothing launched")
    runs = [(c, r) for c in cfgs for r in range(1, a.reps + 1)]
    random.Random(a.seed).shuffle(runs)
    outdir = f"{OUT}/{a.phase}"
    os.makedirs(outdir, exist_ok=True)
    idx_path = f"{outdir}/runs_index.csv"
    new_idx = not os.path.exists(idx_path)
    idx = open(idx_path, "a", newline="")
    w = csv.writer(idx)
    if new_idx:
        w.writerow(["order", "config_id", "rep", "start", "end", "wall_s", "exit", "status", "size_order", "log"])
    gt_new = not os.path.exists(GPU_TIME_CSV)
    gt = open(GPU_TIME_CSV, "a", newline="")
    gw = csv.writer(gt)
    if gt_new:
        gw.writerow(["phase", "config_id", "rep", "n_gpus", "wall_s", "gpu_s"])
    print(f"[{now()}] phase={a.phase} configs={len(cfgs)} runs={len(runs)} seed={a.seed} "
          f"gpu_s_used_so_far={gpu_s_used():.1f}", flush=True)
    for order, (c, rep) in enumerate(runs, 1):
        logf = f"{outdir}/{c['id']}__rep{rep}.log"
        if os.path.exists(logf):
            print(f"  skip existing {os.path.basename(logf)}")
            continue
        sizes = list(c["sizes"])
        random.Random(f"{a.seed}-{c['id']}-{rep}").shuffle(sizes)
        cmd = build_cmd(c, sizes)
        env = os.environ.copy()
        env.update(BASE_ENV)
        env.update(c["env"])
        env["GPU_LIST"] = ",".join(map(str, c["gpus"]))
        shown = dict(BASE_ENV, **c["env"], GPU_LIST=env["GPU_LIST"])
        cmd_str = " ".join(f"{k}={shlex.quote(v)}" for k, v in sorted(shown.items())) + " " + \
            " ".join(shlex.quote(x) for x in cmd)
        print(f"[{now()}] ({order}/{len(runs)}) {c['id']} rep{rep} sizes={sizes}", flush=True)
        if a.dry:
            print("   ", cmd_str)
            continue
        used = gpu_s_used()
        if used + 60 > BUDGET_GPU_S:                # keep a 60 GPU-s margin
            print(f"  BUDGET STOP: {used:.0f} GPU-s used, budget {BUDGET_GPU_S}")
            w.writerow([order, c["id"], rep, now(), now(), 0, "", "SKIPPED_BUDGET", "", ""])
            idx.flush()
            break
        ok, smi = wait_idle(c["gpus"])
        if not ok:
            w.writerow([order, c["id"], rep, now(), now(), 0, "", "SKIPPED_GPUS_BUSY", "", ""])
            idx.flush()
            continue
        log_cmd(cmd_str + f"   > {os.path.relpath(logf, REPO)} 2>&1")
        start, t0 = now(), time.time()
        with open(logf, "w") as f:
            f.write(f"# config_id: {c['id']}\n# rep: {rep}\n# order: {order}/{len(runs)} seed={a.seed}\n")
            f.write(f"# start: {start}\n# gpus: {c['gpus']}\n# size_order_mib: {sizes}\n# cmd: {cmd_str}\n")
            f.write("# nvidia-smi pre-run (index, util%, mem MiB):\n")
            for line in smi.strip().splitlines():
                f.write(f"#   {line}\n")
            f.flush()
            try:
                p = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=c["timeout"],
                                   cwd=REPO)
                rc = p.returncode
                status = "OK" if rc == 0 else "FAILED"
            except subprocess.TimeoutExpired:
                rc, status = -9, "TIMEOUT"
            f.flush()
            wall = time.time() - t0
            text = open(logf, errors="replace").read()
            if status == "OK" and "signaled with" in text:
                status = "FAILED_RANK_SIGNAL"
            results = [l for l in text.splitlines() if l.startswith("RESULT ")]
            if status == "OK" and len(results) != len(sizes):
                status = "FAILED_MISSING_RESULTS"
            if status == "OK" and any(" n_bad=0" not in l for l in results):
                status = "FAILED_CORRECTNESS"
            f.write(f"\n# end: {now()}\n# exit: {rc}\n# wall_s: {wall:.2f}\n# status: {status}\n")
        w.writerow([order, c["id"], rep, start, now(), f"{wall:.2f}", rc, status, " ".join(map(str, sizes)),
                    os.path.relpath(logf, REPO)])
        idx.flush()
        gw.writerow([a.phase, c["id"], rep, len(c["gpus"]), f"{wall:.2f}", f"{wall * len(c['gpus']):.2f}"])
        gt.flush()
        print(f"    -> {status} rc={rc} {wall:.1f}s (GPU-s used now {gpu_s_used():.0f})", flush=True)
    idx.close()
    gt.close()


if __name__ == "__main__":
    main()
