#!/usr/bin/env python3
"""Run one benchmark phase from configs.py in shuffled (config, rep) order, waiting for idle GPUs before each run.

Usage: campaign.py <phase> [--reps 5] [--seed N] [--only substr] [--dry]
Each run writes raw/<phase>/<config>__rep<k>.log with a header block (lines starting with '#').
"""
import argparse, csv, datetime as dt, glob, os, random, shlex, subprocess, sys, time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import configs  # noqa: E402

IDLE_UTIL_MAX, IDLE_MEM_MAX, WAIT_MAX_S = 5, 200, 900     # % util, MiB, seconds
GPU_TIME_CSV = os.path.join(REPO, "raw", "gpu_time.csv")


def now():
    return dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def log_cmd(cmd_str):
    with open(os.path.join(REPO, "commands.log"), "a") as f:
        f.write(f"[{now()}] (cwd={REPO}) {cmd_str}\n")


def gpu_state():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20261112)
    ap.add_argument("--only", default=None)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--any-rep", action="store_true",
                    help="skip a config if any rep of it already has a completed log (used for validation)")
    a = ap.parse_args()

    cfgs = configs.PHASES[a.phase]()
    unsafe = [(c["id"], configs.check_safe(c)) for c in cfgs if configs.check_safe(c)]
    if unsafe:
        for cid, why in unsafe:
            print(f"REFUSING {cid}: {why}")
        sys.exit("unsafe configs in phase; fix configs.py")
    if a.only:
        cfgs = [c for c in cfgs if a.only in c["id"]]
    runs = [(c, r) for c in cfgs for r in range(1, a.reps + 1)]
    random.Random(a.seed).shuffle(runs)

    outdir = os.path.join(REPO, "raw", a.phase)
    os.makedirs(outdir, exist_ok=True)
    index_path = os.path.join(outdir, "runs_index.csv")
    new_index = not os.path.exists(index_path)
    idx = open(index_path, "a", newline="")
    w = csv.writer(idx)
    if new_index:
        w.writerow(["order", "config_id", "rep", "start", "end", "wall_s", "exit", "status", "log"])
    gt_new = not os.path.exists(GPU_TIME_CSV)
    gt = open(GPU_TIME_CSV, "a", newline="")
    gw = csv.writer(gt)
    if gt_new:
        gw.writerow(["phase", "config_id", "rep", "n_gpus", "wall_s", "gpu_s"])

    print(f"[{now()}] phase={a.phase} configs={len(cfgs)} runs={len(runs)} seed={a.seed}", flush=True)
    for order, (c, rep) in enumerate(runs, 1):
        logf = os.path.join(outdir, f"{c['id']}__rep{rep}.log")
        if os.path.exists(logf):
            print(f"  skip existing {os.path.basename(logf)}")
            continue
        if a.any_rep and glob.glob(os.path.join(outdir, f"{c['id']}__rep*.log")):
            print(f"  skip {c['id']}: a rep already exists (--any-rep)")
            continue
        cmd = c["cmd"]
        env = os.environ.copy()
        env.update(c.get("env", {}))
        env_str = " ".join(f"{k}={shlex.quote(v)}" for k, v in sorted(c.get("env", {}).items()))
        cmd_str = (env_str + " " if env_str else "") + " ".join(shlex.quote(x) for x in cmd)
        print(f"[{now()}] ({order}/{len(runs)}) {c['id']} rep{rep}", flush=True)
        if a.dry:
            print("   ", cmd_str)
            continue
        ok, smi = wait_idle(c["gpus"])
        if not ok:
            w.writerow([order, c["id"], rep, now(), now(), 0, "", "SKIPPED_GPUS_BUSY", ""])
            idx.flush()
            continue
        log_cmd(cmd_str + f"   > {os.path.relpath(logf, REPO)} 2>&1")
        start = now()
        t0 = time.time()
        with open(logf, "w") as f:
            f.write(f"# config_id: {c['id']}\n# rep: {rep}\n# order: {order}/{len(runs)} seed={a.seed}\n")
            f.write(f"# start: {start}\n# gpus: {c['gpus']}\n# cmd: {cmd_str}\n")
            for k, v in c.get("meta", {}).items():
                f.write(f"# meta.{k}: {v}\n")
            f.write("# nvidia-smi pre-run (index, util%, mem MiB):\n")
            for line in smi.strip().splitlines():
                f.write(f"#   {line}\n")
            f.flush()
            try:
                p = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT,
                                   timeout=c.get("timeout", 900), cwd=REPO)
                rc = p.returncode
                status = "OK" if rc == 0 else "FAILED"
                # gforker's mpiexec can return 0 even when ranks die from a signal; check the log text
                f.flush()
                if status == "OK" and "signaled with" in open(logf, errors="replace").read():
                    status = "FAILED_RANK_SIGNAL"
            except subprocess.TimeoutExpired:
                rc, status = -9, "TIMEOUT"
            wall = time.time() - t0
            f.write(f"\n# end: {now()}\n# exit: {rc}\n# wall_s: {wall:.2f}\n# status: {status}\n")
        w.writerow([order, c["id"], rep, start, now(), f"{wall:.2f}", rc, status, os.path.relpath(logf, REPO)])
        idx.flush()
        gw.writerow([a.phase, c["id"], rep, len(c["gpus"]), f"{wall:.2f}", f"{wall * len(c['gpus']):.2f}"])
        gt.flush()
        print(f"    -> {status} rc={rc} {wall:.1f}s", flush=True)
    idx.close()
    gt.close()


if __name__ == "__main__":
    main()
