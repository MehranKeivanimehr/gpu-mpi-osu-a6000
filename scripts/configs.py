"""Benchmark configurations for campaign.py (MVAPICH 4.1 + OMB 7.5.3, nccl-tests 2.21.1 + NCCL 2.21.5).

GPU sets use GPUs 2 to 6 only (see raw/env/nvidia-smi_topo.txt for the links). On this machine PCIe peer-to-peer
between non-NVLink GPUs returned corrupt data, so check_safe() refuses GPU 4 and allows P2P-capable modes
(mv, ucxipc, NCCL without NCCL_P2P_LEVEL=NVL) only on a single NVLink pair.
"""
import os

H = os.path.expanduser("~")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MV = f"{H}/opt/mvapich-4.1-cuda"
MPIEXEC = f"{MV}/bin/mpiexec"            # gforker launcher (no TCP listener)
OMB = f"{H}/opt/omb-7.5.3-mvapich-4.1/libexec/osu-micro-benchmarks/mpi"
NCCLT = f"{H}/opt/nccl-tests-2.21.1-mvapich-4.1/build"
NCCL_HOME = f"{H}/opt/nccl-2.21.5-pipwheel"
WRAP = f"{REPO}/scripts/gpu_bind.sh"

PATHS = {               # name -> GPU list (rank order)
    "nv23": [2, 3],     # NV4 NVLink bridge, NUMA 0
    "node56": [5, 6],   # cross-pair, same socket (NUMA 1), PCIe via host bridges (NODE)
    "sys36": [3, 6],    # cross-pair, across sockets (SYS)  (was 3,4 before GPU 4 got stuck)
}
NV_PAIRS = [{0, 1}, {2, 3}, {4, 5}, {6, 7}]
BROKEN_GPUS = {4}
SETS = {
    "g2_nv23": [2, 3],          # one NVLink pair
    "g4_nv23_5_6": [2, 3, 5, 6],  # NVLink pair 2-3 plus 5 and 6, which are not bridged to each other
}

# MPL dlopen()s the unversioned libcuda.so; ~/opt/cuda-driver-link holds a symlink to the driver's libcuda.so.1
LIBPATH = f"{H}/opt/cuda-driver-link:/usr/local/cuda-12.4/lib64"
BASE_ENV = {
    "LD_LIBRARY_PATH": LIBPATH,
    "UCX_TLS": "self,sm,cuda",              # never tcp: it would open a listening socket
    "UCX_WARN_UNUSED_ENV_VARS": "n",
}
MODES = {
    "mv": {},                               # MVAPICH default intra-node path (CH4 shared memory + GPU IPC)
    # NOLOCAL sends intra-node traffic through the ch4:ucx netmod, so UCX_TLS decides the transport
    "ucxipc": {"MPIR_CVAR_NOLOCAL": "1", "UCX_TLS": "self,sm,cuda_copy,cuda_ipc"},
    "ucxnoipc": {"MPIR_CVAR_NOLOCAL": "1", "UCX_TLS": "self,sm,cuda_copy"},
}

RANGES = {
    "S": ("1:4194304", []),
    "L": ("8388608:1073741824", ["-i", "10", "-x", "3"]),   # fewer iterations so 1 GiB stays affordable
    "V": ("1:1073741824", ["-i", "3", "-x", "1"]),          # validation sweep, checked with -c
}
P2P = {"osu_latency": "pt2pt/osu_latency", "osu_bw": "pt2pt/osu_bw",
       "osu_bibw": "pt2pt/osu_bibw"}
COLL = {"osu_allreduce": "collective/osu_allreduce",
        "osu_allgather": "collective/osu_allgather",
        "osu_reduce_scatter": "collective/osu_reduce_scatter"}
NCCL = {"all_reduce_perf": "all_reduce_perf", "all_gather_perf": "all_gather_perf",
        "reduce_scatter_perf": "reduce_scatter_perf"}


def _env(gpus, mode="mv", extra=None):
    e = dict(BASE_ENV)
    e.update(MODES[mode])
    e["GPU_LIST"] = ",".join(map(str, gpus))
    if extra:
        e.update(extra)
    return e


def omb_cfg(phase, bench, path_name, gpus, rng, mode="mv", validate=False, extra_env=None):
    rel = P2P.get(bench) or COLL[bench]
    is_p2p = bench in P2P
    mrange, rargs = RANGES[rng]
    args = ["-d", "cuda", "-m", mrange] + list(rargs)
    if is_p2p and bench != "osu_latency" and rng == "L":
        args += ["-W", "8"]          # window 8 (default 64) for >= 8 MiB messages
    if is_p2p and bench != "osu_latency" and rng == "V":
        args += ["-W", "2"]          # validation: window 2 keeps host-side checking affordable
    if not is_p2p:
        args += ["-f"]               # full stats: min/max/iterations
    if validate:
        args += ["-c"]
    if is_p2p:
        args += ["D", "D"]
    cmd = [MPIEXEC, "-n", str(len(gpus)), WRAP, f"{OMB}/{rel}"] + args
    cid = f"{bench}__{path_name}__{mode}__{rng}" + ("__val" if validate else "")
    return {"id": cid, "gpus": gpus, "cmd": cmd, "env": _env(gpus, mode, extra_env),
            "timeout": 300,
            "meta": {"kind": "omb", "bench": bench, "path": path_name, "mode": mode, "range": rng,
                     "validate": validate, "n": len(gpus)}}


def nccl_cfg(bench, set_name, gpus, p2p_level=None, check=True, iters=20, warm=5, tag=""):
    extra = {"GPU_BIND_MODE": "all", "NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": "INIT,TUNING",
             "NCCL_SOCKET_IFNAME": "lo", "LD_LIBRARY_PATH": f"{NCCL_HOME}/lib:{MV}/lib:{LIBPATH}",
             "MPIR_CVAR_ENABLE_GPU": "0"}     # MPI only bootstraps nccl-tests
    if p2p_level:
        extra["NCCL_P2P_LEVEL"] = p2p_level
    args = ["-b", "8", "-e", "1G", "-f", "2", "-g", "1", "-n", str(iters), "-w", str(warm),
            "-c", "1" if check else "0"]
    cmd = [MPIEXEC, "-n", str(len(gpus)), WRAP, f"{NCCLT}/{NCCL[bench]}"] + args
    p2p_tag = f"p2p{p2p_level}" if p2p_level else "p2pdefault"
    cid = f"{bench}__{set_name}__{p2p_tag}{tag}"
    return {"id": cid, "gpus": gpus, "cmd": cmd, "env": _env(gpus, "mv", extra), "timeout": 300,
            "meta": {"kind": "nccl", "bench": bench, "set": set_name, "p2p_level": p2p_level or "default",
                     "n": len(gpus)}}


def check_safe(c):
    g = set(c["gpus"])
    if g & BROKEN_GPUS:
        return f"uses broken GPU(s) {sorted(g & BROKEN_GPUS)}"
    single_nv_pair = len(g) == 2 and g in NV_PAIRS
    m = c["meta"]
    if m["kind"] == "omb" and m["mode"] in ("mv", "ucxipc") and not single_nv_pair:
        return f"mode {m['mode']} can use PCIe P2P/IPC and {sorted(g)} is not one NVLink pair"
    if m["kind"] == "nccl" and m["p2p_level"] != "NVL" and not single_nv_pair:
        return f"NCCL without NCCL_P2P_LEVEL=NVL on {sorted(g)} (not one NVLink pair)"
    return None


def phase_smoke():
    return [omb_cfg("smoke", "osu_latency", "nv23", [2, 3], "S", validate=False)]


def phase_validate():
    c = []
    for b in P2P:                                   # NVLink pair: all three transport modes
        for m in MODES:
            c.append(omb_cfg("validate", b, "nv23", [2, 3], "V", mode=m, validate=True))
    for pn in ["node56", "sys36"]:                  # cross pairs: host staging only
        for b in P2P:
            c.append(omb_cfg("validate", b, pn, PATHS[pn], "V", mode="ucxnoipc", validate=True))
    for b in COLL:
        c.append(omb_cfg("validate", b, "g2_nv23", SETS["g2_nv23"], "V", mode="mv", validate=True))
        c.append(omb_cfg("validate", b, "g4_nv23_5_6", SETS["g4_nv23_5_6"], "V", mode="ucxnoipc", validate=True))
    return c


def phase_p2p():
    c = [omb_cfg("p2p", b, "nv23", [2, 3], rng, mode="mv") for b in P2P for rng in ["S", "L"]]
    c += [omb_cfg("p2p", b, pn, PATHS[pn], rng, mode="ucxnoipc")
          for pn in ["node56", "sys36"] for b in P2P for rng in ["S", "L"]]
    return c


def phase_ucx():
    """UCX transport contrast on the NVLink pair: cuda_ipc allowed vs. cuda_copy only (host staging)."""
    return [omb_cfg("ucx", b, "nv23", [2, 3], rng, mode=m)
            for b in ["osu_latency", "osu_bw"] for m in ["ucxipc", "ucxnoipc"] for rng in ["S", "L"]]


def phase_ucxlog():
    """One run per mode with UCX transport logging, to confirm which transport carried the data."""
    extra = {"UCX_LOG_LEVEL": "info", "UCX_PROTO_INFO": "y"}
    c = []
    for pn, m in [("nv23", "ucxipc"), ("nv23", "ucxnoipc"), ("node56", "ucxnoipc")]:
        cfg = omb_cfg("ucxlog", "osu_bw", pn, PATHS[pn], "S", mode=m, extra_env=extra)
        cfg["cmd"] = [x if x != "1:4194304" else "1048576:4194304" for x in cfg["cmd"]]
        cfg["id"] += "__ucxlog"
        c.append(cfg)
    return c


def phase_coll():
    c = [omb_cfg("coll", b, "g2_nv23", SETS["g2_nv23"], rng, mode="mv") for b in COLL for rng in ["S", "L"]]
    c += [omb_cfg("coll", b, "g4_nv23_5_6", SETS["g4_nv23_5_6"], rng, mode="ucxnoipc") for b in COLL for rng in ["S", "L"]]
    return c


def phase_nccl():
    c = []
    for sn, g in SETS.items():
        lvl = None if sn == "g2_nv23" else "NVL"   # cross-pair set -> NVLink-only P2P (PCIe P2P proven unsafe)
        for b in NCCL:
            c.append(nccl_cfg(b, sn, g, p2p_level=lvl))
    return c


PHASES = {k[len("phase_"):]: v for k, v in globals().items() if k.startswith("phase_")}
