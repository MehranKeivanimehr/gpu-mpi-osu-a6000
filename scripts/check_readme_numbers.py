#!/usr/bin/env python3
"""Check every number in README.md against the CSV, log or git record it comes from.

Each check computes the text the README should contain from a source file and asserts that the text is there. Afterwards every
number left in the README prose must be inside one of the checked texts. Inline code with digits must be an existing path or
appear verbatim in a file of the repository. The Mermaid diagram is checked against the nvidia-smi topology.
Usage: check_readme_numbers.py   (exit status 1 on any failure)
"""
import gzip
import os
import re
import subprocess
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README = open(f"{REPO}/README.md", encoding="utf-8").read()
FAILS, CHECKED = [], []
NUM = re.compile(r"(?<![\w./\-])\d+(?:,\d{3})*(?:\.\d+)*(?![\w])")


def read(rel):
    path = f"{REPO}/{rel}"
    if path.endswith(".gz"):
        return gzip.open(path, "rt", errors="replace").read()
    return open(path, encoding="utf-8", errors="replace").read()


def csv(rel):
    return pd.read_csv(f"{REPO}/{rel}")


def norm(text):
    return re.sub(r"\s+", " ", text).strip()


def fmt(x, nd=2):
    # same rule as fmt() in scripts/analyze.py
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    if abs(x) >= 100:
        return f"{x:.1f}"
    return f"{x:.{nd}f}"


def split_readme(text):
    fences = re.findall(r"```(\w*)\n(.*?)```", text, re.S)
    prose = re.sub(r"```\w*\n.*?```", " ", text, flags=re.S)
    prose = re.sub(r"(?m)^\s*\d+\.\s", " ", prose)          # ordered-list markers
    prose = re.sub(r"\]\([^)]*\)", "]", prose)               # link targets
    return fences, prose


FENCES, PROSE = split_readme(README)


def code_spans_ok(prose):
    files = []
    for root, dirs, names in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in (".git", "build", "__pycache__")]
        files += [os.path.join(root, n) for n in names if not n.endswith((".png", ".gif", ".html"))]
    corpus = "\n".join(read(os.path.relpath(f, REPO)) for f in files)
    out = prose
    for span in sorted(set(re.findall(r"`([^`]+)`", prose))):
        if not re.search(r"\d", span):
            continue
        ok = os.path.exists(f"{REPO}/{span.rstrip('/')}") or span in corpus
        CHECKED.append((f"code `{span}`", "path exists" if os.path.exists(f"{REPO}/{span.rstrip('/')}") else "found in repo"))
        if not ok:
            FAILS.append(f"inline code `{span}` is neither a path nor text found in the repository")
        out = out.replace(f"`{span}`", "`code`")
    return out


PROSE_N = norm(code_spans_ok(PROSE))
COVER = [False] * len(PROSE_N)


def claim(text, source, ok=True):
    """Assert that `text` is in the README prose and that the source agrees; mark its numbers as checked."""
    text = norm(text)
    hits = [m.start() for m in re.finditer(re.escape(text), PROSE_N)]
    if not ok:
        FAILS.append(f"source does not support '{text}' ({source})")
    if not hits:
        FAILS.append(f"README does not contain '{text}' ({source})")
    for h in hits:
        for i in range(h, h + len(text)):
            COVER[i] = True
    CHECKED.append((text, source))


# results
p = csv("data/p2p_summary.csv")
bw = lambda path, mode: p[(p.bench == "osu_bw") & (p.path == path) & (p["mode"] == mode) & (p["size"] == 1 << 30)].iloc[0]
claim(f"reached {bw('nv23', 'mv')['median']:,.0f} MB/s at 1 GiB", "data/p2p_summary.csv: osu_bw nv23 mv, size 1073741824")
claim(f"it reached {bw('nv23', 'ucxnoipc')['median']:,.0f} MB/s", "data/p2p_summary.csv: osu_bw nv23 ucxnoipc, size 1073741824")
nreps = set(p.nreps) | set(csv("data/coll_mpi_summary.csv").t_nreps) | set(csv("data/coll_nccl_summary.csv").t_nreps)
nreps |= set(csv("data/allreduce_sweep.csv").n_runs) | set(csv("data/allreduce_knobs.csv").n_runs)
claim("of 5 runs", "runs per point in data/*summary*.csv and data/allreduce_*.csv", nreps == {5})

sw = csv("data/allreduce_sweep.csv").set_index(["config", "size_mib"])
cs = csv("data/allreduce_cliff_sweep.csv").set_index("config")
c = cs.loc["dev_default"]
claim(f"between {c.cliff_from_mib} and {c.cliff_to_mib} MiB, from {sw.loc[('dev_default', c.cliff_from_mib)].t_median_us:,.2f} "
      f"to {sw.loc[('dev_default', c.cliff_to_mib)].t_median_us:,.2f} microseconds per call",
      "data/allreduce_cliff_sweep.csv and data/allreduce_sweep.csv, dev_default")
kn = csv("data/allreduce_knobs.csv").set_index(["config", "size_mib"])
slow, fast = kn.loc[("dev_default", 128)].t_median_us, kn.loc[("dev_lmem128M", 128)].t_median_us
default_pool = re.search(r"MPIR_CVAR_LMEM_POOL_SIZE.*?default\s*:\s*32 \* 1024 \* 1024", read("raw/verify/lmem_pool_source.txt"), re.S)
claim(f"raising it from 32 to 128 MiB made a 128 MiB allreduce {slow / fast:.1f} times faster ({slow:,.2f} to {fast:,.2f} microseconds)",
      "data/allreduce_knobs.csv dev_default and dev_lmem128M at 128 MiB; default pool size from raw/verify/lmem_pool_source.txt",
      bool(default_pool))
pools = sorted({int(m) for m in re.findall(r"dev_lmem(\d+)M", " ".join(kn.index.get_level_values(0)))} | {32})
claim(f"steps through {', '.join(map(str, pools[:-1]))} and {pools[-1]} MiB", "configs in data/allreduce_knobs.csv")

probe34, probe23 = read("raw/p2p_probe/probe_3_4_dma.log"), read("raw/p2p_probe/probe_2_3_dma.log")
nbytes, bad = map(int, re.findall(r"memcpyPeer peer_access=1 bytes=(\d+) .* bad_words=(\d+)", probe34)[-1])
claim(f"A {nbytes >> 20} MiB CUDA peer copy across PCIe between GPUs 3 and 4", "raw/p2p_probe/probe_3_4_dma.log",
      "CUDA_VISIBLE_DEVICES=3,4" in probe34)
claim(f"returned {bad:,} wrong words out of {nbytes // 4:,}", "raw/p2p_probe/probe_3_4_dma.log, 4-byte words")
ok23 = re.search(rf"memcpyPeer peer_access=1 bytes={nbytes} .* bad_words=0", probe23) and "CUDA_VISIBLE_DEVICES=2,3" in probe23
claim("between the NVLink partners 2 and 3 was correct", "raw/p2p_probe/probe_2_3_dma.log", bool(ok23))

# topology, hardware and versions
topo = read("raw/env/nvidia-smi_topo.txt")
rows = {int(r[0][3:]): r for r in (l.split() for l in topo.splitlines()) if r and re.fullmatch(r"GPU\d", r[0])}
nv_pairs = sorted({tuple(sorted((g, j))) for g, r in rows.items() for j, v in enumerate(r[1:9]) if v == "NV4"})
numa = {g: int(r[10]) for g, r in rows.items()}
claim("GPUs 2 and 3", "scripts/configs.py g2_nv23 and scripts/run_allreduce.py PAIR; NVLink pair in raw/env/nvidia-smi_topo.txt",
      (2, 3) in nv_pairs and '"g2_nv23": [2, 3]' in read("scripts/configs.py") and "PAIR = [2, 3]" in read("scripts/run_allreduce.py"))
smi_q = read("raw/env/nvidia-smi_q.txt")
ngpu = int(re.search(r"Attached GPUs\s*:\s*(\d+)", smi_q).group(1))
claim(f"{ngpu}-GPU RTX A6000 node", "raw/env/nvidia-smi_q.txt Attached GPUs")
claim(f"with {ngpu} NVIDIA RTX A6000 GPUs", "raw/env/nvidia-smi_q.txt Attached GPUs")
mem = re.search(r"FB Memory Usage\s*\n\s*Total\s*:\s*(\d+) MiB", smi_q).group(1)
gen = re.search(r"PCIe Generation\s*\n\s*Max\s*:\s*(\d+)", smi_q).group(1)
width = re.search(r"Link Width\s*\n\s*Max\s*:\s*(\d+)x", smi_q).group(1)
claim(f"| GPUs | {ngpu} x NVIDIA RTX A6000, {mem} MiB each, PCIe Gen{gen} x{width} |", "raw/env/nvidia-smi_q.txt")
lscpu = read("raw/env/lscpu.txt")
sockets = re.search(r"Socket\(s\):\s*(\d+)", lscpu).group(1)
model = re.search(r"Model name:\s*AMD (EPYC \d+)", lscpu).group(1)
claim(f"| CPUs | {sockets} x AMD {model} |", "raw/env/lscpu.txt")
osd = read("raw/env/os_driver.txt")
ubuntu = re.search(r'PRETTY_NAME="Ubuntu ([\d.]+ LTS)"', osd).group(1)
kernel = re.search(r"^Linux \S+ (\S+)", osd).group(1)
claim(f"| OS | Ubuntu {ubuntu}, kernel {kernel} |", "raw/env/os_driver.txt")
driver = re.search(r"Driver Version: ([\d.]+)", read("raw/env/nvidia-smi.txt")).group(1)
cuda = re.search(r"release ([\d.]+),", read("raw/env/cuda_12.4.txt")).group(1)
claim(f"| driver and CUDA | NVIDIA driver {driver}, CUDA toolkit {cuda} |", "raw/env/nvidia-smi.txt, raw/env/cuda_12.4.txt")
ucx = re.search(r"Library version: ([\d.]+)", read("raw/verify/ucx_info_v.txt")).group(1)
claim(f"| UCX | {ucx}, built with", "raw/verify/ucx_info_v.txt")
mv = re.search(r"MVAPICH Version:\s*([\d.]+)", read("raw/verify/mvapich_mpichversion.txt")).group(1)
mpich = re.search(r"MPICH_VERSION_m4\],\[([\d.]+)\]", read("raw/verify/lmem_pool_source.txt")).group(1)
claim(f"| MVAPICH | {mv}, based on MPICH {mpich}, built with", "raw/verify/mvapich_mpichversion.txt, raw/verify/lmem_pool_source.txt")
claim(f"MVAPICH {mv}", "raw/verify/mvapich_mpichversion.txt")
claim(f"MPICH {mpich} source", "raw/verify/lmem_pool_source.txt (maint/version.m4)")
omb = re.search(r"VER=([\d.]+)", read("scripts/build_omb.sh")).group(1)
claim(f"| OSU Micro-Benchmarks | {omb}, built with", "scripts/build_omb.sh; OMB header in raw/p2p/osu_bw__nv23__mv__L__rep1.log",
      f"Bandwidth Test v{omb}" in read("raw/p2p/osu_bw__nv23__mv__L__rep1.log"))
nccl = re.search(r"NCCL version ([\d.]+)\+", read("raw/nccl/all_reduce_perf__g2_nv23__p2pdefault__rep1.log")).group(1)
claim(f"| NCCL | {nccl}, from the", "raw/nccl/*.log NCCL version line; raw/env/python_torch_nccl.txt",
      f"nvidia_nccl_cu12-{nccl}" in read("raw/env/python_torch_nccl.txt"))
nt = re.search(r"VER=([\d.]+)", read("scripts/build_nccl_tests.sh")).group(1)
claim(f"| nccl-tests | {nt}, built with", "scripts/build_nccl_tests.sh")

# MPI and NCCL table
t = csv("data/mpi_vs_nccl_table.csv")
t = t[t.gpu_set == "g2_nv23"]
for _, r in t[t.collective.isin(["allreduce", "allgather"])].iterrows():
    claim(f"| {r.S} | {r.collective} | {fmt(r.mpi_t_med_us)} | {fmt(r.nccl_t_med_us)} | "
          f"{fmt(r.mpi_busbw_med_GBps)} | {fmt(r.nccl_busbw_med_GBps)} |", "data/mpi_vs_nccl_table.csv, g2_nv23")
ar, ag = t[t.collective == "allreduce"], t[t.collective == "allgather"]
claim("NCCL took less time for allreduce at all three sizes", "data/mpi_vs_nccl_table.csv",
      len(ar) == 3 and (ar.nccl_t_med_us < ar.mpi_t_med_us).all())
faster = ag[ag.mpi_t_med_us < ag.nccl_t_med_us].S.tolist()
claim(f"MVAPICH took less time for allgather at {faster[0]} and {faster[1]}", "data/mpi_vs_nccl_table.csv", len(faster) == 2)
cm = csv("data/coll_mpi_summary.csv")
cm = cm[(cm.kind == "allreduce") & (cm.path == "g2_nv23")].set_index("S").busbw_median
claim("drops between 32 and 64 MiB in this OMB sweep", "data/coll_mpi_summary.csv allreduce g2_nv23 bus bandwidth",
      cm[64 << 20] / cm[32 << 20] < 0.6 and cm[32 << 20] / cm[16 << 20] >= 0.6)

# trust list
v = csv("data/validation_summary.csv")
ab = v[v.abnormal_exit]
claim(f"one 4-GPU sweep stopped after its {int(ab.max_size.iloc[0]) >> 20} MiB size had passed", "data/validation_summary.csv",
      len(ab) == 1 and ab.path.iloc[0] == "g4_nv23_5_6" and v.failed.sum() == 0)
claim("4-GPU", "scripts/configs.py g4_nv23_5_6 has 4 GPUs", '"g4_nv23_5_6": [2, 3, 5, 6]' in read("scripts/configs.py"))
w = csv("data/nccl_wrong_summary.csv")
claim("reported 0 wrong results", "data/nccl_wrong_summary.csv", (w.oop_wrong_total.sum() + w.ip_wrong_total.sum()) == 0)
ca = csv("data/allreduce_cliff_algo.csv")
forced = ca[ca.config.str.startswith("dev_dcnone_") & (ca.config != "dev_dcnone_auto")]
claim(f"All {len(forced)} allreduce algorithms that I could force show it", "data/allreduce_cliff_algo.csv",
      forced.cliff_from_mib.notna().all())


def first_commit(path):
    out = subprocess.run(["git", "-C", REPO, "log", "--diff-filter=A", "--format=%ad", "--date=format:%Y-%m-%d %H:%M:%S %z",
                          "--", path], capture_output=True, text=True).stdout.split("\n")
    return [l for l in out if l][-1]


pre, dat = first_commit("followup/PREREGISTRATION.md"), first_commit("followup/raw/sweep")
claim(f"hypotheses at {pre[11:19]} EST on {pre[:10]} and its data at {dat[11:19]} EST the same day",
      "git log: first commits of followup/PREREGISTRATION.md and followup/raw/sweep",
      pre.endswith("-0500") and dat.endswith("-0500") and pre[:10] == dat[:10] and pre < dat)
claim("section 3.4.2.5", "raw/verify/doc_checks.txt", "### 3.4.2.5. Host IOMMU Hardware" in read("raw/verify/doc_checks.txt"))
claim("with `MPI_FLOAT` on 2 ranks", "scripts/run_allreduce.py and src/allreduce_sizes.c",
      '[MPIEXEC, "-n", "2", WRAP]' in read("scripts/run_allreduce.py") and "MPI_FLOAT" in read("src/allreduce_sizes.c"))
smi = read("raw/env/nvidia-smi.txt")
claim("GPUs 0 and 1 during all my measurements", "raw/env/nvidia-smi.txt process table",
      bool(re.search(r"\|\s+0\s+N/A\s+N/A\s+<pid>\s+C\s+<other user's process>", smi))
      and bool(re.search(r"\|\s+1\s+N/A\s+N/A\s+<pid>\s+C\s+<other user's process>", smi)))

# fenced blocks
for lang, body in FENCES:
    if lang == "mermaid":
        decl = {int(g): body[: m.start()] for m in re.finditer(r'gpu(\d)\["GPU \1"\]', body) for g in [m.group(1)]}
        for g, before in decl.items():
            sub = re.findall(r"NUMA node (\d)", before)
            if not sub or int(sub[-1]) != numa[g]:
                FAILS.append(f"mermaid: GPU {g} drawn in the wrong NUMA node")
        drawn = sorted(tuple(sorted(map(int, x))) for x in re.findall(r"gpu(\d)\[[^\]]*\] <-->\|NVLink\| gpu(\d)", body))
        if drawn != nv_pairs or sorted(decl) != list(range(ngpu)):
            FAILS.append(f"mermaid: NVLink pairs {drawn} or GPUs {sorted(decl)} do not match raw/env/nvidia-smi_topo.txt {nv_pairs}")
        if model.split()[1] not in body:
            FAILS.append("mermaid: CPU model differs from raw/env/lscpu.txt")
        extra = set(NUM.findall(body)) - {str(i) for i in range(ngpu)} - {model.split()[1]}
        if extra:
            FAILS.append(f"mermaid: unchecked numbers {sorted(extra)}")
        CHECKED.append(("mermaid diagram", "raw/env/nvidia-smi_topo.txt, raw/env/lscpu.txt"))
    elif lang == "bash":
        for path in re.findall(r"(scripts/\S+\.py|requirements\.txt)", body):
            if not os.path.exists(f"{REPO}/{path}"):
                FAILS.append(f"bash block: {path} does not exist")
        if NUM.findall(body):
            FAILS.append(f"bash block: unchecked numbers {NUM.findall(body)}")
    elif lang == "bibtex":
        year = subprocess.run(["git", "-C", REPO, "log", "-1", "--format=%ad", "--date=format:%Y"],
                              capture_output=True, text=True).stdout.strip()
        if f"year         = {{{year}}}" not in body or f"{ngpu}-GPU" not in body:
            FAILS.append("bibtex: year or GPU count does not match")
        if set(NUM.findall(body)) - {year, str(ngpu)}:
            FAILS.append(f"bibtex: unchecked numbers {set(NUM.findall(body)) - {year, str(ngpu)}}")
        CHECKED.append(("bibtex year and title", "git log year of the last commit; raw/env/nvidia-smi_q.txt"))
    else:
        FAILS.append(f"unexpected code block of type '{lang}'")

for m in NUM.finditer(PROSE_N):
    if not all(COVER[m.start():m.end()]):
        ctx = PROSE_N[max(0, m.start() - 40):m.end() + 40]
        FAILS.append(f"number '{m.group()}' is not covered by any check: ...{ctx}...")

print(f"{len(CHECKED)} checks")
for text, src in CHECKED:
    print(f"  ok? {text[:90]!r:95} <- {src}")
if FAILS:
    print(f"\n{len(FAILS)} FAILED:")
    for f in FAILS:
        print("  -", f)
    sys.exit(1)
print("\nall README numbers match their sources")
