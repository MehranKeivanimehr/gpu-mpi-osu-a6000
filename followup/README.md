# Follow-up (2026-11-12, 23:04–23:50 EST)

This follow-up closes three open items from the first study. Nothing in the repo's existing `raw/` was modified. Every shell command
is in `commands.log`.

| item | result | file |
|---|---|---|
| A. GPU 4 status | Still stuck; the driver reports `GPU Recovery Action: Reset`. Kernel log unreadable without root. | `raw/gpu4/` |
| B. IOMMU evidence (read-only, no P2P probes) | All 8 GPUs `DMA-FQ` (translated), no `iommu=pt`; NVIDIA documents this as unsupported for PCIe P2P on bare-metal Linux | `IOMMU_EVIDENCE.md`, `env/iommu/`, `env/docs_fetch/` |
| C. Allreduce cliff | Starts at 32→36 MiB; host buffers too; not an algorithm switch; moves with `MPIR_CVAR_LMEM_POOL_SIZE` | `ALLREDUCE_CLIFF.md`, `PREREGISTRATION.md` |

**Layout**
* `scripts/`
  * `allreduce_sizes.c`: the timing and correctness program;
  * `run_followup.py`: the driver, with safety, idle and budget checks;
  * `analyze_followup.py`, `plot_followup.py`, `nsys_trace_summary.py`, `html2text.py`, `envvar_excerpt.py`;
  * `log.sh`: command logging.
* `raw/<phase>/`: one log per run (`smoke`, `sweep`, `knobs`, `algo`, `nsys`, `glibc`), plus `raw/gpu_time.csv`.
* `raw/cvars/mpivars.txt`: all control variables of this MVAPICH 4.1 install.
* `env/`: documentation excerpts used as DOCUMENTED sources.
* `results/`: `summary_<phase>.csv`, `cliff_<phase>.csv`, `nsys_trace_summary.csv`, `gpu_time_summary.csv` and `plots/`.

**Reproduce** (needs an idle NVLink pair; this setup uses GPUs 2–3):
```
cd ~/mpi-osu-a6000/followup
~/miniconda3/bin/python -I scripts/run_followup.py sweep    # also: knobs, algo, nsys, glibc
~/miniconda3/bin/python -I scripts/analyze_followup.py sweep knobs algo glibc
~/miniconda3/bin/python -I scripts/plot_followup.py
```
The binary is built into `build/`, which is not in git. The build command is the first `mpicc` line in `commands.log`.
