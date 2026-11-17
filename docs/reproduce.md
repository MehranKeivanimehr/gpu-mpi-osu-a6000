# How to rerun everything

## Rebuild the data and figures from the logs (no GPU needed)

```bash
pip install -r requirements.txt
python3 scripts/analyze.py
python3 scripts/analyze_allreduce.py sweep knobs algo glibc gpu_time
python3 scripts/nsys_summary.py raw/allreduce/nsys/trace_dev_32MiB_rank0_cuda_gpu_trace.csv raw/allreduce/nsys/trace_dev_64MiB_rank0_cuda_gpu_trace.csv raw/allreduce/nsys/trace_dev_32MiB_rank1_cuda_gpu_trace.csv raw/allreduce/nsys/trace_dev_64MiB_rank1_cuda_gpu_trace.csv > data/allreduce_nsys.csv
python3 scripts/plot.py
python3 scripts/make_interactive.py
python3 scripts/check_readme_numbers.py
```

`analyze.py` and `analyze_allreduce.py` rewrite the CSVs in `data/` from `raw/`; `git diff data/` should then show no change.
`plot.py` writes `figures/`, and `make_interactive.py` writes `docs/interactive.html`. Both scripts accept `--out` to write
somewhere else.

## Rerun the measurements

Read [p2p_corruption.md](p2p_corruption.md) first. The run scripts refuse configurations that could put traffic on PCIe
peer-to-peer between GPUs that are not NVLink partners, but the GPU numbers in `scripts/configs.py` and
`scripts/run_allreduce.py` are those of my machine. Check your own topology with `nvidia-smi topo -m` and change them before
running anything.

1. Build the libraries as in [build.md](build.md):

   ```bash
   mkdir -p ~/opt/src ~/opt/build
   cd ~/opt/src
   curl -sSLO https://github.com/openucx/ucx/releases/download/v1.22.0/ucx-1.22.0.tar.gz
   curl -sSLO https://mvapich.cse.ohio-state.edu/download/mvapich/mv2/mvapich-4.1.tar.gz
   curl -sSLO https://mvapich.cse.ohio-state.edu/download/mvapich/osu-micro-benchmarks-7.5.3.tar.gz
   curl -sSL -o nccl-tests-2.21.1.tar.gz https://github.com/NVIDIA/nccl-tests/archive/refs/tags/v2.21.1.tar.gz
   ```

   Compare the files with `raw/build/sha256sums.txt`. The build scripts expect UCX and MVAPICH unpacked in `~/opt/build`, and the
   OMB and nccl-tests tarballs in `~/opt/src/omb/` and `~/opt/src/nccl-tests/`:

   ```bash
   tar xzf ucx-1.22.0.tar.gz -C ~/opt/build
   tar xzf mvapich-4.1.tar.gz -C ~/opt/build
   mkdir -p omb nccl-tests && mv osu-micro-benchmarks-7.5.3.tar.gz omb/ && mv nccl-tests-2.21.1.tar.gz nccl-tests/
   ```

   Then, from the repository:

   ```bash
   bash scripts/build_ucx.sh
   bash scripts/build_mvapich.sh
   OMB_EXTRA_LIBS=-lcudart bash scripts/build_omb.sh mvapich-4.1 ~/opt/mvapich-4.1-cuda
   bash scripts/build_nccl_tests.sh
   ```

   The NCCL library and the `libcuda.so` symlink are set up by hand, as described in [build.md](build.md).

2. Build the allreduce program:

   ```bash
   mkdir -p build && ~/opt/mvapich-4.1-cuda/bin/mpicc -O2 -Wall -o build/allreduce_sizes src/allreduce_sizes.c -I/usr/local/cuda-12.4/include -L/usr/local/cuda-12.4/lib64 -Wl,-rpath,/usr/local/cuda-12.4/lib64 -lcudart
   ```

3. Run the first part, phase by phase, in the order I used:

   ```bash
   bash scripts/run_all.sh
   ```

   `campaign.py <phase> --dry` prints the commands without running them.

4. Run the allreduce follow-up, in the order I used:

   ```bash
   python3 scripts/run_allreduce.py smoke --reps 1
   python3 scripts/run_allreduce.py sweep
   python3 scripts/run_allreduce.py knobs
   python3 scripts/run_allreduce.py algo
   python3 scripts/run_allreduce.py nsys --reps 1
   python3 scripts/run_allreduce.py glibc
   ```

   Each phase writes `raw/allreduce/<phase>/`, skips runs whose log already exists, and stops before 3300 GPU-seconds in total.
   The `nsys` phase needs Nsight Systems; the path is set in `scripts/run_allreduce.py`.

Every run checks that its GPUs are idle before it starts and records an `nvidia-smi` snapshot in its log. The scripts append
every command they launch to `commands.log`, which git ignores.

## What not to run

`src/p2p_probe.cu`, `scripts/run_p2p_probe.sh` and `scripts/diag_hang.sh` produced the evidence in
[p2p_corruption.md](p2p_corruption.md). On a machine where the IOMMU translates GPU DMA, the probe corrupted data and left a GPU
that needed a reset. Do not run them on a shared machine, and do not run them across GPUs that are not NVLink partners unless
the IOMMU is off or in passthrough mode for those GPUs.
