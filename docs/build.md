# How I built the libraries without root

I built everything as a normal user under `~/opt`. The machine had the CUDA 12.4 toolkit in `/usr/local/cuda-12.4` (with
`nvcc` not on `PATH`) and GCC 13.3.0, but no MPI, NCCL, UCX, gfortran or autotools (`raw/env/toolchain.txt`,
`raw/env/cuda_12.4.txt`). Each build script is in `scripts/`, and the full output of every attempt, including the failed ones, is
in `raw/build/` (gzip-compressed).

## Versions and sources

| component | version | source | build script | build log |
|---|---|---|---|---|
| UCX | 1.22.0 | https://github.com/openucx/ucx/releases/download/v1.22.0/ucx-1.22.0.tar.gz | `build_ucx.sh` | `ucx-1.22.0_build.log.gz` |
| MVAPICH | 4.1 (based on MPICH 4.3.0) | https://mvapich.cse.ohio-state.edu/download/mvapich/mv2/mvapich-4.1.tar.gz | `build_mvapich.sh` | `mvapich-4.1_build.log.gz` |
| OSU Micro-Benchmarks | 7.5.3 | https://mvapich.cse.ohio-state.edu/download/mvapich/osu-micro-benchmarks-7.5.3.tar.gz | `build_omb.sh` | `omb-7.5.3-mvapich-4.1_build.log.gz` |
| NCCL | 2.21.5 | the `nvidia-nccl-cu12` 2.21.5 wheel of a PyTorch 2.6.0+cu124 environment | none (symlinks) | |
| nccl-tests | 2.21.1 | https://github.com/NVIDIA/nccl-tests/archive/refs/tags/v2.21.1.tar.gz | `build_nccl_tests.sh` | `nccl-tests-2.21.1_build.log.gz` |
| Open MPI | 5.0.11 (built, not used) | https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.11.tar.bz2 | `build_openmpi.sh` | `openmpi-5.0.11_build.log.gz` |

The sha256 sums of the tarballs are in `raw/build/sha256sums.txt`. The MPICH base version comes from
`maint/version.m4` in the MVAPICH tarball (`raw/verify/lmem_pool_source.txt`).

## Flags

* UCX 1.22.0: `--with-cuda=/usr/local/cuda-12.4 --without-verbs --without-rdmacm --without-knem --without-xpmem --without-gdrcopy
  --without-java --without-go --enable-mt --enable-cma --disable-debug --disable-assertions --disable-params-check
  --enable-compiler-opt=3`. The resulting transports were `self, tcp, sysv, posix, cuda_copy, cuda_ipc, cma`
  (`raw/verify/ucx_info_d.txt`). gdrcopy was not possible: it needs a kernel module that only root can load.
* MVAPICH 4.1: `--with-device=ch4:ucx --with-ucx=$HOME/opt/ucx-1.22.0 --with-cuda=/usr/local/cuda-12.4 --with-cuda-sm=86
  --with-pm=gforker --disable-fortran --enable-fast=O3`, with `LDFLAGS="-L$CUDA/lib64 -Wl,-rpath,$CUDA/lib64 -L$CUDA/lib64/stubs"`.
  The configure line the library reports is in `raw/verify/mvapich_mpichversion.txt`.
* OSU Micro-Benchmarks 7.5.3: `CC=mpicc CXX=mpicxx --enable-cuda --with-cuda-include=$CUDA/include
  --with-cuda-libpath=$CUDA/lib64 --with-cuda-binpath=$CUDA/bin NVCC=$CUDA/bin/nvcc NVCCFLAGS=-arch=sm_86`, with
  `LDFLAGS="-L$CUDA/lib64/stubs ..."` and, for MVAPICH, `LIBS=-lcudart`.
* nccl-tests 2.21.1: `make MPI=1 MPI_HOME=$HOME/opt/mvapich-4.1-cuda CUDA_HOME=/usr/local/cuda-12.4
  NCCL_HOME=$HOME/opt/nccl-2.21.5-pipwheel NVCC_GENCODE=-gencode=arch=compute_86,code=sm_86`.
* NCCL: I linked the wheel's `nccl.h`, `nccl_net.h` and `libnccl.so.2` into `~/opt/nccl-2.21.5-pipwheel/{include,lib}` and added
  `libnccl.so -> libnccl.so.2`. I used this version on purpose: it is the one from the PyTorch stack where the original GPU 0
  and 2 problem appeared (`raw/env/python_torch_nccl.txt`).
* Open MPI 5.0.11: `--with-cuda=$CUDA --with-cuda-libdir=$CUDA/lib64/stubs --with-ucx=$HOME/opt/ucx-1.22.0
  --disable-mpi-fortran --without-verbs --without-ofi --without-psm2 --without-ucc --without-hcoll`.

## Problems I hit and what I did

1. MVAPICH configure failed twice in the MPL sub-configure: `--with-cuda` looked for `libcudart` in `$CUDA/lib` instead of
   `lib64`, and `--with-cuda-lib` was not passed to the link test. Putting `-L$CUDA/lib64` in `LDFLAGS` fixed it
   (`mvapich-4.1_build_attempt1.log.gz`, `mvapich-4.1_build_attempt2.log.gz`).
2. There is no unversioned `libcuda.so` outside `$CUDA/lib64/stubs`, so `-lcuda` failed. I linked against the stub; at run time
   the driver's `libcuda.so.1` is used.
3. The first MVAPICH run stopped at `mpl_gpu_cuda.c:489: gpu_mem_hook_init: Assertion 'libcuda_handle' failed`, because MPL calls
   `dlopen("libcuda.so")` with the unversioned name. I made a symlink `~/opt/cuda-driver-link/libcuda.so ->
   /usr/lib/x86_64-linux-gnu/libcuda.so.1` and put that directory first in `LD_LIBRARY_PATH` for every run.
4. OMB with MVAPICH's `mpicc` failed to link (`DSO missing from command line` for `libcudart`). `LIBS=-lcudart` fixed it.
5. OMB needs `nvcc` for `util/kernel.cpp`, and `nvcc` was not on `PATH`. `--with-cuda-binpath` fixed it.
6. `osu_latency --help` in OMB 7.5.3 crashed with `*** buffer overflow detected ***` (`raw/verify/omb_osu_latency_help.txt`), so I
   read the options from `c/util/osu_util_options.h` in the OMB source instead.

## Launcher and network listeners

I did not want any process listening on the network.

* MVAPICH's default launcher, Hydra, binds its control socket to `INADDR_ANY` (`src/pm/hydra/lib/utils/sock.c`). I built MVAPICH
  with the gforker launcher instead. It works on one node only and uses `socketpair(AF_UNIX)` for PMI
  (`src/pm/util/pmiserv.c`). During an MPI job, `ss -ltnup` showed no listener (`raw/verify/listen_mvapich_gforker.txt` is
  empty), and `raw/verify/listen_during_campaign.txt` sampled every 15 s during the measurements.
* Open MPI's launcher `prterun` always listened on `0.0.0.0` (`raw/verify/listen_ompi_default.txt`). The source binds
  `INADDR_ANY` unconditionally (`3rd-party/prrte/src/mca/oob/tcp/oob_tcp_listener.c`), and restricting it to `lo` failed. So I
  built Open MPI but did not benchmark it. Before I found this, I ran five short Open MPI test launches of 8 s or less, and each
  had that listener open briefly.
* UCX's `tcp` transport was never enabled (`UCX_TLS` always listed explicit transports), and NCCL bootstrapped on loopback
  (`NCCL_SOCKET_IFNAME=lo`).

## Checks that the build is CUDA-aware

* `MPIX_GPU_query_support(MPIX_GPU_SUPPORT_CUDA)` and `MPIX_Query_cuda_support()` both returned 1 with two ranks on GPUs 2 and 3
  ([`src/cuda_aware_check.c`](../src/cuda_aware_check.c), `raw/verify/mvapich_cuda_aware_check.txt`).
* `libmpi.so` links `libcudart.so.12`, `libcuda.so.1` and UCX (`raw/verify/mvapich_ldd_libmpi.txt`).
* `ucx_info -d` lists the `cuda_copy` and `cuda_ipc` transports (`raw/verify/ucx_info_d.txt`).

## The small test programs

```bash
~/opt/mvapich-4.1-cuda/bin/mpicc -o build/cuda_aware_check src/cuda_aware_check.c
~/opt/mvapich-4.1-cuda/bin/mpicc -o build/init_sleep src/init_sleep.c
~/opt/mvapich-4.1-cuda/bin/mpicc -O2 -Wall -o build/allreduce_sizes src/allreduce_sizes.c -I/usr/local/cuda-12.4/include -L/usr/local/cuda-12.4/lib64 -Wl,-rpath,/usr/local/cuda-12.4/lib64 -lcudart
```

`src/p2p_probe.cu` was built with `/usr/local/cuda-12.4/bin/nvcc -O2 -arch=sm_86`. Read the warning in
[p2p_corruption.md](p2p_corruption.md) before building or running it.
