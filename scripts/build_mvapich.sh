#!/bin/bash
# Build MVAPICH 4.1 (open-source, MPICH-4.3.0 based) with CUDA support from the source tarball.
# Attempt 1 failed: MPL searched $CUDA/lib for libcudart; CUDA 12.4 uses lib64, attempt 2 (--with-cuda-lib) also ignored;
# so add -L$CUDA/lib64 (+rpath) to LDFLAGS directly.
# Device ch4:ucx against our UCX 1.22.0; yaksa (datatype engine) GPU kernels built for sm_86 (RTX A6000) only.
# Process manager: gforker (single-node fork launcher, PMI over AF_UNIX socketpairs -> no TCP listener).
# hydra was dropped because its control socket binds INADDR_ANY (src/pm/hydra/lib/utils/sock.c -> MPL_listen_anyport).
set -euo pipefail
VER=4.1
CUDA=/usr/local/cuda-12.4
UCX=$HOME/opt/ucx-1.22.0
PREFIX=$HOME/opt/mvapich-$VER-cuda
BLD=$HOME/opt/build/mvapich-$VER-objdir
mkdir -p $BLD && cd $BLD
$HOME/opt/build/mvapich-$VER/configure --prefix=$PREFIX \
  --with-device=ch4:ucx --with-ucx=$UCX \
  --with-cuda=$CUDA --with-cuda-sm=86 \
  --with-pm=gforker --disable-fortran --enable-fast=O3 \
  LDFLAGS="-L$CUDA/lib64 -Wl,-rpath,$CUDA/lib64 -L$CUDA/lib64/stubs"
make -j48
make install
