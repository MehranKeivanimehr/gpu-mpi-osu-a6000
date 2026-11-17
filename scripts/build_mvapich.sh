#!/bin/bash
# Build MVAPICH 4.1 (based on MPICH 4.3.0) with CUDA, ch4:ucx and the gforker launcher.
# -L$CUDA/lib64 goes in LDFLAGS because MPL's configure looks for libcudart in $CUDA/lib and ignores --with-cuda-lib.
# gforker uses AF_UNIX socketpairs for PMI; hydra binds INADDR_ANY (src/pm/hydra/lib/utils/sock.c).
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
