#!/bin/bash
# Build Open MPI 5.0.11 with CUDA and UCX. I built it but did not use it (see docs/method.md).
set -euo pipefail
VER=5.0.11
CUDA=/usr/local/cuda-12.4
UCX=$HOME/opt/ucx-1.22.0
PREFIX=$HOME/opt/openmpi-$VER-ucx-cuda
BLD=$HOME/opt/build/openmpi-$VER-objdir
mkdir -p $BLD && cd $BLD
$HOME/opt/build/openmpi-$VER/configure --prefix=$PREFIX \
  --with-cuda=$CUDA --with-cuda-libdir=$CUDA/lib64/stubs \
  --with-ucx=$UCX \
  --disable-mpi-fortran --without-verbs --without-ofi --without-psm2 --without-ucc --without-hcoll
make -j32
make install
