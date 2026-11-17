#!/bin/bash
# Build OSU Micro-Benchmarks 7.5.3 with CUDA against a given MPI.
# Usage: build_omb.sh <tag> <mpi_prefix>     e.g. build_omb.sh ompi-5.0.11 $HOME/opt/openmpi-5.0.11-ucx-cuda
# With MVAPICH set OMB_EXTRA_LIBS=-lcudart; its mpicc does not add libcudart to the link line.
# The unversioned libcuda.so exists only in $CUDA/lib64/stubs, so the build links against the stub.
set -euo pipefail
VER=7.5.3
TAG=$1; MPI=$2
CUDA=/usr/local/cuda-12.4
PREFIX=$HOME/opt/omb-$VER-$TAG
SRC=$HOME/opt/build/omb-$VER-$TAG
rm -rf $SRC && mkdir -p $SRC
tar xzf $HOME/opt/src/omb/osu-micro-benchmarks-$VER.tar.gz -C $SRC --strip-components=1
cd $SRC
./configure --prefix=$PREFIX CC=$MPI/bin/mpicc CXX=$MPI/bin/mpicxx \
  --enable-cuda --with-cuda-include=$CUDA/include --with-cuda-libpath=$CUDA/lib64 --with-cuda-binpath=$CUDA/bin \
  NVCC=$CUDA/bin/nvcc NVCCFLAGS="-arch=sm_86" \
  LDFLAGS="-L$CUDA/lib64/stubs -Wl,-rpath,$CUDA/lib64 -Wl,-rpath,$MPI/lib" LIBS="${OMB_EXTRA_LIBS:-}"
make -j16
make install
