#!/bin/bash
# Build nccl-tests 2.21.1 (MPI=1) against MVAPICH 4.1 and NCCL 2.21.5 from the nvidia-nccl-cu12 wheel (see docs/build.md).
set -euo pipefail
VER=2.21.1
CUDA=/usr/local/cuda-12.4
MPI=$HOME/opt/mvapich-4.1-cuda
NCCL=$HOME/opt/nccl-2.21.5-pipwheel
DST=$HOME/opt/nccl-tests-$VER-mvapich-4.1
rm -rf $DST && mkdir -p $DST
tar xzf $HOME/opt/src/nccl-tests/nccl-tests-$VER.tar.gz -C $DST --strip-components=1
cd $DST
make -j16 MPI=1 MPI_HOME=$MPI CUDA_HOME=$CUDA NCCL_HOME=$NCCL NVCC_GENCODE="-gencode=arch=compute_86,code=sm_86"
