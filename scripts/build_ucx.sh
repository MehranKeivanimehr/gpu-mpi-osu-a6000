#!/bin/bash
# Build UCX 1.22.0 with CUDA into $HOME/opt (user space, no root).
# No verbs/rdmacm/knem/xpmem/gdrcopy on this machine (no headers / kernel modules; gdrcopy needs a root-installed kernel module).
set -euo pipefail
VER=1.22.0
CUDA=/usr/local/cuda-12.4
PREFIX=$HOME/opt/ucx-$VER
cd $HOME/opt/build/ucx-$VER
./configure --prefix=$PREFIX --with-cuda=$CUDA \
  --without-verbs --without-rdmacm --without-knem --without-xpmem --without-gdrcopy \
  --without-java --without-go \
  --enable-mt --enable-cma --disable-debug --disable-assertions --disable-params-check --enable-compiler-opt=3
make -j32
make install
