#!/bin/bash
set -euo pipefail


SRC="$1"

clang -O2 -std=c++17 \
    -x aica -D__AICA__ -D__AICA_ARCH__=900 \
    -L/usr/local/aica/lib \
    -laicart -fPIC --shared \
    -I/usr/local/aica/include \
    -I/root/work/tilelang_aica/src \
    -I/root/work/cutlass/include \
    "$SRC" -o kernel_lib.so
