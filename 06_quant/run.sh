#!/bin/bash
# Build the dequantization implementations and run a small shape sweep.
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p build

CUDA_ARCH="${CUDA_ARCH:-sm_89}"

nvcc -arch="${CUDA_ARCH}" -O3 -o build/bench_dequant \
  bench_dequant.cu dequant_v*.cu

run_case() {
  local M="$1"
  local N="$2"
  local TILE_SIZE="$3"
  printf "\n==== M=%s N=%s TILE_SIZE=%s ====\n" "$M" "$N" "$TILE_SIZE"
  ./build/bench_dequant "$M" "$N" "$TILE_SIZE"
}

# Include rectangular and non-divisible shapes to exercise boundary handling.
run_case 1024 1024 128
run_case 2048 4096 128
run_case 4096 2048 64
run_case 1023 2051 127
