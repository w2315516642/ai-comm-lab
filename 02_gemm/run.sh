#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p build

CUDA_ARCH="${CUDA_ARCH:-sm_89}"

nvcc -arch="${CUDA_ARCH}" -O3 \
  --expt-relaxed-constexpr \
  -I../third_party/cutlass/include \
  -o build/bench_gemm \
  bench_gemm.cu gemm_v0.cu gemm_v1.cu gemm_v3.cu gemm_v4.cu gemm_v5.cu \
  gemm_v6.cu gemm_v7.cu gemm_v8.cu gemm_cutlass.cu

run_case() {
  local M="$1"
  local N="$2"
  local K="$3"
  printf "\n==== M=%s N=%s K=%s ====\n" "$M" "$N" "$K"
  ./build/bench_gemm "$M" "$N" "$K"
}

# Include non-multiples of the tile size to exercise boundary handling.
# run_case 127 129 131
# run_case 256 256 256
# run_case 511 257 263
# run_case 512 512 512
# run_case 1024 1024 512
run_case 1024 1024 1024
