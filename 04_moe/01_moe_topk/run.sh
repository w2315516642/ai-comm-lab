#!/bin/bash
# Build all registered MoE top-k implementations and run a small shape sweep.
set -euo pipefail

cd "$(dirname "$0")"

mkdir -p build

CUDA_ARCH="${CUDA_ARCH:-sm_89}"

nvcc -arch="${CUDA_ARCH}" -O3 -o build/bench_moe_topk \
  bench_moe_topk.cu \
  moe_topk_v*.cu moe_topk_ans*.cu

run_case() {
  local M="$1"
  local E="$2"
  local K="$3"
  printf "\n==== M=%s E=%s k=%s ====\n" "$M" "$E" "$K"
  ./build/bench_moe_topk "$M" "$E" "$K"
}

# Shape sweep: M E k
run_case 1024 64 2
run_case 4096 64 2
run_case 4096 128 2
run_case 4096 256 2
run_case 4096 128 4
