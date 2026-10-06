#!/bin/bash
# Build all registered FlashAttention implementations and run a shape sweep.
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p build

CUDA_ARCH="${CUDA_ARCH:-sm_89}"

nvcc -arch="${CUDA_ARCH}" -O3 -o build/bench_flashAttn \
  bench_flashAttn.cu flashAttn_v*.cu

run_case() {
  local N="$1"
  local D="$2"
  local CAUSAL="$3"
  printf \
    "\n==== N=%s D=%s causal=%s ====\n" \
    "$N" "$D" "$CAUSAL"
  ./build/bench_flashAttn "$N" "$D" "$CAUSAL"
}

# Cover causal/non-causal paths and dimensions that are not common tile sizes.
run_case 64 32 0
run_case 128 64 0
run_case 1024 128 0
