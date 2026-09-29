#!/bin/bash
# Build all registered FlashAttention implementations and run a shape sweep.
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p build

CUDA_ARCH="${CUDA_ARCH:-sm_89}"

nvcc -arch="${CUDA_ARCH}" -O3 -o build/bench_flashAttn \
  bench_flashAttn.cu flashAttn_v*.cu

run_case() {
  local B="$1"
  local H="$2"
  local N="$3"
  local D="$4"
  local CAUSAL="$5"
  printf \
    "\n==== B=%s H=%s N=%s D=%s causal=%s ====\n" \
    "$B" "$H" "$N" "$D" "$CAUSAL"
  ./build/bench_flashAttn "$B" "$H" "$N" "$D" "$CAUSAL"
}

# Cover causal/non-causal paths and dimensions that are not common tile sizes.
run_case 1 4 64 32 0
run_case 1 4 128 64 0
run_case 2 3 127 40 1
