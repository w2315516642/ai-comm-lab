#!/bin/bash
# Profile one registered MoE top-k implementation with Nsight Compute.
#
# Usage:
#   ./ncu.sh [impl] [M] [E] [k] [ncu_set] [extra ncu args...]
#
# Examples:
#   ./ncu.sh v0 4096 128 2
#   ./ncu.sh ans0 4096 128 2 basic
#   ./ncu.sh ans0 4096 128 2 speed-of-light
#   ./ncu.sh 2 4096 256 2 full --section SourceCounters
#   NCU_COPY_DIR=/mnt/e/custom_reports ./ncu.sh v0 4096 128 2
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v ncu >/dev/null 2>&1; then
  echo "error: ncu not found in PATH" >&2
  exit 1
fi

impl="${1:-v0}"
M="${2:-4096}"
E="${3:-128}"
K="${4:-2}"
NCU_SET="${5:-basic}"
COPY_DIR="${NCU_COPY_DIR:-/mnt/e/ncu_reports}"
EXTRA_NCU_ARGS=()
if [ "$#" -gt 5 ]; then
  EXTRA_NCU_ARGS=("${@:6}")
fi

case "$impl" in
0 | v0 | v0-baseline)
  IMPL_IDX=0
  IMPL_NAME="v0"
  KERNEL_REGEX="^moe_topk_kernel_v0"
  ;;
1 | ans0)
  IMPL_IDX=1
  IMPL_NAME="ans0"
  KERNEL_REGEX="^topk_kernel"
  ;;
2 | ans1)
  IMPL_IDX=2
  IMPL_NAME="ans1"
  KERNEL_REGEX="^moe_topk\\("
  ;;
3 | ans2)
  IMPL_IDX=3
  IMPL_NAME="ans2"
  KERNEL_REGEX="^topk_gating_kernel"
  ;;
4 | v1 | v1-warp-per-t)
  IMPL_IDX=4
  IMPL_NAME="v1-warp-per-t"
  KERNEL_REGEX="^moe_topk_kernel_v1"
  ;;
*)
  echo "error: unknown impl '$impl' (use v0/ans0/ans1/ans2 or 0/1/2/3)" >&2
  exit 1
  ;;
esac

mkdir -p build/ncu_reports

CUDA_ARCH="${CUDA_ARCH:-sm_89}"
nvcc -arch="${CUDA_ARCH}" -O3 -lineinfo -o build/bench_moe_topk_ncu \
  bench_moe_topk.cu \
  moe_topk_v*.cu moe_topk_ans*.cu

safe_set="${NCU_SET//[^A-Za-z0-9_.-]/_}"
report="build/ncu_reports/${IMPL_NAME}_M${M}_E${E}_k${K}_${safe_set}"

NCU_PROFILE_ARGS=()
case "$NCU_SET" in
speed-of-light | sol)
  NCU_PROFILE_ARGS=(
    --section LaunchStats
    --section Occupancy
    --section SpeedOfLight
    --section WorkloadDistribution
  )
  ;;
*)
  NCU_PROFILE_ARGS=(--set "$NCU_SET")
  ;;
esac

printf "Profiling impl=%s idx=%s kernel_regex=%s M=%s E=%s k=%s set=%s\n" \
  "$IMPL_NAME" "$IMPL_IDX" "$KERNEL_REGEX" "$M" "$E" "$K" "$NCU_SET"
printf "Report: %s.ncu-rep\n" "$report"

MOE_TOPK_SKIP_CORRECT=1 ncu \
  "${NCU_PROFILE_ARGS[@]}" \
  --target-processes all \
  --kernel-name "regex:${KERNEL_REGEX}" \
  --launch-count 1 \
  --force-overwrite \
  --export "$report" \
  "${EXTRA_NCU_ARGS[@]}" \
  ./build/bench_moe_topk_ncu "$M" "$E" "$K" 1 0 "$IMPL_IDX"

report_file="${report}.ncu-rep"
if [ -d "$COPY_DIR" ] && [ -f "$report_file" ]; then
  cp -f "$report_file" "$COPY_DIR/"
  printf "Copied report to: %s/%s\n" "$COPY_DIR" "$(basename "$report_file")"
else
  printf "Skip copy: destination not found or report missing: %s\n" "$COPY_DIR"
fi
