#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p build

# 为另一台机器交叉编译时，可以手动设置 CUDA_ARCH。
if [[ -z "${CUDA_ARCH:-}" ]]; then
  GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
  GPU_ID="${GPU_ID%%,*}"
  GPU_CC=$(nvidia-smi -i "${GPU_ID}" --query-gpu=compute_cap \
    --format=csv,noheader,nounits 2>/dev/null) || GPU_CC=""
  GPU_CC="${GPU_CC//[[:space:]]/}"
  if [[ ! "${GPU_CC}" =~ ^[0-9]+\.[0-9]+$ ]]; then
    echo "Cannot detect GPU architecture; set CUDA_ARCH=sm_89 or sm_90a explicitly." >&2
    exit 1
  fi
  CUDA_ARCH="sm_${GPU_CC//./}"
fi
# WGMMA 需要启用 Hopper 架构专用特性的编译目标。
if [[ "${CUDA_ARCH}" == "sm_90" ]]; then
  CUDA_ARCH=sm_90a
fi

ARCH_FLAGS=(-arch="${CUDA_ARCH}")
EXTRA_SOURCES=()
EXTRA_FLAGS=()
EXTRA_LIBS=()
CUTLASS_SOURCE=gemm_cutlass.cu
if [[ "${CUDA_ARCH}" == "sm_90a" ]]; then
  # 只生成 sm_90a 机器码，避免 nvcc 同时生成通用的 compute_90 PTX 回退版本。
  ARCH_FLAGS=(-gencode=arch=compute_90a,code=sm_90a)
  EXTRA_SOURCES+=(gemm_v9.cu gemm_v10.cu gemm_v11.cu gemm_v12.cu)
  EXTRA_FLAGS+=(-DENABLE_GEMM_V9 -DENABLE_GEMM_V10 -DENABLE_GEMM_V11 -DENABLE_GEMM_V12)
  # v10/v11/v12 使用 Driver API 构造 TMA tensor map。
  EXTRA_LIBS+=(-lcuda)
  CUTLASS_SOURCE=gemm_cutlass_sm90.cu
fi

echo "Building for ${CUDA_ARCH} (optional sources: ${EXTRA_SOURCES[*]:-none})"
echo "CUTLASS backend: ${CUTLASS_SOURCE} (autotune on first call)"
# 使用发布构建：CUTLASS 的设备端断言调用可能让 WGMMA 流水线被强制串行化。
# NDEBUG 只关闭 assert，不关闭 bench 的数值正确性检查。
nvcc "${ARCH_FLAGS[@]}" -std=c++17 -O3 -DNDEBUG "${EXTRA_FLAGS[@]}" \
  --expt-relaxed-constexpr \
  -I../third_party/cutlass/include \
  -o build/bench_gemm \
  bench_gemm.cu gemm_v0.cu gemm_v1.cu gemm_v3.cu gemm_v4.cu gemm_v5.cu \
  gemm_v6.cu gemm_v7.cu gemm_v8.cu "${CUTLASS_SOURCE}" "${EXTRA_SOURCES[@]}" "${EXTRA_LIBS[@]}"

run_case() {
  local M="$1"
  local N="$2"
  local K="$3"
  printf "\n==== M=%s N=%s K=%s ====\n" "$M" "$N" "$K"
  ./build/bench_gemm "$M" "$N" "$K"
}

# 可启用下列非整块尺寸用例，检查边界处理。
# run_case 127 129 131
# run_case 127 136 24  # v10：M/N tile 尾部与 K 尾部补零
# run_case 256 256 256
# run_case 511 257 263
# run_case 512 512 512
# run_case 1024 1024 512
run_case 1024 1024 1024
