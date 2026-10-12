#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

# Environment and dependency checks.
REPO_ROOT="$(cd ../../.. && pwd)"
FLASHMLA_DIR="$REPO_ROOT/third_party/flashmla"
CUTLASS_DIR="${CUTLASS_DIR:-$PWD/build/cutlass}"
if [[ -z "${CUDA_HOME:-}" ]]; then
    CUDA_HOME=/usr/local/cuda
    if [[ -x build/cuda-12.9/bin/nvcc ]]; then CUDA_HOME="$PWD/build/cuda-12.9"; fi
fi
NVCC="$CUDA_HOME/bin/nvcc"
fail() { echo "$*" >&2; exit 1; }
command -v g++ >/dev/null || fail 'g++ is required.'
command -v git >/dev/null || fail 'git is required.'
[[ -x "$NVCC" ]] || fail "nvcc not found: $NVCC (check CUDA_HOME)."
VERSION="$("$NVCC" --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')"
[[ -n "$VERSION" ]] || fail 'Cannot determine CUDA version.'
(( ${VERSION%%.*} > 12 || (${VERSION%%.*} == 12 && ${VERSION##*.} >= 9) )) \
    || fail "CUDA >=12.9 is required; found $VERSION."
[[ -f "$FLASHMLA_DIR/csrc/kerutils/include/kerutils/kerutils.cuh" ]] \
    || fail 'FlashMLA is missing; initialize third_party/flashmla with git submodule update --init.'
[[ -f "$CUTLASS_DIR/include/cute/tensor.hpp" ]] || fail "CUTLASS not found: $CUTLASS_DIR."
[[ "$(git -C "$FLASHMLA_DIR" rev-parse HEAD)" == 0eee43b12f034b657133cf2afca6a72ebb6efccf ]] \
    || fail 'FlashMLA does not match the fixed v0 version.'
[[ "$(git -C "$CUTLASS_DIR" rev-parse HEAD)" == 147f5673d0c1c3dcf66f78d677fd647e4a020219 ]] \
    || fail 'CUTLASS does not match the fixed v0 version.'

# Build v0 and add v1 automatically when present.
mkdir -p build
SOURCES=(megaAttn_v0.cu)
FLAGS=()
if [[ -f megaAttn_v1.cu ]]; then
    SOURCES+=(megaAttn_v1.cu)
    FLAGS+=(-DENABLE_MEGA_ATTN_V1)
fi
echo "Building SM100, CUDA $VERSION: ${SOURCES[*]}"
"$NVCC" -std=c++20 -O3 -DNDEBUG -lineinfo \
    -gencode=arch=compute_100a,code=sm_100a \
    --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math \
    --ptxas-options=-v,--register-usage-level=10,--warn-on-spills,--warn-on-local-memory-usage \
    -I"$FLASHMLA_DIR/csrc" -I"$FLASHMLA_DIR/csrc/kerutils/include" \
    -I"$CUTLASS_DIR/include" -I"$CUTLASS_DIR/tools/util/include" \
    -I"$CUDA_HOME/include/cccl" \
    -L/usr/lib/wsl/lib -L"$CUDA_HOME/lib" -L"$CUDA_HOME/lib64" \
    -L"$CUDA_HOME/lib/stubs" -L"$CUDA_HOME/lib64/stubs" \
    -lcuda "${FLAGS[@]}" bench_megaAttn.cu "${SOURCES[@]}" -o build/bench_megaAttn

exec ./build/bench_megaAttn
