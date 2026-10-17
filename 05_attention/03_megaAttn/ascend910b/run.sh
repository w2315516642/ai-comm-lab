#!/bin/bash
set -eo pipefail
cd "$(dirname "$0")"

CANN_ROOT="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.0}"
[[ -f "$CANN_ROOT/set_env.sh" ]] || { echo "CANN set_env.sh not found: $CANN_ROOT" >&2; exit 1; }
source "$CANN_ROOT/set_env.sh"
set -u
for tool in bisheng cmake g++; do
    command -v "$tool" >/dev/null || { echo "Missing tool: $tool" >&2; exit 1; }
done
[[ -f ../sm100/bench_megaAttn.cu ]] || { echo 'Please copy the sm100 directory too (CPU reference).' >&2; exit 1; }

cmake -S . -B build/npu -DASCEND_CANN_PACKAGE_PATH="$CANN_ROOT" -DSOC_VERSION=Ascend910B2C
cmake --build build/npu -j "$(nproc)"
exec ./build/npu/bench_megaAttn
