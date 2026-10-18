#!/bin/bash
set -eo pipefail
cd "$(dirname "$0")"

CANN_ROOT="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.0}"
[[ -f "$CANN_ROOT/set_env.sh" ]] || { echo "CANN set_env.sh not found: $CANN_ROOT" >&2; exit 1; }
source "$CANN_ROOT/set_env.sh"
set -u
for tool in bisheng cmake g++ python3; do
    command -v "$tool" >/dev/null || { echo "Missing tool: $tool" >&2; exit 1; }
done
[[ -f ../sm100/bench_megaAttn.cu ]] || { echo 'Please copy the sm100 directory too (CPU reference).' >&2; exit 1; }

# 从 Runtime 查询实际芯片，B2/B2C 不再需要手改编译参数。
SOC_VERSION="$(python3 - "$CANN_ROOT/lib64/libascendcl.so" <<'PY_SOC'
import ctypes, sys
acl = ctypes.CDLL(sys.argv[1])
acl.aclrtGetSocName.restype = ctypes.c_char_p
soc = acl.aclrtGetSocName()
if not soc:
    raise SystemExit("Cannot query Ascend SoC")
print(soc.decode())
PY_SOC
)"
echo "Target SoC: $SOC_VERSION"
cmake -S . -B build/v2 -DASCEND_CANN_PACKAGE_PATH="$CANN_ROOT" -DSOC_VERSION="$SOC_VERSION"
cmake --build build/v2 -j "$(nproc)"
exec ./build/v2/bench_megaAttn
