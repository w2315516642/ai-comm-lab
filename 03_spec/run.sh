#!/bin/bash
# 编译所有版本算子 + bench,并跑一组尺寸对比
# 本地 GPU 是 RTX 4070 Laptop (sm_89);如果换机器记得改 -arch
set -euo pipefail

cd "$(dirname "$0")"

mkdir -p build

# 新版本文件加进来后,在这一行追加即可(或写进一个 wildcard)
nvcc -arch=sm_89 -O3 -o build/bench_spec_dec \
  bench_spec_dec.cu \
  spec_dec_v0.cu spec_dec_v1.cu spec_dec_v2.cu spec_dec_ans0.cu spec_dec_ans1.cu

# 尺寸扫描:B  T  V
./build/bench_spec_dec 1 8 32768
./build/bench_spec_dec 8 8 32768
./build/bench_spec_dec 8 16 32768
./build/bench_spec_dec 64 8 32768
