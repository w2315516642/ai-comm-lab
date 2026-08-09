#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")"

mkdir -p build

nvcc -arch=sm_80 -O3 -o build/bandwidth_pcie bandwidth_pcie.cu
./build/bandwidth_pcie