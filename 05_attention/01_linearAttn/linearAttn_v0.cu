#include <cmath>
#include <cuda_runtime.h>

namespace LinearAttn {

__device__ __forceinline__ float ELU(float x, int N) {
    bool mask = x > 0.0f;
    float out = mask ? x + 1 : expf(x);
    return out;
}

} // namespace LinearAttn

__global__ void linear_attention_kernel(
    const float *__restrict__ Q, const float *__restrict__ K,
    const float *__restrict__ V, float *__restrict__ output, int M, int d
) {}

extern "C" void solve_v0(
    const float *Q, const float *K, const float *V, float *output, int M, int d
) {}
