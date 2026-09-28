#include <cuda_runtime.h>

__global__ void gemm_kernel_v0(
    const float *__restrict__ A,
    const float *__restrict__ B,
    float *C,
    int M,
    int N,
    int K
) {
    int m = blockDim.x * blockIdx.x + threadIdx.x;
    int n = blockDim.y * blockIdx.y + threadIdx.y;

    if (m < M && n < N) {
        float dot = 0.0f;
        for (int k = 0; k < K; k++) {
            dot += A[m * K + k] * B[k * N + n];
        }
        C[m * N + n] = dot;
    }
}

void solve(const float *A, const float *B, float *C, int M, int N, int K) {
    dim3 threads(32, 32, 1);
    dim3 blocks((M + 31) / 32, (N + 31 / 32), 1);

    gemm_kernel_v0<<<blocks, threads>>>(A, B, C, M, N, K);
}
