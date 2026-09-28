#include <cuda_runtime.h>

template <int BM = 64, int BN = 64, int BK = 32, int TM = 2, int TN = 2>
__global__ void gemm_kernel_smem(
    const float *__restrict__ A,
    const float *__restrict__ B,
    float *C,
    int M,
    int N,
    int K
) {}

void solve_v1(const float *A, const float *B, float *C, int M, int N, int K) {
    const int BLOCK_SIZE = 16;
    // 每个线程处理 C 中的 TM * TN 小块
    const int TM = 2;
    const int TN = 2;
    const int BK = 32;

    const int BM = BLOCK_SIZE * TM;
    const int BN = BLOCK_SIZE * TN;

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 blocks((M + BM - 1) / BM, (N + BN - 1) / BN, 1);

    gemm_kernel_smem<<<blocks, threads>>>(A, B, C, M, N, K);
}
