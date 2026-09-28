#include <cuda_runtime.h>

template <int BM = 16, int BN = 16, int BK = 16>
__global__ void gemm_kernel_smem(
    const float *__restrict__ A,
    const float *__restrict__ B,
    float *C,
    int M,
    int N,
    int K
) {
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    int m = blockDim.y * blockIdx.y + threadIdx.y;

    __shared__ float smem_A[BM * BK];
    __shared__ float smem_B[BK * BN];

    int sm_idx = threadIdx.y;
    int sn_idx = threadIdx.x;

    int smk = threadIdx.y % BK;
    int snk = threadIdx.x % BK;

    float tmp = 0.0f;
    for (int bk = 0; bk < K; bk += BK) {
        // 1. 把数据搬运到 smem 中，注意 coalesced
        int lo = bk;
        int hi = min(bk + BK, K);

        for (int i = 0; i < BK; i += BN) {
            bool mask = (i + snk < BK) && (bk + i + snk < K) && m < M;
            smem_A[sm_idx * BK + i + snk] =
                mask ? A[m * K + bk + i + snk] : 0.0f;
        }
        for (int i = 0; i < BK; i += BM) {
            bool mask = (i + smk < BK) && (bk + i + smk < K) && n < N;
            smem_B[(i + smk) * BN + sn_idx] =
                mask ? B[(bk + i + smk) * N + n] : 0.0f;
        }

        __syncthreads();

        // 2. 对数据进行乘加
        for (int k = 0; k < hi - lo; k++) {
            tmp += smem_A[sm_idx * BK + k] * smem_B[k * BN + sn_idx];
        }
        // 3. 等所有数据被用完再开始下一次搬运
        __syncthreads();
    }
    if (m < M && n < N)
        C[m * N + n] = tmp;
}

void solve_v1(const float *A, const float *B, float *C, int M, int N, int K) {
    dim3 threads(16, 16, 1);
    dim3 blocks((N + 16 - 1) / 16, (M + 16 - 1) / 16, 1);

    gemm_kernel_smem<16, 16, 16><<<blocks, threads>>>(A, B, C, M, N, K);
}
