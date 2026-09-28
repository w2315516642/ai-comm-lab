#include <cuda_runtime.h>

#define FLOAT4(x) *((float4 *)(&(x)))
#define CEIL_DIV(a, b) ((a) + (b) - 1) / (b)

template <
    const int BM = 64,
    const int BN = 64,
    const int BK = 32,
    const int TM = 4,
    const int TN = 4>
__global__ void gemm_2d_btiling_vec(
    const float *A, const float *B, float *C, int M, int N, int K
) {

    extern __shared__ float s_mem[];
    float *As = s_mem;
    float *Bs = &s_mem[BM * BK];

    const uint cCol = blockIdx.x;
    const uint cRow = blockIdx.y;

    // __shared__ alignas(16) float As[BM * BK];
    // __shared__ alignas(16) float Bs[BK * BN];

    const uint strideA = BM / TM;
    const uint strideB = BN / TN;
    const uint threadCol = threadIdx.x % strideB;
    const uint threadRow = threadIdx.x / strideB;

    const uint vBK = BK / 4;
    const uint vBN = BN / 4;

    const uint innerColA = threadIdx.x % vBK;
    const uint innerRowA = threadIdx.x / vBK;
    const uint innerColB = threadIdx.x % vBN;
    const uint innerRowB = threadIdx.x / vBN;

    const uint NUM_THREADS = (BM * BN) / (TM * TN);
    const uint strideLoadA = NUM_THREADS / vBK;
    const uint strideLoadB = NUM_THREADS / vBN;

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    float tmp[TM * TN] = {0.0f};

    for (int k = 0; k < K; k += BK) {

        for (int loadOffset = 0; loadOffset < BM; loadOffset += strideLoadA) {
            uint sRow = innerRowA + loadOffset;
            uint sCol = innerColA * 4;
            FLOAT4(As[sRow * BK + sCol]) = FLOAT4(A[sRow * K + sCol]);
        }

        for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideLoadB) {
            uint sRow = innerRowB + loadOffset;
            uint sCol = innerColB * 4;
            FLOAT4(Bs[sRow * BN + sCol]) = FLOAT4(B[sRow * N + sCol]);
        }
        __syncthreads();

        A += BK;
        B += BK * N;

        for (uint dotIdx = 0; dotIdx < BK; dotIdx++) {
            for (uint resIdxB = 0; resIdxB < TN / 4; resIdxB++) {
                float4 tmpB = FLOAT4(
                    Bs[dotIdx * BN + threadCol * 4 + resIdxB * strideB * 4]
                );
                for (uint resIdxA = 0; resIdxA < TM; resIdxA++) {
                    float tmpA =
                        As[(threadRow + resIdxA * strideA) * BK + dotIdx];
                    tmp[resIdxA * TN + resIdxB * 4 + 0] += tmpA * tmpB.x;
                    tmp[resIdxA * TN + resIdxB * 4 + 1] += tmpA * tmpB.y;
                    tmp[resIdxA * TN + resIdxB * 4 + 2] += tmpA * tmpB.z;
                    tmp[resIdxA * TN + resIdxB * 4 + 3] += tmpA * tmpB.w;
                }
            }
        }
        __syncthreads();
    }
    for (uint resIdxA = 0; resIdxA < TM; resIdxA++) {
        uint cRowLocal = threadRow + resIdxA * strideA;
        for (uint resIdxB = 0; resIdxB < TN / 4; resIdxB++) {
            uint cColLocal = (threadCol + resIdxB * strideB) * 4;
            float4 res;
            uint cIdx = cRowLocal * N + cColLocal;
            uint tmpIdx = resIdxA * TN + resIdxB * 4;
            res.x = tmp[tmpIdx + 0];
            res.y = tmp[tmpIdx + 1];
            res.z = tmp[tmpIdx + 2];
            res.w = tmp[tmpIdx + 3];
            FLOAT4(C[cIdx]) = res;
        }
    }
}

void solve_v3(const float *A, const float *B, float *C, int M, int N, int K) {
    const int BLOCK_SIZE = 32;
    const int TILING_SIZE = BLOCK_SIZE;
    const int TM = 4;
    const int TN = 4;
    constexpr int BM = BLOCK_SIZE * TM;
    constexpr int BN = BLOCK_SIZE * TN * 2;
    constexpr int BK = TILING_SIZE;
    if (M % BM != 0 || N % BN != 0 || K % BK != 0)
        return;

    int shmem_needed_2dbtiling_vec = (BM * BK + BK * BN) * sizeof(float);
    cudaFuncSetAttribute(
        gemm_2d_btiling_vec<BM, BN, BK, TM, TN * 4>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shmem_needed_2dbtiling_vec
    );

    dim3 blockSize(BLOCK_SIZE * BLOCK_SIZE / 2);
    dim3 gridSize(CEIL_DIV(N, BN), CEIL_DIV(M, BM), 1);

    gemm_2d_btiling_vec<BM, BN, BK, TM, TN * 4>
        <<<gridSize, blockSize, shmem_needed_2dbtiling_vec>>>(A, B, C, M, N, K);
}
