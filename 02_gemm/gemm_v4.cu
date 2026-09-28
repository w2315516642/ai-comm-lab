#include <cuda_pipeline.h>
#include <cuda_runtime.h>

#define FLOAT4(x) *((float4 *)(&(x)))
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

namespace {

constexpr int BM = 128;
constexpr int BN = 256;
constexpr int BK = 32;
constexpr int TM = 4;
constexpr int TN = 16;
constexpr int THREADS = (BM / TM) * (BN / TN);

// Every thread copies several aligned float4 values. The caller commits all
// copies as one asynchronous group after both A and B have been issued.
__device__ __forceinline__ void async_load_tile(
    float *As, float *Bs, const float *A, const float *B, int N, int K, int tid
) {
    constexpr int A_COLS4 = BK / 4;
    constexpr int B_COLS4 = BN / 4;

    int a_row = tid / A_COLS4;
    int a_col4 = tid % A_COLS4;
    for (; a_row < BM; a_row += THREADS / A_COLS4) {
        int col = a_col4 * 4;
        __pipeline_memcpy_async(
            &As[a_row * BK + col], &A[a_row * K + col], sizeof(float4)
        );
    }

    int b_row = tid / B_COLS4;
    int b_col4 = tid % B_COLS4;
    for (; b_row < BK; b_row += THREADS / B_COLS4) {
        int col = b_col4 * 4;
        __pipeline_memcpy_async(
            &Bs[b_row * BN + col], &B[b_row * N + col], sizeof(float4)
        );
    }
}

__global__ void gemm_async_double_buffer(
    const float *__restrict__ A,
    const float *__restrict__ B,
    float *__restrict__ C,
    int M,
    int N,
    int K
) {
    extern __shared__ float shared[];
    constexpr int A_STAGE_SIZE = BM * BK;
    constexpr int B_STAGE_SIZE = BK * BN;
    constexpr int STAGE_SIZE = A_STAGE_SIZE + B_STAGE_SIZE;

    float *As[2] = {shared, shared + STAGE_SIZE};
    float *Bs[2] = {shared + A_STAGE_SIZE, shared + STAGE_SIZE + A_STAGE_SIZE};

    int tid = threadIdx.x;
    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;
    const float *A_block = A + block_row * K;
    const float *B_block = B + block_col;
    float *C_block = C + block_row * N + block_col;

    constexpr int THREAD_COLS = BN / TN;
    int thread_col = tid % THREAD_COLS;
    int thread_row = tid / THREAD_COLS;
    float accum[TM * TN] = {0.0f};

    // Prologue: make the first K tile available before computation starts.
    async_load_tile(As[0], Bs[0], A_block, B_block, N, K, tid);
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    int tile_count = K / BK;
    for (int tile = 0; tile < tile_count; tile++) {
        int read_stage = tile & 1;
        int write_stage = read_stage ^ 1;
        bool has_next = tile + 1 < tile_count;

        // Start loading the next tile before computing the current tile.
        if (has_next) {
            const float *next_A = A_block + (tile + 1) * BK;
            const float *next_B = B_block + (tile + 1) * BK * N;
            async_load_tile(
                As[write_stage], Bs[write_stage], next_A, next_B, N, K, tid
            );
            __pipeline_commit();
        }

        for (int kk = 0; kk < BK; kk++) {
            for (int col_group = 0; col_group < TN / 4; col_group++) {
                int shared_col = (thread_col + col_group * THREAD_COLS) * 4;
                float4 b = FLOAT4(Bs[read_stage][kk * BN + shared_col]);

                for (int row_item = 0; row_item < TM; row_item++) {
                    int shared_row = thread_row + row_item * (BM / TM);
                    float a = As[read_stage][shared_row * BK + kk];
                    int out = row_item * TN + col_group * 4;
                    accum[out + 0] += a * b.x;
                    accum[out + 1] += a * b.y;
                    accum[out + 2] += a * b.z;
                    accum[out + 3] += a * b.w;
                }
            }
        }

        // The next stage may be consumed only after every thread has finished
        // its asynchronous copies. The barrier also protects the old stage
        // before it is reused two iterations later.
        if (has_next) {
            __pipeline_wait_prior(0);
            __syncthreads();
        }
    }

    for (int row_item = 0; row_item < TM; row_item++) {
        int row = thread_row + row_item * (BM / TM);
        for (int col_group = 0; col_group < TN / 4; col_group++) {
            int col = (thread_col + col_group * THREAD_COLS) * 4;
            int out = row_item * TN + col_group * 4;
            FLOAT4(C_block[row * N + col]) = make_float4(
                accum[out + 0], accum[out + 1], accum[out + 2], accum[out + 3]
            );
        }
    }
}

} // namespace

void solve_v4(const float *A, const float *B, float *C, int M, int N, int K) {
    if (M % BM != 0 || N % BN != 0 || K % BK != 0)
        return;

    constexpr int SHARED_BYTES = 2 * (BM * BK + BK * BN) * sizeof(float);
    cudaFuncSetAttribute(
        gemm_async_double_buffer,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        SHARED_BYTES
    );

    dim3 block(THREADS);
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    gemm_async_double_buffer<<<grid, block, SHARED_BYTES>>>(A, B, C, M, N, K);
}
