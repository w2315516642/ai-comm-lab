#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>
#include <mma.h>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

namespace {

namespace wmma = nvcuda::wmma;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr int WARPS_M = 2;
constexpr int WARPS_N = 2;
constexpr int BM = WARPS_M * WMMA_M;
constexpr int BN = WARPS_N * WMMA_N;
constexpr int BK = WMMA_K;
constexpr int WARPS = WARPS_M * WARPS_N;
constexpr int THREADS = WARPS * 32;
constexpr int A_STAGE_SIZE = BM * BK;
constexpr int B_STAGE_SIZE = BK * BN;
constexpr int STAGE_SIZE = A_STAGE_SIZE + B_STAGE_SIZE;

using Accumulator =
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>;
using MatrixA = wmma::fragment<
    wmma::matrix_a,
    WMMA_M,
    WMMA_N,
    WMMA_K,
    __nv_bfloat16,
    wmma::row_major>;
using MatrixB = wmma::fragment<
    wmma::matrix_b,
    WMMA_M,
    WMMA_N,
    WMMA_K,
    __nv_bfloat16,
    wmma::row_major>;

// Each thread moves four consecutive BF16 values (8 bytes) from A and B.
// The aligned dimensions enforced by solve_v6 make every copy cp.async-safe.
__device__ __forceinline__ void async_load_tile(
    __nv_bfloat16 *As,
    __nv_bfloat16 *Bs,
    const __nv_bfloat16 *A,
    const __nv_bfloat16 *B,
    int N,
    int K,
    int tid
) {
    constexpr int VALUES_PER_COPY = 4;
    int a_index = tid * VALUES_PER_COPY;
    int a_row = a_index / BK;
    int a_col = a_index % BK;
    __pipeline_memcpy_async(
        &As[a_index],
        &A[a_row * K + a_col],
        VALUES_PER_COPY * sizeof(__nv_bfloat16)
    );

    int b_index = tid * VALUES_PER_COPY;
    int b_row = b_index / BN;
    int b_col = b_index % BN;
    __pipeline_memcpy_async(
        &Bs[b_index],
        &B[b_row * N + b_col],
        VALUES_PER_COPY * sizeof(__nv_bfloat16)
    );
}

__global__ void gemm_bf16_wmma_double_buffer(
    const __nv_bfloat16 *__restrict__ A,
    const __nv_bfloat16 *__restrict__ B,
    float *__restrict__ C,
    int M,
    int N,
    int K
) {
    extern __shared__ __nv_bfloat16 shared[];
    __nv_bfloat16 *As[2] = {shared, shared + STAGE_SIZE};
    __nv_bfloat16 *Bs[2] = {
        shared + A_STAGE_SIZE, shared + STAGE_SIZE + A_STAGE_SIZE
    };

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int warp_row = warp_id / WARPS_N;
    int warp_col = warp_id % WARPS_N;
    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;
    const __nv_bfloat16 *A_block = A + block_row * K;
    const __nv_bfloat16 *B_block = B + block_col;

    Accumulator accum;
    wmma::fill_fragment(accum, 0.0f);

    // Prologue: stage tile 0 before the first MMA operation.
    async_load_tile(As[0], Bs[0], A_block, B_block, N, K, tid);
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    int tile_count = K / BK;
    for (int tile = 0; tile < tile_count; tile++) {
        int read_stage = tile & 1;
        int write_stage = read_stage ^ 1;
        bool has_next = tile + 1 < tile_count;

        // While Tensor Cores consume the current stage, cp.async fills the
        // next.
        if (has_next) {
            const __nv_bfloat16 *next_A = A_block + (tile + 1) * BK;
            const __nv_bfloat16 *next_B = B_block + (tile + 1) * BK * N;
            async_load_tile(
                As[write_stage], Bs[write_stage], next_A, next_B, N, K, tid
            );
            __pipeline_commit();
        }

        MatrixA a_frag;
        MatrixB b_frag;
        const __nv_bfloat16 *warp_A = As[read_stage] + warp_row * WMMA_M * BK;
        const __nv_bfloat16 *warp_B = Bs[read_stage] + warp_col * WMMA_N;
        wmma::load_matrix_sync(a_frag, warp_A, BK);
        wmma::load_matrix_sync(b_frag, warp_B, BN);
        wmma::mma_sync(accum, a_frag, b_frag, accum);

        if (has_next) {
            __pipeline_wait_prior(0);
            __syncthreads();
        }
    }

    float *warp_C =
        C + (block_row + warp_row * WMMA_M) * N + block_col + warp_col * WMMA_N;
    wmma::store_matrix_sync(warp_C, accum, N, wmma::mem_row_major);
}

} // namespace

void solve_v6(
    const __nv_bfloat16 *A,
    const __nv_bfloat16 *B,
    float *C,
    int M,
    int N,
    int K
) {
    if (M % BM != 0 || N % BN != 0 || K % BK != 0)
        return;

    constexpr int SHARED_BYTES = 2 * STAGE_SIZE * sizeof(__nv_bfloat16);
    dim3 block(THREADS);
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    gemm_bf16_wmma_double_buffer<<<grid, block, SHARED_BYTES>>>(
        A, B, C, M, N, K
    );
}
