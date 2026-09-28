#include <cuda_bf16.h>
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

__global__ void gemm_bf16_wmma(
    const __nv_bfloat16 *__restrict__ A,
    const __nv_bfloat16 *__restrict__ B,
    float *__restrict__ C,
    int M,
    int N,
    int K
) {
    __shared__ __nv_bfloat16 As[BM * BK];
    __shared__ __nv_bfloat16 Bs[BK * BN];

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int warp_row = warp_id / WARPS_N;
    int warp_col = warp_id % WARPS_N;
    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> accum;
    wmma::fill_fragment(accum, 0.0f);

    for (int k0 = 0; k0 < K; k0 += BK) {
        // Cooperatively stage the BF16 input tiles in shared memory.
        for (int idx = tid; idx < BM * BK; idx += THREADS) {
            int row = idx / BK;
            int col = idx % BK;
            As[idx] = A[(block_row + row) * K + k0 + col];
        }
        for (int idx = tid; idx < BK * BN; idx += THREADS) {
            int row = idx / BN;
            int col = idx % BN;
            Bs[idx] = B[(k0 + row) * N + block_col + col];
        }
        __syncthreads();

        wmma::fragment<
            wmma::matrix_a,
            WMMA_M,
            WMMA_N,
            WMMA_K,
            __nv_bfloat16,
            wmma::row_major>
            a_frag;
        wmma::fragment<
            wmma::matrix_b,
            WMMA_M,
            WMMA_N,
            WMMA_K,
            __nv_bfloat16,
            wmma::row_major>
            b_frag;

        const __nv_bfloat16 *warp_A = As + warp_row * WMMA_M * BK;
        const __nv_bfloat16 *warp_B = Bs + warp_col * WMMA_N;
        wmma::load_matrix_sync(a_frag, warp_A, BK);
        wmma::load_matrix_sync(b_frag, warp_B, BN);
        wmma::mma_sync(accum, a_frag, b_frag, accum);
        __syncthreads();
    }

    float *warp_C =
        C + (block_row + warp_row * WMMA_M) * N + block_col + warp_col * WMMA_N;
    wmma::store_matrix_sync(warp_C, accum, N, wmma::mem_row_major);
}

} // namespace

void solve_v5(
    const __nv_bfloat16 *A,
    const __nv_bfloat16 *B,
    float *C,
    int M,
    int N,
    int K
) {
    if (M % BM != 0 || N % BN != 0 || K % BK != 0)
        return;

    dim3 block(THREADS);
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    gemm_bf16_wmma<<<grid, block>>>(A, B, C, M, N, K);
}
