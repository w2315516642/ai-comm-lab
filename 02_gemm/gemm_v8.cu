#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

namespace {
constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 16;

constexpr int WARPS_M = 2;
constexpr int WARPS_N = 2;
constexpr int WARP_TILES_M = 2;
constexpr int WARP_TILES_N = 8;
constexpr int BK = 32;

constexpr int WARP_M = WARP_TILES_M * MMA_M;
constexpr int WARP_N = WARP_TILES_N * MMA_N;
constexpr int BM = WARPS_M * WARP_M;
constexpr int BN = WARPS_N * WARP_N;
constexpr int WARPS = WARPS_M * WARPS_N;
constexpr int THREADS = WARPS * 32;
constexpr int COPY_BYTES = 16;
constexpr int VALUES_PER_COPY = COPY_BYTES / sizeof(__nv_bfloat16);

constexpr int A_STAGE_SIZE = BM * BK;
constexpr int B_STAGE_SIZE = BN * BK;
constexpr int STAGE_SIZE = A_STAGE_SIZE + B_STAGE_SIZE;

static_assert(BK % MMA_K == 0);
static_assert(BK % VALUES_PER_COPY == 0);
static_assert(BN % VALUES_PER_COPY == 0);
static_assert(THREADS <= 1024);

__device__ __forceinline__ uint32_t shared_address(const void *ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ int swizzle_index_a(int row, int col) {
    constexpr int CHUNKS_PER_ROW = BK / VALUES_PER_COPY;

    int chunk = col / VALUES_PER_COPY;
    int offset = col % VALUES_PER_COPY;
    int swizzled_chunk = chunk ^ (row & (CHUNKS_PER_ROW - 1));

    return row * BK + swizzled_chunk * VALUES_PER_COPY + offset;
}

__device__ __forceinline__ int swizzle_index_b(int row, int col) {
    constexpr int CHUNKS_PER_ROW = BN / VALUES_PER_COPY;

    int chunk = col / VALUES_PER_COPY;
    int offset = col % VALUES_PER_COPY;
    int swizzled_chunk = chunk ^ (row & (CHUNKS_PER_ROW - 1));

    return row * BN + swizzled_chunk * VALUES_PER_COPY + offset;
}

__device__ __forceinline__ void async_load_tile(
    __nv_bfloat16 *As,
    __nv_bfloat16 *Bs,
    const __nv_bfloat16 *A,
    const __nv_bfloat16 *B,
    int N,
    int K,
    int tid
) {
    constexpr int STRIDE = THREADS * VALUES_PER_COPY;

    for (int index = tid * VALUES_PER_COPY; index < A_STAGE_SIZE;
         index += STRIDE) {
        int row = index / BK;
        int col = index % BK;
        int swizzled_index = swizzle_index_a(row, col);

        __pipeline_memcpy_async(
            &As[swizzled_index], &A[row * K + col], COPY_BYTES
        );
    }

    for (int index = tid * VALUES_PER_COPY; index < B_STAGE_SIZE;
         index += STRIDE) {
        int row = index / BN;
        int col = index % BN;
        int swizzled_index = swizzle_index_b(row, col);

        __pipeline_memcpy_async(
            &Bs[swizzled_index], &B[row * N + col], COPY_BYTES
        );
    }
}

__device__ __forceinline__ void load_matrix_a(
    uint32_t (&a)[4],
    const __nv_bfloat16 *As,
    int logical_row,
    int logical_col,
    int lane_id
) {
    int row = logical_row + lane_id % 16;
    int col = logical_col + (lane_id / 16) * 8;

    int index = swizzle_index_a(row, col);
    uint32_t address = shared_address(&As[index]);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                 "{%0, %1, %2, %3}, [%4];\n"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                 : "r"(address));
}

__device__ __forceinline__ void load_matrix_b(
    uint32_t (&b)[2],
    const __nv_bfloat16 *Bs,
    int logical_row,
    int logical_col,
    int lane_id
) {
    int row = logical_row + lane_id % 16;
    int col = logical_col;

    int index = swizzle_index_b(row, col);
    uint32_t address = shared_address(&Bs[index]);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(b[0]), "=r"(b[1])
        : "r"(address)
    );
}

__device__ __forceinline__ void mma_m16n8k16(
    float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1])
    );
}

__device__ __forceinline__ void store_accumulator(
    float *C,
    const float (&accum)[WARP_TILES_M][WARP_TILES_N][4],
    int N,
    int warp_row,
    int warp_col,
    int lane_id
) {
    int row_in_half = lane_id / 4;
    int col_pair = (lane_id % 4) * 2;
    int base_row = warp_row * WARP_M;
    int base_col = warp_col * WARP_N;

#pragma unroll
    for (int m_tile = 0; m_tile < WARP_TILES_M; m_tile++) {
#pragma unroll
        for (int n_tile = 0; n_tile < WARP_TILES_N; n_tile++) {
            int row = base_row + m_tile * MMA_M + row_in_half;
            int col = base_col + n_tile * MMA_N + col_pair;
            C[row * N + col + 0] = accum[m_tile][n_tile][0];
            C[row * N + col + 1] = accum[m_tile][n_tile][1];
            C[(row + 8) * N + col + 0] = accum[m_tile][n_tile][2];
            C[(row + 8) * N + col + 1] = accum[m_tile][n_tile][3];
        }
    }
}

__global__ void gemm_bf16_double_buffer(
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
        shared + A_STAGE_SIZE, shared + A_STAGE_SIZE + STAGE_SIZE
    };

    int tid = threadIdx.x;
    int lane_id = tid & 31;
    int warp_id = tid >> 5;

    // 四个 warp 在 block 输出 tile 的位置
    //
    // warp 0 | warp 1
    // ---------------
    // warp 2 | warp 3
    //
    // 每个格子是 16x16，所以整个 block 覆盖 32x32
    int warp_row = warp_id / WARPS_N;
    int warp_col = warp_id % WARPS_N;
    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;
    const __nv_bfloat16 *A_block = A + block_row * K;
    const __nv_bfloat16 *B_block = B + block_col;

    float accum[WARP_TILES_M][WARP_TILES_N][4] = {};

    async_load_tile(As[0], Bs[0], A_block, B_block, N, K, tid);
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    int tile_count = K / BK;
    for (int tile = 0; tile < tile_count; tile++) {
        // stage 0 和 stage 1 奇偶交替读算
        int read_stage = tile & 1;
        int write_stage = read_stage ^ 1;
        bool has_next = tile + 1 < tile_count;

        if (has_next) {
            async_load_tile(
                As[write_stage],
                Bs[write_stage],
                A_block + (tile + 1) * BK,
                B_block + (tile + 1) * BK * N,
                N,
                K,
                tid
            );
            __pipeline_commit();
        }

#pragma unroll
        for (int kk = 0; kk < BK; kk += MMA_K) {
            uint32_t a_frag[WARP_TILES_M][4];
            uint32_t b_frag[WARP_TILES_N][2];

#pragma unroll
            for (int m_tile = 0; m_tile < WARP_TILES_M; m_tile++) {
                int a_row = warp_row * WARP_M + m_tile * MMA_M;
                int a_col = kk;
                load_matrix_a(
                    a_frag[m_tile], As[read_stage], a_row, a_col, lane_id
                );
            }

#pragma unroll
            for (int n_tile = 0; n_tile < WARP_TILES_N; n_tile++) {
                int b_row = kk;
                int b_col = warp_col * WARP_N + n_tile * MMA_N;
                load_matrix_b(
                    b_frag[n_tile], Bs[read_stage], b_row, b_col, lane_id
                );
            }

#pragma unroll
            for (int m_tile = 0; m_tile < WARP_TILES_M; m_tile++) {
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_TILES_N; n_tile++) {
                    mma_m16n8k16(
                        accum[m_tile][n_tile], a_frag[m_tile], b_frag[n_tile]
                    );
                }
            }
        }

        if (has_next) {
            __pipeline_wait_prior(0);
            __syncthreads();
        }
    }

    float *C_block = C + block_row * N + block_col;
    store_accumulator(C_block, accum, N, warp_row, warp_col, lane_id);
}

} // namespace

bool supports_v8(int M, int N, int K) {
    return M % BM == 0 && N % BN == 0 && K % BK == 0;
}

void solve_v8(
    const __nv_bfloat16 *A,
    const __nv_bfloat16 *B,
    float *C,
    int M,
    int N,
    int K
) {
    if (!supports_v8(M, N, K)) {
        return;
    }

    constexpr int SHARED_BYTES = 2 * STAGE_SIZE * sizeof(__nv_bfloat16);

    dim3 block(THREADS);
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));

    gemm_bf16_double_buffer<<<grid, block, SHARED_BYTES>>>(A, B, C, M, N, K);
}
