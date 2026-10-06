#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int WGMMA_M = 64;
constexpr int WGMMA_N = 128;
constexpr int WGMMA_K = 16;

constexpr int WARPS_PER_WG = 4;
constexpr int WG_M = 2;
constexpr int WG_N = 2;
constexpr int WG_PER_BLOCK = WG_M * WG_N;

constexpr int THREADS = WARPS_PER_WG * WG_PER_BLOCK * 32;
constexpr int BM = WG_M * WGMMA_M;
constexpr int BN = WG_N * WGMMA_N;
constexpr int BK = WGMMA_K;

static_assert(WGMMA_M == 64 && WGMMA_N == 128 && WGMMA_K == 16,
              "the PTX wrapper below implements m64n128k16");
static_assert(WG_M > 0 && WG_N > 0 && THREADS <= 1024);
static_assert((BM + BN) * BK * sizeof(__nv_bfloat16) <= 48 * 1024);

constexpr int ACCUM_SIZE = WGMMA_M * WGMMA_N / 128;

// 无 swizzle 的 K-major 布局：以连续的 8×8 BF16 小块为单位存储。
// A 的逻辑坐标为 (m,k)，B 为 (n,k)。每 8 行先存 k=0..7 的小块，
// 再存 k=8..15 的小块，因此这里不是普通的行主序布局。
__host__ __device__ constexpr int smem_index(int row, int k) {
    return (row / 8) * (8 * BK) + (k / 8) * 64 + (row % 8) * 8 + k % 8;
}

__device__ __forceinline__ uint64_t make_smem_descriptor(const void *ptr) {
    const uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    constexpr uint64_t LEADING_BYTES = 8 * 8 * sizeof(__nv_bfloat16);
    constexpr uint64_t STRIDE_BYTES = 8 * BK * sizeof(__nv_bfloat16);

    // WGMMA 描述符：起始地址占 [13:0]，leading 偏移占 [29:16]，stride 偏移占 [45:32]。
    // 三个字段均以 16 字节为单位编码；基址偏移和 swizzle 模式均为 0。
    return ((address >> 4) & 0x3fffULL) |
           ((LEADING_BYTES >> 4) << 16) | ((STRIDE_BYTES >> 4) << 32);
}

__device__ __forceinline__ void load_tile(
    __nv_bfloat16 *As,
    __nv_bfloat16 *Bs,
    const __nv_bfloat16 *A,
    const __nv_bfloat16 *B,
    int M, int N, int K,
    int block_row, int block_col, int k_start
) {
    for (int i = threadIdx.x; i < BM * BK; i += blockDim.x) {
        int row = i / BK;
        int k = i % BK;
        As[smem_index(row, k)] = block_row + row < M && k_start + k < K
            ? A[(size_t)(block_row + row) * K + k_start + k]
            : __float2bfloat16_rn(0.0f);
    }
    // 合并读取全局内存中行主序的 B[K,N]，再按 K-major 的 B[n,k] 布局写入共享内存。
    for (int i = threadIdx.x; i < BK * BN; i += blockDim.x) {
        int k = i / BN;
        int col = i % BN;
        Bs[smem_index(col, k)] = k_start + k < K && block_col + col < N
            ? B[(size_t)(k_start + k) * N + block_col + col]
            : __float2bfloat16_rn(0.0f);
    }
}

// 普通共享内存写入使用 generic proxy，WGMMA 读取使用 async proxy。
// 每个写入线程先用 fence 保证跨 proxy 的可见性，再进行整个 block 的同步。
__device__ __forceinline__ void publish_shared_tile() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
}

// 同一个 warpgroup 的 128 个线程必须共同执行此函数。
// 基线版提交后立即等待完成；将 fence、MMA、commit、wait 放在同一个 asm 块中，
// 让编译器通过输入输出约束识别整个指令序列对累加寄存器的依赖。
__device__ __forceinline__ void wgmma_m64n128k16(
    float (&d)[ACCUM_SIZE], uint64_t desc_a, uint64_t desc_b
) {
    asm volatile(
        "{\n"
        ".reg .pred accumulate;\n"
        "setp.ne.u32 accumulate, 1, 0;\n"
        "wgmma.fence.sync.aligned;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, "
        " %8, %9, %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31, "
        " %32, %33, %34, %35, %36, %37, %38, %39, "
        " %40, %41, %42, %43, %44, %45, %46, %47, "
        " %48, %49, %50, %51, %52, %53, %54, %55, "
        " %56, %57, %58, %59, %60, %61, %62, %63}, "
        "%64, %65, accumulate, 1, 1, 0, 0;\n"
        "wgmma.commit_group.sync.aligned;\n"
        "wgmma.wait_group.sync.aligned 0;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
          "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
          "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]),
          "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
          "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
          "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
          "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]),
          "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),
          "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]),
          "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),
          "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]),
          "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
          "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]),
          "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
        : "l"(desc_a), "l"(desc_b)
        : "memory"
    );
}

// 每个 warp 负责 16 行，每个 lane 持有第 r 行和第 r+8 行上的相邻两列。
// 每连续 4 个累加寄存器对应一组这样的元素，下一组在 warpgroup 的输出 tile 中右移 8 列。
__host__ __device__ constexpr int accumulator_row(int wg_thread, int reg) {
    return (wg_thread / 32) * 16 + (wg_thread % 32) / 4 + ((reg % 4) / 2) * 8;
}

__host__ __device__ constexpr int accumulator_col(int wg_thread, int reg) {
    return (reg / 4) * 8 + (wg_thread % 4) * 2 + reg % 2;
}

__device__ __forceinline__ void store_accumulator(
    float *C, const float (&accum)[ACCUM_SIZE],
    int M, int N, int row_start, int col_start, int wg_thread
) {
#pragma unroll
    for (int r = 0; r < ACCUM_SIZE; r++) {
        int row = row_start + accumulator_row(wg_thread, r);
        int col = col_start + accumulator_col(wg_thread, r);
        if (row < M && col < N)
            C[(size_t)row * N + col] = accum[r];
    }
}

__global__ void gemm_bf16_hopper(
    const __nv_bfloat16 *__restrict__ A,
    const __nv_bfloat16 *__restrict__ B,
    float *__restrict__ C,
    int M,
    int N,
    int K
) {
    __shared__ __align__(16) __nv_bfloat16 As[BM * BK];
    __shared__ __align__(16) __nv_bfloat16 Bs[BN * BK];

    int wg_id = threadIdx.x / 128;
    int wg_thread = threadIdx.x % 128;
    int wg_row = (wg_id / WG_N) * WGMMA_M;
    int wg_col = (wg_id % WG_N) * WGMMA_N;
    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    uint64_t desc_a = make_smem_descriptor(As + smem_index(wg_row, 0));
    uint64_t desc_b = make_smem_descriptor(Bs + smem_index(wg_col, 0));
    float accum[ACCUM_SIZE] = {};

    for (int k_start = 0; k_start < K; k_start += BK) {
        load_tile(As, Bs, A, B, M, N, K, block_row, block_col, k_start);
        publish_shared_tile();
        wgmma_m64n128k16(accum, desc_a, desc_b);
        // 当前 warpgroup 的 WGMMA 已完成；所有 warpgroup 都结束后才能覆盖共享内存。
        __syncthreads();
    }

    store_accumulator(C, accum, M, N,
                      block_row + wg_row, block_col + wg_col, wg_thread);
}

} // 匿名命名空间

bool supports_v9(int M, int N, int K) {
    return M > 0 && N > 0 && K >= 0;
}

void solve_v9(
    const __nv_bfloat16 *A,
    const __nv_bfloat16 *B,
    float *C,
    int M,
    int N,
    int K
) {
    if (!supports_v9(M, N, K))
        return;
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    gemm_bf16_hopper<<<grid, THREADS>>>(A, B, C, M, N, K);
}
