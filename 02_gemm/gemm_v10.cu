#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <cstdio>
#include <cstdlib>

namespace {

// 保留 v9 的 warpgroup 划分，先实现单缓冲 TMA + WGMMA。
// 输入仍为行主序 A[M,K]、B[K,N]，输出为 FP32 C[M,N]。
constexpr int WGMMA_M = 64;
constexpr int WGMMA_N = 128;
constexpr int WGMMA_K = 16;
constexpr int WG_M = 2;
constexpr int WG_N = 2;
constexpr int THREADS = WG_M * WG_N * 128;
constexpr int BM = WG_M * WGMMA_M;
constexpr int BN = WG_N * WGMMA_N;
constexpr int BK = 16;

// B 每次搬 64 列，即每行 128 字节；宽 tile 分成多个这样的条带。
// A 的连续维度是 K，B 的连续维度是 N，不需要额外转置 B。
constexpr int B_TMA_N = 64;
constexpr int ACCUM_SIZE = WGMMA_M * WGMMA_N / 128;
constexpr int TILE_BYTES = (BM + BN) * BK * sizeof(__nv_bfloat16);
constexpr int A_SWIZZLE_MODE = BK == 16 ? 3 : (BK == 32 ? 2 : 1);
constexpr CUtensorMapSwizzle A_TMA_SWIZZLE = BK == 16
    ? CU_TENSOR_MAP_SWIZZLE_32B
    : (BK == 32 ? CU_TENSOR_MAP_SWIZZLE_64B : CU_TENSOR_MAP_SWIZZLE_128B);

static_assert(WGMMA_M == 64 && WGMMA_N == 128 && WGMMA_K == 16);
static_assert(WG_M > 0 && WG_N > 0 && THREADS <= 1024);
static_assert(BK == 16 || BK == 32 || BK == 64);
static_assert(B_TMA_N == 64);
static_assert(BM <= 256 && BN % B_TMA_N == 0);
static_assert(TILE_BYTES + sizeof(uint64_t) <= 48 * 1024);

// TMA 与 WGMMA 使用同一种硬件 swizzle。下标单位是 BF16 元素。
// 这里仅计算每次 WGMMA 子 tile 的起点，不再由线程逐元素加载。
__host__ __device__ constexpr int swizzle_index(int index, int width_bf16) {
    return index ^ (((index >> 6) & (width_bf16 / 8 - 1)) << 3);
}

__host__ __device__ constexpr int smem_a_index(int m, int k) {
    return swizzle_index(m * BK + k, BK);
}

__host__ __device__ constexpr int smem_b_index(int n, int k) {
    int index = (n / B_TMA_N) * (BK * B_TMA_N) + k * B_TMA_N + n % B_TMA_N;
    return swizzle_index(index, B_TMA_N);
}

__device__ __forceinline__ uint32_t shared_address(const void *ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ uint64_t make_smem_descriptor(
    const void *ptr, int leading_bytes, int stride_bytes, int swizzle_mode
) {
    // 起始地址、leading、stride 都按 16 字节编码。
    // 缓冲区及各 swizzle 周期起点按 1024 字节对齐，因此 base offset 为 0。
    return ((shared_address(ptr) >> 4) & 0x3fffULL)
         | (uint64_t(leading_bytes >> 4) << 16)
         | (uint64_t(stride_bytes >> 4) << 32)
         | (uint64_t(swizzle_mode) << 62);
}

__device__ __forceinline__ void init_tma_barrier(uint64_t *barrier) {
    // 只有发起搬运的线程执行 arrive；其他线程仅等待，不计入到达人数。
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                 :: "r"(shared_address(barrier)) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

__device__ __forceinline__ void expect_tma_bytes(uint64_t *barrier) {
    // 所有 A/B 搬运共用一个 barrier，预登记本轮完整 tile 的字节数。
    // 越界补零部分同样计入完成字节数。
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                 :: "r"(shared_address(barrier)), "r"(TILE_BYTES) : "memory");
}

__device__ __forceinline__ void tma_load_2d(
    void *dst, const CUtensorMap *map, int x, int y, uint64_t *barrier
) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3}], [%4];"
        :: "r"(shared_address(dst)), "l"(map), "r"(x), "r"(y),
           "r"(shared_address(barrier)) : "memory");
}

__device__ __forceinline__ void load_tiles_tma(
    __nv_bfloat16 *As, __nv_bfloat16 *Bs,
    const CUtensorMap *map_a, const CUtensorMap *map_b,
    int block_row, int block_col, int k_start, uint64_t *barrier
) {
    expect_tma_bytes(barrier);
    tma_load_2d(As, map_a, k_start, block_row, barrier);
#pragma unroll
    for (int n = 0; n < BN; n += B_TMA_N)
        tma_load_2d(Bs + (n / B_TMA_N) * BK * B_TMA_N,
                    map_b, block_col + n, k_start, barrier);
}

__device__ __forceinline__ void wait_tma(uint64_t *barrier, uint32_t phase) {
    asm volatile(
        "{\n"
        ".reg .pred done;\n"
        "wait_tma_loop:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 done, [%0], %1;\n"
        "@!done bra wait_tma_loop;\n"
        "}\n"
        :: "r"(shared_address(barrier)), "r"(phase) : "memory");
}

// A 采用 K-major，B 采用 N-major，因此 WGMMA 最后的 trans-b 参数为 1。
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
        "%64, %65, accumulate, 1, 1, 0, 1;\n"
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

__global__ void gemm_bf16_hopper_tma(
    const __grid_constant__ CUtensorMap map_a,
    const __grid_constant__ CUtensorMap map_b,
    float *__restrict__ C, int M, int N, int K
) {
    // 1024 对齐覆盖 32/64/128B swizzle 的完整重复周期。
    __shared__ __align__(1024) __nv_bfloat16 As[BM * BK];
    __shared__ __align__(1024) __nv_bfloat16 Bs[BN * BK];
    __shared__ __align__(8) uint64_t barrier;
    if (threadIdx.x == 0)
        init_tma_barrier(&barrier);
    __syncthreads();

    int wg_id = threadIdx.x / 128;
    int wg_thread = threadIdx.x % 128;
    int wg_row = (wg_id / WG_N) * WGMMA_M;
    int wg_col = (wg_id % WG_N) * WGMMA_N;
    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;
    float accum[ACCUM_SIZE] = {};
    uint32_t phase = 0;

    for (int k_start = 0; k_start < K; k_start += BK) {
        if (threadIdx.x == 0)
            load_tiles_tma(As, Bs, &map_a, &map_b,
                           block_row, block_col, k_start, &barrier);
        // TMA 与 WGMMA 都访问 async proxy。每个消费者等到搬运完成后才能读。
        wait_tma(&barrier, phase);
        phase ^= 1;
#pragma unroll
        for (int k = 0; k < BK; k += WGMMA_K) {
            // A：K-major swizzle，leading 字段按隐含值 1 编码。
            uint64_t desc_a = make_smem_descriptor(
                As + smem_a_index(wg_row, k), 16, 8 * BK * 2, A_SWIZZLE_MODE);
            // B：N-major 128B swizzle，横跨 64 列跳到下一条带，
            // 沿 K 跨 8 行则跳过 8 * 128 字节。
            uint64_t desc_b = make_smem_descriptor(
                Bs + smem_b_index(wg_col, k), BK * B_TMA_N * 2,
                8 * B_TMA_N * 2, 1);
            wgmma_m64n128k16(accum, desc_a, desc_b);
        }
        // 单缓冲：所有 warpgroup 完成 WGMMA 后才能发起下一轮覆盖。
        __syncthreads();
    }
    if (threadIdx.x == 0)
        asm volatile("mbarrier.inval.shared::cta.b64 [%0];"
                     :: "r"(shared_address(&barrier)) : "memory");
    store_accumulator(C, accum, M, N,
                      block_row + wg_row, block_col + wg_col, wg_thread);
}

void check_driver(CUresult result, const char *operation) {
    if (result != CUDA_SUCCESS) {
        const char *message = nullptr;
        cuGetErrorString(result, &message);
        std::fprintf(stderr, "v10 %s: %s (code=%d)\n", operation,
                     message ? message : "unknown CUDA error", int(result));
        std::abort();
    }
}

CUtensorMap make_tensor_map(
    const __nv_bfloat16 *ptr, int rows, int cols,
    int tile_rows, int tile_cols, CUtensorMapSwizzle swizzle
) {
    // TMA 维度从最快变化的维度开始，因此行主序矩阵描述为 {cols, rows}。
    CUtensorMap map{};
    uint64_t dims[2] = {uint64_t(cols), uint64_t(rows)};
    uint64_t strides[1] = {uint64_t(cols) * sizeof(__nv_bfloat16)};
    uint32_t box[2] = {uint32_t(tile_cols), uint32_t(tile_rows)};
    uint32_t element_strides[2] = {1, 1};
    check_driver(cuTensorMapEncodeTiled(
        &map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
        const_cast<__nv_bfloat16 *>(ptr), dims, strides, box, element_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE), "cuTensorMapEncodeTiled");
    return map;
}

// 描述符只描述地址和形状，不保存矩阵数据。相同输入地址/形状重复计时时复用，
// 避免把 CPU 构造描述符的时间混入每次 kernel 提交间隔；不缓存任何 GPU 分配。
struct TensorMapCache {
    const __nv_bfloat16 *a = nullptr;
    const __nv_bfloat16 *b = nullptr;
    int m = 0, n = 0, k = 0;
    CUtensorMap map_a{}, map_b{};

    void update(const __nv_bfloat16 *A, const __nv_bfloat16 *B, int M, int N, int K) {
        if (a == A && b == B && m == M && n == N && k == K)
            return;
        map_a = make_tensor_map(A, M, K, BM, BK, A_TMA_SWIZZLE);
        map_b = make_tensor_map(B, K, N, BK, B_TMA_N, CU_TENSOR_MAP_SWIZZLE_128B);
        a = A;
        b = B;
        m = M;
        n = N;
        k = K;
    }
};

} // 匿名命名空间

bool supports_v10(int M, int N, int K) {
    // 行跨度需为 16 字节的倍数；M/N/K 不要求整除 BM/BN/BK。
    // TMA 自动对 tile 越界区域补零，输出侧仍做 M/N 边界检查。
    return M > 0 && N > 0 && K > 0 && N % 8 == 0 && K % 8 == 0;
}

void solve_v10(
    const __nv_bfloat16 *A, const __nv_bfloat16 *B, float *C,
    int M, int N, int K
) {
    if (!supports_v10(M, N, K))
        return;
    // cudaMalloc 的地址满足对齐；调用者传入切片指针时也必须满足 16 字节对齐。
    if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)) & 15) {
        std::fprintf(stderr, "v10 requires 16-byte-aligned A and B\n");
        std::abort();
    }
    static thread_local TensorMapCache maps;
    maps.update(A, B, M, N, K);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    gemm_bf16_hopper_tma<<<grid, THREADS>>>(maps.map_a, maps.map_b, C, M, N, K);
}
