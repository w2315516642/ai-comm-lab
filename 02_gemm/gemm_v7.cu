#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

namespace {
// 匿名 namespace：这些名字只在当前 .cu 文件中可见，避免与其他文件中的
// BM、BK 等同名符号发生链接冲突。

// 一条 mma.sync 指令计算 D[16, 8] = A[16, 16] * B[16, 8] + C[16, 8]。
// 一个 warp 连续执行两条指令，最终得到一个 16x16 的输出 tile。
constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 16;

// 一个 block 有 2x2 个 warp，每个 warp 负责 16x16，因此 block 输出 32x32。
// K 方向每轮只处理 16，随后沿 K 方向循环累加。
constexpr int WARPS_M = 2;
constexpr int WARPS_N = 2;
constexpr int BM = WARPS_M * MMA_M;
constexpr int BN = WARPS_N * (2 * MMA_N);
constexpr int BK = MMA_K;
constexpr int WARPS = WARPS_M * WARPS_N;
constexpr int THREADS = WARPS * 32;

// 一个 shared-memory stage 包含一个 A[BM, BK] tile 和一个 B[BK, BN]
// tile。v7 准备两个 stage：一个用于当前计算，另一个用于 cp.async 预取。
constexpr int A_STAGE_SIZE = BM * BK;
constexpr int B_STAGE_SIZE = BK * BN;
constexpr int STAGE_SIZE = A_STAGE_SIZE + B_STAGE_SIZE;

// __device__：函数只能在 GPU 上调用。
// __forceinline__：要求编译器尽量把函数体直接展开到调用处，省掉函数调用，
// 同时也方便编译器继续优化这些很短的辅助函数。
__device__ __forceinline__ uint32_t shared_address(const void *ptr) {
    // ldmatrix 的 PTX 操作数要求 shared-memory 地址，而普通 C++ 指针是
    // generic address，因此需要显式转换成 shared address。
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void load_matrix_a(
    uint32_t (&a)[4], const __nv_bfloat16 *tile, int lane
) {
    // uint32_t (&a)[4] 是“长度为 4 的 uint32_t 数组的引用”。
    // 它不会复制数组，并且编译器能够在函数签名中检查数组长度。
    // 如果写成 uint32_t *a，就会丢失“长度必须为 4”这条信息。
    // const __nv_bfloat16 * 表示通过 tile 只能读取 BF16，不能修改它。
    // ldmatrix 的基本单位是一个 8x8、每元素 16 bit 的小矩阵。
    // A 是 16x16，因此使用 x4，一次得到四个 8x8 象限。
    //
    // lane 0..15 提供每一行前 8 个元素的地址；
    // lane 16..31 提供相同行后 8 个元素的地址。
    // ldmatrix 会在 warp 内重新分发数据，每个 lane 最终得到 4 个 32-bit
    // 寄存器，每个寄存器装着两个 BF16。
    int row = lane % 16;
    int col = (lane / 16) * 8;
    uint32_t address = shared_address(&tile[row * BK + col]);
    // asm volatile(...)：在 CUDA C++ 中嵌入一条 PTX 汇编指令。
    // volatile 防止编译器认为汇编没有副作用而删除或随意移动它。
    // 冒号之后依次是“输出操作数 : 输入操作数”。
    // "=r" 表示写入一个 32-bit 整数寄存器，"r" 表示读取整数寄存器；
    // %0、%1 等占位符按操作数出现的顺序编号。
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                 "{%0, %1, %2, %3}, [%4];\n"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                 : "r"(address));
}

__device__ __forceinline__ void load_matrix_b(
    uint32_t (&b)[2], const __nv_bfloat16 *tile, int lane, int col
) {
    // 每条 mma 指令需要 B[16, 8]。它由两个 8x8 小矩阵组成，因此使用 x2。
    // B 在 shared memory 中仍按 row-major 保存，而下面的 mma 使用 row.col，
    // 所以通过 .trans 得到 operand B 所要求的 column-major 寄存器布局。
    // x2 只需要 16 个行地址，硬件读取 lane 0..15 提供的地址；这里让
    // lane 16..31 重复前 16 个地址，写法更统一。
    int row = lane % 16;
    uint32_t address = shared_address(&tile[row * BN + col]);
    // 与上面的汇编约束相同：b[0]、b[1] 是输出，address 是输入。
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(b[0]), "=r"(b[1])
        : "r"(address)
    );
}

__device__ __forceinline__ void mma_m16n8k16(
    float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]
) {
    // const uint32_t (&a)[4] 表示“对数组的只读引用”：既不复制数组，也不
    // 允许此函数修改 a。d 没有 const，因为 MMA 要原地更新 accumulator。
    //
    // d 同时作为输入 accumulator C 和输出 D，因此约束使用 "+f"。
    // "+" 表示该操作数既读又写，"f" 表示 FP32 寄存器。
    // a[4] 共包含 8 个 BF16，b[2] 共包含 4 个 BF16；这些并不是某个
    // 线程独立拥有的完整矩阵，而是 MMA 规定的 warp 分布式 fragment。
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

__device__ __forceinline__ void async_load_tile(
    __nv_bfloat16 *As,
    __nv_bfloat16 *Bs,
    const __nv_bfloat16 *A,
    const __nv_bfloat16 *B,
    int N,
    int K,
    int tid
) {
    // 指针参数没有 __restrict__，因为这里的重点是表达 shared/global 两侧
    // 地址；主 kernel 上的 __restrict__ 才负责告诉编译器三块全局内存不重叠。
    //
    // A tile 和 B tile 都有 512 个 BF16。block 有 128 个线程，因此每个
    // 线程恰好为 A、B 各搬 4 个 BF16，也就是两次 8-byte cp.async。
    // solve_v7 限制了矩阵维度，保证源地址与目标地址满足对齐要求。
    constexpr int VALUES_PER_COPY = 4;
    int index = tid * VALUES_PER_COPY;

    int a_row = index / BK;
    int a_col = index % BK;
    // __pipeline_memcpy_async(dst, src, bytes) 是 CUDA 提供的 device
    // intrinsic。 在 sm_80+ 且满足大小/对齐要求时，它会生成 global -> shared 的
    // cp.async。
    __pipeline_memcpy_async(
        &As[index],
        &A[a_row * K + a_col],
        VALUES_PER_COPY * sizeof(__nv_bfloat16)
    );

    int b_row = index / BN;
    int b_col = index % BN;
    __pipeline_memcpy_async(
        &Bs[index],
        &B[b_row * N + b_col],
        VALUES_PER_COPY * sizeof(__nv_bfloat16)
    );
}

__device__ __forceinline__ void store_accumulator(
    float *C,
    const float (&accum)[2][4],
    int N,
    int warp_row,
    int warp_col,
    int lane
) {
    // const float (&accum)[2][4] 是“2x4 二维数组的只读引用”。和指针相比，
    // 这种写法保留了两个维度，因而可以继续使用 accum[i][j]，且不会复制数据。
    // 一条 m16n8k16 的四个 FP32 accumulator 在 lane 间的布局为：
    //
    //   lane / 4       -> 第 0 个输出行
    //   lane / 4 + 8   -> 第 1 个输出行
    //   (lane % 4) * 2 -> 每行相邻两个输出列
    //
    // accum[n_half][0:1] 属于第一行，accum[n_half][2:3] 属于第二行。
    // n_half=0/1 分别来自两条 MMA，对应列区间 0..7 和 8..15。
    int row_in_half = lane / 4;
    int col_pair = (lane % 4) * 2;
    int base_row = warp_row * 16;
    int base_col = warp_col * 16;

    for (int n_half = 0; n_half < 2; n_half++) {
        int col = base_col + n_half * 8 + col_pair;
        C[(base_row + row_in_half) * N + col + 0] = accum[n_half][0];
        C[(base_row + row_in_half) * N + col + 1] = accum[n_half][1];
        C[(base_row + row_in_half + 8) * N + col + 0] = accum[n_half][2];
        C[(base_row + row_in_half + 8) * N + col + 1] = accum[n_half][3];
    }
}

// __global__：这是一个 kernel，函数在 GPU 上执行、由 CPU 通过 <<<>>> 启动。
// kernel 必须返回 void。
__global__ void gemm_bf16_ldmatrix_mma(
    const __nv_bfloat16 *__restrict__ A,
    const __nv_bfloat16 *__restrict__ B,
    float *__restrict__ C,
    int M,
    int N,
    int K
) {
    // __restrict__ 告诉编译器 A、B、C 指向的内存彼此不重叠，使其可以更大胆地
    // 重排 load/store。它是优化承诺：调用者必须保证这个承诺确实成立。
    //
    // extern __shared__ T name[] 声明“动态 shared memory”。数组大小不是在这里
    // 写死，而是由启动 kernel 时 <<<grid, block, SHARED_BYTES>>> 的第三项决定。
    // shared 的实际排列如下：
    //
    //   stage 0: [A0 | B0]
    //   stage 1: [A1 | B1]
    //
    // As/Bs 只是给这四段内存取了更容易理解的名字。
    extern __shared__ __nv_bfloat16 shared[];
    __nv_bfloat16 *As[2] = {shared, shared + STAGE_SIZE};
    __nv_bfloat16 *Bs[2] = {
        shared + A_STAGE_SIZE, shared + STAGE_SIZE + A_STAGE_SIZE
    };

    // threadIdx/blockIdx 是 CUDA 内建变量；这里使用一维线程块和二维 grid。
    // threadIdx.x：线程在当前 block 内的编号。
    // blockIdx.x/y：当前 block 在 grid 的列/行编号。
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;

    // 四个 warp 在 block 输出 tile 中排成：
    //
    //   warp 0 | warp 1
    //   -------+-------
    //   warp 2 | warp 3
    //
    // 每个格子是 16x16，整个 block 因而覆盖 32x32。
    int warp_row = warp_id / WARPS_N;
    int warp_col = warp_id % WARPS_N;
    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;
    const __nv_bfloat16 *A_block = A + block_row * K;
    const __nv_bfloat16 *B_block = B + block_col;

    // 两组 accumulator 对应输出 tile 左右两个 16x8 半块。
    // 它们在整个 K 循环期间都保留在寄存器里。
    // = {} 对聚合类型做零初始化，因此 8 个 float 全部从 0.0f 开始。
    float accum[2][4] = {};

    // Pipeline prologue：循环开始前必须先让 stage 0 可用。
    async_load_tile(As[0], Bs[0], A_block, B_block, N, K, tid);
    // commit 把之前发出的异步 copy 组成一个 group；wait_prior(0) 等待所有
    // 已提交 group 完成。__syncthreads() 则是整个 block 的线程屏障。
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    int tile_count = K / BK;
    for (int tile = 0; tile < tile_count; tile++) {
        // stage 0 和 stage 1 随 tile 奇偶交替承担读、写角色。
        int read_stage = tile & 1;
        int write_stage = read_stage ^ 1;
        bool has_next = tile + 1 < tile_count;

        if (has_next) {
            // 只提交异步请求，不在这里等待。随后执行的 ldmatrix/mma 使用
            // read_stage，因此可以和写入 write_stage 的 cp.async 重叠。
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

        uint32_t a[4];
        uint32_t b[2];

        // 每个 warp 选择 A 中自己的 16 行，以及 B 中自己的 16 列。
        // 两个 warp 可以复用相同的 A，另外两个 warp 可以复用相同的 B。
        const __nv_bfloat16 *warp_A = As[read_stage] + warp_row * MMA_M * BK;
        const __nv_bfloat16 *warp_B = Bs[read_stage] + warp_col * 2 * MMA_N;
        load_matrix_a(a, warp_A, lane);

        // 第一条 MMA 计算该 warp 输出 tile 的左半边 16x8。
        load_matrix_b(b, warp_B, lane, 0);
        mma_m16n8k16(accum[0], a, b);

        // A fragment 可以复用；重新加载 B 的后 8 列，计算右半边 16x8。
        load_matrix_b(b, warp_B, lane, MMA_N);
        mma_m16n8k16(accum[1], a, b);

        if (has_next) {
            // 等待下一 stage 的异步拷贝完成，再做 block barrier。
            // barrier 还有第二个作用：保证所有 warp 都读完当前 stage，之后
            // 它才可以在下一轮被 cp.async 安全覆盖。
            __pipeline_wait_prior(0);
            __syncthreads();
        }
    }

    // K 方向全部累加完成后，每个 lane 把自己的 8 个 FP32 结果写回。
    float *C_block = C + block_row * N + block_col;
    store_accumulator(C_block, accum, N, warp_row, warp_col, lane);
}

} // namespace

void solve_v7(
    const __nv_bfloat16 *A,
    const __nv_bfloat16 *B,
    float *C,
    int M,
    int N,
    int K
) {
    // solve_v7 是普通 host 函数：CPU 调用它，它再负责配置并启动 GPU kernel。
    // 为了把边界判断从教学 kernel 中拿掉，launcher 只接受完整 tile。
    // 这也确保了 8-byte cp.async 和 ldmatrix 所需的地址对齐。
    if (M % BM != 0 || N % BN != 0 || K % BK != 0)
        return;

    // constexpr 表示编译期常量；sizeof 返回一个 BF16 元素所占的字节数。
    constexpr int SHARED_BYTES = 2 * STAGE_SIZE * sizeof(__nv_bfloat16);

    // dim3 是 CUDA 的三维尺寸类型。只提供一个参数时，y 和 z 默认为 1。
    dim3 block(THREADS);
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));

    // CUDA kernel 启动语法：kernel<<<grid, block, dynamic_smem_bytes>>>(参数)。
    // 调用是异步的：这里只把工作提交给 GPU，不会等待 kernel 执行完毕。
    gemm_bf16_ldmatrix_mma<<<grid, block, SHARED_BYTES>>>(A, B, C, M, N, K);
}
