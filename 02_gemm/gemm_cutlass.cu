#include <cutlass/gemm/device/gemm.h>
#include "cutlass_benchmark.cuh"

namespace {

using Element = cutlass::bfloat16_t;
using RowMajor = cutlass::layout::RowMajor;
using Output = cutlass::epilogue::thread::LinearCombination<float, 4, float, float>;

// Ampere/Ada 的 BF16 路径使用 Sm80 指令族；nvcc 仍按实际 GPU（如 sm_89）生成机器码。
// Sm89 标签的专用配置主要面向 FP8，不能靠替换这个标签自动升级 BF16 主循环。
template<int M, int N, int K, int WarpM, int WarpN, int Stages>
struct Ampere {
    using Gemm = cutlass::gemm::device::Gemm<
        Element, RowMajor, Element, RowMajor, float, RowMajor, float,
        cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<M, N, K>,
        cutlass::gemm::GemmShape<WarpM, WarpN, K>,
        cutlass::gemm::GemmShape<16, 8, 16>, Output,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, Stages
    >;

    static typename Gemm::Arguments arguments(const cutlass_bench::Problem &p) {
        return {
            {p.m, p.n, p.k},
            {reinterpret_cast<const Element *>(p.a), p.k},
            {reinterpret_cast<const Element *>(p.b), p.n},
            {p.c, p.n}, {p.c, p.n}, {1.0f, 0.0f}
        };
    }
};

} // 匿名命名空间

void solve_cutlass(
    const __nv_bfloat16 *A, const __nv_bfloat16 *B, float *C, int M, int N, int K
) {
    using namespace cutlass_bench;
    static thread_local Candidate<Ampere<64, 128, 32, 32, 64, 3>> small("64x128x32-s3");
    static thread_local Candidate<Ampere<128, 128, 32, 64, 64, 3>> medium("128x128x32-s3");
    // 保留旧配置作为候选，避免仅靠架构/尺寸经验替换导致回退。
    static thread_local Candidate<Ampere<128, 256, 64, 64, 64, 2>> original("128x256x64-s2");
    static thread_local Autotuner tuner("SM80-compatible BF16", {&small, &medium, &original});
    tuner.run(A, B, C, M, N, K);
}
