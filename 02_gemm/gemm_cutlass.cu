#include <cstdio>
#include <cstdlib>

#include <cuda_bf16.h>

#include <cutlass/arch/arch.h>
#include <cutlass/arch/mma.h>
#include <cutlass/bfloat16.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>

using RowMajor = cutlass::layout::RowMajor;
using CutlassConfig = cutlass::gemm::device::DefaultGemmConfiguration<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::bfloat16_t,
    cutlass::bfloat16_t,
    float,
    float>;

using CutlassGemm = cutlass::gemm::device::Gemm<
    cutlass::bfloat16_t,
    RowMajor,
    cutlass::bfloat16_t,
    RowMajor,
    float,
    RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    CutlassConfig::ThreadblockShape,
    CutlassConfig::WarpShape,
    CutlassConfig::InstructionShape,
    CutlassConfig::EpilogueOutputOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2>;

void solve_cutlass(
    const __nv_bfloat16 *A,
    const __nv_bfloat16 *B,
    float *C,
    int M,
    int N,
    int K
) {
    CutlassGemm gemm_operator;

    auto cutlass_A = reinterpret_cast<const cutlass::bfloat16_t *>(A);
    auto cutlass_B = reinterpret_cast<const cutlass::bfloat16_t *>(B);

    // clang-format off
    CutlassGemm::Arguments arguments(
        {M, N, K},
        {cutlass_A, K},
        {cutlass_B, N},
        {C, N},
        {C, N},
        {1.0f, 0.0f}
    );
    // clang-format on

    cutlass::Status status = gemm_operator(arguments);

    if (status != cutlass::Status::kSuccess) {
        fprintf(
            stderr,
            "CUTLASS GEMM failed: %s\n",
            cutlass::cutlassGetStatusString(status)
        );
        abort();
    }
}
