#include <cute/tensor.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include "cutlass_benchmark.cuh"

namespace {

using Element = cutlass::bfloat16_t;
using RowMajor = cutlass::layout::RowMajor;
using Arch = cutlass::arch::Sm90;
using TensorOp = cutlass::arch::OpClassTensorOp;

// Hopper 原生 TMA + WGMMA + warp specialization。
// 按 tile 选择匹配的主循环/写回调度，Auto 按共享内存预算选择流水线级数。
template<int M, int N, int K>
struct Hopper {
    using Tile = cute::Shape<cute::Int<M>, cute::Int<N>, cute::Int<K>>;
    using Cluster = cute::Shape<cute::_1, cute::_1, cute::_1>;
    using MainloopSchedule = cute::conditional_t<M == 64,
        cutlass::gemm::KernelTmaWarpSpecializedPingpong,
        cutlass::gemm::KernelTmaWarpSpecializedCooperative>;
    using EpilogueSchedule = cute::conditional_t<M == 64,
        cutlass::epilogue::TmaWarpSpecialized,
        cutlass::epilogue::TmaWarpSpecializedCooperative>;
    using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        Arch, TensorOp, Tile, Cluster, cutlass::epilogue::collective::EpilogueTileAuto,
        float, float, float, RowMajor, 4, float, RowMajor, 4,
        EpilogueSchedule
    >::CollectiveOp;
    using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        Arch, TensorOp, Element, RowMajor, 8, Element, RowMajor, 8,
        float, Tile, Cluster,
        cutlass::gemm::collective::StageCountAutoCarveout<int(sizeof(typename Epilogue::SharedStorage))>,
        MainloopSchedule
    >::CollectiveOp;
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int, int, int, int>, Mainloop, Epilogue
    >;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    static typename Gemm::Arguments arguments(const cutlass_bench::Problem &p) {
        // CUTLASS 3.x 的 B 使用逻辑 (N,K,L) 坐标；行主序 B[K,N] 的步长为 (1,N,0)。
        typename GemmKernel::StrideA stride_a{p.k, cute::_1{}, int64_t(0)};
        typename GemmKernel::StrideB stride_b{cute::_1{}, p.n, int64_t(0)};
        typename GemmKernel::StrideC stride_c{p.n, cute::_1{}, int64_t(0)};
        typename GemmKernel::StrideD stride_d{p.n, cute::_1{}, int64_t(0)};
        cutlass::KernelHardwareInfo hardware;
        hardware.device_id = p.device;
        hardware.sm_count = p.sm_count;
        typename Gemm::Arguments args{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {p.m, p.n, p.k, 1},
            {reinterpret_cast<const Element *>(p.a), stride_a,
             reinterpret_cast<const Element *>(p.b), stride_b},
            {{}, p.c, stride_c, p.c, stride_d},
            hardware
        };
        args.epilogue.thread.alpha = 1.0f;
        args.epilogue.thread.beta = 0.0f;
        return args;
    }
};

} // 匿名命名空间

void solve_cutlass(
    const __nv_bfloat16 *A, const __nv_bfloat16 *B, float *C, int M, int N, int K
) {
    using namespace cutlass_bench;
    static thread_local Candidate<Hopper<64, 128, 64>> small("64x128x64-auto");
    static thread_local Candidate<Hopper<128, 128, 64>> medium("128x128x64-auto");
    static thread_local Candidate<Hopper<128, 256, 64>> large("128x256x64-auto");
    static thread_local Autotuner tuner("SM90 TMA/WGMMA", {&small, &medium, &large});
    tuner.run(A, B, C, M, N, K);
}
