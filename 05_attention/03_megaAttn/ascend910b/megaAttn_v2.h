#pragma once
#include "megaAttn_v1.h"

namespace mega_ascend_v2 {
using Input = mega_ascend_v1::Input;
using Output = mega_ascend_v1::Output;
constexpr int HEADS=64, DIM=512, ROPE_DIM=64, KV_TILE=64;
constexpr int SCORE_ELEMS=HEADS*KV_TILE, RAW_ROW_BYTES=544;
constexpr float SCORE_SCALE=mega_ascend_v1::SCORE_SCALE;
constexpr float RESCALE_DELTA=4.1588830833596715f; // ln(64)，指数权重保持在 BF16 可表示范围内
// CANN 9.0 / 910B 在 UB 的最后 8 KiB 使用内部临时区，Select 等 API 会写入。
constexpr int SDK_UB_RESERVE=8*1024, STATE_BYTES=1408;
constexpr int MATMUL_UB_BYTES=0; // TSCM/NZ 输入、GM 输出的这两份 tiling 无需 UB 转换缓存
// raw KV 与 scores/P 的生命周期互斥，复用同一段 UB；work 也用于分批读回 PV。
constexpr int ARENA_BYTES=KV_TILE*RAW_ROW_BYTES;
constexpr int USER_UB_BYTES=HEADS*DIM*4+ARENA_BYTES+5*DIM*4+DIM*2+
                            3*DIM*4+128*4+64+528*4+STATE_BYTES+192*4;
static_assert(USER_UB_BYTES+2*MATMUL_UB_BYTES+SDK_UB_RESERVE<=192*1024);
constexpr uint64_t SCRATCH_PER_CORE=(SCORE_ELEMS+HEADS*DIM)*4;
constexpr uint64_t ScratchBytes(uint32_t blocks) { return blocks*SCRATCH_PER_CORE; }

// 一个任务覆盖一个 query 的全部 64 个 head，共享同一次 KV 搬运和解码。
struct QueryConfig {
    int32_t query=0, position=0;
    int32_t topk=0, extra_topk=0, length=0, extra_length=0;
    int32_t page_size=0, page_stride=0, extra_page_size=0, extra_page_stride=0;
    int32_t indices_stride=0, extra_indices_stride=0;
};
static_assert(sizeof(QueryConfig)==12*sizeof(int32_t));

// 可选诊断输出：AIV 经过各代码段的周期数，异步计算会重叠，不能当作独立 Cube 耗时。
enum ProfilePart { Query, PrepareKv, WaitQk, Softmax, WaitPv, Accumulate, Finish, Total, ProfileCount };
struct Profile { uint64_t cycles[ProfileCount]; };

// 查表内容由精确的 FP32 位模式生成；同一实现也供 CPU 的穷举编解码测试使用。
#if defined(__CCE__)
__aicore__
#endif
constexpr uint32_t E4M3Bits(uint8_t code) {
    uint32_t e=(code>>3)&15, m=code&7, bits=0;
    if (e) bits=e==15 && m==7 ? 0x7fc00000u : ((e+120)<<23)|(m<<20);
    else if (m>=4) bits=(120u<<23)|((m-4)<<21);
    else if (m>=2) bits=(119u<<23)|((m-2)<<22);
    else bits=m ? (118u<<23) : 0u;
    return bits|(uint32_t(code&128)<<24);
}
#if defined(__CCE__)
__aicore__
#endif
constexpr uint32_t E8M0Bits(uint8_t code) {
    return code==0 ? 0x00400000u : code==255 ? 0x7fc00000u : uint32_t(code)<<23;
}
} // namespace mega_ascend_v2

// Q/KV/P 使用 TSCM；两份 KV 的 L1 缓冲共 128 KiB，全部用户 L1 共 200 KiB。
// qk_tiling: A=TSCM/NZ BF16，B=TSCM/NZ BF16 转置，C=GM/ND FP32，形状 64x64x512。
// pv_tiling: A=TSCM/NZ BF16，B=TSCM/NZ BF16 不转置，C=GM/ND FP32，形状 64x512x64。
// 两份 tiling 的 shareUbSize 必须为 0，L1 预算各 64 KiB；scratch 为 ScratchBytes(block_num)。
// configs/profiles 均为设备地址；profiles 可为 nullptr，关闭诊断计时。
// block_num<=min(query_count,可用 AIC 核数)，各配置 query 不重复，索引由调用方保证合法。
extern "C" void launch_mega_attn_v2(
    const mega_ascend_v2::Input &in, const mega_ascend_v2::Output &out,
    const mega_ascend_v2::QueryConfig *configs, uint32_t query_count, uint32_t block_num,
    void *scratch, void *qk_tiling, void *pv_tiling, void *system_workspace, void *stream,
    mega_ascend_v2::Profile *profiles=nullptr);
