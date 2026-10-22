#pragma once
#include "megaAttn_v3.h"

namespace mega_ascend_v4 {
// 输入输出、Matmul 形状和只读表与 v3 一致，只增加 KV 分片的中间结果。
using namespace mega_ascend_v3;
constexpr int KV_SPLITS=2;
constexpr int PARTIAL_FLOATS=TOTAL_HEADS*(DIM+2);
constexpr uint64_t ScratchBytes(uint32_t blocks,uint32_t queries) {
    return blocks*SCRATCH_PER_CORE+uint64_t(queries)*KV_SPLITS*PARTIAL_FLOATS*4;
}

} // namespace mega_ascend_v4

// 一次算子调用依次启动计算核与合并核；Graph 计时也包含两者。
// scratch 至少 ScratchBytes(block_num,query_count) 字节。
// profiles 可为空，否则需 query_count*KV_SPLITS*2 项，只记录分片计算，不含合并。
// block_num 不超过可用 AIC 核组数；输入索引由调用方保证合法。
extern "C" void launch_mega_attn_v4(
    const mega_ascend_v4::Input &in,const mega_ascend_v4::Output &out,
    const mega_ascend_v4::QueryConfig *configs,uint32_t query_count,uint32_t block_num,
    void *scratch,void *qk_tiling,void *pv_tiling,void *system_workspace,void *lookup_tables,void *stream,
    mega_ascend_v4::Profile *profiles=nullptr);

// 合并所有 KV 分片后，统一做 sink、逆 RoPE 与量化；与计算核位于同一 stream。
extern "C" void finish_mega_attn_v4(const mega_ascend_v4::Input &in,
    const mega_ascend_v4::Output &out,const mega_ascend_v4::QueryConfig *configs,
    uint32_t count,uint32_t blocks,void *partials,void *tables,void *stream);
