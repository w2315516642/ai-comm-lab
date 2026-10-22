#pragma once
#include "megaAttn_v4.h"

namespace mega_ascend_v6 {
// 复用 v4 的输入输出和分片布局，使用本版的合并核；只读表在 v3 前缀后增加 Q 布局索引。
using mega_ascend_v3::Input;
using mega_ascend_v3::Output;
using mega_ascend_v3::QueryConfig;
using mega_ascend_v3::Profile;
using mega_ascend_v3::TOTAL_HEADS;
using mega_ascend_v3::HEADS;
using mega_ascend_v3::DIM;
using mega_ascend_v3::ROPE_DIM;
using mega_ascend_v3::DECODE_ROWS;
using mega_ascend_v3::DECODE_ELEMS;
using mega_ascend_v3::INDEX_WORDS;
using mega_ascend_v3::WORK_BYTES;
using mega_ascend_v3::STATE_BYTES;
using mega_ascend_v3::SDK_UB_RESERVE;
using mega_ascend_v3::MATMUL_UB_BYTES;
using mega_ascend_v3::SCORE_SCALE;
using mega_ascend_v3::RESCALE_DELTA;
using mega_ascend_v3::Query;
using mega_ascend_v3::PrepareKv;
using mega_ascend_v3::WaitQk;
using mega_ascend_v3::Softmax;
using mega_ascend_v3::WaitPv;
using mega_ascend_v3::Accumulate;
using mega_ascend_v3::Finish;
using mega_ascend_v3::Total;
using mega_ascend_v3::ProfileCount;
// 每个 AIV 解码半个 tile，并计算 32 个 head。
constexpr int KV_TILE=128, KV_ROWS=KV_TILE/2;
constexpr int SCORE_ELEMS=HEADS*KV_TILE, ARENA_BYTES=KV_ROWS*(DIM+32);
constexpr uint64_t SCRATCH_PER_CORE=TOTAL_HEADS*(KV_TILE+DIM)*4+
    (TOTAL_HEADS*DIM+2*KV_TILE*DIM+TOTAL_HEADS*KV_TILE)*2;
constexpr int USER_UB_BYTES=HEADS*DIM*4+ARENA_BYTES+SCORE_ELEMS*2+WORK_BYTES+DECODE_ELEMS*2+
    2*DECODE_ELEMS*2+INDEX_WORDS*4+64+768*4+STATE_BYTES+192*4+KV_TILE*4;
constexpr int KV_SPLITS=2;
constexpr int PARTIAL_FLOATS=mega_ascend_v3::TOTAL_HEADS*(mega_ascend_v3::DIM+2);
static_assert(KV_SPLITS==mega_ascend_v4::KV_SPLITS && PARTIAL_FLOATS==mega_ascend_v4::PARTIAL_FLOATS);
constexpr uint64_t ScratchBytes(uint32_t blocks,uint32_t queries) {
    return blocks*SCRATCH_PER_CORE+uint64_t(queries)*KV_SPLITS*PARTIAL_FLOATS*4;
}

constexpr int QUERY_INDEX_WORDS=4*mega_ascend_v3::DIM;
constexpr int TABLE_WORDS=mega_ascend_v3::TABLE_WORDS+QUERY_INDEX_WORDS;
// 主机一次生成并上传，所有核与 Graph 重放共用。
struct LookupTables { uint32_t words[TABLE_WORDS]{}; };
#if !defined(__CCE__)
inline LookupTables MakeLookupTables() {
    LookupTables table;
    auto base=mega_ascend_v3::MakeLookupTables();
    for(int i=0;i<mega_ascend_v3::TABLE_WORDS;++i) table.words[i]=base.words[i];
    // 一次搬 4 个 head 后，Q 的 UB 布局为 [D/16,4,16]，转换到 [4,D]。
    for(int h=0;h<4;++h) for(int d=0;d<mega_ascend_v3::DIM;++d)
        table.words[mega_ascend_v3::TABLE_WORDS+h*mega_ascend_v3::DIM+d]=((d/16)*64+h*16+d%16)*4;
    return table;
}
#endif

} // namespace mega_ascend_v6

// 一次算子调用依次启动计算核与合并核；Graph 计时也包含两者。
// scratch 至少 ScratchBytes(block_num,query_count) 字节。
// profiles 可为空，否则需 query_count*KV_SPLITS*2 项，只记录分片计算，不含合并。
// 针对 24 个 AIC 核组调优；少开核组仍正确，分片边界不随启动核数变化。
// lookup_tables 必须来自本版 MakeLookupTables()；输入索引由调用方保证合法。
extern "C" void launch_mega_attn_v6(
    const mega_ascend_v6::Input &in,const mega_ascend_v6::Output &out,
    const mega_ascend_v6::QueryConfig *configs,uint32_t query_count,uint32_t block_num,
    void *scratch,void *qk_tiling,void *pv_tiling,void *system_workspace,void *lookup_tables,void *stream,
    mega_ascend_v6::Profile *profiles=nullptr);


extern "C" void finish_mega_attn_v6(const mega_ascend_v6::Input&,const mega_ascend_v6::Output&,const mega_ascend_v6::QueryConfig*,uint32_t,uint32_t,void*,void*,void*);
