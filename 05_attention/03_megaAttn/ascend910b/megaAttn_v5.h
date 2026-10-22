#pragma once
#include "megaAttn_v4.h"

namespace mega_ascend_v5 {
// 复用 v4 的输入输出、分片布局及合并核；只读表在 v3 前缀后增加 Q 布局索引。
using namespace mega_ascend_v3;
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

} // namespace mega_ascend_v5

// 一次算子调用依次启动计算核与合并核；Graph 计时也包含两者。
// scratch 至少 ScratchBytes(block_num,query_count) 字节。
// profiles 可为空，否则需 query_count*KV_SPLITS*2 项，只记录分片计算，不含合并。
// 针对 24 个 AIC 核组调优；少开核组仍正确，分片边界不随启动核数变化。
// lookup_tables 必须来自本版 MakeLookupTables()；输入索引由调用方保证合法。
extern "C" void launch_mega_attn_v5(
    const mega_ascend_v5::Input &in,const mega_ascend_v5::Output &out,
    const mega_ascend_v5::QueryConfig *configs,uint32_t query_count,uint32_t block_num,
    void *scratch,void *qk_tiling,void *pv_tiling,void *system_workspace,void *lookup_tables,void *stream,
    mega_ascend_v5::Profile *profiles=nullptr);

