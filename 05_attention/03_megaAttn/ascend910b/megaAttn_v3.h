#pragma once
#include "megaAttn_v2.h"

namespace mega_ascend_v3 {
using Input=mega_ascend_v2::Input;
using Output=mega_ascend_v2::Output;
using QueryConfig=mega_ascend_v2::QueryConfig;
using Profile=mega_ascend_v2::Profile;
using mega_ascend_v2::E4M3Bits;
using mega_ascend_v2::E8M0Bits;
using mega_ascend_v2::ProfilePart;
constexpr auto Query=mega_ascend_v2::Query, PrepareKv=mega_ascend_v2::PrepareKv,
    WaitQk=mega_ascend_v2::WaitQk, Softmax=mega_ascend_v2::Softmax,
    WaitPv=mega_ascend_v2::WaitPv, Accumulate=mega_ascend_v2::Accumulate,
    Finish=mega_ascend_v2::Finish, Total=mega_ascend_v2::Total;
constexpr int ProfileCount=mega_ascend_v2::ProfileCount;
constexpr int TOTAL_HEADS=64, HEADS=32, DIM=512, ROPE_DIM=64, KV_TILE=64;
constexpr int DECODE_ROWS=4, DECODE_ELEMS=DECODE_ROWS*DIM;
constexpr int SCORE_ELEMS=HEADS*KV_TILE, ARENA_BYTES=KV_TILE*(DIM+32);
constexpr int STATE_BYTES=1408, SDK_UB_RESERVE=8192, MATMUL_UB_BYTES=0;
constexpr int INDEX_WORDS=3*DIM+128+2*DECODE_ELEMS;
constexpr int WORK_BYTES=3*DECODE_ELEMS*4;
constexpr int USER_UB_BYTES=HEADS*DIM*4+ARENA_BYTES+SCORE_ELEMS*2+WORK_BYTES+DECODE_ELEMS*2+
    2*DECODE_ELEMS*2+INDEX_WORDS*4+64+768*4+STATE_BYTES+192*4+KV_TILE*4;
static_assert(USER_UB_BYTES+SDK_UB_RESERVE<=192*1024);
constexpr float SCORE_SCALE=mega_ascend_v2::SCORE_SCALE;
constexpr float RESCALE_DELTA=mega_ascend_v2::RESCALE_DELTA;
constexpr uint64_t SCRATCH_PER_CORE=TOTAL_HEADS*(KV_TILE+DIM)*4+
    (TOTAL_HEADS*DIM+2*KV_TILE*DIM+TOTAL_HEADS*KV_TILE)*2;
// 每组共享完整 64 head 的分数/PV；两个 AIV 分别消费自己的 32 行。
constexpr uint64_t ScratchBytes(uint32_t blocks) { return blocks*SCRATCH_PER_CORE; }

// 一个 FP4 字节直接查出相邻两个 BF16，保留正负零。
#if defined(__CCE__)
__aicore__
#endif
constexpr uint32_t Fp4PairBits(uint8_t code) {
    constexpr uint16_t values[8]={0,0x3f00,0x3f80,0x3fc0,0x4000,0x4040,0x4080,0x40c0};
    uint32_t lo=values[code&7]|((uint32_t(code)&8)<<12);
    uint32_t hi=values[(code>>4)&7]|((uint32_t(code)&128)<<8);
    return lo|(hi<<16);
}
constexpr int TABLE_WORDS=768+INDEX_WORDS+16;
// 固定查表数据由主机生成一次，所有核和所有 Graph 重放复用同一份只读 GM。
struct LookupTables { uint32_t words[TABLE_WORDS]{}; };
#if !defined(__CCE__)
inline LookupTables MakeLookupTables() {
    LookupTables data;
    for (int i=0;i<256;++i) {
        data.words[i]=E4M3Bits(uint8_t(i));
        data.words[256+i]=E8M0Bits(uint8_t(i));
        data.words[512+i]=Fp4PairBits(uint8_t(i));
    }
    auto idx=data.words+768;
    for (int d=0;d<DIM;++d) {
        idx[d]=(d/2)*4; idx[DIM+d]=(d/32)*4; idx[2*DIM+d]=(d/16)*4;
    }
    for (int d=0;d<64;++d) { idx[3*DIM+d]=(d^1)*4; idx[3*DIM+64+d]=(d/2)*4; }
    for (int d=0;d<DECODE_ELEMS;++d) {
        idx[3*DIM+128+d]=((d/DIM)*32+(d%DIM)/32)*4;
        idx[3*DIM+128+DECODE_ELEMS+d]=((d/DIM)*32+(d%DIM)/16)*4;
    }
    for (int i=0;i<16;++i) data.words[768+INDEX_WORDS+i]=0xaaaaaaaau;
    return data;
}
#endif
} // namespace mega_ascend_v3

// 每个 AIV 处理 32 head；Cube QK=64x64x512，PV=64x512x64。
// scratch 含两侧共享的 ND 输入和 FP32 结果；Matmul 各保留 128 KiB L1。
// lookup_tables 指向上传后的 MakeLookupTables()；profiles 可为空，否则需 query_count*2 项。
extern "C" void launch_mega_attn_v3(
    const mega_ascend_v3::Input &in,const mega_ascend_v3::Output &out,
    const mega_ascend_v3::QueryConfig *configs,uint32_t query_count,uint32_t block_num,
    void *scratch,void *qk_tiling,void *pv_tiling,void *system_workspace,void *lookup_tables,void *stream,
    mega_ascend_v3::Profile *profiles=nullptr);
