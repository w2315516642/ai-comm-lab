#pragma once
#include <cstdint>
#include <cuda_runtime_api.h>

// 固定 Decode：Hq=64、Hkv=1、D=512、RoPE=64，不做 Q norm。
// 所有指针指向当前 SM100 设备；必填缓冲区由调用方分配并保证大小、对齐正确。
// Q 为 BF16，物理布局 [N,32,64,16]；KV 入缓存前已完成位置变换。
// 每页先放所有 token 的量化数据，再放 scale，最后补齐 page_stride。
// 物理索引的负数/越界值会被屏蔽；position 必须有效，length 必须在 [0,topk] 内。
// 调用算子时传入的参数，汇总输入输出指针、数据尺寸和存储步长。
struct MegaAttnParams {
    const uint16_t *q = nullptr;
    // 主缓存：每 token 512 字节 E4M3 数据和 16 字节 UE8M0 scale；二者在页内分区存放。
    const uint8_t *kv_fp8 = nullptr;
    // 额外缓存：每 token 256 字节 E2M1 数据和 32 字节 E4M3 scale；数据每字节装两个值。
    const uint8_t *kv_fp4 = nullptr;
    // 两份索引分别定位各自缓存，值=物理页号*该缓存的 page_size+页内 token 序号。
    const int32_t *indices = nullptr;
    const int32_t *extra_indices = nullptr;
    const int32_t *positions = nullptr; // [N]
    const float *cos_sin =
        nullptr; // [max_position,64]，前 32 个 cos，后 32 个 sin
    const float *attn_sink = nullptr;     // 可选 [64]，不计入 LSE
    const int32_t *topk_length = nullptr; // 可选 [N]，为空时使用 topk
    const int32_t *extra_topk_length =
        nullptr;                // 可选 [N]，为空时使用 extra_topk
    // 每Q输出64×512=32768 B；拆成8组×16维度块×8head×32值，乘积仍为32768。
    uint8_t *out_fp8 = nullptr; // E4M3，物理布局 [N,8,16,8,32]
    // 每Q有64×(512/32)=1024个scale字节，打包后是256个uint32；含padding的总容量为256×sf_stride个uint32。
    uint32_t *out_sf = nullptr; // 每四个 UE8M0 指数打包为 uint32
    float *lse = nullptr; // [N,64]，自然对数；无有效索引时为 +inf
    // n 是一次调用的 Q 数量，每个 Q 包含固定的 64 个 head。
    int n = 0;
    int num_pages = 0, page_size = 0, page_stride = 0; // page_stride 单位为字节
    int extra_num_pages = 0, extra_page_size = 0, extra_page_stride = 0;
    // topk/extra_topk 是两路索引行的容量；逐 Q 的 length 只规定该行前多少个位置有效。
    int topk = 0, extra_topk = 0;
    // stride 是相邻 Q 的索引行起点间距，允许尾部 padding，因此不必等于 topk。
    int indices_stride = 0, extra_indices_stride = 0; // int32 元素数，8 的倍数
    int sf_stride =
        0; // uint32 元素数，>=N 且为 4 的倍数；组间步长为 32*sf_stride
    float sm_scale = 0.04419417382415922f; // 1/sqrt(512)
};

using SolveFn = void (*)(const MegaAttnParams &, cudaStream_t);
extern "C" void solve_v0(const MegaAttnParams &params, cudaStream_t stream);

// 新建 megaAttn_v1.cu 并实现相同接口后，run.sh 会自动加入比较。
#ifdef ENABLE_MEGA_ATTN_V1
extern "C" void solve_v1(const MegaAttnParams &params, cudaStream_t stream);
#endif
