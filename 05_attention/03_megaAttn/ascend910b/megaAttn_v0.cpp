#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "megaAttn_api.h"

using namespace AscendC;
using namespace mega_ascend;

// 阶段 1 使用 GM 暂存连接 Vector/Cube，先让每个观测点都能独立对拍。
using QType = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;
using KType = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, true>;
using ScoreType = MatmulType<TPosition::GM, CubeFormat::ND, float>;
using QkMatmul = Matmul<QType, KType, ScoreType>;

// 你主要编辑这个类中的六个 TODO；Init/Process/启动入口由骨架负责。
class KernelMegaStage1 {
public:
    QkMatmul mm;

    __aicore__ inline void Init(
        GM_ADDR q, GM_ADDR kv8, GM_ADDR kv4, GM_ADDR ids, GM_ADDR extra_ids,
        GM_ADDR rope, GM_ADDR config, GM_ADDR q_loaded, GM_ADDR q_rope,
        GM_ADDR kv, GM_ADDR valid, GM_ADDR scores, GM_ADDR status, TPipe *pipe) {
        // DataCopy 的数量通常是元素数，InitBuffer 的大小是字节数。
        GlobalTensor<int32_t> configGm;
        configGm.SetGlobalBuffer((__gm__ int32_t *)config, 16);
        int32_t *fields = reinterpret_cast<int32_t *>(&cfg);
        for (int i = 0; i < 16; ++i) fields[i] = configGm.GetValue(i);
        qGm.SetGlobalBuffer((__gm__ bfloat16_t *)q);
        kv8Gm.SetGlobalBuffer((__gm__ uint8_t *)kv8);
        kv4Gm.SetGlobalBuffer((__gm__ uint8_t *)kv4);
        idsGm.SetGlobalBuffer((__gm__ int32_t *)ids);
        extraIdsGm.SetGlobalBuffer((__gm__ int32_t *)extra_ids);
        ropeGm.SetGlobalBuffer((__gm__ float *)rope);
        qLoadedGm.SetGlobalBuffer((__gm__ bfloat16_t *)q_loaded, Q_ELEMS);
        qRopeGm.SetGlobalBuffer((__gm__ bfloat16_t *)q_rope, Q_ELEMS);
        kvGm.SetGlobalBuffer((__gm__ bfloat16_t *)kv, KV_ELEMS);
        validGm.SetGlobalBuffer((__gm__ int32_t *)valid, KV_TILE);
        scoreGm.SetGlobalBuffer((__gm__ float *)scores, SCORE_ELEMS);
        statusGm.SetGlobalBuffer((__gm__ int32_t *)status, 8);
        pipe->InitBuffer(qBuf, Q_ELEMS * sizeof(bfloat16_t));
        pipe->InitBuffer(rawBuf, KV_TILE * RAW_ROW_BYTES);
        pipe->InitBuffer(kvBuf, KV_ELEMS * sizeof(bfloat16_t));
        pipe->InitBuffer(validBuf, KV_TILE * sizeof(int32_t));
        pipe->InitBuffer(scoreBuf, SCORE_ELEMS * sizeof(float));
        pipe->InitBuffer(statusBuf, 32);
    }

    __aicore__ inline void Process() {
        if (!LoadQuery())       { Report(Step::LoadQuery); return; }
        if (!ApplyQueryRope())  { Report(Step::ApplyQueryRope); return; }
        if (!LoadKvTile())      { Report(Step::LoadKvTile); return; }
        if (!DecodeKvTile())    { Report(Step::DecodeKvTile); return; }
        if (!ComputeQk())       { Report(Step::ComputeQk); return; }
        if (!MaskScores())      { Report(Step::MaskScores); return; }
        Report(Step::Done);
    }

private:
    __aicore__ inline bool LoadQuery() {
        // TODO 1：把 Q 的 head_begin..head_begin+15 重排进 qBuf，并导出 qLoadedGm。
        // 原始索引：query*64*512 + (d/16)*64*16 + head*16 + d%16。
        // 每次可搬同一个 head 的 16 个 BF16，正好 32 B；目标是 [16,512]。
        return false;
    }

    __aicore__ inline bool ApplyQueryRope() {
        // TODO 2：只旋转 qBuf 的最后 64 维，舍入回 BF16，再导出 qRopeGm。
        // position 决定 cos_sin 行；旋转配对为 (448+2*j,449+2*j)。
        return false;
    }

    __aicore__ inline bool LoadKvTile() {
        // TODO 3：顺序拼接 topk 个 FP8 位置和 extra_topk 个 FP4 位置。
        // 拼接分界按容量 topk，不是 length；length 只决定位置是否有效。
        // 索引已是物理槽号：page=id/page_size，token=id%page_size，不再查页表。
        // rawBuf 每行 544 B：数据放 [0,512)，scale 放 [512,544)；FP4 数据只有 256 B。
        // 无效索引不得发起越界搬运；记录 validBuf 并导出 validGm。
        return false;
    }

    __aicore__ inline bool DecodeKvTile() {
        // TODO 4：rawBuf → kvBuf 的 BF16 [64,512]，无效行填 0，导出 kvGm。
        // FP8：E4M3 数据，32 个值共享一个 UE8M0 scale。
        // FP4：E2M1 两值/字节（低半字节在前），16 个值共享一个 E4M3 scale。
        return false;
    }

    __aicore__ inline bool ComputeQk() {
        // TODO 5：用已注册的 mm 计算 qRopeGm * transpose(kvGm) → scoreGm。
        // 形状是 [16,512] * [512,64]，B 的物理存储仍为 [64,512]。
        // Vector 写 GM 必须完成后再发起 Matmul；读取结果前等待 Matmul 完成。
        return false;
    }

    __aicore__ inline bool MaskScores() {
        // TODO 6：scoreGm → scoreBuf，乘 SCORE_SCALE，无效列写 -inf，再导出。
        // 整列屏蔽：同一个 KV 的 valid 标记供 16 个 head 共用。
        return false;
    }

    __aicore__ inline void Report(Step step) {
        auto local = statusBuf.Get<int32_t>();
        for (int i = 0; i < 8; ++i) local.SetValue(i, static_cast<int32_t>(step));
        auto event = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(event);
        WaitFlag<HardEvent::S_MTE3>(event);
        DataCopy(statusGm, local, 8);
    }

    Stage1Config cfg;
    GlobalTensor<bfloat16_t> qGm, qLoadedGm, qRopeGm, kvGm;
    GlobalTensor<uint8_t> kv8Gm, kv4Gm;
    GlobalTensor<int32_t> idsGm, extraIdsGm, validGm, statusGm;
    GlobalTensor<float> ropeGm, scoreGm;
    TBuf<TPosition::VECCALC> qBuf, rawBuf, kvBuf, validBuf, scoreBuf, statusBuf;
};

extern "C" __global__ __aicore__ void mega_attn_stage1(
    GM_ADDR q, GM_ADDR kv8, GM_ADDR kv4, GM_ADDR ids, GM_ADDR extra_ids,
    GM_ADDR rope, GM_ADDR config, GM_ADDR q_loaded, GM_ADDR q_rope,
    GM_ADDR kv, GM_ADDR valid, GM_ADDR scores, GM_ADDR status,
    GM_ADDR cube_tiling, GM_ADDR workspace) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    TPipe pipe;
    KernelMegaStage1 op;
    TCubeTiling tiling;
    auto src = reinterpret_cast<__gm__ uint32_t *>(cube_tiling);
    auto dst = reinterpret_cast<uint32_t *>(&tiling);
    for (int i = 0; i < sizeof(TCubeTiling) / sizeof(uint32_t); ++i) dst[i] = src[i];
    SetSysWorkspace(workspace);
    // 注册宏让 AIC 处理 Matmul 消息；其余核心算法从 AIV 侧驱动。
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), op.mm, &tiling);
    op.Init(q, kv8, kv4, ids, extra_ids, rope, config, q_loaded, q_rope,
            kv, valid, scores, status, &pipe);
    op.Process();
}

extern "C" void launch_mega_attn_stage1(
    const Stage1Input &in, const Stage1Output &out,
    void *config, void *cube_tiling, void *workspace, void *stream) {
    mega_attn_stage1<<<1, nullptr, stream>>>(
        (GM_ADDR)in.q, (GM_ADDR)in.kv_fp8, (GM_ADDR)in.kv_fp4,
        (GM_ADDR)in.indices, (GM_ADDR)in.extra_indices, (GM_ADDR)in.cos_sin,
        (GM_ADDR)config, (GM_ADDR)out.q_loaded, (GM_ADDR)out.q_rope,
        (GM_ADDR)out.kv, (GM_ADDR)out.valid, (GM_ADDR)out.scores,
        (GM_ADDR)out.status, (GM_ADDR)cube_tiling, (GM_ADDR)workspace);
}
