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
        GM_ADDR kv, GM_ADDR valid, GM_ADDR scores, GM_ADDR status, TPipe *pipe
    ) {
        // DataCopy 的数量通常是元素数，InitBuffer 的大小是字节数。
        GlobalTensor<int32_t> configGm;
        configGm.SetGlobalBuffer((__gm__ int32_t *)config, 16);
        int32_t *fields = reinterpret_cast<int32_t *>(&cfg);
        for (int i = 0; i < 16; ++i)
            fields[i] = configGm.GetValue(i);

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
        if (!LoadQuery()) {
            Report(Step::LoadQuery);
            return;
        }
        if (!ApplyQueryRope()) {
            Report(Step::ApplyQueryRope);
            return;
        }
        if (!LoadKvTile()) {
            Report(Step::LoadKvTile);
            return;
        }
        if (!DecodeKvTile()) {
            Report(Step::DecodeKvTile);
            return;
        }
        if (!ComputeQk()) {
            Report(Step::ComputeQk);
            return;
        }
        if (!MaskScores()) {
            Report(Step::MaskScores);
            return;
        }
        Report(Step::Done);
    }

  protected:
    __aicore__ inline bool LoadQuery() {
        // TODO 1：把 Q 的 head_begin..head_begin+15 重排进 qBuf，并导出
        // qLoadedGm。 原始索引：query*64*512 + (d/16)*64*16 + head*16 + d%16。
        // 每次可搬同一个 head 的 16 个 BF16，正好 32 B；目标是 [16,512]。
        auto qLocal = qBuf.Get<bfloat16_t>();

        constexpr uint32_t CHUNK = 16;
        uint64_t queryBase = uint64_t(cfg.query) * HEADS * DIM;

        for (uint32_t h = 0; h < HEAD_TILE; ++h) {
            uint32_t head = cfg.head_begin + h;

            for (uint32_t d = 0; d < DIM; d += CHUNK) {
                uint64_t srcOffset = queryBase + d * HEADS + head * CHUNK;
                uint64_t dstOffset = h * DIM + d;

                DataCopy(qLocal[dstOffset], qGm[srcOffset], CHUNK);
            }
        }
        PipeBarrier<PIPE_ALL>();

        DataCopy(qLoadedGm, qLocal, Q_ELEMS);
        PipeBarrier<PIPE_ALL>();

        return true;
    }

    __aicore__ inline bool ApplyQueryRope() {
        // TODO 2：只旋转 qBuf 的最后 64 维，舍入回 BF16，再导出 qRopeGm。
        // position 决定 cos_sin 行；旋转配对为 (448+2*j,449+2*j)。
        auto qLocal = qBuf.Get<bfloat16_t>();

        constexpr uint32_t PAIRS = ROPE_DIM / 2;
        constexpr uint32_t ROPE_START = DIM - ROPE_DIM;
        uint64_t ropeBase = uint64_t(cfg.position) * ROPE_DIM;

        for (uint32_t j = 0; j < PAIRS; ++j) {
            float cos = ropeGm.GetValue(ropeBase + j);
            float sin = ropeGm.GetValue(ropeBase + j + PAIRS);

            for (uint32_t h = 0; h < HEAD_TILE; ++h) {
                uint32_t offset = h * DIM + ROPE_START + j * 2;

                float x = AscendC::ToFloat(qLocal.GetValue(offset + 0));
                float y = AscendC::ToFloat(qLocal.GetValue(offset + 1));

                float rotatedX = x * cos - y * sin;
                float rotatedY = x * sin + y * cos;

                qLocal.SetValue(offset + 0, AscendC::Cast(rotatedX));
                qLocal.SetValue(offset + 1, AscendC::Cast(rotatedY));
            }
        }

        auto event = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::S_MTE3)
        );
        SetFlag<HardEvent::S_MTE3>(event);
        WaitFlag<HardEvent::S_MTE3>(event);

        DataCopy(qRopeGm, qLocal, Q_ELEMS);
        PipeBarrier<PIPE_ALL>();

        return true;
    }

    __aicore__ inline bool LoadKvTile() {
        // TODO 3：顺序拼接 topk 个 FP8 位置和 extra_topk 个 FP4 位置。
        // 拼接分界按容量 topk，不是 length；length 只决定位置是否有效。
        // 索引已是物理槽号：page=id/page_size，token=id%page_size，不再查页表。
        // rawBuf 每行 544 B：数据放 [0,512)，scale 放 [512,544)；FP4 数据只有
        // 256 B。 无效索引不得发起越界搬运；记录 validBuf 并导出 validGm。
        auto rawLocal = rawBuf.Get<uint8_t>();
        auto validLocal = validBuf.Get<int32_t>();

        for (int32_t t = 0; t < KV_TILE; ++t) {
            validLocal.SetValue(t, 0);

            int32_t pos = cfg.kv_begin + t;
            bool isExtra = pos >= cfg.topk;

            int32_t index = isExtra ? pos - cfg.topk : pos;
            int32_t capacity = isExtra ? cfg.extra_topk : cfg.topk;
            int32_t length = isExtra ? cfg.extra_length : cfg.length;
            if (index >= capacity || index >= length)
                continue;

            auto indicesGm = isExtra ? extraIdsGm : idsGm;
            int32_t indicesStride =
                isExtra ? cfg.extra_indices_stride : cfg.indices_stride;

            uint64_t indexOffset = uint64_t(cfg.query) * indicesStride + index;
            int32_t id = indicesGm.GetValue(indexOffset);

            int32_t pageSize = isExtra ? cfg.extra_page_size : cfg.page_size;
            int32_t pages = isExtra ? cfg.extra_pages : cfg.pages;
            int32_t pageStride =
                isExtra ? cfg.extra_page_stride : cfg.page_stride;

            auto cache = isExtra ? kv4Gm : kv8Gm;

            uint32_t dataBytes = isExtra ? DIM / 2 : DIM;
            // fp8 每 32 维一个 scale，fp4 每 16 维一个，scale 都是 1B 的
            uint32_t scaleBytes = isExtra ? DIM / 16 : DIM / 32;

            int32_t page = id / pageSize;
            int32_t token = id % pageSize;

            uint64_t pageBase = uint64_t(page) * pageStride;
            uint64_t dataOffset = pageBase + uint64_t(token) * dataBytes;
            uint64_t scaleOffset = pageBase + uint64_t(pageSize) * dataBytes +
                                   uint64_t(token) * scaleBytes;

            uint32_t rowBase = t * RAW_ROW_BYTES;

            DataCopyExtParams dataCopy{1, dataBytes, 0, 0, 0};
            DataCopyExtParams scaleCopy{1, scaleBytes, 0, 0, 0};
            // 不使用额外 pad 方法，后面由参数自行算出合法 scale 范围就行
            DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};

            DataCopyPad(rawLocal[rowBase], cache[dataOffset], dataCopy, pad);
            DataCopyPad(
                rawLocal[rowBase + DIM], cache[scaleOffset], scaleCopy, pad
            );

            validLocal.SetValue(t, 1);
        }

        auto event = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::S_MTE3)
        );
        SetFlag<HardEvent::S_MTE3>(event);
        WaitFlag<HardEvent::S_MTE3>(event);

        DataCopy(validGm, validLocal, KV_TILE);
        PipeBarrier<PIPE_ALL>();

        return true;
    }

    __aicore__ inline bool DecodeKvTile() {
        // TODO 4：rawBuf → kvBuf 的 BF16 [64,512]，无效行填 0，导出 kvGm。
        // FP8：E4M3 数据，32 个值共享一个 UE8M0 scale。
        // FP4：E2M1 两值/字节（低半字节在前），16 个值共享一个 E4M3 scale。
        auto rawLocal = rawBuf.Get<uint8_t>();
        auto validLocal = validBuf.Get<int32_t>();
        auto kvLocal = kvBuf.Get<bfloat16_t>();

        auto loadEvent = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE2_S)
        );
        SetFlag<HardEvent::MTE2_S>(loadEvent);
        WaitFlag<HardEvent::MTE2_S>(loadEvent);

        for (uint32_t t = 0; t < KV_TILE; ++t) {
            uint32_t rawBase = t * RAW_ROW_BYTES;
            uint32_t kvBase = t * DIM;

            if (validLocal.GetValue(t) == 0) {
                for (int32_t d = 0; d < DIM; ++d) {
                    kvLocal.SetValue(kvBase + d, AscendC::Cast(0.0f));
                }
                continue;
            }

            bool isFp4 = cfg.kv_begin + t >= cfg.topk;
            int32_t groupSize = isFp4 ? 16 : 32;

            for (int32_t group = 0; group < DIM / groupSize; ++group) {
                uint8_t sCode = rawLocal.GetValue(rawBase + DIM + group);
                float scale = isFp4 ? DecodeE4M3(sCode) : DecodeE8M0(sCode);

                for (int32_t j = 0; j < groupSize; ++j) {
                    int32_t d = group * groupSize + j;
                    float value;

                    if (isFp4) {
                        // fp4 时 512 维度总共占 256 字节
                        // 一个字节放两维，奇数维放高位，偶数维放低位
                        uint8_t packed = rawLocal.GetValue(rawBase + d / 2);
                        uint8_t code = (packed >> ((d % 2) * 4)) & 0xf;
                        value = DecodeE2M1(code);
                    } else {
                        uint8_t code = rawLocal.GetValue(rawBase + d);
                        value = DecodeE4M3(code);
                    }

                    kvLocal.SetValue(kvBase + d, AscendC::Cast(scale * value));
                }
            }
        }

        auto storeEvent = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::S_MTE3)
        );
        SetFlag<HardEvent::S_MTE3>(storeEvent);
        WaitFlag<HardEvent::S_MTE3>(storeEvent);

        DataCopy(kvGm, kvLocal, KV_ELEMS);
        PipeBarrier<PIPE_ALL>();

        return true;
    }

    __aicore__ inline bool ComputeQk() {
        // TODO 5：用已注册的 mm 计算 qRopeGm * transpose(kvGm) → scoreGm。
        // 形状是 [16,512] * [512,64]，B 的物理存储仍为 [64,512]。
        // Vector 写 GM 必须完成后再发起 Matmul；读取结果前等待 Matmul 完成。
        auto event = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE3_S)
        );
        SetFlag<HardEvent::MTE3_S>(event);
        WaitFlag<HardEvent::MTE3_S>(event);

        mm.SetTensorA(qRopeGm);
        mm.SetTensorB(kvGm, true);

        mm.IterateAll<true>(scoreGm);
        mm.End();

        return true;
    }

    __aicore__ inline bool MaskScores() {
        // TODO 6：scoreGm → scoreBuf，乘 SCORE_SCALE，无效列写 -inf，再导出。
        // 整列屏蔽：同一个 KV 的 valid 标记供 16 个 head 共用。
        auto scoreLocal = scoreBuf.Get<float>();
        auto validLocal = validBuf.Get<uint32_t>();

        DataCopy(scoreLocal, scoreGm, SCORE_ELEMS);

        auto loadEvent = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE2_S)
        );
        SetFlag<HardEvent::MTE2_S>(loadEvent);
        WaitFlag<HardEvent::MTE2_S>(loadEvent);

        float negInf =
            AscendC::GetScalarBitcodeValue<uint32_t, float>(0xff800000u);

        for (int32_t t = 0; t < KV_TILE; ++t) {
            bool valid = validLocal.GetValue(t);

            for (int32_t h = 0; h < HEAD_TILE; ++h) {
                uint32_t offset = h * KV_TILE + t;
                float value = negInf;

                if (valid)
                    value = scoreLocal.GetValue(offset) * SCORE_SCALE;

                scoreLocal.SetValue(offset, value);
            }
        }

        auto storeEvent = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::S_MTE3)
        );
        SetFlag<HardEvent::S_MTE3>(storeEvent);
        WaitFlag<HardEvent::S_MTE3>(storeEvent);

        DataCopy(scoreGm, scoreLocal, SCORE_ELEMS);
        PipeBarrier<PIPE_ALL>();

        return true;
    }

    __aicore__ inline float DecodeE4M3(uint8_t code) {
        uint32_t exponent = (code >> 3) & 0xf;
        uint32_t mantissa = code & 0x7;

        float value;
        if (exponent == 0) {
            // AI Core 不支持 uint32_t 直接转 float；尾数仅 0~7，可先转
            // int32_t。
            value = float(int32_t(mantissa)) / 512.0f;
        } else {
            // 转成 fp32，指数偏置由 7 变为 127，尾数由 3 位扩到 23 位
            uint32_t bits = ((exponent + 120) << 23) | (mantissa << 20);
            if (exponent == 15 && mantissa == 7)
                bits = 0x7fc00000u;

            value = AscendC::GetScalarBitcodeValue<uint32_t, float>(bits);
        }

        return (code & 0x80) ? -value : value;
    }

    __aicore__ inline float DecodeE2M1(uint8_t code) {
        constexpr float magnitudes[8] = {0.0f, 0.5f, 1.0f, 1.5f,
                                         2.0f, 3.0f, 4.0f, 6.0f};

        float value = magnitudes[(code & 0x7)];
        return (code & 0x8) ? -value : value;
    }

    // E8M0 没有符号和尾数，表示 2^(s - 127)
    __aicore__ inline float DecodeE8M0(uint8_t code) {
        uint32_t bits = uint32_t(code) << 23;

        if (code == 0)
            bits = 0x00400000u;
        else if (code == 255)
            bits = 0x7fc00000u; // NaN

        return AscendC::GetScalarBitcodeValue<uint32_t, float>(bits);
    }

    __aicore__ inline void Report(Step step) {
        auto local = statusBuf.Get<int32_t>();
        for (int i = 0; i < 8; ++i)
            local.SetValue(i, static_cast<int32_t>(step));
        auto event = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::S_MTE3)
        );
        SetFlag<HardEvent::S_MTE3>(event);
        WaitFlag<HardEvent::S_MTE3>(event);
        DataCopy(statusGm, local, 8);
    }

    template <HardEvent eventType> __aicore__ inline void SyncPipeline() {
        auto event =
            static_cast<event_t>(GetTPipePtr()->FetchEventID(eventType));
        SetFlag<eventType>(event);
        WaitFlag<eventType>(event);
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
    GM_ADDR rope, GM_ADDR config, GM_ADDR q_loaded, GM_ADDR q_rope, GM_ADDR kv,
    GM_ADDR valid, GM_ADDR scores, GM_ADDR status, GM_ADDR cube_tiling,
    GM_ADDR workspace
) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    TPipe pipe;
    KernelMegaStage1 op;
    TCubeTiling tiling;
    auto src = reinterpret_cast<__gm__ uint32_t *>(cube_tiling);
    auto dst = reinterpret_cast<uint32_t *>(&tiling);
    for (int i = 0; i < sizeof(TCubeTiling) / sizeof(uint32_t); ++i)
        dst[i] = src[i];
    // HAVE_WORKSPACE 已初始化系统通信区；入参 workspace 已偏移到用户区。
    // Matmul 必须取系统区地址，AIC 在注册宏内服务，后续逻辑由 AIV 执行。
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), op.mm, &tiling);
    op.Init(
        q, kv8, kv4, ids, extra_ids, rope, config, q_loaded, q_rope, kv, valid,
        scores, status, &pipe
    );
    op.Process();
}

extern "C" void launch_mega_attn_stage1(
    const Stage1Input &in, const Stage1Output &out, void *config,
    void *cube_tiling, void *workspace, void *stream
) {
    mega_attn_stage1<<<1, nullptr, stream>>>(
        (GM_ADDR)in.q, (GM_ADDR)in.kv_fp8, (GM_ADDR)in.kv_fp4,
        (GM_ADDR)in.indices, (GM_ADDR)in.extra_indices, (GM_ADDR)in.cos_sin,
        (GM_ADDR)config, (GM_ADDR)out.q_loaded, (GM_ADDR)out.q_rope,
        (GM_ADDR)out.kv, (GM_ADDR)out.valid, (GM_ADDR)out.scores,
        (GM_ADDR)out.status, (GM_ADDR)cube_tiling, (GM_ADDR)workspace
    );
}

// 阶段 2 复用阶段 1 的搬运/QK，只增加跨 KV tile 的 softmax 与 PV 累加。
class KernelMegaStage2 : public KernelMegaStage1 {
  public:
    using VType = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;
    Matmul<QType, VType, ScoreType> pvMm;

    __aicore__ inline void InitOnline(
        GM_ADDR p, GM_ADDR pv, GM_ADDR output, GM_ADDR row_max, GM_ADDR row_sum,
        TPipe *pipe
    ) {
        pGm.SetGlobalBuffer((__gm__ bfloat16_t *)p, SCORE_ELEMS);
        pvGm.SetGlobalBuffer((__gm__ float *)pv, Q_ELEMS);
        outputGm.SetGlobalBuffer((__gm__ float *)output, Q_ELEMS);
        maxGm.SetGlobalBuffer((__gm__ float *)row_max, HEAD_TILE);
        sumGm.SetGlobalBuffer((__gm__ float *)row_sum, HEAD_TILE);

        pipe->InitBuffer(pBuf, SCORE_ELEMS * sizeof(bfloat16_t));
        pipe->InitBuffer(accBuf, Q_ELEMS * sizeof(float));
        pipe->InitBuffer(maxBuf, HEAD_TILE * sizeof(float));
        pipe->InitBuffer(sumBuf, HEAD_TILE * sizeof(float));
        pipe->InitBuffer(alphaBuf, HEAD_TILE * sizeof(float));
    }

    __aicore__ inline void ProcessOnline() {
        if (!LoadQuery()) {
            Report(Step::LoadQuery);
            return;
        }
        if (!ApplyQueryRope()) {
            Report(Step::ApplyQueryRope);
            return;
        }
        if (!InitOnlineState()) {
            Report(Step::InitOnlineState);
            return;
        }
        for (cfg.kv_begin = 0; cfg.kv_begin < cfg.topk + cfg.extra_topk;
             cfg.kv_begin += KV_TILE) {
            if (!LoadKvTile()) {
                Report(Step::LoadKvTile);
                return;
            }
            if (!DecodeKvTile()) {
                Report(Step::DecodeKvTile);
                return;
            }
            if (!ComputeQk()) {
                Report(Step::ComputeQk);
                return;
            }
            if (!MaskScores()) {
                Report(Step::MaskScores);
                return;
            }
            if (!OnlineSoftmax()) {
                ExportOnlineState();
                Report(Step::OnlineSoftmax);
                return;
            }
            if (!ComputePv()) {
                Report(Step::ComputePv);
                return;
            }
            if (!AccumulateOutput()) {
                Report(Step::AccumulateOutput);
                return;
            }
        }
        if (!NormalizeOutput()) {
            Report(Step::NormalizeOutput);
            return;
        }
        ExportOnlineState();
        Report(Step::Done);
    }

  private:
    __aicore__ inline bool InitOnlineState() {
        // TODO 7：每个 head 的 m=-inf、l=0、alpha=1；accBuf 的 [16,512] 全部置
        // 0。
        auto maxLocal = maxBuf.Get<float>();
        auto sumLocal = sumBuf.Get<float>();
        auto alphaLocal = alphaBuf.Get<float>();
        auto accLocal = accBuf.Get<float>();

        float negInf =
            AscendC::GetScalarBitcodeValue<uint32_t, float>(0xff800000u);

        Duplicate(maxLocal, negInf, HEAD_TILE);
        Duplicate(sumLocal, 0.0f, HEAD_TILE);
        Duplicate(alphaLocal, 1.0f, HEAD_TILE);
        Duplicate(accLocal, 0.0f, Q_ELEMS);

        // Duplicate 走的 Vector 通道
        SyncPipeline<HardEvent::V_S>();

        return true;
    }

    __aicore__ inline bool OnlineSoftmax() {
        // TODO 8：m'=max(m,max(score))，alpha=exp(m-m')，P=exp(score-m')。
        // l'=alpha*l+sum(P)；分母用 FP32 的 P，送 Cube 的 P 舍入为 BF16。
        // 全空前缀保持 m=-inf、l=0、P=0、alpha=1，避免 -inf-(-inf)。
        // 输入在 scoreBuf；更新 maxBuf/sumBuf/alphaBuf，把 pBuf 导出到 pGm。
        auto scoreLocal = scoreBuf.Get<float>();
        auto pLocal = pBuf.Get<bfloat16_t>();
        auto maxLocal = maxBuf.Get<float>();
        auto sumLocal = sumBuf.Get<float>();
        auto alphaLocal = alphaBuf.Get<float>();

        float negInf =
            AscendC::GetScalarBitcodeValue<uint32_t, float>(0xff800000u);

        for (int32_t h = 0; h < HEAD_TILE; ++h) {
            int32_t headBase = h * KV_TILE;
            float oldMax = maxLocal.GetValue(h);
            float newMax = oldMax;

            for (int32_t t = 0; t < KV_TILE; ++t) {
                float score = scoreLocal.GetValue(headBase + t);
                if (score > newMax) {
                    newMax = score;
                }
            }

            // 避免 -inf - (-inf)
            bool empty = newMax == negInf;
            alphaLocal.SetValue(h, empty ? 0.0f : oldMax - newMax);

            // 减最大值
            for (int32_t t = 0; t < KV_TILE; ++t) {
                float shifted =
                    empty ? negInf : scoreLocal.GetValue(headBase + t) - newMax;
                scoreLocal.SetValue(headBase + t, shifted);
            }
            maxLocal.SetValue(h, newMax);
        }

        SyncPipeline<HardEvent::S_V>();
        Exp(scoreLocal, scoreLocal, SCORE_ELEMS);
        Exp(alphaLocal, alphaLocal, HEAD_TILE);
        SyncPipeline<HardEvent::V_S>();

        // 更新 sum
        for (int32_t h = 0; h < HEAD_TILE; ++h) {
            int32_t headBase = h * KV_TILE;
            float tileSum = 0.0f;

            for (int32_t t = 0; t < KV_TILE; ++t) {
                float p = scoreLocal.GetValue(headBase + t);
                tileSum += p;
                pLocal.SetValue(headBase + t, AscendC::Cast(p));
            }

            float oldSum = sumLocal.GetValue(h);
            float alpha = alphaLocal.GetValue(h);
            sumLocal.SetValue(h, alpha * oldSum + tileSum);
        }

        SyncPipeline<HardEvent::S_MTE3>();
        DataCopy(pGm, pLocal, SCORE_ELEMS);
        PipeBarrier<PIPE_ALL>();
        return true;
    }

    __aicore__ inline bool ComputePv() {
        // TODO 9：pvMm 计算 [16,64] * [64,512] -> pvGm[16,512]，V 不转置。
        // 当前 MLA 中 K 和 V 共用 kvGm；发送请求前等 pGm 搬出完成。
        SyncPipeline<HardEvent::MTE3_S>();

        pvMm.SetTensorA(pGm);
        pvMm.SetTensorB(kvGm, false);

        pvMm.IterateAll<true>(pvGm);
        pvMm.End();

        return true;
    }

    __aicore__ inline bool AccumulateOutput() {
        // TODO 10：acc[h,d] = alpha[h]*acc[h,d] + pv[h,d]，全部 FP32。
        // rawBuf 在本轮解码后已不用，可 Get<float>() 暂存完整 PV（32768 B）。
        auto pvLocal = rawBuf.Get<float>();
        auto accLocal = accBuf.Get<float>();
        auto alphaLocal = alphaBuf.Get<float>();

        DataCopy(pvLocal, pvGm, Q_ELEMS);
        SyncPipeline<HardEvent::MTE2_S>();

        for (int32_t h = 0; h < HEAD_TILE; ++h) {
            int32_t headBase = h * DIM;
            float alpha = alphaLocal.GetValue(h);

            for (int32_t d = 0; d < DIM; ++d) {
                int32_t offset = headBase + d;
                float oldAcc = accLocal.GetValue(offset);
                float tilePV = pvLocal.GetValue(offset);

                accLocal.SetValue(offset, alpha * oldAcc + tilePV);
            }
        }

        SyncPipeline<HardEvent::S_MTE2>();
        return true;
    }

    __aicore__ inline bool NormalizeOutput() {
        // TODO 11：acc[h,d] /= sum[h]；sum[h]==0 的 head 输出 0。
        // 本阶段输出 FP32 attention；sink、逆 RoPE、输出量化留到阶段 3。
        auto accLocal = accBuf.Get<float>();
        auto sumLocal = sumBuf.Get<float>();

        for (int32_t h = 0; h < HEAD_TILE; ++h) {
            int32_t headBase = h * DIM;
            float sum = sumLocal.GetValue(h);
            bool empty = sum == 0.0f;

            float invSum = empty ? 0.0f : 1.0f / sum;

            for (int32_t d = 0; d < DIM; ++d) {
                int32_t offset = headBase + d;
                float value = empty ? 0.0f : accLocal.GetValue(offset) * invSum;

                accLocal.SetValue(offset, value);
            }
        }

        return true;
    }

    __aicore__ inline void ExportOnlineState() {
        auto event = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::S_MTE3)
        );
        SetFlag<HardEvent::S_MTE3>(event);
        WaitFlag<HardEvent::S_MTE3>(event);
        DataCopy(outputGm, accBuf.Get<float>(), Q_ELEMS);
        DataCopy(maxGm, maxBuf.Get<float>(), HEAD_TILE);
        DataCopy(sumGm, sumBuf.Get<float>(), HEAD_TILE);
        PipeBarrier<PIPE_ALL>();
    }

    GlobalTensor<bfloat16_t> pGm;
    GlobalTensor<float> pvGm, outputGm, maxGm, sumGm;
    TBuf<TPosition::VECCALC> pBuf, accBuf, maxBuf, sumBuf, alphaBuf;
};

extern "C" __global__ __aicore__ void mega_attn_stage2(
    GM_ADDR q, GM_ADDR kv8, GM_ADDR kv4, GM_ADDR ids, GM_ADDR extra_ids,
    GM_ADDR rope, GM_ADDR config, GM_ADDR q_loaded, GM_ADDR q_rope, GM_ADDR kv,
    GM_ADDR valid, GM_ADDR scores, GM_ADDR status, GM_ADDR p, GM_ADDR pv,
    GM_ADDR output, GM_ADDR row_max, GM_ADDR row_sum, GM_ADDR qk_tiling,
    GM_ADDR pv_tiling, GM_ADDR workspace
) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    TPipe pipe;
    KernelMegaStage2 op;
    TCubeTiling qkTiling, pvTiling;
    auto qkSrc = reinterpret_cast<__gm__ uint32_t *>(qk_tiling);
    auto pvSrc = reinterpret_cast<__gm__ uint32_t *>(pv_tiling);
    auto qkDst = reinterpret_cast<uint32_t *>(&qkTiling);
    auto pvDst = reinterpret_cast<uint32_t *>(&pvTiling);
    for (int i = 0; i < sizeof(TCubeTiling) / sizeof(uint32_t); ++i) {
        qkDst[i] = qkSrc[i];
        pvDst[i] = pvSrc[i];
    }
    REGIST_MATMUL_OBJ(
        &pipe, GetSysWorkSpacePtr(), op.mm, &qkTiling, op.pvMm, &pvTiling
    );
    op.Init(
        q, kv8, kv4, ids, extra_ids, rope, config, q_loaded, q_rope, kv, valid,
        scores, status, &pipe
    );
    op.InitOnline(p, pv, output, row_max, row_sum, &pipe);
    op.ProcessOnline();
}

extern "C" void launch_mega_attn_stage2(
    const Stage1Input &in, const Stage1Output &scratch, const Stage2Output &out,
    void *config, void *qk_tiling, void *pv_tiling, void *workspace,
    void *stream
) {
    mega_attn_stage2<<<1, nullptr, stream>>>(
        (GM_ADDR)in.q, (GM_ADDR)in.kv_fp8, (GM_ADDR)in.kv_fp4,
        (GM_ADDR)in.indices, (GM_ADDR)in.extra_indices, (GM_ADDR)in.cos_sin,
        (GM_ADDR)config, (GM_ADDR)scratch.q_loaded, (GM_ADDR)scratch.q_rope,
        (GM_ADDR)scratch.kv, (GM_ADDR)scratch.valid, (GM_ADDR)scratch.scores,
        (GM_ADDR)scratch.status, (GM_ADDR)out.p, (GM_ADDR)out.pv,
        (GM_ADDR)out.output, (GM_ADDR)out.row_max, (GM_ADDR)out.row_sum,
        (GM_ADDR)qk_tiling, (GM_ADDR)pv_tiling, (GM_ADDR)workspace
    );
}
