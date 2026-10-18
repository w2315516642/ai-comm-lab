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

// Q/KV 加载、解码和 QK；各步骤保留独立入口，便于查看中间结果。
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
    __aicore__ inline bool LoadQuery(bool exportLoaded = true) {
        // 把 Q 的 head_begin..head_begin+15 重排进 qBuf，并导出
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

        if (exportLoaded) {
            DataCopy(qLoadedGm, qLocal, Q_ELEMS);
            PipeBarrier<PIPE_ALL>();
        }

        return true;
    }

    __aicore__ inline bool ApplyQueryRope() {
        // 只旋转 qBuf 的最后 64 维，舍入回 BF16，再导出 qRopeGm。
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
        // 顺序拼接 topk 个 FP8 位置和 extra_topk 个 FP4 位置。
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
        // rawBuf → kvBuf 的 BF16 [64,512]，无效行填 0，导出 kvGm。
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
        // 用已注册的 mm 计算 qRopeGm * transpose(kvGm) → scoreGm。
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
        // scoreGm → scoreBuf，乘 SCORE_SCALE，无效列写 -inf，再导出。
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

    __aicore__ inline bool ProcessOnline() {
        if (!LoadQuery()) {
            Report(Step::LoadQuery);
            return false;
        }
        if (!ApplyQueryRope()) {
            Report(Step::ApplyQueryRope);
            return false;
        }
        if (!InitOnlineState()) {
            Report(Step::InitOnlineState);
            return false;
        }
        for (cfg.kv_begin = 0; cfg.kv_begin < cfg.topk + cfg.extra_topk;
             cfg.kv_begin += KV_TILE) {
            if (!LoadKvTile()) {
                Report(Step::LoadKvTile);
                return false;
            }
            if (!DecodeKvTile()) {
                Report(Step::DecodeKvTile);
                return false;
            }
            if (!ComputeQk()) {
                Report(Step::ComputeQk);
                return false;
            }
            if (!MaskScores()) {
                Report(Step::MaskScores);
                return false;
            }
            if (!OnlineSoftmax()) {
                ExportOnlineState();
                Report(Step::OnlineSoftmax);
                return false;
            }
            if (!ComputePv()) {
                Report(Step::ComputePv);
                return false;
            }
            if (!AccumulateOutput()) {
                Report(Step::AccumulateOutput);
                return false;
            }
        }
        if (!NormalizeOutput()) {
            Report(Step::NormalizeOutput);
            return false;
        }
        ExportOnlineState();
        Report(Step::Done);
        return true;
    }

  protected:
    __aicore__ inline bool InitOnlineState() {
        // 每个 head 的 m=-inf、l=0、alpha=1；accBuf 的 [16,512] 全部置
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
        // m'=max(m,max(score))，alpha=exp(m-m')，P=exp(score-m')。
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
        // pvMm 计算 [16,64] * [64,512] -> pvGm[16,512]，V 不转置。
        // 当前 MLA 中 K 和 V 共用 kvGm；发送请求前等 pGm 搬出完成。
        SyncPipeline<HardEvent::MTE3_S>();

        pvMm.SetTensorA(pGm);
        pvMm.SetTensorB(kvGm, false);

        pvMm.IterateAll<true>(pvGm);
        pvMm.End();

        return true;
    }

    __aicore__ inline bool AccumulateOutput() {
        // acc[h,d] = alpha[h]*acc[h,d] + pv[h,d]，全部 FP32。
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
        // acc[h,d] /= sum[h]；sum[h]==0 的 head 输出 0。
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

// 完整输出阶段复用 online 状态；不新增 UB，alpha/score/raw 在末尾可作临时缓冲。
class KernelMegaStage3 : public KernelMegaStage2 {
  public:
    __aicore__ inline void InitFinal(
        GM_ADDR sink, GM_ADDR fp8, GM_ADDR scales, GM_ADDR lse,
        int32_t sf_stride
    ) {
        hasSink = sink != nullptr;
        if (hasSink)
            sinkGm.SetGlobalBuffer((__gm__ float *)sink, HEADS);
        fp8Gm.SetGlobalBuffer((__gm__ uint8_t *)fp8);
        scalesGm.SetGlobalBuffer((__gm__ uint32_t *)scales);
        // 每次只导出当前 query 的 16 个 head，16*4=64 B 满足搬运对齐。
        lseGm.SetGlobalBuffer(
            (__gm__ float *)lse + cfg.query * HEADS + cfg.head_begin, HEAD_TILE
        );
        sfStride = sf_stride;
    }

    __aicore__ inline void ProcessFinal() {
        if (!ProcessOnline())
            return;
        if (!ApplySinkAndLse()) {
            Report(Step::ApplySinkAndLse);
            return;
        }
        if (!ApplyOutputRope()) {
            ExportOnlineState();
            Report(Step::ApplyOutputRope);
            return;
        }
        if (!QuantizeOutput()) {
            ExportOnlineState();
            Report(Step::QuantizeOutput);
            return;
        }
        ExportOnlineState();
        Report(Step::Done);
    }

  protected:
    __aicore__ inline bool ApplySinkAndLse(bool vectorScale = false) {
        // LSE=m+ln(l)，不含 sink；空行写 +inf，输出保持 0。
        // acc 已除过 l；再乘 1/(1+exp(sink-LSE))，避免直接 exp 大正数。
        // alphaBuf 可暂存 LSE，scoreBuf 可暂存 gate；最后导出到 lseGm。
        auto accLocal = accBuf.Get<float>();
        auto maxLocal = maxBuf.Get<float>();
        auto sumLocal = sumBuf.Get<float>();
        auto lseLocal = alphaBuf.Get<float>();
        auto gateLocal = scoreBuf.Get<float>();
        auto deltaLocal = gateLocal[HEAD_TILE];

        float negInf =
            AscendC::GetScalarBitcodeValue<uint32_t, float>(0xff800000u);
        float posInf =
            AscendC::GetScalarBitcodeValue<uint32_t, float>(0x7f800000u);

        for (int32_t h = 0; h < HEAD_TILE; ++h) {
            float sum = sumLocal.GetValue(h);
            lseLocal.SetValue(h, sum == 0.0f ? 1.0f : sum);
        }
        SyncPipeline<HardEvent::S_V>();
        Ln(lseLocal, lseLocal, HEAD_TILE);
        SyncPipeline<HardEvent::V_S>();

        for (int32_t h = 0; h < HEAD_TILE; ++h) {
            bool empty = sumLocal.GetValue(h) == 0.0f;
            float lse =
                empty ? posInf : maxLocal.GetValue(h) + lseLocal.GetValue(h);
            lseLocal.SetValue(h, lse);

            float sink = hasSink ? sinkGm.GetValue(cfg.head_begin + h) : negInf;
            float delta = empty ? 0.0f : sink - lse;
            deltaLocal.SetValue(h, delta);

            gateLocal.SetValue(h, delta > 0.0f ? -delta : delta);
        }
        SyncPipeline<HardEvent::S_V>();
        Exp(gateLocal, gateLocal, HEAD_TILE);
        SyncPipeline<HardEvent::V_S>();

        for (int32_t h = 0; h < HEAD_TILE; ++h) {
            float t = gateLocal.GetValue(h);
            float delta = deltaLocal.GetValue(h);

            float gate;
            if (sumLocal.GetValue(h) == 0.0f)
                gate = 0.0f;
            else
                gate = delta >= 0.0f ? t / (1.0f + t) : 1.0f / (1.0f + t);

            if (vectorScale) {
                if (gate == 0.0f) Duplicate(accLocal[h * DIM], 0.0f, DIM);
                else Muls(accLocal[h * DIM], accLocal[h * DIM], gate, DIM);
            } else {
                for (int32_t d = 0; d < DIM; ++d) {
                    int32_t offset = h * DIM + d;
                    float value = gate == 0.0f ? 0.0f : accLocal.GetValue(offset) * gate;
                    accLocal.SetValue(offset, value);
                }
            }
        }

        if (vectorScale) SyncPipeline<HardEvent::V_S>();
        SyncPipeline<HardEvent::S_MTE3>();
        DataCopy(lseGm, lseLocal, HEAD_TILE);
        PipeBarrier<PIPE_ALL>();
        return true;
    }

    __aicore__ inline bool ApplyOutputRope() {
        // acc[h,448+2*j:450+2*j] 乘 R(-theta)，保持 FP32。
        // cos/sin 沿用 cfg.position 的同一行，前 448 维不变。
        auto accLocal = accBuf.Get<float>();

        constexpr int32_t PAIRS = ROPE_DIM / 2;
        constexpr int32_t ROPE_START = DIM - ROPE_DIM;
        uint64_t ropeBase = uint64_t(cfg.position) * ROPE_DIM;

        for (int32_t j = 0; j < PAIRS; ++j) {
            float cos = ropeGm.GetValue(ropeBase + j);
            float sin = ropeGm.GetValue(ropeBase + j + PAIRS);

            for (int32_t h = 0; h < HEAD_TILE; ++h) {
                int32_t offset = h * DIM + ROPE_START + j * 2;

                float x = accLocal.GetValue(offset);
                float y = accLocal.GetValue(offset + 1);

                accLocal.SetValue(offset, x * cos + y * sin);
                accLocal.SetValue(offset + 1, -x * sin + y * cos);
            }
        }

        return true;
    }

    __aicore__ inline bool QuantizeOutput(bool fastEncode = false) {
        // 每 head 每 32 维一组，s=2^ceil(log2(max(amax,1e-4)/448))。
        // E4M3 最近偶数舍入，scale 写 UE8M0；rawBuf 可暂存 8192 B 输出及
        // scale。 fp8 为 [N,8,16,8,32]；scales 为 [256,sfStride]，只写本
        // query/head。
        auto accLocal = accBuf.Get<float>();
        auto packedLocal = rawBuf.Get<uint8_t>();
        auto scaleLocal = rawBuf.Get<uint32_t>()[Q_ELEMS / 4];

        constexpr int32_t GROUP_SIZE = 32;
        constexpr int32_t GROUPS = DIM / GROUP_SIZE;
        constexpr int32_t SCALE_WORDS = HEAD_TILE * GROUPS / 4;

        for (int32_t w = 0; w < SCALE_WORDS; ++w) {
            scaleLocal.SetValue(w * 8, 0u);
        }

        for (int32_t h = 0; h < HEAD_TILE; ++h) {
            for (int32_t g = 0; g < GROUPS; ++g) {
                int32_t src = h * DIM + g * GROUP_SIZE;

                float amax = 0.0f;
                for (int32_t j = 0; j < GROUP_SIZE; ++j) {
                    float x = accLocal.GetValue(src + j);
                    float a = x < 0.0f ? -x : x;
                    if (a > amax)
                        amax = a;
                }

                float invScale;
                uint8_t scale = MakeOutputScale(amax, invScale);

                int32_t word = (h / 8) * 32 + g * 2 + (h % 8) / 4;
                int32_t shift = (h % 4) * 8;
                uint32_t old = scaleLocal.GetValue(word * 8);
                scaleLocal.SetValue(word * 8, old | (uint32_t(scale) << shift));

                int32_t dst = (h / 8) * 4096 + g * 256 + (h % 8) * 32;
                for (int32_t j = 0; j < GROUP_SIZE; ++j) {
                    float x = accLocal.GetValue(src + j) * invScale;
                    uint8_t code = fastEncode ? EncodeE4M3Fast(x) : EncodeE4M3(x);
                    packedLocal.SetValue(dst + j, code);
                }
            }
        }

        SyncPipeline<HardEvent::S_MTE3>();

        uint64_t dataBase = uint64_t(cfg.query) * HEADS * DIM +
                            uint64_t(cfg.head_begin / 8) * 4096;
        DataCopy(fp8Gm[dataBase], packedLocal, Q_ELEMS);

        DataCopyExtParams copyWord{1, 4, 0, 0, 0};
        int32_t firstWord = (cfg.head_begin / 8) * 32;
        for (int32_t w = 0; w < SCALE_WORDS; ++w) {
            uint64_t dst = uint64_t(firstWord + w) * sfStride + cfg.query;
            DataCopyPad(scalesGm[dst], scaleLocal[w * 8], copyWord);
        }
        PipeBarrier<PIPE_ALL>();

        return true;
    }

    __aicore__ inline uint8_t EncodeE4M3(float x) {
        uint32_t bits = AscendC::GetScalarBitcodeValue<float, uint32_t>(x);
        uint8_t sign = uint8_t((bits >> 24) & 0x80);
        float a =
            AscendC::GetScalarBitcodeValue<uint32_t, float>(bits & 0x7fffffffu);

        if (a > 448.0f)
            a = 448.0f;

        int32_t lo = 0, hi = 126;
        while (lo < hi) {
            int32_t mid = (lo + hi) / 2;
            if (DecodeE4M3(uint8_t(mid)) < a)
                lo = mid + 1;
            else
                hi = mid;
        }

        int32_t upper = lo;
        int32_t lower = upper == 0 ? 0 : upper - 1;
        float left = a - DecodeE4M3(uint8_t(lower));
        float right = DecodeE4M3(upper) - a;

        // 距离相同时选择最低位为 0 的编码，即最近偶数舍入
        int32_t code = upper;
        if (left < right || (left == right && (lower & 1) == 0))
            code = lower;

        return uint8_t(code) | sign;
    }

    __aicore__ inline uint8_t MakeOutputScale(float amax, float &invScale) {
        if (amax < 1e-4f)
            amax = 1e-4f;
        float required = amax / 448.0f;
        uint32_t bits =
            AscendC::GetScalarBitcodeValue<float, uint32_t>(required);

        int32_t e = int32_t((bits - 1u) >> 23) - 126;
        uint32_t invBits = uint32_t(127 - e) << 23;
        invScale = AscendC::GetScalarBitcodeValue<uint32_t, float>(invBits);

        return uint8_t(e + 127);
    }

    // FP32 -> E4M3 的直接 RNE 编码；省去每个输出元素的二分查找。
    __aicore__ inline uint8_t EncodeE4M3Fast(float x) {
        uint32_t bits = AscendC::GetScalarBitcodeValue<float, uint32_t>(x);
        uint32_t sign = (bits >> 24) & 0x80u;
        uint32_t mag = bits & 0x7fffffffu;
        if (mag >= 0x43e00000u) return uint8_t(sign | 126u); // 448
        uint32_t exponent = mag >> 23;
        uint32_t code;
        if (exponent >= 121u) {
            // 23 位 FP32 尾数缩到 3 位：丢弃 20 位，半 ULP 时按保留位奇偶舍入。
            code = (mag - (120u << 23) + 0x7ffffu + ((mag >> 20) & 1u)) >> 20;
        } else if (exponent >= 117u) {
            // E4M3 非规格化数间隔 2^-9；低于半间隔 2^-10 的值舍入到零。
            uint32_t significand = (mag & 0x7fffffu) | 0x800000u;
            uint32_t shift = 141u - exponent;
            code = (significand + ((1u << (shift - 1)) - 1u) +
                    ((significand >> shift) & 1u)) >> shift;
        } else {
            code = 0;
        }
        return uint8_t(sign | code);
    }

    bool hasSink;
    int32_t sfStride;
    GlobalTensor<float> sinkGm, lseGm;
    GlobalTensor<uint8_t> fp8Gm;
    GlobalTensor<uint32_t> scalesGm;
};

extern "C" __global__ __aicore__ void mega_attn_stage3(
    GM_ADDR q, GM_ADDR kv8, GM_ADDR kv4, GM_ADDR ids, GM_ADDR extra_ids,
    GM_ADDR rope, GM_ADDR config, GM_ADDR q_loaded, GM_ADDR q_rope, GM_ADDR kv,
    GM_ADDR valid, GM_ADDR scores, GM_ADDR status, GM_ADDR p, GM_ADDR pv,
    GM_ADDR output, GM_ADDR row_max, GM_ADDR row_sum, GM_ADDR sink, GM_ADDR fp8,
    GM_ADDR scales, GM_ADDR lse, int32_t sf_stride, GM_ADDR qk_tiling,
    GM_ADDR pv_tiling, GM_ADDR workspace
) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    TPipe pipe;
    KernelMegaStage3 op;
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
    op.InitFinal(sink, fp8, scales, lse, sf_stride);
    op.ProcessFinal();
}

extern "C" void launch_mega_attn_stage3(
    const Stage1Input &in, const Stage1Output &scratch,
    const Stage2Output &online, const float *sink, const Stage3Output &out,
    void *config, void *qk_tiling, void *pv_tiling, void *workspace,
    void *stream
) {
    mega_attn_stage3<<<1, nullptr, stream>>>(
        (GM_ADDR)in.q, (GM_ADDR)in.kv_fp8, (GM_ADDR)in.kv_fp4,
        (GM_ADDR)in.indices, (GM_ADDR)in.extra_indices, (GM_ADDR)in.cos_sin,
        (GM_ADDR)config, (GM_ADDR)scratch.q_loaded, (GM_ADDR)scratch.q_rope,
        (GM_ADDR)scratch.kv, (GM_ADDR)scratch.valid, (GM_ADDR)scratch.scores,
        (GM_ADDR)scratch.status, (GM_ADDR)online.p, (GM_ADDR)online.pv,
        (GM_ADDR)online.output, (GM_ADDR)online.row_max,
        (GM_ADDR)online.row_sum, (GM_ADDR)sink, (GM_ADDR)out.fp8,
        (GM_ADDR)out.scales, (GM_ADDR)out.lse, out.sf_stride,
        (GM_ADDR)qk_tiling, (GM_ADDR)pv_tiling, (GM_ADDR)workspace
    );
}

// 每个 AIC/AIV 组复用一份 UB 和 GM 暂存，顺序完成分配给自己的完整 attention
// 任务。
class KernelMegaStage4 : public KernelMegaStage3 {
  public:
    __aicore__ inline void InitTasks(
        GM_ADDR configs, GM_ADDR status, uint32_t task_count, GM_ADDR sink,
        GM_ADDR fp8, GM_ADDR scales, GM_ADDR lse, int32_t sf_stride, int32_t mode
    ) {
        taskConfigsGm.SetGlobalBuffer((__gm__ int32_t *)configs);
        taskStatus = status;
        taskCount = task_count;
        sinkAddr = sink;
        fp8Addr = fp8;
        scalesAddr = scales;
        lseAddr = lse;
        finalSfStride = sf_stride;
        runMode = static_cast<Stage4Mode>(mode);
    }

    __aicore__ inline void RunTasks() {
        if (!ProcessTasks())
            Report(Step::ProcessTasks);
    }

  private:
    __aicore__ inline bool ProcessTasks() {
        // 当前核从 GetBlockIdx() 出发，每次增加 GetBlockNum()。
        // 每个 task 先 BindTask，再执行所选路径；同一核的 UB/GM 暂存顺序复用。
        uint32_t core = GetBlockIdx();
        uint32_t cores = GetBlockNum();

        for (uint32_t task = core; task < taskCount; task += cores) {
            if (!BindTask(task))
                return false;
            if (runMode == Stage4Mode::Reference) ProcessFinal();
            else ProcessOptimized(runMode == Stage4Mode::Pipeline);
        }
        return true;
    }

    __aicore__ inline bool BindTask(uint32_t task) {
        // 从 taskConfigsGm 的 task*16 处读取本任务的 16 个 int32
        // 配置。 statusGm 指向 taskStatus 的第 task*8 个 int32；调用 InitFinal
        // 更新 LSE 切片。
        uint64_t configBase = uint64_t(task) * 16;
        int32_t *fields = reinterpret_cast<int32_t *>(&cfg);
        for (int32_t i = 0; i < 16; ++i) {
            fields[i] = taskConfigsGm.GetValue(configBase + i);
        }

        statusGm.SetGlobalBuffer(
            (__gm__ int32_t *)taskStatus + uint64_t(task) * 8, 8
        );

        InitFinal(sinkAddr, fp8Addr, scalesAddr, lseAddr, finalSfStride);
        return true;
    }

    __aicore__ inline void ProcessOptimized(bool pipeline) {
        LoadQuery(false);
        ApplyQueryRope();
        InitOnlineState();
        int32_t total = cfg.topk + cfg.extra_topk;
        if (total > 0) PrefetchKv(0);
        for (int32_t begin = 0; begin < total; begin += KV_TILE) {
            cfg.kv_begin = begin;
            ConsumePrefetchedKv();
            DecodeKvTile();
            int32_t next = begin + KV_TILE;
            if (pipeline) {
                StartQkAsync();
                // Cube 读取当前 kvGm；MTE2 向已释放的 rawBuf 搬下一块。
                if (next < total) PrefetchKv(next);
                mm.WaitIterateAll();
                mm.End();
            } else {
                ComputeQk();
            }
            MaskScoresVector();
            OnlineSoftmaxVector();
            ComputePv();
            AccumulateOutputVector();
            if (!pipeline && next < total) PrefetchKv(next);
        }
        NormalizeOutputVector();
        ApplySinkAndLse(true);
        ApplyOutputRope();
        QuantizeOutput(true);
        // 优化路径只写最终 FP8/scale/LSE 和完成状态，不导出观测张量。
        Report(Step::Done);
    }

    __aicore__ inline void PrefetchKv(int32_t begin) {
        auto raw = rawBuf.Get<uint8_t>();
        // Q 已在 qRopeGm，qBuf 的前 256 B 存下一块 valid，不能覆盖当前 validBuf。
        auto nextValid = qBuf.Get<int32_t>();
        for (int32_t t = 0; t < KV_TILE; ++t) {
            nextValid.SetValue(t, 0);
            int32_t pos = begin + t;
            bool extra = pos >= cfg.topk;
            int32_t index = extra ? pos - cfg.topk : pos;
            int32_t capacity = extra ? cfg.extra_topk : cfg.topk;
            int32_t length = extra ? cfg.extra_length : cfg.length;
            if (index >= capacity || index >= length) continue;
            auto ids = extra ? extraIdsGm : idsGm;
            int32_t stride = extra ? cfg.extra_indices_stride : cfg.indices_stride;
            int32_t id = ids.GetValue(uint64_t(cfg.query) * stride + index);
            int32_t pageSize = extra ? cfg.extra_page_size : cfg.page_size;
            int32_t pageStride = extra ? cfg.extra_page_stride : cfg.page_stride;
            uint32_t dataBytes = extra ? DIM / 2 : DIM;
            uint32_t scaleBytes = extra ? DIM / 16 : DIM / 32;
            uint64_t base = uint64_t(id / pageSize) * pageStride;
            uint64_t dataOffset = base + uint64_t(id % pageSize) * dataBytes;
            uint64_t scaleOffset = base + uint64_t(pageSize) * dataBytes +
                                   uint64_t(id % pageSize) * scaleBytes;
            auto cache = extra ? kv4Gm : kv8Gm;
            DataCopyExtParams dataCopy{1, dataBytes, 0, 0, 0};
            DataCopyExtParams scaleCopy{1, scaleBytes, 0, 0, 0};
            DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};
            DataCopyPad(raw[t * RAW_ROW_BYTES], cache[dataOffset], dataCopy, pad);
            DataCopyPad(raw[t * RAW_ROW_BYTES + DIM], cache[scaleOffset], scaleCopy, pad);
            nextValid.SetValue(t, 1);
        }
        // 只提交搬运；在真正解码下一块前等待，避免提前阻塞 Cube。
    }

    __aicore__ inline void ConsumePrefetchedKv() {
        SyncPipeline<HardEvent::MTE2_S>();
        auto nextValid = qBuf.Get<int32_t>();
        auto valid = validBuf.Get<int32_t>();
        for (int32_t t = 0; t < KV_TILE; ++t)
            valid.SetValue(t, nextValid.GetValue(t));
    }

    __aicore__ inline void StartQkAsync() {
        SyncPipeline<HardEvent::MTE3_S>();
        mm.SetTensorA(qRopeGm);
        mm.SetTensorB(kvGm, true);
        // 第四个参数启用完成通知；false 保持普通 ND 输出布局。
        mm.IterateAll<false>(scoreGm, 0, false, true);
    }

    __aicore__ inline void MaskScoresVector() {
        auto scores = scoreBuf.Get<float>();
        // 前 64 个 int32 留给下一块 valid；mask 从第 256 B 开始，共 64 个 float。
        auto mask = qBuf.Get<float>()[64];
        auto valid = validBuf.Get<int32_t>();
        float negInf = AscendC::GetScalarBitcodeValue<uint32_t, float>(0xff800000u);
        for (int32_t t = 0; t < KV_TILE; ++t)
            mask.SetValue(t, valid.GetValue(t) ? 0.0f : negInf);
        DataCopy(scores, scoreGm, SCORE_ELEMS);
        SyncPipeline<HardEvent::MTE2_V>();
        SyncPipeline<HardEvent::S_V>();
        Muls(scores, scores, SCORE_SCALE, SCORE_ELEMS);
        PipeBarrier<PIPE_V>();
        for (int32_t h = 0; h < HEAD_TILE; ++h)
            Add(scores[h * KV_TILE], scores[h * KV_TILE], mask, KV_TILE);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void OnlineSoftmaxVector() {
        auto scores = scoreBuf.Get<float>();
        auto prob = pBuf.Get<bfloat16_t>();
        auto rowMax = maxBuf.Get<float>();
        auto rowSum = sumBuf.Get<float>();
        auto alpha = alphaBuf.Get<float>();
        // valid/mask 各占 256 B；max/sum 接着各占 16*4=64 B，互不覆盖。
        auto tileMax = qBuf.Get<float>()[128];
        auto tileSum = qBuf.Get<float>()[144];
        float negInf = AscendC::GetScalarBitcodeValue<uint32_t, float>(0xff800000u);
        // 每行 64 个 float 正好一个 repeat；行跨度 256 B，即 8 个 data block。
        WholeReduceMax(tileMax, scores, KV_TILE, HEAD_TILE, 1, 1, 8,
                       ReduceOrder::ORDER_ONLY_VALUE);
        SyncPipeline<HardEvent::V_S>();
        for (int32_t h = 0; h < HEAD_TILE; ++h) {
            float oldMax = rowMax.GetValue(h);
            float newMax = tileMax.GetValue(h);
            if (oldMax > newMax) newMax = oldMax;
            bool empty = newMax == negInf;
            alpha.SetValue(h, empty ? 0.0f : oldMax - newMax);
            rowMax.SetValue(h, newMax);
            if (empty) Duplicate(scores[h * KV_TILE], negInf, KV_TILE);
            else Adds(scores[h * KV_TILE], scores[h * KV_TILE], -newMax, KV_TILE);
        }
        SyncPipeline<HardEvent::S_V>();
        PipeBarrier<PIPE_V>();
        Exp(scores, scores, SCORE_ELEMS);
        Exp(alpha, alpha, HEAD_TILE);
        PipeBarrier<PIPE_V>();
        WholeReduceSum(tileSum, scores, KV_TILE, HEAD_TILE, 1, 1, 8);
        Cast(prob, scores, RoundMode::CAST_RINT, SCORE_ELEMS);
        SyncPipeline<HardEvent::V_S>();
        // m 改变后先把旧分母乘 exp(m_old-m_new)，再加当前块的 FP32 概率和。
        for (int32_t h = 0; h < HEAD_TILE; ++h)
            rowSum.SetValue(h, alpha.GetValue(h) * rowSum.GetValue(h) + tileSum.GetValue(h));
        SyncPipeline<HardEvent::V_MTE3>();
        DataCopy(pGm, prob, SCORE_ELEMS);
    }

    __aicore__ inline void AccumulateOutputVector() {
        // kvBuf 已写回 kvGm；Cube 只读 GM，因此可复用前 32 KiB 暂存 PV。
        // rawBuf 此时留给下一块预取，不能再作为 PV 暂存。
        auto pv = kvBuf.Get<float>();
        auto acc = accBuf.Get<float>();
        auto alpha = alphaBuf.Get<float>();
        DataCopy(pv, pvGm, Q_ELEMS);
        SyncPipeline<HardEvent::MTE2_V>();
        // O_new = exp(m_old-m_new)*O_old + P*V；每个 head 共用一个 alpha。
        for (int32_t h = 0; h < HEAD_TILE; ++h)
            Muls(acc[h * DIM], acc[h * DIM], alpha.GetValue(h), DIM);
        PipeBarrier<PIPE_V>();
        Add(acc, acc, pv, Q_ELEMS);
        // 下一轮解码会标量覆盖 kvBuf，要等 Vector 读完当前 PV。
        SyncPipeline<HardEvent::V_S>();
    }

    __aicore__ inline void NormalizeOutputVector() {
        auto acc = accBuf.Get<float>();
        auto sum = sumBuf.Get<float>();
        for (int32_t h = 0; h < HEAD_TILE; ++h) {
            float l = sum.GetValue(h);
            if (l == 0.0f) Duplicate(acc[h * DIM], 0.0f, DIM);
            else Muls(acc[h * DIM], acc[h * DIM], 1.0f / l, DIM);
        }
        SyncPipeline<HardEvent::V_S>();
    }

    Stage4Mode runMode;
    GlobalTensor<int32_t> taskConfigsGm;
    GM_ADDR taskStatus;
    GM_ADDR sinkAddr;
    GM_ADDR fp8Addr;
    GM_ADDR scalesAddr;
    GM_ADDR lseAddr;
    uint32_t taskCount;
    int32_t finalSfStride;
};

extern "C" __global__ __aicore__ void mega_attn_stage4(
    GM_ADDR q, GM_ADDR kv8, GM_ADDR kv4, GM_ADDR ids, GM_ADDR extra_ids,
    GM_ADDR rope, GM_ADDR configs, GM_ADDR q_loaded, GM_ADDR q_rope, GM_ADDR kv,
    GM_ADDR valid, GM_ADDR scores, GM_ADDR status, GM_ADDR p, GM_ADDR pv,
    GM_ADDR output, GM_ADDR row_max, GM_ADDR row_sum, GM_ADDR sink, GM_ADDR fp8,
    GM_ADDR scales, GM_ADDR lse, int32_t sf_stride, uint32_t task_count, int32_t mode,
    GM_ADDR qk_tiling, GM_ADDR pv_tiling, GM_ADDR workspace
) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    TPipe pipe;
    KernelMegaStage4 op;
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
    // GM_ADDR 的指针运算以字节为单位；每核只初始化一次缓冲。
    uint64_t core = GetBlockIdx();
    op.Init(
        q, kv8, kv4, ids, extra_ids, rope,
        configs + core * sizeof(Stage1Config), q_loaded + core * Q_ELEMS * 2,
        q_rope + core * Q_ELEMS * 2, kv + core * KV_ELEMS * 2,
        valid + core * KV_TILE * 4, scores + core * SCORE_ELEMS * 4,
        status + core * 8 * 4, &pipe
    );
    op.InitOnline(
        p + core * SCORE_ELEMS * 2, pv + core * Q_ELEMS * 4,
        output + core * Q_ELEMS * 4, row_max + core * HEAD_TILE * 4,
        row_sum + core * HEAD_TILE * 4, &pipe
    );
    op.InitTasks(
        configs, status, task_count, sink, fp8, scales, lse, sf_stride, mode
    );
    op.RunTasks();
}

extern "C" void launch_mega_attn_stage4(
    const Stage1Input &in, const Stage1Output &scratch,
    const Stage2Output &online, const float *sink, const Stage3Output &out,
    uint32_t task_count, uint32_t block_num, void *configs, void *qk_tiling,
    void *pv_tiling, void *workspace, void *stream, Stage4Mode mode
) {
    mega_attn_stage4<<<block_num, nullptr, stream>>>(
        (GM_ADDR)in.q, (GM_ADDR)in.kv_fp8, (GM_ADDR)in.kv_fp4,
        (GM_ADDR)in.indices, (GM_ADDR)in.extra_indices, (GM_ADDR)in.cos_sin,
        (GM_ADDR)configs, (GM_ADDR)scratch.q_loaded, (GM_ADDR)scratch.q_rope,
        (GM_ADDR)scratch.kv, (GM_ADDR)scratch.valid, (GM_ADDR)scratch.scores,
        (GM_ADDR)scratch.status, (GM_ADDR)online.p, (GM_ADDR)online.pv,
        (GM_ADDR)online.output, (GM_ADDR)online.row_max,
        (GM_ADDR)online.row_sum, (GM_ADDR)sink, (GM_ADDR)out.fp8,
        (GM_ADDR)out.scales, (GM_ADDR)out.lse, out.sf_stride, task_count, static_cast<int32_t>(mode),
        (GM_ADDR)qk_tiling, (GM_ADDR)pv_tiling, (GM_ADDR)workspace
    );
}
