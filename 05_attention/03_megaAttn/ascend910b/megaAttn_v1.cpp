#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "megaAttn_v1.h"

using namespace AscendC;
using namespace mega_ascend_v1;

// K 的物理布局是 [64,512]，QK 计算时转置；PV 使用同一份 KV，但不转置。
using QType = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;
using KType = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, true>;
using VType = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;
using AccType = MatmulType<TPosition::GM, CubeFormat::ND, float>;

// 每个核组复用自己的 UB/GM 暂存，依次完成分配到的 query/head 任务。
class KernelMegaAttention {
  public:
    Matmul<QType, KType, AccType> qkMm;
    Matmul<QType, VType, AccType> pvMm;

    __aicore__ inline void Init(
        GM_ADDR q, GM_ADDR kv8, GM_ADDR kv4, GM_ADDR ids, GM_ADDR extra_ids,
        GM_ADDR rope, GM_ADDR sink, GM_ADDR tasks, uint32_t task_count,
        GM_ADDR fp8, GM_ADDR scales, GM_ADDR lse, int32_t sf_stride,
        GM_ADDR scratch, TPipe *pipe
    ) {
        qGm.SetGlobalBuffer((__gm__ bfloat16_t *)q);
        kv8Gm.SetGlobalBuffer((__gm__ uint8_t *)kv8);
        kv4Gm.SetGlobalBuffer((__gm__ uint8_t *)kv4);
        idsGm.SetGlobalBuffer((__gm__ int32_t *)ids);
        extraIdsGm.SetGlobalBuffer((__gm__ int32_t *)extra_ids);
        ropeGm.SetGlobalBuffer((__gm__ float *)rope);
        taskConfigsGm.SetGlobalBuffer((__gm__ int32_t *)tasks);
        taskCount = task_count;
        hasSink = sink != nullptr;
        if (hasSink) sinkGm.SetGlobalBuffer((__gm__ float *)sink, HEADS);
        fp8Gm.SetGlobalBuffer((__gm__ uint8_t *)fp8);
        scalesGm.SetGlobalBuffer((__gm__ uint32_t *)scales);
        lseGm.SetGlobalBuffer((__gm__ float *)lse);
        sfStride = sf_stride;
        InitScratch(scratch, pipe);
    }

    __aicore__ inline void Process() {
        // task = core, core+核数, ...；不同核写不同的 query/head 切片。
        for (uint32_t task = GetBlockIdx(); task < taskCount; task += GetBlockNum()) {
            BindTask(task);
            ProcessQuery();
        }
    }

  private:
    __aicore__ inline void InitScratch(GM_ADDR scratch, TPipe *pipe) {
        GM_ADDR base = scratch + uint64_t(GetBlockIdx()) * ScratchLayout::BytesPerCore;
        qRopeGm.SetGlobalBuffer((__gm__ bfloat16_t *)(base + ScratchLayout::Q), Q_ELEMS);
        kvGm.SetGlobalBuffer((__gm__ bfloat16_t *)(base + ScratchLayout::KV), KV_ELEMS);
        scoreGm.SetGlobalBuffer((__gm__ float *)(base + ScratchLayout::Scores), SCORE_ELEMS);
        pGm.SetGlobalBuffer((__gm__ bfloat16_t *)(base + ScratchLayout::P), SCORE_ELEMS);
        pvGm.SetGlobalBuffer((__gm__ float *)(base + ScratchLayout::PV), Q_ELEMS);

        pipe->InitBuffer(qBuf, Q_ELEMS * sizeof(bfloat16_t));
        pipe->InitBuffer(rawBuf, KV_TILE * RAW_ROW_BYTES);
        pipe->InitBuffer(kvBuf, KV_ELEMS * sizeof(bfloat16_t));
        pipe->InitBuffer(validBuf, KV_TILE * sizeof(int32_t));
        pipe->InitBuffer(scoreBuf, SCORE_ELEMS * sizeof(float));
        pipe->InitBuffer(pBuf, SCORE_ELEMS * sizeof(bfloat16_t));
        pipe->InitBuffer(accBuf, Q_ELEMS * sizeof(float));
        pipe->InitBuffer(maxBuf, HEAD_TILE * sizeof(float));
        pipe->InitBuffer(sumBuf, HEAD_TILE * sizeof(float));
        pipe->InitBuffer(alphaBuf, HEAD_TILE * sizeof(float));
    }

    __aicore__ inline void BindTask(uint32_t task) {
        constexpr uint32_t FIELDS = sizeof(TaskConfig) / sizeof(int32_t);
        auto fields = reinterpret_cast<int32_t *>(&cfg);
        for (uint32_t i = 0; i < FIELDS; ++i)
            fields[i] = taskConfigsGm.GetValue(uint64_t(task) * FIELDS + i);
    }

    __aicore__ inline void ProcessQuery() {
        LoadQuery();
        ApplyQueryRope();
        InitOnlineState();
        int32_t total = cfg.topk + cfg.extra_topk;
        if (total > 0) PrefetchKv(0);
        for (kvBegin = 0; kvBegin < total; kvBegin += KV_TILE) {
            ConsumePrefetchedKv();
            DecodeKvTile();
            StartQkAsync();
            // Cube 读当前 kvGm 时，MTE2 向已释放的 rawBuf 搬下一块。
            if (kvBegin + KV_TILE < total) PrefetchKv(kvBegin + KV_TILE);
            qkMm.WaitIterateAll();
            qkMm.End();
            MaskScores();
            OnlineSoftmax();
            ComputePv();
            AccumulateOutput();
        }
        NormalizeOutput();
        ApplySinkAndLse();
        ApplyOutputRope();
        QuantizeOutput();
    }

    __aicore__ inline void LoadQuery() {
        // 把当前 16 个 head 重排成 [16,512]。原始索引：query*64*512 + (d/16)*64*16 + head*16 + d%16。
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
    }

    __aicore__ inline void ApplyQueryRope() {
        // 只旋转 qBuf 的最后 64 维，舍入回 BF16，写入 Matmul 的 Q 暂存。
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

    __aicore__ inline void DecodeKvTile() {
        // rawBuf 解码为 BF16 [64,512]，尾部无效行填 0，再写入 Matmul 的 KV 暂存。
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

            bool isFp4 = kvBegin + t >= cfg.topk;
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
    }

    __aicore__ inline void StartQkAsync() {
        SyncPipeline<HardEvent::MTE3_S>();
        qkMm.SetTensorA(qRopeGm);
        qkMm.SetTensorB(kvGm, true);
        // 第四个参数启用完成通知；false 保持普通 ND 输出布局。
        qkMm.IterateAll<false>(scoreGm, 0, false, true);
    }

    __aicore__ inline void MaskScores() {
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

    __aicore__ inline void InitOnlineState() {
        // 每个 head 初始化 m=-inf、l=0、alpha=1，输出累加器清零。
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
    }

    __aicore__ inline void OnlineSoftmax() {
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

    __aicore__ inline void ComputePv() {
        // pvMm 计算 [16,64] * [64,512] -> pvGm[16,512]，V 不转置。
        // 当前 MLA 中 K 和 V 共用 kvGm；发送请求前等 pGm 搬出完成。
        SyncPipeline<HardEvent::MTE3_S>();

        pvMm.SetTensorA(pGm);
        pvMm.SetTensorB(kvGm, false);

        pvMm.IterateAll<true>(pvGm);
        pvMm.End();
    }

    __aicore__ inline void AccumulateOutput() {
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

    __aicore__ inline void NormalizeOutput() {
        auto acc = accBuf.Get<float>();
        auto sum = sumBuf.Get<float>();
        for (int32_t h = 0; h < HEAD_TILE; ++h) {
            float l = sum.GetValue(h);
            if (l == 0.0f) Duplicate(acc[h * DIM], 0.0f, DIM);
            else Muls(acc[h * DIM], acc[h * DIM], 1.0f / l, DIM);
        }
        SyncPipeline<HardEvent::V_S>();
    }

    __aicore__ inline void ApplySinkAndLse() {
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

            if (gate == 0.0f) Duplicate(accLocal[h * DIM], 0.0f, DIM);
            else Muls(accLocal[h * DIM], accLocal[h * DIM], gate, DIM);
        }

        SyncPipeline<HardEvent::V_S>();
        SyncPipeline<HardEvent::S_MTE3>();
        DataCopy(lseGm[uint64_t(cfg.query) * HEADS + cfg.head_begin], lseLocal, HEAD_TILE);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ApplyOutputRope() {
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
    }

    __aicore__ inline void QuantizeOutput() {
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
                    uint8_t code = EncodeE4M3(x);
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

    __aicore__ inline float DecodeE8M0(uint8_t code) {
        uint32_t bits = uint32_t(code) << 23;

        if (code == 0)
            bits = 0x00400000u;
        else if (code == 255)
            bits = 0x7fc00000u; // NaN

        return AscendC::GetScalarBitcodeValue<uint32_t, float>(bits);
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

    // FP32 按最近偶数舍入编码为 E4M3，有限值饱和到 448。
    __aicore__ inline uint8_t EncodeE4M3(float x) {
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

    template <HardEvent eventType> __aicore__ inline void SyncPipeline() {
        auto event = static_cast<event_t>(GetTPipePtr()->FetchEventID(eventType));
        SetFlag<eventType>(event);
        WaitFlag<eventType>(event);
    }

    TaskConfig cfg;
    int32_t kvBegin = 0, sfStride;
    uint32_t taskCount;
    bool hasSink;
    GlobalTensor<bfloat16_t> qGm, qRopeGm, kvGm, pGm;
    GlobalTensor<uint8_t> kv8Gm, kv4Gm, fp8Gm;
    GlobalTensor<int32_t> idsGm, extraIdsGm, taskConfigsGm;
    GlobalTensor<uint32_t> scalesGm;
    GlobalTensor<float> ropeGm, scoreGm, pvGm, sinkGm, lseGm;
    TBuf<TPosition::VECCALC> qBuf, rawBuf, kvBuf, validBuf, scoreBuf;
    TBuf<TPosition::VECCALC> pBuf, accBuf, maxBuf, sumBuf, alphaBuf;
};

extern "C" __global__ __aicore__ void mega_attn_v1(
    GM_ADDR q, GM_ADDR kv8, GM_ADDR kv4, GM_ADDR ids, GM_ADDR extra_ids,
    GM_ADDR rope, GM_ADDR sink, GM_ADDR tasks, uint32_t task_count,
    GM_ADDR fp8, GM_ADDR scales, GM_ADDR lse, int32_t sf_stride,
    GM_ADDR scratch, GM_ADDR qk_tiling, GM_ADDR pv_tiling, GM_ADDR workspace
) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    TPipe pipe;
    KernelMegaAttention op;
    TCubeTiling qkTiling, pvTiling;
    auto qkSrc = reinterpret_cast<__gm__ uint32_t *>(qk_tiling);
    auto pvSrc = reinterpret_cast<__gm__ uint32_t *>(pv_tiling);
    auto qkDst = reinterpret_cast<uint32_t *>(&qkTiling);
    auto pvDst = reinterpret_cast<uint32_t *>(&pvTiling);
    for (int i = 0; i < sizeof(TCubeTiling) / sizeof(uint32_t); ++i) {
        qkDst[i] = qkSrc[i];
        pvDst[i] = pvSrc[i];
    }
    // AIC 在注册宏内服务 Matmul 请求；后续主循环由 AIV 驱动。
    // HAVE_WORKSPACE 初始化系统通信区，workspace 必须是设备入口最后一个参数。
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), op.qkMm, &qkTiling, op.pvMm, &pvTiling);
    op.Init(q, kv8, kv4, ids, extra_ids, rope, sink, tasks, task_count,
            fp8, scales, lse, sf_stride, scratch, &pipe);
    op.Process();
}

extern "C" void launch_mega_attn_v1(
    const Input &in, const Output &out, const TaskConfig *tasks,
    uint32_t task_count, uint32_t block_num, void *scratch, void *qk_tiling,
    void *pv_tiling, void *system_workspace, void *stream
) {
    if (task_count == 0) return;
    mega_attn_v1<<<block_num, nullptr, stream>>>(
        (uint8_t *)in.q, (uint8_t *)in.kv_fp8, (uint8_t *)in.kv_fp4,
        (uint8_t *)in.indices, (uint8_t *)in.extra_indices, (uint8_t *)in.cos_sin,
        (uint8_t *)in.sink, (uint8_t *)tasks, task_count,
        out.fp8, (uint8_t *)out.scales, (uint8_t *)out.lse, out.sf_stride,
        (uint8_t *)scratch, (uint8_t *)qk_tiling, (uint8_t *)pv_tiling,
        (uint8_t *)system_workspace);
}
