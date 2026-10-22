#include "kernel_operator.h"
#include "megaAttn_v6.h"
using namespace AscendC;
using namespace mega_ascend_v6;
constexpr int MERGE_HEADS = 8;
constexpr int MERGE_UB_BYTES = MERGE_HEADS * DIM * 4 + WORK_BYTES +
                               ARENA_BYTES + DECODE_ELEMS * 2 + 2048 +
                               (3 * DIM + 128) * 4 + 64 + 192 * 4;
static_assert(MERGE_UB_BYTES + SDK_UB_RESERVE <= 192 * 1024);

// 每个 AIV 合并一个 query 的 8 个 head；不会在量化后的输出上做归约。
class MergeAttentionV6 {
  public:
    __aicore__ inline void Init(
        GM_ADDR partials, GM_ADDR configs, GM_ADDR rope, GM_ADDR sink,
        GM_ADDR fp8, GM_ADDR scales, GM_ADDR lse, int32_t stride,
        GM_ADDR tables, TPipe *pipe
    ) {
        partialsGm.SetGlobalBuffer((__gm__ float *)partials);
        configsGm.SetGlobalBuffer((__gm__ int32_t *)configs);
        ropeGm.SetGlobalBuffer((__gm__ float *)rope);
        hasSink = sink != nullptr;
        if (hasSink)
            sinkGm.SetGlobalBuffer((__gm__ float *)sink);
        fp8Gm.SetGlobalBuffer((__gm__ uint8_t *)fp8);
        scalesGm.SetGlobalBuffer((__gm__ uint32_t *)scales);
        lseGm.SetGlobalBuffer((__gm__ float *)lse);
        scaleStride = stride;
        pipe->InitBuffer(accBuf, MERGE_HEADS * DIM * 4);
        pipe->InitBuffer(workBuf, WORK_BYTES);
        pipe->InitBuffer(arenaBuf, ARENA_BYTES);
        pipe->InitBuffer(rowBuf, DECODE_ELEMS * 2);
        pipe->InitBuffer(stateBuf, 2048);
        pipe->InitBuffer(indexBuf, (3 * DIM + 128) * 4);
        pipe->InitBuffer(maskBuf, 64);
        pipe->InitBuffer(ropeBuf, 192 * 4);
        GlobalTensor<uint32_t> table;
        table.SetGlobalBuffer((__gm__ uint32_t *)tables);
        DataCopy(indexBuf.Get<uint32_t>(), table[768], 3 * DIM + 128);
        DataCopy(maskBuf.Get<uint32_t>(), table[768 + INDEX_WORDS], 16);
        Sync<HardEvent::MTE2_V>();
    }
    __aicore__ inline void Process(uint32_t count) {
        for (uint32_t task = GetBlockIdx();
             task < count * (TOTAL_HEADS / MERGE_HEADS);
             task += GetBlockNum()) {
            uint32_t queryTask = task / (TOTAL_HEADS / MERGE_HEADS);
            headBase = (task % (TOTAL_HEADS / MERGE_HEADS)) * MERGE_HEADS;
            auto fields = reinterpret_cast<int32_t *>(&cfg);
            for (int i = 0; i < 12; ++i)
                fields[i] = configsGm.GetValue(uint64_t(queryTask) * 12 + i);
            auto m = stateBuf.Get<float>(), sum = m[64], alpha = m[128],
                 partM = m[192], partSum = m[256], next = m[320], beta = m[384];
            auto acc = accBuf.Get<float>(), pv = workBuf.Get<float>();
            auto a = workBuf.Get<float>()[2 * DECODE_ELEMS],
                 b = a[MERGE_HEADS * 8];
            // 第一片直接成为累积状态，只对后续分片做指数基准对齐。
            uint64_t first = uint64_t(queryTask) * KV_SPLITS * PARTIAL_FLOATS;
            DataCopy(m, partialsGm[first + headBase], MERGE_HEADS);
            DataCopy(
                sum, partialsGm[first + TOTAL_HEADS + headBase], MERGE_HEADS
            );
            DataCopy(
                acc, partialsGm[first + 2 * TOTAL_HEADS + headBase * DIM],
                MERGE_HEADS * DIM
            );
            Sync<HardEvent::MTE2_V>();
            for (int split = 1; split < KV_SPLITS; ++split) {
                uint64_t base =
                    (uint64_t(queryTask) * KV_SPLITS + split) * PARTIAL_FLOATS;
                DataCopy(partM, partialsGm[base + headBase], MERGE_HEADS);
                DataCopy(
                    partSum, partialsGm[base + TOTAL_HEADS + headBase],
                    MERGE_HEADS
                );
                Sync<HardEvent::MTE2_V>();
                // 统一到更大的指数基准：l 和 O 都乘 exp(旧基准-新基准) 再相加。
                Max(next, m, partM, MERGE_HEADS);
                VDone();
                Sub(alpha, m, next, MERGE_HEADS);
                Sub(beta, partM, next, MERGE_HEADS);
                VDone();
                Exp(alpha, alpha, MERGE_HEADS);
                Exp(beta, beta, MERGE_HEADS);
                VDone();
                Mul(sum, sum, alpha, MERGE_HEADS);
                Mul(partSum, partSum, beta, MERGE_HEADS);
                VDone();
                Add(sum, sum, partSum, MERGE_HEADS);
                Adds(m, next, 0.0f, MERGE_HEADS);
                Brcb(a, alpha, MERGE_HEADS / 8, {1, 8});
                Brcb(b, beta, MERGE_HEADS / 8, {1, 8});
                VDone();
                for (int h = 0; h < MERGE_HEADS; h += 8) {
                    DataCopy(
                        pv,
                        partialsGm
                            [base + 2 * TOTAL_HEADS + (headBase + h) * DIM],
                        8 * DIM
                    );
                    Sync<HardEvent::MTE2_V>();
                    for (int j = 0; j < 8; ++j) {
                        Mul(acc[(h + j) * DIM], acc[(h + j) * DIM],
                            a[(h + j) * 8], 64, DIM / 64, {1, 1, 0, 8, 8, 0});
                        Mul(pv[j * DIM], pv[j * DIM], b[(h + j) * 8], 64,
                            DIM / 64, {1, 1, 0, 8, 8, 0});
                    }
                    VDone();
                    Add(acc[h * DIM], acc[h * DIM], pv, 8 * DIM);
                    Sync<HardEvent::V_MTE2>();
                }
            }
            auto rope = ropeBuf.Get<float>();
            DataCopy(rope, ropeGm[uint64_t(cfg.position) * ROPE_DIM], ROPE_DIM);
            Sync<HardEvent::MTE2_V>();
            Gather(
                rope[64], rope, indexBuf.Get<uint32_t>()[3 * DIM + 64], 0, 64
            );
            Gather(
                rope[128], rope[32], indexBuf.Get<uint32_t>()[3 * DIM + 64], 0,
                64
            );
            VDone();
            Muls(Work(0), rope[128], -1.0f, 64);
            VDone();
            Select(
                rope[128], maskBuf.Get<uint8_t>(), rope[128], Work(0),
                SELMODE::VSEL_TENSOR_TENSOR_MODE, 64
            );
            Sync<HardEvent::V_S>();
            Finalize();
        }
    }

  private:
    template <HardEvent E> __aicore__ inline void Sync() {
        auto id = static_cast<event_t>(GetTPipePtr()->FetchEventID(E));
        SetFlag<E>(id);
        WaitFlag<E>(id);
    }
    __aicore__ inline void VDone() { PipeBarrier<PIPE_V>(); }
    __aicore__ inline LocalTensor<float> Work(int slot) {
        return workBuf.Get<float>()[slot * DIM];
    }
    __aicore__ inline float NegInf() {
        return GetScalarBitcodeValue<uint32_t, float>(0xff800000u);
    }
    __aicore__ inline void RotateFourRows(const LocalTensor<float> &rows) {
        auto paired = Work(0), product = Work(1), rope = ropeBuf.Get<float>();
        for (int j = 0; j < DECODE_ROWS; ++j)
            Gather(
                paired[j * ROPE_DIM], rows[j * DIM + DIM - ROPE_DIM],
                indexBuf.Get<uint32_t>()[3 * DIM], 0, ROPE_DIM
            );
        VDone();
        // 每次 repeat 一行，源/目的的行步长分别为 512 或 64 个 float。
        Mul(paired, paired, rope[128], 64, DECODE_ROWS, {1, 1, 1, 8, 8, 0});
        Mul(product, rows[DIM - ROPE_DIM], rope[64], 64, DECODE_ROWS,
            {1, 1, 1, 8, 64, 0});
        VDone();
        Sub(rows[DIM - ROPE_DIM], product, paired, 64, DECODE_ROWS,
            {1, 1, 1, 64, 8, 8});
        VDone();
    }

    __aicore__ inline void Finalize() {
        auto m = stateBuf.Get<float>(), sum = m[64], lse = m[192],
             gate = m[256], factor = m[128];
        float inf = GetScalarBitcodeValue<uint32_t, float>(0x7f800000u);
        for (int h = 0; h < MERGE_HEADS; ++h)
            lse.SetValue(h, sum.GetValue(h) == 0 ? 1.0f : sum.GetValue(h));
        Sync<HardEvent::S_V>();
        Ln(lse, lse, MERGE_HEADS);
        Sync<HardEvent::V_S>();
        for (int h = 0; h < MERGE_HEADS; ++h) {
            bool empty = sum.GetValue(h) == 0;
            float l = empty ? inf : m.GetValue(h) + lse.GetValue(h);
            lse.SetValue(h, l);
            float delta =
                empty
                    ? 0.0f
                    : (hasSink ? sinkGm.GetValue(headBase + h) : NegInf()) - l;
            factor.SetValue(h, delta);
            gate.SetValue(h, delta > 0 ? -delta : delta);
        }
        Sync<HardEvent::S_V>();
        Exp(gate, gate, MERGE_HEADS);
        Sync<HardEvent::V_S>();
        for (int h = 0; h < MERGE_HEADS; ++h) {
            float s = sum.GetValue(h), t = gate.GetValue(h),
                  delta = factor.GetValue(h);
            float g = delta >= 0 ? t / (1 + t) : 1 / (1 + t);
            factor.SetValue(h, s == 0 ? 0.0f : g / s);
        }
        Sync<HardEvent::S_MTE3>();
        DataCopy(
            lseGm[uint64_t(cfg.query) * TOTAL_HEADS + headBase], lse,
            MERGE_HEADS
        );
        Sync<HardEvent::MTE3_S>();
        for (int h = 0; h < MERGE_HEADS; ++h) {
            auto row = accBuf.Get<float>()[h * DIM];
            float f = factor.GetValue(h);
            if (f == 0)
                Duplicate(row, 0.0f, DIM);
            else
                Muls(row, row, f, DIM);
        }
        VDone();
        for (int h = 0; h < MERGE_HEADS; h += DECODE_ROWS) {
            RotateFourRows(accBuf.Get<float>()[h * DIM]);
            QuantizeRows(accBuf.Get<float>()[h * DIM], h);
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void QuantizeRows(const LocalTensor<float> &row, int h) {
        auto abs = workBuf.Get<float>(), exp = abs[DECODE_ELEMS],
             inv = abs[2 * DECODE_ELEMS];
        auto temp = arenaBuf.Get<float>(), code = temp[DECODE_ELEMS];
        auto groupMax = temp[2 * DECODE_ELEMS], invScales = groupMax[64];
        auto words = invScales[128].ReinterpretCast<uint32_t>();
        Abs(abs, row, DECODE_ELEMS);
        VDone();
        WholeReduceMax(
            groupMax, abs, 32, DECODE_ROWS * 16, 1, 1, 4,
            ReduceOrder::ORDER_ONLY_VALUE
        );
        VDone();
        Maxs(groupMax, groupMax, 1e-4f, 64);
        VDone();
        Muls(groupMax, groupMax, 1.0f / 448.0f, 64);
        VDone();
        // bits-1 把恰好为 2 的幂的边界归入较小指数；e=E-126，scale 字节=E+1。
        Adds(
            code.ReinterpretCast<int32_t>(),
            groupMax.ReinterpretCast<int32_t>(), int32_t(-1), 64
        );
        VDone();
        ShiftRight(
            code.ReinterpretCast<uint32_t>(), code.ReinterpretCast<uint32_t>(),
            uint32_t(23), 64
        );
        VDone();
        Cast(exp, code.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, 64);
        VDone();
        Adds(groupMax, exp, 1.0f, 64);
        VDone();
        Cast(
            code.ReinterpretCast<int32_t>(), groupMax, RoundMode::CAST_RINT, 64
        );
        VDone();
        // 1/scale 的 FP32 指数位为 127-e=253-E，尾数为 0。
        Muls(invScales, exp, -1.0f, 64);
        VDone();
        Adds(invScales, invScales, 253.0f, 64);
        VDone();
        Cast(
            invScales.ReinterpretCast<int32_t>(), invScales,
            RoundMode::CAST_RINT, 64
        );
        VDone();
        ShiftLeft(
            invScales.ReinterpretCast<uint32_t>(),
            invScales.ReinterpretCast<uint32_t>(), uint32_t(23), 64
        );
        Sync<HardEvent::V_S>();
        auto scaleCodes = code.ReinterpretCast<uint32_t>();
        for (int g = 0; g < 16; ++g)
            words.SetValue(
                g * 8, scaleCodes.GetValue(g) |
                           (scaleCodes.GetValue(16 + g) << 8) |
                           (scaleCodes.GetValue(32 + g) << 16) |
                           (scaleCodes.GetValue(48 + g) << 24)
            );
        for (int j = 0; j < DECODE_ROWS; ++j)
            Gather(
                inv[j * DIM], invScales[j * 16], indexBuf.Get<uint32_t>()[DIM],
                0, DIM
            );
        VDone();
        Mul(abs, abs, inv, DECODE_ELEMS);
        VDone();
        Mins(abs, abs, 448.0f, DECODE_ELEMS);
        VDone();
        // E=max(FP32 指数,121)；量化步长为 2^(E-130)，非规格化段固定为 2^-9。
        ShiftRight(
            temp.ReinterpretCast<uint32_t>(), abs.ReinterpretCast<uint32_t>(),
            uint32_t(23), DECODE_ELEMS
        );
        VDone();
        Cast(
            exp, temp.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE,
            DECODE_ELEMS
        );
        VDone();
        Maxs(exp, exp, 121.0f, DECODE_ELEMS);
        VDone();
        Muls(inv, exp, -1.0f, DECODE_ELEMS);
        VDone();
        Adds(inv, inv, 257.0f, DECODE_ELEMS);
        VDone();
        Cast(
            temp.ReinterpretCast<int32_t>(), inv, RoundMode::CAST_RINT,
            DECODE_ELEMS
        );
        VDone();
        ShiftLeft(
            temp.ReinterpretCast<uint32_t>(), temp.ReinterpretCast<uint32_t>(),
            uint32_t(23), DECODE_ELEMS
        );
        VDone();
        Mul(inv, abs, temp, DECODE_ELEMS);
        VDone();
        Cast(
            code.ReinterpretCast<int32_t>(), inv, RoundMode::CAST_RINT,
            DECODE_ELEMS
        );
        VDone();
        Cast(
            code, code.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE,
            DECODE_ELEMS
        );
        VDone();
        Adds(exp, exp, -121.0f, DECODE_ELEMS);
        VDone();
        Muls(exp, exp, 8.0f, DECODE_ELEMS);
        VDone();
        Add(code, code, exp, DECODE_ELEMS);
        ShiftRight(
            temp.ReinterpretCast<uint32_t>(), row.ReinterpretCast<uint32_t>(),
            uint32_t(31), DECODE_ELEMS
        );
        VDone();
        Cast(
            temp, temp.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE,
            DECODE_ELEMS
        );
        VDone();
        Muls(temp, temp, 128.0f, DECODE_ELEMS);
        VDone();
        Add(code, code, temp, DECODE_ELEMS);
        VDone();
        // 码字 0..254 可经 half 精确转成 uint8；符号位也保留负零。
        Cast(rowBuf.Get<half>(), code, RoundMode::CAST_RINT, DECODE_ELEMS);
        VDone();
        auto packed = abs.ReinterpretCast<uint8_t>();
        Cast(packed, rowBuf.Get<half>(), RoundMode::CAST_RINT, DECODE_ELEMS);
        Sync<HardEvent::V_MTE3>();
        for (int j = 0; j < DECODE_ROWS; ++j) {
            int globalHead = headBase + h + j;
            uint64_t base = uint64_t(cfg.query) * TOTAL_HEADS * DIM +
                            (globalHead / 8) * 4096 + (globalHead % 8) * 32;
            DataCopy(fp8Gm[base], packed[j * DIM], {16, 1, 0, 7});
        }
        Sync<HardEvent::S_MTE3>();
        for (int g = 0; g < 16; ++g) {
            int globalHead = headBase + h;
            int word = (globalHead / 8) * 32 + g * 2 + (globalHead % 8) / 4;
            DataCopyPad(
                scalesGm[uint64_t(word) * scaleStride + cfg.query],
                words[g * 8], {1, 4, 0, 0, 0}
            );
        }
        Sync<HardEvent::MTE3_V>();
        Sync<HardEvent::MTE3_S>();
    }

    QueryConfig cfg;
    int32_t headBase, scaleStride;
    bool hasSink;
    GlobalTensor<int32_t> configsGm;
    GlobalTensor<float> partialsGm, ropeGm, sinkGm, lseGm;
    GlobalTensor<uint8_t> fp8Gm;
    GlobalTensor<uint32_t> scalesGm;
    TBuf<TPosition::VECCALC> accBuf, workBuf, arenaBuf, rowBuf, stateBuf,
        indexBuf, maskBuf, ropeBuf;
};
extern "C" __global__ __aicore__ void mega_attn_v6_merge(
    GM_ADDR partials, GM_ADDR configs, GM_ADDR rope, GM_ADDR sink, GM_ADDR fp8,
    GM_ADDR scales, GM_ADDR lse, int32_t stride, GM_ADDR tables, uint32_t count
) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    MergeAttentionV6 op;
    op.Init(
        partials, configs, rope, sink, fp8, scales, lse, stride, tables, &pipe
    );
    op.Process(count);
}
extern "C" void finish_mega_attn_v6(
    const Input &in, const Output &out, const QueryConfig *configs,
    uint32_t count, uint32_t blocks, void *partials, void *tables, void *stream
) {
    mega_attn_v6_merge<<<blocks * 2, nullptr, stream>>>(
        (uint8_t *)partials, (uint8_t *)configs, (uint8_t *)in.cos_sin,
        (uint8_t *)in.sink, out.fp8, (uint8_t *)out.scales, (uint8_t *)out.lse,
        out.sf_stride, (uint8_t *)tables, count
    );
}
