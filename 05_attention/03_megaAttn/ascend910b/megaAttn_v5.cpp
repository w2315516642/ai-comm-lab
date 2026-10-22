#include "megaAttn_v5.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;
using namespace mega_ascend_v5;
constexpr int COMPUTE_UB_BYTES =
    USER_UB_BYTES - INDEX_WORDS * 4 + 128 * 4 + QUERY_INDEX_WORDS * 4;
static_assert(COMPUTE_UB_BYTES + SDK_UB_RESERVE <= 192 * 1024);

// 两个 AIV 准备同一份 GM 矩阵；IBSHARE=true 让 Cube 等两侧都写完再计算。
using AType = MatmulType<
    TPosition::GM, CubeFormat::ND, bfloat16_t, false, LayoutMode::NONE, true>;
using KType = MatmulType<
    TPosition::GM, CubeFormat::ND, bfloat16_t, true, LayoutMode::NONE, true>;
using CType = MatmulType<TPosition::GM, CubeFormat::ND, float>;

// AIC 自己推进 QK/PV，AIV 只发准备完成通知，省去逐次 Matmul 消息解析。
constexpr auto V5_MM_CONFIG = GetNormalConfig(
    false, false, false, BatchMode::BATCH_LESS_THAN_L1, true,
    IterateOrder::UNDEF, ScheduleType::INNER_PRODUCT, true, true
);
// 两个 AIV 各处理 32 head/32 行 KV；Cube 每次计算完整 64 head。
class KernelMegaAttentionV5 {
  public:
    Matmul<AType, KType, CType, CType, V5_MM_CONFIG> qkMm;
    Matmul<AType, AType, CType, CType, V5_MM_CONFIG> pvMm;

    __aicore__ inline void Init(
        GM_ADDR q, GM_ADDR kv8, GM_ADDR kv4, GM_ADDR ids, GM_ADDR extraIds,
        GM_ADDR rope, GM_ADDR configs, uint32_t count, GM_ADDR scratch,
        GM_ADDR profiles, GM_ADDR tables, TPipe *pipe
    ) {
        qGm.SetGlobalBuffer((__gm__ bfloat16_t *)q);
        kv8Gm.SetGlobalBuffer((__gm__ uint8_t *)kv8);
        kv4Gm.SetGlobalBuffer((__gm__ uint8_t *)kv4);
        idsGm.SetGlobalBuffer((__gm__ int32_t *)ids);
        extraIdsGm.SetGlobalBuffer((__gm__ int32_t *)extraIds);
        ropeGm.SetGlobalBuffer((__gm__ float *)rope);
        configsGm.SetGlobalBuffer((__gm__ int32_t *)configs);
        profiling = profiles != nullptr;
        if (profiling)
            profileGm.SetGlobalBuffer((__gm__ uint64_t *)profiles);
        taskCount = count;
        headBase = GetSubBlockIdx() * HEADS;
        uint32_t group = GetBlockIdx();
        if ASCEND_IS_AIV {
            group /= 2;
        }
        GM_ADDR base = scratch + uint64_t(group) * SCRATCH_PER_CORE;
        scoresGm.SetGlobalBuffer((__gm__ float *)base, TOTAL_HEADS * KV_TILE);
        pvGm.SetGlobalBuffer(
            (__gm__ float *)(base + TOTAL_HEADS * KV_TILE * 4),
            TOTAL_HEADS * DIM
        );

        GM_ADDR shared = base + TOTAL_HEADS * (KV_TILE + DIM) * 4;
        qPrepared.SetGlobalBuffer(
            (__gm__ bfloat16_t *)shared, TOTAL_HEADS * DIM
        );
        shared += TOTAL_HEADS * DIM * 2;
        for (int i = 0; i < 2; ++i)
            kvPrepared[i].SetGlobalBuffer(
                (__gm__ bfloat16_t *)(shared + i * KV_TILE * DIM * 2),
                KV_TILE * DIM
            );
        shared += 2 * KV_TILE * DIM * 2;
        pPrepared.SetGlobalBuffer(
            (__gm__ bfloat16_t *)shared, TOTAL_HEADS * KV_TILE
        );
        partialsGm.SetGlobalBuffer((__gm__ float *)(scratch +
                                                    uint64_t(GetBlockNum()) *
                                                        SCRATCH_PER_CORE));
        if ASCEND_IS_AIC {
            return;
        }
        pipe->InitBuffer(accBuf, HEADS * DIM * 4);
        pipe->InitBuffer(arenaBuf, ARENA_BYTES);
        pipe->InitBuffer(probBuf, SCORE_ELEMS * 2);
        pipe->InitBuffer(workBuf, WORK_BYTES);
        pipe->InitBuffer(rowBuf, DECODE_ELEMS * 2);
        pipe->InitBuffer(decodeOutBuf, 2 * DECODE_ELEMS * 2);
        pipe->InitBuffer(idsBuf, KV_TILE * 4);
        pipe->InitBuffer(ropeIndexBuf, 128 * 4);
        pipe->InitBuffer(maskBuf, 64);
        pipe->InitBuffer(lutBuf, 768 * 4);
        pipe->InitBuffer(stateBuf, STATE_BYTES);
        pipe->InitBuffer(ropeBuf, 192 * 4);
        pipe->InitBuffer(queryIndexBuf, QUERY_INDEX_WORDS * 4);
        InitTables(tables);
    }

    __aicore__ inline void Process() {
        if ASCEND_IS_AIC {
            ProcessCube();
            return;
        }
        for (uint32_t task = GetBlockIdx() / 2; task < taskCount * KV_SPLITS;
             task += GetBlockNum()) {
            queryTask = task % taskCount;
            split = task / taskCount;
            auto fields = reinterpret_cast<int32_t *>(&cfg);
            for (int i = 0; i < 12; ++i)
                fields[i] = configsGm.GetValue(uint64_t(queryTask) * 12 + i);
            for (int i = 0; i < ProfileCount; ++i)
                cycles[i] = 0;
            auto begin = Tick();
            ProcessQuery();
            cycles[Total] = Tick() - begin;
            if (profiling) {
                auto local = stateBuf.Get<uint64_t>();
                for (int i = 0; i < ProfileCount; ++i)
                    local.SetValue(i, cycles[i]);
                Sync<HardEvent::S_MTE3>();
                DataCopy(
                    profileGm
                        [(uint64_t(task) * 2 + GetSubBlockIdx()) *
                         ProfileCount],
                    local, ProfileCount
                );
                Sync<HardEvent::MTE3_S>();
            }
        }
    }

  private:
    template <HardEvent E> __aicore__ inline void Sync() {
        auto id = static_cast<event_t>(GetTPipePtr()->FetchEventID(E));
        SetFlag<E>(id);
        WaitFlag<E>(id);
    }
    __aicore__ inline uint64_t Tick() {
        return profiling ? GetSystemCycle() : 0;
    }
    __aicore__ inline void VDone() { PipeBarrier<PIPE_V>(); }
    __aicore__ inline LocalTensor<float> Work(int slot) {
        return workBuf.Get<float>()[slot * DIM];
    }
    __aicore__ inline float NegInf() {
        return GetScalarBitcodeValue<uint32_t, float>(0xff800000u);
    }

    __aicore__ inline void InitTables(GM_ADDR address) {
        GlobalTensor<uint32_t> table;
        table.SetGlobalBuffer(
            (__gm__ uint32_t *)address, mega_ascend_v5::TABLE_WORDS
        );
        DataCopy(
            queryIndexBuf.Get<uint32_t>(), table[mega_ascend_v3::TABLE_WORDS],
            QUERY_INDEX_WORDS
        );
        DataCopy(lutBuf.Get<uint32_t>(), table, 768);
        DataCopy(ropeIndexBuf.Get<uint32_t>(), table[768 + 3 * DIM], 128);
        DataCopy(maskBuf.Get<uint32_t>(), table[768 + INDEX_WORDS], 16);
        Sync<HardEvent::MTE2_V>();
    }

    __aicore__ inline void ProcessCube() {
        for (uint32_t task = GetBlockIdx(); task < taskCount * KV_SPLITS;
             task += GetBlockNum()) {
            queryTask = task % taskCount;
            split = task / taskCount;
            auto fields = reinterpret_cast<int32_t *>(&cfg);
            for (int i = 0; i < 12; ++i)
                fields[i] = configsGm.GetValue(uint64_t(queryTask) * 12 + i);
            int tiles = (cfg.topk + cfg.extra_topk + KV_TILE - 1) / KV_TILE;
            int boundary = SplitBoundary(tiles), end = split ? tiles : boundary,
                slot = 0;
            int current = NextTile(split ? boundary : 0, end);
            if (current < end) {
                qkMm.SetTensorA(qPrepared);
                qkMm.SetTensorB(kvPrepared[0], true);
                qkMm.IterateAll<false>(scoresGm, 0, false, true);
            }
            while (current < end) {
                int next = NextTile(current + 1, end);
                // 与 AIV 通知顺序一致：先下一块 QK，再当前块 PV。
                if (next < end) {
                    qkMm.SetTensorA(qPrepared);
                    qkMm.SetTensorB(kvPrepared[slot ^ 1], true);
                    qkMm.IterateAll<false>(scoresGm, 0, false, true);
                }
                pvMm.SetTensorA(pPrepared);
                pvMm.SetTensorB(kvPrepared[slot], false);
                pvMm.IterateAll<false>(pvGm, 0, false, true);
                current = next;
                slot ^= 1;
            }
        }
    }

    // 目标卡有 24 个 AIC 核组。分到 3 个任务的组少分 tile，2 个任务的组多分。
    // 固定逻辑调度宽度，实际只启动 1/2/5 个组验证时仍保持同一归约顺序。
    __aicore__ inline int SplitBoundary(int tiles) {
        constexpr uint32_t groups = 24;
        uint32_t g0 = queryTask % groups, g1 = (queryTask + taskCount) % groups;
        uint32_t n0 = (taskCount * KV_SPLITS + groups - 1 - g0) / groups;
        uint32_t n1 = (taskCount * KV_SPLITS + groups - 1 - g1) / groups;
        // 按每组任务数的倒数分配：t0*n0 约等于 t1*n1，且 t0+t1=tiles。
        return (tiles * n1 + (n0 + n1) / 2) / (n0 + n1);
    }

    // 保留原始 tile 边界，只跳过与两个有效长度区间均无交集的块。
    __aicore__ inline int NextTile(int tile, int end) {
        while (tile < end) {
            int begin = tile * KV_TILE;
            if ((begin < cfg.topk && begin < cfg.length) ||
                (begin + KV_TILE > cfg.topk &&
                 begin < cfg.topk + cfg.extra_length))
                break;
            ++tile;
        }
        return tile;
    }
    __aicore__ inline void ProcessQuery() {
        auto mark = Tick();
        InitOnline();
        int allTiles = (cfg.topk + cfg.extra_topk + KV_TILE - 1) / KV_TILE;
        int boundary = SplitBoundary(allTiles),
            end = split ? allTiles : boundary;
        int current = NextTile(split ? boundary : 0, end);
        if (current < end)
            LoadQuery();
        cycles[Query] += Tick() - mark;
        if (current < end) {
            int next = NextTile(current + 1, end),
                later = NextTile(next + 1, end), slot = 0;
            mark = Tick();
            PrepareKv(0, current * KV_TILE);
            cycles[mega_ascend_v5::PrepareKv] += Tick() - mark;
            StartQk(0);
            if (next < end) {
                mark = Tick();
                PrepareKv(1, next * KV_TILE);
                cycles[mega_ascend_v5::PrepareKv] += Tick() - mark;
            }
            while (current < end) {
                mark = Tick();
                qkMm.WaitIterateAll();
                qkMm.End();
                LoadScores(slot);
                cycles[WaitQk] += Tick() - mark;
                if (next < end)
                    StartQk(slot ^ 1);
                mark = Tick();
                OnlineSoftmax();
                cycles[Softmax] += Tick() - mark;
                StartPv(slot);
                if (later < end) {
                    mark = Tick();
                    PrefetchKv(later * KV_TILE);
                    cycles[mega_ascend_v5::PrepareKv] += Tick() - mark;
                }
                mark = Tick();
                pvMm.WaitIterateAll();
                pvMm.End();
                cycles[WaitPv] += Tick() - mark;
                mark = Tick();
                AccumulateOutput();
                cycles[Accumulate] += Tick() - mark;
                if (later < end) {
                    mark = Tick();
                    DecodeKv(slot, later * KV_TILE);
                    cycles[mega_ascend_v5::PrepareKv] += Tick() - mark;
                }
                current = next;
                next = later;
                later = NextTile(later + 1, end);
                slot ^= 1;
            }
        }
        mark = Tick();
        StorePartial();
        cycles[Finish] += Tick() - mark;
    }

    __aicore__ inline void LoadQuery() {
        auto rope = ropeBuf.Get<float>();
        DataCopy(rope, ropeGm[uint64_t(cfg.position) * ROPE_DIM], ROPE_DIM);
        Sync<HardEvent::MTE2_V>();
        Gather(rope[64], rope, ropeIndexBuf.Get<uint32_t>()[64], 0, 64);
        Gather(rope[128], rope[32], ropeIndexBuf.Get<uint32_t>()[64], 0, 64);
        VDone();
        Muls(Work(0), rope[128], -1.0f, 64);
        VDone();
        Select(
            rope[128], maskBuf.Get<uint8_t>(), rope[128], Work(0),
            SELMODE::VSEL_TENSOR_TENSOR_MODE, 64
        );
        VDone();

        auto row = rowBuf.Get<bfloat16_t>();
        auto packed = workBuf.Get<float>(), rows = packed[DECODE_ELEMS];
        for (int h = 0; h < HEADS; h += DECODE_ROWS) {
            // 32 个 D 分块，每块连续读取 4 个 head 的 16 个 BF16。
            DataCopy(
                row,
                qGm[uint64_t(cfg.query) * TOTAL_HEADS * DIM +
                    (headBase + h) * 16],
                {32, 4, TOTAL_HEADS - 4, 0}
            );
            Sync<HardEvent::MTE2_V>();
            Cast(packed, row, RoundMode::CAST_NONE, DECODE_ELEMS);
            VDone();
            Gather(
                rows, packed, queryIndexBuf.Get<uint32_t>(), 0, DECODE_ELEMS
            );
            VDone();
            RotateQueryRows(rows);
            Cast(row, rows, RoundMode::CAST_RINT, DECODE_ELEMS);
            Sync<HardEvent::V_MTE3>();
            DataCopy(qPrepared[(headBase + h) * DIM], row, DECODE_ELEMS);
            Sync<HardEvent::MTE3_MTE2>();
        }
        Sync<HardEvent::MTE3_S>();
    }

    __aicore__ inline void RotateQueryRows(const LocalTensor<float> &rows) {
        auto paired = Work(0), product = Work(1), rope = ropeBuf.Get<float>();
        for (int j = 0; j < DECODE_ROWS; ++j)
            Gather(
                paired[j * ROPE_DIM], rows[j * DIM + DIM - ROPE_DIM],
                ropeIndexBuf.Get<uint32_t>(), 0, ROPE_DIM
            );
        VDone();
        Mul(paired, paired, rope[128], 64, DECODE_ROWS, {1, 1, 1, 8, 8, 0});
        Mul(product, rows[DIM - ROPE_DIM], rope[64], 64, DECODE_ROWS,
            {1, 1, 1, 8, 64, 0});
        VDone();
        Add(rows[DIM - ROPE_DIM], product, paired, 64, DECODE_ROWS,
            {1, 1, 1, 64, 8, 8});
        VDone();
    }

    __aicore__ inline void PrepareKv(int slot, int begin) {
        PrefetchKv(begin);
        DecodeKv(slot, begin);
    }
    __aicore__ inline void PrefetchKv(int begin) {
        // arena 即将复用：Vector 清零和 MTE2 搬入都要等此前的 MTE3 读取结束。
        Sync<HardEvent::MTE3_V>();
        Sync<HardEvent::MTE3_MTE2>();
        auto raw = arenaBuf.Get<uint8_t>();
        int mainRows = cfg.topk - begin;
        mainRows = mainRows < 0 ? 0 : (mainRows > KV_TILE ? KV_TILE : mainRows);
        int extraBegin = begin + mainRows - cfg.topk;
        int extraRows = cfg.extra_topk - extraBegin;
        extraRows = extraRows < 0
                        ? 0
                        : (extraRows > KV_TILE - mainRows ? KV_TILE - mainRows
                                                          : extraRows);
        bool full = mainRows + extraRows == KV_TILE &&
                    (!mainRows || begin + mainRows <= cfg.length) &&
                    (!extraRows || extraBegin + extraRows <= cfg.extra_length);
        // 完整 tile 不清零；FP8 scale 的尾部由搬运指令补齐为零。
        if (!full) {
            Duplicate(
                raw.ReinterpretCast<uint16_t>(), uint16_t(0), ARENA_BYTES / 2
            );
            Sync<HardEvent::V_MTE2>();
        }
        auto ids = idsBuf.Get<int32_t>();
        DataCopyPadExtParams<int32_t> padIds{false, 0, 0, 0};
        if (mainRows)
            DataCopyPad(
                ids, idsGm[uint64_t(cfg.query) * cfg.indices_stride + begin],
                {1, uint32_t(mainRows * 4), 0, 0, 0}, padIds
            );
        // extra 起点可能不对齐，分别搬入 work，再用标量读；两个来源不互相覆盖。
        if (extraRows)
            DataCopyPad(
                workBuf.Get<int32_t>(),
                extraIdsGm
                    [uint64_t(cfg.query) * cfg.extra_indices_stride +
                     extraBegin],
                {1, uint32_t(extraRows * 4), 0, 0, 0}, padIds
            );
        Sync<HardEvent::MTE2_S>();
        for (int t = headBase; t < headBase + HEADS; ++t) {
            bool extra = t >= mainRows;
            int idx = extra ? extraBegin + t - mainRows : begin + t;
            int capacity = extra ? cfg.extra_topk : cfg.topk,
                length = extra ? cfg.extra_length : cfg.length;
            if (!full && (idx >= capacity || idx >= length))
                continue;
            int id = extra ? workBuf.Get<int32_t>().GetValue(t - mainRows)
                           : ids.GetValue(t);
            int page = extra ? cfg.extra_page_size : cfg.page_size;
            int pageStride = extra ? cfg.extra_page_stride : cfg.page_stride;
            uint32_t bytes = extra ? DIM / 2 : DIM,
                     scales = extra ? DIM / 16 : DIM / 32;
            int pageId = page == 128 ? (id >> 7) : (id / page);
            int inPage = page == 128 ? (id & 127) : (id % page);
            uint64_t base = uint64_t(pageId) * pageStride;
            auto cache = extra ? kv4Gm : kv8Gm;
            int payload =
                extra ? mainRows * DIM + (t - mainRows) * (DIM / 2) : t * DIM;
            DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};
            DataCopyPad(
                raw[payload], cache[base + uint64_t(inPage) * bytes],
                {1, bytes, 0, 0, 0}, pad
            );
            DataCopyPadExtParams<uint8_t> scalePad{
                true, 0, uint8_t(32 - scales), 0
            };
            DataCopyPad(
                raw[KV_TILE * DIM + t * 32],
                cache
                    [base + uint64_t(page) * bytes + uint64_t(inPage) * scales],
                {1, scales, 0, 0, 0}, scalePad
            );
        }
    }

    __aicore__ inline void ByteIndices(
        const LocalTensor<uint8_t> &src, int count
    ) {
        auto offsets = workBuf.Get<int32_t>()[2 * DECODE_ELEMS];
        Cast(rowBuf.Get<half>(), src, RoundMode::CAST_NONE, count);
        VDone();
        Cast(offsets, rowBuf.Get<half>(), RoundMode::CAST_RINT, count);
        VDone();
        ShiftLeft(
            offsets.ReinterpretCast<uint32_t>(),
            offsets.ReinterpretCast<uint32_t>(), uint32_t(2), count
        );
        VDone();
    }

    __aicore__ inline void DecodeKv(int slot, int begin) {
        Sync<HardEvent::MTE2_V>();
        auto raw = arenaBuf.Get<uint8_t>();
        auto lut = lutBuf.Get<float>();
        auto values = workBuf.Get<float>(), scales = values[DECODE_ELEMS];
        auto offsets = values[2 * DECODE_ELEMS].ReinterpretCast<uint32_t>();

        int mainRows = cfg.topk - begin;
        mainRows = mainRows < 0 ? 0 : (mainRows > KV_TILE ? KV_TILE : mainRows);
        tileBegin[slot] = begin;
        int batch = 0;
        for (int t = headBase; t < headBase + HEADS; ++batch) {
            bool fp4 = t >= mainRows;
            int limit = headBase + HEADS;
            int rows =
                fp4 ? limit - t : (mainRows < limit ? mainRows : limit) - t;
            rows = rows > DECODE_ROWS ? DECODE_ROWS : rows;
            int elems = rows * DIM,
                payload =
                    fp4 ? mainRows * DIM + (t - mainRows) * (DIM / 2) : t * DIM;
            ByteIndices(raw[payload], fp4 ? elems / 2 : elems);
            if (fp4) {
                Gather(
                    rowBuf.Get<uint32_t>(), lutBuf.Get<uint32_t>()[512],
                    offsets, 0, elems / 2
                );
                VDone();
                Cast(
                    values, rowBuf.Get<bfloat16_t>(), RoundMode::CAST_NONE,
                    elems
                );
                VDone();
            } else {
                Gather(values, lut, offsets, 0, elems);
                VDone();
            }
            ByteIndices(raw[KV_TILE * DIM + t * 32], rows * 32);
            Gather(scales, lut[fp4 ? 0 : 256], offsets, 0, rows * 32);
            VDone();
            // 每个 scale 先广播成 8 个 float；一次 repeat 用同一 scale 乘 16/32
            // 个值。
            auto expanded = values[2 * DECODE_ELEMS];
            Brcb(expanded, scales, rows * 4, {1, 8});
            VDone();
            for (int r = 0; r < rows; ++r) {
                if (fp4)
                    Mul(values[r * DIM], values[r * DIM], expanded[r * 256], 16,
                        32, {1, 1, 0, 2, 2, 1});
                else
                    Mul(values[r * DIM], values[r * DIM], expanded[r * 256], 32,
                        16, {1, 1, 0, 4, 4, 1});
            }
            VDone();
            // 两块 BF16 缓冲交替写；只在即将复用前等待上一轮 MTE3。
            auto output =
                decodeOutBuf.Get<bfloat16_t>()[(batch & 1) * DECODE_ELEMS];
            Sync<HardEvent::MTE3_V>();
            Cast(output, values, RoundMode::CAST_RINT, elems);
            Sync<HardEvent::V_MTE3>();
            DataCopy(kvPrepared[slot][t * DIM], output, elems);
            t += rows;
        }
        Sync<HardEvent::MTE3_S>();
    }

    __aicore__ inline void InitOnline() {
        // 用最小有限值初始化，空块的最大值差不会出现 -inf-(-inf)。
        Duplicate(stateBuf.Get<float>(), -3.402823466e38f, HEADS);
        Duplicate(stateBuf.Get<float>()[64], 0.0f, HEADS);
        Duplicate(accBuf.Get<float>(), 0.0f, HEADS * DIM);
        Sync<HardEvent::V_S>();
    }
    __aicore__ inline void StartQk(int slot) {
        auto mark = Tick();
        qkMm.SetTensorA(qPrepared);
        qkMm.SetTensorB(kvPrepared[slot], true);
        qkMm.IterateAll<false>(scoresGm, 0, false, true);
        cycles[WaitQk] += Tick() - mark;
    }
    __aicore__ inline void LoadScores(int slot) {
        auto scores = arenaBuf.Get<float>(), mask = Work(0);
        int begin = tileBegin[slot], mainRows = cfg.topk - begin;
        mainRows = mainRows < 0 ? 0 : (mainRows > KV_TILE ? KV_TILE : mainRows);
        int extraBegin = begin + mainRows - cfg.topk;
        bool full = (!mainRows || begin + mainRows <= cfg.length) &&
                    (mainRows == KV_TILE ||
                     extraBegin + KV_TILE - mainRows <= cfg.extra_length);
        if (!full) {
            for (int t = 0; t < KV_TILE; ++t) {
                int idx = t < mainRows ? begin + t : extraBegin + t - mainRows;
                int length = t < mainRows ? cfg.length : cfg.extra_length;
                mask.SetValue(t, idx < length ? 0.0f : NegInf());
            }
            Sync<HardEvent::S_V>();
        }
        DataCopy(scores, scoresGm[headBase * KV_TILE], SCORE_ELEMS);
        Sync<HardEvent::MTE2_V>();
        // 下一块 QK 会覆盖同一 GM 输出，标量发射前必须确认这次读取已完成。
        Sync<HardEvent::MTE2_S>();
        Muls(scores, scores, SCORE_SCALE, SCORE_ELEMS);
        VDone();
        if (!full)
            for (int h = 0; h < HEADS; ++h)
                Add(scores[h * KV_TILE], scores[h * KV_TILE], mask, KV_TILE);
        VDone();
    }
    __aicore__ inline void OnlineSoftmax() {
        auto scores = arenaBuf.Get<float>();
        auto prob = probBuf.Get<bfloat16_t>();
        auto m = stateBuf.Get<float>(), sum = m[64], alpha = m[128],
             tileMax = m[192], tileSum = m[256];
        auto delta = Work(0), safeMax = Work(1), broadcast = Work(2);
        auto update = rowBuf.Get<uint8_t>();
        WholeReduceMax(
            tileMax, scores, KV_TILE, HEADS, 1, 1, 8,
            ReduceOrder::ORDER_ONLY_VALUE
        );
        VDone();
        Adds(alpha, m, 0.0f, HEADS);
        Sub(delta, tileMax, m, HEADS);
        VDone();
        // Compares 的 calCount 版本要求 64 个 float；用显式 mask 处理 32 head。
        Compares(
            update, delta, RESCALE_DELTA, CMPMODE::GT, uint64_t(HEADS), 1,
            {1, 1, 8, 8}
        );
        VDone();
        Select(m, update, tileMax, m, SELMODE::VSEL_TENSOR_TENSOR_MODE, HEADS);
        VDone();
        // 有限下界配合被 mask 成 -inf 的分数，让空行的 exp 和行和均为 0。
        Maxs(safeMax, m, -3.402823466e38f, HEADS);
        VDone();
        Sub(alpha, alpha, safeMax, HEADS);
        Brcb(broadcast, safeMax, HEADS / 8, {1, 8});
        VDone();
        // 每个 head 广播成一个 32B 块，src1 的块步长为 0，覆盖整行 64 个分数。
        Sub(scores, scores, broadcast, 64, HEADS, {1, 1, 0, 8, 8, 1});
        VDone();
        Exp(scores, scores, SCORE_ELEMS);
        Exp(alpha, alpha, HEADS);
        VDone();
        WholeReduceSum(tileSum, scores, KV_TILE, HEADS, 1, 1, 8);
        Cast(prob, scores, RoundMode::CAST_RINT, SCORE_ELEMS);
        VDone();
        Mul(sum, sum, alpha, HEADS);
        VDone();
        Add(sum, sum, tileSum, HEADS);
        VDone();

        Sync<HardEvent::V_MTE3>();
        DataCopy(pPrepared[headBase * KV_TILE], prob, SCORE_ELEMS);
        Sync<HardEvent::MTE3_S>();
    }
    __aicore__ inline void StartPv(int slot) {
        auto mark = Tick();
        pvMm.SetTensorA(pPrepared);
        pvMm.SetTensorB(kvPrepared[slot], false);
        pvMm.IterateAll<false>(pvGm, 0, false, true);
        cycles[WaitPv] += Tick() - mark;
    }
    __aicore__ inline void AccumulateOutput() {
        auto acc = accBuf.Get<float>(), pv = workBuf.Get<float>();
        auto broadcast = workBuf.Get<float>()[2 * DECODE_ELEMS];
        auto alpha = stateBuf.Get<float>()[128];
        Brcb(broadcast, alpha, HEADS / 8, {1, 8});
        VDone();
        for (int h = 0; h < HEADS; h += 8) {
            DataCopy(pv, pvGm[(headBase + h) * DIM], 8 * DIM);
            Sync<HardEvent::MTE2_V>();
            for (int j = 0; j < 8; ++j)
                Mul(acc[(h + j) * DIM], acc[(h + j) * DIM],
                    broadcast[(h + j) * 8], 64, DIM / 64, {1, 1, 0, 8, 8, 0});
            VDone();
            Add(acc[h * DIM], acc[h * DIM], pv, 8 * DIM);
            Sync<HardEvent::V_MTE2>();
        }
        Sync<HardEvent::V_S>();
    }

    // 分片保存指数基准 m、指数和 l、未归一化 FP32 输出，不能先量化。
    __aicore__ inline void StorePartial() {
        uint64_t base =
            (uint64_t(queryTask) * KV_SPLITS + split) * PARTIAL_FLOATS;
        Sync<HardEvent::V_MTE3>();
        DataCopy(partialsGm[base + headBase], stateBuf.Get<float>(), HEADS);
        DataCopy(
            partialsGm[base + TOTAL_HEADS + headBase],
            stateBuf.Get<float>()[64], HEADS
        );
        DataCopy(
            partialsGm[base + 2 * TOTAL_HEADS + headBase * DIM],
            accBuf.Get<float>(), HEADS * DIM
        );
        Sync<HardEvent::MTE3_V>();
        Sync<HardEvent::MTE3_S>();
    }

    QueryConfig cfg;
    uint32_t queryTask, split;
    GlobalTensor<float> partialsGm;
    uint32_t taskCount;
    int32_t headBase;
    bool profiling;
    int tileBegin[2];
    uint64_t cycles[ProfileCount];
    GlobalTensor<bfloat16_t> qGm;
    GlobalTensor<uint8_t> kv8Gm, kv4Gm;
    GlobalTensor<int32_t> idsGm, extraIdsGm, configsGm;
    GlobalTensor<float> ropeGm, scoresGm, pvGm;
    GlobalTensor<uint64_t> profileGm;
    TBuf<TPosition::VECCALC> accBuf, workBuf, ropeIndexBuf, maskBuf, lutBuf,
        stateBuf, ropeBuf;
    TBuf<TPosition::VECOUT> arenaBuf, rowBuf, decodeOutBuf, probBuf;
    TBuf<TPosition::VECCALC> idsBuf, queryIndexBuf;
    GlobalTensor<bfloat16_t> qPrepared, pPrepared, kvPrepared[2];
};

extern "C" __global__ __aicore__ void mega_attn_v5(
    GM_ADDR q, GM_ADDR kv8, GM_ADDR kv4, GM_ADDR ids, GM_ADDR extraIds,
    GM_ADDR rope, GM_ADDR configs, uint32_t count, GM_ADDR scratch,
    GM_ADDR profiles, GM_ADDR tables, GM_ADDR qkTilingGm, GM_ADDR pvTilingGm,
    GM_ADDR workspace
) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe pipe;
    KernelMegaAttentionV5 op;
    TCubeTiling qk, pv;
    auto a = (__gm__ uint32_t *)qkTilingGm, b = (__gm__ uint32_t *)pvTilingGm;
    for (int i = 0; i < sizeof(TCubeTiling) / 4; ++i) {
        ((uint32_t *)&qk)[i] = a[i];
        ((uint32_t *)&pv)[i] = b[i];
    }
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), op.qkMm, &qk, op.pvMm, &pv);
    op.Init(
        q, kv8, kv4, ids, extraIds, rope, configs, count, scratch, profiles,
        tables, &pipe
    );
    op.Process();
}

extern "C" void launch_mega_attn_v5(
    const Input &in, const Output &out, const QueryConfig *configs,
    uint32_t count, uint32_t blocks, void *scratch, void *qkTiling,
    void *pvTiling, void *workspace, void *tables, void *stream,
    Profile *profiles
) {
    if (!count)
        return;
    mega_attn_v5<<<blocks, nullptr, stream>>>(
        (uint8_t *)in.q, (uint8_t *)in.kv_fp8, (uint8_t *)in.kv_fp4,
        (uint8_t *)in.indices, (uint8_t *)in.extra_indices,
        (uint8_t *)in.cos_sin, (uint8_t *)configs, count, (uint8_t *)scratch,
        (uint8_t *)profiles, (uint8_t *)tables, (uint8_t *)qkTiling,
        (uint8_t *)pvTiling, (uint8_t *)workspace
    );
    finish_mega_attn_v4(
        in, out, configs, count, blocks,
        (uint8_t *)scratch + uint64_t(blocks) * SCRATCH_PER_CORE, tables, stream
    );
}
