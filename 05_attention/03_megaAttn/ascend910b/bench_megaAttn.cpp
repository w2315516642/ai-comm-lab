// 复用原 benchmark 的 CPU 数据生成/编解码；不编译 CUDA 部分，也不复制一套 oracle。
#define MEGA_ATTN_REFERENCE_ONLY
#include "../sm100/bench_megaAttn.cu"
#undef MEGA_ATTN_REFERENCE_ONLY
#include "megaAttn_api.h"
#include <cstdio>
#include <cstdlib>

using namespace mega_ascend;
using mega_test::check;

namespace {
// 每个阶段观测点的 CPU 预期值；Q/KV 保留 BF16 存储位，分数用 FP32。
struct Expected {
    std::vector<uint16_t> q = std::vector<uint16_t>(Q_ELEMS);
    std::vector<uint16_t> rope = std::vector<uint16_t>(Q_ELEMS);
    std::vector<uint16_t> kv = std::vector<uint16_t>(KV_ELEMS, 0);
    std::vector<int32_t> valid = std::vector<int32_t>(KV_TILE, 0);
    std::vector<float> raw = std::vector<float>(SCORE_ELEMS, 0);
    std::vector<float> masked = std::vector<float>(SCORE_ELEMS, -INFINITY);
};

Stage1Config MakeConfig(const mega_test::Case &c, int q, int head, int begin) {
    check(q >= 0 && q < c.n && head >= 0 && head + HEAD_TILE <= HEADS &&
          head % HEAD_TILE == 0 && begin >= 0 && begin < c.k + c.ek &&
          begin % KV_TILE == 0, "Invalid stage-1 coordinates");
    return {q, head, begin, c.positions[q], c.k, c.ek, c.lengths[q], c.extra_lengths[q],
            c.main.size, c.main.pages, c.main.stride, c.extra.size, c.extra.pages,
            c.extra.stride, c.istride, c.eistride};
}

Expected ReferenceTile(const mega_test::Case &c, const Stage1Config &p) {
    Expected r;
    for (int h = 0; h < HEAD_TILE; ++h) {
        for (int d = 0; d < DIM; ++d)
            r.q[h * DIM + d] = c.q[mega_test::q_offset(p.query, p.head_begin + h, d)];
    }
    r.rope = r.q;
    const float *cs = c.cos_sin.data() + p.position * ROPE_DIM;
    for (int h = 0; h < HEAD_TILE; ++h) {
        for (int j = 0; j < ROPE_DIM / 2; ++j) {
            int d = h * DIM + DIM - ROPE_DIM + 2 * j;
            float x = mega_test::from_bf16(r.q[d]), y = mega_test::from_bf16(r.q[d+1]);
            r.rope[d] = mega_test::bf16(std::fma(x, cs[j], -y * cs[32+j]));
            r.rope[d+1] = mega_test::bf16(std::fma(x, cs[32+j], y * cs[j]));
        }
    }
    for (int t = 0; t < KV_TILE; ++t) {
        int pos = p.kv_begin + t;
        bool extra = pos >= p.topk;
        int local = extra ? pos - p.topk : pos;
        const auto &cache = extra ? c.extra : c.main;
        const auto &ids = extra ? c.extra_indices : c.indices;
        int cap = extra ? p.extra_topk : p.topk;
        int length = extra ? p.extra_length : p.length;
        int stride = extra ? p.extra_indices_stride : p.indices_stride;
        if (local >= cap || local >= length) continue;
        int id = ids[size_t(p.query) * stride + local];
        // 有效长度内的 id 由 indexer 保证合法。
        r.valid[t] = 1;
        for (int d = 0; d < DIM; ++d)
            r.kv[t * DIM + d] = mega_test::bf16(cache.decoded[size_t(id) * DIM + d]);
    }
    for (int h = 0; h < HEAD_TILE; ++h) {
        for (int t = 0; t < KV_TILE; ++t) {
            double dot = 0;
            for (int d = 0; d < DIM; ++d)
                dot += double(mega_test::from_bf16(r.rope[h * DIM + d])) *
                       mega_test::from_bf16(r.kv[t * DIM + d]);
            r.raw[h * KV_TILE + t] = float(dot);
            if (r.valid[t]) r.masked[h * KV_TILE + t] = float(dot * SCORE_SCALE);
        }
    }
    return r;
}

void CompareFloats(const std::vector<float> &got, const std::vector<float> &ref,
                   float atol, float rtol, const char *name) {
    check(got.size() == ref.size(), "Wrong output size");
    for (size_t i = 0; i < ref.size(); ++i) {
        bool ok = std::isinf(ref[i]) ? got[i] == ref[i] :
            std::isfinite(got[i]) && std::abs(got[i]-ref[i]) <= atol + rtol*std::abs(ref[i]);
        if (!ok) throw std::runtime_error(std::string(name) + " mismatch at " +
            std::to_string(i) + ": got=" + std::to_string(got[i]) +
            ", expected=" + std::to_string(ref[i]));
    }
}

void CompareBf16(const std::vector<uint16_t> &got, const std::vector<uint16_t> &ref,
                 bool exact, const char *name) {
    check(got.size() == ref.size(), "Wrong BF16 output size");
    if (exact) {
        for (size_t i = 0; i < ref.size(); ++i)
            if (got[i] != ref[i]) throw std::runtime_error(std::string(name) +
                " BF16 bits mismatch at " + std::to_string(i));
    } else {
        std::vector<float> a(got.size()), b(ref.size());
        for (size_t i = 0; i < ref.size(); ++i) {
            a[i] = mega_test::from_bf16(got[i]); b[i] = mega_test::from_bf16(ref[i]);
        }
        // 允许跨平台 RoPE 的 FP32 运算顺序造成约一个 BF16 ULP 的差别。
        CompareFloats(a, b, 2e-5f, 8e-3f, name);
    }
}

template<class F> void ExpectFailure(F fn, const char *name) {
    bool threw = false;
    try { fn(); } catch (const std::runtime_error &) { threw = true; }
    check(threw, name);
}


// 独立参考：收集全部 KV 后一次性 softmax，不使用设备侧的 online 递推公式。
struct OnlineExpected {
    std::vector<float> output = std::vector<float>(Q_ELEMS, 0);
    std::vector<float> row_max = std::vector<float>(HEAD_TILE, -INFINITY);
    std::vector<float> row_sum = std::vector<float>(HEAD_TILE, 0);
};
OnlineExpected ReferenceOnline(const mega_test::Case &c, Stage1Config p) {
    std::vector<Expected> tiles;
    for (p.kv_begin = 0; p.kv_begin < p.topk + p.extra_topk; p.kv_begin += KV_TILE)
        tiles.push_back(ReferenceTile(c, p));
    OnlineExpected r;
    for (int h = 0; h < HEAD_TILE; ++h) {
        double m = -INFINITY, l = 0;
        std::vector<double> acc(DIM, 0);
        for (const auto &tile : tiles)
            for (int t = 0; t < KV_TILE; ++t)
                if (tile.valid[t]) m = std::max(m, double(tile.masked[h*KV_TILE+t]));
        if (m == -INFINITY) continue;
        for (const auto &tile : tiles) {
            for (int t = 0; t < KV_TILE; ++t) {
                if (!tile.valid[t]) continue;
                double weight = std::exp(double(tile.masked[h*KV_TILE+t])-m);
                l += weight;
                for (int d = 0; d < DIM; ++d)
                    acc[d] += weight * mega_test::from_bf16(tile.kv[t*DIM+d]);
            }
        }
        r.row_max[h] = float(m);
        r.row_sum[h] = float(l);
        for (int d = 0; d < DIM; ++d) r.output[h*DIM+d] = float(acc[d]/l);
    }
    return r;
}

void OnlineOracleTests() {
    mega_test::Case c(3,72,121,5);
    auto empty = ReferenceOnline(c,MakeConfig(c,0,0,0));
    for (float x : empty.output) check(x == 0, "Empty attention must be zero");
    for (float x : empty.row_max) check(x == -INFINITY, "Empty max");
    for (float x : empty.row_sum) check(x == 0, "Empty sum");
    // Q=0 使所有有效分数相同：输出必须等于 V 的平均，分母就是有效个数。
    std::fill(c.q.begin(),c.q.end(),mega_test::bf16(0));
    c.main.pack(std::vector<float>(c.main.decoded.size(),1));
    c.extra.pack(std::vector<float>(c.extra.decoded.size(),6));
    auto uniform = ReferenceOnline(c,MakeConfig(c,2,0,0));
    float mean = float(c.k+6*c.ek)/(c.k+c.ek);
    for (float x : uniform.output) check(std::abs(x-mean)<1e-6f, "Uniform attention");
    for (float x : uniform.row_max) check(x==0, "Uniform max");
    for (float x : uniform.row_sum) check(x==c.k+c.ek, "Uniform sum");
    // 后续块的最大值更大，另用两组常量的闭式权重核对参考结果。
    std::fill(c.q.begin(),c.q.end(),mega_test::bf16(.125f));
    std::fill(c.positions.begin(),c.positions.end(),0);
    c.main.pack(std::vector<float>(c.main.decoded.size(),.5f));
    c.extra.pack(std::vector<float>(c.extra.decoded.size(),.75f));
    auto rising = ReferenceOnline(c,MakeConfig(c,2,0,0));
    double a = std::exp(-16.0*SCORE_SCALE);
    double sum = c.k*a+c.ek;
    double out = (.5*c.k*a+.75*c.ek)/sum;
    for (float x : rising.output) check(std::abs(x-out)<1e-6, "Rising max output");
    for (float x : rising.row_sum) check(std::abs(x-sum)<2e-5, "Rising max sum");
    std::puts("PASS CPU: stage-2 full softmax oracle, empty/uniform/rising-max");
}

void OracleTests() {
    mega_test::cpu_tests();
    mega_test::Case c(3, 72, 121, 5);
    auto p = MakeConfig(c, 2, 32, 64);
    auto r = ReferenceTile(c, p);
    check(r.valid[0] == 1 && r.valid[7] == 1 && r.valid[8] == 1 && r.valid[9] == 1,
          "Mixed block FP8/FP4 boundary");
    for (int h = 0; h < HEAD_TILE; ++h)
        for (int d = 0; d < DIM; ++d)
            check(r.q[h*DIM+d] == c.q[mega_test::q_offset(2,32+h,d)], "Q offset");
    auto empty = ReferenceTile(c, MakeConfig(c, 0, 0, 0));
    for (float x : empty.masked) check(x == -INFINITY, "Empty tile mask");
    auto tail = ReferenceTile(c, MakeConfig(c, 2, 48, 192));
    for (int t = 1; t < KV_TILE; ++t) check(tail.valid[t] == 0, "Tile tail");
    // 独立的已知值：没有旋转，Q=1，FP8 KV=1，FP4 KV=6（可精确编码）。
    std::fill(c.q.begin(), c.q.end(), mega_test::bf16(1));
    std::fill(c.positions.begin(), c.positions.end(), 0);
    c.main.pack(std::vector<float>(c.main.decoded.size(), 1));
    c.extra.pack(std::vector<float>(c.extra.decoded.size(), 6));
    std::fill(c.indices.begin(), c.indices.end(), 0);
    std::fill(c.extra_indices.begin(), c.extra_indices.end(), 0);
    auto known = ReferenceTile(c, MakeConfig(c, 2, 16, 64));
    check(known.raw[0] == 512 && known.raw[8] == 3072, "Known QK dot products");
    check(known.masked[8] == 3072*SCORE_SCALE, "Known QK scale");
    auto broken = known.q; broken[0] ^= 1;
    ExpectFailure([&]{ CompareBf16(broken, known.q, true, "Q"); }, "Q checker");
    auto bad_scores = known.masked; bad_scores[1] += 1;
    ExpectFailure([&]{ CompareFloats(bad_scores, known.masked, 1e-3f, 1e-3f, "score"); }, "Score checker");
    auto bad_mask = empty.masked; bad_mask[0] = 0;
    ExpectFailure([&]{ CompareFloats(bad_mask, empty.masked, 0, 0, "mask"); }, "Mask checker");
    std::puts("PASS CPU: codecs, stage-1 Q/KV layouts, mixed/tail/empty tiles and QK oracle");
}
} // namespace

#ifndef MEGA_ASCEND_CPU_ONLY
#include <acl/acl.h>
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
void AclCheck(aclError code, const char *op) {
    if (code != ACL_SUCCESS) throw std::runtime_error(std::string(op) +
        " failed: " + std::to_string(code));
}

// 单卡上下文；容器里的逻辑 device 0 不必等于 npu-smi 显示的物理 NPU 编号。
struct Runtime {
    int device = 0;
    aclrtStream stream = nullptr;
    Runtime() {
        if (const char *s = std::getenv("ASCEND_DEVICE_ID")) device = std::stoi(s);
        std::fprintf(stderr, "NPU: aclInit\n");
        AclCheck(aclInit(nullptr), "aclInit");
        std::fprintf(stderr, "NPU: aclrtSetDevice(%d)\n", device);
        AclCheck(aclrtSetDevice(device), "aclrtSetDevice");
        std::fprintf(stderr, "NPU: aclrtCreateStream\n");
        AclCheck(aclrtCreateStream(&stream), "aclrtCreateStream");
        std::fprintf(stderr, "NPU: runtime ready\n");
    }
    ~Runtime() {
        if (stream) aclrtDestroyStream(stream);
        aclrtResetDevice(device); aclFinalize();
    }
};

// 分配首尾各 512 B 的保护区；同步后检查，定位越界写入。
template<class T> struct DeviceBuffer {
    void *base = nullptr;
    T *data = nullptr;
    size_t count;
    explicit DeviceBuffer(size_t n) : count(n) {
        AclCheck(aclrtMalloc(&base, n*sizeof(T)+1024, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
        data = reinterpret_cast<T *>(static_cast<uint8_t *>(base)+512);
        AclCheck(aclrtMemset(base, n*sizeof(T)+1024, 0xa5, n*sizeof(T)+1024), "aclrtMemset");
    }
    explicit DeviceBuffer(const std::vector<T> &v) : DeviceBuffer(v.size()) { Upload(v); }
    ~DeviceBuffer() { if (base) aclrtFree(base); }
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
    void Upload(const std::vector<T> &v) {
        check(v.size() == count, "Upload size mismatch");
        AclCheck(aclrtMemcpy(data, count*sizeof(T), v.data(), count*sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE), "H2D");
    }
    std::vector<T> Download() const {
        std::vector<T> v(count);
        AclCheck(aclrtMemcpy(v.data(), count*sizeof(T), data, count*sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST), "D2H");
        return v;
    }
    void CheckGuard() const {
        std::array<uint8_t,512> bytes{};
        for (const void *p : {base, static_cast<void *>(data+count)}) {
            AclCheck(aclrtMemcpy(bytes.data(), bytes.size(), p, bytes.size(), ACL_MEMCPY_DEVICE_TO_HOST), "guard D2H");
            for (uint8_t v : bytes) check(v == 0xa5, "Device guard overwritten");
        }
    }
};

// 在 Host 生成固定 QK 的 tiling；GM 输入输出让阶段 1 的观测点可直接复用。
std::vector<uint8_t> MakeCubeTiling(platform_ascendc::PlatformAscendC &platform, bool pv = false, int ubBytes = 64*1024) {
    using namespace matmul_tiling;
    MatmulApiTiling api(platform);
    api.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_BF16, false);
    api.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_BF16, !pv);
    api.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT);
    api.SetBias(false);
    int n = pv ? DIM : KV_TILE, k = pv ? KV_TILE : DIM;
    api.SetShape(HEAD_TILE, n, k);
    api.SetOrgShape(HEAD_TILE, n, k);
    // 剩余 UB 留给约 118 KiB 的用户缓冲；不要默认把全部 UB 交给 Matmul。
    api.SetBufferSpace(-1, -1, ubBytes);
    optiling::TCubeTiling tiling;
    check(api.GetTiling(tiling) != -1, "QK tiling failed");
    std::vector<uint8_t> bytes(tiling.GetDataSize());
    tiling.SaveToBuffer(bytes.data(), bytes.size());
    return bytes;
}

void RunCase(Runtime &rt, const char *name, const mega_test::Case &c,
             const Stage1Config &p, void *tiling, void *workspace) {
    std::fprintf(stderr, "NPU [%s]: prepare inputs\n", name);
    auto ref = ReferenceTile(c,p);
    DeviceBuffer<uint16_t> q(c.q), q_loaded(Q_ELEMS), q_rope(Q_ELEMS), kv(KV_ELEMS);
    DeviceBuffer<uint8_t> main(c.main.bytes), extra(c.extra.bytes);
    DeviceBuffer<int32_t> ids(c.indices), extra_ids(c.extra_indices), valid(KV_TILE);
    DeviceBuffer<float> rope(c.cos_sin), scores(SCORE_ELEMS);
    DeviceBuffer<int32_t> status(std::vector<int32_t>(8,-1));
    DeviceBuffer<Stage1Config> config(std::vector<Stage1Config>{p});
    Stage1Input in{q.data,main.data,extra.data,ids.data,extra_ids.data,rope.data};
    Stage1Output out{q_loaded.data,q_rope.data,kv.data,valid.data,scores.data,status.data};
    std::fprintf(stderr, "NPU [%s]: launch kernel\n", name);
    launch_mega_attn_stage1(in,out,config.data,tiling,workspace,rt.stream);
    std::fprintf(stderr, "NPU [%s]: wait kernel (30 s timeout)\n", name);
    aclError syncStatus = aclrtSynchronizeStreamWithTimeout(rt.stream, 30000);
    // 先输出错误，再清理资源，避免清理也阻塞时丢失定位信息。
    if (syncStatus != ACL_SUCCESS)
        std::fprintf(stderr, "NPU [%s]: synchronize failed, code=%d\n", name, int(syncStatus));
    AclCheck(syncStatus, "kernel synchronize");
    std::fprintf(stderr, "NPU [%s]: kernel complete, checking outputs\n", name);
    q.CheckGuard(); main.CheckGuard(); extra.CheckGuard(); ids.CheckGuard();
    extra_ids.CheckGuard(); rope.CheckGuard(); config.CheckGuard();
    q_loaded.CheckGuard(); q_rope.CheckGuard(); kv.CheckGuard(); valid.CheckGuard();
    scores.CheckGuard(); status.CheckGuard();
    int step = status.Download()[0];
    check(step >= 0 && step <= 6, "Missing/invalid device completion status");
    int completed = step == 0 ? 6 : step-1;
    if (completed >= 1) CompareBf16(q_loaded.Download(),ref.q,true,"LoadQuery");
    if (completed >= 2) CompareBf16(q_rope.Download(),ref.rope,false,"RoPE");
    if (completed >= 3) check(valid.Download() == ref.valid,"KV valid mask mismatch");
    if (completed >= 4) CompareBf16(kv.Download(),ref.kv,true,"KV decode");
    // QK 分数使用独立 CPU RoPE 结果，容纳一次 BF16 舍入差异的传播。
    if (completed >= 5) CompareFloats(scores.Download(), step == 6 ? ref.raw : ref.masked,
                                     step == 6 ? .03f : .0015f, .01f, "QK scores");
    std::printf("%s: verified %d/6 steps\n",name,completed);
    if (step != 0) throw std::runtime_error(std::string("TODO stage1: ")+StepName(step));
    std::printf("PASS NPU stage1: %s\n",name);
}


void RunOnlineCase(Runtime &rt, const char *name, const mega_test::Case &c,
                   const Stage1Config &p, void *qkTiling, void *pvTiling, void *workspace) {
    auto ref = ReferenceOnline(c,p);
    DeviceBuffer<uint16_t> q(c.q), qLoaded(Q_ELEMS), qRope(Q_ELEMS), kv(KV_ELEMS), prob(SCORE_ELEMS);
    DeviceBuffer<uint8_t> main(c.main.bytes), extra(c.extra.bytes);
    DeviceBuffer<int32_t> ids(c.indices), extraIds(c.extra_indices), valid(KV_TILE);
    DeviceBuffer<float> rope(c.cos_sin), scores(SCORE_ELEMS), pv(Q_ELEMS), output(Q_ELEMS);
    DeviceBuffer<float> rowMax(HEAD_TILE), rowSum(HEAD_TILE);
    DeviceBuffer<int32_t> status(std::vector<int32_t>(8,-1));
    DeviceBuffer<Stage1Config> config(std::vector<Stage1Config>{p});
    Stage1Input in{q.data,main.data,extra.data,ids.data,extraIds.data,rope.data};
    Stage1Output scratch{qLoaded.data,qRope.data,kv.data,valid.data,scores.data,status.data};
    Stage2Output out{prob.data,pv.data,output.data,rowMax.data,rowSum.data};
    std::fprintf(stderr,"NPU stage2 [%s]: launch + wait (30 s timeout)\n",name);
    launch_mega_attn_stage2(in,scratch,out,config.data,qkTiling,pvTiling,workspace,rt.stream);
    auto sync = aclrtSynchronizeStreamWithTimeout(rt.stream,30000);
    if (sync != ACL_SUCCESS) std::fprintf(stderr,"NPU stage2 synchronize failed: %d\n",int(sync));
    AclCheck(sync,"stage2 synchronize");
    q.CheckGuard(); main.CheckGuard(); extra.CheckGuard(); ids.CheckGuard(); extraIds.CheckGuard();
    rope.CheckGuard(); config.CheckGuard(); qLoaded.CheckGuard(); qRope.CheckGuard(); kv.CheckGuard();
    valid.CheckGuard(); scores.CheckGuard(); status.CheckGuard(); prob.CheckGuard(); pv.CheckGuard();
    output.CheckGuard(); rowMax.CheckGuard(); rowSum.CheckGuard();
    int step = status.Download()[0];
    check(step==0 || (step>=1 && step<=6) || (step>=10 && step<=14), "Stage2 completion status");
    if (step==int(Step::OnlineSoftmax)) {
        // 此时第一块的 softmax 尚未实现，导出的状态应仍为初始化值。
        CompareFloats(rowMax.Download(),std::vector<float>(HEAD_TILE,-INFINITY),0,0,"Initial max");
        CompareFloats(rowSum.Download(),std::vector<float>(HEAD_TILE,0),0,0,"Initial sum");
        CompareFloats(output.Download(),std::vector<float>(Q_ELEMS,0),0,0,"Initial accumulator");
        std::puts("PASS NPU: InitOnlineState");
    }
    if (step!=0) throw std::runtime_error(std::string("TODO stage2: ")+StepName(step));
    // P 舍入为 BF16 后进入 Cube；参考用 FP64 softmax，容纳该舍入与 RoPE 误差。
    CompareFloats(output.Download(),ref.output,.002f,.02f,"Online attention");
    CompareFloats(rowMax.Download(),ref.row_max,.002f,.01f,"Online max");
    CompareFloats(rowSum.Download(),ref.row_sum,.03f,.01f,"Online sum");
    if (p.length==0 && p.extra_length==0)
        CompareFloats(output.Download(),ref.output,0,0,"Empty attention");
    std::printf("PASS NPU stage2: %s\n",name);
}

void NpuTests() {
    Runtime rt;
    std::fprintf(stderr, "NPU: query platform\n");
    auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    check(platform != nullptr,"Cannot get Ascend platform");
    std::fprintf(stderr, "NPU: create and upload tiling\n");
    DeviceBuffer<uint8_t> tiling(MakeCubeTiling(*platform));
    std::fprintf(stderr, "NPU: allocate workspace\n");
    DeviceBuffer<uint8_t> workspace(platform->GetLibApiWorkSpaceSize());
    std::fprintf(stderr, "NPU: generate test inputs\n");
    mega_test::Case c(3,72,121,65,17);
    RunCase(rt,"FP8",c,MakeConfig(c,2,0,0),tiling.data,workspace.data);
    RunCase(rt,"mixed FP8/FP4",c,MakeConfig(c,2,16,64),tiling.data,workspace.data);
    RunCase(rt,"FP4",c,MakeConfig(c,2,32,128),tiling.data,workspace.data);
    RunCase(rt,"tail",c,MakeConfig(c,2,48,192),tiling.data,workspace.data);
    RunCase(rt,"empty",c,MakeConfig(c,0,0,0),tiling.data,workspace.data);
    RunCase(rt,"short length",c,MakeConfig(c,1,0,0),tiling.data,workspace.data);
    RunCase(rt,"short extra length",c,MakeConfig(c,1,16,64),tiling.data,workspace.data);
    mega_test::Case small(1,8,9,5);
    RunCase(rt,"small page + mixed tail",small,MakeConfig(small,0,0,0),tiling.data,workspace.data);
    std::puts("PASS: stage 1 regression (8 cases)");
    // 用户 UB 约 153 KiB；QK/PV 各限 16 KiB，合计不超过 910B 的 192 KiB。
    DeviceBuffer<uint8_t> qkTiling(MakeCubeTiling(*platform,false,16*1024));
    DeviceBuffer<uint8_t> pvTiling(MakeCubeTiling(*platform,true,16*1024));
    auto run = [&](const char *name, const mega_test::Case &input, int query, int head) {
        RunOnlineCase(rt,name,input,MakeConfig(input,query,head,0),
                      qkTiling.data,pvTiling.data,workspace.data);
    };
    run("mixed + tail",c,2,0);
    run("short lengths",c,1,16);
    run("all empty",c,0,32);
    c.lengths[2]=0;
    run("empty prefix then FP4",c,2,48);
    c.lengths[2]=13; c.extra_lengths[2]=0;
    run("empty suffix after FP8",c,2,0);
    c.lengths[2]=1;
    run("single valid KV",c,2,16);
    run("single tile + small pages",small,0,0);
    mega_test::Case rising(1,128,65,5);
    std::fill(rising.q.begin(),rising.q.end(),mega_test::bf16(.125f));
    rising.main.pack(std::vector<float>(rising.main.decoded.size(),.5f));
    rising.extra.pack(std::vector<float>(rising.extra.decoded.size(),.75f));
    run("maximum rises in later tile",rising,0,0);
    std::fill(rising.q.begin(),rising.q.end(),mega_test::bf16(16));
    run("large positive scores",rising,0,0);
    qkTiling.CheckGuard(); pvTiling.CheckGuard();
    tiling.CheckGuard(); workspace.CheckGuard();
}
} // namespace
#endif

int main() {
    try {
        OracleTests();
        OnlineOracleTests();
#ifdef MEGA_ASCEND_CPU_ONLY
        std::puts("CPU oracle only: NPU compilation/execution NOT tested; learner TODOs remain.");
#else
        NpuTests();
        std::puts("PASS: stage 2 online attention (sink/inverse RoPE/quantization remain)");
#endif
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr,"FAIL: %s\n",e.what());
        return 1;
    }
}
