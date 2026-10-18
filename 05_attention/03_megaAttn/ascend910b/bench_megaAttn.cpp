// 复用原 benchmark 的 CPU 数据生成/编解码；不编译 CUDA 部分，也不复制一套 oracle。
#define MEGA_ATTN_REFERENCE_ONLY
#include "../sm100/bench_megaAttn.cu"
#undef MEGA_ATTN_REFERENCE_ONLY
#include "megaAttn_v1.h"
#include "megaAttn_v2.h"
#include <cstdio>
#include <cstdlib>

using namespace mega_ascend_v1;
using mega_test::check;

namespace {
// CPU 参考逐块构造 QK，再收集全部 KV 做独立的完整 softmax。
struct TileConfig : TaskConfig {
    int32_t kv_begin = 0;
};

// CPU 单块参考值；不要求设备导出任何中间张量。
struct Expected {
    std::vector<uint16_t> q = std::vector<uint16_t>(Q_ELEMS);
    std::vector<uint16_t> rope = std::vector<uint16_t>(Q_ELEMS);
    std::vector<uint16_t> kv = std::vector<uint16_t>(KV_ELEMS, 0);
    std::vector<int32_t> valid = std::vector<int32_t>(KV_TILE, 0);
    std::vector<float> raw = std::vector<float>(SCORE_ELEMS, 0);
    std::vector<float> masked = std::vector<float>(SCORE_ELEMS, -INFINITY);
};

TileConfig MakeConfig(const mega_test::Case &c, int q, int head, int begin) {
    check(q >= 0 && q < c.n && head >= 0 && head + HEAD_TILE <= HEADS &&
          head % HEAD_TILE == 0 && begin >= 0 &&
          begin % KV_TILE == 0, "Invalid tile coordinates");
    return {{q, head, c.positions[q], c.k, c.ek, c.lengths[q], c.extra_lengths[q],
             c.main.size, c.main.stride, c.extra.size, c.extra.stride, c.istride, c.eistride}, begin};
}

Expected ReferenceTile(const mega_test::Case &c, const TileConfig &p) {
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
OnlineExpected ReferenceOnline(const mega_test::Case &c, TileConfig p) {
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
    std::puts("PASS CPU: full softmax oracle, empty/uniform/rising-max");
}


// 末处理的 CPU 参考；有效 softmax 分母用双精度计算，空行沿用原版 LSE=+inf。
struct FinalExpected {
    std::vector<float> output, lse;
};
FinalExpected ReferenceFinal(const mega_test::Case &c, const TileConfig &p, bool rotate) {
    auto online = ReferenceOnline(c,p);
    FinalExpected r{online.output,std::vector<float>(HEAD_TILE,INFINITY)};
    for (int h=0; h<HEAD_TILE; ++h) {
        if (online.row_sum[h]==0) continue;
        double lse = double(online.row_max[h])+std::log(double(online.row_sum[h]));
        r.lse[h]=float(lse);
        double sink = c.use_sink ? c.sink[p.head_begin+h] : -INFINITY;
        // logaddexp 求含 sink 的总分母，独立于设备侧的分段 sigmoid 写法。
        double gate=0;
        if (sink!=INFINITY) {
            double m=std::max(lse,sink);
            gate=std::exp(lse-m)/(std::exp(lse-m)+std::exp(sink-m));
        }
        for (int d=0; d<DIM; ++d) r.output[h*DIM+d]*=float(gate);
        if (!rotate) continue;
        const float *cs=c.cos_sin.data()+p.position*ROPE_DIM;
        for (int j=0; j<ROPE_DIM/2; ++j) {
            int d=h*DIM+DIM-ROPE_DIM+2*j;
            float a=r.output[d], b=r.output[d+1];
            r.output[d]=std::fma(a,cs[j],b*cs[32+j]);
            r.output[d+1]=std::fma(-a,cs[32+j],b*cs[j]);
        }
    }
    return r;
}

// 从量化前的实际 FP32 输出核对字节和布局，避免把 BF16 attention 误差误判为编码错误。
void PackFinalTile(const std::vector<float> &input, int q, int head, int stride,
                   std::vector<uint8_t> &data, std::vector<uint32_t> &scales) {
    for (int h=0; h<HEAD_TILE; ++h) {
        int gh=head+h;
        for (int group=0; group<DIM/32; ++group) {
            float amax=0;
            for (int j=0; j<32; ++j) amax=std::max(amax,std::abs(input[h*DIM+group*32+j]));
            float required=std::max(amax,1e-4f)/448.f;
            int e=int(std::ceil(std::log2(double(required))));
            float scale=std::ldexp(1.f,e);
            int slot=group*8+gh%8;
            size_t word=size_t((gh/8)*32+slot/4)*stride+q;
            int shift=8*(slot%4);
            scales[word]=(scales[word]&~(0xffu<<shift))|(uint32_t(e+127)<<shift);
            for (int j=0; j<32; ++j)
                data[mega_test::out_offset(q,gh,group*32+j)]=
                    mega_test::to_e4m3(input[h*DIM+group*32+j]/scale);
        }
    }
}

void FinalOracleTests() {
    mega_test::Case c(3,8,9,5);
    std::fill(c.q.begin(),c.q.end(),mega_test::bf16(0));
    c.main.pack(std::vector<float>(c.main.decoded.size(),6));
    c.extra.pack(std::vector<float>(c.extra.decoded.size(),6));
    std::fill(c.sink.begin(),c.sink.end(),std::log(float(c.k+c.ek)));
    auto p=MakeConfig(c,2,0,0);
    auto half=ReferenceFinal(c,p,false);
    for (float x:half.output) check(std::abs(x-3.f)<1e-6f,"Sink doubles denominator");
    for (float x:half.lse) check(std::abs(x-std::log(float(c.k+c.ek)))<1e-6f,"LSE excludes sink");
    c.sink[0]=-INFINITY; c.sink[1]=INFINITY; c.sink[2]=1000; c.sink[3]=-1000;
    auto extremes=ReferenceFinal(c,p,false);
    for (int d=0; d<DIM; ++d) {
        check(extremes.output[d]==6 && extremes.output[DIM+d]==0,"Infinite sink");
        check(extremes.output[2*DIM+d]==0 && extremes.output[3*DIM+d]==6,"Stable finite sink");
    }
    auto empty=ReferenceFinal(c,MakeConfig(c,0,0,0),true);
    for (float x:empty.lse) check(x==INFINITY,"Empty final LSE");
    for (float x:empty.output) check(x==0,"Empty final output");
    c.use_sink=false;
    auto r=ReferenceFinal(c,p,true);
    auto original=mega_test::reference(c);
    size_t begin=size_t(p.query)*HEADS*DIM;
    CompareFloats(r.output,{original.out.begin()+begin,original.out.begin()+begin+Q_ELEMS},1e-6f,1e-6f,"Original final semantics");
    // 用可精确编码的正负常量验证 scale 取整、跨 head/query 索引与 padding。
    std::vector<float> input(Q_ELEMS);
    std::vector<uint8_t> data(size_t(c.n)*HEADS*DIM,0xa5);
    std::vector<uint32_t> scales(256*c.sfstride,0xa5a5a5a5u);
    for (int h=0; h<HEAD_TILE; ++h)
        for (int d=0; d<DIM; ++d) input[h*DIM+d]=(h%2?-1.f:1.f)*std::ldexp(448.f,(d/32)%5-4);
    PackFinalTile(input,2,16,c.sfstride,data,scales);
    auto unpacked=mega_test::unpack_output(data,scales,c.n,c.sfstride);
    begin=(size_t(2)*HEADS+16)*DIM;
    CompareFloats({unpacked.begin()+begin,unpacked.begin()+begin+Q_ELEMS},input,0,0,"Output packing");
    for (int word=0; word<256; ++word)
        for (int q=0; q<c.sfstride; ++q)
            if (word<64 || word>=128 || q!=2)
                check(scales[word*c.sfstride+q]==0xa5a5a5a5u,"Scale writes outside tile");
    std::fill(input.begin(),input.end(),0);
    PackFinalTile(input,0,0,c.sfstride,data,scales);
    check(scales[0]==0x69696969u,"Zero amax scale must be 2^-22 (UE8M0=105)");
    std::puts("PASS CPU: sink/LSE, inverse RoPE, FP8/UE8M0 and output layouts");
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
    std::puts("PASS CPU: codecs, Q/KV layouts, mixed/tail/empty tiles and QK oracle");
}
// 穷举设备查表的全部编码，对照原有独立 codec，包含非规格化数和 NaN。
void V2CodecTests() {
    for (int code=0;code<256;++code) {
        float got=std::bit_cast<float>(mega_ascend_v2::E4M3Bits(uint8_t(code)));
        float ref=mega_test::e4m3(uint8_t(code));
        check(std::isnan(ref)?std::isnan(got):std::bit_cast<uint32_t>(got)==std::bit_cast<uint32_t>(ref),
              "v2 E4M3 table bits");
        got=std::bit_cast<float>(mega_ascend_v2::E8M0Bits(uint8_t(code)));
        ref=code==255?NAN:std::ldexp(1.0f,code-127);
        check(std::isnan(ref)?std::isnan(got):got==ref,"v2 UE8M0 table bits");
    }
    // v2 的 NZ 布局应与原始 Q 布局相同；所有 64x512 元素各占唯一位置。
    std::vector<bool> occupied(64*512);
    for (int h=0;h<64;++h) for (int d=0;d<512;++d) {
        size_t nz=(d/16)*64*16+h*16+d%16;
        check(nz==mega_test::q_offset(0,h,d) && !occupied[nz],"v2 NZ layout");
        occupied[nz]=true;
    }
    std::printf("PASS CPU v2: exhaustive codec tables; NZ layout; UB=%d+%d bytes, L1=200 KiB\n",
                mega_ascend_v2::USER_UB_BYTES,2*mega_ascend_v2::MATMUL_UB_BYTES);
}

template<class F> void ForEachCase(F run) {
    mega_test::Case mixed(3,72,121,65,17);
    run("mixed/tail/empty",mixed);
    mixed.lengths[2]=0; mixed.use_sink=false;
    run("empty prefix + null sink",mixed);
    mixed.lengths[2]=mixed.k; mixed.extra_lengths[2]=0;
    run("empty suffix",mixed);
    mega_test::Case small(1,8,9,5);
    for (int h=0; h<HEADS; ++h) {
        const float sinks[4]={INFINITY,-INFINITY,10000.0f,-10000.0f};
        small.sink[h]=sinks[h%4];
    }
    run("finite/infinite sinks",small);
    small.main.pack(std::vector<float>(small.main.decoded.size(),0));
    small.extra.pack(std::vector<float>(small.extra.decoded.size(),0));
    run("single tile + zero output",small);
    mega_test::Case rising(1,128,65,5);
    std::fill(rising.q.begin(),rising.q.end(),mega_test::bf16(16));
    rising.main.pack(std::vector<float>(rising.main.decoded.size(),.5f));
    rising.extra.pack(std::vector<float>(rising.extra.decoded.size(),.75f));
    run("rising maximum + large scores",rising);
    // 生成器沿用 CUDA 的非空容量限制；生成后收缩逻辑容量，保留合法分配。
    mega_test::Case noKv(1,8,9,5);
    noKv.k=0; noKv.ek=0; noKv.lengths[0]=0; noKv.extra_lengths[0]=0;
    run("zero KV capacity",noKv);
    mega_test::Case fp8Only(1,72,9,5);
    fp8Only.ek=0; fp8Only.extra_lengths[0]=0;
    run("FP8 only",fp8Only);
    mega_test::Case fp4Only(1,8,65,5);
    fp4Only.k=0; fp4Only.lengths[0]=0;
    run("FP4 only",fp4Only);
    mega_test::Case throughput(8,128,65,65,17);
    std::fill(throughput.lengths.begin(),throughput.lengths.end(),throughput.k);
    std::fill(throughput.extra_lengths.begin(),throughput.extra_lengths.end(),throughput.ek);
    run("full batch",throughput,true);
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

// 固定 QK/PV 形状：v1 从 GM 读 ND，v2 从 L1 读 NZ；结果均写 GM。
std::vector<uint8_t> MakeCubeTiling(platform_ascendc::PlatformAscendC &platform, bool pv = false, bool v2 = false) {
    using namespace matmul_tiling;
    MatmulApiTiling api(platform);
    auto pos=v2?TPosition::TSCM:TPosition::GM;
    auto format=v2?CubeFormat::NZ:CubeFormat::ND;
    api.SetAType(pos, format, DataType::DT_BF16, false);
    api.SetBType(pos, format, DataType::DT_BF16, !pv);
    api.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT);
    api.SetBias(false);
    int tile=v2?mega_ascend_v2::KV_TILE:KV_TILE;
    int n = pv ? DIM : tile, k = pv ? tile : DIM;
    int heads=v2?HEADS:HEAD_TILE;
    api.SetShape(heads, n, k);
    api.SetOrgShape(heads, n, k);
    // v2 的用户 L1 占 200 KiB，两个 Matmul 各留 64 KiB；UB 预算见 v2.h。
    api.SetBufferSpace(v2?64*1024:-1, -1, v2?mega_ascend_v2::MATMUL_UB_BYTES:16*1024);
    optiling::TCubeTiling tiling;
    check(api.GetTiling(tiling) != -1, pv?"PV tiling failed":"QK tiling failed");
    std::vector<uint8_t> bytes(tiling.GetDataSize());
    tiling.SaveToBuffer(bytes.data(), bytes.size());
    return bytes;
}

// 输出量化会在舍入边界改变字节；按每组 amax/16 的 E4M3 半步误差检查数值。
void CompareQuantized(const mega_test::Case &c, const mega_test::Reference &ref,
                      const std::vector<uint8_t> &packed, const std::vector<uint32_t> &scales,
                      const std::vector<float> &lse) {
    auto got=mega_test::unpack_output(packed,scales,c.n,c.sfstride);
    for (size_t base=0; base<ref.out.size(); base+=32) {
        float amax=0;
        for (int j=0; j<32; ++j) amax=std::max(amax,std::abs(ref.out[base+j]));
        for (int j=0; j<32; ++j) {
            float want=ref.out[base+j], value=got[base+j];
            float tolerance=.003f+.025f*std::abs(want)+amax/16;
            if (!std::isfinite(value) || std::abs(value-want)>tolerance)
                throw std::runtime_error("quantized output mismatch at "+std::to_string(base+j));
        }
    }
    CompareFloats(lse,ref.lse,.003f,.002f,"reference LSE");
    for (int q=0; q<c.n; ++q) for (int h=0; h<HEADS; ++h) {
        bool zero=(c.lengths[q]==0 && c.extra_lengths[q]==0) || (c.use_sink && c.sink[h]==INFINITY);
        if (zero) for (int d=0; d<DIM; ++d)
            check(got[(size_t(q)*HEADS+h)*DIM+d]==0,"Empty/sink-inf output must be zero");
    }
    for (int word=0; word<256; ++word) for (int q=c.n; q<c.sfstride; ++q)
        check(scales[word*c.sfstride+q]==0xa5a5a5a5u,"Scale padding overwritten");
}

// 一次样本计时包含 5 个完整 batch；不含分配、H2D、D2H 和 CPU 对拍。
struct DeviceTimer {
    aclrtEvent begin=nullptr, end=nullptr;
    DeviceTimer() {
        AclCheck(aclrtCreateEventWithFlag(&begin,ACL_EVENT_TIME_LINE),"create timing event");
        AclCheck(aclrtCreateEventWithFlag(&end,ACL_EVENT_TIME_LINE),"create timing event");
    }
    ~DeviceTimer() { if (end) aclrtDestroyEvent(end); if (begin) aclrtDestroyEvent(begin); }
};
template<class F> void MeasureBatch(Runtime &rt,const char *name,int queries,F launch) {
    constexpr int warmup=2, repeats=5, samples=3;
    for (int i=0; i<warmup; ++i) launch();
    AclCheck(aclrtSynchronizeStreamWithTimeout(rt.stream,30000),"benchmark warmup");
    DeviceTimer timer;
    std::array<float,samples> times{};
    for (int sample=0; sample<samples; ++sample) {
        AclCheck(aclrtRecordEvent(timer.begin,rt.stream),"record begin");
        for (int i=0; i<repeats; ++i) launch();
        AclCheck(aclrtRecordEvent(timer.end,rt.stream),"record end");
        AclCheck(aclrtSynchronizeStreamWithTimeout(rt.stream,30000),"benchmark synchronize");
        AclCheck(aclrtEventElapsedTime(&times[sample],timer.begin,timer.end),"event elapsed time");
        times[sample]/=repeats;
    }
    std::sort(times.begin(),times.end());
    float ms=times[samples/2];
    check(ms>0 && std::isfinite(ms),"Invalid benchmark duration");
    std::printf("BENCH %-20s batch_us=%10.3f query/s=%12.1f (median of %d x %d)\n",
                name,ms*1000,double(queries)*1000/ms,samples,repeats);
}

// v2 与同一输入的 v1/独立 CPU 参考对拍；核数变化仍须保持逐字节一致。
void RunV2Case(Runtime &rt,const char *name,const mega_test::Case &c,uint32_t availableCores,
               const Input &in,const Output &out,DeviceBuffer<uint8_t> &packed,
               DeviceBuffer<uint32_t> &scales,DeviceBuffer<float> &lse,
               void *qkTiling,void *pvTiling,void *workspace,bool benchmark) {
    auto ref=mega_test::reference(c);
    auto v1Lse=lse.Download();
    auto v1Output=mega_test::unpack_output(packed.Download(),scales.Download(),c.n,c.sfstride);
    std::vector<mega_ascend_v2::QueryConfig> queries;
    for (int q=0;q<c.n;++q) queries.push_back({q,c.positions[q],c.k,c.ek,c.lengths[q],c.extra_lengths[q],
        c.main.size,c.main.stride,c.extra.size,c.extra.stride,c.istride,c.eistride});
    uint32_t maxBlocks=std::min(uint32_t(c.n),availableCores);
    DeviceBuffer<mega_ascend_v2::QueryConfig> configs(queries);
    DeviceBuffer<uint8_t> scratch(mega_ascend_v2::ScratchBytes(maxBlocks));
    DeviceBuffer<mega_ascend_v2::Profile> profiles(c.n);
    auto launch=[&](uint32_t blocks,mega_ascend_v2::Profile *profile=nullptr) {
        launch_mega_attn_v2(in,out,configs.data,c.n,blocks,scratch.data,qkTiling,pvTiling,workspace,rt.stream,profile);
    };
    std::vector<uint32_t> blockCounts{1};
    for (uint32_t n:{2u,5u,maxBlocks})
        if (n<=maxBlocks && std::find(blockCounts.begin(),blockCounts.end(),n)==blockCounts.end()) blockCounts.push_back(n);
    std::vector<uint8_t> serialPacked;
    std::vector<uint32_t> serialScales;
    std::vector<float> serialLse;
    for (uint32_t blocks:blockCounts) {
        packed.Upload(std::vector<uint8_t>(packed.count,0xa5));
        scales.Upload(std::vector<uint32_t>(scales.count,0xa5a5a5a5u));
        lse.Upload(std::vector<float>(lse.count,NAN));
        std::fprintf(stderr,"NPU v2 [%s]: queries=%d groups=%u\n",name,c.n,blocks);
        launch(blocks);
        AclCheck(aclrtSynchronizeStreamWithTimeout(rt.stream,30000),"v2 synchronize");
        auto data=packed.Download(); auto sf=scales.Download(); auto ls=lse.Download();
        CompareQuantized(c,ref,data,sf,ls);
        CompareFloats(ls,v1Lse,.003f,.002f,"v2 vs v1 LSE");
        auto values=mega_test::unpack_output(data,sf,c.n,c.sfstride);
        for (size_t base=0;base<values.size();base+=32) {
            float amax=0;
            for (int j=0;j<32;++j) amax=std::max(amax,std::abs(ref.out[base+j]));
            for (int j=0;j<32;++j)
                check(std::abs(values[base+j]-v1Output[base+j])<=.006f+.05f*std::abs(ref.out[base+j])+amax/8,
                      "v2 vs v1 quantized output");
        }
        if (blocks==1) { serialPacked=data; serialScales=sf; serialLse=ls; }
        else {
            check(data==serialPacked && sf==serialScales,"v2 changed with block count");
            CompareFloats(ls,serialLse,0,0,"v2 multi-core LSE");
        }
        packed.CheckGuard(); scales.CheckGuard(); lse.CheckGuard(); scratch.CheckGuard();
        std::printf("PASS NPU v2: %s queries=%d groups=%u\n",name,c.n,blocks);
    }
    if (benchmark) {
        std::printf("BENCH v2: same input as v1; queries=%d groups=%u, 64 heads/group\n",c.n,maxBlocks);
        MeasureBatch(rt,"v2",c.n,[&] { launch(maxBlocks); });
        // 与无计时版本逐字节比较，保证诊断开关没有改变结果。
        launch(maxBlocks,profiles.data);
        AclCheck(aclrtSynchronizeStreamWithTimeout(rt.stream,30000),"v2 profile synchronize");
        check(packed.Download()==serialPacked && scales.Download()==serialScales,"v2 profiling changed outputs");
        CompareFloats(lse.Download(),serialLse,0,0,"v2 profiling LSE");
        auto samples=profiles.Download();
        const char *labels[]={"query","prepare-kv","wait-qk/load","softmax","wait-pv","accumulate","finish","total"};
        std::puts("PROFILE v2: AIV cycles/query, includes waits; asynchronous work overlaps; excludes InitTables");
        for (int part=0;part<mega_ascend_v2::ProfileCount;++part) {
            double sum=0; for (const auto &sample:samples) sum+=sample.cycles[part];
            std::printf("  %-14s %.0f\n",labels[part],sum/c.n);
        }
    }
    configs.CheckGuard(); scratch.CheckGuard(); profiles.CheckGuard();
}

// 同一组输入分别用一个核组和多个核组运行，检查跨任务复用和核间隔离。
void RunNpuCase(Runtime &rt, const char *name, const mega_test::Case &c, uint32_t availableCores,
                void *qkTiling, void *pvTiling, void *v2Qk, void *v2Pv, void *workspace, bool benchmark=false) {
    auto ref=mega_test::reference(c);
    std::vector<TaskConfig> tasks;
    for (int query=0; query<c.n; ++query)
        for (int head=0; head<HEADS; head+=HEAD_TILE)
            tasks.push_back(MakeConfig(c,query,head,0));
    uint32_t taskCount=uint32_t(tasks.size());
    uint32_t maxBlocks=std::min(taskCount,availableCores);
    check(maxBlocks>0,"No available AIC cores");
    DeviceBuffer<uint16_t> q(c.q);
    DeviceBuffer<uint8_t> main(c.main.bytes), extra(c.extra.bytes), packed(size_t(c.n)*HEADS*DIM);
    DeviceBuffer<int32_t> ids(c.indices), extraIds(c.extra_indices);
    DeviceBuffer<float> rope(c.cos_sin), sink(c.sink), lse(size_t(c.n)*HEADS);
    DeviceBuffer<uint32_t> scales(256*c.sfstride);
    DeviceBuffer<TaskConfig> configs(tasks);
    DeviceBuffer<uint8_t> scratch(ScratchBytes(maxBlocks));
    Input in{q.data,main.data,extra.data,ids.data,extraIds.data,rope.data,c.use_sink?sink.data:nullptr};
    Output out{packed.data,scales.data,lse.data,c.sfstride};
    auto launch=[&](uint32_t blocks) {
        launch_mega_attn_v1(in,out,configs.data,taskCount,blocks,scratch.data,
                           qkTiling,pvTiling,workspace,rt.stream);
    };
    std::vector<uint32_t> blockCounts{1};
    for (uint32_t blocks: {2u,5u,maxBlocks})
        if (blocks<=maxBlocks && std::find(blockCounts.begin(),blockCounts.end(),blocks)==blockCounts.end())
            blockCounts.push_back(blocks);
    std::vector<uint8_t> serialPacked;
    std::vector<uint32_t> serialScales;
    std::vector<float> serialLse;
    for (uint32_t blocks:blockCounts) {
        packed.Upload(std::vector<uint8_t>(packed.count,0xa5));
        scales.Upload(std::vector<uint32_t>(scales.count,0xa5a5a5a5u));
        lse.Upload(std::vector<float>(lse.count,NAN));
        std::fprintf(stderr,"NPU v1 [%s]: tasks=%u groups=%u\n",name,taskCount,blocks);
        launch(blocks);
        AclCheck(aclrtSynchronizeStreamWithTimeout(rt.stream,30000),"v1 synchronize");
        auto actualPacked=packed.Download();
        auto actualScales=scales.Download();
        auto actualLse=lse.Download();
        CompareQuantized(c,ref,actualPacked,actualScales,actualLse);
        if (blocks==1) {
            serialPacked=actualPacked; serialScales=actualScales; serialLse=actualLse;
        } else {
            check(actualPacked==serialPacked,"Multi-core FP8 differs from one group");
            check(actualScales==serialScales,"Multi-core scales differ from one group");
            CompareFloats(actualLse,serialLse,0,0,"Multi-core LSE");
        }
        scratch.CheckGuard(); packed.CheckGuard(); scales.CheckGuard(); lse.CheckGuard();
        std::printf("PASS NPU v1: %s tasks=%u blocks=%u\n",name,taskCount,blocks);
    }
    if (benchmark) {
        std::printf("BENCH shape: N=%d heads=%d dim=%d FP8_K=%d FP4_K=%d groups=%u\n",
                    c.n,HEADS,DIM,c.k,c.ek,maxBlocks);
        MeasureBatch(rt,"v1",c.n,[&] { launch(maxBlocks); });
        CompareQuantized(c,ref,packed.Download(),scales.Download(),lse.Download());
    }
    RunV2Case(rt,name,c,availableCores,in,out,packed,scales,lse,v2Qk,v2Pv,workspace,benchmark);
    q.CheckGuard(); main.CheckGuard(); extra.CheckGuard(); ids.CheckGuard(); extraIds.CheckGuard();
    rope.CheckGuard(); sink.CheckGuard(); configs.CheckGuard(); scratch.CheckGuard();
    packed.CheckGuard(); scales.CheckGuard(); lse.CheckGuard();
}

void NpuTests() {
    Runtime rt;
    auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    check(platform != nullptr,"Cannot get Ascend platform");
    DeviceBuffer<uint8_t> qkTiling(MakeCubeTiling(*platform,false));
    DeviceBuffer<uint8_t> pvTiling(MakeCubeTiling(*platform,true));
    DeviceBuffer<uint8_t> v2Qk(MakeCubeTiling(*platform,false,true));
    DeviceBuffer<uint8_t> v2Pv(MakeCubeTiling(*platform,true,true));
    DeviceBuffer<uint8_t> workspace(platform->GetLibApiWorkSpaceSize());
    ForEachCase([&](const char *name, const mega_test::Case &c, bool bench=false) {
        RunNpuCase(rt,name,c,platform->GetCoreNumAic(),qkTiling.data,pvTiling.data,v2Qk.data,v2Pv.data,workspace.data,bench);
    });
    qkTiling.CheckGuard(); pvTiling.CheckGuard(); v2Qk.CheckGuard(); v2Pv.CheckGuard(); workspace.CheckGuard();
}
} // namespace
#endif

int main() {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    try {
        OracleTests();
        OnlineOracleTests();
        FinalOracleTests();
        V2CodecTests();
#ifdef MEGA_ASCEND_CPU_ONLY
        ForEachCase([](const char *name, const mega_test::Case &c, bool = false) {
            auto ref=mega_test::reference(c);
            for (float x:ref.out) check(std::isfinite(x),"Nonfinite CPU reference output");
            for (int q=0; q<c.n; ++q) for (int h=0; h<HEADS; h+=HEAD_TILE)
                (void)MakeConfig(c,q,h,0);
            std::printf("PASS CPU fixture: %s\n",name);
        });
        std::puts("CPU oracle only: device execution NOT tested.");
#else
        NpuTests();
        std::puts("PASS: v1/v2 complete operator correctness and benchmark");
#endif
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr,"FAIL: %s\n",e.what());
        return 1;
    }
}
