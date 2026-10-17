// Portable CPU oracle and cache codecs. No CUDA/CUTLASS dependency.
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace mega_test {
constexpr int H = 64, D = 512, R = 64;
constexpr float SCALE = 0.04419417382415922f;
inline void check(bool ok, const char *msg) {
    if (!ok)
        throw std::runtime_error(msg);
}
inline int align_up(int x, int a) { return (x + a - 1) / a * a; }
inline uint16_t bf16(float x) {
    uint32_t bits = std::bit_cast<uint32_t>(x);
    if ((bits & 0x7fffffff) > 0x7f800000)
        return uint16_t((bits >> 16) | 0x40);
    return uint16_t((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}
inline float from_bf16(uint16_t x) {
    return std::bit_cast<float>(uint32_t(x) << 16);
}
inline float e4m3(uint8_t x) {
    int e = (x >> 3) & 15, m = x & 7;
    float v = e == 0 ? std::ldexp(float(m), -9)
                     : std::ldexp(1.f + float(m) / 8, e - 7);
    if (e == 15 && m == 7)
        v = std::numeric_limits<float>::quiet_NaN();
    return (x & 128) ? -v : v;
}
inline uint8_t to_e4m3(float x) {
    check(std::isfinite(x), "Test packer expects finite FP8 inputs");
    static const auto table = [] {
        std::array<float, 127> t{};
        for (int i = 0; i < 127; ++i)
            t[i] = e4m3(uint8_t(i));
        return t;
    }();
    float a = std::min(std::abs(x), 448.f);
    int hi =
        int(std::lower_bound(table.begin(), table.end(), a) - table.begin());
    int lo = std::max(0, hi - 1), code = hi;
    if (a - table[lo] < table[hi] - a ||
        (a - table[lo] == table[hi] - a && !(lo & 1)))
        code = lo;
    return uint8_t(code | (std::signbit(x) ? 128 : 0));
}
inline float e2m1(uint8_t x) {
    constexpr float mags[] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
    return (x & 8) ? -mags[x & 7] : mags[x & 7];
}
inline uint8_t to_e2m1(float x) {
    check(std::isfinite(x), "Test packer expects finite FP4 inputs");
    float a = std::min(std::abs(x), 6.f), best = 100;
    int code = 0;
    for (int i = 0; i < 8; ++i) {
        float distance = std::abs(a - e2m1(uint8_t(i)));
        if (distance < best || (distance == best && !(i & 1))) {
            code = i;
            best = distance;
        }
    }
    return uint8_t(code | (std::signbit(x) ? 8 : 0));
}
inline size_t q_offset(int n, int h, int d) {
    return size_t(n) * H * D + (d / 16) * H * 16 + h * 16 + d % 16;
}
inline size_t out_offset(int n, int h, int d) {
    return size_t(n) * H * D + (h / 8) * 4096 + (d / 32) * 256 + (h % 8) * 32 +
           d % 32;
}
inline int physical_index(
    int logical, const std::vector<int> &pages, int page_size
) {
    if (logical < 0 || logical >= int(pages.size()) * page_size)
        return -1;
    int page = pages[logical / page_size];
    return page < 0 ? -1 : page * page_size + logical % page_size;
}

// 构造 CPU 侧的量化分页缓存，并保存解码后的数值供参考计算使用。
struct Cache {
    int pages, size, stride;
    bool fp4;
    std::vector<uint8_t> bytes;
    // Decoded FROM bytes for the oracle, not a copy of the pre-quantized input.
    std::vector<float> decoded;
    Cache(int p, int s, bool four)
        : pages(p), size(s), stride(align_up(s * (four ? 288 : 528), 512)),
          fp4(four), bytes(size_t(p) * stride, 0xcd),
          decoded(size_t(p) * s * D) {}
    int slots() const { return pages * size; }
    void pack(const std::vector<float> &values) {
        check(values.size() == decoded.size(), "Wrong cache source shape");
        int tile = fp4 ? 16 : 32, data_bytes = fp4 ? 256 : 512,
            scales = fp4 ? 32 : 16;
        for (int p = 0; p < pages; ++p)
            for (int t = 0; t < size; ++t)
                for (int c = 0; c < scales; ++c) {
                    size_t logical = (size_t(p) * size + t) * D + c * tile;
                    float amax = 0;
                    for (int d = 0; d < tile; ++d) {
                        check(
                            std::isfinite(values[logical + d]),
                            "Nonfinite cache source"
                        );
                        amax = std::max(amax, std::abs(values[logical + d]));
                    }
                    uint8_t scale_byte;
                    float scale;
                    if (fp4) {
                        scale_byte = to_e4m3(
                            std::clamp(amax / 6.f, std::ldexp(1.f, -9), 448.f)
                        );
                        scale = e4m3(scale_byte);
                    } else {
                        int e = int(
                            std::ceil(std::log2(std::max(amax / 448.f, 1e-4f)))
                        );
                        scale_byte = uint8_t(e + 127);
                        scale = std::ldexp(1.f, e);
                    }
                    bytes
                        [size_t(p) * stride + size * data_bytes + t * scales +
                         c] = scale_byte;
                    for (int d = 0; d < tile; ++d) {
                        size_t data = size_t(p) * stride + t * data_bytes;
                        if (fp4) {
                            int dim = c * tile + d;
                            uint8_t code = to_e2m1(values[logical + d] / scale);
                            if (dim % 2 == 0)
                                bytes[data + dim / 2] = code;
                            else
                                bytes[data + dim / 2] |= uint8_t(code << 4);
                        } else
                            bytes[data + c * tile + d] =
                                to_e4m3(values[logical + d] / scale);
                    }
                }
        unpack();
    }
    void unpack() {
        int tile = fp4 ? 16 : 32, data_bytes = fp4 ? 256 : 512,
            scales = fp4 ? 32 : 16;
        for (int p = 0; p < pages; ++p)
            for (int t = 0; t < size; ++t)
                for (int d = 0; d < D; ++d) {
                    uint8_t b = bytes
                        [size_t(p) * stride + t * data_bytes +
                         (fp4 ? d / 2 : d)];
                    uint8_t s = bytes
                        [size_t(p) * stride + size * data_bytes + t * scales +
                         d / tile];
                    float value = fp4 ? e2m1(uint8_t((b >> (4 * (d % 2))) & 15))
                                      : e4m3(b);
                    float scale = fp4 ? e4m3(s) : std::ldexp(1.f, int(s) - 127);
                    decoded[(size_t(p) * size + t) * D + d] =
                        from_bf16(bf16(value * scale));
                }
    }
};

// 保存一组测试输入，包括 Q、两份 KV 缓存、稀疏索引和 RoPE 参数。
struct Case {
    int n, k, ek, istride, eistride, sfstride;
    Cache main, extra;
    std::vector<uint16_t> q;
    std::vector<int32_t> indices, extra_indices, positions, lengths,
        extra_lengths;
    std::vector<float> cos_sin, sink;
    bool use_lengths = true, use_sink = true;
    Case(int n_, int k_, int ek_, int page, int seed = 1)
        : n(n_), k(k_), ek(ek_), istride(align_up(k, 8)),
          eistride(align_up(ek, 8)), sfstride(align_up(n, 4)),
          main(5, page, false), extra(7, page, true), q(size_t(n) * H * D),
          indices(size_t(n) * istride, 0),
          extra_indices(size_t(n) * eistride, 0), positions(n), lengths(n, k),
          extra_lengths(n, ek), cos_sin(32 * R), sink(H) {
        check(n > 0 && k > 0 && k % 8 == 0 && ek > 0, "Invalid test shape");
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> uniform(-.5f, .5f);
        for (int i = 0; i < n; ++i)
            for (int h = 0; h < H; ++h)
                for (int d = 0; d < D; ++d)
                    q[q_offset(i, h, d)] = bf16(uniform(rng));
        for (Cache *cache : {&main, &extra}) {
            std::vector<float> values(cache->decoded.size());
            for (float &v : values)
                v = from_bf16(bf16(uniform(rng)));
            cache->pack(values);
        }
        // 有效长度内只生成合法索引，空集合由 length=0 表示。
        const std::vector<int> mt = {3, 0, 4, 1, 2}, et = {6, 2, 5, 0, 4, 1, 3};
        for (int i = 0; i < n; ++i) {
            positions[i] = i % 32;
            for (int j = 0; j < k; ++j)
                indices[size_t(i) * istride + j] =
                    physical_index(int(rng() % main.slots()), mt, page);
            for (int j = 0; j < ek; ++j)
                extra_indices[size_t(i) * eistride + j] =
                    physical_index(int(rng() % extra.slots()), et, page);
        }
        if (n >= 3) {
            lengths[0] = 0;
            extra_lengths[0] = 0;
            lengths[1] = std::min(k, 13);
            extra_lengths[1] = std::min(ek, 19);
        }
        for (int pos = 0; pos < 32; ++pos)
            for (int d = 0; d < 32; ++d) {
                float theta =
                    float(pos) * std::pow(100000.f, -float(2 * d) / 64);
                cos_sin[pos * 64 + d] = std::cos(theta);
                cos_sin[pos * 64 + 32 + d] = std::sin(theta);
            }
        for (float &v : sink)
            v = uniform(rng);
        sink[0] = -INFINITY;
        sink[1] = INFINITY;
    }
};

// 保存 CPU 参考实现的输出和 LSE，用于核对 GPU 结果。
struct Reference {
    std::vector<float> out, lse;
};
inline Reference reference(const Case &c) {
    Reference r{
        std::vector<float>(size_t(c.n) * H * D, 0),
        std::vector<float>(size_t(c.n) * H, INFINITY)
    };
    for (int i = 0; i < c.n; ++i) {
        std::vector<const float *> rows;
        auto gather = [&](const Cache &cache,
                          const std::vector<int32_t> &ids,
                          int stride,
                          int count,
                          int length) {
            for (int j = 0; j < std::min(count, length); ++j) {
                int id = ids[size_t(i) * stride + j];
                rows.push_back(cache.decoded.data() + size_t(id) * D);
            }
        };
        gather(
            c.main,
            c.indices,
            c.istride,
            c.k,
            c.use_lengths ? c.lengths[i] : c.k
        );
        gather(
            c.extra,
            c.extra_indices,
            c.eistride,
            c.ek,
            c.use_lengths ? c.extra_lengths[i] : c.ek
        );
        if (rows.empty())
            continue;
        const float *cs = c.cos_sin.data() + c.positions[i] * 64;
        for (int h = 0; h < H; ++h) {
            std::array<float, D> q{};
            for (int d = 0; d < D; ++d)
                q[d] = from_bf16(c.q[q_offset(i, h, d)]);
            for (int pair = 0; pair < 32; ++pair) {
                int d = 448 + 2 * pair;
                float a = q[d], b = q[d + 1], co = cs[pair], si = cs[32 + pair];
                q[d] = from_bf16(bf16(std::fma(a, co, -b * si)));
                q[d + 1] = from_bf16(bf16(std::fma(a, si, b * co)));
            }
            std::vector<double> scores(rows.size());
            double m = -INFINITY;
            for (size_t j = 0; j < rows.size(); ++j) {
                double dot = 0;
                for (int d = 0; d < D; ++d)
                    dot += double(q[d]) * rows[j][d];
                scores[j] = dot * SCALE;
                m = std::max(m, scores[j]);
            }
            double sum = 0;
            for (double &score : scores) {
                score = std::exp(score - m);
                sum += score;
            }
            r.lse[size_t(i) * H + h] = float(m + std::log(sum));
            double denominator =
                sum + (c.use_sink ? std::exp(double(c.sink[h]) - m) : 0);
            std::array<double, D> accum{};
            if (!std::isinf(denominator))
                for (size_t j = 0; j < rows.size(); ++j)
                    for (int d = 0; d < D; ++d)
                        accum[d] += scores[j] / denominator * rows[j][d];
            float *out = r.out.data() + (size_t(i) * H + h) * D;
            for (int d = 0; d < D; ++d)
                out[d] = float(accum[d]);
            for (int pair = 0; pair < 32; ++pair) {
                int d = 448 + 2 * pair;
                float a = out[d], b = out[d + 1], co = cs[pair],
                      si = cs[32 + pair];
                out[d] = std::fma(a, co, b * si);
                out[d + 1] = std::fma(-a, si, b * co);
            }
        }
    }
    return r;
}

inline std::vector<float> unpack_output(
    const std::vector<uint8_t> &data,
    const std::vector<uint32_t> &scales,
    int n,
    int sfstride
) {
    std::vector<float> out(size_t(n) * H * D);
    for (int i = 0; i < n; ++i)
        for (int h = 0; h < H; ++h)
            for (int d = 0; d < D; ++d) {
                int scale_slot = (d / 32) * 8 + h % 8;
                uint32_t word = scales
                    [size_t(h / 8) * 32 * sfstride +
                     (scale_slot / 4) * sfstride + i];
                int exponent = int((word >> (8 * (scale_slot % 4))) & 255);
                out[(size_t(i) * H + h) * D + d] =
                    e4m3(data[out_offset(i, h, d)]) *
                    std::ldexp(1.f, exponent - 127);
            }
    return out;
}

inline void compare(
    const Reference &ref,
    const std::vector<float> &out,
    const std::vector<float> &lse
) {
    check(
        ref.out.size() == out.size() && ref.lse.size() == lse.size(),
        "Wrong result sizes"
    );
    for (size_t row = 0; row < lse.size(); ++row) {
        if (std::isinf(ref.lse[row]))
            check(lse[row] == ref.lse[row], "Empty row LSE must be +inf");
        else if (!std::isfinite(lse[row]) ||
                 std::abs(lse[row] - ref.lse[row]) >
                     1e-3f * (1 + std::abs(ref.lse[row])))
            throw std::runtime_error(
                "LSE mismatch at head row " + std::to_string(row)
            );
        double err = 0, norm = 0, actualnorm = 0, dot = 0;
        for (int d = 0; d < D; ++d) {
            double a = out[row * D + d], b = ref.out[row * D + d];
            check(std::isfinite(a), "Nonfinite FP8 output");
            err += (a - b) * (a - b);
            norm += b * b;
            actualnorm += a * a;
            dot += a * b;
        }
        if (norm < 1e-14)
            check(
                actualnorm < 1e-14, "Empty/sink-infinite head must output zero"
            );
        else {
            double relative = std::sqrt(err / norm),
                   cosine =
                       1 - dot / std::sqrt(norm * std::max(actualnorm, 1e-30));
            if (relative > .06 || cosine > .002)
                throw std::runtime_error(
                    "Output mismatch at head row " + std::to_string(row) +
                    ", relative L2=" + std::to_string(relative) +
                    ", cosine error=" + std::to_string(cosine)
                );
        }
    }
}

inline void cpu_tests() {
    check(bf16(1.f) == 0x3f80 && from_bf16(0x3f80) == 1.f, "BF16 codec");
    check(
        e4m3(0x38) == 1 && e4m3(0x7e) == 448 && e4m3(1) == std::ldexp(1.f, -9),
        "FP8 known values"
    );
    for (int code = 0; code < 127; ++code) {
        check(
            to_e4m3(e4m3(uint8_t(code))) == code,
            "FP8 exhaustive positive roundtrip"
        );
        check(
            to_e4m3(e4m3(uint8_t(code | 128))) == (code | 128),
            "FP8 exhaustive negative roundtrip"
        );
    }
    float mids[] = {.25f, .75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.f};
    int even[] = {0, 2, 2, 4, 4, 6, 6};
    for (int i = 0; i < 7; ++i) {
        check(to_e2m1(mids[i]) == even[i], "FP4 tie-to-even");
        check(to_e2m1(-mids[i]) == (even[i] | 8), "FP4 negative tie");
    }
    check(to_e2m1(99) == 7 && to_e2m1(-99) == 15, "FP4 saturation");
    check(
        physical_index(4, {7, 2, 9}, 4) == 8,
        "Page mapping"
    );
    check(q_offset(1, 3, 17) == 32768 + 1024 + 48 + 1, "Q chunk16 offset");
    for (bool four : {false, true}) {
        Cache c(3, 5, four);
        float value = four ? 6.f : 448.f;
        c.pack(std::vector<float>(c.decoded.size(), value));
        int db = four ? 256 : 512, sb = four ? 32 : 16;
        for (int p = 0; p < c.pages; ++p) {
            for (int i = 0; i < c.size * db; ++i)
                check(
                    c.bytes[size_t(p) * c.stride + i] == (four ? 0x77 : 0x7e),
                    "Known cache data bytes"
                );
            for (int i = 0; i < c.size * sb; ++i)
                check(
                    c.bytes[size_t(p) * c.stride + c.size * db + i] ==
                        (four ? 0x38 : 127),
                    "Known cache scale bytes"
                );
            for (int i = c.size * (db + sb); i < c.stride; ++i)
                check(
                    c.bytes[size_t(p) * c.stride + i] == 0xcd,
                    "Page padding preserved"
                );
        }
        for (float v : c.decoded)
            check(v == value, "Known cache dequantization");
    }
    for (int n : {1, 3, 4, 5}) {
        int stride = align_up(n, 4);
        std::vector<uint8_t> data(size_t(n) * H * D, 0x38);
        std::vector<uint32_t> scales(256 * stride, 0x8281807fu);
        auto out = unpack_output(data, scales, n, stride);
        for (int i = 0; i < n; ++i)
            for (int h = 0; h < H; ++h)
                for (int d = 0; d < D; ++d)
                    check(
                        out[(size_t(i) * H + h) * D + d] ==
                            std::ldexp(1.f, h % 4),
                        "Output scale/head layout"
                    );
    }
    Case c(3, 8, 9, 5);
    std::fill(c.q.begin(), c.q.end(), 0);
    std::fill(c.positions.begin(), c.positions.end(), 0);
    c.main.pack(std::vector<float>(c.main.decoded.size(), 1.f));
    c.extra.pack(std::vector<float>(c.extra.decoded.size(), 6.f));
    std::fill(c.lengths.begin(), c.lengths.end(), 0);
    std::fill(c.extra_lengths.begin(), c.extra_lengths.end(), 0);
    c.indices[0] = 0;
    c.extra_indices[0] = 1;
    c.lengths[0] = 1;
    c.extra_lengths[0] = 1;
    std::fill(c.sink.begin(), c.sink.end(), 0);
    c.sink[0] = -INFINITY;
    c.sink[1] = INFINITY;
    auto r = reference(c);
    check(std::abs(r.lse[0] - std::log(2.f)) < 1e-6, "Two-cache LSE");
    check(
        r.out[0] == 3.5f && r.out[D] == 0 &&
            std::abs(r.out[2 * D] - 7.f / 3) < 1e-6,
        "Sink semantics"
    );
    check(
        r.lse[H] == INFINITY && r.out[H * D] == 0, "Empty attention semantics"
    );
}
} // namespace mega_test

#ifndef MEGA_ATTN_REFERENCE_ONLY
// Test runner and GPU benchmark.
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef MEGA_ATTN_CPU_ONLY
int main() {
    try {
        mega_test::cpu_tests();
        std::puts("PASS: CPU codecs, page/Q/output layouts, reference "
                  "attention and sink/empty semantics");
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
#else
#include "megaAttn_api.h"
#include <cuda_runtime.h>

namespace {
void cuda_check(cudaError_t e, const char *file, int line) {
    if (e != cudaSuccess)
        throw std::runtime_error(
            std::string(file) + ":" + std::to_string(line) + ": " +
            cudaGetErrorString(e)
        );
}
#define CUDA_CHECK(call) cuda_check((call), __FILE__, __LINE__)

// 绑定算子版本名和调用入口，方便统一测试 v0、v1。
struct Impl {
    const char *name;
    SolveFn solve;
};
const Impl implementations[] = {
    {"v0", solve_v0},
#ifdef ENABLE_MEGA_ATTN_V1
    {"v1", solve_v1},
#endif
};

// 管理一块设备内存，并用首尾保护区检查越界写入。
template <class T> struct Buffer {
    static constexpr size_t GUARD = 64;
    T *base = nullptr;
    T *data = nullptr;
    size_t count;
    explicit Buffer(size_t n) : count(n) {
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void **>(&base), (n + 2 * GUARD) * sizeof(T)
        ));
        data = base + GUARD;
        CUDA_CHECK(cudaMemset(base, 0xa5, (n + 2 * GUARD) * sizeof(T)));
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
    }
    explicit Buffer(const std::vector<T> &source) : Buffer(source.size()) {
        upload(source);
    }
    Buffer(const Buffer &) = delete;
    ~Buffer() {
        if (base)
            cudaFree(base);
    }
    void upload(const std::vector<T> &source) {
        mega_test::check(source.size() == count, "Upload size mismatch");
        CUDA_CHECK(cudaMemcpy(
            data, source.data(), count * sizeof(T), cudaMemcpyHostToDevice
        ));
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
    }
    void poison(cudaStream_t stream) {
        CUDA_CHECK(cudaMemsetAsync(data, 0xa5, count * sizeof(T), stream));
    }
    std::vector<T> download() const {
        std::vector<T> result(count);
        CUDA_CHECK(cudaMemcpy(
            result.data(), data, count * sizeof(T), cudaMemcpyDeviceToHost
        ));
        return result;
    }
    void check_guard() const {
        std::array<unsigned char, GUARD * sizeof(T)> bytes;
        for (const T *p : {base, data + count}) {
            CUDA_CHECK(cudaMemcpy(
                bytes.data(), p, bytes.size(), cudaMemcpyDeviceToHost
            ));
            for (unsigned char b : bytes)
                mega_test::check(
                    b == 0xa5, "Device allocation guard overwritten"
                );
        }
    }
};

// 把测试输入上传到 GPU，并管理输出缓冲区、调用参数和执行流。
struct DeviceCase {
    Buffer<uint16_t> q;
    Buffer<uint8_t> main, extra, out;
    Buffer<int32_t> indices, extra_indices, positions, lengths, extra_lengths;
    Buffer<float> cos_sin, sink, lse;
    Buffer<uint32_t> sf;
    MegaAttnParams p{};
    cudaStream_t stream = nullptr;
    explicit DeviceCase(const mega_test::Case &c)
        : q(c.q), main(c.main.bytes), extra(c.extra.bytes),
          out(size_t(c.n) * 64 * 512), indices(c.indices),
          extra_indices(c.extra_indices), positions(c.positions),
          lengths(c.lengths), extra_lengths(c.extra_lengths),
          cos_sin(c.cos_sin), sink(c.sink), lse(size_t(c.n) * 64),
          sf(size_t(256) * c.sfstride) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        p.q = q.data;
        p.kv_fp8 = main.data;
        p.kv_fp4 = extra.data;
        p.indices = indices.data;
        p.extra_indices = extra_indices.data;
        p.positions = positions.data;
        p.cos_sin = cos_sin.data;
        p.attn_sink = c.use_sink ? sink.data : nullptr;
        p.topk_length = c.use_lengths ? lengths.data : nullptr;
        p.extra_topk_length = c.use_lengths ? extra_lengths.data : nullptr;
        p.out_fp8 = out.data;
        p.out_sf = sf.data;
        p.lse = lse.data;
        p.n = c.n;
        p.num_pages = c.main.pages;
        p.page_size = c.main.size;
        p.page_stride = c.main.stride;
        p.extra_num_pages = c.extra.pages;
        p.extra_page_size = c.extra.size;
        p.extra_page_stride = c.extra.stride;
        p.topk = c.k;
        p.extra_topk = c.ek;
        p.indices_stride = c.istride;
        p.extra_indices_stride = c.eistride;
        p.sf_stride = c.sfstride;
    }
    ~DeviceCase() {
        if (stream)
            cudaStreamDestroy(stream);
    }
    void poison_outputs() {
        out.poison(stream);
        sf.poison(stream);
        lse.poison(stream);
    }
    void verify(const mega_test::Reference &ref) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto data = out.download();
        auto scales = sf.download();
        auto unpacked =
            mega_test::unpack_output(data, scales, p.n, p.sf_stride);
        mega_test::compare(ref, unpacked, lse.download());
        // Token padding in EVERY output scale column must remain untouched.
        for (int column = 0; column < 256; ++column)
            for (int n = p.n; n < p.sf_stride; ++n)
                mega_test::check(
                    scales[size_t(column) * p.sf_stride + n] == 0xa5a5a5a5u,
                    "Scale padding overwritten"
                );
        q.check_guard();
        main.check_guard();
        extra.check_guard();
        out.check_guard();
        indices.check_guard();
        extra_indices.check_guard();
        positions.check_guard();
        lengths.check_guard();
        extra_lengths.check_guard();
        cos_sin.check_guard();
        sink.check_guard();
        lse.check_guard();
        sf.check_guard();
    }
};

// 封装 CUDA Graph 的捕获、重放和释放，用于测试与计时。
struct Graph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    Graph(SolveFn solve, DeviceCase &d, int launches = 1) {
        CUDA_CHECK(cudaStreamBeginCapture(d.stream, cudaStreamCaptureModeGlobal)
        );
        for (int i = 0; i < launches; ++i)
            solve(d.p, d.stream);
        CUDA_CHECK(cudaStreamEndCapture(d.stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    }
    ~Graph() {
        if (exec)
            cudaGraphExecDestroy(exec);
        if (graph)
            cudaGraphDestroy(graph);
    }
    void replay(cudaStream_t stream) {
        CUDA_CHECK(cudaGraphLaunch(exec, stream));
    }
};

void run_correctness() {
    // 描述测试规模：Q 数量、两路 topk 长度和每页 token 数。
    struct Shape {
        int n, k, ek, page;
    };
    // k=72 forces one tile to stitch FP8 and FP4 rows; ek=121 adds a tail.
    const Shape shapes[] = {
        {1, 8, 9, 5}, {3, 64, 65, 64}, {17, 72, 121, 65}, {257, 8, 9, 5}
    };
    for (const auto &s : shapes) {
        mega_test::Case c(s.n, s.k, s.ek, s.page, s.n);
        auto ref = mega_test::reference(c);
        DeviceCase d(c);
        for (const auto &impl : implementations) {
            d.poison_outputs();
            impl.solve(d.p, d.stream);
            d.verify(ref);
            std::printf(
                "PASS %s N=%d topk=%d+%d page=%d\n",
                impl.name,
                s.n,
                s.k,
                s.ek,
                s.page
            );
        }
    }
    for (int mode = 0; mode < 3; ++mode) {
        mega_test::Case c(3, 64, 65, 64);
        if (mode == 0) {
            c.use_lengths = false;
            c.use_sink = false;
        }
        if (mode == 1) {
            std::fill(c.lengths.begin(), c.lengths.end(), 0);
            std::fill(c.extra_lengths.begin(), c.extra_lengths.end(), 0);
        }
        if (mode == 2) { // Larger logits exercise online rescaling without
                         // codec overflow.
            for (auto &bits : c.q)
                bits = mega_test::bf16(mega_test::from_bf16(bits) * 32);
        }
        auto ref = mega_test::reference(c);
        DeviceCase d(c);
        for (const auto &impl : implementations) {
            d.poison_outputs();
            impl.solve(d.p, d.stream);
            d.verify(ref);
            std::printf("PASS %s edge_mode=%d\n", impl.name, mode);
        }
    }
    for (const auto &impl : implementations) {
        mega_test::Case c(3, 64, 65, 65);
        DeviceCase d(c);
        impl.solve(d.p, d.stream);
        CUDA_CHECK(cudaStreamSynchronize(d.stream));
        Graph g(impl.solve, d);
        // Change contents at the same address after capture.
        for (auto &bits : c.q)
            bits = mega_test::bf16(mega_test::from_bf16(bits) * .5f);
        d.q.upload(c.q);
        d.poison_outputs();
        g.replay(d.stream);
        d.verify(mega_test::reference(c));
        auto invalid = d.p;
        invalid.topk = 7;
        bool threw = false;
        try {
            impl.solve(invalid, d.stream);
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        mega_test::check(threw, "Invalid topk must be rejected before launch");
        std::printf(
            "PASS %s graph replay with changed input + invalid argument\n",
            impl.name
        );
    }
}

void benchmark(int n, int iterations) {
    mega_test::Case c(n, 128, 512, 128);
    DeviceCase d(c);
    for (const auto &impl : implementations) {
        for (int i = 0; i < 3; ++i)
            impl.solve(d.p, d.stream);
        CUDA_CHECK(cudaStreamSynchronize(d.stream));
        constexpr int launches = 10;
        Graph graph(impl.solve, d, launches);
        graph.replay(d.stream);
        CUDA_CHECK(cudaStreamSynchronize(d.stream));
        cudaEvent_t start, end;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&end));
        std::vector<float> times;
        for (int i = 0; i < iterations; ++i) {
            CUDA_CHECK(cudaEventRecord(start, d.stream));
            graph.replay(d.stream);
            CUDA_CHECK(cudaEventRecord(end, d.stream));
            CUDA_CHECK(cudaEventSynchronize(end));
            float ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));
            times.push_back(ms * 1000 / launches);
        }
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(end));
        std::sort(times.begin(), times.end());
        std::printf(
            "%s N=%d topk=128+512: CUDA Graph median %.3f us/launch\n",
            impl.name,
            n,
            times[times.size() / 2]
        );
    }
    std::puts("Kernel-only timing: excludes Q packing, cache "
              "insertion/packing, reference, and projections.");
}
} // namespace

int main(int argc, char **argv) {
    try {
        bool check_only = argc > 1 && std::strcmp(argv[1], "--check-only") == 0;
        int n = 32, iters = 30;
        if (argc > 1 && !check_only)
            n = std::stoi(argv[1]);
        if (argc > 2)
            iters = std::stoi(argv[2]);
        mega_test::check(
            n > 0 && n <= 4096 && iters > 0 && iters <= 10000,
            "Usage: bench_megaAttn [N<=4096] [iters<=10000] or --check-only"
        );
        int dev = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
        if (prop.major != 10 || prop.minor != 0) {
            std::fprintf(
                stderr,
                "NOT RUN: requires SM100; current GPU is %s (SM%d%d).\n",
                prop.name,
                prop.major,
                prop.minor
            );
            return 2;
        }
        std::printf("GPU: %s\n", prop.name);
        // 默认只测性能；正确性用例通过 --check-only 单独运行。
        if (check_only) {
            mega_test::cpu_tests();
            run_correctness();
        } else {
            benchmark(n, iters);
        }
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
#endif

#endif // MEGA_ATTN_REFERENCE_ONLY
