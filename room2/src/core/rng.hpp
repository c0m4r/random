// room2 - deterministic pseudo random number generation.
// The whole project (textures, shard fracture, sound synthesis, gameplay spread)
// uses these generators so that a given seed always produces the same result.
#pragma once

#include <cstdint>
#include <cmath>
#include "core/math.hpp"

namespace room2 {

// Small, fast, high quality 32-bit generator (Sebastiano Vigna's xoroshiro64**).
class Rng {
public:
    explicit Rng(uint64_t seed = 0x9E3779B97F4A7C15ull) { seedWith(seed); }

    void seedWith(uint64_t seed) {
        // SplitMix64 to decorrelate the state from the seed.
        uint64_t z = seed + 0x9E3779B97F4A7C15ull;
        auto next = [&z]() {
            z += 0x9E3779B97F4A7C15ull;
            uint64_t x = z;
            x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
            x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
            return x ^ (x >> 31);
        };
        s_[0] = static_cast<uint32_t>(next());
        s_[1] = static_cast<uint32_t>(next());
        if (s_[0] == 0 && s_[1] == 0) s_[0] = 0x9E3779B9u;
    }

    uint32_t nextU32() {
        uint32_t s0 = s_[0];
        uint32_t s1 = s_[1];
        uint32_t result = s0 * 0x9E3779BBu;
        result = (result << 5 | result >> 27);
        result *= 5u;
        s1 ^= s0;
        s_[0] = (s0 << 26 | s0 >> 6) ^ s1 ^ (s1 << 9);
        s_[1] = (s1 << 13 | s1 >> 19);
        return result;
    }

    // Uniform in [0,1).
    float nextFloat() {
        // 24 mantissa bits is enough for float precision.
        return static_cast<float>(nextU32() >> 8) * (1.0f / 16777216.0f);
    }
    // Uniform in [lo,hi).
    float range(float lo, float hi) { return lo + (hi - lo) * nextFloat(); }
    // Uniform in [-1,1).
    float symmetric() { return nextFloat() * 2.0f - 1.0f; }
    int rangeInt(int lo, int hiExclusive) {
        if (hiExclusive <= lo) return lo;
        return lo + static_cast<int>(nextU32() % static_cast<uint32_t>(hiExclusive - lo));
    }
    bool chance(float p) { return nextFloat() < p; }

    Vec2 unitVec2() {
        float a = nextFloat() * TWO_PI;
        return {std::cos(a), std::sin(a)};
    }
    // Uniformly distributed direction on the unit sphere.
    Vec3 unitVec3() {
        float z = symmetric();
        float a = nextFloat() * TWO_PI;
        float r = std::sqrt(std::max(0.0f, 1.0f - z * z));
        return {r * std::cos(a), r * std::sin(a), z};
    }
    // Cosine-weighted direction around +Z (hemisphere sampling for AO / GI).
    Vec3 cosineHemisphere() {
        float u1 = nextFloat();
        float u2 = nextFloat();
        float r = std::sqrt(u1);
        float theta = TWO_PI * u2;
        return {r * std::cos(theta), r * std::sin(theta), std::sqrt(std::max(0.0f, 1.0f - u1))};
    }
    // Roughly normal distribution via the sum of 3 uniforms.
    float gaussian(float mean = 0.0f, float stddev = 1.0f) {
        float s = nextFloat() + nextFloat() + nextFloat() - 1.5f;
        return mean + s * stddev * 1.1547f;
    }
    void fillUnitVec3(Vec3* out, int count) {
        for (int i = 0; i < count; ++i) out[i] = unitVec3();
    }

private:
    uint32_t s_[2]{0x9E3779B9u, 0x85EBCA6Bu};
};

// ---------------------------------------------------------------- hashing / noise
inline uint32_t hashU32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7FEB352Du;
    x ^= x >> 15;
    x *= 0x846CA68Bu;
    x ^= x >> 16;
    return x;
}
inline uint32_t hashCombine(uint32_t a, uint32_t b) {
    return hashU32(a ^ (b + 0x9E3779B9u + (a << 6) + (a >> 2)));
}
inline float hashToUnit(uint32_t h) { return static_cast<float>(h >> 8) * (1.0f / 16777216.0f); }

// 2D value noise in [0,1].
inline float valueNoise2(float x, float y, uint32_t seed) {
    int xi = static_cast<int>(std::floor(x));
    int yi = static_cast<int>(std::floor(y));
    float xf = x - static_cast<float>(xi);
    float yf = y - static_cast<float>(yi);
    float u = xf * xf * (3.0f - 2.0f * xf);
    float v = yf * yf * (3.0f - 2.0f * yf);
    auto at = [seed](int a, int b) {
        return hashToUnit(hashCombine(hashCombine(static_cast<uint32_t>(a), static_cast<uint32_t>(b)), seed));
    };
    float a = at(xi, yi), b = at(xi + 1, yi), c = at(xi, yi + 1), d = at(xi + 1, yi + 1);
    return lerpf(lerpf(a, b, u), lerpf(c, d, u), v);
}

// Fractal Brownian motion built on value noise.
inline float fbm2(float x, float y, int octaves, uint32_t seed, float lacunarity = 2.0f,
                  float gain = 0.5f) {
    float sum = 0.0f, amp = 1.0f, norm = 0.0f, fx = x, fy = y;
    for (int i = 0; i < octaves; ++i) {
        sum += amp * valueNoise2(fx, fy, seed + static_cast<uint32_t>(i) * 131u);
        norm += amp;
        amp *= gain;
        fx *= lacunarity;
        fy *= lacunarity;
    }
    return norm > 0.0f ? sum / norm : 0.0f;
}

// Tileable value noise: the lattice wraps at `period` cells.
inline float tileableNoise2(float x, float y, int period, uint32_t seed) {
    auto wrap = [period](int v) { return ((v % period) + period) % period; };
    int xi = static_cast<int>(std::floor(x));
    int yi = static_cast<int>(std::floor(y));
    float xf = x - static_cast<float>(xi);
    float yf = y - static_cast<float>(yi);
    float u = xf * xf * (3.0f - 2.0f * xf);
    float v = yf * yf * (3.0f - 2.0f * yf);
    auto at = [&](int a, int b) {
        return hashToUnit(hashCombine(hashCombine(static_cast<uint32_t>(wrap(a)),
                                                  static_cast<uint32_t>(wrap(b))), seed));
    };
    float a = at(xi, yi), b = at(xi + 1, yi), c = at(xi, yi + 1), d = at(xi + 1, yi + 1);
    return lerpf(lerpf(a, b, u), lerpf(c, d, u), v);
}

inline float tileableFbm2(float x, float y, int period, int octaves, uint32_t seed,
                          float gain = 0.5f) {
    float sum = 0.0f, amp = 1.0f, norm = 0.0f;
    int p = period;
    for (int i = 0; i < octaves; ++i) {
        sum += amp * tileableNoise2(x, y, p, seed + static_cast<uint32_t>(i) * 7919u);
        norm += amp;
        amp *= gain;
        x *= 2.0f;
        y *= 2.0f;
        p *= 2;
    }
    return norm > 0.0f ? sum / norm : 0.0f;
}

// Worley / cellular noise: returns the distance to the nearest feature point (in cells).
inline float worley2(float x, float y, uint32_t seed, float* secondOut = nullptr) {
    int xi = static_cast<int>(std::floor(x));
    int yi = static_cast<int>(std::floor(y));
    float best = 1e9f, second = 1e9f;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int cx = xi + dx, cy = yi + dy;
            uint32_t h = hashCombine(hashCombine(static_cast<uint32_t>(cx), static_cast<uint32_t>(cy)), seed);
            float px = static_cast<float>(cx) + hashToUnit(h);
            float py = static_cast<float>(cy) + hashToUnit(hashU32(h));
            float d = std::sqrt((px - x) * (px - x) + (py - y) * (py - y));
            if (d < best) {
                second = best;
                best = d;
            } else if (d < second) {
                second = d;
            }
        }
    }
    if (secondOut) *secondOut = second;
    return best;
}

// ---------------------------------------------------------------- sampling helpers
inline float luma(Vec3 c) { return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z; }

}  // namespace room2
