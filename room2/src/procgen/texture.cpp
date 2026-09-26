// room2 - procedural PBR texture synthesis.
//
// Design notes
// ------------
// * Everything is built from *periodic* noise. Every fbm / worley lattice wraps at the tile
//   edge and every scattered feature (crack, scratch, knot, dent, smudge, pore) is measured
//   with a toroidal distance, so all maps are seamless by construction. Clamping or mirroring
//   at the border is never used - those leave a visible crease where a wall repeats.
// * Physical units: relief is authored in *metres* and mapped onto texels through
//   `SurfaceParams::texelsPerMetre` (tpm). A feature that should be 2 cm across is
//   `0.02 * tpm` texels wide, and the slope handed to the normal/AO maps is the real
//   per-texel slope `dh * tpm` (dh / texel size). If the mapping density changes, the relief
//   changes with it instead of being a fixed bunch of pixels.
// * Detail that is finer than the texel grid (1.8 mm stippling at 512 texels/m is a third of
//   a texel) is widened to a few texels and its amplitude is scaled by the same factor, so the
//   *slope* - what the shading actually sees - stays that of the real micro relief. The look
//   is slightly coarser than reality but lit correctly, which is the honest trade at 1:1.
// * Base colours are LINEAR reflectance values (plaster ~0.75, dark wood ~0.09, blued steel
//   ~0.05, brass ~0.65, glass ~0.96). The uploader sRGB-encodes them on pack; normal and ORM
//   maps stay linear.
// * Determinism: all randomness is hash/seed based, every texel is computed independently, so
//   the row-band threads below produce bit-identical output to a serial run.

#include "procgen/texture.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

#include "core/rng.hpp"

namespace room2::procgen {
namespace {

// ===========================================================================
// small numeric helpers
// ===========================================================================
inline float sqf(float x) { return x * x; }
inline float fracf(float x) { return x - std::floor(x); }
inline int wrapi(int v, int period) { return ((v % period) + period) % period; }
inline uint32_t wrapu(int v, int period) { return static_cast<uint32_t>(wrapi(v, period)); }
// Shortest signed offset on the unit torus, in [-0.5, 0.5].
inline float torusDelta(float d) { return d - std::round(d); }
// Smoothstep that also accepts lo > hi (used constantly for "1 at the core, 0 at the rim").
inline float band(float lo, float hi, float x) { return smoothstepf(lo, hi, x); }

// Runs fn(y0, y1) over contiguous row bands. Texels never depend on each other, so this is a
// pure throughput optimisation and the output is identical to a single-threaded loop.
template <typename Fn>
void parallelBands(uint32_t rows, Fn&& fn) {
    if (rows == 0) return;
    uint32_t hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 1;
    uint32_t bands = std::min<uint32_t>(rows, std::min<uint32_t>(hw, 16u));
    if (bands <= 1 || rows < 64) {
        fn(0u, rows);
        return;
    }
    const uint32_t per = (rows + bands - 1) / bands;
    std::vector<std::thread> pool;
    pool.reserve(bands - 1);
    for (uint32_t b = 1; b < bands; ++b) {
        const uint32_t y0 = std::min(rows, b * per);
        const uint32_t y1 = std::min(rows, y0 + per);
        if (y0 >= y1) break;
        pool.emplace_back([&fn, y0, y1] { fn(y0, y1); });
    }
    fn(0u, std::min(rows, per));
    for (std::thread& t : pool) t.join();
}

// ===========================================================================
// periodic noise
// ===========================================================================
// rng.hpp's tileableNoise2/tileableFbm2 wrap both axes at the same period. Wood grain, fibres
// and machined finishes need different periods per axis (that is what "stretched 20:1" means),
// so the anisotropic generalisation lives here. Wrapping ix modulo px before hashing is what
// makes the lattice periodic; it is the same construction as the header helper.
inline float tileableNoiseAniso(float x, float y, int px, int py, uint32_t seed) {
    const int xi = static_cast<int>(std::floor(x));
    const int yi = static_cast<int>(std::floor(y));
    const float xf = x - static_cast<float>(xi);
    const float yf = y - static_cast<float>(yi);
    const float u = xf * xf * (3.0f - 2.0f * xf);
    const float v = yf * yf * (3.0f - 2.0f * yf);
    auto at = [px, py, seed](int a, int b) {
        return hashToUnit(hashCombine(hashCombine(wrapu(a, px), wrapu(b, py)), seed));
    };
    const float a = at(xi, yi), b = at(xi + 1, yi), c = at(xi, yi + 1), d = at(xi + 1, yi + 1);
    return lerpf(lerpf(a, b, u), lerpf(c, d, u), v);
}

inline float tileableFbmAniso(float x, float y, int px, int py, int octaves, uint32_t seed,
                              float gain = 0.5f) {
    float sum = 0.0f, amp = 1.0f, norm = 0.0f;
    for (int i = 0; i < octaves; ++i) {
        sum += amp * tileableNoiseAniso(x, y, px, py, seed + static_cast<uint32_t>(i) * 7919u);
        norm += amp;
        amp *= gain;
        x *= 2.0f; y *= 2.0f;
        px *= 2; py *= 2;
    }
    return norm > 0.0f ? sum / norm : 0.0f;
}

struct WorleySample {
    float f1 = 0.0f;   // distance to the nearest feature point, in cells
    float f2 = 0.0f;   // distance to the second nearest
    float id = 0.0f;   // stable per-cell random value in [0,1) (stone/knot/thread identity)
};

// Tileable worley. The *unwrapped* cell index is used to place the feature point while the
// *wrapped* index feeds the hash, so the 3x3 neighbourhood search is correct across the seam
// and each cell keeps a single identity. (rng.hpp's worley2 hashes raw cell indices and is
// therefore not periodic - no good for a repeating wall.)
inline WorleySample tileableWorley(float x, float y, int px, int py, uint32_t seed) {
    const int xi = static_cast<int>(std::floor(x));
    const int yi = static_cast<int>(std::floor(y));
    WorleySample out;
    float best = 1e9f, second = 1e9f;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            const int cx = xi + dx, cy = yi + dy;
            const uint32_t h = hashCombine(hashCombine(wrapu(cx, px), wrapu(cy, py)), seed);
            const float jx = hashToUnit(h);
            const float jy = hashToUnit(hashU32(h));
            const float ddx = static_cast<float>(cx) + jx - x;
            const float ddy = static_cast<float>(cy) + jy - y;
            const float d = std::sqrt(ddx * ddx + ddy * ddy);
            if (d < best) {
                second = best;
                best = d;
                out.id = hashToUnit(hashU32(h ^ 0x5BF03635u));
            } else if (d < second) {
                second = d;
            }
        }
    }
    out.f1 = best;
    out.f2 = second;
    return out;
}

// ===========================================================================
// scattered features
// ===========================================================================
// Cracks and scratches are short segments measured on the torus: a mark that runs off the tile
// re-enters on the opposite side, which is exactly what a repeating surface needs.
struct Segment {
    float ax = 0, ay = 0, bx = 0, by = 0;   // endpoints in tile UV
    float cx = 0, cy = 0;                   // bounding-box centre (for the cheap reject test)
    float hx = 0, hy = 0;                   // bounding-box half extents
    float width = 0.002f;                   // half width in UV
    float depth = 0.0f;                     // groove depth in metres
    float rough = 0.0f;                     // roughness delta at the core
    float bright = 0.0f;                    // relative albedo delta at the core
};

struct MarkHit {
    float mask = 0.0f;    // 1 on the centre line, 0 outside the mark
    float depth = 0.0f;   // metres of groove at this texel
    float rough = 0.0f;   // roughness delta
    float bright = 0.0f;  // albedo multiplier delta
};

// Nearest toroidal distance from (u,v) to a segment, testing the 3x3 wrapped copies.
inline float torusSegmentDist(float u, float v, const Segment& s) {
    float best = 1e9f;
    for (int oy = -1; oy <= 1; ++oy) {
        for (int ox = -1; ox <= 1; ++ox) {
            const float ax = s.ax + static_cast<float>(ox), ay = s.ay + static_cast<float>(oy);
            const float ex = s.bx - s.ax, ey = s.by - s.ay;
            const float t = clamp(((u - ax) * ex + (v - ay) * ey) / max2(ex * ex + ey * ey, 1e-9f),
                                  0.0f, 1.0f);
            const float px = ax + ex * t - u, py = ay + ey * t - v;
            best = min2(best, std::sqrt(px * px + py * py));
        }
    }
    return best;
}

inline MarkHit marksAt(const std::vector<Segment>& list, float u, float v) {
    MarkHit hit;
    for (const Segment& s : list) {
        // Conservative reject on the torus: if the nearest wrapped copy of the bounding box is
        // already further away than `width`, the whole 3x3 search would be wasted.
        const float du = torusDelta(u - s.cx), dv = torusDelta(v - s.cy);
        if (std::fabs(du) > s.hx + s.width || std::fabs(dv) > s.hy + s.width) continue;
        const float d = torusSegmentDist(u, v, s);
        if (d >= s.width) continue;
        const float w = 1.0f - d / s.width;
        hit.mask = max2(hit.mask, w);
        hit.depth = max2(hit.depth, w * s.depth);
        hit.rough = max2(hit.rough, w * s.rough);
        hit.bright = max2(hit.bright, w * s.bright);
    }
    return hit;
}

struct Blob {
    float x = 0, y = 0;      // centre in tile UV
    float radius = 0.05f;    // UV radius
    float depth = 0.0f;      // metres (positive = dent / groove)
    float rough = 0.0f;      // roughness delta at the centre
    float bright = 0.0f;     // albedo delta
    float rings = 0.0f;      // >0: concentric ridge count (fingerprint style smudges)
};

// Toroidal radial falloff of a blob: 1 at the centre, 0 at `radius`.
inline float blobFalloff(float u, float v, const Blob& b) {
    const float dx = torusDelta(u - b.x), dy = torusDelta(v - b.y);
    const float d = std::sqrt(dx * dx + dy * dy);
    return band(b.radius, 0.0f, d);
}

// A mark that runs *along* a tile edge for a long stretch is bad news twice over: once the
// texture repeats it shows up as a regular grid of parallel lines, and it dominates the
// wrap-around difference because the seam rows are the only ones that straddle it. What matters
// is not the mark's bounding box but how much of its *length* hugs an edge, so that is what is
// measured here. Such marks are translated half a tile - a torus translation, so the map stays
// perfectly seamless. Marks that merely cross an edge at an angle are left alone: those are
// exactly the features that make a tiling texture read as non-repeating.
inline bool hugsVSeam(float u, float v, float margin) {
    (void)u;
    return std::fabs(v - std::round(v)) < margin;
}
inline bool hugsUSeam(float u, float v, float margin) {
    (void)v;
    return std::fabs(u - std::round(u)) < margin;
}

void appendSegment(std::vector<Segment>& out, float ax, float ay, float bx, float by, float width,
                   float depth, float rough, float bright) {
    Segment s;
    s.ax = ax; s.ay = ay; s.bx = bx; s.by = by;
    s.cx = 0.5f * (ax + bx);
    s.cy = 0.5f * (ay + by);
    s.hx = 0.5f * std::fabs(bx - ax);
    s.hy = 0.5f * std::fabs(by - ay);
    s.width = width;
    s.depth = depth;
    s.rough = rough;
    s.bright = bright;
    out.push_back(s);
}

// Straight scratches: random position, random direction, several length classes. Widths and
// depths are given in metres; when the mark is thinner than the texel grid it is widened to
// ~one texel and (with slopePreserve = 1) the depth is scaled with it, so a 4 um wide scratch
// still cuts a visible groove instead of dissolving into the bilinear filter.
std::vector<Segment> makeScratches(Rng& rng, int count, float lenMin, float lenMax, float widthM,
                                   float depthM, float rough, float bright, float tpm, uint32_t size,
                                   float minTexels = 2.0f, float slopePreserve = 1.0f) {
    std::vector<Segment> out;
    out.reserve(static_cast<size_t>(count));
    const float wantTexels = max2(widthM * tpm, 1e-4f);
    // A groove narrower than ~2 texels has no measurable slope: the central differences on both
    // sides fall outside it, so the mark would vanish from the normal map entirely.
    const float useTexels = max2(minTexels, wantTexels);
    const float widthUV = useTexels / static_cast<float>(size);
    const float depth = depthM * lerpf(1.0f, useTexels / wantTexels, slopePreserve);
    for (int i = 0; i < count; ++i) {
        const float ang = rng.nextFloat() * TWO_PI;
        const float len = rng.range(lenMin, lenMax);
        float x = rng.nextFloat(), y = rng.nextFloat();
        float dx = std::cos(ang) * len, dy = std::sin(ang) * len;
        const float margin = widthUV * 2.0f + 0.015f;
        const float mx = x + dx * 0.5f, my = y + dy * 0.5f;
        if (std::fabs(dx) > std::fabs(dy) && hugsVSeam(mx, my, margin)) y += 0.5f;
        if (std::fabs(dy) > std::fabs(dx) && hugsUSeam(mx, my, margin)) x += 0.5f;
        // Slight per-scratch variation so a batch of identical marks does not look stamped.
        const float w = widthUV * rng.range(0.6f, 1.6f);
        const float d = depth * rng.range(0.5f, 1.2f);
        appendSegment(out, x, y, x + dx, y + dy, w, d, rough * rng.range(0.6f, 1.3f),
                      bright * rng.range(0.6f, 1.3f));
    }
    return out;
}

// Hairline cracks: a random walk of short segments, so the result meanders like a shrinkage
// crack instead of looking like a ruler line. Same physical-width handling as the scratches.
std::vector<Segment> makeCracks(Rng& rng, int count, int segments, float stepMin, float stepMax,
                                float widthM, float depthM, float tpm, uint32_t size,
                                float minTexels = 2.0f) {
    std::vector<Segment> out;
    const float wantTexels = max2(widthM * tpm, 1e-4f);
    const float useTexels = max2(minTexels, wantTexels);
    const float widthUV = useTexels / static_cast<float>(size);
    const float depth = depthM * (useTexels / wantTexels);
    for (int c = 0; c < count; ++c) {
        std::vector<Segment> chain;
        float x = rng.nextFloat(), y = rng.nextFloat();
        float ang = rng.nextFloat() * TWO_PI;
        float lenNearV = 0.0f, lenNearU = 0.0f, lenTotal = 0.0f;
        for (int s = 0; s < segments; ++s) {
            ang += rng.symmetric() * 0.9f;   // wander
            const float len = rng.range(stepMin, stepMax);
            const float nx = x + std::cos(ang) * len;
            const float ny = y + std::sin(ang) * len;
            const float w = widthUV * rng.range(0.7f, 1.5f) * (1.0f - 0.35f * static_cast<float>(s) /
                                                                         static_cast<float>(segments));
            const float d = depth * rng.range(0.6f, 1.2f);
            appendSegment(chain, x, y, nx, ny, w, d, 0.0f, 0.0f);
            const float segLen = std::sqrt((nx - x) * (nx - x) + (ny - y) * (ny - y));
            lenTotal += segLen;
            if (hugsVSeam(0.0f, 0.5f * (y + ny), widthUV * 2.0f + 0.02f)) lenNearV += segLen;
            if (hugsUSeam(0.5f * (x + nx), 0.0f, widthUV * 2.0f + 0.02f)) lenNearU += segLen;
            x = nx;
            y = ny;
        }
        float shiftU = 0.0f, shiftV = 0.0f;
        if (lenTotal > 0.0f && lenNearV > 0.30f * lenTotal && lenNearV >= lenNearU) shiftV = 0.5f;
        else if (lenTotal > 0.0f && lenNearU > 0.30f * lenTotal) shiftU = 0.5f;
        for (Segment& seg : chain) {
            seg.ax += shiftU; seg.bx += shiftU; seg.cx += shiftU;
            seg.ay += shiftV; seg.by += shiftV; seg.cy += shiftV;
            out.push_back(seg);
        }
    }
    return out;
}

// ===========================================================================
// per-material bake buffers
// ===========================================================================
struct Bake {
    uint32_t size = 0;
    uint32_t seed = 0;
    float tpm = 512.0f;     // texels per metre
    float tileM = 2.0f;     // physical size of the tile
    float inv = 0.0f;       // 1 / size
    std::vector<float> rgb;      // 3 * n linear albedo
    std::vector<float> height;   // n metres
    std::vector<float> rough;    // n
    std::vector<float> metal;    // n
    std::vector<float> ao;       // n multiplicative cavity term

    Bake(uint32_t n, const SurfaceParams& p, uint32_t salt)
        : size(n),
          seed(hashCombine(p.seed, salt)),
          tpm(p.texelsPerMetre > 1.0f ? p.texelsPerMetre : 512.0f),
          tileM(static_cast<float>(n) / (p.texelsPerMetre > 1.0f ? p.texelsPerMetre : 512.0f)),
          inv(n > 0 ? 1.0f / static_cast<float>(n) : 0.0f),
          rgb(static_cast<size_t>(n) * n * 3, 0.0f),
          height(static_cast<size_t>(n) * n, 0.0f),
          rough(static_cast<size_t>(n) * n, 0.5f),
          metal(static_cast<size_t>(n) * n, 0.0f),
          ao(static_cast<size_t>(n) * n, 1.0f) {}

    size_t idx(uint32_t x, uint32_t y) const { return static_cast<size_t>(y) * size + x; }

    // Lattice period (in cells across the tile) for a feature of `featureMetres`. `widen`
    // reports how far the feature had to be stretched to stay representable: multiply the
    // relief amplitude by it to preserve the real slope (see the file header). `octaves`
    // reserves room for the fbm's frequency doubling so the finest octave never aliases, and
    // `minTexels` keeps *structured* patterns (stipple grids, weave, chequering) at a few
    // texels per cell - a 1.5 mm pitch at 512 texels/m would otherwise collapse to one texel
    // per cell and alias into flat noise instead of a pattern.
    int cells(float featureMetres, int octaves, float& widen, int minTexels = 1) const {
        const float req = tileM / max2(featureMetres, 1e-6f);
        const int aa = max2(1, static_cast<int>(size) >> (octaves - 1));
        const int detail = max2(1, static_cast<int>(size) / max2(1, minTexels));
        const int maxP = min2(aa, detail);
        const int p = clamp(static_cast<int>(std::lround(req)), 1, maxP);
        widen = clamp(req / static_cast<float>(p), 0.05f, 64.0f);
        return p;
    }
    int cells(float featureMetres, int octaves = 1) const {
        float unused = 1.0f;
        return cells(featureMetres, octaves, unused);
    }

    // Number of scattered features (cracks, scratches, knots, smudges) for this tile, from a
    // density per square metre. Scattering a fixed count per *tile* would make the features much
    // denser - and much wider relative to the tile - as soon as the tile covers less area, which
    // is both unphysical and what pushed sparse features onto the tile seam. A fractional
    // expectation is resolved with one hash draw, so the result stays deterministic.
    int featureCount(float perSquareMetre, uint32_t salt) const {
        const float expected = perSquareMetre * tileM * tileM;
        int n = static_cast<int>(expected);
        const float frac = expected - static_cast<float>(n);
        if (frac > 0.0f && hashToUnit(hashCombine(seed, salt)) < frac) ++n;
        return n;
    }
};

// Horizon-based ambient occlusion over 8 fixed directions. For every direction we look for the
// steepest elevation angle to a neighbour within `radius` texels and take its sine as the
// occlusion of that direction; the average over the 8 directions is the AO term. `strength`
// converts a height difference into a slope tangent (pass texelsPerMetre for physically
// correct AO, more to exaggerate micro cavities the way a cavity map would).
void horizonAo(const std::vector<float>& h, uint32_t width, uint32_t heightPx, float strength,
               int radius, std::vector<float>& out) {
    const size_t n = static_cast<size_t>(width) * heightPx;
    out.assign(n, 1.0f);
    if (n == 0 || h.size() < n) return;
    static const int DX[8] = {1, 1, 0, -1, -1, -1, 0, 1};
    static const int DY[8] = {0, 1, 1, 1, 0, -1, -1, -1};
    // A diagonal step covers sqrt(2) texels, so the horizon tangent must divide by the true
    // distance - otherwise the four diagonal directions would be over-weighted.
    static const float DSTEP[8] = {1.0f, 1.41421356f, 1.0f, 1.41421356f,
                                   1.0f, 1.41421356f, 1.0f, 1.41421356f};
    const int r = clamp(radius, 1, 64);
    const int w = static_cast<int>(width), hh = static_cast<int>(heightPx);
    parallelBands(heightPx, [&](uint32_t y0, uint32_t y1) {
        for (uint32_t y = y0; y < y1; ++y) {
            const int yi = static_cast<int>(y);
            for (uint32_t x = 0; x < width; ++x) {
                const int xi = static_cast<int>(x);
                const float h0 = h[static_cast<size_t>(y) * width + x];
                float occ = 0.0f;
                for (int d = 0; d < 8; ++d) {
                    float maxTan = 0.0f;
                    for (int t = 1; t <= r; ++t) {
                        const int sx = wrapi(xi + DX[d] * t, w);
                        const int sy = wrapi(yi + DY[d] * t, hh);
                        const float dh = (h[static_cast<size_t>(sy) * width + static_cast<size_t>(sx)] - h0) *
                                         strength;
                        const float tan = dh / (static_cast<float>(t) * DSTEP[d]);
                        if (tan > maxTan) maxTan = tan;
                    }
                    occ += maxTan / std::sqrt(1.0f + maxTan * maxTan);   // sin(elevation)
                }
                out[static_cast<size_t>(y) * width + x] = clamp(1.0f - occ * 0.125f, 0.0f, 1.0f);
            }
        }
    });
}

// Turns the bake buffers into a TextureSet: base colour, normal from the height field, and an
// ORM map whose occlusion combines the horizon term with the material's own cavity term.
TextureSet finishBake(const Bake& b, float normalStrength, float aoStrength, int aoRadius) {
    TextureSet set;
    if (b.size == 0) return set;
    const size_t n = static_cast<size_t>(b.size) * b.size;
    set.baseColor.resize(b.size, b.size);
    for (size_t i = 0; i < n; ++i) {
        set.baseColor.pixels[i * 4 + 0] = clamp(b.rgb[i * 3 + 0], 0.0f, 1.0f);
        set.baseColor.pixels[i * 4 + 1] = clamp(b.rgb[i * 3 + 1], 0.0f, 1.0f);
        set.baseColor.pixels[i * 4 + 2] = clamp(b.rgb[i * 3 + 2], 0.0f, 1.0f);
        set.baseColor.pixels[i * 4 + 3] = 1.0f;
    }
    set.normal = normalFromHeight(b.height, b.size, b.size, normalStrength);
    std::vector<float> horizon;
    horizonAo(b.height, b.size, b.size, aoStrength, aoRadius, horizon);
    std::vector<float> ao(n);
    for (size_t i = 0; i < n; ++i) ao[i] = clamp(horizon[i] * b.ao[i], 0.0f, 1.0f);
    set.orm = packOrm(ao, b.rough, b.metal, b.size, b.size);
    set.hasNormal = true;
    set.hasOrm = true;
    return set;
}

// ===========================================================================
// PNG writer internals (no external libraries: stored deflate blocks + own CRC/Adler)
// ===========================================================================
uint32_t crc32Bytes(const uint8_t* data, size_t len, uint32_t crc = 0xFFFFFFFFu) {
    static const std::vector<uint32_t> table = [] {
        std::vector<uint32_t> t(256);
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            t[i] = c;
        }
        return t;
    }();
    for (size_t i = 0; i < len; ++i) crc = table[(crc ^ data[i]) & 0xFFu] ^ (crc >> 8);
    return crc;
}

void putU32be(std::vector<uint8_t>& out, uint32_t v) {
    out.push_back(static_cast<uint8_t>(v >> 24));
    out.push_back(static_cast<uint8_t>(v >> 16));
    out.push_back(static_cast<uint8_t>(v >> 8));
    out.push_back(static_cast<uint8_t>(v));
}

void putChunk(std::vector<uint8_t>& out, const char type[4], const uint8_t* data, size_t len) {
    putU32be(out, static_cast<uint32_t>(len));
    const size_t start = out.size();
    out.insert(out.end(), type, type + 4);
    if (len) out.insert(out.end(), data, data + len);
    const uint32_t crc = crc32Bytes(out.data() + start, out.size() - start) ^ 0xFFFFFFFFu;
    putU32be(out, crc);
}

// zlib stream with "stored" deflate blocks: legal, trivial and easy to verify by hand.
void appendZlibStored(std::vector<uint8_t>& out, const std::vector<uint8_t>& raw) {
    out.push_back(0x78);   // CM=8 (deflate), CINFO=7 (32K window)
    out.push_back(0x01);   // FCHECK so that (0x78<<8|0x01) % 31 == 0, no preset dictionary
    size_t pos = 0;
    const size_t total = raw.size();
    if (total == 0) {
        out.push_back(0x01);   // final, stored
        out.push_back(0x00); out.push_back(0x00);
        out.push_back(0xFF); out.push_back(0xFF);
    }
    while (pos < total) {
        const size_t len = std::min<size_t>(65535, total - pos);
        const bool last = (pos + len) >= total;
        out.push_back(last ? 0x01 : 0x00);
        out.push_back(static_cast<uint8_t>(len & 0xFF));
        out.push_back(static_cast<uint8_t>((len >> 8) & 0xFF));
        const uint16_t nlen = static_cast<uint16_t>(~static_cast<uint16_t>(len));
        out.push_back(static_cast<uint8_t>(nlen & 0xFF));
        out.push_back(static_cast<uint8_t>((nlen >> 8) & 0xFF));
        out.insert(out.end(), raw.begin() + static_cast<long>(pos),
                   raw.begin() + static_cast<long>(pos + len));
        pos += len;
    }
    uint32_t a = 1, bsum = 0;
    for (uint8_t byte : raw) {
        a = (a + byte) % 65521u;
        bsum = (bsum + a) % 65521u;
    }
    putU32be(out, (bsum << 16) | a);   // Adler-32, big endian
}

bool writePngFile(const std::string& path, uint32_t width, uint32_t heightPx, uint8_t colorType,
                  const std::vector<uint8_t>& raw, bool srgbChunk) {
    std::vector<uint8_t> file;
    file.reserve(raw.size() + 256);
    static const uint8_t kSignature[8] = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    file.insert(file.end(), kSignature, kSignature + 8);

    std::vector<uint8_t> ihdr;
    putU32be(ihdr, width);
    putU32be(ihdr, heightPx);
    ihdr.push_back(8);            // bit depth
    ihdr.push_back(colorType);    // 0 = grey, 6 = RGBA
    ihdr.push_back(0);            // compression: deflate
    ihdr.push_back(0);            // filter method 0
    ihdr.push_back(0);            // no interlace
    putChunk(file, "IHDR", ihdr.data(), ihdr.size());

    if (srgbChunk) {
        const uint8_t intent = 0;   // perceptual
        putChunk(file, "sRGB", &intent, 1);
    }

    std::vector<uint8_t> z;
    appendZlibStored(z, raw);
    putChunk(file, "IDAT", z.data(), z.size());
    putChunk(file, "IEND", nullptr, 0);

    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {
        // The dump mode may be pointed at a folder that does not exist yet; create it once.
        std::error_code ec;
        const std::filesystem::path p(path);
        if (!p.parent_path().empty()) std::filesystem::create_directories(p.parent_path(), ec);
        f = std::fopen(path.c_str(), "wb");
        if (!f) return false;
    }
    const size_t written = std::fwrite(file.data(), 1, file.size(), f);
    const bool ok = (written == file.size());
    std::fclose(f);
    return ok;
}

}  // namespace

// ===========================================================================
// public utilities
// ===========================================================================
Vec4 TextureData::bilinear(float u, float v) const {
    if (!valid()) return Vec4(0, 0, 0, 1);
    // Sample positions are texel centres, and out-of-range taps wrap: the maps are tileable,
    // so wrapping is the correct continuation at the border.
    const float fx = u * static_cast<float>(width) - 0.5f;
    const float fy = v * static_cast<float>(height) - 0.5f;
    const int x0 = static_cast<int>(std::floor(fx));
    const int y0 = static_cast<int>(std::floor(fy));
    const float tx = fx - static_cast<float>(x0);
    const float ty = fy - static_cast<float>(y0);
    const Vec4 a = sample(x0, y0), b = sample(x0 + 1, y0);
    const Vec4 c = sample(x0, y0 + 1), d = sample(x0 + 1, y0 + 1);
    return lerp(lerp(a, b, tx), lerp(c, d, tx), ty);
}

TextureData flatNormal(uint32_t size) {
    TextureData out;
    out.resize(size, size, Vec4(0.5f, 0.5f, 1.0f, 1.0f));
    return out;
}

TextureData normalFromHeight(const std::vector<float>& height, uint32_t width, uint32_t heightPx,
                             float strength) {
    TextureData out;
    if (width == 0 || heightPx == 0 || height.size() < static_cast<size_t>(width) * heightPx) {
        return out;
    }
    out.resize(width, heightPx, Vec4(0.5f, 0.5f, 1.0f, 1.0f));
    const int w = static_cast<int>(width), h = static_cast<int>(heightPx);
    parallelBands(heightPx, [&](uint32_t y0, uint32_t y1) {
        for (uint32_t y = y0; y < y1; ++y) {
            const int yi = static_cast<int>(y);
            const int yUp = wrapi(yi + 1, h), yDn = wrapi(yi - 1, h);
            for (uint32_t x = 0; x < width; ++x) {
                const int xi = static_cast<int>(x);
                const int xL = wrapi(xi - 1, w), xR = wrapi(xi + 1, w);
                // Central differences on the wrapped grid; `strength` is the slope scale
                // (metres per height unit, divided by the texel spacing).
                const float dhx = (height[static_cast<size_t>(yi) * width + static_cast<size_t>(xR)] -
                                   height[static_cast<size_t>(yi) * width + static_cast<size_t>(xL)]) *
                                  0.5f * strength;
                const float dhy = (height[static_cast<size_t>(yUp) * width + x] -
                                   height[static_cast<size_t>(yDn) * width + x]) *
                                  0.5f * strength;
                // Tangent space: +X = U, +Y = V, +Z = out of the surface, R/G = the normal's
                // X/Y component (OpenGL / glTF green-up). A surface that rises along +U leans
                // towards -X, hence the minus signs; normalising keeps X^2+Y^2 < 1 so the
                // stored Z stays real.
                const Vec3 nrm = normalize(Vec3(-dhx, -dhy, 1.0f));
                out.setTexel(x, y, Vec4(nrm.x * 0.5f + 0.5f, nrm.y * 0.5f + 0.5f, nrm.z * 0.5f + 0.5f,
                                        1.0f));
            }
        }
    });
    return out;
}

TextureData aoFromHeight(const std::vector<float>& height, uint32_t width, uint32_t heightPx,
                         float strength, int radius) {
    TextureData out;
    if (width == 0 || heightPx == 0 || height.size() < static_cast<size_t>(width) * heightPx) {
        return out;
    }
    std::vector<float> ao;
    horizonAo(height, width, heightPx, strength, radius, ao);
    out.resize(width, heightPx, Vec4(1, 1, 1, 1));
    for (uint32_t y = 0; y < heightPx; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            // R = G = B = occlusion (1 = fully open); A = 1 so the map can be packed directly.
            const float a = ao[static_cast<size_t>(y) * width + x];
            out.setTexel(x, y, Vec4(a, a, a, 1.0f));
        }
    }
    return out;
}

TextureData packOrm(const std::vector<float>& ao, const std::vector<float>& roughness,
                    const std::vector<float>& metallic, uint32_t width, uint32_t heightPx) {
    TextureData out;
    const size_t n = static_cast<size_t>(width) * heightPx;
    if (n == 0 || ao.size() < n || roughness.size() < n || metallic.size() < n) return out;
    out.resize(width, heightPx, Vec4(1, 1, 1, 1));
    for (size_t i = 0; i < n; ++i) {
        float* p = &out.pixels[i * 4];
        p[0] = clamp(ao[i], 0.0f, 1.0f);
        p[1] = clamp(roughness[i], 0.0f, 1.0f);
        p[2] = clamp(metallic[i], 0.0f, 1.0f);
        p[3] = 1.0f;
    }
    return out;
}

std::vector<uint8_t> toSrgb8(const TextureData& src) {
    std::vector<uint8_t> out;
    if (!src.valid()) return out;
    const size_t n = static_cast<size_t>(src.width) * src.height;
    out.resize(n * 4);
    for (size_t i = 0; i < n; ++i) {
        const float* p = &src.pixels[i * 4];
        const Vec3 lin = clampv(Vec3(p[0], p[1], p[2]), 0.0f, 1.0f);
        const Vec3 enc = linearToSrgb(lin);   // sRGB transfer curve, then quantise
        out[i * 4 + 0] = static_cast<uint8_t>(std::lround(clamp(enc.x, 0.0f, 1.0f) * 255.0f));
        out[i * 4 + 1] = static_cast<uint8_t>(std::lround(clamp(enc.y, 0.0f, 1.0f) * 255.0f));
        out[i * 4 + 2] = static_cast<uint8_t>(std::lround(clamp(enc.z, 0.0f, 1.0f) * 255.0f));
        out[i * 4 + 3] = static_cast<uint8_t>(std::lround(clamp(p[3], 0.0f, 1.0f) * 255.0f));
    }
    return out;
}

std::vector<uint8_t> toLinear8(const TextureData& src) {
    std::vector<uint8_t> out;
    if (!src.valid()) return out;
    const size_t n = static_cast<size_t>(src.width) * src.height;
    out.resize(n * 4);
    for (size_t i = 0; i < n; ++i) {
        const float* p = &src.pixels[i * 4];
        for (int c = 0; c < 4; ++c) {   // data maps: no transfer curve, straight quantisation
            out[i * 4 + static_cast<size_t>(c)] =
                static_cast<uint8_t>(std::lround(clamp(p[c], 0.0f, 1.0f) * 255.0f));
        }
    }
    return out;
}

TextureData downsample(const TextureData& src) {
    TextureData out;
    if (!src.valid()) return out;
    const uint32_t w = max2(1u, src.width / 2), h = max2(1u, src.height / 2);
    out.resize(w, h);
    for (uint32_t y = 0; y < h; ++y) {
        const uint32_t y0 = min2(y * 2, src.height - 1);
        const uint32_t y1 = min2(y * 2 + 1, src.height - 1);
        for (uint32_t x = 0; x < w; ++x) {
            const uint32_t x0 = min2(x * 2, src.width - 1);
            const uint32_t x1 = min2(x * 2 + 1, src.width - 1);
            const Vec4 s = src.texel(x0, y0) + src.texel(x1, y0) + src.texel(x0, y1) + src.texel(x1, y1);
            out.setTexel(x, y, s * 0.25f);
        }
    }
    return out;
}

// ===========================================================================
// materials
// ===========================================================================

// ---------------------------------------------------------------- plaster wall
// roller-applied emulsion over plaster: broad paint unevenness, a fine grain from the roller
// nap and the filler, sparse round pits (air bubbles / trowel digs) and a couple of hairline
// cracks. Relief is sub-millimetre, so the normal map stays very subtle.
TextureSet plasterWall(uint32_t size, const SurfaceParams& params) {
    Bake b(size, params, 0x706C6173u);   // "plas"
    const int pBlotch = b.cells(0.45f, 3);
    const int pGrain = b.cells(0.020f, 4);
    const int pRoll = b.cells(0.028f, 3);      // across the roller direction: fine stipple
    const int pRollX = b.cells(0.22f, 3);      // along the roller direction: stretched
    const int pPit = b.cells(0.055f);
    const int pSheen = b.cells(0.09f, 3);      // paint sheen patchiness
    Rng rng(b.seed ^ 0xC0FFEEu);
    const std::vector<Segment> cracks =
        makeCracks(rng, b.featureCount(0.75f, 101u), 7, 0.05f, 0.11f, 0.0009f, 0.0007f, b.tpm, size);
    const float inv = b.inv;
    parallelBands(size, [&](uint32_t y0, uint32_t y1) {
        for (uint32_t y = y0; y < y1; ++y) {
            const float v = (static_cast<float>(y) + 0.5f) * inv;
            for (uint32_t x = 0; x < size; ++x) {
                const float u = (static_cast<float>(x) + 0.5f) * inv;
                const float blotch = tileableFbm2(u * pBlotch, v * pBlotch, pBlotch, 3, b.seed + 11u);
                const float grain = tileableFbm2(u * pGrain, v * pGrain, pGrain, 4, b.seed + 23u);
                const float roller =
                    tileableFbmAniso(u * pRollX, v * pRoll, pRollX, pRoll, 3, b.seed + 31u);
                const WorleySample pit = tileableWorley(u * pPit, v * pPit, pPit, pPit, b.seed + 41u);
                const float pitMask = band(0.11f, 0.03f, pit.f1);   // a few % of the area
                const float sheen = tileableFbm2(u * pSheen, v * pSheen, pSheen, 3, b.seed + 47u);
                const MarkHit cr = marksAt(cracks, u, v);
                const float hole = max2(pitMask, cr.mask);
                const float mottle = 0.042f * (blotch - 0.5f) + 0.020f * (grain - 0.5f) +
                                     0.014f * (roller - 0.5f);
                const float h = 0.00012f * (grain - 0.5f) * 2.0f + 0.00016f * (roller - 0.5f) +
                                0.00005f * (blotch - 0.5f) - 0.00040f * pitMask - cr.depth;
                const Vec3 alb = Vec3(0.762f, 0.744f, 0.706f) *
                                 (1.0f + mottle - 0.07f * pitMask - 0.13f * cr.mask);
                // Emulsion sheen follows the roller texture and dries unevenly: the grain and
                // the mid-scale sheen patches move roughness by a few percent either way.
                const float rough = 0.900f + 0.050f * (grain - 0.5f) * 2.0f +
                                    0.032f * (sheen - 0.5f) * 2.0f + 0.018f * (blotch - 0.5f) +
                                    0.030f * pitMask + 0.045f * cr.mask;
                const size_t i = b.idx(x, y);
                b.rgb[i * 3 + 0] = alb.x;
                b.rgb[i * 3 + 1] = alb.y;
                b.rgb[i * 3 + 2] = alb.z;
                b.height[i] = h;
                b.rough[i] = clamp(rough, 0.83f, 0.97f);
                b.metal[i] = 0.0f;
                // Cavity term: pits and cracks hold dust and are always a bit darker.
                b.ao[i] = clamp((1.0f - 0.22f * hole) * (0.99f + 0.02f * (blotch - 0.5f)), 0.0f, 1.0f);
            }
        }
    });
    return finishBake(b, b.tpm, b.tpm * 3.0f, 12);
}

// ---------------------------------------------------------------- painted ceiling
// Same family as the wall but flatter, cooler and slightly brighter, with the roller's pass
// overlap (bands of slightly different stipple density) showing through the paint.
TextureSet paintedCeiling(uint32_t size, const SurfaceParams& params) {
    Bake b(size, params, 0x6365696Cu);   // "ceil"
    const int pBlotch = b.cells(0.55f, 3);
    const int pStipple = b.cells(0.014f, 3);
    const int pPass = b.cells(0.30f, 3);
    const int pRoll = b.cells(0.024f, 3);
    const int pRollX = b.cells(0.26f, 3);
    const float inv = b.inv;
    parallelBands(size, [&](uint32_t y0, uint32_t y1) {
        for (uint32_t y = y0; y < y1; ++y) {
            const float v = (static_cast<float>(y) + 0.5f) * inv;
            for (uint32_t x = 0; x < size; ++x) {
                const float u = (static_cast<float>(x) + 0.5f) * inv;
                const float blotch = tileableFbm2(u * pBlotch, v * pBlotch, pBlotch, 3, b.seed + 11u);
                const float stipple = tileableFbm2(u * pStipple, v * pStipple, pStipple, 3, b.seed + 17u);
                const float roll = tileableFbmAniso(u * pRollX, v * pRoll, pRollX, pRoll, 3, b.seed + 19u);
                // Roller pass overlap: a soft band pattern (periodic in u, ie. across the
                // ceiling) whose edges wander with a periodic fbm.
                const float passF = tileableFbm2(u * pPass, v * pPass, pPass, 3, b.seed + 29u);
                const float passT = fracf(u * 4.0f + 0.45f * (passF - 0.5f));
                const float pass = band(0.0f, 0.30f, passT) * band(1.0f, 0.70f, passT);
                const float mottle = 0.016f * (blotch - 0.5f) + 0.010f * (stipple - 0.5f) +
                                     0.010f * (pass - 0.5f);
                const float relief = 0.7f + 0.5f * pass;   // more nap texture inside a pass
                const float h = 0.00014f * (stipple - 0.5f) * 2.0f * relief +
                                0.00016f * (roll - 0.5f) * relief + 0.00003f * (blotch - 0.5f);
                const Vec3 alb = Vec3(0.792f, 0.797f, 0.806f) * (1.0f + mottle);
                const float rough = 0.900f + 0.042f * (stipple - 0.5f) * 2.0f +
                                    0.020f * (blotch - 0.5f) + 0.030f * (pass - 0.5f) * 2.0f;
                const size_t i = b.idx(x, y);
                b.rgb[i * 3 + 0] = alb.x;
                b.rgb[i * 3 + 1] = alb.y;
                b.rgb[i * 3 + 2] = alb.z;
                b.height[i] = h;
                b.rough[i] = clamp(rough, 0.84f, 0.96f);
                b.metal[i] = 0.0f;
                b.ao[i] = clamp(1.0f - 0.10f * (1.0f - stipple) * relief, 0.0f, 1.0f);
            }
        }
    });
    return finishBake(b, b.tpm, b.tpm * 4.0f, 12);
}

// ---------------------------------------------------------------- concrete floor
// Cast concrete: broad colour variation from the pour, dense aggregate speckle, a rounded
// stone mosaic with mortar lines, air-bubble pores, faint trowel swirls and a few shrinkage
// cracks. Patches that were power-floated are noticeably smoother (lower roughness).
TextureSet concreteFloor(uint32_t size, const SurfaceParams& params) {
    Bake b(size, params, 0x636F6E63u);   // "conc"
    const int pBroad = b.cells(1.10f, 3);     // pour-scale colour variation
    const int pMottle = b.cells(0.24f, 4);
    float wAgg = 1.0f, wSpeck = 1.0f, wPore = 1.0f;
    const int pAgg = b.cells(0.011f, 1, wAgg);       // ~1 cm aggregate stones
    const int pSpeck = b.cells(0.004f, 2, wSpeck);   // fine sand speckle
    const int pPore = b.cells(0.030f, 1, wPore);     // air bubbles
    const int pSwirl = b.cells(0.45f, 3);
    Rng rng(b.seed ^ 0xBEEF01u);
    const std::vector<Segment> cracks =
        makeCracks(rng, b.featureCount(1.25f, 103u), 9, 0.04f, 0.10f, 0.0012f, 0.0012f, b.tpm, size);
    const float inv = b.inv;
    parallelBands(size, [&](uint32_t y0, uint32_t y1) {
        for (uint32_t y = y0; y < y1; ++y) {
            const float v = (static_cast<float>(y) + 0.5f) * inv;
            for (uint32_t x = 0; x < size; ++x) {
                const float u = (static_cast<float>(x) + 0.5f) * inv;
                const float broad = tileableFbm2(u * pBroad, v * pBroad, pBroad, 3, b.seed + 11u);
                const float mottle = tileableFbm2(u * pMottle, v * pMottle, pMottle, 4, b.seed + 13u);
                const float speck = tileableFbm2(u * pSpeck, v * pSpeck, pSpeck, 2, b.seed + 17u);
                const float swirlF = tileableFbm2(u * pSwirl, v * pSwirl, pSwirl, 3, b.seed + 19u);
                const WorleySample agg = tileableWorley(u * pAgg, v * pAgg, pAgg, pAgg, b.seed + 23u);
                const WorleySample pore = tileableWorley(u * pPore, v * pPore, pPore, pPore, b.seed + 29u);
                // Mortar line between stones: small where two cells meet.
                const float seam = band(0.0f, 0.30f, agg.f2 - agg.f1);
                const float stone = band(0.55f, 0.15f, agg.f1);
                const float poreMask = band(0.10f, 0.02f, pore.f1);
                const MarkHit cr = marksAt(cracks, u, v);
                // Trowel swirl: soft periodic bands whose edges are pushed around by fbm.
                const float swT = fracf(v * 3.0f + 0.55f * (swirlF - 0.5f));
                const float swirl = band(0.0f, 0.35f, swT) * band(1.0f, 0.65f, swT);
                // Power-floated (polished) patches.
                const float polish = band(0.52f, 0.72f, mottle * 0.6f + broad * 0.4f);
                const float sp = clamp((speck - 0.5f) * 2.6f + 0.5f, 0.0f, 1.0f);
                const float stoneTone = 1.0f + 0.13f * (agg.id - 0.5f);
                const float h = 0.00030f * wAgg * (stone - 0.45f) - 0.00045f * wAgg * (1.0f - seam) +
                                0.00018f * wSpeck * (sp - 0.5f) - 0.00040f * poreMask -
                                0.00020f * swirl + 0.00008f * (mottle - 0.5f) - cr.depth;
                Vec3 alb(0.258f, 0.256f, 0.250f);
                alb = alb * stoneTone * (1.0f - 0.16f * (1.0f - seam)) *
                      (1.0f + 0.10f * (broad - 0.5f) + 0.07f * (mottle - 0.5f)) *
                      (1.0f + 0.09f * (sp - 0.5f));
                alb = alb * (1.0f - 0.10f * poreMask - 0.40f * cr.mask - 0.05f * polish);
                const float rough = 0.740f - 0.13f * polish + 0.05f * (mottle - 0.5f) +
                                    0.03f * (sp - 0.5f) + 0.06f * poreMask + 0.05f * (1.0f - seam) +
                                    0.07f * cr.mask;
                const size_t i = b.idx(x, y);
                b.rgb[i * 3 + 0] = alb.x;
                b.rgb[i * 3 + 1] = alb.y;
                b.rgb[i * 3 + 2] = alb.z;
                b.height[i] = h;
                b.rough[i] = clamp(rough, 0.55f, 0.86f);
                b.metal[i] = 0.0f;
                b.ao[i] = clamp((1.0f - 0.20f * poreMask - 0.25f * cr.mask -
                                 0.10f * (1.0f - seam) - 0.06f * (1.0f - stone)),
                                0.0f, 1.0f);
            }
        }
    });
    return finishBake(b, b.tpm, b.tpm * 3.0f, 14);
}

// ---------------------------------------------------------------- wood engine
// One model serves the plank floor, the table top and the varnished legs:
//   * growth rings: the level sets of a warped phase coordinate. The phase runs across the
//     plank, so the rings come out stretched along it exactly like flat-sawn timber.
//   * fine grain: an fbm stretched ~20:1 along the grain, thresholded into pore streaks.
//   * optional planks with dark gaps, per-plank tone/offset and a sawn-edge bevel.
//   * optional knots (dark, hard inclusions that sit slightly proud) and a varnish layer
//     (lower roughness, faint scratches and dents).
struct WoodStyle {
    Vec3 base{0.15f, 0.10f, 0.06f};
    float colorVar = 0.30f;      // large-scale / per-plank tone variation
    float ringCells = 9.0f;      // growth rings across the grain
    float ringDark = 0.42f;      // albedo reduction on a late-wood line
    float ringWarp = 0.55f;      // ring wander, in ring units
    float grainM = 0.004f;       // fine grain feature size across the grain
    float grainDark = 0.18f;
    float grainReliefM = 0.00030f;   // fine grain depth
    float ringReliefM = 0.00010f;    // late-wood proudness
    float poreThreshold = 0.62f; // 1 = no pores
    float roughMean = 0.45f, roughGrain = 0.06f, roughDetail = 0.05f;
    float varnish = 0.0f;        // 0 = raw wood, 1 = satin finish
    int planks = 0;              // 0 = single slab
    float gapDepthM = 0.0f;
    float knotsPerM2 = 0.0f;
    float scratchesPerM2 = 0.0f;
    float dentsPerM2 = 0.0f;
};

TextureSet makeWood(uint32_t size, const SurfaceParams& params, const WoodStyle& st, uint32_t salt) {
    Bake b(size, params, salt);
    const float inv = b.inv;
    const int pRing = b.cells(0.35f, 2);                       // ring wander field
    const int pRingAlong = max2(1, pRing / 8);                 // ... stretched along the plank
    float wGrain = 1.0f;
    const int pGrain = b.cells(st.grainM, 3, wGrain);           // fine grain streaks
    // The grain is stretched along the plank: the "along" feature is ~20x the across one.
    const int pGrainAlong = max2(1, pGrain / 20);
    const int pTone = b.cells(0.9f, 3);

    Rng rng(b.seed ^ 0x776F6F64u);   // "wood"
    const int plankCount = st.planks > 0 ? st.planks : 1;
    // Per-plank tone, thickness and ring phase offset (indexed by the wrapped plank id, so the
    // plank that straddles the tile seam keeps a single identity).
    std::vector<float> plankTone(static_cast<size_t>(plankCount), 1.0f);
    std::vector<float> plankRing(static_cast<size_t>(plankCount), 0.0f);
    std::vector<float> plankH(static_cast<size_t>(plankCount), 0.0f);
    std::vector<float> plankGap(static_cast<size_t>(plankCount), 1.0f);
    for (int i = 0; i < plankCount; ++i) {
        plankTone[static_cast<size_t>(i)] = 1.0f + st.colorVar * rng.symmetric();
        plankRing[static_cast<size_t>(i)] = rng.nextFloat() * 32.0f;
        plankH[static_cast<size_t>(i)] = rng.symmetric() * 0.00016f;
        plankGap[static_cast<size_t>(i)] = 1.0f + 0.35f * rng.symmetric();
    }
    std::vector<Blob> knots;
    const int knotCount = b.featureCount(st.knotsPerM2, 201u);
    for (int i = 0; i < knotCount; ++i) {
        Blob k;
        // Kept well inside the tile: a knot is a local feature, not something to wrap.
        k.x = rng.range(0.15f, 0.85f);
        k.y = rng.range(0.15f, 0.85f);
        k.radius = rng.range(0.025f, 0.045f);
        k.depth = -0.00035f;              // knots sit slightly proud
        k.rough = 0.10f;
        k.bright = -0.45f;                // and much darker
        knots.push_back(k);
    }
    const std::vector<Segment> scratches =
        makeScratches(rng, b.featureCount(st.scratchesPerM2, 203u), 0.05f, 0.30f, 0.00012f,
                      0.000015f, 0.10f, 0.06f, b.tpm, size, 1.2f);
    std::vector<Blob> dents;
    const int dentCount = b.featureCount(st.dentsPerM2, 205u);
    for (int i = 0; i < dentCount; ++i) {
        Blob d;
        d.x = rng.range(0.1f, 0.9f);
        d.y = rng.range(0.1f, 0.9f);
        d.radius = rng.range(0.02f, 0.05f);
        d.depth = rng.range(0.00008f, 0.00020f);
        d.rough = 0.05f;
        d.bright = -0.04f;
        dents.push_back(d);
    }
    // Varnish hides the micro relief (it fills the pores) and adds a sheen variation.
    const float reliefScale = 1.0f - 0.55f * st.varnish;
    const int pSheen = b.cells(0.18f, 3);
    float wMicro = 1.0f;
    const int pMicro = b.cells(0.0012f, 2, wMicro);   // sanding / raised-grain micro texture

    parallelBands(size, [&](uint32_t y0, uint32_t y1) {
        for (uint32_t y = y0; y < y1; ++y) {
            const float v = (static_cast<float>(y) + 0.5f) * inv;
            // Plank layout: an eighth-tile offset puts the tile seam in the middle of a plank,
            // so the wrapping board is continuous and its per-plank constants match up.
            const float pv = st.planks > 0 ? fracf(v + 0.125f) * static_cast<float>(plankCount) : v;
            const int plank = st.planks > 0 ? (static_cast<int>(pv) % plankCount) : 0;
            const float local = pv - std::floor(pv);
            const float tone = plankTone[static_cast<size_t>(plank)];
            const float ringOffset = plankRing[static_cast<size_t>(plank)];
            // Gap profile: dark, deep and slightly bevelled at both plank ends.
            float gap = 0.0f;
            if (st.planks > 0) {
                const float edge = min2(local, 1.0f - local);   // distance to the plank end
                const float gw = 0.012f * plankGap[static_cast<size_t>(plank)];
                gap = band(gw, 0.0f, edge);
            }
            for (uint32_t x = 0; x < size; ++x) {
                const float u = (static_cast<float>(x) + 0.5f) * inv;
                const float toneF = tileableFbm2(u * pTone, v * pTone, pTone, 3, b.seed + 11u);
                // Ring wander field: low frequency so the rings stay coherent, stretched along
                // the plank so their wobble is stretched too. Both periods must be the actual
                // lattice periods - scaling the *coordinate* by a fraction of a cell would
                // break periodicity and leave a hard seam at u = 0/1.
                const float warp = tileableFbmAniso(u * pRingAlong, local * pRing, pRingAlong,
                                                    pRing, 3, b.seed + 13u);
                // Bunching: the ring spacing is squeezed towards one edge of the board and
                // stretched towards the other, the way flat-sawn timber looks. Without it the
                // rings come out as an even barcode, which is the classic procedural-wood tell.
                const float bunch = (warp - 0.5f) * 2.0f;
                const float shaped = local + 0.85f * bunch * local * (1.0f - local);
                const float phase = shaped * st.ringCells + ringOffset + st.ringWarp * (warp - 0.5f);
                const float ringWave = 0.5f - 0.5f * std::cos(TWO_PI * phase);
                const float rings = ringWave * ringWave * ringWave;   // thin, dark late-wood lines
                // Fine grain, stretched 20:1 along the plank.
                const float grain = tileableFbmAniso(u * pGrainAlong, local * pGrain, pGrainAlong,
                                                     pGrain, 3, b.seed + 17u);
                const float pores = band(st.poreThreshold, st.poreThreshold + 0.22f, grain);
                const float sheen = tileableFbm2(u * pSheen, v * pSheen, pSheen, 3, b.seed + 19u);
                const float microR = tileableFbm2(u * pMicro, v * pMicro, pMicro, 2, b.seed + 23u);
                float knot = 0.0f, knotRing = 0.0f, knotRough = 0.0f, knotDark = 0.0f;
                for (const Blob& k : knots) {
                    const float f = blobFalloff(u, v, k);
                    if (f <= 0.0f) continue;
                    knot = max2(knot, f);
                    knotRing = max2(knotRing, band(0.55f, 1.0f, f) * (1.0f - f));
                    knotRough = max2(knotRough, f * k.rough);
                    knotDark = max2(knotDark, f);
                }
                float dent = 0.0f, dentRough = 0.0f;
                for (const Blob& d : dents) {
                    const float f = blobFalloff(u, v, d);
                    if (f <= 0.0f) continue;
                    dent += f * d.depth;
                    dentRough = max2(dentRough, f * d.rough);
                }
                const MarkHit sc = marksAt(scratches, u, v);

                const float grainTone = 1.0f + st.grainDark * (grain - 0.5f) * 2.0f;
                float alb = 0.0f;
                Vec3 col = st.base * tone * (1.0f + 0.25f * (toneF - 0.5f)) * grainTone;
                alb = 1.0f - st.ringDark * rings;
                col = col * alb;
                col = col * (1.0f - 0.30f * pores) * (1.0f - 0.55f * knotDark);
                col = col * (1.0f - 0.75f * gap) * (1.0f + sc.bright * sc.mask);
                // Late wood is denser: a hair proud of the early wood, pores recessed.
                const float h = plankH[static_cast<size_t>(plank)] +
                                st.ringReliefM * reliefScale * rings +
                                st.grainReliefM * wGrain * reliefScale * (grain - 0.5f) * 2.0f -
                                1.6f * st.grainReliefM * wGrain * reliefScale * pores +
                                0.00035f * knotRing + 0.00030f * knot - dent - sc.depth -
                                st.gapDepthM * gap + 0.00003f * (toneF - 0.5f);
                const float rough = st.roughMean + st.roughGrain * (grain - 0.5f) * 2.0f +
                                    st.roughDetail * (sheen - 0.5f) * 2.0f + 0.075f * pores +
                                    0.10f * gap + knotRough + dentRough + sc.rough * sc.mask -
                                    0.30f * st.varnish * (sheen - 0.5f) +
                                    0.040f * (microR - 0.5f) * 2.0f;
                const size_t i = b.idx(x, y);
                b.rgb[i * 3 + 0] = col.x;
                b.rgb[i * 3 + 1] = col.y;
                b.rgb[i * 3 + 2] = col.z;
                b.height[i] = h;
                b.rough[i] = clamp(rough, 0.08f, 0.95f);
                b.metal[i] = 0.0f;
                b.ao[i] = clamp((1.0f - 0.45f * gap - 0.25f * pores - 0.20f * knotDark -
                                 0.30f * knotRing - 0.25f * dent - 0.35f * sc.mask) *
                                    (0.99f + 0.02f * (toneF - 0.5f)),
                                0.0f, 1.0f);
            }
        }
    });
    // The varnish layer already smooths the amplitudes above, so the normal is built with the
    // plain physical slope scale.
    return finishBake(b, b.tpm, b.tpm * 3.0f, 14);
}

TextureSet woodPlanks(uint32_t size, const SurfaceParams& params) {
    WoodStyle st;
    st.base = Vec3(0.170f, 0.108f, 0.058f);   // warm brown, dark-ish floor boards
    st.colorVar = 0.34f;
    st.ringCells = 6.0f;
    st.ringDark = 0.40f;
    st.ringWarp = 0.75f;
    st.grainM = 0.0045f;
    st.grainDark = 0.20f;
    st.grainReliefM = 0.00034f;
    st.ringReliefM = 0.00012f;
    st.poreThreshold = 0.60f;
    st.roughMean = 0.450f;
    st.roughGrain = 0.070f;
    st.roughDetail = 0.045f;
    st.planks = 4;
    st.gapDepthM = 0.0016f;
    st.knotsPerM2 = 0.5f;
    st.scratchesPerM2 = 6.5f;
    st.dentsPerM2 = 0.5f;
    return makeWood(size, params, st, 0x706C616Eu);   // "plan"
}

TextureSet woodTableTop(uint32_t size, const SurfaceParams& params) {
    WoodStyle st;
    st.base = Vec3(0.190f, 0.118f, 0.062f);   // denser hardwood, slightly redder
    st.colorVar = 0.14f;
    st.ringCells = 13.0f;                     // tighter grain
    st.ringDark = 0.26f;
    st.ringWarp = 0.35f;
    st.grainM = 0.0022f;
    st.grainDark = 0.12f;
    st.grainReliefM = 0.00020f;
    st.ringReliefM = 0.00007f;
    st.poreThreshold = 0.70f;                 // fewer, finer pores
    st.roughMean = 0.320f;                    // satin varnish
    st.roughGrain = 0.030f;
    st.roughDetail = 0.045f;
    st.varnish = 0.8f;
    st.knotsPerM2 = 0.25f;
    st.scratchesPerM2 = 10.0f;
    st.dentsPerM2 = 0.75f;
    return makeWood(size, params, st, 0x7461626Cu);   // "tabl"
}

TextureSet varnishedWood(uint32_t size, const SurfaceParams& params) {
    WoodStyle st;
    st.base = Vec3(0.098f, 0.056f, 0.030f);   // legs / apron: darker stain
    st.colorVar = 0.12f;
    st.ringCells = 15.0f;
    st.ringDark = 0.22f;
    st.ringWarp = 0.30f;
    st.grainM = 0.0020f;
    st.grainDark = 0.10f;
    st.grainReliefM = 0.00016f;
    st.ringReliefM = 0.00006f;
    st.poreThreshold = 0.74f;
    st.roughMean = 0.245f;                    // glossier
    st.roughGrain = 0.022f;
    st.roughDetail = 0.035f;
    st.varnish = 1.0f;
    st.knotsPerM2 = 0.0f;
    st.scratchesPerM2 = 8.5f;
    st.dentsPerM2 = 0.5f;
    return makeWood(size, params, st, 0x7661726Eu);   // "varn"
}

// ---------------------------------------------------------------- blued steel
// Firearm slide: a very dark, near-black oxide finish over polished steel, metallic = 1. The
// oxide layer is thin enough that the base colour tints the specular reflection, which is why
// metals are authored with a real (dark, slightly blue) base colour instead of black.
TextureSet bluedSteel(uint32_t size, const SurfaceParams& params) {
    Bake b(size, params, 0x73746565u);   // "stee"
    float wPolish = 1.0f, wFine = 1.0f;
    const int pPolish = b.cells(0.0035f, 2, wPolish);   // fine lengthwise polish marks
    const int pFine = b.cells(0.0012f, 2, wFine);       // micro tooling texture
    const int pWear = b.cells(0.30f, 3);                // holster / handling wear
    const int pPolishAlong = max2(1, pPolish / 24);
    Rng rng(b.seed ^ 0x51D3u);
    // Micro scratches are brighter (they cut through the oxide into bare steel) and rougher.
    // 4 um wide / 0.4 um deep is below the texel grid, so the groove is widened to a texel with
    // its depth scaled to match - the scratch keeps its real slope and stays visible.
    const std::vector<Segment> scratches =
        makeScratches(rng, b.featureCount(5.5f, 301u), 0.02f, 0.16f, 0.000004f, 0.0000004f, 0.28f,
                      1.15f, b.tpm, size);
    const float inv = b.inv;
    parallelBands(size, [&](uint32_t y0, uint32_t y1) {
        for (uint32_t y = y0; y < y1; ++y) {
            const float v = (static_cast<float>(y) + 0.5f) * inv;
            for (uint32_t x = 0; x < size; ++x) {
                const float u = (static_cast<float>(x) + 0.5f) * inv;
                // Machining / polishing marks run along the slide (u), so the noise is heavily
                // stretched along u and fine across v.
                const float polish = tileableFbmAniso(u * pPolishAlong, v * pPolish, pPolishAlong,
                                                      pPolish, 3, b.seed + 11u);
                const float fine = tileableFbmAniso(u * max2(1, pFine / 24), v * pFine,
                                                    max2(1, pFine / 24), pFine, 2, b.seed + 13u);
                const float wear = tileableFbm2(u * pWear, v * pWear, pWear, 3, b.seed + 17u);
                const float wearMask = band(0.45f, 0.75f, wear);     // rubbed-dull patches
                const float buff = band(0.55f, 0.30f, wear);         // hand-polished patches
                const MarkHit sc = marksAt(scratches, u, v);
                // Thin oxide: dark, faintly blue, and its thickness wobbles with the polish
                // (interference-ish tone shift across the streaks).
                Vec3 col(0.041f, 0.043f, 0.054f);
                col = col * (1.0f + 0.18f * (polish - 0.5f) + 0.10f * (fine - 0.5f));
                col = col * (1.0f + 0.20f * buff - 0.06f * wearMask);
                col = col * (1.0f + sc.bright * sc.mask);
                const float h = 0.0000020f * wPolish * (polish - 0.5f) * 2.0f +
                                0.0000008f * wFine * (fine - 0.5f) * 2.0f - sc.depth;
                const float rough = 0.300f + 0.055f * (polish - 0.5f) * 2.0f +
                                    0.035f * (fine - 0.5f) * 2.0f + 0.075f * wearMask -
                                    0.055f * buff + sc.rough * sc.mask +
                                    0.05f * (tileableFbm2(u * pWear, v * pWear, pWear, 3, b.seed + 19u) -
                                             0.5f) * 2.0f;
                const size_t i = b.idx(x, y);
                b.rgb[i * 3 + 0] = col.x;
                b.rgb[i * 3 + 1] = col.y;
                b.rgb[i * 3 + 2] = col.z;
                b.height[i] = h;
                b.rough[i] = clamp(rough, 0.20f, 0.44f);
                b.metal[i] = 1.0f;
                b.ao[i] = clamp(1.0f - 0.30f * sc.mask - 0.05f * (1.0f - polish) -
                                    0.08f * (1.0f - fine) - 0.04f * wearMask,
                                0.0f, 1.0f);
            }
        }
    });
    // Metal: normal strength follows the real micro relief (a few micrometres).
    return finishBake(b, b.tpm, b.tpm * 4.0f, 12);
}

// ---------------------------------------------------------------- black polymer
// Injection-moulded glass-filled nylon (USP frame): near-black, matte, with a moulded stipple
// grip. The stipple is a jittered grid of small rounded pyramids; 1.8 mm at 512 texels/m is a
// third of a texel, so the lattice is widened to a few texels and the height scaled with it,
// keeping the slope (and therefore the shading) of the real pattern.
TextureSet blackPolymer(uint32_t size, const SurfaceParams& params) {
    Bake b(size, params, 0x706F6C79u);   // "poly"
    float wStipple = 1.0f;
    // 1.8 mm stipple at 512 texels/m is a third of a texel: keep >= 3 texels per bump and scale
    // the bump height with the widening so the stipple keeps its real slope.
    const int pStipple = b.cells(0.0018f, 1, wStipple, 3);
    float wMicro = 1.0f;
    const int pMicro = b.cells(0.0009f, 2, wMicro);
    const int pFlow = b.cells(0.25f, 3);       // mould flow / gloss variation
    const float inv = b.inv;
    parallelBands(size, [&](uint32_t y0, uint32_t y1) {
        for (uint32_t y = y0; y < y1; ++y) {
            const float v = (static_cast<float>(y) + 0.5f) * inv;
            for (uint32_t x = 0; x < size; ++x) {
                const float u = (static_cast<float>(x) + 0.5f) * inv;
                const WorleySample stp = tileableWorley(u * pStipple, v * pStipple, pStipple,
                                                        pStipple, b.seed + 11u);
                // Round-topped pyramid per stipple; F1 alone already gives a Voronoi bump,
                // smoothstep rounds the top and leaves the valley between the bumps.
                const float pyr = band(0.52f, 0.10f, stp.f1);
                const float pyrH = 0.35f + 0.16f * stp.id;   // per-stipple height variation
                const float micro = tileableFbm2(u * pMicro, v * pMicro, pMicro, 2, b.seed + 13u);
                const float flow = tileableFbm2(u * pFlow, v * pFlow, pFlow, 3, b.seed + 17u);
                // Mould parting line: a faint vertical seam, kept away from the tile edge and
                // wobbled by a periodic field so it does not read as a straight ruler line.
                const float wob = tileableFbm2(u * 4, v * 4, 4, 2, b.seed + 19u);
                const float partD = std::fabs(u - 0.27f - 0.012f * (wob - 0.5f));
                const float parting = band(0.0035f, 0.0f, partD);
                const float h = 0.00035f * wStipple * pyr * pyrH +
                                0.00002f * wMicro * (micro - 0.5f) * 2.0f +
                                0.00010f * parting + 0.00001f * (flow - 0.5f);
                // Moulded tips get polished by handling: slightly brighter and smoother.
                const float tip = pyr;
                Vec3 col(0.0275f, 0.0275f, 0.0300f);
                col = col * (1.0f + 0.34f * tip + 0.06f * (flow - 0.5f) + 0.05f * (micro - 0.5f));
                const float rough = 0.640f - 0.075f * tip + 0.045f * (flow - 0.5f) -
                                    0.05f * parting + 0.02f * (micro - 0.5f);
                const size_t i = b.idx(x, y);
                b.rgb[i * 3 + 0] = col.x;
                b.rgb[i * 3 + 1] = col.y;
                b.rgb[i * 3 + 2] = col.z;
                b.height[i] = h;
                b.rough[i] = clamp(rough, 0.52f, 0.74f);
                b.metal[i] = 0.0f;
                b.ao[i] = clamp(1.0f - 0.22f * (1.0f - pyr) - 0.10f * parting, 0.0f, 1.0f);
            }
        }
    });
    return finishBake(b, b.tpm, b.tpm * 1.2f, 10);
}

// ---------------------------------------------------------------- grip panel
// Rubberised chequering: two families of V-grooves at 90 degrees to each other leave a grid of
// raised diamonds with pointed tops. Both groove coordinates are integer shears of (u,v), which
// is what keeps the pattern exactly periodic on the torus.
TextureSet gripPanel(uint32_t size, const SurfaceParams& params) {
    Bake b(size, params, 0x67726970u);   // "grip"
    float wGrid = 1.0f;
    // 1.5 mm chequering needs >= 3 texels per groove period or the sheared lattice aliases into
    // a constant (which is exactly what happened before this clamp existed).
    const int pGrid = b.cells(0.0015f, 1, wGrid, 3);
    float wMicro = 1.0f;
    const int pMicro = b.cells(0.0008f, 2, wMicro);
    const int pTone = b.cells(0.20f, 3);
    const float inv = b.inv;
    const float cells = static_cast<float>(pGrid);
    parallelBands(size, [&](uint32_t y0, uint32_t y1) {
        for (uint32_t y = y0; y < y1; ++y) {
            const float v = (static_cast<float>(y) + 0.5f) * inv;
            for (uint32_t x = 0; x < size; ++x) {
                const float u = (static_cast<float>(x) + 0.5f) * inv;
                const float g1 = fracf((u + v) * cells);
                const float g2 = fracf((u - v) * cells);
                const float cut1 = band(0.34f, 0.0f, min2(g1, 1.0f - g1));
                const float cut2 = band(0.34f, 0.0f, min2(g2, 1.0f - g2));
                const float cut = max2(cut1, cut2);
                // Per-diamond variation comes from smooth periodic fields rather than a cell
                // hash: the diamond lattice is indexed by sheared coordinates, and a hash would
                // step whenever a wrapped coordinate lands a hair on the other side of a cell.
                const float tone = tileableFbm2(u * pTone, v * pTone, pTone, 3, b.seed + 11u);
                const float micro = tileableFbm2(u * pMicro, v * pMicro, pMicro, 2, b.seed + 13u);
                const float dome = (1.0f - cut) * (1.0f - cut);
                const float h = -0.00040f * wGrid * cut + 0.00006f * wGrid * dome +
                                0.00003f * wMicro * (micro - 0.5f) * 2.0f;
                Vec3 col(0.0560f, 0.0505f, 0.0455f);   // dark grey-brown rubber
                col = col * (1.0f + 0.20f * dome + 0.10f * (tone - 0.5f) + 0.06f * (micro - 0.5f));
                const float rough = 0.845f + 0.085f * cut - 0.070f * dome +
                                    0.05f * (tone - 0.5f) * 2.0f + 0.03f * (micro - 0.5f) * 2.0f;
                const size_t i = b.idx(x, y);
                b.rgb[i * 3 + 0] = col.x;
                b.rgb[i * 3 + 1] = col.y;
                b.rgb[i * 3 + 2] = col.z;
                b.height[i] = h;
                b.rough[i] = clamp(rough, 0.72f, 0.93f);
                b.metal[i] = 0.0f;
                b.ao[i] = clamp(1.0f - 0.25f * cut - 0.05f * (1.0f - dome), 0.0f, 1.0f);
            }
        }
    });
    return finishBake(b, b.tpm, b.tpm * 0.8f, 12);
}

// ---------------------------------------------------------------- brass
// Cartridge brass: metallic, warm, quite smooth, with the fine circumferential polishing marks
// a drawing / tumbling operation leaves (lines of constant v, i.e. running around the case) and
// a handful of tiny scratches.
TextureSet brass(uint32_t size, const SurfaceParams& params) {
    Bake b(size, params, 0x62726173u);   // "bras"
    float wPolish = 1.0f;
    const int pPolish = b.cells(0.0025f, 2, wPolish);
    float wFine = 1.0f;
    const int pFine = b.cells(0.0008f, 2, wFine);
    const int pTarnish = b.cells(0.16f, 3);
    const int pAlong = max2(1, pPolish / 30);
    Rng rng(b.seed ^ 0x0B4A55u);
    const std::vector<Segment> scratches =
        makeScratches(rng, b.featureCount(4.5f, 401u), 0.02f, 0.12f, 0.000006f, 0.0000003f, 0.22f,
                      0.10f, b.tpm, size);
    const float inv = b.inv;
    parallelBands(size, [&](uint32_t y0, uint32_t y1) {
        for (uint32_t y = y0; y < y1; ++y) {
            const float v = (static_cast<float>(y) + 0.5f) * inv;
            for (uint32_t x = 0; x < size; ++x) {
                const float u = (static_cast<float>(x) + 0.5f) * inv;
                const float polish = tileableFbmAniso(u * pAlong, v * pPolish, pAlong, pPolish, 3,
                                                      b.seed + 11u);
                const float fine = tileableFbmAniso(u * max2(1, pFine / 30), v * pFine,
                                                    max2(1, pFine / 30), pFine, 2, b.seed + 13u);
                const float tarnish = tileableFbm2(u * pTarnish, v * pTarnish, pTarnish, 3, b.seed + 17u);
                const float dull = band(0.55f, 0.85f, tarnish);   // slightly oxidised patches
                const MarkHit sc = marksAt(scratches, u, v);
                const float h = 0.0000012f * wPolish * (polish - 0.5f) * 2.0f +
                                0.0000005f * wFine * (fine - 0.5f) * 2.0f - sc.depth;
                // Cartridge brass: 70/30 alpha brass is the warm (0.62, 0.45, 0.18) linear.
                Vec3 col(0.620f, 0.450f, 0.180f);
                col = col * (1.0f + 0.05f * (polish - 0.5f) * 2.0f + 0.03f * (fine - 0.5f)) *
                      (1.0f - 0.10f * dull) * (1.0f + sc.bright * sc.mask);
                const float rough = 0.180f + 0.035f * (polish - 0.5f) * 2.0f +
                                    0.020f * (fine - 0.5f) * 2.0f + 0.09f * dull + sc.rough * sc.mask;
                const size_t i = b.idx(x, y);
                b.rgb[i * 3 + 0] = col.x;
                b.rgb[i * 3 + 1] = col.y;
                b.rgb[i * 3 + 2] = col.z;
                b.height[i] = h;
                b.rough[i] = clamp(rough, 0.10f, 0.30f);
                b.metal[i] = 1.0f;
                b.ao[i] = clamp(1.0f - 0.35f * sc.mask - 0.06f * dull - 0.05f * (1.0f - polish) -
                                    0.08f * (1.0f - fine),
                                0.0f, 1.0f);
            }
        }
    });
    return finishBake(b, b.tpm, b.tpm * 4.0f, 12);
}

// ---------------------------------------------------------------- glass
// Soda-lime glass: almost perfectly smooth. Only fingerprints (concentric ridge pattern from a
// fingertip), a faint colour shift where a smudge sits, and a couple of microscopic scratches
// keep it from being a perfect mirror - and even those barely move the normal.
TextureSet glassSurface(uint32_t size, const SurfaceParams& params) {
    Bake b(size, params, 0x676C6173u);   // "glas"
    float wMicro = 1.0f;
    const int pMicro = b.cells(0.0015f, 2, wMicro);
    const int pFilm = b.cells(0.28f, 3);       // smudge / film blotches
    Rng rng(b.seed ^ 0x91A55u);
    std::vector<Blob> smudges;
    const int smudgeCount = b.featureCount(1.75f, 501u);
    for (int i = 0; i < smudgeCount; ++i) {
        Blob s;
        s.x = rng.range(0.08f, 0.92f);
        s.y = rng.range(0.08f, 0.92f);
        s.radius = rng.range(0.05f, 0.11f);
        s.rings = rng.range(6.0f, 11.0f);
        s.rough = 0.035f;
        s.bright = 0.012f;
        smudges.push_back(s);
    }
    // Glass scratches really are microscopic (20 nm deep), so unlike the metal scratches they
    // are deliberately *not* slope-preserved: they show in roughness, not in the normal.
    const std::vector<Segment> scratches =
        makeScratches(rng, b.featureCount(3.0f, 503u), 0.02f, 0.12f, 0.000003f, 0.00000002f, 0.05f,
                      0.0f, b.tpm, size, 1.0f, 0.0f);
    const float inv = b.inv;
    parallelBands(size, [&](uint32_t y0, uint32_t y1) {
        for (uint32_t y = y0; y < y1; ++y) {
            const float v = (static_cast<float>(y) + 0.5f) * inv;
            for (uint32_t x = 0; x < size; ++x) {
                const float u = (static_cast<float>(x) + 0.5f) * inv;
                const float micro = tileableFbm2(u * pMicro, v * pMicro, pMicro, 2, b.seed + 11u);
                const float film = tileableFbm2(u * pFilm, v * pFilm, pFilm, 3, b.seed + 13u);
                float smudge = 0.0f, ridge = 0.0f;
                for (const Blob& s : smudges) {
                    const float dx = torusDelta(u - s.x), dy = torusDelta(v - s.y);
                    const float d = std::sqrt(dx * dx + dy * dy);
                    const float f = band(s.radius, 0.0f, d);
                    if (f <= 0.0f) continue;
                    // Fingertip: concentric ridges, fading at the rim of the print.
                    const float rr = 0.5f + 0.5f * std::cos(TWO_PI * s.rings * d / s.radius);
                    smudge = max2(smudge, f);
                    ridge = max2(ridge, f * rr);
                }
                const MarkHit sc = marksAt(scratches, u, v);
                // Threshold measured from the mid value rather than high in the tail: a
                // low-octave fbm on a small tile has a compressed range, and a fixed high
                // threshold would make the whole film map vanish there.
                const float filmMask = band(0.50f, 0.80f, film) * 0.5f;
                const float h = 0.000000004f * (micro - 0.5f) * 2.0f +   // nanometre-scale
                                0.000000002f * (ridge - 0.5f) * 2.0f - sc.depth;
                Vec3 col(0.962f, 0.966f, 0.960f);
                // A fingerprint is a thin organic film: a hair warmer and slightly hazy.
                col = col + Vec3(0.010f, 0.004f, -0.008f) * (smudge + filmMask) * 0.5f;
                // Float glass is uniform, but not perfectly so: the polished surface wanders by
                // a couple of hundredths across the sheet. The micro field is contrast-stretched
                // because a smooth fbm has a narrow distribution - without it the map would read
                // as flat wherever the smudges are absent.
                const float microS = clamp((micro - 0.5f) * 2.6f + 0.5f, 0.0f, 1.0f);
                const float rough = 0.021f + 0.030f * microS + 0.018f * (film - 0.5f) * 2.0f +
                                    0.070f * smudge + 0.030f * ridge + 0.060f * filmMask +
                                    sc.rough * sc.mask;
                const size_t i = b.idx(x, y);
                b.rgb[i * 3 + 0] = col.x;
                b.rgb[i * 3 + 1] = col.y;
                b.rgb[i * 3 + 2] = col.z;
                b.height[i] = h;
                b.rough[i] = clamp(rough, 0.015f, 0.110f);
                b.metal[i] = 0.0f;
                b.ao[i] = clamp(1.0f - 0.16f * smudge - 0.16f * filmMask - 0.10f * sc.mask -
                                    0.10f * (1.0f - micro),
                                0.0f, 1.0f);
            }
        }
    });
    // Deliberately physical (not boosted): glass relief really is nanometres, so the normal map
    // stays essentially flat while the roughness map carries the smudges.
    return finishBake(b, b.tpm, b.tpm, 8);
}

// ---------------------------------------------------------------- white painted metal
// Ceiling lamp housing: a white topcoat over sheet metal. The paint flows and cures into the
// classic "orange peel" - a cellular dimple pattern a couple of millimetres across - over a
// faint sanding/brush texture. Matte-ish, but not flat.
TextureSet paintedMetalWhite(uint32_t size, const SurfaceParams& params) {
    Bake b(size, params, 0x6C616D70u);   // "lamp"
    float wPeel = 1.0f;
    const int pPeel = b.cells(0.0028f, 1, wPeel, 3);  // orange-peel cells (>= 3 texels each)
    float wBrush = 1.0f;
    const int pBrush = b.cells(0.0016f, 2, wBrush);
    const int pThick = b.cells(0.22f, 3);
    const int pDust = b.cells(0.012f, 2);
    const float inv = b.inv;
    parallelBands(size, [&](uint32_t y0, uint32_t y1) {
        for (uint32_t y = y0; y < y1; ++y) {
            const float v = (static_cast<float>(y) + 0.5f) * inv;
            for (uint32_t x = 0; x < size; ++x) {
                const float u = (static_cast<float>(x) + 0.5f) * inv;
                const WorleySample peel = tileableWorley(u * pPeel, v * pPeel, pPeel, pPeel,
                                                         b.seed + 11u);
                // Orange peel: rounded cells with pinched valleys where the cells meet.
                const float dimple = band(0.55f, 0.05f, peel.f1) * (0.6f + 0.4f * peel.id);
                const float valley = 1.0f - band(0.0f, 0.22f, peel.f2 - peel.f1);
                const float brush = tileableFbmAniso(u * max2(1, pBrush / 12), v * pBrush,
                                                     max2(1, pBrush / 12), pBrush, 2, b.seed + 13u);
                const float thick = tileableFbm2(u * pThick, v * pThick, pThick, 3, b.seed + 17u);
                const float dust = tileableFbm2(u * pDust, v * pDust, pDust, 2, b.seed + 19u);
                const float h = 0.000040f * wPeel * dimple - 0.000030f * wPeel * valley +
                                0.000004f * wBrush * (brush - 0.5f) * 2.0f +
                                0.000003f * (dust - 0.5f) * 2.0f;
                Vec3 col(0.800f, 0.802f, 0.800f);
                col = col * (1.0f + 0.02f * (thick - 0.5f) + 0.010f * (dust - 0.5f)) *
                      (1.0f - 0.03f * valley);
                const float rough = 0.400f - 0.030f * dimple + 0.045f * valley +
                                    0.02f * (brush - 0.5f) + 0.015f * (thick - 0.5f) +
                                    0.02f * (dust - 0.5f);
                const size_t i = b.idx(x, y);
                b.rgb[i * 3 + 0] = col.x;
                b.rgb[i * 3 + 1] = col.y;
                b.rgb[i * 3 + 2] = col.z;
                b.height[i] = h;
                b.rough[i] = clamp(rough, 0.32f, 0.50f);
                b.metal[i] = 0.0f;
                b.ao[i] = clamp(1.0f - 0.12f * valley - 0.05f * (1.0f - dimple), 0.0f, 1.0f);
            }
        }
    });
    return finishBake(b, b.tpm, b.tpm * 3.0f, 12);
}

// ---------------------------------------------------------------- lamp diffuser
// Frosted diffuser / bulb envelope. Used with an emissive factor, so it is kept bright and
// clean: near-white, diffuse mottling only, no grime, no cracks. The "radial" mottling is a
// product of periodic cosines (exactly tileable) rather than a distance-to-centre gradient.
TextureSet lampEmissive(uint32_t size, const SurfaceParams& params) {
    Bake b(size, params, 0x656D6974u);   // "emit"
    float wFrost = 1.0f;
    const int pFrost = b.cells(0.0009f, 2, wFrost);
    const int pMottle = b.cells(0.55f, 3);
    const int pFine = b.cells(0.05f, 3);
    const float inv = b.inv;
    parallelBands(size, [&](uint32_t y0, uint32_t y1) {
        for (uint32_t y = y0; y < y1; ++y) {
            const float v = (static_cast<float>(y) + 0.5f) * inv;
            for (uint32_t x = 0; x < size; ++x) {
                const float u = (static_cast<float>(x) + 0.5f) * inv;
                const float mottle = tileableFbm2(u * pMottle, v * pMottle, pMottle, 3, b.seed + 11u);
                const float fine = tileableFbm2(u * pFine, v * pFine, pFine, 3, b.seed + 13u);
                const float frost = tileableFbm2(u * pFrost, v * pFrost, pFrost, 2, b.seed + 17u);
                const float radial = 0.5f + 0.5f * std::cos(TWO_PI * u) * std::cos(TWO_PI * v);
                const float diff = 0.55f * mottle + 0.25f * fine + 0.20f * radial;
                const Vec3 col = Vec3(0.905f, 0.902f, 0.895f) *
                                 (1.0f + 0.045f * (diff - 0.5f) + 0.02f * (frost - 0.5f));
                const float h = 0.000004f * wFrost * (frost - 0.5f) * 2.0f +
                                0.000003f * (fine - 0.5f) * 2.0f;
                const float rough = 0.360f + 0.09f * (diff - 0.5f) + 0.05f * (frost - 0.5f);
                const size_t i = b.idx(x, y);
                b.rgb[i * 3 + 0] = col.x;
                b.rgb[i * 3 + 1] = col.y;
                b.rgb[i * 3 + 2] = col.z;
                b.height[i] = h;
                b.rough[i] = clamp(rough, 0.28f, 0.47f);
                b.metal[i] = 0.0f;
                b.ao[i] = clamp(1.0f - 0.10f * (1.0f - diff) - 0.05f * (1.0f - frost), 0.0f, 1.0f);
            }
        }
    });
    return finishBake(b, b.tpm, b.tpm * 6.0f, 10);
}

// ---------------------------------------------------------------- fabric
// Plain weave: warp and weft threads alternate over/under in a checkerboard, so the height is
// the crown of whichever thread is on top. Thread pitch is 1.2 mm; that is below the texel grid
// at 512 texels/m, so the weave is widened to ~4 texels per thread and the crown height scaled
// with it to keep the cloth's characteristic steep thread profile.
TextureSet fabric(uint32_t size, const SurfaceParams& params) {
    Bake b(size, params, 0x66616272u);   // "fabr"
    float wThread = 1.0f;
    // A 1.2 mm thread pitch is far below the texel grid at 512 texels/m: widen it to 4 texels
    // per thread and scale the crown height with it, so the weave keeps its steep profile.
    const int threads = b.cells(0.0012f, 1, wThread, 4);
    const int pFuzz = b.cells(0.0004f, 2);
    const int pSlub = b.cells(0.03f, 3);              // yarn thickness variation
    const float n = static_cast<float>(threads);
    const float inv = b.inv;
    parallelBands(size, [&](uint32_t y0, uint32_t y1) {
        for (uint32_t y = y0; y < y1; ++y) {
            const float v = (static_cast<float>(y) + 0.5f) * inv;
            for (uint32_t x = 0; x < size; ++x) {
                const float u = (static_cast<float>(x) + 0.5f) * inv;
                const float fu = u * n, fv = v * n;
                const int iu = static_cast<int>(fu), iv = static_cast<int>(fv);
                const float tu = fu - static_cast<float>(iu), tv = fv - static_cast<float>(iv);
                const bool warpOnTop = ((iu + iv) & 1) == 0;
                // Rounded thread cross-section plus a crown that peaks at the cell centre.
                const float crossU = std::sin(PI * tu);
                const float crossV = std::sin(PI * tv);
                const float crown = warpOnTop ? crossV * (0.78f + 0.22f * crossU)
                                              : crossU * (0.78f + 0.22f * crossV);
                const float slub = tileableFbm2(u * pSlub, v * pSlub, pSlub, 3, b.seed + 11u);
                const float slubWarp = tileableFbm2(u * pSlub, v * pSlub + 3.0f, pSlub, 3, b.seed + 13u);
                const float fuzz = tileableFbm2(u * pFuzz, v * pFuzz, pFuzz, 2, b.seed + 17u);
                const float threadTone = warpOnTop ? slub : slubWarp;
                const float h = 0.00040f * wThread * crown * (0.85f + 0.30f * (threadTone - 0.5f)) +
                                0.00005f * wThread * (fuzz - 0.5f) * 2.0f;
                Vec3 col(0.0700f, 0.0720f, 0.0850f);   // dark blue-grey cloth
                col = col * (1.0f + 0.22f * (threadTone - 0.5f) + 0.12f * (crown - 0.5f)) *
                      (1.0f + 0.10f * (fuzz - 0.5f));
                const float rough = 0.880f - 0.075f * crown + 0.06f * (fuzz - 0.5f) * 2.0f +
                                    0.045f * (threadTone - 0.5f) * 2.0f;
                const size_t i = b.idx(x, y);
                b.rgb[i * 3 + 0] = col.x;
                b.rgb[i * 3 + 1] = col.y;
                b.rgb[i * 3 + 2] = col.z;
                b.height[i] = h;
                b.rough[i] = clamp(rough, 0.78f, 0.92f);
                b.metal[i] = 0.0f;
                b.ao[i] = clamp(1.0f - 0.18f * (1.0f - crown), 0.0f, 1.0f);
            }
        }
    });
    return finishBake(b, b.tpm, b.tpm * 1.0f, 12);
}

// ---------------------------------------------------------------- cardboard
// Corrugated board seen from the liner side: the flutes telegraph through as a soft regular
// ripple, on top of a kraft-paper surface made of long fibres, fines and a faint score line
// where the sheet was folded.
TextureSet cardboard(uint32_t size, const SurfaceParams& params) {
    Bake b(size, params, 0x63617264u);   // "card"
    float wFlute = 1.0f;
    const int pFlute = b.cells(0.0055f, 1, wFlute);   // B-flute pitch
    float wFibre = 1.0f;
    const int pFibre = b.cells(0.0006f, 3, wFibre);
    const int pFibreAlong = max2(1, pFibre / 8);
    const int pFines = b.cells(0.0035f, 2);
    const int pTone = b.cells(0.30f, 3);
    const float inv = b.inv;
    parallelBands(size, [&](uint32_t y0, uint32_t y1) {
        for (uint32_t y = y0; y < y1; ++y) {
            const float v = (static_cast<float>(y) + 0.5f) * inv;
            for (uint32_t x = 0; x < size; ++x) {
                const float u = (static_cast<float>(x) + 0.5f) * inv;
                const float tone = tileableFbm2(u * pTone, v * pTone, pTone, 3, b.seed + 11u);
                // Flute: a periodic ripple across v, its phase wobbled by a periodic field so
                // the corrugation does not read as a machine-perfect sine.
                const float wob = tileableFbmAniso(u * max2(1, pFlute / 6), v * pFlute,
                                                   max2(1, pFlute / 6), pFlute, 2, b.seed + 13u);
                const float flute = 0.5f - 0.5f * std::cos(TWO_PI * (v * static_cast<float>(pFlute) +
                                                                     0.10f * (wob - 0.5f)));
                // Kraft fibres: long strands (stretched ~8:1) plus short fines.
                const float fibre = tileableFbmAniso(u * pFibreAlong, v * pFibre, pFibreAlong,
                                                     pFibre, 3, b.seed + 17u);
                const float strands = band(0.55f, 0.80f, fibre);
                const float fines = tileableFbm2(u * pFines, v * pFines, pFines, 2, b.seed + 19u);
                // Score line: a straight fold crease with a little wander, kept off the seam.
                const float scoreD = std::fabs(v - 0.72f - 0.008f * (tone - 0.5f));
                const float score = band(0.006f, 0.0f, scoreD);
                const float h = 0.00022f * wFlute * (flute - 0.5f) * 2.0f +
                                0.00004f * wFibre * (fibre - 0.5f) * 2.0f +
                                0.00002f * (fines - 0.5f) * 2.0f - 0.00012f * score;
                // Kraft liner: warm tan, ~0.30 linear.
                Vec3 col(0.352f, 0.284f, 0.196f);
                col = col * (1.0f + 0.10f * (tone - 0.5f) + 0.09f * (fibre - 0.5f) * 2.0f +
                             0.05f * (fines - 0.5f)) *
                      (1.0f - 0.22f * strands) * (1.0f - 0.10f * score) *
                      (1.0f + 0.04f * (flute - 0.5f));
                const float rough = 0.905f + 0.025f * (fibre - 0.5f) * 2.0f +
                                    0.02f * (fines - 0.5f) - 0.035f * flute + 0.03f * score;
                const size_t i = b.idx(x, y);
                b.rgb[i * 3 + 0] = col.x;
                b.rgb[i * 3 + 1] = col.y;
                b.rgb[i * 3 + 2] = col.z;
                b.height[i] = h;
                b.rough[i] = clamp(rough, 0.84f, 0.95f);
                b.metal[i] = 0.0f;
                b.ao[i] = clamp(1.0f - 0.10f * strands - 0.12f * score - 0.06f * (1.0f - flute),
                                0.0f, 1.0f);
            }
        }
    });
    return finishBake(b, b.tpm, b.tpm * 3.0f, 12);
}

// ===========================================================================
// PNG output
// ===========================================================================
bool writePng(const std::string& path, const TextureData& image, bool srgb) {
    if (!image.valid()) return false;
    const std::vector<uint8_t> bytes = srgb ? toSrgb8(image) : toLinear8(image);
    std::vector<uint8_t> raw;
    raw.reserve(static_cast<size_t>(image.height) * (1 + static_cast<size_t>(image.width) * 4));
    for (uint32_t y = 0; y < image.height; ++y) {
        raw.push_back(0);   // filter type 0 (None) for every scanline
        const uint8_t* row = &bytes[static_cast<size_t>(y) * image.width * 4];
        raw.insert(raw.end(), row, row + static_cast<size_t>(image.width) * 4);
    }
    return writePngFile(path, image.width, image.height, 6, raw, srgb);
}

bool writePngGrey(const std::string& path, const std::vector<float>& data, uint32_t width,
                  uint32_t heightPx) {
    const size_t n = static_cast<size_t>(width) * heightPx;
    if (n == 0 || data.size() < n) return false;
    std::vector<uint8_t> raw;
    raw.reserve(static_cast<size_t>(heightPx) * (1 + width));
    for (uint32_t y = 0; y < heightPx; ++y) {
        raw.push_back(0);
        for (uint32_t x = 0; x < width; ++x) {
            raw.push_back(static_cast<uint8_t>(
                std::lround(clamp(data[static_cast<size_t>(y) * width + x], 0.0f, 1.0f) * 255.0f)));
        }
    }
    return writePngFile(path, width, heightPx, 0, raw, false);
}

}  // namespace room2::procgen
