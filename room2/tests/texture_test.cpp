// room2 - verification suite for the procedural texture generator (src/procgen/texture.cpp).
//
// Checks, in order:
//   a. structure of every generator's TextureSet (sizes, finiteness, ranges, metallic, relief)
//   b. tileability: wrap-around difference vs. interior difference for every map
//   c. roughness / AO statistics against documented plausible bands
//   d. normalFromHeight against an analytic height ramp and a constant field
//   e. aoFromHeight on a flat field and on a deep narrow pit
//   f. packOrm round-trip
//   g. toSrgb8 transfer curve (and toLinear8 for comparison)
//   h. writePng / writePngGrey: signature, chunk CRCs, zlib header, Adler-32, exact round-trip
//      through a minimal reader written here; raw expectations are dumped next to the PNGs so an
//      independent decoder (python3 zlib) can cross-check them
//   i. performance: all generators at 1024x1024
//
// Build (the extra -include is a workaround for a defect in core/math.hpp, see the note there):
//   g++ -std=c++20 -O2 -Wall -Wextra -I src tests/texture_test.cpp src/procgen/texture.cpp -o ...

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "core/math.hpp"
#include "core/rng.hpp"
#include "procgen/texture.hpp"

using room2::Vec3;
using room2::Vec4;
namespace procgen = room2::procgen;

// ---------------------------------------------------------------- tiny test harness
static int g_pass = 0;
static int g_fail = 0;
static std::string g_section;

static void section(const std::string& s) {
    g_section = s;
    std::printf("\n== %s ==\n", s.c_str());
}
static bool check(bool ok, const std::string& what) {
    if (ok) {
        ++g_pass;
    } else {
        ++g_fail;
        std::printf("  FAIL [%s] %s\n", g_section.c_str(), what.c_str());
    }
    return ok;
}
static bool checkNear(double got, double want, double tol, const std::string& what) {
    char buf[256];
    std::snprintf(buf, sizeof buf, "%s (got %.6f, want %.6f +- %.6g)", what.c_str(), got, want, tol);
    return check(std::fabs(got - want) <= tol, buf);
}

struct Stat {
    double mean = 0, sd = 0, mn = 0, mx = 0;
};
static Stat statsOf(const std::vector<float>& v) {
    Stat s;
    if (v.empty()) return s;
    s.mn = s.mx = v[0];
    double sum = 0;
    for (float x : v) {
        sum += x;
        s.mn = std::min<double>(s.mn, x);
        s.mx = std::max<double>(s.mx, x);
    }
    s.mean = sum / static_cast<double>(v.size());
    double var = 0;
    for (float x : v) var += (x - s.mean) * (x - s.mean);
    s.sd = std::sqrt(var / static_cast<double>(v.size()));
    return s;
}

// ---------------------------------------------------------------- generator registry
struct GenInfo {
    const char* name;
    procgen::TextureSet (*fn)(uint32_t, const procgen::SurfaceParams&);
    double metal;        // expected metallic value (0 for dielectrics, 1 for metals)
    double roughLo, roughHi;
    double albedoLo, albedoHi;
    double reliefMin;    // minimum mean |n.xy - 0.5| sum proving the height field is used
    double reliefMax;    // maximum (used to prove e.g. that glass stays mirror-flat)
    double reliefP99;    // minimum 99th percentile of that sum: catches sparse but *real*
                         // relief such as micro-scratches, which barely move the mean
    double reliefPeak;   // minimum *strongest* relief: what a scratch groove has to reach
};

static const GenInfo kGens[] = {
    // The relief thresholds encode the design intent: broad relief (weave, stipple, aggregate,
    // plank gaps) must show up in the mean, sparse relief (micro-scratches on polished metal)
    // only in the tail, and glass must stay essentially flat.
    // name                  fn                     metal roughLo roughHi albLo  albHi  relMin relMax relP99
    {"plasterWall",          procgen::plasterWall,    0.0, 0.85, 0.95, 0.70, 0.80, 0.004, 0.30, 0.05, 0.0},
    {"paintedCeiling",       procgen::paintedCeiling, 0.0, 0.86, 0.95, 0.74, 0.84, 0.003, 0.30, 0.008, 0.0},
    {"concreteFloor",        procgen::concreteFloor,  0.0, 0.58, 0.82, 0.20, 0.32, 0.025, 0.60, 0.12, 0.0},
    {"woodPlanks",           procgen::woodPlanks,     0.0, 0.32, 0.58, 0.08, 0.24, 0.015, 0.60, 0.06, 0.0},
    {"woodTableTop",         procgen::woodTableTop,   0.0, 0.22, 0.42, 0.10, 0.28, 0.006, 0.60, 0.04, 0.0},
    {"varnishedWood",        procgen::varnishedWood,  0.0, 0.15, 0.34, 0.05, 0.17, 0.004, 0.60, 0.02, 0.05},
    {"bluedSteel",           procgen::bluedSteel,     1.0, 0.20, 0.40, 0.030, 0.075, 0.0,  0.05, 0.0,  0.06},
    {"blackPolymer",         procgen::blackPolymer,   0.0, 0.52, 0.74, 0.016, 0.045, 0.04,  0.90, 0.07, 0.0},
    {"gripPanel",            procgen::gripPanel,      0.0, 0.72, 0.93, 0.030, 0.090, 0.06,  0.90, 0.25, 0.0},
    {"brass",                procgen::brass,          1.0, 0.12, 0.28, 0.35, 0.60, 0.0,   0.05, 0.0,  0.02},
    {"glassSurface",         procgen::glassSurface,   0.0, 0.02, 0.09, 0.90, 0.99, 0.0,   0.02, 0.0, 0.0},
    {"paintedMetalWhite",    procgen::paintedMetalWhite, 0.0, 0.32, 0.50, 0.74, 0.86, 0.004, 0.60, 0.015, 0.0},
    {"lampEmissive",         procgen::lampEmissive,   0.0, 0.30, 0.45, 0.84, 0.96, 0.0008, 0.30, 0.0015, 0.0},
    {"fabric",               procgen::fabric,         0.0, 0.78, 0.92, 0.04, 0.13, 0.04,  0.90, 0.20, 0.0},
    {"cardboard",            procgen::cardboard,      0.0, 0.84, 0.95, 0.22, 0.36, 0.012, 0.60, 0.05, 0.0},
};
static const int kGenCount = static_cast<int>(sizeof(kGens) / sizeof(kGens[0]));

// ---------------------------------------------------------------- stats helpers
struct MapStats {
    Stat albedo;    // luma of the linear base colour
    Stat rough;     // ORM.g
    Stat ao;        // ORM.r
    Stat metal;     // ORM.b
    double normalRelief = 0;      // mean(|n.x-0.5| + |n.y-0.5|)
    double normalReliefP99 = 0;   // 99th percentile of the same: the sparse features
    double normalReliefMax = 0;   // strongest relief anywhere (scratches, cracks, gaps)
};

static MapStats analyse(const procgen::TextureSet& s) {
    MapStats m;
    const size_t n = static_cast<size_t>(s.width()) * s.height();
    std::vector<float> alb(n), rough(n), ao(n), met(n), relief(n);
    double reliefSum = 0;
    for (size_t i = 0; i < n; ++i) {
        const float* b = &s.baseColor.pixels[i * 4];
        const float* o = &s.orm.pixels[i * 4];
        const float* nm = &s.normal.pixels[i * 4];
        const Vec3 c(b[0], b[1], b[2]);
        alb[i] = room2::luma(c);
        rough[i] = o[1];
        ao[i] = o[0];
        met[i] = o[2];
        relief[i] = static_cast<float>(std::fabs(nm[0] - 0.5) + std::fabs(nm[1] - 0.5));
        reliefSum += relief[i];
    }
    m.albedo = statsOf(alb);
    m.rough = statsOf(rough);
    m.ao = statsOf(ao);
    m.metal = statsOf(met);
    m.normalRelief = n ? reliefSum / static_cast<double>(n) : 0.0;
    if (n) {
        const size_t k = std::min(n - 1, static_cast<size_t>(0.99 * static_cast<double>(n)));
        std::nth_element(relief.begin(), relief.begin() + static_cast<long>(k), relief.end());
        m.normalReliefP99 = relief[k];
        m.normalReliefMax = *std::max_element(relief.begin(), relief.end());
    }
    return m;
}

// (b) tileability: mean absolute difference across the wrap seam compared with the mean
// absolute difference between neighbouring interior columns / rows. A perfectly periodic map
// gives ~1.0; the test allows up to 2.0.
struct TileRatio {
    double col = 0, row = 0;
};
static TileRatio tileRatio(const procgen::TextureData& t) {
    TileRatio r;
    const uint32_t w = t.width, h = t.height;
    if (w < 4 || h < 4) return r;
    const uint32_t stride = w * 4;
    double wrapCol = 0, intCol = 0, wrapRow = 0, intRow = 0;
    for (uint32_t y = 0; y < h; ++y) {
        const float* a = &t.pixels[static_cast<size_t>(y) * stride];
        const float* b = &t.pixels[static_cast<size_t>(y) * stride + (w - 1) * 4];
        for (int c = 0; c < 3; ++c) wrapCol += std::fabs(a[c] - b[c]);
    }
    for (uint32_t y = 0; y < h; ++y) {
        for (uint32_t x = 0; x + 1 < w; ++x) {
            const float* a = &t.pixels[static_cast<size_t>(y) * stride + static_cast<size_t>(x) * 4];
            const float* b = a + 4;
            for (int c = 0; c < 3; ++c) intCol += std::fabs(a[c] - b[c]);
        }
    }
    for (uint32_t x = 0; x < w; ++x) {
        const float* a = &t.pixels[static_cast<size_t>(x) * 4];
        const float* b = &t.pixels[static_cast<size_t>(h - 1) * stride + static_cast<size_t>(x) * 4];
        for (int c = 0; c < 3; ++c) wrapRow += std::fabs(a[c] - b[c]);
    }
    for (uint32_t y = 0; y + 1 < h; ++y) {
        for (uint32_t x = 0; x < w; ++x) {
            const float* a = &t.pixels[static_cast<size_t>(y) * stride + static_cast<size_t>(x) * 4];
            const float* b = a + stride;
            for (int c = 0; c < 3; ++c) intRow += std::fabs(a[c] - b[c]);
        }
    }
    const double colWrapN = static_cast<double>(h) * 3.0;
    const double colIntN = static_cast<double>(h) * static_cast<double>(w - 1) * 3.0;
    const double rowWrapN = static_cast<double>(w) * 3.0;
    const double rowIntN = static_cast<double>(w) * static_cast<double>(h - 1) * 3.0;
    r.col = (wrapCol / colWrapN) / std::max(intCol / colIntN, 1e-9);
    r.row = (wrapRow / rowWrapN) / std::max(intRow / rowIntN, 1e-9);
    return r;
}

// ---------------------------------------------------------------- (a) structural checks
// `checkRelief` is only enabled at the production size: the amount of scattered detail
// (cracks, scratches, knots) follows a physical density per square metre, so a 256-texel tile
// legitimately contains fewer of them than the 1024-texel one.
static bool structureOk(const std::string& name, const procgen::TextureSet& s, uint32_t expect,
                        const GenInfo& gi, bool print, bool checkRelief = true) {
    const bool dims = s.baseColor.width == expect && s.baseColor.height == expect &&
                      s.normal.width == expect && s.normal.height == expect &&
                      s.orm.width == expect && s.orm.height == expect;
    check(dims, name + ": all three maps are " + std::to_string(expect) + "x" +
                    std::to_string(expect));
    check(s.hasNormal && s.hasOrm, name + ": hasNormal/hasOrm set");
    check(s.baseColor.valid() && s.normal.valid() && s.orm.valid(), name + ": buffers sized correctly");
    if (!dims) return false;

    const size_t n = static_cast<size_t>(expect) * expect;
    bool finite = true, baseRange = true, alphaOk = true, normalRange = true, ormRange = true;
    bool zOk = true, unitOk = true;
    for (size_t i = 0; i < n; ++i) {
        const float* b = &s.baseColor.pixels[i * 4];
        const float* nm = &s.normal.pixels[i * 4];
        const float* o = &s.orm.pixels[i * 4];
        for (int c = 0; c < 4; ++c) {
            if (!std::isfinite(b[c]) || !std::isfinite(nm[c]) || !std::isfinite(o[c])) finite = false;
        }
        for (int c = 0; c < 3; ++c) {
            if (b[c] < 0.0f || b[c] > 1.0f) baseRange = false;
            if (nm[c] < 0.0f || nm[c] > 1.0f) normalRange = false;
            if (o[c] < 0.0f || o[c] > 1.0f) ormRange = false;
        }
        if (b[3] < 0.0f || b[3] > 1.0f) alphaOk = false;
        const double nx = 2.0 * nm[0] - 1.0, ny = 2.0 * nm[1] - 1.0, nz = 2.0 * nm[2] - 1.0;
        if (!(1.0 - nx * nx - ny * ny >= -1e-6)) zOk = false;      // decoded z would be imaginary
        if (nz < 0.0) zOk = false;                                  // normal must face outwards
        const double len = std::sqrt(nx * nx + ny * ny + nz * nz);
        if (std::fabs(len - 1.0) > 2e-3) unitOk = false;
    }
    check(finite, name + ": all components finite");
    check(baseRange, name + ": base colour in [0,1]");
    check(alphaOk, name + ": base colour alpha in [0,1]");
    check(normalRange, name + ": normal components in [0,1]");
    check(zOk, name + ": normal decodes to sqrt(1-x^2-y^2) >= 0 and +Z");
    check(unitOk, name + ": decoded normals are unit length");
    check(ormRange, name + ": ORM components in [0,1]");

    const MapStats m = analyse(s);
    const double metalTol = 0.02;
    check(std::fabs(m.metal.mean - gi.metal) <= metalTol,
          name + ": metallic mean " + std::to_string(m.metal.mean) + " vs expected " +
              std::to_string(gi.metal));
    check(m.metal.sd <= 1e-6, name + ": metallic is constant (sd " + std::to_string(m.metal.sd) + ")");
    check(m.rough.mean >= gi.roughLo && m.rough.mean <= gi.roughHi,
          name + ": roughness mean " + std::to_string(m.rough.mean) + " in [" +
              std::to_string(gi.roughLo) + ", " + std::to_string(gi.roughHi) + "]");
    check(m.rough.sd > 0.01 && m.ao.sd > 0.01,
          name + ": roughness sd " + std::to_string(m.rough.sd) + " and AO sd " +
              std::to_string(m.ao.sd) + " both > 0.01");
    check(m.albedo.mean >= gi.albedoLo && m.albedo.mean <= gi.albedoHi,
          name + ": albedo luma " + std::to_string(m.albedo.mean) + " in [" +
              std::to_string(gi.albedoLo) + ", " + std::to_string(gi.albedoHi) + "]");
    check(m.ao.mn >= 0.0 && m.ao.mx <= 1.0, name + ": AO within [0,1]");
    if (checkRelief && gi.reliefMin > 0.0) {
        check(m.normalRelief >= gi.reliefMin,
              name + ": normal relief " + std::to_string(m.normalRelief) + " >= " +
                  std::to_string(gi.reliefMin));
    }
    if (checkRelief && gi.reliefMax > 0.0) {
        check(m.normalRelief <= gi.reliefMax,
              name + ": normal relief " + std::to_string(m.normalRelief) + " <= " +
                  std::to_string(gi.reliefMax));
    }
    if (checkRelief && gi.reliefPeak > 0.0) {
        check(m.normalReliefMax >= gi.reliefPeak,
              name + ": strongest normal relief " + std::to_string(m.normalReliefMax) + " >= " +
                  std::to_string(gi.reliefPeak));
    }
    if (checkRelief && gi.reliefP99 > 0.0) {
        check(m.normalReliefP99 >= gi.reliefP99,
              name + ": normal relief p99 " + std::to_string(m.normalReliefP99) + " >= " +
                  std::to_string(gi.reliefP99));
    }

    if (print) {
        const TileRatio tr = tileRatio(s.baseColor);
        const TileRatio trn = tileRatio(s.normal);
        const TileRatio tro = tileRatio(s.orm);
        std::printf("  %-17s %4u alb %.3f  rough %.3f+-%.3f  ao %.3f+-%.3f  "
                    "relief mean/p99/max %.4f/%.3f/%.2f  tile bc %.2f/%.2f nrm %.2f/%.2f "
                    "orm %.2f/%.2f\n",
                    name.c_str(), expect, m.albedo.mean, m.rough.mean, m.rough.sd, m.ao.mean,
                    m.ao.sd, m.normalRelief, m.normalReliefP99, m.normalReliefMax, tr.col, tr.row,
                    trn.col, trn.row, tro.col, tro.row);
        check(tr.col <= 2.0 && tr.row <= 2.0 && trn.col <= 2.0 && trn.row <= 2.0 &&
                  tro.col <= 2.0 && tro.row <= 2.0,
              name + ": tileability ratio <= 2.0 for all maps");
    }
    return true;
}

// ---------------------------------------------------------------- (d) normalFromHeight
static void testNormalFromHeight() {
    section("normalFromHeight: analytic ramp + constant field");
    const uint32_t N = 64;
    const double k = 0.01;   // height step per texel
    const double expected = 0.5 - 0.5 * k / std::sqrt(1.0 + k * k);

    std::vector<float> ramp(static_cast<size_t>(N) * N);
    for (uint32_t y = 0; y < N; ++y)
        for (uint32_t x = 0; x < N; ++x) ramp[static_cast<size_t>(y) * N + x] = static_cast<float>(k * y);
    procgen::TextureData nUp = procgen::normalFromHeight(ramp, N, N, 1.0f);
    // Plane tilted about X: height rises along +V, so the surface normal leans towards -Y and
    // the green channel (OpenGL / glTF green-up tangent space) must sit *below* 0.5.
    checkNear(nUp.texel(32, 32).y, expected, 1e-5, "tilt about X: green channel below 0.5");
    check(nUp.texel(32, 32).y < 0.5f, "tilt about X: sign of the green deviation");
    checkNear(nUp.texel(32, 32).x, 0.5, 1e-6, "tilt about X: red channel stays flat");
    checkNear(nUp.texel(32, 32).z, 0.5 + 0.5 / std::sqrt(1.0 + k * k), 1e-5, "tilt about X: blue channel");

    std::vector<float> rampDown(static_cast<size_t>(N) * N);
    for (uint32_t y = 0; y < N; ++y)
        for (uint32_t x = 0; x < N; ++x)
            rampDown[static_cast<size_t>(y) * N + x] = static_cast<float>(-k * y);
    const procgen::TextureData nDn = procgen::normalFromHeight(rampDown, N, N, 1.0f);
    checkNear(nDn.texel(32, 32).y, 0.5 + (0.5 - expected), 1e-5,
              "opposite tilt about X: mirrored green channel");

    std::vector<float> rampX(static_cast<size_t>(N) * N);
    for (uint32_t y = 0; y < N; ++y)
        for (uint32_t x = 0; x < N; ++x) rampX[static_cast<size_t>(y) * N + x] = static_cast<float>(k * x);
    const procgen::TextureData nX = procgen::normalFromHeight(rampX, N, N, 1.0f);
    // A surface rising along +U leans towards -X: red < 0.5. Magnitude identical by symmetry.
    checkNear(nX.texel(32, 32).x, expected, 1e-5, "tilt about Y: red channel below 0.5");

    // Strength scales the slope: doubling it must double the deviation from 0.5.
    const procgen::TextureData nUp2 = procgen::normalFromHeight(ramp, N, N, 2.0f);
    checkNear(nUp2.texel(32, 32).y, 0.5 - 2.0 * (0.5 - expected), 1e-4,
              "strength scales the slope linearly");

    std::vector<float> flat(static_cast<size_t>(N) * N, 0.37f);
    const procgen::TextureData nFlat = procgen::normalFromHeight(flat, N, N, 5.0f);
    bool exact = true;
    for (uint32_t y = 0; y < N; ++y)
        for (uint32_t x = 0; x < N; ++x) {
            const Vec4 t = nFlat.texel(x, y);
            if (std::fabs(t.x - 0.5f) > 1e-6f || std::fabs(t.y - 0.5f) > 1e-6f ||
                std::fabs(t.z - 1.0f) > 1e-6f || t.w != 1.0f)
                exact = false;
        }
    check(exact, "constant height field yields exactly (0.5, 0.5, 1.0, 1)");
    check(nFlat.width == N && nFlat.height == N, "normal map size matches the height field");
}

// ---------------------------------------------------------------- (e) aoFromHeight
static void testAoFromHeight() {
    section("aoFromHeight: flat field and deep narrow pit");
    const uint32_t N = 64;
    std::vector<float> flat(static_cast<size_t>(N) * N, 0.0f);
    const procgen::TextureData aFlat = procgen::aoFromHeight(flat, N, N, 1.0f, 8);
    double mn = 1e9, mx = -1e9;
    for (uint32_t y = 0; y < N; ++y)
        for (uint32_t x = 0; x < N; ++x) {
            const Vec4 t = aFlat.texel(x, y);
            mn = std::min<double>(mn, t.x);
            mx = std::max<double>(mx, t.x);
            if (t.x != t.y || t.y != t.z || t.w != 1.0f) check(false, "AO map must be grey RGBA");
        }
    checkNear(mn, 1.0, 1e-6, "flat field: min AO");
    checkNear(mx, 1.0, 1e-6, "flat field: max AO");

    // A 2x2 texel pit: narrow compared with the 8 texel horizon radius, so every direction
    // sees a steep wall. Note that AO only darkens *inside* a depression - a hole does not
    // occlude the ground around it, so the surroundings legitimately stay at 1.0.
    std::vector<float> pit(static_cast<size_t>(N) * N, 0.0f);
    for (uint32_t y = 30; y <= 32; ++y)
        for (uint32_t x = 30; x <= 32; ++x) pit[static_cast<size_t>(y) * N + x] = -1.0f;
    const procgen::TextureData aPit = procgen::aoFromHeight(pit, N, N, 1.0f, 8);
    const double centre = aPit.texel(31, 31).x;
    const double corner = aPit.texel(30, 30).x;
    const double outside = aPit.texel(35, 31).x;
    const double far = aPit.texel(4, 4).x;
    std::printf("  pit  AO: centre %.4f, pit corner %.4f, 3 texels outside %.4f, far corner %.4f\n",
                centre, corner, outside, far);
    check(centre < 0.7, "deep narrow pit: AO well below 1 inside the pit");
    check(centre > 0.0, "deep narrow pit: AO stays positive");
    check(corner < centre, "deep narrow pit: a pit corner (two walls) is darker than the centre");
    checkNear(outside, 1.0, 1e-6, "deep narrow pit: a hole does not occlude the ground outside it");
    checkNear(far, 1.0, 1e-6, "deep narrow pit: AO ~1 far away");

    // The mirror case: raised terrain *does* occlude its surroundings, and only out to the
    // horizon radius.
    std::vector<float> bump(static_cast<size_t>(N) * N, 0.0f);
    for (uint32_t y = 31; y <= 32; ++y)
        for (uint32_t x = 31; x <= 32; ++x) bump[static_cast<size_t>(y) * N + x] = 1.0f;
    const procgen::TextureData aBump = procgen::aoFromHeight(bump, N, N, 1.0f, 8);
    const double onTop = aBump.texel(31, 31).x;
    const double beside = aBump.texel(35, 31).x;
    const double beyond = aBump.texel(45, 31).x;
    std::printf("  bump AO: on top %.4f, 3 texels beside %.4f, 13 texels away %.4f\n", onTop, beside,
                beyond);
    checkNear(onTop, 1.0, 1e-6, "a plateau is fully open (AO 1 on top)");
    check(beside < 1.0, "raised terrain occludes the ground beside it");
    check(beside > 0.9, "AO beside a 1-unit step 3 texels away is only lightly occluded");
    checkNear(beyond, 1.0, 1e-6, "occlusion stops at the horizon radius");

    // Radius must actually widen the search: with radius 1 the step 3 texels away is invisible.
    const procgen::TextureData aSmall = procgen::aoFromHeight(bump, N, N, 1.0f, 1);
    check(aSmall.texel(35, 31).x > beside, "a larger radius reaches further (radius 1 misses it)");
    check(aPit.texel(31, 31).x < aPit.texel(35, 34).x, "AO increases away from the pit wall");
}

// ---------------------------------------------------------------- (f) packOrm
static void testPackOrm() {
    section("packOrm: round-trip of the three input buffers");
    const uint32_t W = 17, H = 9;
    std::vector<float> ao(W * H), rough(W * H), metal(W * H);
    for (uint32_t y = 0; y < H; ++y)
        for (uint32_t x = 0; x < W; ++x) {
            const size_t i = static_cast<size_t>(y) * W + x;
            ao[i] = static_cast<float>(x) / static_cast<float>(W - 1);
            rough[i] = static_cast<float>(y) / static_cast<float>(H - 1);
            metal[i] = ((x + y) & 1) ? 1.0f : 0.0f;
        }
    const procgen::TextureData orm = procgen::packOrm(ao, rough, metal, W, H);
    bool ok = orm.width == W && orm.height == H;
    for (size_t i = 0; i < static_cast<size_t>(W) * H && ok; ++i) {
        const Vec4 t = orm.texel(static_cast<uint32_t>(i % W), static_cast<uint32_t>(i / W));
        ok = t.x == ao[i] && t.y == rough[i] && t.z == metal[i] && t.w == 1.0f;
    }
    check(ok, "R=ao, G=roughness, B=metallic, A=1 for every texel");
    // Out-of-range inputs must be clamped, not wrapped or propagated.
    std::vector<float> bad(3, 0.0f);
    std::vector<float> rr{-0.5f, 0.5f, 1.5f};
    std::vector<float> mm{2.0f, -1.0f, 0.5f};
    const procgen::TextureData o2 = procgen::packOrm(bad, rr, mm, 3, 1);
    check(o2.texel(0, 0).y == 0.0f && o2.texel(2, 0).y == 1.0f && o2.texel(0, 0).z == 1.0f &&
              o2.texel(1, 0).z == 0.0f,
          "out-of-range ORM inputs are clamped");
    check(procgen::packOrm({}, {}, {}, 4, 4).width == 0, "mismatched buffers produce an empty map");
}

// ---------------------------------------------------------------- (g) colour conversion
static void testColourConversion() {
    section("toSrgb8 / toLinear8 / downsample / bilinear / flatNormal");
    procgen::TextureData img;
    img.resize(4, 1);
    img.setTexel(0, 0, Vec4(0.216f, 0.216f, 0.216f, 1.0f));   // ~0.5 in sRGB
    img.setTexel(1, 0, Vec4(0.0f, 0.0f, 0.0f, 1.0f));
    img.setTexel(2, 0, Vec4(1.0f, 1.0f, 1.0f, 1.0f));
    img.setTexel(3, 0, Vec4(0.0025f, 0.5f, 0.9f, 0.5f));
    const std::vector<uint8_t> srgb = procgen::toSrgb8(img);
    const std::vector<uint8_t> lin = procgen::toLinear8(img);
    check(srgb.size() == 16 && lin.size() == 16, "8-bit output is width*height*4 bytes");
    // 0.216 linear is the canonical mid grey: the sRGB curve lifts it to 0.5022 -> byte 128,
    // while a linear (data map) quantisation would give 55. Asserting both pins the curve down.
    const double wantMid = room2::linearToSrgb(Vec3(0.216f, 0, 0)).x;
    checkNear(wantMid, 0.5022, 0.001, "linearToSrgb(0.216) = 0.5022 (the 0.5 mid grey)");
    checkNear(srgb[0] / 255.0, wantMid, 1.0 / 255.0, "toSrgb8 applies the sRGB curve to mid grey");
    check(srgb[0] == 128, "toSrgb8(0.216) lands on byte 128 (~0.502 sRGB), not the linear 55");
    check(lin[0] == 55, "toLinear8(0.216) is a straight quantisation (55)");
    check(srgb[4] == 0 && srgb[8] == 255, "toSrgb8 endpoints: 0 -> 0, 1 -> 255");
    check(lin[8] == 255 && lin[4] == 0, "toLinear8 endpoints");
    // 0.0025 is below the 0.0031308 knee, where sRGB is the straight 12.92*v segment.
    check(srgb[12] == 8 && lin[12] == 1,
          "toSrgb8 uses the linear segment below the knee (0.0025 -> 0.0323 -> byte 8) while "
          "toLinear8 rounds to 1");
    check(srgb[13] == 188, "toSrgb8(0.5) = 188 (1.055*0.5^(1/2.4)-0.055)");
    check(srgb[12] != lin[12], "a transfer curve really is applied");
    check(srgb[15] == 128 && lin[15] == 128, "alpha is quantised without a transfer curve");
    check(procgen::toSrgb8(procgen::TextureData{}).empty(), "toSrgb8 on an empty image is safe");

    // downsample: 2x box filter, exact on a constant image, correct on a checkerboard.
    procgen::TextureData big;
    big.resize(4, 4, Vec4(0.25f, 0.5f, 0.75f, 1.0f));
    const procgen::TextureData small = procgen::downsample(big);
    check(small.width == 2 && small.height == 2, "downsample halves both dimensions");
    checkNear(small.texel(1, 1).x, 0.25, 1e-6, "downsample of a constant image is constant");
    big.setTexel(0, 0, Vec4(1, 1, 1, 1));
    big.setTexel(1, 0, Vec4(0, 0, 0, 1));
    big.setTexel(0, 1, Vec4(0, 0, 0, 1));
    big.setTexel(1, 1, Vec4(1, 1, 1, 1));
    const procgen::TextureData small2 = procgen::downsample(big);
    checkNear(small2.texel(0, 0).x, 0.5, 1e-6, "downsample averages the 2x2 block");
    checkNear(small2.texel(1, 0).x, 0.25, 1e-6, "downsample only touches its own block");
    check(procgen::downsample(procgen::TextureData{}).width == 0, "downsample of an empty image");

    // bilinear: exact at texel centres, correct midway, wraps at the border.
    procgen::TextureData t;
    t.resize(4, 4, Vec4(0, 0, 0, 1));
    t.setTexel(0, 0, Vec4(1, 1, 1, 1));
    checkNear(t.bilinear(0.125f, 0.125f).x, 1.0, 1e-6, "bilinear is exact at a texel centre");
    checkNear(t.bilinear(0.25f, 0.125f).x, 0.5, 1e-6, "bilinear interpolates between texels");
    checkNear(t.bilinear(0.0f, 0.125f).x, 0.5, 1e-6, "bilinear wraps across the border");
    check(t.bilinear(-1.3f, 2.7f).x >= 0.0f && t.bilinear(-1.3f, 2.7f).x <= 1.0f,
          "bilinear accepts out-of-range coordinates");

    const procgen::TextureData fn = procgen::flatNormal(8);
    bool flatOk = fn.width == 8 && fn.height == 8;
    for (uint32_t y = 0; y < 8 && flatOk; ++y)
        for (uint32_t x = 0; x < 8 && flatOk; ++x) {
            const Vec4 c = fn.texel(x, y);
            flatOk = c.x == 0.5f && c.y == 0.5f && c.z == 1.0f && c.w == 1.0f;
        }
    check(flatOk, "flatNormal is (0.5, 0.5, 1.0, 1.0) everywhere");
}

// ---------------------------------------------------------------- (h) PNG
namespace {

uint32_t readU32be(const uint8_t* p) {
    return (static_cast<uint32_t>(p[0]) << 24) | (static_cast<uint32_t>(p[1]) << 16) |
           (static_cast<uint32_t>(p[2]) << 8) | static_cast<uint32_t>(p[3]);
}
uint16_t readU16le(const uint8_t* p) {
    return static_cast<uint16_t>(p[0] | (static_cast<uint16_t>(p[1]) << 8));
}
uint32_t crc32Bytes(const uint8_t* data, size_t len, uint32_t crc = 0xFFFFFFFFu) {
    for (size_t i = 0; i < len; ++i) {
        crc ^= data[i];
        for (int k = 0; k < 8; ++k) crc = (crc & 1u) ? (0xEDB88320u ^ (crc >> 1)) : (crc >> 1);
    }
    return crc;
}

struct PngImage {
    uint32_t width = 0, height = 0;
    int colorType = -1;
    std::vector<uint8_t> pixels;   // 4 bytes/px for colour type 6, 1 for type 0
    std::string error;
    int chunks = 0;
};

// Deliberately strict: only the subset of PNG this module writes (8-bit, no interlace, filter
// type 0, zlib "stored" blocks), but every CRC, the zlib header and the Adler-32 are verified,
// and anything unexpected is reported as an error rather than silently accepted.
bool readPng(const std::string& path, PngImage& out) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        out.error = "cannot open";
        return false;
    }
    std::vector<uint8_t> buf;
    uint8_t tmp[4096];
    size_t got;
    while ((got = std::fread(tmp, 1, sizeof tmp, f)) > 0) buf.insert(buf.end(), tmp, tmp + got);
    std::fclose(f);
    static const uint8_t sig[8] = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    if (buf.size() < 8 || std::memcmp(buf.data(), sig, 8) != 0) {
        out.error = "bad signature";
        return false;
    }
    size_t pos = 8;
    std::vector<uint8_t> idat;
    bool sawIhdr = false, sawIend = false;
    while (pos + 12 <= buf.size()) {
        const uint32_t len = readU32be(&buf[pos]);
        if (pos + 12 + len > buf.size()) {
            out.error = "truncated chunk";
            return false;
        }
        const uint8_t* type = &buf[pos + 4];
        const uint8_t* data = &buf[pos + 8];
        const uint32_t storedCrc = readU32be(data + len);
        const uint32_t calcCrc = crc32Bytes(&buf[pos + 4], len + 4) ^ 0xFFFFFFFFu;
        if (storedCrc != calcCrc) {
            out.error = "CRC mismatch in chunk " + std::string(reinterpret_cast<const char*>(type), 4);
            return false;
        }
        ++out.chunks;
        const std::string t(reinterpret_cast<const char*>(type), 4);
        if (t == "IHDR") {
            if (len != 13) {
                out.error = "bad IHDR length";
                return false;
            }
            out.width = readU32be(data);
            out.height = readU32be(data + 4);
            if (data[8] != 8) {
                out.error = "only 8-bit PNGs are written";
                return false;
            }
            out.colorType = data[9];
            if (data[10] != 0 || data[11] != 0 || data[12] != 0) {
                out.error = "unexpected compression/filter/interlace";
                return false;
            }
            if (out.colorType != 0 && out.colorType != 6) {
                out.error = "unexpected colour type";
                return false;
            }
            sawIhdr = true;
        } else if (t == "IDAT") {
            idat.insert(idat.end(), data, data + len);
        } else if (t == "IEND") {
            sawIend = true;
        }
        pos += 12 + static_cast<size_t>(len);
    }
    if (!sawIhdr || !sawIend || idat.size() < 6) {
        out.error = "missing IHDR/IDAT/IEND";
        return false;
    }
    // ---- zlib container
    const uint8_t cmf = idat[0], flg = idat[1];
    if ((cmf & 0x0Fu) != 8) {
        out.error = "zlib: not deflate";
        return false;
    }
    if (((static_cast<uint32_t>(cmf) << 8) | flg) % 31u != 0) {
        out.error = "zlib: FCHECK failed";
        return false;
    }
    if (flg & 0x20u) {
        out.error = "zlib: unexpected preset dictionary";
        return false;
    }
    size_t z = 2;
    std::vector<uint8_t> raw;
    for (;;) {
        if (z >= idat.size()) {
            out.error = "zlib: truncated deflate stream";
            return false;
        }
        const uint8_t hdr = idat[z++];
        const int bfinal = hdr & 1;
        const int btype = (hdr >> 1) & 3;
        if (btype != 0) {
            out.error = "zlib: expected stored blocks";
            return false;
        }
        if (z + 4 > idat.size()) {
            out.error = "zlib: truncated block header";
            return false;
        }
        const uint16_t blen = readU16le(&idat[z]);
        const uint16_t nlen = readU16le(&idat[z + 2]);
        z += 4;
        if (static_cast<uint16_t>(~blen) != nlen) {
            out.error = "zlib: LEN/NLEN mismatch";
            return false;
        }
        if (z + blen > idat.size()) {
            out.error = "zlib: truncated block";
            return false;
        }
        raw.insert(raw.end(), idat.begin() + static_cast<long>(z),
                   idat.begin() + static_cast<long>(z + blen));
        z += blen;
        if (bfinal) break;
    }
    if (z + 4 > idat.size()) {
        out.error = "zlib: missing Adler-32";
        return false;
    }
    uint32_t a = 1, b = 0;
    for (uint8_t byte : raw) {
        a = (a + byte) % 65521u;
        b = (b + a) % 65521u;
    }
    const uint32_t adler = (b << 16) | a;
    if (adler != readU32be(&idat[z])) {
        out.error = "zlib: Adler-32 mismatch";
        return false;
    }
    // ---- unfilter (we only ever write filter type 0)
    const size_t bpp = (out.colorType == 6) ? 4u : 1u;
    const size_t rowBytes = static_cast<size_t>(out.width) * bpp;
    if (raw.size() != (rowBytes + 1) * static_cast<size_t>(out.height)) {
        out.error = "raw scanline data has the wrong size";
        return false;
    }
    out.pixels.resize(rowBytes * out.height);
    for (uint32_t y = 0; y < out.height; ++y) {
        const uint8_t* row = &raw[(rowBytes + 1) * static_cast<size_t>(y)];
        if (row[0] != 0) {
            out.error = "unexpected scanline filter type";
            return false;
        }
        std::memcpy(&out.pixels[rowBytes * y], row + 1, rowBytes);
    }
    return true;
}

bool writeFile(const std::string& path, const std::vector<uint8_t>& bytes) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    const size_t w = bytes.empty() ? 0 : std::fwrite(bytes.data(), 1, bytes.size(), f);
    std::fclose(f);
    return w == bytes.size();
}

// Deterministic test pattern (no rand()).
uint8_t pattern(uint32_t x, uint32_t y, int c) {
    uint32_t h = room2::hashCombine(room2::hashCombine(x, y), static_cast<uint32_t>(c) * 2654435761u);
    return static_cast<uint8_t>(h >> 13);
}

}  // namespace

// Dumps go to <project>/build/texture_dump/ regardless of the working directory: the source
// path the compiler was given (__FILE__) is the anchor, so running the binary from anywhere
// still writes where the build rules expect it.
static std::string dumpDir() {
    std::error_code ec;
    const std::filesystem::path here(__FILE__);
    std::filesystem::path dir = here.parent_path().parent_path() / "build" / "texture_dump";
    if (dir.is_relative()) dir = std::filesystem::absolute(dir, ec);
    std::filesystem::create_directories(dir, ec);
    std::string out = dir.string();
    if (!out.empty() && out.back() != '/') out.push_back('/');
    return out;
}

static void testPng() {
    section("writePng / writePngGrey: container validity and exact round-trip");
    const std::string dir = dumpDir();

    // --- small RGBA (colour type 6), sRGB encoded on write
    procgen::TextureData img;
    img.resize(37, 23);
    for (uint32_t y = 0; y < 23; ++y)
        for (uint32_t x = 0; x < 37; ++x)
            img.setTexel(x, y, Vec4(pattern(x, y, 0) / 255.0f, pattern(x, y, 1) / 255.0f,
                                    pattern(x, y, 2) / 255.0f, 1.0f));
    const std::vector<uint8_t> wantSrgb = procgen::toSrgb8(img);
    const std::vector<uint8_t> wantLin = procgen::toLinear8(img);
    check(procgen::writePng(dir + "rgba_srgb.png", img, true), "writePng(srgb=true) succeeds");
    check(procgen::writePng(dir + "rgba_linear.png", img, false), "writePng(srgb=false) succeeds");
    check(writeFile(dir + "expected_rgba_srgb.bin", wantSrgb), "dumped raw expectation bytes");
    check(writeFile(dir + "expected_rgba_linear.bin", wantLin), "dumped raw linear expectation");

    PngImage a, b;
    check(readPng(dir + "rgba_srgb.png", a), "reader accepts rgba_srgb.png: " + a.error);
    check(readPng(dir + "rgba_linear.png", b), "reader accepts rgba_linear.png: " + b.error);
    check(a.width == 37 && a.height == 23 && a.colorType == 6, "PNG header carries width/height/type");
    check(a.pixels == wantSrgb, "sRGB PNG round-trips byte-exactly");
    check(b.pixels == wantLin, "linear PNG round-trips byte-exactly");
    check(a.chunks >= 4, "file has at least IHDR/IDAT/IEND (plus sRGB) chunks");

    // --- big RGBA: forces several stored deflate blocks (raw > 65535 bytes)
    procgen::TextureData big;
    big.resize(256, 256);
    for (uint32_t y = 0; y < 256; ++y)
        for (uint32_t x = 0; x < 256; ++x) {
            // A smooth gradient plus hard edges: a filter or offset bug would show immediately.
            const uint8_t ramp = static_cast<uint8_t>((x * 3 + y * 5) & 0xFFu);
            big.setTexel(x, y, Vec4(((x / 16 + y / 16) & 1) ? 1.0f : 0.0f, ramp / 255.0f,
                                    ((x ^ y) & 0x40) ? 0.75f : 0.0f, 1.0f));
        }
    check(procgen::writePng(dir + "big_rgba.png", big, false), "writePng of a multi-block image");
    check(writeFile(dir + "expected_big_rgba.bin", procgen::toLinear8(big)), "dumped big expectation");
    PngImage c;
    check(readPng(dir + "big_rgba.png", c), "reader accepts big_rgba.png: " + c.error);
    check(c.pixels == procgen::toLinear8(big), "multi-block PNG round-trips byte-exactly");

    // --- grey (colour type 0)
    const uint32_t gw = 91, gh = 47;
    std::vector<float> grey(static_cast<size_t>(gw) * gh);
    for (uint32_t y = 0; y < gh; ++y)
        for (uint32_t x = 0; x < gw; ++x)
            grey[static_cast<size_t>(y) * gw + x] =
                std::min(1.0f, std::max(0.0f, static_cast<float>(x) / static_cast<float>(gw - 1) * 1.2f -
                                                0.1f));
    check(procgen::writePngGrey(dir + "grey.png", grey, gw, gh), "writePngGrey succeeds");
    std::vector<uint8_t> wantGrey(grey.size());
    for (size_t i = 0; i < grey.size(); ++i)
        wantGrey[i] = static_cast<uint8_t>(std::lround(grey[i] * 255.0f));
    check(writeFile(dir + "expected_grey.bin", wantGrey), "dumped grey expectation");
    PngImage g;
    check(readPng(dir + "grey.png", g), "reader accepts grey.png: " + g.error);
    check(g.width == gw && g.height == gh && g.colorType == 0 && g.pixels == wantGrey,
          "grey PNG round-trips byte-exactly (colour type 0)");

    // --- failure paths
    check(!procgen::writePng(dir + "nothing.png", procgen::TextureData{}, true),
          "writePng refuses an invalid image");
    check(!procgen::writePngGrey(dir + "nothing.png", {}, 0, 0), "writePngGrey refuses an empty buffer");
    check(procgen::writePngGrey(dir + "sub/dir/grey2.png", grey, gw, gh),
          "writePngGrey creates missing parent directories");
    std::printf("  wrote PNGs to %s (verified: signature, per-chunk CRC-32, zlib header,\n"
                "  stored-block LEN/NLEN, Adler-32, filter bytes, exact pixel round-trip)\n",
                dir.c_str());
}

// ---------------------------------------------------------------- determinism
static void testDeterminism() {
    section("determinism: identical seed and params give bit-identical maps");
    procgen::SurfaceParams p;
    p.seed = 7;
    p.texelsPerMetre = 256.0f;
    struct Pair {
        const char* name;
        procgen::TextureSet (*fn)(uint32_t, const procgen::SurfaceParams&);
    };
    const Pair pairs[] = {{"plasterWall", procgen::plasterWall},
                          {"woodPlanks", procgen::woodPlanks},
                          {"blackPolymer", procgen::blackPolymer},
                          {"brass", procgen::brass}};
    for (const Pair& q : pairs) {
        const procgen::TextureSet s1 = q.fn(64, p);
        const procgen::TextureSet s2 = q.fn(64, p);
        const bool same = s1.baseColor.pixels == s2.baseColor.pixels &&
                          s1.normal.pixels == s2.normal.pixels && s1.orm.pixels == s2.orm.pixels;
        check(same, std::string(q.name) + ": two runs are bit-identical");
    }
    procgen::SurfaceParams p2 = p;
    p2.seed = 8;
    const procgen::TextureSet d1 = procgen::plasterWall(64, p);
    const procgen::TextureSet d2 = procgen::plasterWall(64, p2);
    check(d1.baseColor.pixels != d2.baseColor.pixels, "a different seed changes the result");
    // texelsPerMetre must actually drive feature size, so the maps must differ.
    procgen::SurfaceParams p3 = p;
    p3.texelsPerMetre = 2048.0f;
    const procgen::TextureSet d3 = procgen::plasterWall(64, p3);
    check(d1.normal.pixels != d3.normal.pixels, "texelsPerMetre changes the physical detail scale");
}

// ---------------------------------------------------------------- main
int main() {
    std::printf("room2 procedural texture test\n");
    std::printf("generators: %d\n", kGenCount);

    testNormalFromHeight();
    testAoFromHeight();
    testPackOrm();
    testColourConversion();
    testPng();
    testDeterminism();

    // ---- (a)(b)(c) every generator, two sizes
    section("generators: structure, ranges, tileability, roughness/AO statistics (size 256)");
    procgen::SurfaceParams params;   // defaults: seed 1234, 512 texels/m
    for (int i = 0; i < kGenCount; ++i) {
        const procgen::TextureSet s = kGens[i].fn(256, params);
        structureOk(kGens[i].name, s, 256, kGens[i], true, false);
    }

    section("generators: robustness at small and odd sizes");
    for (int i = 0; i < kGenCount; ++i) {
        const procgen::TextureSet s = kGens[i].fn(64, params);
        structureOk(kGens[i].name, s, 64, kGens[i], false, false);
    }
    for (int i = 0; i < kGenCount; ++i) {
        const procgen::TextureSet s = kGens[i].fn(1, params);
        check(s.width() == 1 && s.height() == 1 && s.hasNormal && s.hasOrm,
              std::string(kGens[i].name) + ": 1x1 texture is still well formed");
    }
    for (int i = 0; i < kGenCount; ++i) {
        const procgen::TextureSet s = kGens[i].fn(0, params);
        check(s.width() == 0 && !s.hasNormal, std::string(kGens[i].name) + ": size 0 yields an empty set");
    }

    section("generators: one physical tile (2 m) at three texel densities");
    for (float tpm : {128.0f, 256.0f, 512.0f}) {
        const uint32_t dim = static_cast<uint32_t>(2.0f * tpm);
        procgen::SurfaceParams p;
        p.texelsPerMetre = tpm;
        bool allOk = true;
        double reliefSum = 0;
        for (int i = 0; i < kGenCount; ++i) {
            const procgen::TextureSet s = kGens[i].fn(dim, p);
            const MapStats m = analyse(s);
            const TileRatio tr = tileRatio(s.baseColor);
            const TileRatio tn = tileRatio(s.normal);
            const TileRatio to = tileRatio(s.orm);
            const bool bad = !(s.width() == dim && s.hasNormal && s.hasOrm && m.rough.sd > 0.005 &&
                               m.ao.sd > 0.005 && tr.col <= 2.0 && tr.row <= 2.0 && tn.col <= 2.0 &&
                               tn.row <= 2.0 && to.col <= 2.0 && to.row <= 2.0);
            if (bad) {
                std::printf("    %s: size %u rough sd %.4f ao sd %.4f tile %.2f/%.2f %.2f/%.2f "
                            "%.2f/%.2f\n",
                            kGens[i].name, s.width(), m.rough.sd, m.ao.sd, tr.col, tr.row, tn.col,
                            tn.row, to.col, to.row);
            }
            allOk = allOk && !bad;
            reliefSum += m.normalRelief;
        }
        std::printf("  %4u texels for 2 m (tpm %3.0f): %d materials valid, tileable, non-flat "
                    "(mean relief %.4f)\n",
                    dim, tpm, kGenCount, reliefSum / kGenCount);
        check(allOk, "all materials valid + tileable at texelsPerMetre = " + std::to_string(tpm));
    }

    // ---- (i) performance, and a full re-check of the production-size maps
    section("performance: all generators at 1024x1024 (with full checks)");
    double total = 0;
    double totalBytes = 0;
    for (int i = 0; i < kGenCount; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        const procgen::TextureSet s = kGens[i].fn(1024, params);
        const auto t1 = std::chrono::steady_clock::now();
        const double secs = std::chrono::duration<double>(t1 - t0).count();
        total += secs;
        totalBytes += static_cast<double>(s.baseColor.pixels.size() + s.normal.pixels.size() +
                                          s.orm.pixels.size()) * 4.0;
        std::printf("  %-18s %6.3f s\n", kGens[i].name, secs);
        structureOk(kGens[i].name, s, 1024, kGens[i], true);
    }
    std::printf("  TOTAL %.3f s for %d materials at 1024x1024 (%.1f MB of maps)\n", total, kGenCount,
                totalBytes / (1024.0 * 1024.0));
    check(total < 30.0, "all materials at 1024x1024 generated in under 30 s");

    std::printf("\n%s: %d checks passed, %d failed\n", g_fail == 0 ? "PASS" : "FAIL", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
