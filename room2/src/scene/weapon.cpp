// room2 - procedural HK USP .45 pistol, a spent brass cartridge case and a shard of
// broken drinking glass.
//
// ------------------------------------------------------------------ conventions
// Right handed, Y up, +Z towards the muzzle.  The origin is the bore axis at the rear
// face of the slide (the breech face plane), so:
//
//   slide        z in [0.000, 0.100]      (rear face exactly at z = 0)
//   barrel       z in [0.000, 0.108]      (muzzle crown at z = +0.108)
//   beavertail   z ~ -0.070,  rearmost point (grip backstrap base) z ~ -0.086
//   slide top    y = +0.014,  rear sight top y = +0.019
//   magazine floorplate bottom y = -0.117
//
// Every part is a separate, closed MeshData appended through Scene::addMesh so the game
// can animate the slide, magazine, trigger and hammer independently.
//
// All geometry is a pure function of the arguments: no globals, no time, no RNG except
// buildGlassShard(), which copies the caller's Rng (the signature takes it by const
// reference, so the sequence can not be advanced for the caller).

#include "scene/builders.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "core/rng.hpp"
#include "procgen/mesh.hpp"

namespace room2::scene {
namespace {

using procgen::MeshData;
using procgen::SubMesh;

// Texture repeats per metre, shared by every part so the PBR maps keep a constant
// texel density across the whole weapon.
constexpr float kUv = 10.0f;

// ------------------------------------------------------------------ dimensions
// Reference stations of the layout.  Each builder below spells its own section table
// out so the silhouette can be tuned station by station; these are the values the rest
// of the project (animation, HUD, physics) agrees on.
constexpr float kSlideRearZ = 0.0000f;
constexpr float kSlideFrontZ = 0.1000f;
constexpr float kMuzzleZ = 0.1080f;

constexpr float kSlideHalfW = 0.0145f;   // 29 mm slide
constexpr float kSlideTopY = 0.0140f;
constexpr float kSlideBotY = -0.0140f;

constexpr float kRearSightTopY = 0.0190f;
constexpr float kMagFloorBottomY = -0.1170f;
constexpr float kRearmostZ = -0.0860f;

// Barrel.
constexpr float kBoreR = 0.00570f;       // .45 calibre bore
constexpr float kBarrelR = 0.00750f;     // barrel outside diameter in front of the chamber
constexpr float kChamberR = 0.00980f;

// Frame / grip.
constexpr float kFrameTopY = -0.0135f;   // frame deck, just under the slide
constexpr float kDustBotY = -0.0255f;    // dust cover underside
constexpr float kRailBotY = -0.0300f;    // accessory rail underside
constexpr float kGuideRodY = -0.0185f;   // recoil assembly, below the bore axis
constexpr float kGripTopY = -0.0280f;    // where the grip shell starts
constexpr float kGripBotY = -0.1130f;    // magwell opening
constexpr float kGripHalfW = 0.0155f;

// Ejection port window (right hand side of the slide).
constexpr float kPortInnerX = 0.00600f;
constexpr float kPortY0 = -0.00500f;
constexpr float kPortY1 = 0.00650f;

// Animated pivots - also exposed to the test harness at the bottom of this file.
constexpr float kTriggerPivotY = -0.0285f;
constexpr float kTriggerPivotZ = 0.0105f;
constexpr float kHammerPivotY = -0.0145f;
constexpr float kHammerPivotZ = -0.0125f;

// The grip rakes down and rearward by ~20 degrees: both the frontstrap and the backstrap
// are straight lines in the YZ plane, parameterised by height.
inline float gripT(float y) { return (y - kGripTopY) / (kGripBotY - kGripTopY); }
inline float frontstrapZ(float y) { return lerpf(0.0060f, -0.0270f, clamp(gripT(y), -0.5f, 1.5f)); }
inline float backstrapZ(float y) { return lerpf(-0.0530f, kRearmostZ, clamp(gripT(y), -0.5f, 1.5f)); }

// ------------------------------------------------------------------ 2D helpers
inline float cross2(Vec2 a, Vec2 b) { return a.x * b.y - a.y * b.x; }

float polygonArea(const std::vector<Vec2>& p) {
    float a = 0.0f;
    const size_t n = p.size();
    for (size_t i = 0; i < n; ++i) a += cross2(p[i], p[(i + 1) % n]);
    return a * 0.5f;
}

bool pointInTriangle(Vec2 p, Vec2 a, Vec2 b, Vec2 c) {
    const float d1 = cross2(b - a, p - a);
    const float d2 = cross2(c - b, p - b);
    const float d3 = cross2(a - c, p - c);
    const bool neg = (d1 < 0.0f) || (d2 < 0.0f) || (d3 < 0.0f);
    const bool pos = (d1 > 0.0f) || (d2 > 0.0f) || (d3 > 0.0f);
    return !(neg && pos);
}

// Ear clipping triangulation of a simple polygon.  Output triangles are CCW in the
// polygon's own plane (the input is reversed first when it is wound clockwise).
void triangulatePolygon(const std::vector<Vec2>& poly, std::vector<uint32_t>& out) {
    out.clear();
    const int n = static_cast<int>(poly.size());
    if (n < 3) return;
    std::vector<int> idx(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) idx[static_cast<size_t>(i)] = i;
    if (polygonArea(poly) < 0.0f) std::reverse(idx.begin(), idx.end());
    int guard = 0;
    while (idx.size() > 3 && guard++ < 4 * n * n + 16) {
        const int m = static_cast<int>(idx.size());
        bool clipped = false;
        for (int i = 0; i < m; ++i) {
            const int ia = idx[static_cast<size_t>((i + m - 1) % m)];
            const int ib = idx[static_cast<size_t>(i)];
            const int ic = idx[static_cast<size_t>((i + 1) % m)];
            const Vec2 a = poly[static_cast<size_t>(ia)];
            const Vec2 b = poly[static_cast<size_t>(ib)];
            const Vec2 c = poly[static_cast<size_t>(ic)];
            if (cross2(b - a, c - b) <= 1e-14f) continue;  // reflex or degenerate corner
            bool ok = true;
            for (int k = 0; k < m && ok; ++k) {
                const int ik = idx[static_cast<size_t>(k)];
                if (ik == ia || ik == ib || ik == ic) continue;
                if (pointInTriangle(poly[static_cast<size_t>(ik)], a, b, c)) ok = false;
            }
            if (!ok) continue;
            out.push_back(static_cast<uint32_t>(ia));
            out.push_back(static_cast<uint32_t>(ib));
            out.push_back(static_cast<uint32_t>(ic));
            idx.erase(idx.begin() + i);
            clipped = true;
            break;
        }
        if (!clipped) {  // degenerate input: fall back to a fan so we still emit a cap
            for (size_t k = 1; k + 1 < idx.size(); ++k) {
                out.push_back(static_cast<uint32_t>(idx[0]));
                out.push_back(static_cast<uint32_t>(idx[k]));
                out.push_back(static_cast<uint32_t>(idx[k + 1]));
            }
            return;
        }
    }
    if (idx.size() == 3) {
        out.push_back(static_cast<uint32_t>(idx[0]));
        out.push_back(static_cast<uint32_t>(idx[1]));
        out.push_back(static_cast<uint32_t>(idx[2]));
    }
}

// Inward (miter) offset of a simple polygon, used to chamfer the flat sides of the
// trigger guard and of the levers.
std::vector<Vec2> offsetPolygon(const std::vector<Vec2>& poly, float d) {
    const int n = static_cast<int>(poly.size());
    std::vector<Vec2> out(poly);
    if (n < 3 || std::fabs(d) < 1e-9f) return out;
    const bool ccw = polygonArea(poly) > 0.0f;
    for (int i = 0; i < n; ++i) {
        const Vec2 p0 = poly[static_cast<size_t>((i + n - 1) % n)];
        const Vec2 p1 = poly[static_cast<size_t>(i)];
        const Vec2 p2 = poly[static_cast<size_t>((i + 1) % n)];
        Vec2 t0 = normalize(p1 - p0);
        Vec2 t1 = normalize(p2 - p1);
        if (dot(t0, t0) < 0.5f) t0 = t1;
        if (dot(t1, t1) < 0.5f) t1 = t0;
        Vec2 n0(-t0.y, t0.x), n1(-t1.y, t1.x);
        if (!ccw) { n0 = -n0; n1 = -n1; }
        const Vec2 a = p1 + n0 * d;
        const Vec2 b = p1 + n1 * d;
        const float den = cross2(t0, t1);
        if (std::fabs(den) < 1e-6f) {
            out[static_cast<size_t>(i)] = a;
            continue;
        }
        const float t = cross2(b - a, t1) / den;
        const Vec2 q = a + t0 * t;
        // Clamp pathological miters on thin spikes.
        const float lim = std::fabs(d) * 6.0f + 1e-5f;
        out[static_cast<size_t>(i)] = (length(q - p1) > lim) ? p1 + normalize(n0 + n1) * d : q;
    }
    return out;
}

// Rounded rectangle in 2D with an independent radius per corner:
// r00 at (u0,v0), r10 at (u1,v0), r11 at (u1,v1), r01 at (u0,v1).  Radii are clamped so
// the shape stays valid.  Point count is 4*edgeSegs + 4*arcSegs whatever the radii,
// which the lofts rely on.
std::vector<Vec2> roundedRect4(float u0, float u1, float v0, float v1, float r00, float r10,
                               float r11, float r01, int edgeSegs, int arcSegs) {
    std::vector<Vec2> p;
    const int es = max2(edgeSegs, 1);
    const int as = max2(arcSegs, 1);
    const float room = 0.5f * min2(std::fabs(u1 - u0), std::fabs(v1 - v0));
    const float ra = clamp(r00, 1.5e-4f, room);
    const float rb = clamp(r10, 1.5e-4f, room);
    const float rc = clamp(r11, 1.5e-4f, room);
    const float rd = clamp(r01, 1.5e-4f, room);
    auto push = [&p](Vec2 q) {
        if (!p.empty() && length(q - p.back()) < 1e-7f) return;
        p.push_back(q);
    };
    for (int i = 0; i <= es; ++i) push(Vec2(lerpf(u0 + ra, u1 - rb, float(i) / float(es)), v0));
    for (int i = 1; i <= as; ++i) {
        const float a = -HALF_PI + HALF_PI * float(i) / float(as);
        push(Vec2(u1 - rb + rb * std::cos(a), v0 + rb + rb * std::sin(a)));
    }
    for (int i = 1; i <= es; ++i) push(Vec2(u1, lerpf(v0 + rb, v1 - rc, float(i) / float(es))));
    for (int i = 1; i <= as; ++i) {
        const float a = HALF_PI * float(i) / float(as);
        push(Vec2(u1 - rc + rc * std::cos(a), v1 - rc + rc * std::sin(a)));
    }
    for (int i = 1; i <= es; ++i) push(Vec2(lerpf(u1 - rc, u0 + rd, float(i) / float(es)), v1));
    for (int i = 1; i <= as; ++i) {
        const float a = HALF_PI + HALF_PI * float(i) / float(as);
        push(Vec2(u0 + rd + rd * std::cos(a), v1 - rd + rd * std::sin(a)));
    }
    for (int i = 1; i <= es; ++i) push(Vec2(u0, lerpf(v1 - rd, v0 + ra, float(i) / float(es))));
    for (int i = 1; i < as; ++i) {
        const float a = PI + HALF_PI * float(i) / float(as);
        push(Vec2(u0 + ra + ra * std::cos(a), v0 + ra + ra * std::sin(a)));
    }
    return p;
}

// Same shape with one radius per u end (rA at u0, rB at u1).
std::vector<Vec2> roundedRect(float u0, float u1, float v0, float v1, float rA, float rB,
                              int edgeSegs, int arcSegs) {
    return roundedRect4(u0, u1, v0, v1, rA, rB, rB, rA, edgeSegs, arcSegs);
}

// ------------------------------------------------------------------ lofting
enum class Axis { X = 0, Y = 1, Z = 2 };

struct Section {
    float pos = 0.0f;
    std::vector<Vec2> poly;
};

// Maps a section coordinate (u,v) onto the plane of `axis`.  The frame is chosen so
// that (u, v, axis) is right handed, which makes CCW sections produce outward normals.
inline Vec3 place(Axis axis, Vec2 p, float pos) {
    switch (axis) {
        case Axis::X: return Vec3(pos, p.x, p.y);
        case Axis::Y: return Vec3(p.y, pos, p.x);
        default:      return Vec3(p.x, p.y, pos);
    }
}

inline void tri(MeshData& m, uint32_t a, uint32_t b, uint32_t c) {
    m.indices.push_back(a);
    m.indices.push_back(b);
    m.indices.push_back(c);
}

void addCap(MeshData& m, const Section& s, Axis axis, float uvScale, bool endCap) {
    std::vector<uint32_t> tris;
    triangulatePolygon(s.poly, tris);
    if (tris.empty()) return;
    const uint32_t base = m.vertexCount();
    for (Vec2 p : s.poly) {
        Vertex v;
        v.position = place(axis, p, s.pos);
        v.normal = Vec3(0, 1, 0);
        v.uv = p * uvScale;
        v.uv1 = v.uv;
        v.tangent = Vec4(1, 0, 0, 1);
        m.vertices.push_back(v);
    }
    for (size_t k = 0; k + 2 < tris.size(); k += 3) {
        if (endCap) tri(m, base + tris[k], base + tris[k + 1], base + tris[k + 2]);
        else tri(m, base + tris[k + 2], base + tris[k + 1], base + tris[k]);
    }
}

// Sweeps a closed section along an axis.  Sections must share a point count and be
// sorted by increasing `pos`; they are re-oriented to CCW first so the winding of the
// caller's polygon cannot invert the solid.  UVs are the perimeter distance (u)
// against the axial distance (v), i.e. a proper cylindrical unwrap.
MeshData loft(const std::vector<Section>& sectionsIn, Axis axis, float uvScale,
              bool capStart = true, bool capEnd = true) {
    MeshData m;
    if (sectionsIn.size() < 2) return m;
    std::vector<Section> sections = sectionsIn;
    for (Section& s : sections)
        if (polygonArea(s.poly) < 0.0f) std::reverse(s.poly.begin(), s.poly.end());
    const int n = static_cast<int>(sections.front().poly.size());
    if (n < 3) return m;
    for (const Section& s : sections)
        if (static_cast<int>(s.poly.size()) != n) return m;

    const uint32_t stride = static_cast<uint32_t>(n + 1);
    std::vector<float> uAt(static_cast<size_t>(n) + 1, 0.0f);
    for (size_t i = 0; i < sections.size(); ++i) {
        const std::vector<Vec2>& poly = sections[i].poly;
        float acc = 0.0f;
        uAt[0] = 0.0f;
        for (int j = 0; j < n; ++j) {
            acc += length(poly[static_cast<size_t>((j + 1) % n)] - poly[static_cast<size_t>(j)]);
            uAt[static_cast<size_t>(j) + 1] = acc;
        }
        const float v = (sections[i].pos - sections.front().pos) * uvScale;
        for (int j = 0; j <= n; ++j) {
            Vertex vert;
            vert.position = place(axis, poly[static_cast<size_t>(j % n)], sections[i].pos);
            vert.normal = Vec3(0, 1, 0);
            vert.uv = Vec2(uAt[static_cast<size_t>(j)] * uvScale, v);
            vert.uv1 = vert.uv;
            vert.tangent = Vec4(1, 0, 0, 1);
            m.vertices.push_back(vert);
        }
    }
    for (size_t i = 0; i + 1 < sections.size(); ++i) {
        for (int j = 0; j < n; ++j) {
            const uint32_t a = static_cast<uint32_t>(i) * stride + static_cast<uint32_t>(j);
            const uint32_t b = a + 1;
            const uint32_t c = static_cast<uint32_t>(i + 1) * stride + static_cast<uint32_t>(j) + 1;
            const uint32_t d = c - 1;
            tri(m, a, b, c);
            tri(m, a, c, d);
        }
    }
    if (capStart) addCap(m, sections.front(), axis, uvScale, false);
    if (capEnd) addCap(m, sections.back(), axis, uvScale, true);
    m.computeBounds();
    return m;
}

// ------------------------------------------------------------------ part builder
// Collects pieces into one MeshData and books named submeshes for the scene.
class Part {
public:
    void add(const MeshData& piece, uint32_t material, const char* name = "", bool merge = true) {
        if (piece.indices.empty()) return;
        const uint32_t first = mesh_.indexCount();
        mesh_.append(piece);
        if (merge && !ranges_.empty() && ranges_.back().material == material) {
            ranges_.back().count += mesh_.indexCount() - first;
            return;
        }
        Range r;
        r.first = first;
        r.count = mesh_.indexCount() - first;
        r.material = material;
        r.name = name != nullptr ? name : "";
        ranges_.push_back(r);
    }

    MeshData finish(float smoothAngleDeg, bool boxUv) {
        if (boxUv) procgen::boxProjectUv(mesh_, kUv);
        mesh_.submeshes.clear();
        for (const Range& r : ranges_) {
            if (r.count == 0) continue;
            SubMesh s;
            s.indexOffset = r.first;
            s.indexCount = r.count;
            s.material = r.material;
            s.name = r.name;
            Aabb b;
            for (uint32_t k = r.first; k < r.first + r.count && k < mesh_.indices.size(); ++k)
                b.expand(mesh_.vertices[mesh_.indices[k]].position);
            s.bounds = b;
            mesh_.submeshes.push_back(s);
        }
        mesh_.computeNormals(smoothAngleDeg);
        mesh_.computeBounds();
        mesh_.computeTangents();
        return std::move(mesh_);
    }

private:
    struct Range {
        uint32_t first = 0;
        uint32_t count = 0;
        uint32_t material = 0;
        std::string name;
    };
    MeshData mesh_;
    std::vector<Range> ranges_;
};

// ------------------------------------------------------------------ primitive glue
MeshData boxAt(Vec3 center, Vec3 half, uint32_t material, Quat rot = Quat(0, 0, 0, 1)) {
    MeshData m = procgen::makeBox(half, Vec2(kUv, kUv), material);
    m.transform(mat4TRS(center, rot, Vec3(1, 1, 1)));
    return m;
}

// Cylinder along an arbitrary axis, centred on `center`.
MeshData cylAt(Vec3 center, Axis axis, float radius, float length, int segments,
               uint32_t material) {
    MeshData m = procgen::makeCylinder(radius, length, segments, true, material);
    procgen::scaleUv(m, Vec2(TWO_PI * radius * kUv, length * kUv));
    Quat rot(0, 0, 0, 1);
    if (axis == Axis::Z) rot = quatFromAxisAngle(Vec3(1, 0, 0), HALF_PI);
    else if (axis == Axis::X) rot = quatFromAxisAngle(Vec3(0, 0, 1), HALF_PI);
    m.transform(mat4TRS(center, rot, Vec3(1, 1, 1)));
    return m;
}

MeshData cylZ(float radius, float z0, float z1, int segments, uint32_t material) {
    return cylAt(Vec3(0, 0, (z0 + z1) * 0.5f), Axis::Z, radius, std::fabs(z1 - z0), segments,
                 material);
}

// Z aligned cylinder centred on an arbitrary axis point (sight dots).
MeshData cylZAt(Vec3 center, float radius, float length, int segments, uint32_t material) {
    return cylAt(center, Axis::Z, radius, length, segments, material);
}

// X aligned pin centred on `center` (hammer, slide stop, safety, magazine release).
MeshData cylXAt(Vec3 center, float radius, float length, int segments, uint32_t material) {
    return cylAt(center, Axis::X, radius, length, segments, material);
}

// Torus lying in the XY plane (axis along Z) - used for the recoil spring coils.
MeshData coilZ(float majorRadius, float minorRadius, int majorSegs, int minorSegs, float z,
               uint32_t material) {
    MeshData m = procgen::makeTorus(majorRadius, minorRadius, majorSegs, minorSegs, material);
    procgen::scaleUv(m, Vec2(TWO_PI * majorRadius * kUv, TWO_PI * minorRadius * kUv));
    m.transform(mat4TRS(Vec3(0, 0, z), quatFromAxisAngle(Vec3(1, 0, 0), HALF_PI), Vec3(1, 1, 1)));
    return m;
}

// Surface of revolution about the Y axis (makeLathe) moved so that +Y maps to +Z.
// makeLathe emits u = 0..1 around the axis and v = 0..1 along the profile arc, so both
// are rescaled into metres to keep the texel density of the rest of the weapon.
MeshData revolveZ(const std::vector<Vec2>& profile, int segments, float uvScale,
                  uint32_t material) {
    MeshData m = procgen::makeLathe(profile, segments, material);
    m.transform(mat4FromQuat(quatFromAxisAngle(Vec3(1, 0, 0), HALF_PI)));
    float radiusSum = 0.0f;
    int radiusCount = 0;
    float arcLength = 0.0f;
    for (size_t i = 0; i < profile.size(); ++i) {
        if (profile[i].x > 1e-4f) { radiusSum += profile[i].x; ++radiusCount; }
        arcLength += length(profile[(i + 1) % profile.size()] - profile[i]);
    }
    const float rMean = radiusCount > 0 ? radiusSum / float(radiusCount) : 0.005f;
    procgen::scaleUv(m, Vec2(TWO_PI * rMean * uvScale, max2(arcLength, 1e-4f) * uvScale));
    return m;
}

// ===========================================================================
// slide
// ===========================================================================
// Cross section: rounded top/bottom rectangle.  The right hand wall carries a station
// list dense around the ejection port; stations inside the port window are pulled
// inwards by `portT`, which is what cuts the recess into the lofted solid.
std::vector<Vec2> slideSection(float halfW, float yBot, float yTop, float rTop, float rBot,
                               float portT, float portInner) {
    // Wall stations are parametric so that every section keeps the same vertex count
    // whatever the corner radii are; two of them straddle each edge of the port window
    // so the recess gets a crisp lip.
    static const float kT[] = {0.000f, 0.190f, 0.335f, 0.383f, 0.520f,
                               0.660f, 0.800f, 0.886f, 0.930f, 1.000f};
    const int kWallN = static_cast<int>(sizeof(kT) / sizeof(kT[0]));
    const int arcSegs = 10;
    const int edgeSegs = 4;
    std::vector<Vec2> p;
    auto push = [&p](Vec2 q) {
        if (!p.empty() && length(q - p.back()) < 1e-7f) return;
        p.push_back(q);
    };
    // Bottom edge, left to right.
    for (int i = 0; i <= edgeSegs; ++i)
        push(Vec2(lerpf(-(halfW - rBot), halfW - rBot, float(i) / float(edgeSegs)), yBot));
    // Bottom right corner.
    for (int i = 1; i <= arcSegs; ++i) {
        const float a = -HALF_PI + HALF_PI * float(i) / float(arcSegs);
        push(Vec2(halfW - rBot + rBot * std::cos(a), yBot + rBot + rBot * std::sin(a)));
    }
    // Right wall, bottom to top (the port stations move inboard).
    for (int i = 0; i < kWallN; ++i) {
        const float y = lerpf(yBot + rBot, yTop - rTop, kT[i]);
        float x = halfW;
        if (portT > 0.0f && y > kPortY0 && y < kPortY1) x = lerpf(halfW, portInner, portT);
        push(Vec2(x, y));
    }
    // Top right corner.
    for (int i = 1; i <= arcSegs; ++i) {
        const float a = HALF_PI * float(i) / float(arcSegs);
        push(Vec2(halfW - rTop + rTop * std::cos(a), yTop - rTop + rTop * std::sin(a)));
    }
    // Top edge, right to left.
    for (int i = 1; i <= edgeSegs; ++i)
        push(Vec2(lerpf(halfW - rTop, -(halfW - rTop), float(i) / float(edgeSegs)), yTop));
    // Top left corner.
    for (int i = 1; i <= arcSegs; ++i) {
        const float a = HALF_PI + HALF_PI * float(i) / float(arcSegs);
        push(Vec2(-(halfW - rTop) + rTop * std::cos(a), yTop - rTop + rTop * std::sin(a)));
    }
    // Left wall, top to bottom (mirrored stations, never displaced).
    for (int i = kWallN - 1; i >= 0; --i)
        push(Vec2(-halfW, lerpf(yBot + rBot, yTop - rTop, kT[i])));
    // Bottom left corner (the closing point duplicates the first vertex).
    for (int i = 0; i < arcSegs; ++i) {
        const float a = PI + HALF_PI * float(i) / float(arcSegs);
        push(Vec2(-(halfW - rBot) + rBot * std::cos(a), yBot + rBot + rBot * std::sin(a)));
    }
    return p;
}

MeshData buildSlide(uint32_t matSteel) {
    Part part;
    struct Sec {
        float z, portT, halfW, yTop, yBot, rTop, rBot;
    };
    const Sec secs[] = {
        {0.0000f, 0.0f, kSlideHalfW, kSlideTopY, kSlideBotY, 0.0060f, 0.0022f},
        {0.0025f, 0.0f, kSlideHalfW, kSlideTopY, kSlideBotY, 0.0060f, 0.0022f},
        {0.0440f, 0.0f, kSlideHalfW, kSlideTopY, kSlideBotY, 0.0060f, 0.0022f},
        {0.0465f, 1.0f, kSlideHalfW, kSlideTopY, kSlideBotY, 0.0060f, 0.0022f},
        {0.0810f, 1.0f, kSlideHalfW, kSlideTopY, kSlideBotY, 0.0060f, 0.0022f},
        {0.0835f, 0.0f, kSlideHalfW, kSlideTopY, kSlideBotY, 0.0060f, 0.0022f},
        {0.0900f, 0.0f, kSlideHalfW, kSlideTopY, kSlideBotY, 0.0060f, 0.0022f},
        {0.0950f, 0.0f, kSlideHalfW, kSlideTopY, kSlideBotY, 0.0060f, 0.0022f},
        {0.0978f, 0.0f, 0.01410f, 0.01320f, -0.01360f, 0.0055f, 0.0022f},
        {0.1000f, 0.0f, 0.01300f, 0.01180f, -0.01280f, 0.0050f, 0.0022f},
    };
    std::vector<Section> sections;
    for (const Sec& s : secs) {
        Section sec;
        sec.pos = s.z;
        sec.poly = slideSection(s.halfW, s.yBot, s.yTop, s.rTop, s.rBot, s.portT, kPortInnerX);
        sections.push_back(std::move(sec));
    }
    part.add(loft(sections, Axis::Z, kUv), matSteel, "slide-body");

    // Cocking serrations: thin raised ribs on both flanks (rear bank of twelve, plus a
    // short forward bank).
    auto serrations = [&](float z0, float z1, int count) {
        for (int i = 0; i < count; ++i) {
            const float t = (float(i) + 0.5f) / float(count);
            const float z = lerpf(z0, z1, t);
            for (int side = 0; side < 2; ++side) {
                const float sx = side == 0 ? 1.0f : -1.0f;
                part.add(boxAt(Vec3(sx * 0.01485f, -0.00200f, z), Vec3(0.00055f, 0.00950f, 0.00062f),
                               matSteel),
                         matSteel, "serrations");
            }
        }
    };
    serrations(0.0058f, 0.0290f, 14);
    serrations(0.0855f, 0.0955f, 8);

    // Rear and front sight dovetails: an undercut trapezoid boss on the slide top,
    // lofted so the base is narrower than the crown (a real dovetail).
    auto dovetail = [&](float z0, float z1, float halfBot, float halfTop, float height) {
        std::vector<Section> sections;
        const float zs[2] = {z0, z1};
        for (int i = 0; i < 2; ++i) {
            Section s;
            s.pos = zs[i];
            s.poly = {Vec2(-halfBot, -0.0003f), Vec2(halfBot, -0.0003f),
                      Vec2(halfTop, height), Vec2(-halfTop, height)};
            sections.push_back(std::move(s));
        }
        part.add(loft(sections, Axis::Z, kUv), matSteel, "dovetail");
    };
    dovetail(0.0055f, 0.0265f, 0.0050f, 0.0058f, 0.0018f);
    dovetail(0.0850f, 0.0960f, 0.0046f, 0.0054f, 0.0018f);
    return part.finish(35.0f, false);
}

// ===========================================================================
// barrel
// ===========================================================================
MeshData buildBarrel(uint32_t matBarrel) {
    // Closed profile in (radius, height); CCW keeps the outward normals outside.
    const std::vector<Vec2> profile = {
        {kChamberR, 0.0000f},   // chamber block, rear
        {kChamberR, 0.0245f},
        {kBarrelR,  0.0265f},   // step down in front of the chamber
        {kBarrelR,  0.0700f},   // visible through the ejection port
        {0.00720f,  0.0720f},
        {0.00720f,  0.1040f},
        {0.00660f,  0.1072f},   // muzzle crown chamfer
        {0.00660f,  0.1080f},   // muzzle face
        {kBoreR,    0.1080f},
        {kBoreR,    0.0000f},   // bore, back to the breech
    };
    return revolveZ(profile, 72, kUv, matBarrel);
}

// ===========================================================================
// frame
// ===========================================================================
MeshData buildFrame(uint32_t matPolymer, uint32_t matSteel) {
    Part part;

    // Rear receiver / beavertail: a lofted spine that sweeps back and up behind the
    // slide, ending in the beavertail tang.
    struct Back {
        float z, yBot, yTop, halfW, rBot, rTop;
    };
    const Back backs[] = {
        {-0.0700f, -0.0190f, -0.0075f, 0.0090f, 0.0020f, 0.0020f},
        {-0.0670f, -0.0203f, -0.0077f, 0.0098f, 0.0022f, 0.0022f},
        {-0.0640f, -0.0215f, -0.0080f, 0.0105f, 0.0025f, 0.0024f},
        {-0.0590f, -0.0231f, -0.0085f, 0.0113f, 0.0028f, 0.0026f},
        {-0.0540f, -0.0245f, -0.0090f, 0.0120f, 0.0030f, 0.0028f},
        {-0.0490f, -0.0256f, -0.0095f, 0.0126f, 0.0032f, 0.0029f},
        {-0.0440f, -0.0265f, -0.0100f, 0.0132f, 0.0035f, 0.0030f},
        {-0.0390f, -0.0276f, -0.0107f, 0.0137f, 0.0037f, 0.0031f},
        {-0.0340f, -0.0285f, -0.0115f, 0.0142f, 0.0040f, 0.0032f},
        {-0.0290f, -0.0291f, -0.0121f, 0.0145f, 0.0041f, 0.0033f},
        {-0.0240f, -0.0295f, -0.0125f, 0.0147f, 0.0042f, 0.0034f},
        {-0.0140f, -0.0300f, -0.0130f, 0.0150f, 0.0044f, 0.0034f},
        {-0.0040f, -0.0310f, -0.0135f, 0.0150f, 0.0044f, 0.0034f},
        {0.0100f,  -0.0315f, -0.0135f, 0.0150f, 0.0044f, 0.0034f},
        {0.0300f,  -0.0310f, -0.0135f, 0.0148f, 0.0044f, 0.0034f},
    };
    {
        std::vector<Section> sections;
        for (const Back& b : backs) {
            Section s;
            s.pos = b.z;
            // Axis::Z lofts take the section as (u,v) = (x,y).
            s.poly = roundedRect4(-b.halfW, b.halfW, b.yBot, b.yTop, b.rBot, b.rBot, b.rTop,
                                  b.rTop, 4, 6);
            sections.push_back(std::move(s));
        }
        part.add(loft(sections, Axis::Z, kUv), matPolymer, "receiver");
    }

    // Dust cover over the front of the frame.
    struct Dust {
        float z, yBot, yTop, halfW;
    };
    const Dust dust[] = {
        {0.0300f, kDustBotY, kFrameTopY, 0.0144f},
        {0.0360f, kDustBotY, kFrameTopY, 0.0143f},
        {0.0500f, kDustBotY, kFrameTopY, 0.0142f},
        {0.0700f, kDustBotY, kFrameTopY, 0.0140f},
        {0.0900f, -0.0254f, kFrameTopY, 0.0137f},
        {0.0970f, -0.0250f, kFrameTopY, 0.0134f},
        {0.1000f, -0.0245f, -0.0140f, 0.0126f},
    };
    {
        std::vector<Section> sections;
        for (const Dust& d : dust) {
            Section s;
            s.pos = d.z;
            s.poly = roundedRect4(-d.halfW, d.halfW, d.yBot, d.yTop, 0.0022f, 0.0022f, 0.0050f,
                                  0.0050f, 4, 6);
            sections.push_back(std::move(s));
        }
        part.add(loft(sections, Axis::Z, kUv), matPolymer, "dust-cover");
    }

    // Accessory rail: a base bar with Picatinny cross ribs forward of the trigger
    // guard (z = 0.058 .. 0.098), i.e. underneath the front of the dust cover.
    part.add(boxAt(Vec3(0, -0.0262f, 0.0780f), Vec3(0.0078f, 0.0010f, 0.0200f), matPolymer),
             matPolymer, "rail-base");
    for (int i = 0; i < 4; ++i) {
        const float z = 0.0640f + 0.01010f * float(i);
        part.add(boxAt(Vec3(0, -0.0287f, z), Vec3(0.0104f, 0.0014f, 0.00235f), matPolymer),
                 matPolymer, "rail-rib");
    }

    // Trigger guard: an extruded U in the YZ plane with chamfered flanks.  The front
    // face is squared off and carries checkering ridges.
    // (u, v) = (y, z); a squared front, a rounded bottom and a rear arm that blends
    // into the frontstrap.
    const std::vector<Vec2> guardU = {
        {-0.0250f, 0.0575f}, {-0.0330f, 0.0581f}, {-0.0400f, 0.0584f}, {-0.0470f, 0.0581f},
        {-0.0520f, 0.0572f}, {-0.0560f, 0.0556f}, {-0.0592f, 0.0532f}, {-0.0618f, 0.0500f},
        {-0.0638f, 0.0460f}, {-0.0650f, 0.0415f}, {-0.0657f, 0.0370f}, {-0.0660f, 0.0320f},
        {-0.0659f, 0.0270f}, {-0.0654f, 0.0225f}, {-0.0645f, 0.0180f}, {-0.0630f, 0.0135f},
        {-0.0605f, 0.0090f}, {-0.0570f, 0.0048f}, {-0.0520f, 0.0010f}, {-0.0465f, -0.0018f},
        {-0.0405f, -0.0038f}, {-0.0350f, -0.0048f}, {-0.0300f, -0.0050f},
        {-0.0320f, 0.0090f}, {-0.0355f, 0.0105f}, {-0.0395f, 0.0122f}, {-0.0440f, 0.0143f},
        {-0.0485f, 0.0170f}, {-0.0525f, 0.0205f}, {-0.0558f, 0.0250f}, {-0.0578f, 0.0300f},
        {-0.0587f, 0.0350f}, {-0.0580f, 0.0400f}, {-0.0565f, 0.0445f}, {-0.0540f, 0.0475f},
        {-0.0500f, 0.0495f}, {-0.0450f, 0.0502f}, {-0.0400f, 0.0503f}, {-0.0350f, 0.0498f},
        {-0.0300f, 0.0490f}, {-0.0250f, 0.0480f},
    };
    {
        std::vector<Section> sections;
        const float xs[5] = {-0.0110f, -0.0100f, 0.0f, 0.0100f, 0.0110f};
        const float off[5] = {0.0012f, 0.0f, 0.0f, 0.0f, 0.0012f};
        for (int i = 0; i < 5; ++i) {
            Section s;
            s.pos = xs[i];
            s.poly = offsetPolygon(guardU, -off[i]);
            sections.push_back(std::move(s));
        }
        part.add(loft(sections, Axis::X, kUv), matPolymer, "trigger-guard");
    }
    // Checkering on the squared front face of the guard.
    for (int i = 0; i < 4; ++i) {
        const float y = -0.0300f - 0.0048f * float(i);
        part.add(boxAt(Vec3(0, y, 0.0580f), Vec3(0.0095f, 0.0012f, 0.0011f), matPolymer,
                       quatFromAxisAngle(Vec3(1, 0, 0), -3.0f * DEG2RAD)),
                 matPolymer, "guard-checkering");
    }

    // Frame top deck and rails: the slot the slide rides in (hidden by the slide in
    // battery, exposed as soon as the slide cycles).
    part.add(boxAt(Vec3(0, -0.0115f, 0.0120f), Vec3(0.0120f, 0.0025f, 0.0200f), matPolymer),
             matPolymer, "top-deck");
    for (int side = 0; side < 2; ++side) {
        const float sx = side == 0 ? 1.0f : -1.0f;
        part.add(boxAt(Vec3(sx * 0.0128f, -0.0120f, 0.0110f), Vec3(0.0008f, 0.0020f, 0.0190f),
                       matPolymer),
                 matPolymer, "frame-rail");
    }

    // Magazine release at the trigger guard junction (left side, as on the USP).
    part.add(boxAt(Vec3(-0.0152f, -0.0350f, 0.0035f), Vec3(0.0008f, 0.0044f, 0.0040f), matPolymer),
             matPolymer, "mag-release");
    part.add(cylXAt(Vec3(-0.01530f, -0.0350f, 0.0035f), 0.0022f, 0.0018f, 10, matSteel), matSteel,
             "mag-release-pin");
    return part.finish(35.0f, true);
}

// ===========================================================================
// grip
// ===========================================================================
MeshData buildGrip(uint32_t matGrip) {
    Part part;

    // Palm swell: the backstrap bulges rearward around the middle of the grip.
    auto backSwell = [](float y) {
        const float t = clamp((y - kGripTopY) / (kGripBotY - kGripTopY), 0.0f, 1.0f);
        return -0.0020f * std::sin(PI * clamp((t - 0.12f) / 0.76f, 0.0f, 1.0f));
    };
    struct Row {
        float y, halfW, rFront, rBack;
    };
    const Row rows[] = {
        {kGripBotY, 0.01560f, 0.0060f, 0.0075f},   // slight flare at the magwell
        {-0.1000f, 0.01500f, 0.0060f, 0.0075f},
        {-0.0850f, 0.01530f, 0.0060f, 0.0080f},
        {-0.0680f, 0.01550f, 0.0062f, 0.0085f},
        {-0.0520f, 0.01550f, 0.0062f, 0.0085f},
        {-0.0400f, 0.01520f, 0.0060f, 0.0080f},
        {kGripTopY, 0.01500f, 0.0055f, 0.0070f},
    };
    {
        std::vector<Section> sections;
        for (const Row& r : rows) {
            const float zf = frontstrapZ(r.y);
            const float zb = backstrapZ(r.y) + backSwell(r.y);
            Section s;
            s.pos = r.y;
            s.poly = roundedRect(zb, zf, -r.halfW, r.halfW, r.rBack, r.rFront, 5, 6);
            sections.push_back(std::move(s));
        }
        part.add(loft(sections, Axis::Y, kUv), matGrip, "grip-shell");
    }

    // Half width of the grip shell at an arbitrary height (the stipple studs are placed
    // against the local surface so none of them float).
    auto halfWAt = [&rows](float y) {
        if (y <= rows[0].y) return rows[0].halfW;
        const int n = static_cast<int>(sizeof(rows) / sizeof(rows[0]));
        for (int i = 1; i < n; ++i) {
            if (y <= rows[i].y) {
                const float t = (y - rows[i - 1].y) / (rows[i].y - rows[i - 1].y);
                return lerpf(rows[i - 1].halfW, rows[i].halfW, t);
            }
        }
        return rows[n - 1].halfW;
    };

    // Stippled side panels: a grid of small raised studs on both flanks.
    const int cols = 7;
    const int rws = 10;
    for (int side = 0; side < 2; ++side) {
        const float sx = side == 0 ? 1.0f : -1.0f;
        for (int r = 0; r < rws; ++r) {
            const float ty = (float(r) + 0.5f) / float(rws);
            const float y = lerpf(-0.1055f, -0.0425f, ty);
            const float zf = frontstrapZ(y) - 0.0052f;
            const float zb = backstrapZ(y) + 0.0060f;
            const float hw = halfWAt(y);
            for (int c = 0; c < cols; ++c) {
                const float tc = (float(c) + 0.5f) / float(cols);
                const float z = lerpf(zb, zf, tc);
                // Embedded 0.5 mm, proud 0.7 mm.
                part.add(boxAt(Vec3(sx * (hw + 0.00010f), y, z),
                               Vec3(0.00060f, 0.00145f, 0.00145f), matGrip),
                         matGrip, "stippling");
            }
        }
    }

    // Frontstrap finger grooves: three ridges across the flat middle of the frontstrap.
    for (int i = 0; i < 3; ++i) {
        const float y = -0.0470f - 0.0155f * float(i);
        const float z = frontstrapZ(y) - 0.0002f;
        part.add(boxAt(Vec3(0, y, z), Vec3(0.0090f, 0.0017f, 0.0012f), matGrip,
                       quatFromAxisAngle(Vec3(1, 0, 0), -20.0f * DEG2RAD)),
                 matGrip, "finger-groove");
    }
    return part.finish(35.0f, false);
}

// ===========================================================================
// trigger
// ===========================================================================
MeshData buildTrigger(uint32_t matPolymer, uint32_t matSteel) {
    Part part;
    // Blade cross section in (y, z); the pivot sits at the top rear.
    const std::vector<Vec2> blade = {
        {-0.0250f, 0.0040f}, {-0.0250f, 0.0140f}, {-0.0330f, 0.0172f}, {-0.0430f, 0.0208f},
        {-0.0510f, 0.0202f}, {-0.0555f, 0.0176f}, {-0.0560f, 0.0126f}, {-0.0500f, 0.0106f},
        {-0.0400f, 0.0116f}, {-0.0310f, 0.0092f},
    };
    {
        std::vector<Section> sections;
        const float xs[4] = {-0.0055f, -0.0047f, 0.0047f, 0.0055f};
        const float off[4] = {0.0010f, 0.0f, 0.0f, 0.0010f};
        for (int i = 0; i < 4; ++i) {
            Section s;
            s.pos = xs[i];
            s.poly = offsetPolygon(blade, -off[i]);
            sections.push_back(std::move(s));
        }
        part.add(loft(sections, Axis::X, kUv), matPolymer, "trigger-blade");
    }
    // Trigger safety blade: a narrow steel blade let into the face of the trigger,
    // standing ~0.6 mm proud of it (the "safety blade in trigger" detail).
    {
        std::vector<Vec2> safe = offsetPolygon(blade, -0.0015f);
        for (Vec2& p : safe)
            if (p.x < -0.0330f) p = Vec2(p.x, p.y + 0.0006f);
        std::vector<Section> sections;
        for (int i = 0; i < 3; ++i) {
            Section s;
            s.pos = -0.0019f + 0.0019f * float(i);
            s.poly = safe;
            sections.push_back(std::move(s));
        }
        part.add(loft(sections, Axis::X, kUv), matSteel, "trigger-safety-blade", false);
    }
    return part.finish(35.0f, true);
}

// ===========================================================================
// hammer (bobbed, spurless)
// ===========================================================================
MeshData buildHammer(uint32_t matSteel) {
    const std::vector<Vec2> body = {
        {-0.0158f, -0.0148f}, {-0.0158f, -0.0075f}, {-0.0125f, -0.0055f}, {-0.0060f, -0.0062f},
        {-0.0012f, -0.0082f}, {0.0005f, -0.0112f},  {-0.0006f, -0.0148f}, {-0.0044f, -0.0172f},
        {-0.0096f, -0.0166f}, {-0.0138f, -0.0152f},
    };
    Part part;
    std::vector<Section> sections;
    const float xs[4] = {-0.0045f, -0.0037f, 0.0037f, 0.0045f};
    const float off[4] = {0.0009f, 0.0f, 0.0f, 0.0009f};
    for (int i = 0; i < 4; ++i) {
        Section s;
        s.pos = xs[i];
        s.poly = offsetPolygon(body, -off[i]);
        sections.push_back(std::move(s));
    }
    part.add(loft(sections, Axis::X, kUv), matSteel, "hammer-body");
    part.add(cylXAt(Vec3(0.0f, kHammerPivotY, kHammerPivotZ), 0.0022f, 0.0104f, 10, matSteel),
             matSteel, "hammer-pin");
    return part.finish(35.0f, true);
}

// ===========================================================================
// magazine + floorplate
// ===========================================================================
MeshData buildMagazine(uint32_t matSteel) {
    struct Row {
        float y, halfW, inset;
    };
    const Row rows[] = {
        {-0.1125f, 0.01220f, 0.0042f},
        {-0.1050f, 0.01230f, 0.0040f},
        {-0.0800f, 0.01250f, 0.0038f},
        {-0.0500f, 0.01270f, 0.0036f},
        {-0.0300f, 0.01280f, 0.0034f},
    };
    std::vector<Section> sections;
    for (const Row& r : rows) {
        Section s;
        s.pos = r.y;
        s.poly = roundedRect(backstrapZ(r.y) + r.inset, frontstrapZ(r.y) - r.inset, -r.halfW,
                             r.halfW, 0.0030f, 0.0030f, 3, 4);
        sections.push_back(std::move(s));
    }
    MeshData m = loft(sections, Axis::Y, kUv);
    m.submeshes.clear();
    SubMesh sm;
    sm.indexOffset = 0;
    sm.indexCount = m.indexCount();
    sm.material = matSteel;
    sm.name = "magazine-body";
    sm.bounds = m.bounds;
    m.submeshes.push_back(sm);
    return m;
}

MeshData buildFloorplate(uint32_t matGrip) {
    const float zc = backstrapZ(kGripBotY) + 0.0302f;
    MeshData m = procgen::makeRoundedBox(Vec3(0.01580f, 0.00220f, 0.03050f), 0.0018f, 2, matGrip);
    procgen::scaleUv(m, Vec2(kUv, kUv));
    m.transform(mat4Translate(Vec3(0, kMagFloorBottomY + 0.0022f, zc)));
    return m;
}

// ===========================================================================
// sights
// ===========================================================================
MeshData buildFrontSight(uint32_t matSteel) {
    Part part;
    {
        std::vector<Section> sections;
        const float zs[3] = {0.0878f, 0.0920f, 0.0955f};
        for (int i = 0; i < 3; ++i) {
            Section s;
            s.pos = zs[i];
            const float hw = (i == 2) ? 0.0012f : 0.0018f;
            const float top = (i == 0) ? 0.0178f : (i == 2 ? 0.0184f : 0.0186f);
            s.poly = roundedRect(-hw, hw, kSlideTopY, top, 0.0006f, 0.0006f, 2, 2);
            sections.push_back(std::move(s));
        }
        part.add(loft(sections, Axis::Z, kUv), matSteel, "front-sight-post");
    }
    // White aiming dot (raised disc on the rear face of the post; see the note in the
    // report - SceneMaterials has no dedicated white paint slot).
    part.add(cylZAt(Vec3(0.0f, 0.0166f, 0.0876f), 0.00110f, 0.0007f, 12, matSteel), matSteel,
             "front-sight-dot", false);
    return part.finish(35.0f, true);
}

MeshData buildRearSight(uint32_t matSteel) {
    Part part;
    {
        // Notched block; the notch runs front to back so a small loft is enough.
        const std::vector<Vec2> sec = {
            {-0.0070f, 0.0145f}, {0.0070f, 0.0145f}, {0.0070f, kRearSightTopY},
            {0.0012f, kRearSightTopY}, {0.0012f, 0.0168f}, {-0.0012f, 0.0168f},
            {-0.0012f, kRearSightTopY}, {-0.0070f, kRearSightTopY},
        };
        std::vector<Section> sections;
        const float zs[3] = {0.0060f, 0.0160f, 0.0240f};
        for (int i = 0; i < 3; ++i) {
            Section s;
            s.pos = zs[i];
            s.poly = sec;
            sections.push_back(std::move(s));
        }
        part.add(loft(sections, Axis::Z, kUv), matSteel, "rear-sight-body");
    }
    for (int i = 0; i < 2; ++i) {
        const float x = i == 0 ? 0.0046f : -0.0046f;
        part.add(cylZAt(Vec3(x, 0.0172f, 0.0059f), 0.00110f, 0.0007f, 12, matSteel), matSteel,
                 "rear-sight-dot", false);
    }
    return part.finish(35.0f, true);
}

// ===========================================================================
// small controls
// ===========================================================================
MeshData buildSlideStop(uint32_t matSteel) {
    Part part;
    const std::vector<Vec2> lever = {
        {-0.0110f, -0.0060f}, {-0.0210f, -0.0075f}, {-0.0235f, -0.0130f},
        {-0.0195f, -0.0245f}, {-0.0135f, -0.0230f}, {-0.0105f, -0.0120f},
    };
    std::vector<Section> sections;
    const float xs[4] = {-0.0162f, -0.0154f, -0.0146f, -0.0140f};
    const float off[4] = {0.0009f, 0.0f, 0.0f, 0.0009f};
    for (int i = 0; i < 4; ++i) {
        Section s;
        s.pos = xs[i];
        s.poly = offsetPolygon(lever, -off[i]);
        sections.push_back(std::move(s));
    }
    part.add(loft(sections, Axis::X, kUv), matSteel, "slide-stop-lever");
    part.add(cylXAt(Vec3(-0.01530f, -0.0110f, -0.0015f), 0.0026f, 0.0022f, 12, matSteel), matSteel,
             "slide-stop-pin");
    return part.finish(35.0f, true);
}

MeshData buildSafetyLever(uint32_t matSteel) {
    Part part;
    const std::vector<Vec2> paddle = {
        {-0.0080f, -0.0040f}, {-0.0075f, -0.0140f}, {-0.0105f, -0.0290f}, {-0.0150f, -0.0320f},
        {-0.0200f, -0.0290f}, {-0.0215f, -0.0190f}, {-0.0180f, -0.0060f}, {-0.0120f, -0.0025f},
    };
    std::vector<Section> sections;
    const float xs[4] = {-0.0163f, -0.0156f, -0.0148f, -0.0142f};
    const float off[4] = {0.0008f, 0.0f, 0.0f, 0.0008f};
    for (int i = 0; i < 4; ++i) {
        Section s;
        s.pos = xs[i];
        s.poly = offsetPolygon(paddle, -off[i]);
        sections.push_back(std::move(s));
    }
    part.add(loft(sections, Axis::X, kUv), matSteel, "safety-lever");
    part.add(cylXAt(Vec3(-0.01540f, -0.0135f, -0.0045f), 0.0024f, 0.0022f, 12, matSteel), matSteel,
             "safety-pin");
    return part.finish(35.0f, true);
}

MeshData buildDecockLever(uint32_t matSteel) {
    const std::vector<Vec2> paddle = {
        {-0.0175f, 0.0005f}, {-0.0265f, -0.0020f}, {-0.0288f, -0.0075f},
        {-0.0245f, -0.0135f}, {-0.0180f, -0.0115f}, {-0.0165f, -0.0050f},
    };
    std::vector<Section> sections;
    const float xs[3] = {-0.0158f, -0.0152f, -0.0146f};
    const float off[3] = {0.0007f, 0.0f, 0.0007f};
    for (int i = 0; i < 3; ++i) {
        Section s;
        s.pos = xs[i];
        s.poly = offsetPolygon(paddle, -off[i]);
        sections.push_back(std::move(s));
    }
    MeshData m = loft(sections, Axis::X, kUv);
    procgen::boxProjectUv(m, kUv);
    m.computeNormals(35.0f);
    m.computeBounds();
    m.computeTangents();
    m.submeshes.clear();
    SubMesh sm;
    sm.indexOffset = 0;
    sm.indexCount = m.indexCount();
    sm.material = matSteel;
    sm.name = "decock-lever";
    sm.bounds = m.bounds;
    m.submeshes.push_back(sm);
    return m;
}

MeshData buildExtractor(uint32_t matSteel) {
    Part part;
    const std::vector<Vec2> claw = {
        {-0.0075f, 0.0290f}, {0.0025f, 0.0295f}, {0.0035f, 0.0360f},
        {0.0008f, 0.0435f},  {-0.0075f, 0.0440f},
    };
    std::vector<Section> sections;
    const float xs[3] = {0.01180f, 0.01440f, 0.01530f};
    const float off[3] = {0.0000f, 0.0000f, 0.0009f};
    for (int i = 0; i < 3; ++i) {
        Section s;
        s.pos = xs[i];
        s.poly = offsetPolygon(claw, -off[i]);
        sections.push_back(std::move(s));
    }
    part.add(loft(sections, Axis::X, kUv), matSteel, "extractor-claw");
    return part.finish(35.0f, true);
}

MeshData buildGuideRod(uint32_t matSteel) {
    // One closed solid of revolution: rod with a flanged head at the muzzle end.
    const std::vector<Vec2> profile = {
        {0.00000f, 0.04000f},  // rear face (pole)
        {0.00320f, 0.04000f},
        {0.00320f, 0.09980f},  // rod
        {0.00480f, 0.09980f},  // flange
        {0.00480f, 0.10250f},
        {0.00360f, 0.10250f},  // head face
        {0.00360f, 0.10150f},
        {0.00000f, 0.10150f},  // front face (pole)
    };
    MeshData m = revolveZ(profile, 24, kUv, matSteel);
    // The recoil assembly hangs under the barrel, not on the bore axis.
    m.transform(mat4Translate(Vec3(0, kGuideRodY, 0)));
    m.submeshes.clear();
    SubMesh sm;
    sm.indexOffset = 0;
    sm.indexCount = m.indexCount();
    sm.material = matSteel;
    sm.name = "guide-rod";
    sm.bounds = m.bounds;
    m.submeshes.push_back(sm);
    return m;
}

MeshData buildRecoilSpring(uint32_t matSteel) {
    Part part;
    const int coils = 13;
    for (int i = 0; i < coils; ++i) {
        const float z = 0.0480f + 0.00370f * float(i);
        part.add(coilZ(0.00460f, 0.00120f, 12, 6, z, matSteel), matSteel, "recoil-spring");
    }
    // Close the ends of the spring with a flat washer so it does not read as a stack of
    // floating rings when the slide is back.
    for (int i = 0; i < 3; ++i) {
        part.add(cylZ(0.00360f, 0.0455f + 0.0006f * float(i), 0.0461f + 0.0006f * float(i), 12,
                      matSteel),
                 matSteel, "spring-collar");
    }
    MeshData m = part.finish(35.0f, false);
    m.transform(mat4Translate(Vec3(0, kGuideRodY, 0)));
    return m;
}

}  // namespace

// ------------------------------------------------------------------ public API
WeaponMeshSet buildWeapon(Scene& scene, const SceneMaterials& materials) {
    WeaponMeshSet out;
    const uint32_t steel = materials.gunSlide;
    const uint32_t polymer = materials.gunFrame;
    const uint32_t gripMat = materials.gunGrip;
    const uint32_t barrelMat = materials.gunBarrel;

    auto put = [&](WeaponPart part, MeshData mesh, const char* name) {
        out.partMeshes[static_cast<uint32_t>(part)] = scene.addMesh(mesh, name);
    };
    put(WeaponPart::Slide, buildSlide(steel), "weapon.slide");
    put(WeaponPart::Barrel, buildBarrel(barrelMat), "weapon.barrel");
    put(WeaponPart::Frame, buildFrame(polymer, steel), "weapon.frame");
    put(WeaponPart::Grip, buildGrip(gripMat), "weapon.grip");
    put(WeaponPart::Trigger, buildTrigger(polymer, steel), "weapon.trigger");
    put(WeaponPart::Hammer, buildHammer(steel), "weapon.hammer");
    put(WeaponPart::Magazine, buildMagazine(steel), "weapon.magazine");
    put(WeaponPart::MagazineFloorplate, buildFloorplate(gripMat), "weapon.magazine-floorplate");
    put(WeaponPart::FrontSight, buildFrontSight(steel), "weapon.front-sight");
    put(WeaponPart::RearSight, buildRearSight(steel), "weapon.rear-sight");
    put(WeaponPart::SlideStop, buildSlideStop(steel), "weapon.slide-stop");
    put(WeaponPart::SafetyLever, buildSafetyLever(steel), "weapon.safety-lever");
    put(WeaponPart::DecockLever, buildDecockLever(steel), "weapon.decock-lever");
    put(WeaponPart::Extractor, buildExtractor(steel), "weapon.extractor");
    put(WeaponPart::GuideRod, buildGuideRod(steel), "weapon.guide-rod");
    put(WeaponPart::RecoilSpring, buildRecoilSpring(steel), "weapon.recoil-spring");

    out.count = static_cast<uint32_t>(WeaponPart::Count);
    Aabb b;
    for (uint32_t i = 0; i < out.count; ++i) {
        const uint32_t mesh = out.partMeshes[i];
        if (mesh < scene.meshCount()) b.expand(scene.mesh(mesh).bounds);
    }
    out.assembledBounds = b;
    return out;
}

uint32_t buildCartridgeCase(Scene& scene, const SceneMaterials& materials) {
    // .45 ACP: 22.81 mm long, 12.09 mm rim, extractor groove, spent primer pocket.
    // The profile is a closed cup of revolution about Y, centred on the case's middle.
    const float L = 0.01135f;   // half length
    const std::vector<Vec2> profile = {
        {0.00000f, -0.01015f},  // primer pocket floor (pole)
        {0.00290f, -0.01015f},
        {0.00300f, -L},         // pocket wall up to the head face
        {0.00590f, -L},         // head face out to the rim
        {0.00605f, -0.01090f},  // rim
        {0.00605f, -0.01000f},
        {0.00570f, -0.00920f},  // extractor groove
        {0.00570f, -0.00820f},
        {0.00590f, -0.00740f},
        {0.00598f, -0.00600f},
        {0.00600f, 0.00000f},   // body
        {0.00595f, 0.00850f},
        {0.00590f, L},          // case mouth, outside
        {0.00520f, L},          // mouth rim (wall thickness)
        {0.00510f, 0.01060f},
        {0.00480f, -0.00650f},  // interior wall
        {0.00000f, -0.00650f},  // interior floor (pole)
    };
    MeshData m = procgen::makeLathe(profile, 24, materials.brass);
    procgen::scaleUv(m, Vec2(TWO_PI * 0.006f * kUv, 1.0f));
    m.submeshes.clear();
    SubMesh sm;
    sm.indexOffset = 0;
    sm.indexCount = m.indexCount();
    sm.material = materials.brass;
    sm.name = "cartridge-case";
    sm.bounds = m.bounds;
    m.submeshes.push_back(sm);
    return scene.addMesh(m, "weapon.cartridge-case");
}

namespace {

// 2D convex hull (monotone chain) with collinear points removed, so the caps of the
// shard never contain a zero-area triangle.
std::vector<Vec2> convexHull2D(std::vector<Vec2> pts) {
    std::sort(pts.begin(), pts.end(), [](Vec2 a, Vec2 b) {
        return a.x < b.x || (a.x == b.x && a.y < b.y);
    });
    pts.erase(std::unique(pts.begin(), pts.end(),
                          [](Vec2 a, Vec2 b) { return length(a - b) < 1e-7f; }),
              pts.end());
    const int n = static_cast<int>(pts.size());
    if (n < 3) return pts;
    std::vector<Vec2> h(static_cast<size_t>(2 * n));
    int k = 0;
    for (int i = 0; i < n; ++i) {
        while (k >= 2 && cross2(h[static_cast<size_t>(k - 1)] - h[static_cast<size_t>(k - 2)],
                                pts[static_cast<size_t>(i)] - h[static_cast<size_t>(k - 2)]) <= 1e-10f)
            --k;
        h[static_cast<size_t>(k++)] = pts[static_cast<size_t>(i)];
    }
    for (int i = n - 2, t = k + 1; i >= 0; --i) {
        while (k >= t && cross2(h[static_cast<size_t>(k - 1)] - h[static_cast<size_t>(k - 2)],
                                pts[static_cast<size_t>(i)] - h[static_cast<size_t>(k - 2)]) <= 1e-10f)
            --k;
        h[static_cast<size_t>(k++)] = pts[static_cast<size_t>(i)];
    }
    h.resize(static_cast<size_t>(max2(k - 1, 0)));
    return h;
}

}  // namespace

uint32_t buildGlassShard(Scene& scene, const SceneMaterials& materials, const Rng& rng,
                         float size, float thickness) {
    // The Rng is taken by const reference, so the caller's sequence can not be advanced
    // from here: the shard is a pure function of (rng state, size, thickness).  The game
    // must vary the generator per shard (e.g. a fresh Rng(seed + i)) to get variety.
    Rng r = rng;
    const float s = max2(size, 1.0e-3f);
    const float th = clamp(thickness, 1.0e-3f, s * 0.30f);

    // 1. irregular radial points, then their convex hull.
    const int wanted = 3 + r.rangeInt(0, 3);  // 3..5
    std::vector<Vec2> pts;
    float ang = r.range(0.0f, TWO_PI);
    for (int i = 0; i < wanted; ++i) {
        ang += r.range(0.45f, 1.15f) * (TWO_PI / float(wanted));
        const float rad = s * r.range(0.30f, 0.50f);
        pts.push_back(Vec2(rad * std::cos(ang), rad * std::sin(ang)));
    }
    std::vector<Vec2> hull = convexHull2D(pts);
    if (hull.size() < 3) {
        hull = {Vec2(-0.5f * s, -0.34f * s), Vec2(0.52f * s, -0.28f * s), Vec2(-0.04f, 0.46f * s)};
    }
    const int n = static_cast<int>(hull.size());

    // 2. two jittered faces: the far one is tapered, which makes the shard a wedge.
    std::vector<Vec2> bot(static_cast<size_t>(n)), top(static_cast<size_t>(n));
    std::vector<float> zb(static_cast<size_t>(n)), zt(static_cast<size_t>(n));
    const float taper = r.range(0.55f, 0.92f);
    for (int i = 0; i < n; ++i) {
        bot[static_cast<size_t>(i)] = hull[static_cast<size_t>(i)];
        top[static_cast<size_t>(i)] = hull[static_cast<size_t>(i)] * taper;
        zb[static_cast<size_t>(i)] = r.range(0.0f, 0.30f) * th;
        zt[static_cast<size_t>(i)] = th * r.range(0.70f, 1.30f);
    }

    // 3. normalise the 2D spread so the bounding box diagonal comes out at `size`.
    float mnx = 1e30f, mxx = -1e30f, mny = 1e30f, mxy = -1e30f, mnz = 1e30f, mxz = -1e30f;
    auto acc = [&](Vec2 p, float z) {
        mnx = min2(mnx, p.x); mxx = max2(mxx, p.x);
        mny = min2(mny, p.y); mxy = max2(mxy, p.y);
        mnz = min2(mnz, z);   mxz = max2(mxz, z);
    };
    for (int i = 0; i < n; ++i) { acc(bot[static_cast<size_t>(i)], zb[static_cast<size_t>(i)]);
                                  acc(top[static_cast<size_t>(i)], zt[static_cast<size_t>(i)]); }
    const float dz = mxz - mnz;
    const float targetXY = std::sqrt(max2(s * s - dz * dz, 0.25f * s * s));
    const float haveXY = std::sqrt((mxx - mnx) * (mxx - mnx) + (mxy - mny) * (mxy - mny));
    const float kxy = haveXY > 1e-6f ? targetXY / haveXY : 1.0f;
    for (int i = 0; i < n; ++i) {
        bot[static_cast<size_t>(i)] = bot[static_cast<size_t>(i)] * kxy;
        top[static_cast<size_t>(i)] = top[static_cast<size_t>(i)] * kxy;
    }

    // 4. build the closed prism with flat-shaded faces (duplicated vertices).
    MeshData m;
    auto push = [&m](Vec3 p) {
        Vertex v;
        v.position = p;
        v.normal = Vec3(0, 0, 1);
        v.uv = Vec2(0, 0);
        v.uv1 = v.uv;
        v.tangent = Vec4(1, 0, 0, 1);
        m.vertices.push_back(v);
        return static_cast<uint32_t>(m.vertices.size() - 1);
    };
    std::vector<uint32_t> bi(static_cast<size_t>(n)), ti(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        bi[static_cast<size_t>(i)] = push(Vec3(bot[static_cast<size_t>(i)].x, bot[static_cast<size_t>(i)].y,
                                               zb[static_cast<size_t>(i)]));
        ti[static_cast<size_t>(i)] = push(Vec3(top[static_cast<size_t>(i)].x, top[static_cast<size_t>(i)].y,
                                               zt[static_cast<size_t>(i)]));
    }
    for (int i = 0; i < n; ++i) {
        const int j = (i + 1) % n;
        tri(m, bi[static_cast<size_t>(i)], bi[static_cast<size_t>(j)], ti[static_cast<size_t>(j)]);
        tri(m, bi[static_cast<size_t>(i)], ti[static_cast<size_t>(j)], ti[static_cast<size_t>(i)]);
    }
    // Caps: a convex polygon, so a fan is safe and stays non-degenerate.
    for (int i = 1; i + 1 < n; ++i) {
        tri(m, bi[0], bi[static_cast<size_t>(i + 1)], bi[static_cast<size_t>(i)]);
        tri(m, ti[0], ti[static_cast<size_t>(i)], ti[static_cast<size_t>(i + 1)]);
    }
    // Planar UVs for the two faces and the rim (cheap but usable on glass).
    for (Vertex& v : m.vertices) v.uv = Vec2(v.position.x, v.position.y) * (kUv * 4.0f);
    m.computeNormals(20.0f);
    m.computeBounds();
    const Vec3 c = m.bounds.center();
    m.transform(mat4Translate(Vec3(-c.x, -c.y, -c.z)));
    m.computeTangents();
    m.submeshes.clear();
    SubMesh sm;
    sm.indexOffset = 0;
    sm.indexCount = m.indexCount();
    sm.material = materials.glass;
    sm.name = "glass-shard";
    sm.bounds = m.bounds;
    m.submeshes.push_back(sm);
    return scene.addMesh(m, "glass.shard");
}

// ------------------------------------------------------------------ test hooks
// Deliberately outside builders.hpp: the animation code only needs the parts, but the
// self test wants to check the pivots the geometry was actually built around.
Vec3 weaponTriggerPivot() { return Vec3(0.0f, kTriggerPivotY, kTriggerPivotZ); }
Vec3 weaponHammerPivot() { return Vec3(0.0f, kHammerPivotY, kHammerPivotZ); }

}  // namespace room2::scene
