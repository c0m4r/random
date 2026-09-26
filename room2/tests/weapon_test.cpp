// room2 - self test for src/scene/weapon.cpp (procedural HK USP, cartridge case and
// broken glass shards).
//
// Build:
//   g++ -std=c++20 -O2 -Wall -Wextra -I src tests/weapon_test.cpp src/scene/weapon.cpp
//       src/scene/scene.cpp src/procgen/mesh.cpp src/core/log.cpp -o build/weapon_test
// (src/core/log.cpp is required because Scene::addMesh logs a warning for empty meshes)
//
// Every check prints PASS/FAIL together with the numbers it measured.

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <array>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "core/math.hpp"
#include "core/rng.hpp"
#include "procgen/mesh.hpp"
#include "scene/builders.hpp"
#include "scene/scene.hpp"

using namespace room2;
using namespace room2::scene;

// Pivots the geometry was actually built around (defined in weapon.cpp; deliberately
// not part of the public builders.hpp API).
namespace room2::scene {
Vec3 weaponTriggerPivot();
Vec3 weaponHammerPivot();
}  // namespace room2::scene

// ---------------------------------------------------------------- test plumbing
static int g_pass = 0;
static int g_fail = 0;

static void check(bool ok, const char* fmt, ...) {
    char buf[1024];
    va_list args;
    va_start(args, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", buf);
    if (ok) ++g_pass; else ++g_fail;
}

static const char* partName(WeaponPart p) {
    switch (p) {
        case WeaponPart::Slide: return "Slide";
        case WeaponPart::Barrel: return "Barrel";
        case WeaponPart::Frame: return "Frame";
        case WeaponPart::Grip: return "Grip";
        case WeaponPart::Trigger: return "Trigger";
        case WeaponPart::Hammer: return "Hammer";
        case WeaponPart::Magazine: return "Magazine";
        case WeaponPart::MagazineFloorplate: return "MagazineFloorplate";
        case WeaponPart::FrontSight: return "FrontSight";
        case WeaponPart::RearSight: return "RearSight";
        case WeaponPart::SlideStop: return "SlideStop";
        case WeaponPart::SafetyLever: return "SafetyLever";
        case WeaponPart::DecockLever: return "DecockLever";
        case WeaponPart::Extractor: return "Extractor";
        case WeaponPart::GuideRod: return "GuideRod";
        case WeaponPart::RecoilSpring: return "RecoilSpring";
        default: return "?";
    }
}

// ---------------------------------------------------------------- mesh analysis
struct MeshStats {
    uint32_t mesh = kInvalidIndex;
    uint32_t vertices = 0;
    uint32_t triangles = 0;
    uint32_t degenerate = 0;
    uint32_t nonFinite = 0;
    uint32_t outOfRange = 0;
    Aabb bounds;
    Vec3 centroid{0, 0, 0};    // area weighted surface centroid
    double area = 0.0;
    double inwardArea = 0.0;   // area whose normal points towards the centroid
    double volume = 0.0;       // signed volume (divergence theorem)
    uint32_t inwardTris = 0;   // triangles whose normal points at the surface centroid
    std::vector<uint32_t> indices;   // expanded triangle corner indices
    std::vector<Vec3> positions;     // one entry per distinct vertex of the part
};

// A part is convex when no vertex lies in front of any of its face planes.  This is
// measured rather than assumed, so the "convex-ish" threshold below only ever applies
// to geometry where the centroid heuristic is actually meaningful.
static float maxPlaneViolation(const MeshStats& s, const std::vector<Vertex>& V) {
    float worst = 0.0f;
    for (uint32_t t = 0; t < s.triangles; ++t) {
        const Vec3 p0 = V[s.indices[t * 3 + 0]].position;
        const Vec3 p1 = V[s.indices[t * 3 + 1]].position;
        const Vec3 p2 = V[s.indices[t * 3 + 2]].position;
        const Vec3 n = cross(p1 - p0, p2 - p0);
        if (lengthSq(n) < 1e-20f) continue;
        const Vec3 nn = normalize(n);
        for (Vec3 p : s.positions) worst = std::max(worst, dot(nn, p - p0));
    }
    return worst;
}

// Splits the part into connected sub-solids (triangles welded through exactly coincident
// vertex positions) and returns the worst signed volume of any of them.  Composite parts
// are unions of overlapping closed solids, so this verifies the winding of every single
// piece instead of only the sum of them.
struct ComponentReport {
    uint32_t count = 0;
    double worstVolume = 0.0;
    uint32_t inverted = 0;
};

static ComponentReport componentVolumes(const MeshStats& s, const std::vector<Vertex>& V) {
    ComponentReport rep;
    const uint32_t n = s.triangles;
    if (n == 0) return rep;
    std::vector<int> parent(static_cast<size_t>(n));
    for (uint32_t i = 0; i < n; ++i) parent[i] = static_cast<int>(i);
    auto find = [&parent](int i) {
        while (parent[static_cast<size_t>(i)] != i) {
            parent[static_cast<size_t>(i)] = parent[static_cast<size_t>(parent[static_cast<size_t>(i)])];
            i = parent[static_cast<size_t>(i)];
        }
        return i;
    };
    std::map<std::array<long long, 3>, int> weld;
    for (uint32_t t = 0; t < n; ++t) {
        for (int c = 0; c < 3; ++c) {
            const Vec3 p = V[s.indices[t * 3 + c]].position;
            const std::array<long long, 3> key{static_cast<long long>(std::llround(p.x * 1e6)),
                                               static_cast<long long>(std::llround(p.y * 1e6)),
                                               static_cast<long long>(std::llround(p.z * 1e6))};
            const auto it = weld.find(key);
            if (it == weld.end()) {
                weld.emplace(key, static_cast<int>(t));
            } else {
                const int a = find(it->second);
                const int b = find(static_cast<int>(t));
                if (a != b) parent[static_cast<size_t>(a)] = b;
            }
        }
    }
    std::map<int, double> vol;
    for (uint32_t t = 0; t < n; ++t) {
        const Vec3 p0 = V[s.indices[t * 3 + 0]].position;
        const Vec3 p1 = V[s.indices[t * 3 + 1]].position;
        const Vec3 p2 = V[s.indices[t * 3 + 2]].position;
        vol[find(static_cast<int>(t))] += static_cast<double>(dot(p0, cross(p1, p2))) / 6.0;
    }
    rep.count = static_cast<uint32_t>(vol.size());
    bool first = true;
    for (const auto& kv : vol) {
        if (first || kv.second < rep.worstVolume) rep.worstVolume = kv.second;
        first = false;
        if (kv.second <= 0.0) ++rep.inverted;
    }
    return rep;
}

static bool isConvex(const MeshStats& s, const std::vector<Vertex>& V, float eps) {
    for (uint32_t t = 0; t < s.triangles; ++t) {
        const Vec3 p0 = V[s.indices[t * 3 + 0]].position;
        const Vec3 p1 = V[s.indices[t * 3 + 1]].position;
        const Vec3 p2 = V[s.indices[t * 3 + 2]].position;
        const Vec3 n = cross(p1 - p0, p2 - p0);
        if (lengthSq(n) < 1e-20f) continue;
        const Vec3 nn = normalize(n);
        for (Vec3 p : s.positions)
            if (dot(nn, p - p0) > eps) return false;
    }
    return true;
}

static bool finite3(Vec3 v) {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

// Scene::addMesh keeps each submesh's indices relative to its own vertex block and
// stores that block's base in MeshRange::vertexOffset (the draw call passes it as
// firstVertex).  Every vertex lookup in this test goes through here.
static uint32_t resolveVertex(const MeshRange& r, uint32_t localIndex, size_t vertexCount,
                              uint32_t* outOfRange) {
    const int64_t base = static_cast<int64_t>(r.vertexOffset);
    const int64_t vi = base + static_cast<int64_t>(localIndex);
    if (vi < 0 || static_cast<size_t>(vi) >= vertexCount) {
        if (outOfRange != nullptr) ++(*outOfRange);
        return 0;
    }
    return static_cast<uint32_t>(vi);
}

static MeshStats analyse(const Scene& scene, uint32_t meshIndex) {
    MeshStats st;
    st.mesh = meshIndex;
    if (meshIndex >= scene.meshCount()) return st;
    const Mesh& mesh = scene.mesh(meshIndex);
    const std::vector<Vertex>& V = scene.vertices();
    const std::vector<uint32_t>& I = scene.indices();

    int64_t lo = INT64_MAX, hi = -1;
    for (const MeshRange& r : mesh.ranges) {
        for (uint32_t k = 0; k + 2 < r.indexCount + 0; k += 3) {
            const uint32_t base = r.firstIndex + k;
            if (base + 2 >= I.size()) break;
            for (int c = 0; c < 3; ++c) {
                const uint32_t vi = resolveVertex(r, I[base + c], V.size(), &st.outOfRange);
                st.indices.push_back(vi);
                if (static_cast<int64_t>(vi) < lo) lo = vi;
                if (static_cast<int64_t>(vi) > hi) hi = vi;
                if (!finite3(V[vi].position)) ++st.nonFinite;
            }
        }
    }
    st.triangles = static_cast<uint32_t>(st.indices.size() / 3);
    if (hi < lo) return st;
    st.vertices = static_cast<uint32_t>(hi - lo + 1);

    // Pass 1: areas, bounds, centroid, signed volume.
    double areaSum = 0.0;
    Vec3 csum(0, 0, 0);
    for (uint32_t t = 0; t < st.triangles; ++t) {
        const Vec3 p0 = V[st.indices[t * 3 + 0]].position;
        const Vec3 p1 = V[st.indices[t * 3 + 1]].position;
        const Vec3 p2 = V[st.indices[t * 3 + 2]].position;
        if (!finite3(p0) || !finite3(p1) || !finite3(p2)) { ++st.degenerate; continue; }
        st.bounds.expand(p0); st.bounds.expand(p1); st.bounds.expand(p2);
        const Vec3 n = cross(p1 - p0, p2 - p0);
        const double a = 0.5 * static_cast<double>(length(n));
        if (!(a > 1e-12)) { ++st.degenerate; continue; }
        areaSum += a;
        csum += (p0 + p1 + p2) * (static_cast<float>(a) / 3.0f);
        st.volume += static_cast<double>(dot(p0, cross(p1, p2))) / 6.0;
    }
    st.area = areaSum;
    st.centroid = areaSum > 0.0 ? csum / static_cast<float>(areaSum) : st.bounds.center();

    // Distinct vertices of this part (for the convexity measurement).
    {
        std::vector<bool> seen(V.size(), false);
        for (uint32_t vi : st.indices) {
            if (vi < seen.size() && !seen[vi]) {
                seen[vi] = true;
                st.positions.push_back(V[vi].position);
            }
        }
    }

    // Pass 2: how much area faces away from the centroid.
    double inward = 0.0;
    for (uint32_t t = 0; t < st.triangles; ++t) {
        const Vec3 p0 = V[st.indices[t * 3 + 0]].position;
        const Vec3 p1 = V[st.indices[t * 3 + 1]].position;
        const Vec3 p2 = V[st.indices[t * 3 + 2]].position;
        const Vec3 n = cross(p1 - p0, p2 - p0);
        const double a = 0.5 * static_cast<double>(length(n));
        if (!(a > 1e-12)) continue;
        const Vec3 fc = (p0 + p1 + p2) / 3.0f;
        if (dot(normalize(n), fc - st.centroid) < 0.0f) {
            inward += a;
            ++st.inwardTris;
        }
    }
    st.inwardArea = inward;
    return st;
}

static double inwardFraction(const MeshStats& s) {
    return s.area > 0.0 ? s.inwardArea / s.area : 0.0;
}

static double diagonal(const Aabb& b) {
    const Vec3 e = b.mx - b.mn;
    return std::sqrt(static_cast<double>(e.x) * e.x + static_cast<double>(e.y) * e.y +
                     static_cast<double>(e.z) * e.z);
}

// Nearest part hit by a ray, used to prove the barrel really is what you see through
// the ejection port (and that the port is not blocked by slide material).
static int nearestPart(const Scene& scene, const WeaponMeshSet& gun, const Ray& ray,
                       float* outT) {
    int best = -1;
    float bestT = 1e30f;
    for (uint32_t p = 0; p < gun.count; ++p) {
        const Mesh& m = scene.mesh(gun.partMeshes[p]);
        for (const MeshRange& r : m.ranges) {
            for (uint32_t k = 0; k + 2 < r.indexCount; k += 3) {
                const Vec3 a = scene.vertices()[resolveVertex(r, scene.indices()[r.firstIndex + k], scene.vertices().size(), nullptr)].position;
                const Vec3 b = scene.vertices()[resolveVertex(r, scene.indices()[r.firstIndex + k + 1], scene.vertices().size(), nullptr)].position;
                const Vec3 c = scene.vertices()[resolveVertex(r, scene.indices()[r.firstIndex + k + 2], scene.vertices().size(), nullptr)].position;
                float t = 0.0f;
                if (rayTriangle(ray, a, b, c, t) && t < bestT) {
                    bestT = t;
                    best = static_cast<int>(p);
                }
            }
        }
    }
    if (outT != nullptr) *outT = bestT;
    return best;
}

static Material makeMaterial(const char* name, float metallic, float rough) {
    Material m;
    m.name = name;
    m.metallic = metallic;
    m.roughness = rough;
    return m;
}

int main() {
    std::printf("=== room2 weapon geometry test ===\n\n");

    Scene scene;
    scene.addMaterial(makeMaterial("white", 0.0f, 0.5f));   // slot 0: never used by the gun
    SceneMaterials mats;
    mats.gunSlide = scene.addMaterial(makeMaterial("gun.slide", 1.0f, 0.28f));
    mats.gunFrame = scene.addMaterial(makeMaterial("gun.frame", 0.0f, 0.55f));
    mats.gunGrip = scene.addMaterial(makeMaterial("gun.grip", 0.0f, 0.75f));
    mats.gunBarrel = scene.addMaterial(makeMaterial("gun.barrel", 1.0f, 0.35f));
    mats.brass = scene.addMaterial(makeMaterial("brass", 1.0f, 0.25f));
    mats.glass = scene.addMaterial(makeMaterial("glass", 0.0f, 0.05f));

    // ---------------------------------------------------------------- (a)
    std::printf("--- (a) parts, mesh handles, vertex/triangle counts ---\n");
    const WeaponMeshSet gun = buildWeapon(scene, mats);
    MeshStats stats[static_cast<uint32_t>(WeaponPart::Count)];
    uint32_t totalTris = 0;
    uint32_t totalVerts = 0;
    Aabb gunBounds;
    for (uint32_t i = 0; i < static_cast<uint32_t>(WeaponPart::Count); ++i) {
        const WeaponPart part = static_cast<WeaponPart>(i);
        const uint32_t mesh = gun.partMeshes[i];
        const bool valid = mesh < scene.meshCount();
        stats[i] = analyse(scene, mesh);
        check(valid && stats[i].triangles > 0 && stats[i].outOfRange == 0,
              "%-18s mesh=%2u verts=%4u tris=%4u  bounds x[%+.4f,%+.4f] y[%+.4f,%+.4f] "
              "z[%+.4f,%+.4f]",
              partName(part), mesh, stats[i].vertices, stats[i].triangles, stats[i].bounds.mn.x,
              stats[i].bounds.mx.x, stats[i].bounds.mn.y, stats[i].bounds.mx.y,
              stats[i].bounds.mn.z, stats[i].bounds.mx.z);
        totalTris += stats[i].triangles;
        totalVerts += stats[i].vertices;
        if (valid) gunBounds.expand(scene.mesh(mesh).bounds);
    }
    check(gun.count == static_cast<uint32_t>(WeaponPart::Count),
          "WeaponMeshSet::count == %u (WeaponPart::Count)", gun.count);
    {
        bool consistent = true;
        for (uint32_t i = 0; i < gun.count; ++i) {
            const Aabb& own = scene.mesh(gun.partMeshes[i]).bounds;
            const Aabb& got = stats[i].bounds;
            const bool ok = std::fabs(own.mn.x - got.mn.x) < 1e-6f &&
                            std::fabs(own.mx.x - got.mx.x) < 1e-6f &&
                            std::fabs(own.mn.y - got.mn.y) < 1e-6f &&
                            std::fabs(own.mx.y - got.mx.y) < 1e-6f &&
                            std::fabs(own.mn.z - got.mn.z) < 1e-6f &&
                            std::fabs(own.mx.z - got.mx.z) < 1e-6f;
            if (!ok) {
                std::printf("      %s: Mesh::bounds z[%+.5f,%+.5f] resolved z[%+.5f,%+.5f] "
                            "x[%+.5f,%+.5f]/[%+.5f,%+.5f]\n", partName(static_cast<WeaponPart>(i)),
                            own.mn.z, own.mx.z, got.mn.z, got.mx.z, own.mn.x, own.mx.x, got.mn.x,
                            got.mx.x);
            }
            consistent = consistent && ok;
        }
        check(consistent, "per-part vertex bounds match Mesh::bounds "
                          "(index convention: MeshRange::vertexOffset + local index)");
    }
    std::printf("  total: %u vertices, %u triangles\n", totalVerts, totalTris);

    // Material plumbing: each part must reference the material slots it was given.
    auto materialSet = [&](WeaponPart p, std::vector<uint32_t> allowed) {
        const Mesh& m = scene.mesh(gun.partMeshes[static_cast<uint32_t>(p)]);
        bool ok = !m.ranges.empty();
        for (const MeshRange& r : m.ranges) {
            bool found = false;
            for (uint32_t a : allowed) found = found || (r.material == a);
            ok = ok && found;
        }
        return ok;
    };
    check(materialSet(WeaponPart::Slide, {mats.gunSlide}) &&
              materialSet(WeaponPart::Barrel, {mats.gunBarrel}) &&
              materialSet(WeaponPart::Frame, {mats.gunFrame, mats.gunSlide}) &&
              materialSet(WeaponPart::Grip, {mats.gunGrip}) &&
              materialSet(WeaponPart::MagazineFloorplate, {mats.gunGrip}),
          "submesh materials: slide=gunSlide barrel=gunBarrel frame=gunFrame grip=gunGrip "
          "floorplate=gunGrip");

    // ---------------------------------------------------------------- (b)
    std::printf("\n--- (b) assembled bounding box vs the real HK USP ---\n");
    const Vec3 ext = gunBounds.mx - gunBounds.mn;
    std::printf("  bounds min (%.4f, %.4f, %.4f)  max (%.4f, %.4f, %.4f)\n", gunBounds.mn.x,
                gunBounds.mn.y, gunBounds.mn.z, gunBounds.mx.x, gunBounds.mx.y, gunBounds.mx.z);
    check(std::fabs(ext.z - 0.194f) <= 0.012f, "length  (Z) = %.4f m  (target 0.194 +-0.012)", ext.z);
    check(std::fabs(ext.y - 0.136f) <= 0.012f, "height  (Y) = %.4f m  (target 0.136 +-0.012)", ext.y);
    check(std::fabs(ext.x - 0.032f) <= 0.006f, "width   (X) = %.4f m  (target 0.032 +-0.006)", ext.x);
    check(std::fabs(gunBounds.mx.z - 0.108f) <= 0.004f && std::fabs(gunBounds.mn.z + 0.086f) <= 0.014f,
          "front-most z = %.4f (muzzle), rear-most z = %.4f", gunBounds.mx.z, gunBounds.mn.z);

    // ---------------------------------------------------------------- (c)
    std::printf("\n--- (c) barrel muzzle and bore axis ---\n");
    {
        const MeshStats& b = stats[static_cast<uint32_t>(WeaponPart::Barrel)];
        const Vec3 bc = b.bounds.center();
        check(std::fabs(b.bounds.mx.z - 0.108f) <= 0.004f, "muzzle z = %.4f (target +0.108 +-0.004)",
              b.bounds.mx.z);
        check(std::fabs(bc.x) <= 0.003f && std::fabs(bc.y) <= 0.003f,
              "barrel bounds centre x = %.4f, y = %.4f (bore axis at 0,0 +-0.003)", bc.x, bc.y);
        // Muzzle ring: every vertex of the barrel near the muzzle must straddle (0,0).
        const Mesh& m = scene.mesh(gun.partMeshes[static_cast<uint32_t>(WeaponPart::Barrel)]);
        Vec3 sum(0, 0, 0);
        uint32_t n = 0;
        for (const MeshRange& r : m.ranges) {
            for (uint32_t k = 0; k < r.indexCount; ++k) {
                const uint32_t vi = resolveVertex(r, scene.indices()[r.firstIndex + k],
                                                 scene.vertices().size(), nullptr);
                const Vec3 p = scene.vertices()[vi].position;
                if (p.z > 0.1060f) { sum += p; ++n; }
            }
        }
        const Vec3 mean = n > 0 ? sum / static_cast<float>(n) : Vec3(0, 0, 0);
        check(n >= 32 && std::fabs(mean.x) < 0.0015f && std::fabs(mean.y) < 0.0015f,
              "muzzle ring: %u vertices, centre (%.5f, %.5f)", n, mean.x, mean.y);
    }

    // ---------------------------------------------------------------- (d)
    std::printf("\n--- (d) slide rear face and slide top ---\n");
    {
        const MeshStats& s = stats[static_cast<uint32_t>(WeaponPart::Slide)];
        check(std::fabs(s.bounds.mn.z - 0.0f) <= 0.002f, "slide rear face z = %+.4f (target 0 +-0.002)",
              s.bounds.mn.z);
        check(std::fabs(s.bounds.mx.y - 0.0140f) <= 0.003f,
              "slide top y = %+.4f (target +0.014 +-0.003, incl. sight dovetails)", s.bounds.mx.y);
        check(s.bounds.mx.z > 0.099f && s.bounds.mx.z <= 0.101f, "slide front z = %.4f (~0.100)",
              s.bounds.mx.z);
    }

    // ---------------------------------------------------------------- (e)
    std::printf("\n--- (e) magazine can be animated out of the magwell ---\n");
    {
        const Aabb& mag = stats[static_cast<uint32_t>(WeaponPart::Magazine)].bounds;
        const Aabb& grip = stats[static_cast<uint32_t>(WeaponPart::Grip)].bounds;
        const float magMaxY = mag.mx.y - 0.11f;
        const float limit = grip.mn.y + 0.02f;
        check(magMaxY < limit,
              "magazine translated by (0,-0.11,0): max y = %.4f < grip min y + 0.02 = %.4f "
              "(clears by %.4f m)",
              magMaxY, limit, limit - magMaxY);
        check(mag.mn.y > grip.mn.y && mag.mx.y < grip.mx.y,
              "magazine is inside the grip in battery (mag y [%.4f, %.4f], grip y [%.4f, %.4f])",
              mag.mn.y, mag.mx.y, grip.mn.y, grip.mx.y);
    }

    // ---------------------------------------------------------------- (f)
    std::printf("\n--- (f) trigger pivot ---\n");
    {
        const Vec3 pivot = weaponTriggerPivot();
        // The design's trigger guard cavity (the enclosed finger volume plus the frame
        // recess the trigger hangs out of).
        const Aabb guardVolume{Vec3(-0.012f, -0.066f, 0.004f), Vec3(0.012f, -0.022f, 0.052f)};
        std::printf("  trigger pivot = (%.4f, %.4f, %.4f)\n", pivot.x, pivot.y, pivot.z);
        check(guardVolume.contains(pivot), "trigger pivot inside the trigger guard volume "
              "x[-0.012,0.012] y[-0.066,-0.022] z[0.004,0.052]");
        const MeshStats& t = stats[static_cast<uint32_t>(WeaponPart::Trigger)];
        check(t.bounds.contains(pivot), "pivot lies inside the trigger mesh bounds "
              "y[%.4f,%.4f] z[%.4f,%.4f]", t.bounds.mn.y, t.bounds.mx.y, t.bounds.mn.z, t.bounds.mx.z);
        check(pivot.y > t.bounds.mn.y && (t.bounds.mx.y - pivot.y) < 0.010f,
              "pivot is at the top of the blade (%.4f m below the blade top)",
              t.bounds.mx.y - pivot.y);

        // Rotating about that pivot must swing the blade rearwards, like a real trigger.
        const Mat4 rot = mat4FromQuat(quatFromAxisAngle(Vec3(1, 0, 0), 14.0f * DEG2RAD));
        const Mat4 about = mat4Translate(pivot) * rot * mat4Translate(-pivot);
        const Mesh& m = scene.mesh(gun.partMeshes[static_cast<uint32_t>(WeaponPart::Trigger)]);
        Vec3 tip(0, 0, 0);
        bool hasTip = false;
        for (const MeshRange& r : m.ranges) {
            for (uint32_t k = 0; k < r.indexCount; ++k) {
                const uint32_t vi = resolveVertex(r, scene.indices()[r.firstIndex + k],
                                                 scene.vertices().size(), nullptr);
                const Vec3 p = scene.vertices()[vi].position;
                if (p.y < t.bounds.mn.y + 0.0005f && (!hasTip || p.z > tip.z)) {
                    tip = p;
                    hasTip = true;
                }
            }
        }
        const Vec3 moved = transformPoint(about, tip);
        const Vec3 pivotAfter = transformPoint(about, pivot);
        check(hasTip && distance(pivotAfter, pivot) < 1e-6f && moved.z < tip.z - 0.003f,
              "14 deg about the pivot: blade tip z %.4f -> %.4f (%.4f m rearward, pivot fixed)",
              tip.z, moved.z, tip.z - moved.z);
    }

    std::printf("\n--- (f2) hammer pivot and ejection port ---\n");
    {
        const Vec3 pivot = weaponHammerPivot();
        const MeshStats& h = stats[static_cast<uint32_t>(WeaponPart::Hammer)];
        const float span = h.bounds.mx.y - h.bounds.mn.y;
        std::printf("  hammer pivot = (%.4f, %.4f, %.4f)\n", pivot.x, pivot.y, pivot.z);
        check(h.bounds.contains(pivot), "hammer pivot inside the hammer bounds "
              "y[%.4f,%.4f] z[%.4f,%.4f]", h.bounds.mn.y, h.bounds.mx.y, h.bounds.mn.z, h.bounds.mx.z);
        check(pivot.y < h.bounds.mn.y + 0.35f * span,
              "hammer pivot at the bottom of the bobbed hammer (%.4f m above its base of "
              "%.4f m total height)", pivot.y - h.bounds.mn.y, span);
        check(h.bounds.mx.z <= 0.0f, "hammer is behind the breech face (max z = %.4f)",
              h.bounds.mx.z);

        // Three rays straight into the ejection port window from the right hand side:
        // the first surface they meet must belong to the barrel.
        int hits[3] = {-1, -1, -1};
        float ts[3] = {0, 0, 0};
        const float zs[3] = {0.0520f, 0.0630f, 0.0760f};
        for (int i = 0; i < 3; ++i) {
            const Ray ray{Vec3(0.30f, 0.0005f, zs[i]), Vec3(-1, 0, 0)};
            hits[i] = nearestPart(scene, gun, ray, &ts[i]);
        }
        const int barrel = static_cast<int>(WeaponPart::Barrel);
        check(hits[0] == barrel && hits[1] == barrel && hits[2] == barrel,
              "through the ejection port the nearest surface is the Barrel "
              "(hits: %d,%d,%d, t = %.4f/%.4f/%.4f m)",
              hits[0], hits[1], hits[2], ts[0], ts[1], ts[2]);
    }

    // ---------------------------------------------------------------- (g)
    std::printf("\n--- (g) winding, normals, degeneracy ---\n");
    {
        const std::vector<Vertex>& V = scene.vertices();
        bool allClosed = true;
        bool strict = true;
        uint32_t convexCount = 0;
        for (uint32_t i = 0; i < static_cast<uint32_t>(WeaponPart::Count); ++i) {
            const MeshStats& s = stats[i];
            // 0.25 mm slack: faceting of a curved surface and the slight twist of a
            // tapered loft quad must not disqualify an otherwise convex shell.  Real
            // concavities in this model (trigger guard loop, ejection port, barrel bore,
            // protruding pins, stippling) are all >= 0.5 mm.
            const bool convex = isConvex(s, V, 2.5e-4f);
            const double triFrac = s.triangles > 0
                                       ? double(s.inwardTris) / double(s.triangles) : 0.0;
            // Convex shells are held to the strict 5 %; composite / concave / hollow
            // parts (overlapping solids, the triggerguard loop, the barrel bore, the
            // coil spring) are checked through their signed volume instead.
            const double limit = convex ? 0.05 : 0.50;
            if (convex) ++convexCount;
            const ComponentReport comp = componentVolumes(s, V);
            const bool ok = s.degenerate == 0 && s.nonFinite == 0 && s.outOfRange == 0 &&
                            s.volume > 0.0 && comp.inverted == 0 && triFrac < limit;
            allClosed = allClosed && ok;
            if (convex) strict = strict && triFrac < 0.05;
            std::printf("  %-18s degen=%u nonfinite=%u vol=%+.2e m^3 solids=%u (worst %+.2e) "
                        "inward=%5.2f%% area %5.2f%% concavity=%.3f mm  %s\n",
                        partName(static_cast<WeaponPart>(i)), s.degenerate, s.nonFinite, s.volume,
                        comp.count, comp.worstVolume, triFrac * 100.0, inwardFraction(s) * 100.0,
                        double(maxPlaneViolation(s, V)) * 1000.0,
                        convex ? "convex shell -> <5%" : "composite/hollow -> <50%");
        }
        std::printf("  %u of %u parts measure as convex shells\n", convexCount,
                    static_cast<uint32_t>(WeaponPart::Count));
        check(allClosed, "every part: 0 degenerate faces, 0 non-finite/oob vertices, and every "
                         "closed sub-solid has a positive signed volume (outward winding)");
        check(convexCount >= 2, "%u parts measure as convex shells (the rest are unions of "
                                "solids, loops or hollow shells)", convexCount);
        check(strict, "every convex part has < 5%% inward-facing triangles");
    }

    // ---------------------------------------------------------------- (h)
    std::printf("\n--- (h) .45 ACP cartridge case ---\n");
    {
        const uint32_t caseMesh = buildCartridgeCase(scene, mats);
        const MeshStats s = analyse(scene, caseMesh);
        const Vec3 e = s.bounds.mx - s.bounds.mn;
        const Vec3 c = s.bounds.center();
        check(s.triangles > 0 &&
                  std::fabs(e.y - 0.0227f) <= 0.0023f,  // 10 %
              "case length (Y) = %.4f m (target 0.0227 +-10%%)", e.y);
        check(std::fabs(e.x - 0.0120f) <= 0.0012f && std::fabs(e.z - 0.0120f) <= 0.0012f,
              "case diameter (X,Z) = %.4f, %.4f m (target 0.0120 +-10%%)", e.x, e.z);
        check(e.y > e.x && e.y > e.z, "axis along +Y (Y extent %.4f > X %.4f, Z %.4f)", e.y, e.x, e.z);
        check(std::fabs(c.x) < 1e-4f && std::fabs(c.z) < 1e-4f, "case centred on its axis "
              "(centre x = %.5f, z = %.5f)", c.x, c.z);
        check(s.degenerate == 0 && s.nonFinite == 0 && s.volume > 0.0,
              "case: closed, outward wound, no degenerate faces (volume %.2e m^3)", s.volume);
        const Mesh& m = scene.mesh(caseMesh);
        check(!m.ranges.empty() && m.ranges[0].material == mats.brass, "case material = brass (%u)",
              m.ranges.empty() ? 0u : m.ranges[0].material);
    }

    // ---------------------------------------------------------------- (i)
    std::printf("\n--- (i) 200 glass shards, fixed seed ---\n");
    {
        uint32_t bad = 0;
        uint32_t nan = 0;
        uint32_t nonDet = 0;
        uint32_t badDiag = 0;
        uint32_t tooManyTris = 0;
        uint32_t notThin = 0;
        double diagMin = 1e30, diagMax = 0.0;
        const float size = 0.035f;
        const float thickness = 0.0045f;
        for (int i = 0; i < 200; ++i) {
            const uint64_t seed = 0xC0FFEEull + static_cast<uint64_t>(i) * 7919ull;
            Rng rng(seed);
            const uint32_t mesh = buildGlassShard(scene, mats, rng, size, thickness);
            const MeshStats s = analyse(scene, mesh);
            const double d = diagonal(s.bounds);
            diagMin = std::min(diagMin, d);
            diagMax = std::max(diagMax, d);
            if (s.degenerate != 0) ++bad;
            if (s.nonFinite != 0) ++nan;
            if (std::fabs(d - size) > 0.40 * size) ++badDiag;
            if (s.triangles > 60) ++tooManyTris;
            const Vec3 e = s.bounds.mx - s.bounds.mn;
            if (!(e.z < e.x && e.z < e.y)) ++notThin;

            // Determinism: the same rng state must produce bit identical geometry.
            Rng again(seed);
            const uint32_t mesh2 = buildGlassShard(scene, mats, again, size, thickness);
            const Mesh& a = scene.mesh(mesh);
            const Mesh& b = scene.mesh(mesh2);
            const std::vector<Vertex>& V = scene.vertices();
            const std::vector<uint32_t>& II = scene.indices();
            bool same = a.ranges.size() == b.ranges.size() &&
                        a.ranges[0].indexCount == b.ranges[0].indexCount;
            if (same) {
                for (uint32_t k = 0; k < a.ranges[0].indexCount && same; ++k) {
                    const uint32_t ia = resolveVertex(a.ranges[0], II[a.ranges[0].firstIndex + k],
                                                      V.size(), nullptr);
                    const uint32_t ib = resolveVertex(b.ranges[0], II[b.ranges[0].firstIndex + k],
                                                      V.size(), nullptr);
                    const Vertex& va = V[ia];
                    const Vertex& vb = V[ib];
                    same = std::memcmp(&va.position, &vb.position, sizeof(Vec3)) == 0 &&
                           std::memcmp(&va.normal, &vb.normal, sizeof(Vec3)) == 0 &&
                           std::memcmp(&va.uv, &vb.uv, sizeof(Vec2)) == 0;
                }
            }
            if (!same) ++nonDet;
        }
        std::printf("  shard diagonal: min %.4f, max %.4f (requested %.4f)\n", diagMin, diagMax, size);
        check(bad == 0, "all 200 shards non-degenerate (degenerate: %u)", bad);
        check(nan == 0, "no NaN/Inf vertices (bad: %u)", nan);
        check(badDiag == 0, "all diagonals within 40%% of size (bad: %u)", badDiag);
        check(nonDet == 0, "same seed -> identical vertices (mismatches: %u)", nonDet);
        check(tooManyTris == 0, "every shard <= 60 triangles (bad: %u)", tooManyTris);
        check(notThin == 0, "thickness is the thin (Z) dimension (bad: %u)", notThin);
    }

    // ---------------------------------------------------------------- (j)
    std::printf("\n--- (j) polygon budget ---\n");
    check(totalTris >= 8000 && totalTris <= 40000,
          "pistol triangle count = %u (required 8000..40000)", totalTris);

    std::printf("\n=== %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
