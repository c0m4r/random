// room2 - procedural mesh construction.
//
// Every drawable in the game is generated here at runtime, so this file is the
// single source of truth for the project's geometry conventions:
//
//   * right-handed, Y-up world.  A triangle is wound counter-clockwise when the
//     surface is seen from *outside*, which is what
//     VK_FRONT_FACE_COUNTER_CLOCKWISE plus back-face culling expects, and every
//     vertex normal points away from the solid (makeBoxInverted points into the
//     room) - so a face is visible exactly when its normal faces the camera.
//   * UVs are "world-ish" wherever a tiling PBR material is expected: on a box
//     face of edge length L the texture repeats L * uvScale times, which keeps
//     texel density constant regardless of how a piece is scaled.  Round
//     primitives use the conventional u = around, v = along parametrisation.
//   * every builder ends in finalize(), which computes `bounds`, installs one
//     submesh covering every index exactly once and derives tangents from the
//     UVs (needed by the normal-mapped PBR pass).
//   * degenerate input (zero/negative size, < 3 segments, empty profiles) is
//     clamped to a minimum instead of producing NaN, Inf or an endless loop.
//   * output is a pure function of the arguments: no RNG, no time, no globals.
//
// All loop indices that become vertex indices are uint32_t; the helpers below
// keep that explicit because a large lathe/sphere can exceed 65535 vertices.

#include "procgen/mesh.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <numeric>
#include <vector>

namespace room2::procgen {
namespace {

// Smallest magnitude a primitive dimension is clamped to.  Keeps divisions and
// normalisations well conditioned for degenerate call sites.
constexpr float kMinSize = 1e-4f;

// Absolute position tolerance (scaled by mesh extent) for coincident-vertex
// clustering inside computeNormals()/weld()/fixSeamNormals().
constexpr float kPosTol = 1e-5f;

// Profile points closer than this to the Y axis collapse into a single pole.
constexpr float kPoleEps = 1e-5f;

// Dihedral angle above which a lathe profile corner becomes a hard crease.
constexpr float kLatheCreaseDeg = 40.0f;

inline bool finite(float v) { return std::isfinite(v); }
inline bool finite(Vec2 v) { return finite(v.x) && finite(v.y); }
inline bool finite(Vec3 v) { return finite(v.x) && finite(v.y) && finite(v.z); }
inline bool finite(Vec4 v) { return finite(v.x) && finite(v.y) && finite(v.z) && finite(v.w); }

// Clamps a primitive dimension to a positive magnitude (NaN falls back to the minimum).
inline float safeSize(float v) {
    if (!finite(v)) return kMinSize;
    const float a = std::fabs(v);
    return a < kMinSize ? kMinSize : a;
}
// Clamps a segment/ring/subdivision count to at least `lo`.
inline int safeCount(int v, int lo) { return v < lo ? lo : v; }
// Guard for caller supplied scale factors: only NaN/Inf are rejected, negative
// scales are legal (they mirror the tiling).
inline float safeScale(float s) { return finite(s) ? s : 1.0f; }
inline Vec2 safeScale(Vec2 s) { return Vec2(safeScale(s.x), safeScale(s.y)); }

inline Vertex mkVertex(Vec3 p, Vec3 n, Vec2 uv) {
    Vertex v;
    v.position = p;
    v.normal = n;
    v.tangent = Vec4(0.0f, 0.0f, 0.0f, 1.0f);
    v.uv = uv;
    v.uv1 = uv;
    return v;
}

// math.hpp only provides lengthSq() for Vec3; the 2D profile math needs its own.
inline float lenSq2(Vec2 v) { return dot(v, v); }

inline void tri(MeshData& m, uint32_t a, uint32_t b, uint32_t c) {
    m.indices.push_back(a);
    m.indices.push_back(b);
    m.indices.push_back(c);
}

// Every builder funnels through here: bounds + a single submesh spanning the
// whole index buffer + per-vertex tangents derived from the UVs.
void finalize(MeshData& m, uint32_t material, const char* name = nullptr) {
    m.computeBounds();
    m.submeshes.clear();
    if (!m.indices.empty()) {
        SubMesh sm;
        sm.indexOffset = 0;
        sm.indexCount = m.indexCount();
        sm.material = material;
        sm.bounds = m.bounds;
        sm.name = name != nullptr ? name : "";
        m.submeshes.push_back(sm);
    }
    m.computeTangents();
}

// Characteristic size of a mesh, used to scale position tolerances so that a
// 3 cm drinking glass and a 6 m room both cluster sensibly.
inline float meshScale(const std::vector<Vertex>& verts) {
    Aabb b;
    for (const Vertex& v : verts) {
        if (finite(v.position)) b.expand(v.position);
    }
    if (!b.valid()) return 1.0f;
    const Vec3 e = b.mx - b.mn;
    return max2(1.0f, max2(e.x, max2(e.y, e.z)));
}

// Groups vertex indices by coincident position using a tolerance-sized spatial
// hash grid.  A sorted-run approach would break whenever unrelated points
// interleave in sort order (e.g. the two coincident equator rings of a capsule,
// which are separated by other ring vertices sharing the same x), leaving the
// duplicates in different clusters - and therefore with different normals.
std::vector<std::vector<uint32_t>> positionClusters(const std::vector<Vertex>& verts, float tol) {
    std::vector<std::vector<uint32_t>> out;
    const uint32_t n = static_cast<uint32_t>(verts.size());
    if (n == 0) return out;
    const float cell = max2(tol, 1e-9f);
    const float inv = 1.0f / cell;
    using Cell = std::array<int64_t, 3>;
    auto cellOf = [inv](Vec3 p) {
        return Cell{static_cast<int64_t>(std::floor(p.x * inv)),
                    static_cast<int64_t>(std::floor(p.y * inv)),
                    static_cast<int64_t>(std::floor(p.z * inv))};
    };
    std::map<Cell, std::vector<uint32_t>> grid;  // cell -> cluster ids touching it
    std::vector<uint32_t> rep;                   // cluster -> representative vertex
    const float tol2 = tol * tol;
    for (uint32_t i = 0; i < n; ++i) {
        const Vec3 p = verts[i].position;
        if (!finite(p)) {  // never NaN-poison a cluster
            out.push_back({i});
            rep.push_back(i);
            continue;
        }
        const Cell c = cellOf(p);
        int found = -1;
        for (int dz = -1; dz <= 1 && found < 0; ++dz) {
            for (int dy = -1; dy <= 1 && found < 0; ++dy) {
                for (int dx = -1; dx <= 1 && found < 0; ++dx) {
                    const Cell key{c[0] + dx, c[1] + dy, c[2] + dz};
                    const auto it = grid.find(key);
                    if (it == grid.end()) continue;
                    for (uint32_t cl : it->second) {
                        if (lengthSq(p - verts[rep[cl]].position) <= tol2) {
                            found = static_cast<int>(cl);
                            break;
                        }
                    }
                }
            }
        }
        if (found < 0) {
            found = static_cast<int>(out.size());
            out.push_back({});
            rep.push_back(i);
        }
        out[static_cast<size_t>(found)].push_back(i);
        grid[c].push_back(static_cast<uint32_t>(found));
    }
    return out;
}

// Smallest non-zero axis-aligned helper vector, used as a tangent fallback.
inline Vec3 perpendicularTo(Vec3 n) {
    const Vec3 axis = std::fabs(n.x) < 0.9f ? Vec3(1, 0, 0) : Vec3(0, 1, 0);
    Vec3 t = cross(axis, n);
    if (lengthSq(t) < 1e-12f) t = cross(Vec3(0, 0, 1), n);
    return normalize(t);
}

}  // namespace

// ===========================================================================
// MeshData
// ===========================================================================

void MeshData::computeBounds() { bounds = procgen::computeBounds(vertices); }

void MeshData::computeNormals(float smoothAngleDeg) {
    const uint32_t nv = vertexCount();
    if (nv == 0 || indices.size() < 3) return;
    const uint32_t nf = static_cast<uint32_t>(indices.size() / 3);

    // Unnormalised face normals: |cross| == 2 * area, so using them directly as
    // accumulation weights gives exactly the requested area weighting.
    std::vector<Vec3> fn(nf);
    for (uint32_t f = 0; f < nf; ++f) {
        const Vec3 p0 = vertices[indices[f * 3 + 0]].position;
        const Vec3 p1 = vertices[indices[f * 3 + 1]].position;
        const Vec3 p2 = vertices[indices[f * 3 + 2]].position;
        fn[f] = cross(p1 - p0, p2 - p0);
    }

    // Corner lists per vertex (CSR of index-buffer entries).
    std::vector<uint32_t> head(nv, UINT32_MAX);
    std::vector<uint32_t> next(indices.size(), UINT32_MAX);
    for (uint32_t k = 0; k < indices.size(); ++k) {
        const uint32_t v = indices[k];
        if (v >= nv) continue;  // malformed input: ignore rather than read OOB
        next[k] = head[v];
        head[v] = k;
    }

    const float cosThr = std::cos(clamp(smoothAngleDeg, 0.0f, 180.0f) * DEG2RAD);
    std::vector<Vec3> bestN(nv, Vec3(0, 0, 0));
    std::vector<float> bestW(nv, -1.0f);

    std::vector<std::vector<uint32_t>> clusters =
        positionClusters(vertices, kPosTol * meshScale(vertices));

    // Per cluster: build the corner list, union corners whose face normals are
    // within the smoothing angle, then area-weight-average each group.  Splitting
    // by face-normal angle is what keeps a 90 degree crease hard while a curved
    // surface (including duplicated UV-seam/pole vertices, which sit at the same
    // position) averages smoothly.
    struct Corner {
        uint32_t face;
        uint32_t vertex;
    };
    std::vector<Corner> corners;
    std::vector<int> parent;
    std::vector<int> rootOf;
    std::vector<Vec3> groupSum;
    std::vector<float> groupWeight;
    for (const std::vector<uint32_t>& cl : clusters) {
        corners.clear();
        for (uint32_t v : cl) {
            for (uint32_t k = head[v]; k != UINT32_MAX; k = next[k]) {
                corners.push_back(Corner{k / 3, v});
            }
        }
        const int nc = static_cast<int>(corners.size());
        if (nc == 0) continue;
        parent.resize(nc);
        for (int i = 0; i < nc; ++i) parent[i] = i;
        auto find = [&parent](int i) {
            while (parent[i] != i) {
                parent[i] = parent[parent[i]];
                i = parent[i];
            }
            return i;
        };
        for (int i = 0; i < nc; ++i) {
            const Vec3 a = fn[corners[i].face];
            if (lengthSq(a) < 1e-24f) continue;
            for (int j = i + 1; j < nc; ++j) {
                const Vec3 b = fn[corners[j].face];
                if (lengthSq(b) < 1e-24f) continue;
                if (dot(normalize(a), normalize(b)) >= cosThr) {
                    const int ra = find(i), rb = find(j);
                    if (ra != rb) parent[rb] = ra;
                }
            }
        }
        // rootOf maps a union-find root to its group slot (indexed by root!).
        rootOf.assign(nc, -1);
        groupSum.clear();
        groupWeight.clear();
        for (int i = 0; i < nc; ++i) {
            const int r = find(i);
            int g = rootOf[r];
            if (g < 0) {
                g = static_cast<int>(groupSum.size());
                rootOf[r] = g;
                groupSum.push_back(Vec3(0, 0, 0));
                groupWeight.push_back(0.0f);
            }
            const Vec3 f = fn[corners[i].face];
            groupSum[g] += f;
            groupWeight[g] += length(f);
        }
        for (int i = 0; i < nc; ++i) {
            const int g = rootOf[find(i)];
            if (g < 0) continue;
            const uint32_t v = corners[i].vertex;
            // A vertex index used by several incompatible groups (possible after a
            // weld) keeps the heaviest one - deterministic and crease preserving.
            if (groupWeight[g] > bestW[v]) {
                bestW[v] = groupWeight[g];
                bestN[v] = groupSum[g];
            }
        }
    }

    // A vertex no triangle references (a duplicated pole column always has one
    // spare copy) inherits the normals of its coincident siblings.
    for (const std::vector<uint32_t>& cl : clusters) {
        Vec3 sum(0, 0, 0);
        int have = 0;
        for (uint32_t v : cl) {
            if (bestW[v] >= 0.0f) { sum += bestN[v]; ++have; }
        }
        if (have == 0 || lengthSq(normalize(sum)) < 0.5f) continue;
        for (uint32_t v : cl) {
            if (bestW[v] < 0.0f) { bestN[v] = sum; bestW[v] = 0.0f; }
        }
    }

    for (uint32_t v = 0; v < nv; ++v) {
        const Vec3 n = normalize(bestN[v]);
        if (lengthSq(n) > 0.5f) {
            vertices[v].normal = n;
            continue;
        }
        // Untouched (unreferenced) or perfectly cancelled vertex: keep a usable normal.
        const Vec3 old = vertices[v].normal;
        vertices[v].normal = (finite(old) && lengthSq(old) > 1e-12f) ? normalize(old) : Vec3(0, 1, 0);
    }
}

void MeshData::computeTangents() {
    const uint32_t nv = vertexCount();
    if (nv == 0) return;
    std::vector<Vec3> tan(nv, Vec3(0, 0, 0));
    std::vector<Vec3> bit(nv, Vec3(0, 0, 0));
    const uint32_t nf = static_cast<uint32_t>(indices.size() / 3);
    for (uint32_t f = 0; f < nf; ++f) {
        const uint32_t i0 = indices[f * 3 + 0], i1 = indices[f * 3 + 1], i2 = indices[f * 3 + 2];
        if (i0 >= nv || i1 >= nv || i2 >= nv) continue;
        const Vec3 p0 = vertices[i0].position, p1 = vertices[i1].position, p2 = vertices[i2].position;
        const Vec2 w0 = vertices[i0].uv, w1 = vertices[i1].uv, w2 = vertices[i2].uv;
        const Vec3 e1 = p1 - p0, e2 = p2 - p0;
        const Vec2 d1 = w1 - w0, d2 = w2 - w0;
        const float det = d1.x * d2.y - d2.x * d1.y;
        if (!finite(det) || std::fabs(det) < 1e-12f) continue;  // degenerate UVs: no frame
        const float inv = 1.0f / det;
        const Vec3 t = (e1 * d2.y - e2 * d1.y) * inv;
        const Vec3 b = (e2 * d1.x - e1 * d2.x) * inv;
        // Weight by triangle area so big faces dominate the average.
        const float w = length(cross(e1, e2));
        if (!finite(w)) continue;
        tan[i0] += t * w; tan[i1] += t * w; tan[i2] += t * w;
        bit[i0] += b * w; bit[i1] += b * w; bit[i2] += b * w;
    }
    for (uint32_t i = 0; i < nv; ++i) {
        Vertex& v = vertices[i];
        Vec3 n = finite(v.normal) ? normalize(v.normal) : Vec3(0, 1, 0);
        if (lengthSq(n) < 0.5f) n = Vec3(0, 1, 0);
        v.normal = n;
        Vec3 t = tan[i] - n * dot(n, tan[i]);          // Gram-Schmidt against the normal
        float w = 1.0f;
        if (lengthSq(t) < 1e-16f) {
            // No usable UV derivative (pole, cap centre, flat-coloured mesh):
            // fall back to the bitangent, then to any perpendicular axis.
            Vec3 b = bit[i] - n * dot(n, bit[i]);
            t = lengthSq(b) > 1e-16f ? b : perpendicularTo(n);
            w = 1.0f;
        } else {
            Vec3 b = bit[i] - n * dot(n, bit[i]);
            if (lengthSq(b) > 1e-16f) {
                w = dot(cross(n, normalize(t)), normalize(b)) < 0.0f ? -1.0f : 1.0f;
            }
        }
        t = normalize(t);
        if (lengthSq(t) < 0.5f) t = perpendicularTo(n);
        v.tangent = Vec4(t, w);
    }
}

void MeshData::transform(const Mat4& m) {
    const Mat3 upper = mat3FromMat4Upper(m);
    const Mat3 nrm = transpose(inverse(upper));
    const bool mirrored = determinant(upper) < 0.0f;
    for (Vertex& v : vertices) {
        v.position = transformPoint(m, v.position);
        const Vec3 n = nrm * v.normal;
        if (finite(n) && lengthSq(n) > 1e-16f) v.normal = normalize(n);
        const Vec3 t = upper * v.tangent.xyz();
        // A mirror flips the geometric handedness, so the bitangent sign does too;
        // the tangent itself still follows dP/du.
        float w = v.tangent.w;
        if (mirrored) w = -w;
        Vec3 tt = t - v.normal * dot(v.normal, t);
        v.tangent = Vec4(lengthSq(tt) > 1e-16f ? normalize(tt) : v.tangent.xyz(), w);
    }
    if (mirrored) {
        // Mirroring inverts the apparent winding: restore "CCW seen from outside".
        for (size_t i = 0; i + 2 < indices.size(); i += 3) std::swap(indices[i + 1], indices[i + 2]);
    }
    computeBounds();
}

void MeshData::append(const MeshData& other, uint32_t materialOffset) {
    if (&other == this) {  // self-append: work on a stable copy
        const MeshData copy = other;
        append(copy, materialOffset);
        return;
    }
    const uint32_t vbase = vertexCount();
    const uint32_t ibase = indexCount();
    if (!other.vertices.empty()) {
        vertices.insert(vertices.end(), other.vertices.begin(), other.vertices.end());
    }
    indices.reserve(indices.size() + other.indices.size());
    for (uint32_t i : other.indices) indices.push_back(i + vbase);
    if (!other.submeshes.empty()) {
        for (const SubMesh& s : other.submeshes) {
            SubMesh c = s;
            c.indexOffset = s.indexOffset + ibase;
            c.material = s.material + materialOffset;
            submeshes.push_back(c);
        }
    } else if (!other.indices.empty()) {
        // Keep the "submeshes cover every index" invariant even for a hand-built mesh.
        SubMesh c;
        c.indexOffset = ibase;
        c.indexCount = static_cast<uint32_t>(other.indices.size());
        c.material = materialOffset;
        c.bounds = other.bounds;
        submeshes.push_back(c);
    }
    if (other.bounds.valid()) bounds.expand(other.bounds);
    if (!bounds.valid() && !vertices.empty()) computeBounds();
}

void MeshData::mergeSubMeshes(uint32_t material, const char* name) {
    submeshes.clear();
    if (indices.empty()) return;
    SubMesh sm;
    sm.indexOffset = 0;
    sm.indexCount = indexCount();
    sm.material = material;
    Aabb b;
    for (uint32_t i : indices) {
        if (i < vertices.size()) b.expand(vertices[i].position);
    }
    sm.bounds = b.valid() ? b : bounds;
    sm.name = name != nullptr ? name : "";
    submeshes.push_back(sm);
}

void MeshData::addSubMesh(uint32_t firstIndex, uint32_t material, const char* name) {
    if (firstIndex >= indices.size()) return;  // nothing left to close
    SubMesh sm;
    sm.indexOffset = firstIndex;
    sm.indexCount = indexCount() - firstIndex;
    sm.material = material;
    Aabb b;
    for (uint32_t k = firstIndex; k < indices.size(); ++k) {
        const uint32_t i = indices[k];
        if (i < vertices.size()) b.expand(vertices[i].position);
    }
    sm.bounds = b.valid() ? b : bounds;
    sm.name = name != nullptr ? name : "";
    submeshes.push_back(sm);
}

void MeshData::flipWinding() {
    for (size_t i = 0; i + 2 < indices.size(); i += 3) std::swap(indices[i + 1], indices[i + 2]);
    for (Vertex& v : vertices) {
        v.normal = -v.normal;
        // Keep tangent.xyz == dP/du (a UV derivative does not depend on which side
        // counts as front) and flip the handedness so bitangent == w * cross(N, T)
        // still matches dP/dv.  This is exactly what computeTangents() derives for
        // the flipped normal, so the two stay interchangeable.
        v.tangent = Vec4(v.tangent.xyz(), -v.tangent.w);
    }
}

void MeshData::weld(float tolerance) {
    const uint32_t n = vertexCount();
    if (n < 2) return;
    const float tol = max2(std::fabs(finite(tolerance) ? tolerance : kPosTol), 1e-7f);
    std::vector<uint32_t> order(n);
    std::iota(order.begin(), order.end(), 0u);
    std::sort(order.begin(), order.end(), [this](uint32_t a, uint32_t b) {
        const Vertex& va = vertices[a];
        const Vertex& vb = vertices[b];
        const float ka[8] = {va.position.x, va.position.y, va.position.z, va.normal.x,
                             va.normal.y,   va.normal.z,   va.uv.x,     va.uv.y};
        const float kb[8] = {vb.position.x, vb.position.y, vb.position.z, vb.normal.x,
                             vb.normal.y,   vb.normal.z,   vb.uv.x,     vb.uv.y};
        for (int i = 0; i < 8; ++i) {
            if (ka[i] != kb[i]) return ka[i] < kb[i];
        }
        return a < b;
    });
    std::vector<uint32_t> remap(n, 0);
    std::vector<Vertex> out;
    out.reserve(n);
    for (uint32_t oi = 0; oi < n; ++oi) {
        const uint32_t i = order[oi];
        bool merged = false;
        if (!out.empty()) {
            const Vertex& r = out.back();
            const Vertex& v = vertices[i];
            merged = std::fabs(v.position.x - r.position.x) <= tol &&
                     std::fabs(v.position.y - r.position.y) <= tol &&
                     std::fabs(v.position.z - r.position.z) <= tol &&
                     std::fabs(v.normal.x - r.normal.x) <= tol &&
                     std::fabs(v.normal.y - r.normal.y) <= tol &&
                     std::fabs(v.normal.z - r.normal.z) <= tol &&
                     std::fabs(v.uv.x - r.uv.x) <= tol && std::fabs(v.uv.y - r.uv.y) <= tol;
        }
        if (merged) {
            remap[i] = static_cast<uint32_t>(out.size() - 1);
        } else {
            remap[i] = static_cast<uint32_t>(out.size());
            out.push_back(vertices[i]);
        }
    }
    for (uint32_t& idx : indices) {
        if (idx < n) idx = remap[idx];
    }
    vertices.swap(out);
    computeBounds();
}

// ===========================================================================
// primitives
// ===========================================================================

MeshData makeBox(Vec3 halfExtents, Vec2 uvScale, uint32_t material) {
    const float hx = safeSize(halfExtents.x), hy = safeSize(halfExtents.y);
    const float hz = safeSize(halfExtents.z);
    const Vec3 h(hx, hy, hz);
    const Vec2 s = safeScale(uvScale);
    // n is the face normal, cross(u, v) == n, so the quad ordered
    // (-u,-v) (+u,-v) (+u,+v) (-u,+v) is CCW as seen from +n.
    struct Face { Vec3 n, u, v; };
    static const Face faces[6] = {
        {{ 1, 0, 0}, { 0, 0,-1}, { 0, 1, 0}},
        {{-1, 0, 0}, { 0, 0, 1}, { 0, 1, 0}},
        {{ 0, 1, 0}, { 1, 0, 0}, { 0, 0,-1}},
        {{ 0,-1, 0}, { 1, 0, 0}, { 0, 0, 1}},
        {{ 0, 0, 1}, { 1, 0, 0}, { 0, 1, 0}},
        {{ 0, 0,-1}, {-1, 0, 0}, { 0, 1, 0}},
    };
    MeshData m;
    m.vertices.reserve(24);
    m.indices.reserve(36);
    for (const Face& f : faces) {
        // n is a signed axis, so the face centre is |dot(n, h)| along n.
        const Vec3 c = f.n * std::fabs(dot(f.n, h));
        const float hu = std::fabs(dot(f.u, h));
        const float hv = std::fabs(dot(f.v, h));
        const uint32_t base = m.vertexCount();
        const Vec3 p0 = c - f.u * hu - f.v * hv;
        const Vec3 p1 = c + f.u * hu - f.v * hv;
        const Vec3 p2 = c + f.u * hu + f.v * hv;
        const Vec3 p3 = c - f.u * hu + f.v * hv;
        // World-ish UVs: a face of size 2*hu maps to 2*hu*uvScale texture repeats.
        for (Vec3 p : {p0, p1, p2, p3}) {
            m.vertices.push_back(mkVertex(p, f.n, Vec2(dot(p, f.u), dot(p, f.v)) * s));
        }
        tri(m, base + 0, base + 1, base + 2);
        tri(m, base + 0, base + 2, base + 3);
    }
    finalize(m, material, "box");
    return m;
}

MeshData makeBoxInverted(Vec3 halfExtents, float uvScale, uint32_t material) {
    const float hx = safeSize(halfExtents.x), hy = safeSize(halfExtents.y);
    const float hz = safeSize(halfExtents.z);
    const Vec3 h(hx, hy, hz);
    const float s = safeScale(uvScale);
    // Room shell: n points *into* the room and cross(u, v) == n.  u/v are chosen as
    // (signed) world axes, so the UV of any wall is just two world coordinates times
    // uvScale - tiling is therefore continuous across wall/floor/ceiling seams.
    struct Face { Vec3 n, u, v; };
    static const Face faces[6] = {
        {{-1, 0, 0}, { 0, 0, 1}, { 0, 1, 0}},   // +X wall
        {{ 1, 0, 0}, { 0, 0, 1}, { 0,-1, 0}},   // -X wall
        {{ 0, 1, 0}, { 1, 0, 0}, { 0, 0,-1}},   // floor
        {{ 0,-1, 0}, { 1, 0, 0}, { 0, 0, 1}},   // ceiling
        {{ 0, 0,-1}, { 1, 0, 0}, { 0,-1, 0}},   // +Z wall
        {{ 0, 0, 1}, { 1, 0, 0}, { 0, 1, 0}},   // -Z wall
    };
    MeshData m;
    m.vertices.reserve(24);
    m.indices.reserve(36);
    for (const Face& f : faces) {
        // n points into the room, so the wall sits at -|dot(n, h)| along n.
        const Vec3 c = f.n * (-std::fabs(dot(f.n, h)));
        const float hu = std::fabs(dot(f.u, h));
        const float hv = std::fabs(dot(f.v, h));
        const uint32_t base = m.vertexCount();
        const Vec3 p0 = c - f.u * hu - f.v * hv;
        const Vec3 p1 = c + f.u * hu - f.v * hv;
        const Vec3 p2 = c + f.u * hu + f.v * hv;
        const Vec3 p3 = c - f.u * hu + f.v * hv;
        for (Vec3 p : {p0, p1, p2, p3}) {
            m.vertices.push_back(mkVertex(p, f.n, Vec2(dot(p, f.u), dot(p, f.v)) * s));
        }
        tri(m, base + 0, base + 1, base + 2);
        tri(m, base + 0, base + 2, base + 3);
    }
    finalize(m, material, "room");
    return m;
}

MeshData makePlane(float sizeX, float sizeZ, int subdivX, int subdivZ, uint32_t material) {
    const float sx = safeSize(sizeX), sz = safeSize(sizeZ);
    const int nx = safeCount(subdivX, 1), nz = safeCount(subdivZ, 1);
    const float hx = sx * 0.5f, hz = sz * 0.5f;
    MeshData m;
    m.vertices.reserve(static_cast<size_t>(nx + 1) * static_cast<size_t>(nz + 1));
    for (int j = 0; j <= nz; ++j) {
        const float z = -hz + sz * (static_cast<float>(j) / static_cast<float>(nz));
        for (int i = 0; i <= nx; ++i) {
            const float x = -hx + sx * (static_cast<float>(i) / static_cast<float>(nx));
            // uv = (x, -z) matches the +Y face of boxProjectUv (unmirrored from above).
            m.vertices.push_back(mkVertex(Vec3(x, 0.0f, z), Vec3(0, 1, 0), Vec2(x, -z)));
        }
    }
    for (int j = 0; j < nz; ++j) {
        for (int i = 0; i < nx; ++i) {
            const uint32_t a = static_cast<uint32_t>(j * (nx + 1) + i);
            const uint32_t b = a + 1;
            const uint32_t d = a + static_cast<uint32_t>(nx + 1);
            const uint32_t c = d + 1;
            tri(m, a, d, c);
            tri(m, a, c, b);
        }
    }
    finalize(m, material, "plane");
    return m;
}

MeshData makeUvSphere(float radius, int segments, int rings, uint32_t material) {
    const float r = safeSize(radius);
    const int S = safeCount(segments, 3), R = safeCount(rings, 2);
    MeshData m;
    m.vertices.reserve(static_cast<size_t>(R + 1) * static_cast<size_t>(S + 1));
    for (int i = 0; i <= R; ++i) {
        const float v = static_cast<float>(i) / static_cast<float>(R);
        const float theta = v * PI;  // 0 at the +Y pole
        const float st = std::sin(theta), ct = std::cos(theta);
        for (int j = 0; j <= S; ++j) {
            const float u = static_cast<float>(j) / static_cast<float>(S);
            const float phi = u * TWO_PI;
            const Vec3 p(r * st * std::cos(phi), r * ct, r * st * std::sin(phi));
            // Poles and the duplicated UV seam get identical normals, so the
            // silhouette stays smooth after an index weld.
            const Vec3 n = normalize(p);
            const Vec3 pn = lengthSq(n) > 0.5f ? n : Vec3(0, ct >= 0 ? 1.0f : -1.0f, 0);
            m.vertices.push_back(mkVertex(p, pn, Vec2(u, v)));
        }
    }
    auto idx = [S](int i, int j) { return static_cast<uint32_t>(i * (S + 1) + j); };
    for (int i = 0; i < R; ++i) {
        for (int j = 0; j < S; ++j) {
            const uint32_t A = idx(i, j), B = idx(i + 1, j), C = idx(i + 1, j + 1), D = idx(i, j + 1);
            if (i == 0) {
                tri(m, D, C, B);  // A and D coincide at the top pole
            } else if (i + 1 == R) {
                tri(m, A, D, B);  // B and C coincide at the bottom pole
            } else {
                tri(m, A, D, C);
                tri(m, A, C, B);
            }
        }
    }
    finalize(m, material, "uvsphere");
    return m;
}

MeshData makeCylinder(float radius, float height, int segments, bool capped, uint32_t material) {
    const float r = safeSize(radius), h = safeSize(height);
    const int S = safeCount(segments, 3);
    const float hy = h * 0.5f;
    MeshData m;
    // ---- side wall: two rings, seam column duplicated for UV continuity.
    for (int ring = 0; ring < 2; ++ring) {
        const float y = ring == 0 ? -hy : hy;
        for (int j = 0; j <= S; ++j) {
            const float u = static_cast<float>(j) / static_cast<float>(S);
            const float phi = u * TWO_PI;
            const float c = std::cos(phi), s = std::sin(phi);
            const float v = ring == 0 ? 0.0f : 1.0f;
            m.vertices.push_back(mkVertex(Vec3(r * c, y, r * s), Vec3(c, 0, s), Vec2(u, v)));
        }
    }
    for (int j = 0; j < S; ++j) {
        const uint32_t b0 = static_cast<uint32_t>(j), b1 = b0 + 1;
        const uint32_t t0 = static_cast<uint32_t>(S + 1 + j), t1 = t0 + 1;
        tri(m, b0, t0, t1);
        tri(m, b0, t1, b1);
    }
    // ---- caps: their own vertices with axis-aligned normals (never averaged rims).
    if (capped) {
        for (int cap = 0; cap < 2; ++cap) {
            const bool top = cap == 0;
            const float y = top ? hy : -hy;
            const Vec3 n(0, top ? 1.0f : -1.0f, 0);
            const uint32_t centre = m.vertexCount();
            m.vertices.push_back(mkVertex(Vec3(0, y, 0), n, Vec2(0.5f, 0.5f)));
            const uint32_t rim = m.vertexCount();
            for (int j = 0; j <= S; ++j) {
                const float phi = static_cast<float>(j) / static_cast<float>(S) * TWO_PI;
                const float c = std::cos(phi), s = std::sin(phi);
                const Vec2 uv(0.5f + 0.5f * c, 0.5f + (top ? -0.5f : 0.5f) * s);
                m.vertices.push_back(mkVertex(Vec3(r * c, y, r * s), n, uv));
            }
            for (int j = 0; j < S; ++j) {
                const uint32_t a = rim + static_cast<uint32_t>(j), b = a + 1;
                if (top) tri(m, centre, b, a);
                else tri(m, centre, a, b);
            }
        }
    }
    finalize(m, material, "cylinder");
    return m;
}

MeshData makeCone(float radius, float height, int segments, uint32_t material) {
    const float r = safeSize(radius), h = safeSize(height);
    const int S = safeCount(segments, 3);
    const float hy = h * 0.5f;
    MeshData m;
    // ---- side: one triangle per segment.  The outward normal of a cone side is
    // normalize(h*cos, r, h*sin): constant along the slant, so apex vertices get
    // the exact cone normal instead of a degenerate pole average.
    const uint32_t base = m.vertexCount();
    for (int j = 0; j <= S; ++j) {
        const float u = static_cast<float>(j) / static_cast<float>(S);
        const float phi = u * TWO_PI;
        const float c = std::cos(phi), s = std::sin(phi);
        const Vec3 n = normalize(Vec3(h * c, r, h * s));
        m.vertices.push_back(mkVertex(Vec3(r * c, -hy, r * s), n, Vec2(u, 0.0f)));
    }
    for (int j = 0; j < S; ++j) {
        const float u = (static_cast<float>(j) + 0.5f) / static_cast<float>(S);
        const float phi = u * TWO_PI;
        const uint32_t apex = m.vertexCount();
        const Vec3 n = normalize(Vec3(h * std::cos(phi), r, h * std::sin(phi)));
        m.vertices.push_back(mkVertex(Vec3(0, hy, 0), n, Vec2(u, 1.0f)));
        tri(m, base + static_cast<uint32_t>(j), apex, base + static_cast<uint32_t>(j) + 1);
    }
    // ---- base cap (the solid is closed, which shadowing and SSAO expect).
    const uint32_t centre = m.vertexCount();
    m.vertices.push_back(mkVertex(Vec3(0, -hy, 0), Vec3(0, -1, 0), Vec2(0.5f, 0.5f)));
    const uint32_t rim = m.vertexCount();
    for (int j = 0; j <= S; ++j) {
        const float phi = static_cast<float>(j) / static_cast<float>(S) * TWO_PI;
        const float c = std::cos(phi), s = std::sin(phi);
        const Vec2 uv(0.5f + 0.5f * c, 0.5f + 0.5f * s);
        m.vertices.push_back(mkVertex(Vec3(r * c, -hy, r * s), Vec3(0, -1, 0), uv));
    }
    for (int j = 0; j < S; ++j) {
        tri(m, centre, rim + static_cast<uint32_t>(j), rim + static_cast<uint32_t>(j) + 1);
    }
    finalize(m, material, "cone");
    return m;
}

MeshData makeCapsule(float radius, float cylinderHeight, int segments, int rings, uint32_t material) {
    const float r = safeSize(radius);
    const float ch = finite(cylinderHeight) ? max2(std::fabs(cylinderHeight), 0.0f) : 0.0f;
    const int S = safeCount(segments, 3), R = safeCount(rings, 2);
    const float hy = ch * 0.5f;
    const float ymin = -hy - r, ymax = hy + r;
    const float total = ymax - ymin;  // > 0 because radius >= kMinSize
    MeshData m;
    auto vAt = [ymin, total](float y) { return (y - ymin) / total; };

    // ---- cylindrical body: outward radial normals, matching the hemisphere
    // normals at the equator so no shading seam appears there.  A zero-height
    // capsule is just a sphere, so the body is skipped entirely then (emitting it
    // would add zero-area triangles that break manifoldness).
    const bool hasBody = ch > 1e-6f;
    for (int ring = 0; hasBody && ring < 2; ++ring) {
        const float y = ring == 0 ? -hy : hy;
        for (int j = 0; j <= S; ++j) {
            const float u = static_cast<float>(j) / static_cast<float>(S);
            const float phi = u * TWO_PI;
            const float c = std::cos(phi), s = std::sin(phi);
            m.vertices.push_back(mkVertex(Vec3(r * c, y, r * s), Vec3(c, 0, s), Vec2(u, vAt(y))));
        }
    }
    for (int j = 0; hasBody && j < S; ++j) {
        const uint32_t b0 = static_cast<uint32_t>(j), b1 = b0 + 1;
        const uint32_t t0 = static_cast<uint32_t>(S + 1 + j), t1 = t0 + 1;
        tri(m, b0, t0, t1);
        tri(m, b0, t1, b1);
    }

    // ---- top hemisphere: ring 0 is the +Y pole, ring R the equator.
    for (int i = 0; i <= R; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(R);
        const float th = t * HALF_PI;
        const float st = std::sin(th), ct = std::cos(th);
        for (int j = 0; j <= S; ++j) {
            const float u = static_cast<float>(j) / static_cast<float>(S);
            const float phi = u * TWO_PI;
            const float c = std::cos(phi), s = std::sin(phi);
            const Vec3 p(r * st * c, hy + r * ct, r * st * s);
            m.vertices.push_back(mkVertex(p, Vec3(st * c, ct, st * s), Vec2(u, vAt(p.y))));
        }
    }
    uint32_t topBase = m.vertexCount() - static_cast<uint32_t>((R + 1) * (S + 1));
    for (int i = 0; i < R; ++i) {
        for (int j = 0; j < S; ++j) {
            const uint32_t A = topBase + static_cast<uint32_t>(i * (S + 1) + j);
            const uint32_t B = A + static_cast<uint32_t>(S + 1);
            const uint32_t C = B + 1, D = A + 1;
            if (i == 0) tri(m, D, C, B);
            else { tri(m, A, D, C); tri(m, A, C, B); }
        }
    }

    // ---- bottom hemisphere: ring 0 is the equator, ring R the -Y pole.
    const uint32_t botBase = m.vertexCount();
    for (int i = 0; i <= R; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(R);
        const float th = (1.0f - t) * HALF_PI;
        const float st = std::sin(th), ct = std::cos(th);
        for (int j = 0; j <= S; ++j) {
            const float u = static_cast<float>(j) / static_cast<float>(S);
            const float phi = u * TWO_PI;
            const float c = std::cos(phi), s = std::sin(phi);
            const Vec3 p(r * st * c, -hy - r * ct, r * st * s);
            m.vertices.push_back(mkVertex(p, Vec3(st * c, -ct, st * s), Vec2(u, vAt(p.y))));
        }
    }
    for (int i = 0; i < R; ++i) {
        for (int j = 0; j < S; ++j) {
            const uint32_t A = botBase + static_cast<uint32_t>(i * (S + 1) + j);
            const uint32_t B = A + static_cast<uint32_t>(S + 1);
            const uint32_t C = B + 1, D = A + 1;
            if (i + 1 == R) tri(m, A, D, B);
            else { tri(m, A, D, C); tri(m, A, C, B); }
        }
    }
    finalize(m, material, "capsule");
    return m;
}

MeshData makeTorus(float majorRadius, float minorRadius, int majorSegments, int minorSegments,
                   uint32_t material) {
    const float R = safeSize(majorRadius), r = safeSize(minorRadius);
    const int MS = safeCount(majorSegments, 3), ms = safeCount(minorSegments, 3);
    MeshData m;
    m.vertices.reserve(static_cast<size_t>(MS + 1) * static_cast<size_t>(ms + 1));
    for (int i = 0; i <= ms; ++i) {
        const float v = static_cast<float>(i) / static_cast<float>(ms);
        const float tv = v * TWO_PI;
        const float cv = std::cos(tv), sv = std::sin(tv);
        for (int j = 0; j <= MS; ++j) {
            const float u = static_cast<float>(j) / static_cast<float>(MS);
            const float tu = u * TWO_PI;
            const float cu = std::cos(tu), su = std::sin(tu);
            const Vec3 p((R + r * cv) * cu, r * sv, (R + r * cv) * su);
            m.vertices.push_back(mkVertex(p, Vec3(cv * cu, sv, cv * su), Vec2(u, v)));
        }
    }
    for (int i = 0; i < ms; ++i) {
        for (int j = 0; j < MS; ++j) {
            const uint32_t A = static_cast<uint32_t>(i * (MS + 1) + j);
            const uint32_t B = A + static_cast<uint32_t>(MS + 1);  // +v (around the tube)
            const uint32_t D = A + 1;                              // +u (around the ring)
            const uint32_t C = B + 1;
            tri(m, A, B, C);
            tri(m, A, C, D);
        }
    }
    finalize(m, material, "torus");
    return m;
}

MeshData makeLathe(const std::vector<Vec2>& profileIn, int segments, uint32_t material) {
    const int S = safeCount(segments, 3);
    MeshData m;

    // ---- sanitise the profile: finite, positive radii, no repeated points.  A
    // first point equal to the last one means a closed loop (solid cross section):
    // the duplicate is dropped and the final band wraps around to point 0.
    std::vector<Vec2> profile;
    profile.reserve(profileIn.size());
    for (Vec2 p : profileIn) {
        if (!finite(p)) continue;
        const Vec2 q(std::fabs(p.x) < kPoleEps ? 0.0f : std::fabs(p.x), p.y);
        if (!profile.empty() && length(q - profile.back()) < 1e-9f) continue;
        profile.push_back(q);
    }
    bool closed = false;
    if (profile.size() >= 3 && length(profile.front() - profile.back()) <= 1e-6f) {
        profile.pop_back();
        closed = true;
    }
    if (profile.size() < 2) {
        finalize(m, material, "lathe");  // empty, but consistently booked
        return m;
    }
    const int P = static_cast<int>(profile.size());
    const int bands = closed ? P : P - 1;
    auto isPole = [&profile](int i) { return profile[i].x <= kPoleEps; };
    // A band whose both ends sit on the axis is a zero-area sliver (the closing
    // edge of a solid profile); it must not influence crease detection.
    std::vector<char> axisBand(static_cast<size_t>(bands), 0);
    for (int b = 0; b < bands; ++b) {
        axisBand[static_cast<size_t>(b)] = (isPole(b) && isPole((b + 1) % P)) ? 1 : 0;
    }

    // Segment normals in the (radius, height) plane.  For a profile wound CCW
    // (interior of the solid on the left) n = (dy, -dr) is the outward direction,
    // and cross(dP/du, dP/dphi) then matches the triangle winding used below.
    std::vector<Vec2> segN(bands, Vec2(0, 0));
    for (int b = 0; b < bands; ++b) {
        const Vec2 a = profile[b], c = profile[(b + 1) % P];
        const Vec2 d = c - a;
        const Vec2 n(d.y, -d.x);
        segN[b] = length(n) > EPS ? normalize(n) : Vec2(0, 0);
    }
    // Any degenerate segment inherits its nearest valid neighbour.
    {
        Vec2 last(0, 0);
        for (int b = 0; b < bands; ++b) {
            if (lenSq2(segN[b]) < 0.5f) segN[b] = last;
            else last = segN[b];
        }
        if (lenSq2(last) > 0.5f) {
            for (int b = 0; b < bands; ++b) {
                if (lenSq2(segN[b]) < 0.5f) segN[b] = last;
            }
        }
        for (int b = 0; b < bands; ++b) {
            if (lenSq2(segN[b]) < 0.5f) segN[b] = Vec2(1, 0);
        }
    }

    // Per profile point: one normal when the surface is smooth through it, two
    // when the profile turns sharper than the crease angle (a crisp glass rim).
    const float cosCrease = std::cos(kLatheCreaseDeg * DEG2RAD);
    std::vector<std::vector<Vec2>> pointN(P);
    std::vector<int> prevSlot(P, 0), nextSlot(P, 0);
    for (int i = 0; i < P; ++i) {
        const size_t prevBand = static_cast<size_t>((i - 1 + bands) % bands);
        const bool hasPrev = (closed || i > 0) && !axisBand[prevBand];
        const bool hasNext = (closed || i < P - 1) && !axisBand[static_cast<size_t>(i % bands)];
        const Vec2 nPrev = hasPrev ? segN[(i - 1 + bands) % bands] : Vec2(0, 0);
        const Vec2 nNext = hasNext ? segN[i % bands] : Vec2(0, 0);
        if (hasPrev && hasNext) {
            if (dot(nPrev, nNext) >= cosCrease) {
                const Vec2 sum = nPrev + nNext;
                pointN[i].push_back(lenSq2(sum) > 1e-8f ? normalize(sum) : nNext);
                prevSlot[i] = nextSlot[i] = 0;
            } else {
                pointN[i].push_back(nPrev);
                pointN[i].push_back(nNext);
                prevSlot[i] = 0;
                nextSlot[i] = 1;
            }
        } else if (hasNext) {
            pointN[i].push_back(nNext);
        } else {
            pointN[i].push_back(nPrev);
        }
    }

    // V coordinate: normalised arc length along the profile, which keeps texel
    // density even on a tall tapered glass.
    std::vector<float> vAt(P, 0.0f);
    {
        float total = 0.0f;
        for (int b = 0; b < bands; ++b) total += length(profile[(b + 1) % P] - profile[b]);
        if (total > 1e-9f) {
            float acc = 0.0f;
            for (int i = 0; i < P; ++i) {
                vAt[i] = acc / total;
                acc += length(profile[(i + 1) % P] - profile[i]);
            }
        }
    }

    // ---- rings.  A zero-radius point collapses to a single vertex on the axis;
    // its radial normal components cancel around the revolution, so the pole
    // normal is axis aligned.
    std::vector<std::vector<uint32_t>> ringStart(P);
    for (int i = 0; i < P; ++i) {
        const float rr = profile[i].x, y = profile[i].y;
        const uint32_t slots = static_cast<uint32_t>(pointN[i].size());
        ringStart[i].assign(slots, 0);
        for (uint32_t s = 0; s < slots; ++s) {
            const Vec2 n2 = pointN[i][s];
            ringStart[i][s] = m.vertexCount();
            if (isPole(i)) {
                const float ny = std::fabs(n2.y) < 1e-4f ? 1.0f : (n2.y < 0.0f ? -1.0f : 1.0f);
                m.vertices.push_back(mkVertex(Vec3(0, y, 0), Vec3(0, ny, 0), Vec2(0.0f, vAt[i])));
            } else {
                for (int j = 0; j <= S; ++j) {
                    const float u = static_cast<float>(j) / static_cast<float>(S);
                    const float phi = u * TWO_PI;
                    const float c = std::cos(phi), s2 = std::sin(phi);
                    const Vec3 n(n2.x * c, n2.y, n2.x * s2);
                    const Vec3 pn = normalize(n);
                    m.vertices.push_back(mkVertex(Vec3(rr * c, y, rr * s2), pn, Vec2(u, vAt[i])));
                }
            }
        }
    }

    // ---- bands: (A,B,C) and (A,C,D) is CCW seen from outside for a CCW profile.
    // A band touching a pole degenerates into a single fan triangle per segment.
    for (int b = 0; b < bands; ++b) {
        const int i0 = b, i1 = (b + 1) % P;
        const uint32_t r0 = ringStart[i0][static_cast<size_t>(nextSlot[i0])];
        const uint32_t r1 = ringStart[i1][static_cast<size_t>(prevSlot[i1])];
        const bool p0 = isPole(i0), p1 = isPole(i1);
        if (p0 && p1) continue;  // axis segment: zero-area band
        auto at = [](uint32_t start, int j, bool p) {
            return p ? start : start + static_cast<uint32_t>(j);
        };
        for (int j = 0; j < S; ++j) {
            const uint32_t A = at(r0, j, p0), D = at(r0, j + 1, p0);
            const uint32_t B = at(r1, j, p1), C = at(r1, j + 1, p1);
            if (p0) tri(m, A, B, C);
            else if (p1) tri(m, A, C, D);
            else {
                tri(m, A, B, C);
                tri(m, A, C, D);
            }
        }
    }
    finalize(m, material, "lathe");
    return m;
}

MeshData makeRoundedBox(Vec3 halfExtents, float cornerRadius, int cornerSegments, uint32_t material) {
    const Vec3 h(safeSize(halfExtents.x), safeSize(halfExtents.y), safeSize(halfExtents.z));
    float r = finite(cornerRadius) ? std::fabs(cornerRadius) : 0.0f;
    r = min2(r, min2(h.x, min2(h.y, h.z)));  // the radius must fit inside the box
    const int cs = safeCount(cornerSegments, 1);
    if (r < kMinSize * 0.5f) return makeBox(h, Vec2(1.0f, 1.0f), material);  // plain box

    const Vec3 inner(max2(h.x - r, 0.0f), max2(h.y - r, 0.0f), max2(h.z - r, 0.0f));

    // Per-axis sample coordinates: `cs` arc samples per rounded end plus the flat
    // span in between (kept as a single quad strip because it is planar).
    std::vector<float> axis[3];
    for (int a = 0; a < 3; ++a) {
        const float hh = h[a], in = inner[a];
        std::vector<float>& c = axis[a];
        auto pushUnique = [&c](float v) {
            if (c.empty() || std::fabs(c.back() - v) > 1e-6f) c.push_back(v);
        };
        for (int k = 0; k <= cs; ++k) {
            const float al = HALF_PI * (1.0f - static_cast<float>(k) / static_cast<float>(cs));
            pushUnique(-(in + r * std::sin(al)));
        }
        pushUnique(in);
        for (int k = 1; k <= cs; ++k) {
            const float al = HALF_PI * (static_cast<float>(k) / static_cast<float>(cs));
            pushUnique(in + r * std::sin(al));
        }
        if (c.size() < 2) { c.clear(); c.push_back(-hh); c.push_back(hh); }
    }

    // Face frames with cross(u, v) == n, exactly like makeBox.
    struct Face { Vec3 n, u, v; int ua, va; };
    static const Face faces[6] = {
        {{ 1, 0, 0}, { 0, 0,-1}, { 0, 1, 0}, 2, 1},
        {{-1, 0, 0}, { 0, 0, 1}, { 0, 1, 0}, 2, 1},
        {{ 0, 1, 0}, { 1, 0, 0}, { 0, 0,-1}, 0, 2},
        {{ 0,-1, 0}, { 1, 0, 0}, { 0, 0, 1}, 0, 2},
        {{ 0, 0, 1}, { 1, 0, 0}, { 0, 1, 0}, 0, 1},
        {{ 0, 0,-1}, {-1, 0, 0}, { 0, 1, 0}, 0, 1},
    };
    MeshData m;
    for (const Face& f : faces) {
        const float hn = std::fabs(dot(f.n, h));
        const std::vector<float>& cu = axis[f.ua];
        const std::vector<float>& cv = axis[f.va];
        const int nu = static_cast<int>(cu.size()), nvv = static_cast<int>(cv.size());
        const uint32_t base = m.vertexCount();
        for (int gv = 0; gv < nvv; ++gv) {
            for (int gu = 0; gu < nu; ++gu) {
                // Round the plain box point by pushing it out of the inner box:
                // q is the closest inner-box point and (p - q) is the SDF gradient,
                // which is therefore also the exact surface normal.
                const Vec3 pFace = f.n * hn + f.u * cu[gu] + f.v * cv[gv];
                const Vec3 q(clamp(pFace.x, -inner.x, inner.x), clamp(pFace.y, -inner.y, inner.y),
                             clamp(pFace.z, -inner.z, inner.z));
                const Vec3 d = pFace - q;
                const float dl = length(d);
                const Vec3 nrm = dl > 1e-12f ? d / dl : f.n;
                const Vec3 p = q + nrm * r;
                m.vertices.push_back(mkVertex(p, nrm, Vec2(dot(p, f.u), dot(p, f.v))));
            }
        }
        for (int gv = 0; gv < nvv - 1; ++gv) {
            for (int gu = 0; gu < nu - 1; ++gu) {
                const uint32_t a = base + static_cast<uint32_t>(gv * nu + gu);
                const uint32_t b = a + 1;
                const uint32_t d = a + static_cast<uint32_t>(nu);
                const uint32_t c = d + 1;
                tri(m, a, b, c);
                tri(m, a, c, d);
            }
        }
    }
    finalize(m, material, "roundedbox");
    return m;
}

MeshData makeFullscreenTriangle() {
    MeshData m;
    m.vertices.reserve(3);
    m.indices.reserve(3);
    // NDC triangle covering [-1,1]^2 with uv == position * 0.5 + 0.5, so `uv`
    // indexes a full-screen texture directly in the post-processing pass.
    m.vertices.push_back(mkVertex(Vec3(-1, -1, 0), Vec3(0, 0, 1), Vec2(0, 0)));
    m.vertices.push_back(mkVertex(Vec3(3, -1, 0), Vec3(0, 0, 1), Vec2(2, 0)));
    m.vertices.push_back(mkVertex(Vec3(-1, 3, 0), Vec3(0, 0, 1), Vec2(0, 2)));
    tri(m, 0, 1, 2);
    finalize(m, 0, "fullscreen");
    return m;
}

// ===========================================================================
// helpers
// ===========================================================================

void fixSeamNormals(MeshData& mesh, int segments) {
    const uint32_t nv = mesh.vertexCount();
    if (nv < 2) return;
    const int segs = safeCount(segments, 1);
    // Seam duplicates differ by roughly 360/segments degrees; merge anything
    // closer than that (plus slack) and leave genuinely hard creases alone.
    const float maxAngle = min2(50.0f, 360.0f / static_cast<float>(segs) + 25.0f);
    const float cosTol = std::cos(maxAngle * DEG2RAD);
    const std::vector<std::vector<uint32_t>> clusters =
        positionClusters(mesh.vertices, kPosTol * meshScale(mesh.vertices));
    for (const std::vector<uint32_t>& cl : clusters) {
        if (cl.size() < 2) continue;
        // Coincident vertices can belong to surfaces that meet at a right angle
        // (a cylinder cap rim on a side seam), so only merge normals that are
        // already within the tolerance of their group's average.
        std::vector<Vec3> groupSum;
        std::vector<std::vector<uint32_t>> groupMembers;
        for (uint32_t i : cl) {
            const Vec3 n = normalize(mesh.vertices[i].normal);
            if (lengthSq(n) < 0.5f) continue;
            int g = -1;
            for (size_t k = 0; k < groupSum.size(); ++k) {
                const Vec3 avg = normalize(groupSum[k]);
                if (lengthSq(avg) > 0.5f && dot(avg, n) >= cosTol) { g = static_cast<int>(k); break; }
            }
            if (g < 0) {
                groupSum.push_back(n);
                groupMembers.push_back({i});
            } else {
                groupSum[static_cast<size_t>(g)] += n;
                groupMembers[static_cast<size_t>(g)].push_back(i);
            }
        }
        for (size_t g = 0; g < groupSum.size(); ++g) {
            const Vec3 avg = normalize(groupSum[g]);
            if (lengthSq(avg) < 0.5f) continue;
            for (uint32_t i : groupMembers[g]) {
                Vertex& v = mesh.vertices[i];
                v.normal = avg;
                // Keep the tangent frame orthonormal against the new normal.
                const Vec3 t = v.tangent.xyz() - avg * dot(avg, v.tangent.xyz());
                if (lengthSq(t) > 1e-16f) v.tangent = Vec4(normalize(t), v.tangent.w);
            }
        }
    }
}

void scaleUv(MeshData& mesh, Vec2 s) {
    const Vec2 k = safeScale(s);
    for (Vertex& v : mesh.vertices) {
        v.uv = v.uv * k;
        v.uv1 = v.uv1 * k;
    }
}

void offsetUv(MeshData& mesh, Vec2 o) {
    for (Vertex& v : mesh.vertices) {
        v.uv += o;
        v.uv1 += o;
    }
}

void boxProjectUv(MeshData& mesh, float scale) {
    const float s = safeScale(scale);
    for (Vertex& v : mesh.vertices) {
        const Vec3 an = absv(v.normal);
        const Vec3 p = v.position;
        // Sign choices keep the projection unmirrored when a face is looked at
        // from outside, and both faces of an axis share the same world axes.
        Vec2 uv;
        if (an.x >= an.y && an.x >= an.z) {
            uv = Vec2(v.normal.x >= 0.0f ? -p.z : p.z, p.y);
        } else if (an.y >= an.x && an.y >= an.z) {
            uv = Vec2(p.x, v.normal.y >= 0.0f ? -p.z : p.z);
        } else {
            uv = Vec2(v.normal.z >= 0.0f ? p.x : -p.x, p.y);
        }
        v.uv = uv * s;
        v.uv1 = v.uv;
    }
}

Aabb computeBounds(const std::vector<Vertex>& vertices) {
    Aabb b;
    for (const Vertex& v : vertices) {
        if (!finite(v.position)) continue;  // never let a NaN poison the bounds
        b.expand(v.position);
    }
    return b;
}

}  // namespace room2::procgen
