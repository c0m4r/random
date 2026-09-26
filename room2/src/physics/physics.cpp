// room2 - compact impulse-based rigid body physics (implementation).
//
// Conventions and design notes
// ----------------------------
// * Body ids: the low 20 bits are a slot index into `bodies_`, the high 12 bits are a slot
//   generation. `removeBody()` retires the slot (bumping its generation) and pushes it onto
//   `freeSlots_`, so a stale id can never resolve to a recycled body. A slot whose generation
//   field would wrap is retired permanently rather than reused.
// * Empty slots: `bodies()` is indexed by slot and keeps removed slots so that live ids stay
//   valid. A removed slot is marked with `flags == 0` (none of BODY_STATIC, BODY_KINEMATIC,
//   BODY_SLEEPING, BODY_ACTIVE set) and with mass/invMass/invInertiaLocal zeroed plus an
//   *invalid* worldBounds (mn > mx) so it can never pass an AABB test. Every live body always
//   has at least one flag bit set (addBody() forces BODY_ACTIVE), so consumers iterate with
//   `if (b.flags == 0) continue;`. Trailing empty slots are trimmed in removeBody(), so
//   `bodyCount()` (defined inline in the frozen header as `bodies_.size()`, i.e. the slot
//   count) matches the live count whenever the newest slots are the ones being removed;
//   `activeBodyCount()` counts live, awake dynamic bodies.
// * Scratch memory: World's members are fixed by the header, so the solver's per-contact
//   temporaries and the broadphase grid live in a file-local, reused scratch block. That is
//   safe because a World is single-threaded and step()/moveCapsule() are not re-entrant, and
//   it keeps the hot path free of per-step heap traffic after the first few frames.
// * Narrow phase: every manifold generator produces signed separations (negative = overlap), so
//   the same code serves both the public `collideShapes()` (margin 0, overlaps only) and the
//   speculative contact generation used by the solver (small positive margin). Contacts with a
//   positive `penetration` mean "overlapping"; negative means "approaching, will touch within
//   this step".
// * Determinism: no randomness, no wall clock, no pointer-ordered containers. Every loop walks
//   slot order or an explicitly sorted key array.
//
#include "physics.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <utility>
#include <vector>

namespace room2::phys {
namespace {

// ---------------------------------------------------------------- constants
constexpr float kSpecMarginMax = 0.1f;          // largest speculative contact margin (m)
constexpr float kRestitutionThreshold = 0.5f;   // below this closing speed: no bounce
constexpr float kBaumgarteMaxVel = 0.6f;        // cap on the Baumgarte bias (m/s)
constexpr float kMaxLinearVelocity = 250.0f;    // sanity clamps
constexpr float kMaxAngularVelocity = 120.0f;
constexpr float kSkin = 0.002f;                 // character-controller back-off (m)
constexpr float kGroundSnap = 0.025f;           // ground probe distance (m)
constexpr int kMaxManifoldPoints = 8;

// ---------------------------------------------------------------- id helpers
constexpr uint32_t kSlotBits = 20;
constexpr uint32_t kSlotMask = (1u << kSlotBits) - 1u;
constexpr uint32_t kGenMax = (1u << (32 - kSlotBits)) - 1u;

inline uint32_t makeBodyId(uint32_t slot, uint32_t gen) { return (gen << kSlotBits) | slot; }
inline uint32_t idSlot(BodyId id) { return id & kSlotMask; }
inline uint32_t idGeneration(BodyId id) { return id >> kSlotBits; }

// A removed slot is detectable by `flags == 0` (a live body always has a flag bit set).
inline bool deadSlot(const Body& b) { return b.flags == 0; }

// ---------------------------------------------------------------- manifolds
struct ManifoldPoint {
    Vec3 point{0, 0, 0};      // world space, midway between the two surfaces
    Vec3 normalLocal{0, 1, 0};  // scratch: per-point normal in the reference shape's space
    float separation = 0.0f;  // signed distance along the normal (negative = overlap)
    uint32_t featureId = 0;
};

struct Manifold {
    Vec3 normal{0, 1, 0};     // unit, points from A towards B
    int count = 0;
    ManifoldPoint pts[kMaxManifoldPoints];
};

// ---------------------------------------------------------------- scratch
// Per-contact solver temporaries, parallel to World::contacts_. The tangent basis and the
// body slots are cached here so the iteration loops do not rebuild them (the basis needs two
// normalisations, which is the most expensive per-contact work in the solver).
struct ContactExtra {
    float separation = 0.0f;  // signed separation at generation time
    float bias = 0.0f;        // velocity target from Baumgarte + speculative contacts
    float restBias = 0.0f;    // restitution target velocity
    float posImpulse = 0.0f;  // accumulated position-correction impulse
    uint32_t slotA = 0;
    uint32_t slotB = 0;
    Vec3 t1{1, 0, 0};         // friction basis, orthogonal to the contact normal
    Vec3 t2{0, 0, 1};
};

// Per-slot solver view (zero inverse mass/inertia for static, kinematic and sleeping bodies).
struct SolverBody {
    float invMass = 0.0f;
    Mat3 invInertia;
    bool movable = false;
};

struct Scratch {
    std::vector<uint64_t> pairKeys;
    std::vector<uint32_t> cellStart;
    std::vector<uint32_t> cellCursor;
    std::vector<uint32_t> cellItems;
    std::vector<uint32_t> prevOrder;
    std::vector<ContactExtra> extra;
    std::vector<SolverBody> solver;
    std::vector<Aabb> inflBounds;
    std::vector<std::pair<BodyId, BodyId>> pairs;
};

Scratch& scratch() {
    static Scratch s;
    return s;
}

// ---------------------------------------------------------------- small math
inline Mat3 zeroMat3() {
    Mat3 m;
    m.c[0] = Vec3(0, 0, 0);
    m.c[1] = Vec3(0, 0, 0);
    m.c[2] = Vec3(0, 0, 0);
    return m;
}

// R * diag(d) * R^T - world inverse inertia from a local diagonal.
inline Mat3 rotateInertia(const Mat3& R, Vec3 d) {
    Mat3 RD;
    RD.c[0] = R.c[0] * d.x;
    RD.c[1] = R.c[1] * d.y;
    RD.c[2] = R.c[2] * d.z;
    return RD * transpose(R);
}

// Deterministic orthonormal basis with `n` as the first axis.
inline void contactBasis(Vec3 n, Vec3& t1, Vec3& t2) {
    Vec3 a = (std::fabs(n.x) < 0.9f) ? Vec3(1, 0, 0) : Vec3(0, 1, 0);
    t1 = normalize(cross(a, n));
    if (lengthSq(t1) < 1e-12f) t1 = normalize(cross(Vec3(0, 0, 1), n));
    t2 = cross(n, t1);
}

// Component-wise clamp of a point to an axis aligned box (math.hpp only offers a scalar
// clampv, so this is the box variant).
inline Vec3 clampToBox(Vec3 v, Vec3 he) { return minv(maxv(v, -he), he); }

inline Vec3 closestPointOnSegment(Vec3 a, Vec3 b, Vec3 p) {
    Vec3 ab = b - a;
    float dd = dot(ab, ab);
    if (dd < 1e-12f) return a;
    return a + ab * clamp(dot(p - a, ab) / dd, 0.0f, 1.0f);
}

// Closest points between two segments (Ericson, Real-Time Collision Detection).
inline void closestPointsSegments(Vec3 p1, Vec3 q1, Vec3 p2, Vec3 q2, Vec3& c1, Vec3& c2) {
    const Vec3 d1 = q1 - p1;
    const Vec3 d2 = q2 - p2;
    const Vec3 r = p1 - p2;
    const float a = dot(d1, d1);
    const float e = dot(d2, d2);
    const float f = dot(d2, r);
    const float eps = 1e-9f;
    float s = 0.0f, t = 0.0f;
    if (a <= eps && e <= eps) {
        c1 = p1;
        c2 = p2;
        return;
    }
    if (a <= eps) {
        t = clamp(f / e, 0.0f, 1.0f);
    } else {
        const float c = dot(d1, r);
        if (e <= eps) {
            s = clamp(-c / a, 0.0f, 1.0f);
        } else {
            const float b = dot(d1, d2);
            const float denom = a * e - b * b;
            s = denom > eps ? clamp((b * f - c * e) / denom, 0.0f, 1.0f) : 0.0f;
            t = (b * s + f) / e;
            if (t < 0.0f) {
                t = 0.0f;
                s = clamp(-c / a, 0.0f, 1.0f);
            } else if (t > 1.0f) {
                t = 1.0f;
                s = clamp((b - c) / a, 0.0f, 1.0f);
            }
        }
    }
    c1 = p1 + d1 * s;
    c2 = p2 + d2 * t;
}

// Segment parameter of the point closest to an axis aligned box. Distance to a convex set
// composed with an affine map is convex in t, so a ternary search is exact to tolerance.
inline float closestSegmentParamToAabb(Vec3 p0, Vec3 p1, Vec3 he) {
    auto f = [&](float t) {
        Vec3 s = lerp(p0, p1, t);
        Vec3 q = clampToBox(s, he);
        return lengthSq(q - s);
    };
    float lo = 0.0f, hi = 1.0f;
    for (int i = 0; i < 60 && (hi - lo) > 1e-5f; ++i) {
        const float m1 = lo + (hi - lo) / 3.0f;
        const float m2 = hi - (hi - lo) / 3.0f;
        if (f(m1) < f(m2)) {
            hi = m2;
        } else {
            lo = m1;
        }
    }
    return (lo + hi) * 0.5f;
}

// Projection radius of an oriented box onto a unit axis.
inline float boxProjectionRadius(Vec3 he, const Mat3& R, Vec3 L) {
    return he.x * std::fabs(dot(R.c[0], L)) + he.y * std::fabs(dot(R.c[1], L)) +
           he.z * std::fabs(dot(R.c[2], L));
}

inline float combineRestitution(const Material& a, const Material& b) {
    return max2(a.restitution, b.restitution);
}

// ---------------------------------------------------------------- box/sphere helper
struct BoxSphereLocal {
    bool hit = false;
    Vec3 normal{0, 1, 0};      // unit, from the box towards the sphere (box local frame)
    float separation = 0.0f;   // negative when overlapping
    Vec3 witnessBox{0, 0, 0};  // box local
    Vec3 witnessSphere{0, 0, 0};
};

// Box (centred at the local origin) against a sphere whose centre is `c` in box local space.
inline BoxSphereLocal boxSphereLocal(Vec3 he, Vec3 c, float radius, float margin) {
    BoxSphereLocal r;
    const Vec3 q = clampToBox(c, he);
    const Vec3 dl = c - q;
    const float dist = length(dl);
    if (dist > 1e-6f) {
        r.normal = dl / dist;
        r.separation = dist - radius;
        r.witnessBox = q;
        r.witnessSphere = c - r.normal * radius;
    } else {
        // Sphere centre inside the box: escape along the closest face.
        int axis = 0;
        float best = FLT_MAX;
        for (int i = 0; i < 3; ++i) {
            const float dd = he[i] - std::fabs(c[i]);
            if (dd < best) {
                best = dd;
                axis = i;
            }
        }
        const float sgn = c[axis] >= 0.0f ? 1.0f : -1.0f;
        r.normal = Vec3(0, 0, 0);
        r.normal[axis] = sgn;
        r.separation = -(best + radius);
        r.witnessBox = c + r.normal * best;
        r.witnessSphere = c + r.normal * radius;
    }
    r.hit = r.separation <= margin;
    return r;
}

// ---------------------------------------------------------------- manifold builders
void manifoldBoxBox(const Shape& sa, Vec3 pa, Quat qa, const Shape& sb, Vec3 pb, Quat qb,
                    float margin, Manifold& m) {
    const Mat3 Ra = mat3FromQuat(qa);
    const Mat3 Rb = mat3FromQuat(qb);
    const Vec3 ha = sa.halfExtents;
    const Vec3 hb = sb.halfExtents;
    const Vec3 d = pb - pa;

    float bestSep = -FLT_MAX;
    Vec3 bestAxis(0, 1, 0);
    int bestOwner = -1;  // 0 = face of A, 1 = face of B, 2 = edge pair
    int bestI = 0, bestJ = 0;
    float bestSign = 1.0f;
    // Separations inside a small band count as ties, so a symmetric configuration (a box
    // resting flat on a box) does not flip which face becomes the reference from frame to
    // frame. Such a flip changes every feature id and destroys the warm start, which makes
    // stacks sink. The band grows with the separation to stay scale independent.
    auto betterAxis = [&bestSep, &bestOwner](float sep) {
        if (bestOwner < 0) return true;
        return sep > bestSep + 1e-4f + 1e-3f * std::fabs(bestSep);
    };

    for (int i = 0; i < 3; ++i) {
        const Vec3 L = Ra.c[i];
        const float s = dot(d, L);
        const float sep =
            std::fabs(s) - (boxProjectionRadius(ha, Ra, L) + boxProjectionRadius(hb, Rb, L));
        if (sep > margin) return;
        if (betterAxis(sep)) {
            bestSep = sep;
            bestAxis = L;
            bestOwner = 0;
            bestI = i;
            bestSign = s < 0.0f ? -1.0f : 1.0f;
        }
    }
    for (int j = 0; j < 3; ++j) {
        const Vec3 L = Rb.c[j];
        const float s = dot(d, L);
        const float sep =
            std::fabs(s) - (boxProjectionRadius(ha, Ra, L) + boxProjectionRadius(hb, Rb, L));
        if (sep > margin) return;
        if (betterAxis(sep)) {
            bestSep = sep;
            bestAxis = L;
            bestOwner = 1;
            bestI = j;
            bestSign = s < 0.0f ? -1.0f : 1.0f;
        }
    }
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            Vec3 L = cross(Ra.c[i], Rb.c[j]);
            const float len = length(L);
            if (len < 1e-6f) continue;  // parallel edges: the face axes already cover this
            L = L / len;
            const float s = dot(d, L);
            const float sep =
                std::fabs(s) - (boxProjectionRadius(ha, Ra, L) + boxProjectionRadius(hb, Rb, L));
            if (sep > margin) return;
            if (betterAxis(sep)) {
                bestSep = sep;
                bestAxis = L;
                bestOwner = 2;
                bestI = i;
                bestJ = j;
                bestSign = s < 0.0f ? -1.0f : 1.0f;
            }
        }
    }
    if (bestSep > margin) return;

    // Normal always points from A towards B.
    const Vec3 n = bestSign > 0.0f ? bestAxis : -bestAxis;
    m.normal = n;
    m.count = 0;

    if (bestOwner == 2) {
        // Edge/edge: single deepest point from the closest points of the two support edges.
        Vec3 ea0 = pa, ea1 = pa;
        for (int k = 0; k < 3; ++k) {
            if (k == bestI) continue;
            const float sgn = dot(n, Ra.c[k]) >= 0.0f ? 1.0f : -1.0f;
            ea0 += Ra.c[k] * (ha[k] * sgn);
            ea1 += Ra.c[k] * (ha[k] * sgn);
        }
        ea0 -= Ra.c[bestI] * ha[bestI];
        ea1 += Ra.c[bestI] * ha[bestI];

        Vec3 eb0 = pb, eb1 = pb;
        for (int k = 0; k < 3; ++k) {
            if (k == bestJ) continue;
            const float sgn = dot(n, Rb.c[k]) <= 0.0f ? 1.0f : -1.0f;
            eb0 += Rb.c[k] * (hb[k] * sgn);
            eb1 += Rb.c[k] * (hb[k] * sgn);
        }
        eb0 -= Rb.c[bestJ] * hb[bestJ];
        eb1 += Rb.c[bestJ] * hb[bestJ];

        Vec3 c1, c2;
        closestPointsSegments(ea0, ea1, eb0, eb1, c1, c2);
        const float sep = dot(c2 - c1, n);
        if (sep > margin) return;
        m.pts[0].point = (c1 + c2) * 0.5f;
        m.pts[0].separation = sep;
        m.pts[0].featureId = 0x8000u | (uint32_t(bestI) << 4) | uint32_t(bestJ);
        m.count = 1;
        return;
    }

    // ---- face contact: clip the incident face against the reference face side planes.
    const bool refIsA = (bestOwner == 0);
    const Vec3 refHe = refIsA ? ha : hb;
    const Vec3 incHe = refIsA ? hb : ha;
    const Vec3 refPos = refIsA ? pa : pb;
    const Vec3 incPos = refIsA ? pb : pa;
    const Mat3& Rref = refIsA ? Ra : Rb;
    const Mat3& Rinc = refIsA ? Rb : Ra;
    const int refAxis = bestI;

    // Outward normal of the reference face (towards the incident box).
    const Vec3 nRef = refIsA ? n : -n;
    const float refSgn = dot(nRef, Rref.c[refAxis]) > 0.0f ? 1.0f : -1.0f;
    const Vec3 refFaceCenter = refPos + Rref.c[refAxis] * (refHe[refAxis] * refSgn);

    // Incident face = the face of the incident box whose outward normal is most anti-parallel
    // to nRef, i.e. the face pointing into the reference box. Both signs of every box axis must
    // be considered: taking only +axis[k] picks a side face when a box lies flat on a face.
    int incAxis = 0;
    float incSign = 1.0f;
    float bestDot = FLT_MAX;
    for (int k = 0; k < 3; ++k) {
        const float dp = dot(Rinc.c[k], nRef);
        const float sgn = dp <= 0.0f ? 1.0f : -1.0f;
        const float aligned = dp * sgn;  // == -|dp|, the most anti-parallel orientation
        if (aligned < bestDot) {
            bestDot = aligned;
            incAxis = k;
            incSign = sgn;
        }
    }
    const Vec3 nInc = Rinc.c[incAxis] * incSign;
    const Vec3 incFaceCenter = incPos + nInc * incHe[incAxis];
    const int u1 = (incAxis + 1) % 3;
    const int u2 = (incAxis + 2) % 3;
    const Vec3 eu = Rinc.c[u1] * incHe[u1];
    const Vec3 ev = Rinc.c[u2] * incHe[u2];

    Vec3 poly[8];
    Vec3 tmp[8];
    int nv = 4;
    poly[0] = incFaceCenter + eu + ev;
    poly[1] = incFaceCenter - eu + ev;
    poly[2] = incFaceCenter - eu - ev;
    poly[3] = incFaceCenter + eu - ev;

    // Clip against the four side planes of the reference face.
    const Vec3 sideAxis[2] = {Rref.c[(refAxis + 1) % 3], Rref.c[(refAxis + 2) % 3]};
    const float sideHe[2] = {refHe[(refAxis + 1) % 3], refHe[(refAxis + 2) % 3]};
    for (int s = 0; s < 2 && nv > 0; ++s) {
        for (int sgn = -1; sgn <= 1 && nv > 0; sgn += 2) {
            const Vec3 pl = sideAxis[s] * float(sgn);
            const float off = dot(pl, refFaceCenter) + sideHe[s];
            int nout = 0;
            for (int i = 0; i < nv; ++i) {
                const Vec3 cur = poly[i];
                const Vec3 nxt = poly[(i + 1) % nv];
                const float dc = dot(pl, cur) - off;
                const float dn = dot(pl, nxt) - off;
                if (dc <= 0.0f && nout < 8) tmp[nout++] = cur;
                if (((dc < 0.0f) != (dn < 0.0f)) && nout < 8) {
                    const float tt = dc / (dc - dn);
                    tmp[nout++] = cur + (nxt - cur) * tt;
                }
            }
            // Remove degenerate/duplicate vertices: a vertex lying exactly on a clip plane
            // emits both itself and a zero-length "intersection", and duplicated manifold
            // points double the solver work for the same corner.
            nv = 0;
            for (int i = 0; i < nout; ++i) {
                bool dup = false;
                for (int k = 0; k < nv; ++k) {
                    if (lengthSq(tmp[i] - poly[k]) < 1e-10f) {
                        dup = true;
                        break;
                    }
                }
                if (!dup) poly[nv++] = tmp[i];
            }
        }
    }
    if (nv < 3) return;

    // Keep the points that touch/overlap the reference face plane, deepest first.
    ManifoldPoint keep[8];
    int nk = 0;
    const float planeOffset = dot(nRef, refFaceCenter);
    for (int i = 0; i < nv && nk < 8; ++i) {
        const float sep = dot(nRef, poly[i]) - planeOffset;
        if (sep > margin) continue;
        const Vec3 onRef = poly[i] - nRef * sep;  // projection onto the reference face
        ManifoldPoint mp;
        mp.point = (poly[i] + onRef) * 0.5f;
        mp.separation = sep;
        mp.featureId = (refIsA ? 0u : 1u) * 4096u + uint32_t(refAxis) * 256u +
                       uint32_t(incAxis) * 16u + uint32_t(i);
        keep[nk++] = mp;
    }
    // Insertion sort by separation (ascending = deepest first); stable and deterministic.
    for (int i = 1; i < nk; ++i) {
        ManifoldPoint key = keep[i];
        int j = i - 1;
        while (j >= 0 && keep[j].separation > key.separation) {
            keep[j + 1] = keep[j];
            --j;
        }
        keep[j + 1] = key;
    }
    m.count = min2(nk, kMaxManifoldPoints);
    for (int i = 0; i < m.count; ++i) m.pts[i] = keep[i];
}

void manifoldBoxSphere(const Shape& box, Vec3 pbox, Quat qbox, const Shape& sphere, Vec3 psph,
                       float margin, Manifold& m) {
    const Vec3 cl = rotate(conjugate(qbox), psph - pbox);
    const BoxSphereLocal r = boxSphereLocal(box.halfExtents, cl, sphere.radius, margin);
    if (!r.hit) return;
    m.normal = rotate(qbox, r.normal);
    m.count = 1;
    m.pts[0].separation = r.separation;
    m.pts[0].featureId = 0x100u;
    const Vec3 wBox = pbox + rotate(qbox, r.witnessBox);
    const Vec3 wSph = pbox + rotate(qbox, r.witnessSphere);
    m.pts[0].point = (wBox + wSph) * 0.5f;
}

void manifoldCapsuleSphere(const Shape& cap, Vec3 pcap, Quat qcap, const Shape& sphere, Vec3 psph,
                           float margin, Manifold& m) {
    const Vec3 axis = rotate(qcap, Vec3(0, 1, 0));
    const Vec3 a0 = pcap - axis * cap.halfHeight;
    const Vec3 a1 = pcap + axis * cap.halfHeight;
    const Vec3 cp = closestPointOnSegment(a0, a1, psph);
    Vec3 dv = cp - psph;
    float dist = length(dv);
    if (dist < 1e-6f) {
        dv = pcap - psph;
        dist = length(dv);
        if (dist < 1e-6f) {
            dv = Vec3(0, 1, 0);
            dist = 1.0f;
        }
    }
    const Vec3 n = dv / dist;  // from the sphere towards the capsule
    const float sep = dist - (cap.radius + sphere.radius);
    if (sep > margin) return;
    // Public normal points A -> B, and A is the capsule here, so flip.
    m.normal = -n;
    m.count = 1;
    m.pts[0].separation = sep;
    m.pts[0].featureId = 0x300u;
    const Vec3 wCap = cp - n * cap.radius;
    const Vec3 wSph = psph + n * sphere.radius;
    m.pts[0].point = (wCap + wSph) * 0.5f;
}

void manifoldSphereSphere(const Shape& sa, Vec3 pa, const Shape& sb, Vec3 pb, float margin,
                          Manifold& m) {
    Vec3 dv = pb - pa;
    float dist = length(dv);
    if (dist < 1e-6f) {
        dv = Vec3(0, 1, 0);
        dist = 1.0f;
    }
    const Vec3 n = dv / dist;
    const float sep = dist - (sa.radius + sb.radius);
    if (sep > margin) return;
    m.normal = n;
    m.count = 1;
    m.pts[0].separation = sep;
    m.pts[0].featureId = 0x200u;
    const Vec3 wa = pa + n * sa.radius;
    const Vec3 wb = pb - n * sb.radius;
    m.pts[0].point = (wa + wb) * 0.5f;
}

void manifoldCapsuleCapsule(const Shape& sa, Vec3 pa, Quat qa, const Shape& sb, Vec3 pb, Quat qb,
                            float margin, Manifold& m) {
    const Vec3 ax = rotate(qa, Vec3(0, 1, 0));
    const Vec3 bx = rotate(qb, Vec3(0, 1, 0));
    Vec3 c1, c2;
    closestPointsSegments(pa - ax * sa.halfHeight, pa + ax * sa.halfHeight,
                          pb - bx * sb.halfHeight, pb + bx * sb.halfHeight, c1, c2);
    Vec3 dv = c2 - c1;
    float dist = length(dv);
    if (dist < 1e-6f) {
        dv = pb - pa;
        dist = length(dv);
        if (dist < 1e-6f) {
            dv = Vec3(0, 1, 0);
            dist = 1.0f;
        }
    }
    const Vec3 n = dv / dist;
    const float sep = dist - (sa.radius + sb.radius);
    if (sep > margin) return;
    m.normal = n;
    m.count = 1;
    m.pts[0].separation = sep;
    m.pts[0].featureId = 0x400u;
    const Vec3 wa = c1 + n * sa.radius;
    const Vec3 wb = c2 - n * sb.radius;
    m.pts[0].point = (wa + wb) * 0.5f;
}

// Box against capsule. The capsule is sampled along its segment (and at the closest point to
// the box) so that a capsule lying flat on a face gets more than one support point.
void manifoldBoxCapsule(const Shape& box, Vec3 pbox, Quat qbox, const Shape& cap, Vec3 pcap,
                        Quat qcap, float margin, Manifold& m) {
    const Quat inv = conjugate(qbox);
    const Vec3 axisW = rotate(qcap, Vec3(0, 1, 0));
    const Vec3 p0w = pcap - axisW * cap.halfHeight;
    const Vec3 p1w = pcap + axisW * cap.halfHeight;
    const Vec3 p0 = rotate(inv, p0w - pbox);
    const Vec3 p1 = rotate(inv, p1w - pbox);
    const Vec3 he = box.halfExtents;

    const float tStar = closestSegmentParamToAabb(p0, p1, he);
    const float ts[4] = {0.0f, tStar, 0.5f, 1.0f};

    ManifoldPoint keep[4];
    int nk = 0;
    for (int i = 0; i < 4; ++i) {
        const float t = ts[i];
        // Skip samples that coincide with one already taken.
        bool dup = false;
        for (int j = 0; j < i && !dup; ++j) {
            if (std::fabs(ts[j] - t) < 0.02f) dup = true;
        }
        if (dup) continue;
        const Vec3 c = lerp(p0, p1, t);
        const BoxSphereLocal r = boxSphereLocal(he, c, cap.radius, margin);
        if (!r.hit) continue;
        ManifoldPoint mp;
        mp.separation = r.separation;
        mp.featureId = 0x500u + uint32_t(i);
        const Vec3 wBox = pbox + rotate(qbox, r.witnessBox);
        const Vec3 wCap = pbox + rotate(qbox, r.witnessSphere);
        mp.point = (wBox + wCap) * 0.5f;
        mp.normalLocal = r.normal;
        keep[nk++] = mp;
    }
    if (nk == 0) return;
    // Sort deepest first.
    for (int i = 1; i < nk; ++i) {
        ManifoldPoint key = keep[i];
        int j = i - 1;
        while (j >= 0 && keep[j].separation > key.separation) {
            keep[j + 1] = keep[j];
            --j;
        }
        keep[j + 1] = key;
    }
    m.count = nk;
    // All samples of a flat contact share the same face normal; use the deepest sample's.
    m.normal = rotate(qbox, keep[0].normalLocal);
    for (int i = 0; i < nk; ++i) {
        m.pts[i].point = keep[i].point;
        m.pts[i].separation = keep[i].separation;
        m.pts[i].featureId = keep[i].featureId;
    }
}

// Reversing the argument order only flips the normal: separations and witness points are
// unchanged, and the normal is rebuilt from the flipped direction by the caller.
inline void flipManifold(Manifold& m) { m.normal = -m.normal; }

// Dispatch: fills `m` for the (A, B) order; `m.normal` points A -> B.
void collideManifold(const Shape& sa, Vec3 pa, Quat qa, const Shape& sb, Vec3 pb, Quat qb,
                     float margin, Manifold& m) {
    m.count = 0;
    m.normal = Vec3(0, 1, 0);
    switch (sa.type) {
        case ShapeType::Box:
            switch (sb.type) {
                case ShapeType::Box: manifoldBoxBox(sa, pa, qa, sb, pb, qb, margin, m); break;
                case ShapeType::Sphere: manifoldBoxSphere(sa, pa, qa, sb, pb, margin, m); break;
                case ShapeType::Capsule:
                    manifoldBoxCapsule(sa, pa, qa, sb, pb, qb, margin, m);
                    break;
            }
            break;
        case ShapeType::Sphere:
            switch (sb.type) {
                case ShapeType::Box:
                    manifoldBoxSphere(sb, pb, qb, sa, pa, margin, m);
                    flipManifold(m);
                    break;
                case ShapeType::Sphere: manifoldSphereSphere(sa, pa, sb, pb, margin, m); break;
                case ShapeType::Capsule:
                    manifoldCapsuleSphere(sb, pb, qb, sa, pa, margin, m);
                    flipManifold(m);
                    break;
            }
            break;
        case ShapeType::Capsule:
            switch (sb.type) {
                case ShapeType::Box:
                    manifoldBoxCapsule(sb, pb, qb, sa, pa, qa, margin, m);
                    flipManifold(m);
                    break;
                case ShapeType::Sphere:
                    manifoldCapsuleSphere(sa, pa, qa, sb, pb, margin, m);
                    break;
                case ShapeType::Capsule:
                    manifoldCapsuleCapsule(sa, pa, qa, sb, pb, qb, margin, m);
                    break;
            }
            break;
    }
}

// ---------------------------------------------------------------- broadphase
inline uint32_t hashCell(int x, int y, int z, uint32_t mask) {
    uint32_t h = uint32_t(x) * 73856093u ^ uint32_t(y) * 19349663u ^ uint32_t(z) * 83492791u;
    return (h ^ (h >> 15)) & mask;
}

// Uniform spatial hash over the dynamic bodies. Static and kinematic bodies are few in a room
// (floor, walls, table), so they are tested against every dynamic body directly instead of
// being inserted into the grid.
struct Broadphase {
    // Per-body speculative inflation: how far the body can travel during one sub-step. This
    // keeps the grid tight for resting piles (zero inflation) while still finding contacts for
    // fast bodies, which is what makes the speculative contacts prevent tunnelling.
    static float bodyInflate(const Body& b, float dt) {
        const float lin = length(b.linearVelocity) * dt;
        const float ang = length(b.angularVelocity) * b.shape.boundingRadius() * dt;
        return clamp(lin + ang, 0.0f, kSpecMarginMax);
    }

    static void computePairs(const std::vector<Body>& bodies, float dt,
                             std::vector<std::pair<BodyId, BodyId>>& out) {
        out.clear();
        Scratch& sc = scratch();
        sc.pairKeys.clear();
        const uint32_t n = uint32_t(bodies.size());

        // Inflated (swept) bounds of every live dynamic body.
        sc.inflBounds.assign(n, Aabb{});
        Aabb world;
        uint32_t dynCount = 0;
        for (uint32_t i = 0; i < n; ++i) {
            const Body& b = bodies[i];
            if (deadSlot(b) || !b.isDynamic()) continue;
            const float inflate = bodyInflate(b, dt);
            Aabb ib{b.worldBounds.mn - Vec3(inflate), b.worldBounds.mx + Vec3(inflate)};
            sc.inflBounds[i] = ib;
            world.expand(ib);
            ++dynCount;
        }

        if (dynCount > 1 && world.valid()) {
            const Vec3 extent = world.mx - world.mn;
            float cell = 0.35f;
            cell = max2(cell, maxComponent(extent) / 128.0f);
            if (cell < 1e-4f) cell = 1e-4f;
            const float invCell = 1.0f / cell;

            uint32_t tableSize = 64;
            while (tableSize < dynCount * 2u && tableSize < (1u << 16)) tableSize <<= 1;
            const uint32_t mask = tableSize - 1u;

            const Vec3 origin = world.mn;
            // Cell range of a body, clamped so a body never spans more than 4 cells per axis
            // (bodies exceeding that are handled by the brute-force pass below).
            auto cellRange = [&](const Aabb& bb, int mn[3], int mx[3], bool& large) {
                large = false;
                for (int a = 0; a < 3; ++a) {
                    int i0 = int(std::floor((bb.mn[a] - origin[a]) * invCell));
                    int i1 = int(std::floor((bb.mx[a] - origin[a]) * invCell));
                    if (i0 > i1) std::swap(i0, i1);
                    if (i1 - i0 > 3) {
                        large = true;
                        i1 = i0 + 3;
                    }
                    mn[a] = i0;
                    mx[a] = i1;
                }
            };

            const uint32_t cells = tableSize;
            sc.cellStart.assign(cells + 1, 0);

            // Pass 1: count entries per bucket.
            for (uint32_t i = 0; i < n; ++i) {
                const Body& b = bodies[i];
                if (deadSlot(b) || !b.isDynamic()) continue;
                int mn[3], mx[3];
                bool large = false;
                cellRange(sc.inflBounds[i], mn, mx, large);
                for (int x = mn[0]; x <= mx[0]; ++x)
                    for (int y = mn[1]; y <= mx[1]; ++y)
                        for (int z = mn[2]; z <= mx[2]; ++z)
                            ++sc.cellStart[hashCell(x, y, z, mask) + 1];
            }
            for (uint32_t i = 0; i < cells; ++i) sc.cellStart[i + 1] += sc.cellStart[i];
            sc.cellItems.resize(sc.cellStart[cells]);
            sc.cellCursor.assign(sc.cellStart.begin(), sc.cellStart.begin() + cells);
            // Pass 2: fill buckets (insertion order = slot order, so buckets stay sorted).
            for (uint32_t i = 0; i < n; ++i) {
                const Body& b = bodies[i];
                if (deadSlot(b) || !b.isDynamic()) continue;
                int mn[3], mx[3];
                bool large = false;
                cellRange(sc.inflBounds[i], mn, mx, large);
                for (int x = mn[0]; x <= mx[0]; ++x)
                    for (int y = mn[1]; y <= mx[1]; ++y)
                        for (int z = mn[2]; z <= mx[2]; ++z)
                            sc.cellItems[sc.cellCursor[hashCell(x, y, z, mask)]++] = i;
            }

            // Pair enumeration. Each body queries the cells it covers; duplicates (two AABBs
            // sharing several cells) are removed by the sort + unique below.
            for (uint32_t i = 0; i < n; ++i) {
                const Body& a = bodies[i];
                if (deadSlot(a) || !a.isDynamic()) continue;
                int mn[3], mx[3];
                bool large = false;
                cellRange(sc.inflBounds[i], mn, mx, large);
                if (large) continue;  // handled by the brute-force pass below
                const Aabb& ib = sc.inflBounds[i];
                for (int x = mn[0]; x <= mx[0]; ++x) {
                    for (int y = mn[1]; y <= mx[1]; ++y) {
                        for (int z = mn[2]; z <= mx[2]; ++z) {
                            const uint32_t h = hashCell(x, y, z, mask);
                            for (uint32_t e = sc.cellStart[h]; e < sc.cellStart[h + 1]; ++e) {
                                const uint32_t j = sc.cellItems[e];
                                if (j <= i) continue;
                                const Body& b = bodies[j];
                                if (!ib.overlaps(sc.inflBounds[j])) continue;
                                if ((a.collisionGroup & b.collisionMask) == 0u) continue;
                                if ((b.collisionGroup & a.collisionMask) == 0u) continue;
                                sc.pairKeys.push_back((uint64_t(i) << 32) | uint64_t(j));
                            }
                        }
                    }
                }
            }

            // Bodies too large for the grid (rare in a room: only something like a moving
            // platform) are tested against every other dynamic body.
            for (uint32_t i = 0; i < n; ++i) {
                const Body& a = bodies[i];
                if (deadSlot(a) || !a.isDynamic()) continue;
                int mn[3], mx[3];
                bool large = false;
                cellRange(sc.inflBounds[i], mn, mx, large);
                if (!large) continue;
                for (uint32_t j = 0; j < n; ++j) {
                    if (j == i) continue;
                    const Body& b = bodies[j];
                    if (deadSlot(b) || !b.isDynamic()) continue;
                    if (!sc.inflBounds[i].overlaps(sc.inflBounds[j])) continue;
                    if ((a.collisionGroup & b.collisionMask) == 0u) continue;
                    if ((b.collisionGroup & a.collisionMask) == 0u) continue;
                    const uint32_t lo = min2(i, j), hi = max2(i, j);
                    sc.pairKeys.push_back((uint64_t(lo) << 32) | uint64_t(hi));
                }
            }
        }

        // Dynamic vs static/kinematic (brute force: a room has only a handful of them).
        for (uint32_t i = 0; i < n; ++i) {
            const Body& a = bodies[i];
            if (deadSlot(a) || !a.isDynamic()) continue;
            const Aabb& ib = sc.inflBounds[i];
            for (uint32_t j = 0; j < n; ++j) {
                const Body& b = bodies[j];
                if (deadSlot(b) || b.isDynamic()) continue;
                if (!ib.overlaps(b.worldBounds)) continue;
                if ((a.collisionGroup & b.collisionMask) == 0u) continue;
                if ((b.collisionGroup & a.collisionMask) == 0u) continue;
                sc.pairKeys.push_back((uint64_t(i) << 32) | uint64_t(j));
            }
        }

        std::sort(sc.pairKeys.begin(), sc.pairKeys.end());
        sc.pairKeys.erase(std::unique(sc.pairKeys.begin(), sc.pairKeys.end()), sc.pairKeys.end());
        out.reserve(sc.pairKeys.size());
        for (uint64_t k : sc.pairKeys) {
            out.emplace_back(BodyId(k >> 32), BodyId(k & 0xFFFFFFFFu));
        }
    }
};

// ---------------------------------------------------------------- shape distance
// Signed separation between a capsule and another shape; `normalOut` points from the capsule
// towards the other shape and is unit length. Used by the character controller.
bool capsuleShapeSeparation(const Shape& cap, Vec3 pcap, Quat qcap, const Shape& other,
                            Vec3 pother, Quat qother, float& sepOut, Vec3& normalOut) {
    const Vec3 axisW = rotate(qcap, Vec3(0, 1, 0));
    const Vec3 c0 = pcap - axisW * cap.halfHeight;
    const Vec3 c1 = pcap + axisW * cap.halfHeight;
    switch (other.type) {
        case ShapeType::Sphere: {
            const Vec3 cp = closestPointOnSegment(c0, c1, pother);
            Vec3 dv = pother - cp;
            float dist = length(dv);
            if (dist < 1e-5f) {
                dv = Vec3(0, 1, 0);
                dist = 1.0f;
            }
            normalOut = dv / dist;
            sepOut = dist - (cap.radius + other.radius);
            return true;
        }
        case ShapeType::Capsule: {
            const Vec3 bx = rotate(qother, Vec3(0, 1, 0));
            Vec3 s1, s2;
            closestPointsSegments(c0, c1, pother - bx * other.halfHeight,
                                  pother + bx * other.halfHeight, s1, s2);
            Vec3 dv = s2 - s1;
            float dist = length(dv);
            if (dist < 1e-5f) {
                dv = pother - pcap;
                dist = length(dv);
                if (dist < 1e-5f) {
                    dv = Vec3(0, 1, 0);
                    dist = 1.0f;
                }
            }
            normalOut = dv / dist;
            sepOut = dist - (cap.radius + other.radius);
            return true;
        }
        case ShapeType::Box: {
            const Quat inv = conjugate(qother);
            const Vec3 p0 = rotate(inv, c0 - pother);
            const Vec3 p1 = rotate(inv, c1 - pother);
            const Vec3 he = other.halfExtents;
            const float t = closestSegmentParamToAabb(p0, p1, he);
            const Vec3 s = lerp(p0, p1, t);
            const Vec3 q = clampToBox(s, he);
            const Vec3 dl = q - s;
            const float dist = length(dl);
            Vec3 nLocal;
            if (dist > 1e-6f) {
                nLocal = dl / dist;
                sepOut = dist - cap.radius;
            } else {
                int axis = 0;
                float best = FLT_MAX;
                for (int i = 0; i < 3; ++i) {
                    const float dd = he[i] - std::fabs(s[i]);
                    if (dd < best) {
                        best = dd;
                        axis = i;
                    }
                }
                const float sgn = s[axis] >= 0.0f ? 1.0f : -1.0f;
                nLocal = Vec3(0, 0, 0);
                nLocal[axis] = sgn;
                sepOut = -(best + cap.radius);
            }
            normalOut = rotate(qother, nLocal);
            return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------- ray/shape tests
bool rayObbLocal(Vec3 origin, Vec3 dir, Vec3 he, float& tHit, Vec3& normalLocal) {
    float t0 = -FLT_MAX, t1 = FLT_MAX;
    float nearAxis = 0.0f;
    for (int i = 0; i < 3; ++i) {
        const float d = dir[i];
        if (std::fabs(d) < 1e-9f) {
            if (origin[i] < -he[i] || origin[i] > he[i]) return false;
        } else {
            const float inv = 1.0f / d;
            float ta = (-he[i] - origin[i]) * inv;
            float tb = (he[i] - origin[i]) * inv;
            if (ta > tb) std::swap(ta, tb);
            if (ta > t0) {
                t0 = ta;
                nearAxis = float(i);
            }
            t1 = min2(t1, tb);
            if (t0 > t1) return false;
        }
    }
    if (t1 < 0.0f) return false;
    tHit = t0 >= 0.0f ? t0 : t1;
    const Vec3 p = origin + dir * tHit;
    // Recover the face normal from the hit point (works for entry and exit hits alike).
    int axis = int(nearAxis);
    float best = -1.0f;
    for (int i = 0; i < 3; ++i) {
        const float ratio = he[i] > 1e-9f ? std::fabs(p[i]) / he[i] : 1.0f;
        if (ratio > best) {
            best = ratio;
            axis = i;
        }
    }
    normalLocal = Vec3(0, 0, 0);
    normalLocal[axis] = p[axis] >= 0.0f ? 1.0f : -1.0f;
    return true;
}

bool rayCapsuleShape(Vec3 origin, Vec3 dir, Vec3 c0, Vec3 c1, float radius, float& tHit,
                     Vec3& normal) {
    const Vec3 ba = c1 - c0;
    const Vec3 oa = origin - c0;
    const float baba = dot(ba, ba);
    const float bard = dot(ba, dir);
    const float baoa = dot(ba, oa);
    const float rdoa = dot(dir, oa);
    const float oaoa = dot(oa, oa);
    const float a = baba - bard * bard;
    float best = FLT_MAX;
    bool found = false;
    if (a > 1e-9f) {
        const float b = baba * rdoa - baoa * bard;
        const float c = baba * oaoa - baoa * baoa - radius * radius * baba;
        const float h = b * b - a * c;
        if (h >= 0.0f) {
            const float sq = std::sqrt(h);
            for (int s = 0; s < 2; ++s) {
                const float t = (-b + (s == 0 ? -sq : sq)) / a;
                if (t < 1e-6f || t >= best) continue;
                const float y = baoa + t * bard;
                if (y > 0.0f && y < baba) {
                    const Vec3 p = origin + dir * t;
                    const Vec3 axisPt = c0 + ba * (y / baba);
                    normal = normalize(p - axisPt);
                    best = t;
                    found = true;
                }
            }
        }
    }
    // Spherical caps.
    const Vec3 caps[2] = {c0, c1};
    for (int i = 0; i < 2; ++i) {
        float t;
        if (raySphere(Ray{origin, dir}, caps[i], radius, t) && t < best) {
            best = t;
            normal = normalize(origin + dir * t - caps[i]);
            found = true;
        }
    }
    if (!found) return false;
    tHit = best;
    return true;
}

// ---------------------------------------------------------------- character controller
struct SlideOutcome {
    Vec3 position{0, 0, 0};
    bool grounded = false;
    Vec3 groundNormal{0, 1, 0};
    bool hitWall = false;
    bool hitCeiling = false;
};

inline bool obstacleBody(const Body& b) {
    return !deadSlot(b) && (b.isStatic() || (b.flags & BODY_KINEMATIC) != 0);
}

// Push the capsule out of every obstacle it currently overlaps.
Vec3 depenetrateCapsule(const std::vector<Body>& bodies, Vec3 pos, const Shape& cap) {
    for (int it = 0; it < 5; ++it) {
        float worst = 0.0f;
        Vec3 worstN(0, 1, 0);
        for (const Body& b : bodies) {
            if (!obstacleBody(b)) continue;
            float sep;
            Vec3 n;
            if (!capsuleShapeSeparation(cap, pos, Quat(0, 0, 0, 1), b.shape, b.position,
                                        b.orientation, sep, n))
                continue;
            const float pen = -sep;
            if (pen > worst) {
                worst = pen;
                worstN = n;
            }
        }
        if (worst <= 1e-5f) break;
        pos -= worstN * (worst + 1e-4f);
    }
    return pos;
}

// Collide-and-slide sweep. The separation normal returned for the capsule points *from the
// capsule towards the obstacle*, so the surface normal is its negation.
SlideOutcome slideCapsule(const std::vector<Body>& bodies, Vec3 pos, const Shape& cap, Vec3 delta,
                          float maxSlopeCos) {
    SlideOutcome out;
    out.position = pos;
    // Long moves are consumed in conservative slices: each iteration advances at most this far
    // so the (exact, but narrow-phase derived) distance query can never be stepped over.
    const float maxSlice = min2(0.25f, max2(0.02f, cap.radius));
    const int maxIters = 64;
    Vec3 remaining = delta;
    for (int iter = 0; iter < maxIters; ++iter) {
        const float len = length(remaining);
        if (len < 1e-5f) break;
        const Vec3 dir = remaining * (1.0f / len);
        float advance = len;
        Vec3 hitN(0, 0, 0);
        bool anyHit = false;
        for (const Body& b : bodies) {
            if (!obstacleBody(b)) continue;
            float sep;
            Vec3 n;
            if (!capsuleShapeSeparation(cap, out.position, Quat(0, 0, 0, 1), b.shape, b.position,
                                        b.orientation, sep, n))
                continue;
            const float denom = dot(dir, n);
            if (sep > 0.0f) {
                if (denom > 1e-5f) {
                    const float t = sep / denom;  // distance until the surfaces touch
                    if (t < advance) {
                        advance = t;
                        hitN = n;
                        anyHit = true;
                    }
                }
            } else if (denom > 0.0f) {
                advance = 0.0f;  // already overlapping and still moving inwards
                hitN = n;
                anyHit = true;
            }
        }
        float step = advance;
        if (anyHit) step = max2(0.0f, advance - kSkin);
        step = min2(step, maxSlice);
        out.position += dir * step;
        remaining -= dir * step;
        if (!anyHit) continue;                 // nothing ahead: keep consuming the motion
        if (advance > maxSlice) continue;      // the obstacle is still further than one slice

        // Standing on the obstacle: classify by the surface normal (obstacle -> capsule).
        const Vec3 surfaceN = -hitN;
        if (surfaceN.y >= maxSlopeCos) {
            out.grounded = true;
            out.groundNormal = surfaceN;
        } else if (surfaceN.y <= -maxSlopeCos) {
            out.hitCeiling = true;
        } else {
            out.hitWall = true;
        }
        // Slide: drop the component of the remaining motion into the surface.
        remaining -= surfaceN * dot(remaining, surfaceN);
    }
    if (lengthSq(delta) > 1e-10f) {
        out.position = depenetrateCapsule(bodies, out.position, cap);
    }
    return out;
}

// Straight descent until the capsule touches something (used to land after an auto-step).
// Unlike slideCapsule this never slides sideways, so the character settles on the step
// instead of being deflected off its edge.
struct DropOutcome {
    Vec3 position{0, 0, 0};
    bool grounded = false;
    Vec3 normal{0, 1, 0};
};

DropOutcome dropCapsule(const std::vector<Body>& bodies, Vec3 pos, const Shape& cap, float drop,
                        float maxSlopeCos) {
    const Vec3 dir(0.0f, -1.0f, 0.0f);
    float advance = drop;
    Vec3 hitN(0, 0, 0);
    bool anyHit = false;
    for (const Body& b : bodies) {
        if (!obstacleBody(b)) continue;
        float sep;
        Vec3 n;
        if (!capsuleShapeSeparation(cap, pos, Quat(0, 0, 0, 1), b.shape, b.position, b.orientation,
                                    sep, n))
            continue;
        const float denom = dot(dir, n);
        if (sep > 0.0f) {
            if (denom > 1e-5f) {
                const float t = sep / denom;
                if (t < advance) {
                    advance = t;
                    hitN = n;
                    anyHit = true;
                }
            }
        } else if (denom > 0.0f) {
            advance = 0.0f;
            hitN = n;
            anyHit = true;
        }
    }
    DropOutcome r;
    r.position = pos + dir * max2(0.0f, anyHit ? advance - kSkin : advance);
    if (anyHit) {
        const Vec3 surfaceN = -hitN;
        if (surfaceN.y >= maxSlopeCos) {
            r.grounded = true;
            r.normal = surfaceN;
        }
    }
    return r;
}

inline float horizontalProgress(Vec3 from, Vec3 to) {
    const Vec3 d = to - from;
    return std::sqrt(d.x * d.x + d.z * d.z);
}

}  // namespace

// ---------------------------------------------------------------- Shape
float Shape::volume() const {
    switch (type) {
        case ShapeType::Box: return 8.0f * halfExtents.x * halfExtents.y * halfExtents.z;
        case ShapeType::Sphere: return (4.0f / 3.0f) * PI * radius * radius * radius;
        case ShapeType::Capsule: {
            const float r2 = radius * radius;
            const float vCyl = PI * r2 * (2.0f * halfHeight);
            const float vSph = (4.0f / 3.0f) * PI * r2 * radius;
            return vCyl + vSph;
        }
    }
    return 0.0f;
}

Vec3 Shape::unitInertiaDiagonal() const {
    switch (type) {
        case ShapeType::Box: {
            const float hx = halfExtents.x, hy = halfExtents.y, hz = halfExtents.z;
            const float v = 8.0f * hx * hy * hz;
            return Vec3(v * (hy * hy + hz * hz) / 3.0f, v * (hx * hx + hz * hz) / 3.0f,
                        v * (hx * hx + hy * hy) / 3.0f);
        }
        case ShapeType::Sphere: {
            const float v = (4.0f / 3.0f) * PI * radius * radius * radius;
            const float i = 0.4f * v * radius * radius;
            return Vec3(i, i, i);
        }
        case ShapeType::Capsule: {
            // Cylinder + two hemispherical caps, all about the capsule centre (which is also
            // the centre of mass). Cylinder: I_perp = m(r^2/4 + h^2/3), I_axis = m r^2 / 2.
            // A solid hemisphere has I = (2/5) m r^2 about both a base diameter and its
            // symmetry axis; parallel axis to the capsule centre adds m(h + 3r/8)^2 for the
            // perpendicular directions (the cap centre of mass is 3r/8 off its base plane).
            const float r = radius, h = halfHeight;
            const float r2 = r * r;
            const float vCyl = PI * r2 * (2.0f * h);
            const float mCyl = vCyl;
            const float vSph = (4.0f / 3.0f) * PI * r2 * r;
            const float mSph = vSph;
            const float mHalf = 0.5f * mSph;
            const float iAxis = 0.4f * mSph * r2 + 0.5f * mCyl * r2;
            const float iPerp = mCyl * (0.25f * r2 + h * h / 3.0f) +
                                2.0f * (0.4f * mHalf * r2 + mHalf * h * h + 0.75f * mHalf * h * r);
            return Vec3(iPerp, iAxis, iPerp);
        }
    }
    return Vec3(0, 0, 0);
}

// ---------------------------------------------------------------- narrow phase
int collideShapes(const Shape& sa, Vec3 pa, Quat qa, const Shape& sb, Vec3 pb, Quat qb,
                  Vec3* outNormal, float* outPenetration, Vec3* outPoint, int maxContacts) {
    if (maxContacts <= 0) return 0;
    Manifold m;
    collideManifold(sa, pa, qa, sb, pb, qb, 0.0f, m);
    // Overlaps only, deepest first.
    int written = 0;
    float deepest = 0.0f;
    for (int i = 0; i < m.count && written < maxContacts; ++i) {
        if (m.pts[i].separation > 0.0f) continue;
        if (outPoint) outPoint[written] = m.pts[i].point;
        deepest = max2(deepest, -m.pts[i].separation);
        ++written;
    }
    if (written == 0) return 0;
    if (outNormal) *outNormal = m.normal;
    if (outPenetration) *outPenetration = deepest;
    return written;
}

// ---------------------------------------------------------------- World
World::World(const WorldConfig& cfg) : cfg_(cfg) {}

World::~World() = default;

BodyId World::addBody(const Body& body) {
    uint32_t slot = 0;
    uint32_t gen = 0;
    bool reused = false;
    // Reuse the newest free slot that is still inside the slot array and still empty: trimming
    // trailing slots in removeBody() can leave stale indices on the free list.
    while (!freeSlots_.empty()) {
        const uint32_t candidate = freeSlots_.back();
        freeSlots_.pop_back();
        if (candidate < bodies_.size() && deadSlot(bodies_[candidate])) {
            slot = candidate;
            gen = generations_[candidate];
            reused = true;
            break;
        }
    }
    if (!reused) {
        slot = uint32_t(bodies_.size());
        gen = 0u;
        bodies_.emplace_back();
        generations_.push_back(0u);
    }
    Body& b = bodies_[slot];
    b = body;
    if ((b.flags & (BODY_STATIC | BODY_KINEMATIC | BODY_ACTIVE)) == 0u) b.flags |= BODY_ACTIVE;
    if (b.isStatic()) {
        b.flags &= ~(BODY_SLEEPING | BODY_ACTIVE);
        b.flags |= BODY_STATIC;
    }
    if ((b.flags & BODY_KINEMATIC) != 0u) {
        b.flags &= ~(BODY_SLEEPING | BODY_ACTIVE | BODY_STATIC);
        b.flags |= BODY_KINEMATIC;
    }
    b.orientation = normalize(b.orientation);
    b.sleepTimer = 0.0f;
    b.force = Vec3(0, 0, 0);
    b.torque = Vec3(0, 0, 0);
    computeMassProperties(b);
    const BodyId id = makeBodyId(slot, gen);
    updateDerived(id);
    return id;
}

void World::removeBody(BodyId id) {
    const uint32_t slot = idSlot(id);
    if (slot >= bodies_.size()) return;
    if (generations_[slot] != idGeneration(id)) return;
    if (deadSlot(bodies_[slot])) return;
    Body& b = bodies_[slot];
    b = Body();          // dead marker: flags == 0, mass/invMass zero
    b.flags = 0u;
    b.mass = 0.0f;
    b.invMass = 0.0f;
    b.worldInvInertia = zeroMat3();
    b.worldBounds = Aabb{};  // mn > mx: never overlaps anything
    // Retire the slot if the generation field would wrap, otherwise make it reusable.
    if (generations_[slot] < kGenMax) {
        ++generations_[slot];
        freeSlots_.push_back(slot);
    } else {
        generations_[slot] = kGenMax;
    }
    // Trim trailing empty slots so bodyCount() stays close to the live body count.
    while (!bodies_.empty() && deadSlot(bodies_.back())) {
        bodies_.pop_back();
        if (generations_.size() > bodies_.size()) generations_.pop_back();
    }
    // Drop free slots that were trimmed away.
    while (!freeSlots_.empty() && freeSlots_.back() >= bodies_.size()) freeSlots_.pop_back();
}

Body* World::getBody(BodyId id) {
    const uint32_t slot = idSlot(id);
    if (slot >= bodies_.size() || slot >= generations_.size()) return nullptr;
    if (generations_[slot] != idGeneration(id)) return nullptr;
    if (deadSlot(bodies_[slot])) return nullptr;
    return &bodies_[slot];
}

const Body* World::getBody(BodyId id) const {
    const uint32_t slot = idSlot(id);
    if (slot >= bodies_.size() || slot >= generations_.size()) return nullptr;
    if (generations_[slot] != idGeneration(id)) return nullptr;
    if (deadSlot(bodies_[slot])) return nullptr;
    return &bodies_[slot];
}

size_t World::activeBodyCount() const {
    size_t n = 0;
    for (const Body& b : bodies_) {
        if (!deadSlot(b) && b.isActive()) ++n;
    }
    return n;
}

void World::computeMassProperties(Body& body) {
    const float v = max2(0.0f, body.shape.volume());
    const float density = body.material.density;
    if (body.isStatic() || (body.flags & BODY_KINEMATIC) != 0u) {
        body.mass = 0.0f;
        body.invMass = 0.0f;
        body.invInertiaLocal = Vec3(0, 0, 0);
        return;
    }
    body.mass = density * v;
    body.invMass = body.mass > 1e-9f ? 1.0f / body.mass : 0.0f;
    const Vec3 inertia = body.shape.unitInertiaDiagonal() * density;
    body.invInertiaLocal =
        Vec3(inertia.x > 1e-12f ? 1.0f / inertia.x : 0.0f,
             inertia.y > 1e-12f ? 1.0f / inertia.y : 0.0f,
             inertia.z > 1e-12f ? 1.0f / inertia.z : 0.0f);
}

void World::setPosition(BodyId id, Vec3 p) {
    Body* b = getBody(id);
    if (!b) return;
    b->position = p;
    wake(id);
    updateDerived(id);
}

void World::setOrientation(BodyId id, Quat q) {
    Body* b = getBody(id);
    if (!b) return;
    b->orientation = normalize(q);
    wake(id);
    updateDerived(id);
}

void World::applyImpulse(BodyId id, Vec3 impulse, Vec3 worldPoint) {
    Body* b = getBody(id);
    if (!b || !b->isDynamic()) return;
    wake(id);
    b->linearVelocity += impulse * b->invMass;
    b->angularVelocity += b->worldInvInertia * cross(worldPoint - b->position, impulse);
}

void World::applyImpulseAtCenter(BodyId id, Vec3 impulse) {
    Body* b = getBody(id);
    if (!b || !b->isDynamic()) return;
    wake(id);
    b->linearVelocity += impulse * b->invMass;
}

void World::applyForce(BodyId id, Vec3 force) {
    Body* b = getBody(id);
    if (!b || !b->isDynamic()) return;
    wake(id);
    b->force += force;
}

void World::applyTorque(BodyId id, Vec3 torque) {
    Body* b = getBody(id);
    if (!b || !b->isDynamic()) return;
    wake(id);
    b->torque += torque;
}

void World::wake(BodyId id) {
    Body* b = getBody(id);
    if (!b) return;
    b->sleepTimer = 0.0f;
    if (b->isDynamic()) {
        b->flags &= ~BODY_SLEEPING;
        b->flags |= BODY_ACTIVE;
    }
}

void World::updateDerived(BodyId id) {
    const uint32_t slot = idSlot(id);
    if (slot >= bodies_.size()) return;
    Body& b = bodies_[slot];
    if (deadSlot(b)) return;
    const Mat3 R = mat3FromQuat(b.orientation);
    b.worldInvInertia = rotateInertia(R, b.invInertiaLocal);
    b.centerOfMassWorld = b.position;  // every supported shape is centred on its local origin
    Vec3 e(0, 0, 0);
    switch (b.shape.type) {
        case ShapeType::Box: {
            const Vec3 he = b.shape.halfExtents;
            e = Vec3(std::fabs(R.c[0][0]) * he.x + std::fabs(R.c[1][0]) * he.y +
                         std::fabs(R.c[2][0]) * he.z,
                     std::fabs(R.c[0][1]) * he.x + std::fabs(R.c[1][1]) * he.y +
                         std::fabs(R.c[2][1]) * he.z,
                     std::fabs(R.c[0][2]) * he.x + std::fabs(R.c[1][2]) * he.y +
                         std::fabs(R.c[2][2]) * he.z);
            break;
        }
        case ShapeType::Sphere: e = Vec3(b.shape.radius, b.shape.radius, b.shape.radius); break;
        case ShapeType::Capsule: {
            const Vec3 axis = R.c[1];
            e = absv(axis) * b.shape.halfHeight + Vec3(b.shape.radius, b.shape.radius,
                                                        b.shape.radius);
            break;
        }
    }
    b.worldBounds.mn = b.position - e;
    b.worldBounds.mx = b.position + e;
}

void World::updateAllDerived() {
    for (size_t i = 0; i < bodies_.size(); ++i) {
        if (deadSlot(bodies_[i])) continue;
        updateDerived(makeBodyId(uint32_t(i), generations_[i]));
    }
}

void World::broadphasePairs(std::vector<std::pair<BodyId, BodyId>>& out) const {
    // `accumulator_` holds the current sub-step length (see generateContacts()).
    Broadphase::computePairs(bodies_, accumulator_, out);
}

void World::integrateVelocities(float dt) {
    for (Body& b : bodies_) {
        if (deadSlot(b) || !b.isDynamic()) {
            b.force = Vec3(0, 0, 0);
            b.torque = Vec3(0, 0, 0);
            continue;
        }
        if (!b.isSleeping()) {
            const Vec3 accel = cfg_.gravity + b.force * b.invMass;
            b.linearVelocity += accel * dt;
            b.angularVelocity += (b.worldInvInertia * b.torque) * dt;
            const float ld = max2(0.0f, b.material.linearDamping);
            const float ad = max2(0.0f, b.material.angularDamping);
            b.linearVelocity *= 1.0f / (1.0f + ld * dt);
            b.angularVelocity *= 1.0f / (1.0f + ad * dt);
            const float lv = length(b.linearVelocity);
            if (lv > kMaxLinearVelocity) b.linearVelocity *= kMaxLinearVelocity / lv;
            const float av = length(b.angularVelocity);
            if (av > kMaxAngularVelocity) b.angularVelocity *= kMaxAngularVelocity / av;
        }
        b.force = Vec3(0, 0, 0);
        b.torque = Vec3(0, 0, 0);
    }
}

void World::integratePositions(float dt) {
    for (Body& b : bodies_) {
        if (deadSlot(b) || b.isStatic() || b.isSleeping()) continue;
        b.position += b.linearVelocity * dt;
        if (lengthSq(b.angularVelocity) > 1e-12f) {
            const Quat wq(b.angularVelocity.x, b.angularVelocity.y, b.angularVelocity.z, 0.0f);
            const Quat dq = (wq * b.orientation) * (0.5f * dt);
            b.orientation = normalize(b.orientation + dq);
        }
    }
}

void World::generateContacts() {
    // `accumulator_` carries the current sub-step length (the frozen header declares
    // generateContacts()/solveVelocities() without parameters).
    const float dt = accumulator_;
    prevContacts_.swap(contacts_);
    contacts_.clear();
    Scratch& sc = scratch();
    sc.extra.clear();
    broadphasePairs(sc.pairs);

    // Contact generation runs *after* velocity integration, so every awake body currently
    // carries this sub-step's gravity increment (|g| * dt) even when it is resting. That must
    // be discounted when deciding whether a body is "moving", otherwise every awake body in a
    // pile looks like it is moving, wakes its sleeping neighbours, and the whole pile can never
    // fall asleep. A sleeping body is never woken by another sleeping body: nothing else can
    // wake it once the pile is quiet.
    const float linThr = cfg_.sleepLinearThreshold + length(cfg_.gravity) * dt;
    const float angThr = cfg_.sleepAngularThreshold;

    for (const auto& pr : sc.pairs) {
        BodyId idA = pr.first;
        BodyId idB = pr.second;
        Body& A = bodies_[idSlot(idA)];
        Body& B = bodies_[idSlot(idB)];
        const bool aSleep = A.isSleeping();
        const bool bSleep = B.isSleeping();
        if (aSleep && bSleep) continue;
        if (aSleep || bSleep) {
            const Body& mover = aSleep ? B : A;
            const bool moving = lengthSq(mover.linearVelocity) > linThr * linThr ||
                                lengthSq(mover.angularVelocity) > angThr * angThr;
            if (moving) wake(aSleep ? idA : idB);
        }

        // Speculative margin: each body contributes the distance it can travel this sub-step
        // (see Broadphase::bodyInflate). Contacts are generated for shapes that will touch
        // within that distance, which is what stops fast bodies tunnelling through thin
        // geometry. Resting bodies have ~zero inflation, so piles stay cheap.
        const float margin = min2(kSpecMarginMax, Broadphase::bodyInflate(A, dt) +
                                                    Broadphase::bodyInflate(B, dt));

        Manifold m;
        collideManifold(A.shape, A.position, A.orientation, B.shape, B.position, B.orientation,
                        margin, m);
        for (int k = 0; k < m.count; ++k) {
            Contact c;
            c.a = idA;
            c.b = idB;
            c.point = m.pts[k].point;
            c.normal = m.normal;
            c.penetration = -m.pts[k].separation;
            c.rA = c.point - A.position;
            c.rB = c.point - B.position;
            c.featureId = m.pts[k].featureId;
            contacts_.push_back(c);
            ContactExtra ex;
            ex.separation = m.pts[k].separation;
            sc.extra.push_back(ex);
        }
    }
    lastContactCount_ = contacts_.size();
}

void World::solveVelocities() {
    const float dt = accumulator_;
    Scratch& sc = scratch();
    const size_t nc = contacts_.size();
    if (nc == 0) return;
    const float invDt = dt > 0.0f ? 1.0f / dt : 0.0f;

    // Per-slot solver views (sleeping bodies act as immovable).
    sc.solver.assign(bodies_.size(), SolverBody{});
    for (size_t i = 0; i < bodies_.size(); ++i) {
        const Body& b = bodies_[i];
        if (deadSlot(b)) continue;
        SolverBody& s = sc.solver[i];
        s.movable = b.isDynamic() && !b.isSleeping();
        s.invMass = s.movable ? b.invMass : 0.0f;
        s.invInertia = s.movable ? b.worldInvInertia : zeroMat3();
    }

    // ---- per-contact solver state
    for (size_t i = 0; i < nc; ++i) {
        Contact& c = contacts_[i];
        ContactExtra& ex = sc.extra[i];
        const SolverBody& sa = sc.solver[idSlot(c.a)];
        const SolverBody& sb = sc.solver[idSlot(c.b)];
        const Mat3& ia = sa.invInertia;
        const Mat3& ib = sb.invInertia;

        float kn = sa.invMass + sb.invMass;
        kn += dot(c.normal, cross(ia * cross(c.rA, c.normal), c.rA));
        kn += dot(c.normal, cross(ib * cross(c.rB, c.normal), c.rB));
        c.effectiveMassN = kn > 1e-12f ? 1.0f / kn : 0.0f;

        ex.slotA = idSlot(c.a);
        ex.slotB = idSlot(c.b);
        contactBasis(c.normal, ex.t1, ex.t2);
        const Vec3 axes[2] = {ex.t1, ex.t2};
        for (int k = 0; k < 2; ++k) {
            float kt = sa.invMass + sb.invMass;
            kt += dot(axes[k], cross(ia * cross(c.rA, axes[k]), c.rA));
            kt += dot(axes[k], cross(ib * cross(c.rB, axes[k]), c.rB));
            c.effectiveMassT[k] = kt > 1e-12f ? 1.0f / kt : 0.0f;
        }

        // Restitution from the relative normal velocity at first contact.
        const Body& A = bodies_[idSlot(c.a)];
        const Body& B = bodies_[idSlot(c.b)];
        const Vec3 vA = A.linearVelocity + cross(A.angularVelocity, c.rA);
        const Vec3 vB = B.linearVelocity + cross(B.angularVelocity, c.rB);
        const float vn = dot(vB - vA, c.normal);
        const float e = combineRestitution(A.material, B.material);
        ex.restBias = vn < -kRestitutionThreshold ? -e * vn : 0.0f;

        // Velocity bias: Baumgarte for existing overlap (clamped), plus the speculative target
        // that lets a fast body close a gap exactly to touching without tunnelling.
        const float pen = -ex.separation;
        float baum = 0.0f;
        if (pen > cfg_.penetrationSlop) {
            baum = min2(kBaumgarteMaxVel, cfg_.baumgarte * (pen - cfg_.penetrationSlop) * invDt);
        }
        const float spec = ex.separation > 0.0f ? -ex.separation * invDt : 0.0f;
        ex.bias = max2(baum, spec);
        ex.posImpulse = 0.0f;
    }

    // ---- warm start: match (a, b, featureId) inside prevContacts_
    sc.prevOrder.resize(prevContacts_.size());
    for (size_t i = 0; i < sc.prevOrder.size(); ++i) sc.prevOrder[i] = uint32_t(i);
    const std::vector<Contact>& prev = prevContacts_;
    std::sort(sc.prevOrder.begin(), sc.prevOrder.end(), [&prev](uint32_t x, uint32_t y) {
        const Contact& cx = prev[x];
        const Contact& cy = prev[y];
        if (cx.a != cy.a) return cx.a < cy.a;
        if (cx.b != cy.b) return cx.b < cy.b;
        if (cx.featureId != cy.featureId) return cx.featureId < cy.featureId;
        return x < y;
    });
    auto keyLess = [&prev](uint32_t idx, const Contact& c) {
        const Contact& p = prev[idx];
        if (p.a != c.a) return p.a < c.a;
        if (p.b != c.b) return p.b < c.b;
        return p.featureId < c.featureId;
    };
    auto keyLessRev = [&prev](const Contact& c, uint32_t idx) {
        const Contact& p = prev[idx];
        if (c.a != p.a) return c.a < p.a;
        if (c.b != p.b) return c.b < p.b;
        return c.featureId < p.featureId;
    };
    auto pairLess = [&prev](uint32_t idx, const Contact& c) {
        const Contact& p = prev[idx];
        if (p.a != c.a) return p.a < c.a;
        return p.b < c.b;
    };
    auto pairLessRev = [&prev](const Contact& c, uint32_t idx) {
        const Contact& p = prev[idx];
        if (c.a != p.a) return c.a < p.a;
        return c.b < p.b;
    };

    for (size_t i = 0; i < nc; ++i) {
        Contact& c = contacts_[i];
        if (c.effectiveMassN <= 0.0f) continue;
        auto lo = std::lower_bound(sc.prevOrder.begin(), sc.prevOrder.end(), c, keyLess);
        auto hi = std::upper_bound(lo, sc.prevOrder.end(), c, keyLessRev);
        float bestDist = cfg_.warmStartDistance;
        const Contact* match = nullptr;
        for (auto it = lo; it != hi; ++it) {
            const Contact& p = prev[*it];
            if (dot(p.normal, c.normal) < 0.5f) continue;
            const float d = distance(p.point, c.point);
            if (d < bestDist) {
                bestDist = d;
                match = &p;
            }
        }
        if (!match) {
            // Feature ids can legitimately change between frames (the SAT reference face may
            // swap on a symmetric contact). Falling back to the nearest contact of the same
            // body pair keeps stacks warm-started instead of resetting their impulses.
            auto lo2 = std::lower_bound(sc.prevOrder.begin(), sc.prevOrder.end(), c, pairLess);
            auto hi2 = std::upper_bound(lo2, sc.prevOrder.end(), c, pairLessRev);
            for (auto it = lo2; it != hi2; ++it) {
                const Contact& p = prev[*it];
                if (dot(p.normal, c.normal) < 0.5f) continue;
                const float d = distance(p.point, c.point);
                if (d < bestDist) {
                    bestDist = d;
                    match = &p;
                }
            }
        }
        if (!match) continue;
        c.normalImpulse = match->normalImpulse;
        c.tangentImpulse[0] = match->tangentImpulse[0];
        c.tangentImpulse[1] = match->tangentImpulse[1];
        // Apply the cached impulse so the first iteration starts from the previous solution.
        const ContactExtra& ex = sc.extra[i];
        const Vec3 P = c.normal * c.normalImpulse + ex.t1 * c.tangentImpulse[0] +
                       ex.t2 * c.tangentImpulse[1];
        Body& A = bodies_[ex.slotA];
        Body& B = bodies_[ex.slotB];
        const SolverBody& sa = sc.solver[ex.slotA];
        const SolverBody& sb = sc.solver[ex.slotB];
        if (sa.movable) {
            A.linearVelocity -= P * sa.invMass;
            A.angularVelocity -= sa.invInertia * cross(c.rA, P);
        }
        if (sb.movable) {
            B.linearVelocity += P * sb.invMass;
            B.angularVelocity += sb.invInertia * cross(c.rB, P);
        }
    }

    // ---- sequential impulse iterations
    for (int iter = 0; iter < cfg_.velocityIterations; ++iter) {
        for (size_t i = 0; i < nc; ++i) {
            Contact& c = contacts_[i];
            if (c.effectiveMassN <= 0.0f) continue;
            ContactExtra& ex = sc.extra[i];
            Body& A = bodies_[ex.slotA];
            Body& B = bodies_[ex.slotB];
            const SolverBody& sa = sc.solver[ex.slotA];
            const SolverBody& sb = sc.solver[ex.slotB];

            Vec3 dv = (B.linearVelocity + cross(B.angularVelocity, c.rB)) -
                      (A.linearVelocity + cross(A.angularVelocity, c.rA));

            // Normal.
            const float vn = dot(dv, c.normal);
            const float target = max2(ex.restBias, ex.bias);
            float lambda = (target - vn) * c.effectiveMassN;
            const float oldN = c.normalImpulse;
            c.normalImpulse = max2(0.0f, oldN + lambda);
            lambda = c.normalImpulse - oldN;
            Vec3 P = c.normal * lambda;
            if (sa.movable) {
                A.linearVelocity -= P * sa.invMass;
                A.angularVelocity -= sa.invInertia * cross(c.rA, P);
            }
            if (sb.movable) {
                B.linearVelocity += P * sb.invMass;
                B.angularVelocity += sb.invInertia * cross(c.rB, P);
            }

            // Friction along two tangents, clamped to the Coulomb cone.
            const float muStatic = std::sqrt(max2(0.0f, A.material.staticFriction) *
                                             max2(0.0f, B.material.staticFriction));
            const float muDynamic = std::sqrt(max2(0.0f, A.material.dynamicFriction) *
                                              max2(0.0f, B.material.dynamicFriction));
            const Vec3 tangents[2] = {ex.t1, ex.t2};
            for (int k = 0; k < 2; ++k) {
                if (c.effectiveMassT[k] <= 0.0f) continue;
                dv = (B.linearVelocity + cross(B.angularVelocity, c.rB)) -
                     (A.linearVelocity + cross(A.angularVelocity, c.rA));
                const float vt = dot(dv, tangents[k]);
                const float mu = std::fabs(vt) > 0.01f ? muDynamic : muStatic;
                float lambdaT = -vt * c.effectiveMassT[k];
                const float maxF = mu * c.normalImpulse;
                const float oldT = c.tangentImpulse[k];
                c.tangentImpulse[k] = clamp(oldT + lambdaT, -maxF, maxF);
                lambdaT = c.tangentImpulse[k] - oldT;
                P = tangents[k] * lambdaT;
                if (sa.movable) {
                    A.linearVelocity -= P * sa.invMass;
                    A.angularVelocity -= sa.invInertia * cross(c.rA, P);
                }
                if (sb.movable) {
                    B.linearVelocity += P * sb.invMass;
                    B.angularVelocity += sb.invInertia * cross(c.rB, P);
                }
            }
        }
    }
}

void World::solvePositions() {
    Scratch& sc = scratch();
    const size_t nc = contacts_.size();
    if (nc == 0) return;
    for (int iter = 0; iter < cfg_.positionIterations; ++iter) {
        for (size_t i = 0; i < nc; ++i) {
            Contact& c = contacts_[i];
            if (c.effectiveMassN <= 0.0f) continue;
            ContactExtra& ex = sc.extra[i];
            const SolverBody& sa = sc.solver[ex.slotA];
            const SolverBody& sb = sc.solver[ex.slotB];
            const float imSum = sa.invMass + sb.invMass;
            const bool angular = sa.movable || sb.movable;
            if (imSum <= 0.0f && !angular) continue;

            // Track the separation change caused by the corrections we already applied
            // (impulse -> separation is linear: d(sep) = lambda / effectiveMassN).
            const float sepNow = ex.separation + ex.posImpulse / c.effectiveMassN;
            const float C = min2(0.0f, sepNow + cfg_.penetrationSlop);
            if (C >= 0.0f) continue;
            float lambda = -cfg_.baumgarte * C * c.effectiveMassN;
            if (lambda <= 0.0f) continue;
            // Safety clamp: never move a body by more than 5 cm in one iteration.
            if (imSum > 0.0f && lambda * imSum > 0.05f) lambda = 0.05f / imSum;
            const float oldPos = ex.posImpulse;
            ex.posImpulse = max2(0.0f, oldPos + lambda);
            lambda = ex.posImpulse - oldPos;
            if (lambda <= 0.0f) continue;

            const Vec3 P = c.normal * lambda;
            Body& A = bodies_[ex.slotA];
            Body& B = bodies_[ex.slotB];
            if (sa.movable) {
                A.position -= P * sa.invMass;
                const Vec3 dw = sa.invInertia * cross(c.rA, P);
                if (lengthSq(dw) > 1e-12f) {
                    A.orientation = normalize(A.orientation + (Quat(dw.x, dw.y, dw.z, 0.0f) *
                                                               A.orientation) * 0.5f);
                }
            }
            if (sb.movable) {
                B.position += P * sb.invMass;
                const Vec3 dw = sb.invInertia * cross(c.rB, P);
                if (lengthSq(dw) > 1e-12f) {
                    B.orientation = normalize(B.orientation + (Quat(dw.x, dw.y, dw.z, 0.0f) *
                                                               B.orientation) * 0.5f);
                }
            }
        }
    }
}

void World::updateSleep(float dt) {
    if (!cfg_.enableSleeping) {
        for (Body& b : bodies_) {
            if (deadSlot(b) || !b.isDynamic()) continue;
            b.flags &= ~BODY_SLEEPING;
            b.flags |= BODY_ACTIVE;
            b.sleepTimer = 0.0f;
        }
        return;
    }
    const float linThr2 = cfg_.sleepLinearThreshold * cfg_.sleepLinearThreshold;
    const float angThr2 = cfg_.sleepAngularThreshold * cfg_.sleepAngularThreshold;
    for (Body& b : bodies_) {
        if (deadSlot(b) || !b.isDynamic()) continue;
        if (b.isSleeping()) continue;  // already asleep; only wake() clears the flag
        const bool slow = lengthSq(b.linearVelocity) < linThr2 &&
                          lengthSq(b.angularVelocity) < angThr2;
        if (slow) {
            b.sleepTimer += dt;
            if (b.sleepTimer >= cfg_.timeToSleep) {
                b.flags |= BODY_SLEEPING;
                b.flags &= ~BODY_ACTIVE;
                b.linearVelocity = Vec3(0, 0, 0);
                b.angularVelocity = Vec3(0, 0, 0);
            }
        } else {
            b.sleepTimer = 0.0f;
            b.flags &= ~BODY_SLEEPING;
            b.flags |= BODY_ACTIVE;
        }
    }
}

void World::step(float dt) {
    if (!(dt > 0.0f)) return;
    if (dt > 0.25f) dt = 0.25f;  // clamp pathological frames
    const int substeps = max2(1, cfg_.substeps);
    const float h = dt / float(substeps);
    for (int s = 0; s < substeps; ++s) {
        accumulator_ = h;
        updateAllDerived();
        integrateVelocities(h);
        generateContacts();
        solveVelocities();
        integratePositions(h);
        updateAllDerived();   // fresh orientation-dependent data for the position pass
        solvePositions();
        updateAllDerived();   // requirement: derived data refreshed after integration
        updateSleep(h);
    }
}

// ---------------------------------------------------------------- queries
RayHit World::raycast(Vec3 origin, Vec3 dir, float maxDist,
                      const std::function<bool(const Body&)>& filter) const {
    RayHit hit;
    const float dl = length(dir);
    if (dl < 1e-9f || maxDist <= 0.0f) return hit;
    const Ray ray{origin, dir / dl};
    float bestT = maxDist;
    for (size_t i = 0; i < bodies_.size(); ++i) {
        const Body& b = bodies_[i];
        if (deadSlot(b)) continue;
        if (filter && !filter(b)) continue;
        float tAabb;
        if (!rayAabb(ray, b.worldBounds, tAabb)) continue;
        if (tAabb > bestT) continue;

        float t = 0.0f;
        Vec3 n(0, 1, 0);
        bool got = false;
        switch (b.shape.type) {
            case ShapeType::Box: {
                const Vec3 o = rotate(conjugate(b.orientation), origin - b.position);
                const Vec3 d = rotate(conjugate(b.orientation), ray.dir);
                Vec3 nl;
                got = rayObbLocal(o, d, b.shape.halfExtents, t, nl);
                if (got) n = rotate(b.orientation, nl);
                break;
            }
            case ShapeType::Sphere: {
                got = raySphere(ray, b.position, b.shape.radius, t);
                if (got) n = normalize(origin + ray.dir * t - b.position);
                break;
            }
            case ShapeType::Capsule: {
                const Vec3 axis = rotate(b.orientation, Vec3(0, 1, 0));
                got = rayCapsuleShape(origin, ray.dir, b.position - axis * b.shape.halfHeight,
                                      b.position + axis * b.shape.halfHeight, b.shape.radius, t,
                                      n);
                break;
            }
        }
        if (!got || t > bestT) continue;
        bestT = t;
        hit.hit = true;
        hit.t = t;
        hit.normal = n;
        hit.body = makeBodyId(uint32_t(i), generations_[i]);
        hit.userData = b.userData;
    }
    if (hit.hit) hit.point = origin + ray.dir * hit.t;
    return hit;
}

bool World::segmentBlocked(Vec3 a, Vec3 b, const std::function<bool(const Body&)>& filter) const {
    const Vec3 d = b - a;
    const float len = length(d);
    if (len < 1e-9f) return false;
    return raycast(a, d, len, filter).hit;
}

std::vector<BodyId> World::querySphere(Vec3 center, float radius) const {
    std::vector<BodyId> out;
    const Aabb probe{center - Vec3(radius), center + Vec3(radius)};
    for (size_t i = 0; i < bodies_.size(); ++i) {
        const Body& b = bodies_[i];
        if (deadSlot(b)) continue;
        if (!b.worldBounds.overlaps(probe)) continue;
        out.push_back(makeBodyId(uint32_t(i), generations_[i]));
    }
    return out;
}

// ---------------------------------------------------------------- character controller
World::CharacterMoveResult World::moveCapsule(Vec3 position, float radius, float halfHeight,
                                              Vec3 delta, float stepHeight,
                                              float maxSlopeCos) const {
    CharacterMoveResult res;
    const Shape cap = Shape::makeCapsule(radius, halfHeight);
    const float slopeCos = clamp(maxSlopeCos, -1.0f, 1.0f);

    // Start from a non-penetrating pose so a slightly embedded capsule cannot tunnel out.
    Vec3 start = depenetrateCapsule(bodies_, position, cap);
    SlideOutcome direct = slideCapsule(bodies_, start, cap, delta, slopeCos);
    Vec3 best = direct.position;
    float bestProgress = horizontalProgress(start, best);
    bool grounded = direct.grounded;
    Vec3 groundNormal = direct.groundNormal;
    bool hitWall = direct.hitWall;
    bool hitCeiling = direct.hitCeiling;

    // Auto-step: retry the move from `stepHeight` higher, then settle straight down onto the
    // ledge. Accepted only if it lands on a walkable surface, keeps more ground than the plain
    // slide and does not drop below the original stance.
    if (stepHeight > 1e-4f && direct.hitWall) {
        const Vec3 up = depenetrateCapsule(bodies_, start + Vec3(0, stepHeight, 0), cap);
        const SlideOutcome lifted = slideCapsule(bodies_, up, cap, delta, slopeCos);
        const DropOutcome down =
            dropCapsule(bodies_, lifted.position, cap, stepHeight + 0.05f, slopeCos);
        const Vec3 candidate = down.position;
        const float progress = horizontalProgress(start, candidate);
        if (down.grounded && candidate.y > start.y - 0.02f && progress > bestProgress + 1e-3f) {
            best = candidate;
            bestProgress = progress;
            grounded = true;
            groundNormal = down.normal;
            hitWall = false;
        }
    }

    // Ground resolution. A walkable contact during the slide means the capsule is standing on
    // something, so settle it onto that surface (this also handles sliding down a slope crest)
    // and drop the grounded flag again if the surface turned out to be out of snap reach.
    if (grounded) {
        const float snapDist = max2(0.05f, min2(stepHeight, 0.3f));
        const DropOutcome snap = dropCapsule(bodies_, best, cap, snapDist, slopeCos);
        best = snap.position;
        grounded = snap.grounded;
        if (snap.grounded) groundNormal = snap.normal;
    }

    // Ground probe: a walkable surface within the snap distance counts as standing on it.
    if (!grounded) {
        float bestSep = kGroundSnap;
        Vec3 bestN(0, 1, 0);
        for (const Body& b : bodies_) {
            if (!obstacleBody(b)) continue;
            float sep;
            Vec3 n;
            if (!capsuleShapeSeparation(cap, best, Quat(0, 0, 0, 1), b.shape, b.position,
                                        b.orientation, sep, n))
                continue;
            const Vec3 surfaceN = -n;  // obstacle -> capsule
            if (surfaceN.y < slopeCos) continue;
            if (sep < bestSep) {
                bestSep = sep;
                bestN = surfaceN;
            }
        }
        if (bestSep < kGroundSnap) {
            grounded = true;
            groundNormal = bestN;
        }
    }

    res.position = best;
    res.grounded = grounded;
    res.groundNormal = grounded ? groundNormal : Vec3(0, 1, 0);
    res.hitWall = hitWall;
    res.hitCeiling = hitCeiling;
    return res;
}

}  // namespace room2::phys
