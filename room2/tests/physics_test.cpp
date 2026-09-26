// room2 - physics module tests.
//
// Scenarios mirror the game: a sealed room with a static table, a dropped box, a box stack,
// 150 glass shards landing on the table, hitscan raycasts and a first-person capsule walking
// into a wall.  Every check prints PASS/FAIL plus the numbers behind it.
//
// Build:
//   g++ -std=c++20 -O2 -Wall -Wextra -Isrc tests/physics_test.cpp src/physics/physics.cpp
//       -o /tmp/physics_test && /tmp/physics_test
//
#include "physics/physics.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <vector>

using namespace room2;
using namespace room2::phys;

// Test-only global new/delete replacement: lets the suite prove that stepping a settled world
// performs no heap traffic (requirement: avoid per-step allocation).
static size_t g_allocations = 0;
void* operator new(size_t size) {
    ++g_allocations;
    void* p = std::malloc(size ? size : 1);
    if (!p) throw std::bad_alloc();
    return p;
}
void* operator new[](size_t size) { return ::operator new(size); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, size_t) noexcept { std::free(p); }
void operator delete[](void* p, size_t) noexcept { std::free(p); }

namespace {

constexpr float kDt = 1.0f / 120.0f;
constexpr float kFloorTop = 0.0f;
constexpr float kTableTop = 0.76f;
const Vec3 kTableHe{0.70f, 0.02f, 0.45f};
const Vec3 kTableCenter{0.0f, 0.74f, 0.0f};

int g_checks = 0;
int g_failures = 0;

std::string fmt(const char* f, ...) {
    char buf[512];
    va_list args;
    va_start(args, f);
    vsnprintf(buf, sizeof(buf), f, args);
    va_end(args);
    return std::string(buf);
}

void report(bool ok, const std::string& what, const std::string& detail) {
    ++g_checks;
    if (!ok) ++g_failures;
    std::printf("%s %-52s %s\n", ok ? "[PASS]" : "[FAIL]", what.c_str(), detail.c_str());
}

WorldConfig defaultConfig() {
    WorldConfig cfg;
    cfg.gravity = Vec3(0.0f, -9.81f, 0.0f);
    cfg.velocityIterations = 10;
    cfg.positionIterations = 3;
    cfg.substeps = 1;
    cfg.enableSleeping = true;
    return cfg;
}

BodyId addBox(World& w, Vec3 center, Vec3 half, const Material& m, uint32_t flags) {
    Body b;
    b.shape = Shape::makeBox(half);
    b.position = center;
    b.material = m;
    b.flags = flags;
    return w.addBody(b);
}

// Static room: 6 x 3 x 8 m interior, plus the table (slab at y = 0.72..0.76 and four legs).
void buildRoom(World& w) {
    Material m;
    m.restitution = 0.05f;
    m.staticFriction = 0.6f;
    m.dynamicFriction = 0.45f;

    addBox(w, Vec3(0.0f, -0.1f, 0.0f), Vec3(3.2f, 0.1f, 4.2f), m, BODY_STATIC);   // floor
    addBox(w, Vec3(0.0f, 3.1f, 0.0f), Vec3(3.2f, 0.1f, 4.2f), m, BODY_STATIC);    // ceiling
    addBox(w, Vec3(3.1f, 1.5f, 0.0f), Vec3(0.1f, 1.5f, 4.2f), m, BODY_STATIC);    // +X wall
    addBox(w, Vec3(-3.1f, 1.5f, 0.0f), Vec3(0.1f, 1.5f, 4.2f), m, BODY_STATIC);   // -X wall
    addBox(w, Vec3(0.0f, 1.5f, 4.1f), Vec3(3.2f, 1.5f, 0.1f), m, BODY_STATIC);    // +Z wall
    addBox(w, Vec3(0.0f, 1.5f, -4.1f), Vec3(3.2f, 1.5f, 0.1f), m, BODY_STATIC);   // -Z wall
    addBox(w, kTableCenter, kTableHe, m, BODY_STATIC);                            // table top
    for (int sx = -1; sx <= 1; sx += 2) {
        for (int sz = -1; sz <= 1; sz += 2) {
            addBox(w, Vec3(0.62f * float(sx), 0.36f, 0.37f * float(sz)),
                   Vec3(0.04f, 0.36f, 0.04f), m, BODY_STATIC);
        }
    }
}

// 150 glass shards (5 cm boxes) in a sheet above the table.
std::vector<BodyId> dropShards(World& w, int nx, int nz, float spacing, float height) {
    Material glass;
    glass.density = 1000.0f;
    glass.restitution = 0.08f;
    glass.staticFriction = 0.5f;
    glass.dynamicFriction = 0.35f;
    glass.linearDamping = 0.03f;
    glass.angularDamping = 0.08f;
    std::vector<BodyId> ids;
    const Vec3 half(0.025f, 0.025f, 0.025f);
    for (int ix = 0; ix < nx; ++ix) {
        for (int iz = 0; iz < nz; ++iz) {
            const float x = (float(ix) - float(nx - 1) * 0.5f) * spacing;
            const float z = (float(iz) - float(nz - 1) * 0.5f) * spacing;
            ids.push_back(addBox(w, Vec3(x, height, z), half, glass, BODY_ACTIVE));
        }
    }
    return ids;
}

// ---------------------------------------------------------------- (a) dropped box
void testDropBox() {
    World w(defaultConfig());
    buildRoom(w);
    Material m;
    Body b;
    b.shape = Shape::makeBox(Vec3(0.2f, 0.2f, 0.2f));
    b.position = Vec3(1.5f, 2.2f, 2.0f);
    b.material = m;
    const BodyId id = w.addBody(b);

    for (int i = 0; i < 600; ++i) w.step(kDt);  // 5 s
    const Body* rb = w.getBody(id);
    const float bottom = rb->worldBounds.mn.y - kFloorTop;
    const float speed = length(rb->linearVelocity);
    report(std::fabs(bottom) <= 0.005f && speed < 1e-3f, "(a) box rests on floor",
           fmt("bottom=%.5f m (tol 5 mm), speed=%.6f m/s, asleep=%d", bottom, speed,
               rb->isSleeping() ? 1 : 0));

    // Same drop without sleeping: the resting state must still be quiet (no jitter).
    WorldConfig cfg = defaultConfig();
    cfg.enableSleeping = false;
    World w2(cfg);
    buildRoom(w2);
    const BodyId id2 = w2.addBody(b);
    float maxSpeed = 0.0f;
    float minBottom = 1e9f;
    float maxBottom = -1e9f;
    for (int i = 0; i < 600; ++i) {
        w2.step(kDt);
        if (i > 360) {  // last 2 s
            const Body* p = w2.getBody(id2);
            maxSpeed = max2(maxSpeed, length(p->linearVelocity));
            minBottom = min2(minBottom, p->worldBounds.mn.y);
            maxBottom = max2(maxBottom, p->worldBounds.mn.y);
        }
    }
    report(maxSpeed < 0.05f && minBottom > -0.006f && maxBottom < 0.006f,
           "(a) no jitter/sinking (sleeping off)",
           fmt("max speed=%.5f m/s, bottom range=[%.5f, %.5f] m", maxSpeed, minBottom,
               maxBottom));
}

// ---------------------------------------------------------------- (b) box stack
void testStack() {
    World w(defaultConfig());
    buildRoom(w);
    Material m;
    std::vector<BodyId> ids;
    for (int i = 0; i < 4; ++i) {
        const float y = 0.1f + 0.2f * float(i);
        ids.push_back(addBox(w, Vec3(1.5f, y, 2.0f), Vec3(0.1f, 0.1f, 0.1f), m, BODY_ACTIVE));
    }
    for (int i = 0; i < 600; ++i) w.step(kDt);  // 5 s
    float drift = 0.0f;
    float lowest = 1e9f;
    for (size_t i = 0; i < ids.size(); ++i) {
        const Body* p = w.getBody(ids[i]);
        const float d = std::sqrt((p->position.x - 1.5f) * (p->position.x - 1.5f) +
                                  (p->position.z - 2.0f) * (p->position.z - 2.0f));
        drift = max2(drift, d);
        lowest = min2(lowest, p->position.y);
    }
    int asleep = 0;
    for (BodyId id : ids) {
        if (w.getBody(id)->isSleeping()) ++asleep;
    }
    report(drift < 0.02f && lowest > 0.08f, "(b) 4-box stack stands for 5 s",
           fmt("max horizontal drift=%.6f m, lowest centre y=%.4f, asleep=%d/4", drift, lowest,
               asleep));
}

// ---------------------------------------------------------------- (c) 150 shards on table
struct ShardResult {
    int onTable = 0;
    int asleep = 0;
    int total = 0;
    float maxPenetration = 0.0f;
    float restingSpeed = 0.0f;
    std::vector<float> finalPositions;
};

ShardResult runShardScenario(std::vector<float>* capture = nullptr) {
    World w(defaultConfig());
    buildRoom(w);
    const std::vector<BodyId> ids = dropShards(w, 15, 10, 0.075f, 1.05f);
    ShardResult r;
    r.total = int(ids.size());
    for (int step = 0; step < 960; ++step) {  // 8 s
        w.step(kDt);
        // Penetration of the table top, only for shards still over the table footprint.
        for (BodyId id : ids) {
            const Body* p = w.getBody(id);
            if (std::fabs(p->position.x) > kTableHe.x || std::fabs(p->position.z) > kTableHe.z) {
                continue;
            }
            r.maxPenetration = max2(r.maxPenetration, kTableTop - p->worldBounds.mn.y);
        }
    }
    for (BodyId id : ids) {
        const Body* p = w.getBody(id);
        const bool overTable = std::fabs(p->position.x) <= kTableHe.x &&
                               std::fabs(p->position.z) <= kTableHe.z;
        if (overTable && p->worldBounds.mn.y >= kTableTop - 0.005f) ++r.onTable;
        if (p->isSleeping()) ++r.asleep;
        r.restingSpeed = max2(r.restingSpeed, length(p->linearVelocity));
        if (capture) {
            capture->push_back(p->position.x);
            capture->push_back(p->position.y);
            capture->push_back(p->position.z);
            capture->push_back(p->orientation.x);
            capture->push_back(p->orientation.y);
            capture->push_back(p->orientation.z);
            capture->push_back(p->orientation.w);
        }
    }
    return r;
}

void testShards() {
    const ShardResult r = runShardScenario();
    report(r.onTable == r.total && r.maxPenetration < 0.005f,
           "(c) 150 shards on table, none below the top",
           fmt("on table=%d/%d, max penetration=%.6f m (tol 5 mm)", r.onTable, r.total,
               max2(0.0f, r.maxPenetration)));
    report(r.asleep == r.total, "(c) all shards sleep",
           fmt("asleep=%d/%d, max resting speed=%.5f m/s", r.asleep, r.total, r.restingSpeed));
}

// ---------------------------------------------------------------- (d) character controller
void testCharacter() {
    World w(defaultConfig());
    buildRoom(w);

    // Walk into the +Z wall: the capsule must stop short of it (interior face at z = 4).
    const float radius = 0.35f;
    const float halfHeight = 0.55f;
    const float standY = radius + halfHeight;  // resting on the floor
    const Vec3 start(0.0f, standY, 2.5f);
    const World::CharacterMoveResult wall =
        w.moveCapsule(start, radius, halfHeight, Vec3(0.0f, 0.0f, 5.0f));
    report(wall.position.z > 3.4f && wall.position.z < 3.75f,
           "(d) capsule blocked by +Z wall",
           fmt("start z=2.5 -> z=%.4f (wall face 4.0, expected ~%.2f), hitWall=%d",
               wall.position.z, 4.0f - radius, wall.hitWall ? 1 : 0));

    // Walking along the floor reports grounded with a +Y normal.
    const World::CharacterMoveResult walk =
        w.moveCapsule(start, radius, halfHeight, Vec3(0.1f, 0.0f, 0.0f));
    report(walk.grounded && walk.groundNormal.y > 0.99f, "(d) walking on floor is grounded",
           fmt("grounded=%d, normal=(%.3f, %.3f, %.3f)", walk.grounded ? 1 : 0,
               walk.groundNormal.x, walk.groundNormal.y, walk.groundNormal.z));

    // The table blocks a waist-high capsule (top at y = 0.76, no step-up that high).
    const World::CharacterMoveResult table =
        w.moveCapsule(Vec3(0.0f, standY, 1.6f), radius, halfHeight, Vec3(0.0f, 0.0f, -3.0f));
    report(table.position.z > 0.75f, "(d) capsule blocked by the table",
           fmt("z: 1.6 -> %.4f (table slab edge z=0.45 + radius)", table.position.z));

    // A 10 cm ledge is stepped over (stepHeight 0.35).
    World w2(defaultConfig());
    buildRoom(w2);
    Material m;
    addBox(w2, Vec3(0.0f, 0.05f, 1.2f), Vec3(3.0f, 0.05f, 0.5f), m, BODY_STATIC);  // 10 cm step
    // 0.707 == 45 degrees, the usual walkable-slope limit; the 10 cm ledge face is a wall.
    const World::CharacterMoveResult step =
        w2.moveCapsule(Vec3(0.0f, standY, 2.5f), radius, halfHeight, Vec3(0.0f, 0.0f, -1.0f),
                       0.35f, 0.707f);
    report(step.position.z < 1.6f && step.position.y > standY + 0.05f,
           "(d) capsule steps over a 10 cm ledge",
           fmt("z=%.4f (ledge edge 1.7, target 1.5), y=%.4f (floor %.2f + 0.10)",
               step.position.z, step.position.y, standY));

    // A capsule that starts embedded in the floor is pushed out, not ejected sideways.
    const World::CharacterMoveResult embedded =
        w.moveCapsule(Vec3(0.0f, standY - 0.12f, 2.5f), radius, halfHeight, Vec3(0.0f, 0.0f, 0.0f));
    report(embedded.position.y > standY - 0.02f, "(d) embedded capsule is depenetrated",
           fmt("y=%.4f -> %.4f", standY - 0.12f, embedded.position.y));
}

// ---------------------------------------------------------------- (e) raycasts
void testRaycast() {
    World w(defaultConfig());
    buildRoom(w);

    const RayHit down = w.raycast(Vec3(0.0f, 2.0f, 0.0f), Vec3(0.0f, -1.0f, 0.0f), 5.0f);
    const float expected = 2.0f - kTableTop;
    report(down.hit && std::fabs(down.t - expected) <= 0.001f && down.normal.y > 0.999f,
           "(e) ray hits table top",
           fmt("t=%.5f (expected %.5f), normal=(%.3f, %.3f, %.3f), point.y=%.5f", down.t, expected,
               down.normal.x, down.normal.y, down.normal.z, down.point.y));

    const RayHit floorHit = w.raycast(Vec3(2.0f, 2.0f, 0.0f), Vec3(0.0f, -1.0f, 0.0f), 5.0f);
    report(floorHit.hit && std::fabs(floorHit.t - 2.0f) < 0.001f && floorHit.normal.y > 0.999f,
           "(e) ray hits floor next to the table",
           fmt("t=%.5f, normal.y=%.3f", floorHit.t, floorHit.normal.y));

    // Straight up from y = 2 hits the ceiling (interior face at y = 3) at t = 1.
    const RayHit up = w.raycast(Vec3(0.0f, 2.0f, 0.0f), Vec3(0.0f, 1.0f, 0.0f), 5.0f);
    report(up.hit && std::fabs(up.t - 1.0f) < 0.001f && up.normal.y < -0.999f,
           "(e) upward ray hits the ceiling",
           fmt("t=%.5f (expected 1.0), normal.y=%.3f", up.t, up.normal.y));
    // A short ray that stops before the ceiling reports nothing.
    const RayHit shortRay = w.raycast(Vec3(0.0f, 2.0f, 0.0f), Vec3(0.0f, 1.0f, 0.0f), 0.5f);
    report(!shortRay.hit, "(e) ray bounded by maxDist misses",
           fmt("hit=%d t=%.3f", shortRay.hit ? 1 : 0, shortRay.t));

    const bool blocked = w.segmentBlocked(Vec3(0.0f, 2.0f, 0.0f), Vec3(0.0f, 0.5f, 0.0f));
    const bool clear = w.segmentBlocked(Vec3(2.0f, 1.0f, 3.0f), Vec3(2.0f, 1.0f, 3.5f));
    report(blocked && !clear, "(e) segmentBlocked",
           fmt("through table=%d, free space=%d", blocked ? 1 : 0, clear ? 1 : 0));

    // A shard body is hit by a ray (dynamic bodies participate in raycasts).
    Material glass;
    Body shard;
    shard.shape = Shape::makeBox(Vec3(0.05f, 0.05f, 0.05f));
    shard.position = Vec3(0.0f, 1.5f, 0.0f);
    shard.material = glass;
    const BodyId shardId = w.addBody(shard);
    const RayHit hitShard = w.raycast(Vec3(0.0f, 2.0f, 0.0f), Vec3(0.0f, -1.0f, 0.0f), 5.0f);
    report(hitShard.hit && hitShard.body == shardId && std::fabs(hitShard.t - 0.45f) < 0.002f,
           "(e) ray hits a dynamic shard first",
           fmt("t=%.5f (expected 0.45), body=%u (shard=%u)", hitShard.t, hitShard.body, shardId));

    const std::vector<BodyId> near = w.querySphere(Vec3(0.0f, 0.76f, 0.0f), 0.05f);
    report(!near.empty(), "(e) querySphere finds the table top",
           fmt("bodies=%zu", near.size()));
}

// ---------------------------------------------------------------- (f) determinism
void testDeterminism() {
    std::vector<float> a, b;
    runShardScenario(&a);
    runShardScenario(&b);
    const bool same = a.size() == b.size() && !a.empty() &&
                      std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
    float maxDiff = 0.0f;
    if (a.size() == b.size()) {
        for (size_t i = 0; i < a.size(); ++i) maxDiff = max2(maxDiff, std::fabs(a[i] - b[i]));
    }
    report(same, "(f) determinism: two identical runs are bit-identical",
           fmt("floats compared=%zu, max |delta|=%.9f", a.size(), maxDiff));
}

// ---------------------------------------------------------------- (g) mass properties
void testInertia() {
    // Box and sphere are closed form; verify against the textbook expressions.
    const Shape box = Shape::makeBox(Vec3(0.2f, 0.3f, 0.4f));
    const float vBox = 8.0f * 0.2f * 0.3f * 0.4f;
    const Vec3 iBox = box.unitInertiaDiagonal();
    const float ex = vBox * (0.3f * 0.3f + 0.4f * 0.4f) / 3.0f;
    const float ey = vBox * (0.2f * 0.2f + 0.4f * 0.4f) / 3.0f;
    report(std::fabs(box.volume() - vBox) < 1e-6f && std::fabs(iBox.x - ex) < 1e-6f &&
               std::fabs(iBox.y - ey) < 1e-6f,
           "(g) box volume/inertia",
           fmt("V=%.6f I=(%.6f, %.6f, %.6f)", box.volume(), iBox.x, iBox.y, iBox.z));

    const Shape sph = Shape::makeSphere(0.25f);
    const float vSph = (4.0f / 3.0f) * PI * 0.25f * 0.25f * 0.25f;
    const Vec3 iSph = sph.unitInertiaDiagonal();
    report(std::fabs(sph.volume() - vSph) < 1e-6f &&
               std::fabs(iSph.x - 0.4f * vSph * 0.0625f) < 1e-6f,
           "(g) sphere volume/inertia", fmt("V=%.6f I=%.6f", sph.volume(), iSph.x));

    // Capsule: numeric integration (midpoint rule) over the exact capsule volume.
    const float r = 0.3f, h = 0.4f;
    const Shape cap = Shape::makeCapsule(r, h);
    const int n = 96;
    const Vec3 lo(-r, -h - r, -r);
    const Vec3 hi(r, h + r, r);
    const Vec3 cell((hi.x - lo.x) / float(n), (hi.y - lo.y) / float(n), (hi.z - lo.z) / float(n));
    const float cellV = cell.x * cell.y * cell.z;
    double vol = 0.0, ix = 0.0, iy = 0.0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                const float x = lo.x + (float(i) + 0.5f) * cell.x;
                const float y = lo.y + (float(j) + 0.5f) * cell.y;
                const float z = lo.z + (float(k) + 0.5f) * cell.z;
                const float ay = std::fabs(y);
                const float rad = ay <= h ? r : std::sqrt(max2(0.0f, r * r - (ay - h) * (ay - h)));
                if (x * x + z * z > rad * rad) continue;
                vol += cellV;
                ix += (double(y) * y + double(z) * z) * cellV;
                iy += (double(x) * x + double(z) * z) * cellV;
            }
        }
    }
    const Vec3 ana = cap.unitInertiaDiagonal();
    const float relV = std::fabs(float(vol) - cap.volume()) / cap.volume();
    const float relX = std::fabs(float(ix) - ana.x) / ana.x;
    const float relY = std::fabs(float(iy) - ana.y) / ana.y;
    report(relV < 0.02f && relX < 0.03f && relY < 0.03f, "(g) capsule volume/inertia vs numeric",
           fmt("V err=%.3f%%, I_perp err=%.3f%% (%.6f vs %.6f), I_axis err=%.3f%%", relV * 100.0f,
               relX * 100.0f, ana.x, float(ix), relY * 100.0f));

    // computeMassProperties wiring: mass = density * volume, inverse inertia diagonal.
    Body b;
    b.shape = Shape::makeBox(Vec3(0.05f, 0.05f, 0.05f));
    b.material.density = 1000.0f;
    World::computeMassProperties(b);
    const float expectedMass = 1000.0f * 8.0f * 0.05f * 0.05f * 0.05f;
    Body stat;
    stat.shape = Shape::makeBox(Vec3(1.0f, 1.0f, 1.0f));
    stat.flags = BODY_STATIC;
    World::computeMassProperties(stat);
    Body kin;
    kin.shape = Shape::makeSphere(0.5f);
    kin.flags = BODY_KINEMATIC;
    World::computeMassProperties(kin);
    const bool massOk = std::fabs(b.mass - expectedMass) < 1e-6f &&
                        std::fabs(b.invMass - 1.0f / expectedMass) < 1e-3f;
    const bool staticOk = stat.invMass == 0.0f && stat.invInertiaLocal.x == 0.0f;
    const bool kinematicOk = kin.invMass == 0.0f && kin.invInertiaLocal.y == 0.0f;
    report(massOk && staticOk && kinematicOk,
           "(g) computeMassProperties: dynamic/static/kinematic",
           fmt("m=%.5f kg invM=%.3f static invM=%.1f kinematic invI=%.1f", b.mass, b.invMass,
               stat.invMass, kin.invInertiaLocal.y));
}

// ---------------------------------------------------------------- (h) narrow phase
void testNarrowPhase() {
    const Shape boxA = Shape::makeBox(Vec3(0.5f, 0.5f, 0.5f));
    const Shape boxB = Shape::makeBox(Vec3(0.5f, 0.5f, 0.5f));
    Vec3 n(0, 0, 0), pts[4];
    float pen = 0.0f;
    const Quat id(0, 0, 0, 1);
    // B sits 0.9 above A: 0.1 m of overlap on Y, expect a 4-point face manifold.
    const int c1 = collideShapes(boxA, Vec3(0, 0, 0), id, boxB, Vec3(0, 0.9f, 0), id, &n, &pen,
                                 pts, 4);
    report(c1 == 4 && std::fabs(n.y - 1.0f) < 1e-4f && std::fabs(pen - 0.1f) < 1e-4f,
           "(h) box/box face manifold",
           fmt("contacts=%d normal=(%.3f,%.3f,%.3f) penetration=%.5f", c1, n.x, n.y, n.z, pen));

    const int none = collideShapes(boxA, Vec3(0, 0, 0), id, boxB, Vec3(0, 1.5f, 0), id, &n, &pen,
                                   pts, 4);
    report(none == 0, "(h) separated boxes produce no contacts", fmt("contacts=%d", none));

    // Sphere above box: normal A -> B is +Y, penetration = r - gap.
    const Shape sph = Shape::makeSphere(0.25f);
    const int c2 = collideShapes(boxA, Vec3(0, 0, 0), id, sph, Vec3(0, 0.7f, 0), id, &n, &pen, pts,
                                 4);
    report(c2 == 1 && n.y > 0.999f && std::fabs(pen - 0.05f) < 1e-4f, "(h) box/sphere",
           fmt("contacts=%d normal.y=%.4f penetration=%.5f", c2, n.y, pen));

    // Capsule lying on a box face: expect support points, normal +Y.
    const Shape cap = Shape::makeCapsule(0.1f, 0.3f);
    const Quat rz = quatFromAxisAngle(Vec3(0, 0, 1), HALF_PI);  // capsule axis -> X
    const int c3 = collideShapes(boxA, Vec3(0, 0, 0), id, cap, Vec3(0, 0.58f, 0), rz, &n, &pen,
                                 pts, 4);
    report(c3 >= 2 && n.y > 0.99f && pen > 0.0f, "(h) box/capsule manifold",
           fmt("contacts=%d normal.y=%.4f penetration=%.5f", c3, n.y, pen));

    // Reversed order must flip the normal.
    const int c4 = collideShapes(sph, Vec3(0, 0.7f, 0), id, boxA, Vec3(0, 0, 0), id, &n, &pen, pts,
                                 4);
    report(c4 == 1 && n.y < -0.999f, "(h) reversed order flips the normal",
           fmt("contacts=%d normal.y=%.4f", c4, n.y));
}

// ---------------------------------------------------------------- (i) tunnelling
void testTunnelling() {
    // 5 cm shard fired downwards at 12 m/s at the 4 cm thick table top: 10 cm per step, so
    // discrete collision detection alone would pass straight through it.
    for (float speed : {5.0f, 12.0f}) {
        World w(defaultConfig());
        buildRoom(w);
        Body b;
        b.shape = Shape::makeBox(Vec3(0.025f, 0.025f, 0.025f));
        b.position = Vec3(0.0f, 1.2f, 0.0f);
        b.linearVelocity = Vec3(0.0f, -speed, 0.0f);
        const BodyId id = w.addBody(b);
        for (int i = 0; i < 240; ++i) w.step(kDt);
        const Body* p = w.getBody(id);
        const bool onOrAbove = p->worldBounds.mn.y > kTableTop - 0.005f;
        const bool fellThroughFloor = p->worldBounds.mn.y < kFloorTop - 0.005f;
        report(onOrAbove && !fellThroughFloor, fmt("(i) no tunnelling at %.0f m/s", speed),
               fmt("bottom=%.5f m (table top %.2f), y=%.4f, speed=%.4f", p->worldBounds.mn.y,
                   kTableTop, p->position.y, length(p->linearVelocity)));
    }
}

// ---------------------------------------------------------------- (j) bodies/ids
void testBodyIds() {
    World w(defaultConfig());
    Material m;
    const BodyId a = addBox(w, Vec3(0, 5, 0), Vec3(0.1f, 0.1f, 0.1f), m, BODY_ACTIVE);
    const BodyId b = addBox(w, Vec3(0, 5, 1), Vec3(0.1f, 0.1f, 0.1f), m, BODY_ACTIVE);
    w.removeBody(a);
    const BodyId c = addBox(w, Vec3(0, 5, 2), Vec3(0.1f, 0.1f, 0.1f), m, BODY_ACTIVE);
    const bool stale = w.getBody(a) == nullptr;
    const bool fresh = w.getBody(b) != nullptr && w.getBody(c) != nullptr;
    report(stale && fresh && c != a, "(j) stale ids never resolve after recycling",
           fmt("recycled id reused=%d, live bodies reachable=%d, count=%zu", c == a ? 1 : 0,
               fresh ? 1 : 0, w.bodyCount()));

    // Churn: many add/remove cycles must never hand out an id that resolves to a live body
    // (this also stresses the free list + trailing-slot trimming under the sanitizers).
    size_t liveBefore = 0;
    for (const Body& b : w.bodies()) {
        if (b.flags != 0) ++liveBefore;
    }
    std::vector<BodyId> live;
    uint32_t state = 12345u;
    int staleHits = 0;
    int liveMismatch = 0;
    for (int i = 0; i < 400; ++i) {
        state = state * 1664525u + 1013904223u;  // deterministic LCG, no library RNG
        const bool doAdd = live.empty() || ((state >> 16) & 3u) != 0u;
        if (doAdd) {
            Body nb;
            nb.shape = Shape::makeSphere(0.05f);
            nb.position = Vec3(float(state % 7u) * 0.1f, 2.0f, float(state % 11u) * 0.1f);
            live.push_back(w.addBody(nb));
        } else {
            const size_t k = (state >> 8) % live.size();
            const BodyId victim = live[k];
            w.removeBody(victim);
            if (w.getBody(victim) != nullptr) ++staleHits;
            live[k] = live.back();
            live.pop_back();
        }
    }
    for (BodyId id : live) {
        if (w.getBody(id) == nullptr) ++liveMismatch;
    }
    // Removed ids must be gone (never recycled to a different body).
    size_t reachable = 0;
    for (const Body& b : w.bodies()) {
        if (b.flags != 0) ++reachable;
    }
    report(staleHits == 0 && liveMismatch == 0 && reachable == liveBefore + live.size(),
           "(j) 400 add/remove cycles keep ids exact",
           fmt("stale resolved=%d, live missing=%d, slots live=%zu (expected %zu)", staleHits,
               liveMismatch, reachable, liveBefore + live.size()));

    // Sleeping bodies wake on impulse, and a sleeping body is never woken by another sleeper.
    WorldConfig cfg = defaultConfig();
    World w2(cfg);
    buildRoom(w2);
    const BodyId s1 = addBox(w2, Vec3(1.5f, 0.1f, 2.0f), Vec3(0.1f, 0.1f, 0.1f), m, BODY_ACTIVE);
    const BodyId s2 = addBox(w2, Vec3(1.5f, 0.3f, 2.0f), Vec3(0.1f, 0.1f, 0.1f), m, BODY_ACTIVE);
    for (int i = 0; i < 600; ++i) w2.step(kDt);
    const bool bothAsleep = w2.getBody(s1)->isSleeping() && w2.getBody(s2)->isSleeping();
    w2.applyImpulseAtCenter(s1, Vec3(0.5f, 0.0f, 0.0f));
    const bool woke = !w2.getBody(s1)->isSleeping();
    report(bothAsleep && woke && w2.activeBodyCount() >= 1, "(j) sleeping + impulse wake",
           fmt("slept=%d, woke on impulse=%d, active=%zu", bothAsleep ? 1 : 0, woke ? 1 : 0,
               w2.activeBodyCount()));
}

// ---------------------------------------------------------------- (l) substepping
void testSubsteps() {
    WorldConfig cfg = defaultConfig();
    cfg.substeps = 4;   // 4 x 1/480 s sub-steps per 1/120 s frame
    World w(cfg);
    buildRoom(w);
    const std::vector<BodyId> ids = dropShards(w, 15, 10, 0.075f, 1.05f);

    float maxPenetration = 0.0f;
    for (int step = 0; step < 480; ++step) {  // 4 s
        w.step(kDt);
        for (BodyId id : ids) {
            const Body* p = w.getBody(id);
            if (std::fabs(p->position.x) > kTableHe.x || std::fabs(p->position.z) > kTableHe.z) {
                continue;
            }
            maxPenetration = max2(maxPenetration, kTableTop - p->worldBounds.mn.y);
        }
    }
    int onTable = 0;
    int asleep = 0;
    for (BodyId id : ids) {
        const Body* p = w.getBody(id);
        if (p->worldBounds.mn.y >= kTableTop - 0.005f) ++onTable;
        if (p->isSleeping()) ++asleep;
    }
    report(onTable == int(ids.size()) && maxPenetration < 0.005f,
           "(l) substeps=4 keeps 150 shards on the table",
           fmt("on table=%d/%zu, max penetration=%.6f m, asleep=%d", onTable, ids.size(),
               max2(0.0f, maxPenetration), asleep));
}

// ---------------------------------------------------------------- (n) glass burst
// The game scenario: a drinking glass on the table is shot and shatters into ~120 shards that
// tumble, bounce, slide off the table edge and land on the floor. Nothing may end up embedded
// in the static geometry, and the whole pile must come to rest.
void testGlassBurst() {
    World w(defaultConfig());
    buildRoom(w);
    std::vector<BodyId> shards = dropShards(w, 12, 10, 0.06f, 0.82f);  // 120 shards

    // Kick every shard outwards from the glass centre, like a bullet impact would.
    for (size_t i = 0; i < shards.size(); ++i) {
        const Body* p = w.getBody(shards[i]);
        Vec3 dir(p->position.x, 0.35f, p->position.z);
        for (int c = 0; c < 3; ++c) {
            // Deterministic per-shard variation from the index (no RNG).
            const uint32_t h = uint32_t(i) * 2654435761u + uint32_t(c) * 40503u;
            (&dir.x)[c] += float((h >> 13) & 7u) * 0.03f;
        }
        dir = normalize(dir);
        const float speed = 1.6f + 0.6f * float(i % 5u) * 0.25f;
        w.applyImpulseAtCenter(shards[i], dir * (speed * w.getBody(shards[i])->mass));
    }

    float minBottom = 1e9f;
    float maxSpeed = 0.0f;
    for (int step = 0; step < 720; ++step) {  // 6 s
        w.step(kDt);
        for (BodyId id : shards) {
            const Body* p = w.getBody(id);
            minBottom = min2(minBottom, p->worldBounds.mn.y);
            maxSpeed = max2(maxSpeed, length(p->linearVelocity));
        }
    }

    int embedded = 0;
    int asleep = 0;
    int onFloor = 0;
    for (BodyId id : shards) {
        const Body* p = w.getBody(id);
        if (p->isSleeping()) ++asleep;
        if (p->worldBounds.mn.y < 0.02f) ++onFloor;   // came to rest on the room floor
        for (const Body& s : w.bodies()) {            // static geometry: shards() slot order
            if (s.flags != BODY_STATIC) continue;
            const Aabb shrunk{s.worldBounds.mn + Vec3(0.005f), s.worldBounds.mx - Vec3(0.005f)};
            if (shrunk.contains(p->position)) ++embedded;
        }
    }
    report(embedded == 0 && minBottom > -0.005f, "(n) glass burst never sinks into geometry",
           fmt("shards embedded in statics=%d, lowest shard bottom=%.5f m, peak speed=%.2f m/s",
               embedded, minBottom, maxSpeed));
    report(asleep == int(shards.size()), "(n) burst shards all come to rest",
           fmt("asleep=%d/%zu after 6 s, shards on the floor=%d", asleep, shards.size(),
               onFloor));
}

// ---------------------------------------------------------------- (m) forces and torques
void testForcesAndTorques() {
    WorldConfig cfg = defaultConfig();
    cfg.gravity = Vec3(0, 0, 0);      // isolate the integrator from contact forces
    cfg.enableSleeping = false;
    World w(cfg);
    Material m;
    m.linearDamping = 0.0f;
    m.angularDamping = 0.0f;
    Body b;
    b.shape = Shape::makeBox(Vec3(0.1f, 0.1f, 0.1f));
    b.material = m;
    const BodyId id = w.addBody(b);
    const float mass = w.getBody(id)->mass;
    const float inertia = w.getBody(id)->mass * (0.01f + 0.01f) / 3.0f;

    for (int i = 0; i < 120; ++i) {  // 1 s of a constant 10 N push (+X) and 1 Nm spin (+Y)
        w.applyForce(id, Vec3(10.0f, 0.0f, 0.0f));
        w.applyTorque(id, Vec3(0.0f, 1.0f, 0.0f));
        w.step(kDt);
    }
    const Body* p = w.getBody(id);
    const float expectedV = 10.0f * 1.0f / mass;
    const float expectedW = 1.0f * 1.0f / inertia;
    const bool vOk = std::fabs(p->linearVelocity.x - expectedV) < 0.02f * expectedV;
    const bool wOk = std::fabs(p->angularVelocity.y - expectedW) < 0.02f * expectedW;
    // Forces must be cleared every step: after this step with no force, velocity is unchanged.
    const float vBefore = p->linearVelocity.x;
    w.step(kDt);
    const bool cleared = std::fabs(w.getBody(id)->linearVelocity.x - vBefore) < 1e-5f;
    report(vOk && wOk && cleared, "(m) applyForce/applyTorque integrate and clear",
           fmt("v=%.5f (expect %.5f), w=%.4f (expect %.4f), forces cleared=%d",
               p->linearVelocity.x, expectedV, p->angularVelocity.y, expectedW,
               cleared ? 1 : 0));

    // Impulse at a point produces the expected linear and angular kick.
    World w2(cfg);
    Body b2;
    b2.shape = Shape::makeBox(Vec3(0.1f, 0.1f, 0.1f));
    b2.material = m;
    const BodyId id2 = w2.addBody(b2);
    const float mass2 = w2.getBody(id2)->mass;
    w2.applyImpulse(id2, Vec3(0.0f, 1.0f, 0.0f), Vec3(0.1f, 0.0f, 0.0f));  // off-centre
    const Body* p2 = w2.getBody(id2);
    const bool linOk = std::fabs(p2->linearVelocity.y - 1.0f / mass2) < 1e-5f;
    const bool angOk = std::fabs(p2->angularVelocity.z) > 1e-3f;  // torque about Z
    report(linOk && angOk, "(m) applyImpulse at a point spins the body",
           fmt("v.y=%.5f (expect %.5f), w.z=%.4f", p2->linearVelocity.y, 1.0f / mass2,
               p2->angularVelocity.z));
}

// ---------------------------------------------------------------- (k) performance
struct StepStats {
    double mean = 0.0;
    double p95 = 0.0;
    double worst = 0.0;
    size_t contacts = 0;
};

StepStats measureSteps(World& w, int warmup, int steps) {
    for (int i = 0; i < warmup; ++i) w.step(kDt);
    std::vector<double> samples;
    samples.reserve(size_t(steps));
    for (int i = 0; i < steps; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        w.step(kDt);
        const auto t1 = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    StepStats s;
    s.contacts = w.lastContactCount();
    for (double v : samples) s.mean += v;
    s.mean /= double(samples.size());
    std::sort(samples.begin(), samples.end());
    s.p95 = samples[size_t(double(samples.size() - 1) * 0.95)];
    s.worst = samples.back();
    return s;
}

void testPerformance() {
    // 260 dynamic bodies: 150 shards on the table, 100 piled on the floor, 10 crates.
    auto buildScene = [](World& w) {
        buildRoom(w);
        dropShards(w, 15, 10, 0.075f, 1.05f);
        dropShards(w, 10, 10, 0.075f, 0.40f);
        Material m;
        for (int i = 0; i < 10; ++i) {
            addBox(w, Vec3(-2.2f + 0.4f * float(i), 1.5f + 0.05f * float(i), 3.0f),
                   Vec3(0.15f, 0.15f, 0.15f), m, BODY_ACTIVE);
        }
    };

#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
    const double budget = 40.0;  // sanitizers cost ~10-30x; the shape of the result is what counts
    const char* note = " [sanitizer build: relaxed budget]";
#else
    const double budget = 2.0;
    const char* note = "";
#endif

    // (A) game-like: sleeping enabled, pile settled.
    WorldConfig cfg = defaultConfig();
    World w(cfg);
    buildScene(w);
    const StepStats settled = measureSteps(w, 240, 240);
    report(settled.mean < budget && settled.p95 < budget, "(k) step time, 260 bodies (awake pile)",
           fmt("mean=%.4f ms, p95=%.4f ms, worst=%.4f ms (%.0f bodies live)%s", settled.mean,
               settled.p95, settled.worst, double(w.bodyCount()), note));

    // (C) no heap traffic once the scratch buffers have grown.
    g_allocations = 0;
    for (int i = 0; i < 120; ++i) w.step(kDt);
    const size_t allocations = g_allocations;
    report(allocations <= 2, "(k) steady-state stepping is allocation free",
           fmt("heap allocations over 120 steps (271 bodies)=%zu", allocations));

    // (B) worst case: sleeping off, so every body stays awake and keeps generating contacts.
    WorldConfig cfg2 = defaultConfig();
    cfg2.enableSleeping = false;
    World w2(cfg2);
    buildScene(w2);
    const StepStats busy = measureSteps(w2, 60, 240);
    report(busy.mean < budget && busy.p95 < budget, "(k) step time, worst case (all awake)",
           fmt("mean=%.4f ms, p95=%.4f ms, worst=%.4f ms, contacts=%zu%s", busy.mean, busy.p95,
               busy.worst, busy.contacts, note));
}

}  // namespace

int main() {
    std::printf("room2 physics tests (dt = 1/120 s)\n");
    std::printf("---------------------------------------------------------------\n");
    testDropBox();
    testStack();
    testShards();
    testCharacter();
    testRaycast();
    testDeterminism();
    testInertia();
    testNarrowPhase();
    testTunnelling();
    testSubsteps();
    testGlassBurst();
    testForcesAndTorques();
    testBodyIds();
    testPerformance();
    std::printf("---------------------------------------------------------------\n");
    std::printf("%d checks, %d failures -> %s\n", g_checks, g_failures,
                g_failures == 0 ? "ALL TESTS PASSED" : "FAILURES PRESENT");
    return g_failures == 0 ? 0 : 1;
}
