// room2 - compact impulse-based rigid body physics.
// Designed for the scale of a single room: a few hundred dynamic bodies, static
// box geometry, plus raycasts for hitscan weapons.
#pragma once

#include <cstdint>
#include <vector>
#include <functional>
#include "../core/math.hpp"

namespace room2::phys {

// ---------------------------------------------------------------- shapes
enum class ShapeType : uint32_t { Box = 0, Sphere = 1, Capsule = 2 };

struct Shape {
    ShapeType type = ShapeType::Box;
    // Box: half extents on each local axis.
    Vec3 halfExtents{0.5f, 0.5f, 0.5f};
    // Sphere / Capsule: radius. Capsule axis is local Y.
    float radius = 0.5f;
    // Capsule: half height of the *cylindrical segment* (total height = 2*(halfHeight+radius)).
    float halfHeight = 0.0f;

    static Shape makeBox(Vec3 he) {
        Shape s;
        s.type = ShapeType::Box;
        s.halfExtents = he;
        return s;
    }
    static Shape makeSphere(float r) {
        Shape s;
        s.type = ShapeType::Sphere;
        s.radius = r;
        return s;
    }
    // Cylinder approximated as a capsule (Y axis, flat-ish ends ignored).
    static Shape makeCapsule(float r, float halfHeight) {
        Shape s;
        s.type = ShapeType::Capsule;
        s.radius = r;
        s.halfHeight = halfHeight;
        return s;
    }
    // Radius of a bounding sphere around the local origin.
    float boundingRadius() const {
        switch (type) {
            case ShapeType::Box: return length(halfExtents);
            case ShapeType::Sphere: return radius;
            case ShapeType::Capsule: return halfHeight + radius;
        }
        return 0.5f;
    }
    Aabb localAabb() const {
        Aabb b;
        switch (type) {
            case ShapeType::Box:
                b.mn = -halfExtents;
                b.mx = halfExtents;
                break;
            case ShapeType::Sphere:
                b.mn = Vec3(-radius);
                b.mx = Vec3(radius);
                break;
            case ShapeType::Capsule:
                b.mn = {-radius, -halfHeight - radius, -radius};
                b.mx = {radius, halfHeight + radius, radius};
                break;
        }
        return b;
    }
    // Volume and inertia diagonal for unit density (used to derive mass properties).
    float volume() const;
    Vec3 unitInertiaDiagonal() const;  // inertia of the shape at unit density, about center
};

// ---------------------------------------------------------------- material
struct Material {
    float restitution = 0.05f;   // 0 = inelastic, 1 = perfectly elastic
    float staticFriction = 0.6f;
    float dynamicFriction = 0.45f;
    float linearDamping = 0.03f;
    float angularDamping = 0.08f;
    float density = 1000.0f;
};

// ---------------------------------------------------------------- body
using BodyId = uint32_t;
inline constexpr BodyId INVALID_BODY = 0xFFFFFFFFu;

enum BodyFlags : uint32_t {
    BODY_STATIC = 1u << 0,
    BODY_KINEMATIC = 1u << 1,
    BODY_SLEEPING = 1u << 2,
    BODY_ACTIVE = 1u << 3,
};

struct Body {
    Shape shape;
    Material material;

    Vec3 position{0, 0, 0};
    Quat orientation{0, 0, 0, 1};
    Vec3 linearVelocity{0, 0, 0};
    Vec3 angularVelocity{0, 0, 0};

    // Accumulated forces/torques cleared each step.
    Vec3 force{0, 0, 0};
    Vec3 torque{0, 0, 0};

    float invMass = 0.0f;
    Vec3 invInertiaLocal{0, 0, 0};   // diagonal inverse inertia in body space

    uint32_t flags = BODY_ACTIVE;
    float sleepTimer = 0.0f;
    float mass = 0.0f;
    uint32_t userData = 0;            // game-side handle
    uint32_t collisionGroup = 0xFFFFFFFFu;
    uint32_t collisionMask = 0xFFFFFFFFu;

    // Cached world-space data, refreshed by World::updateDerived().
    Mat3 worldInvInertia;
    Aabb worldBounds;
    Vec3 centerOfMassWorld{0, 0, 0};   // position + rotated local COM offset (0 for our shapes)

    bool isStatic() const { return (flags & BODY_STATIC) != 0; }
    bool isDynamic() const { return !isStatic() && !(flags & BODY_KINEMATIC); }
    bool isSleeping() const { return (flags & BODY_SLEEPING) != 0; }
    bool isActive() const { return !isSleeping() && !isStatic() && !(flags & BODY_KINEMATIC); }
};

// ---------------------------------------------------------------- contacts
struct Contact {
    BodyId a = INVALID_BODY;
    BodyId b = INVALID_BODY;
    Vec3 point{0, 0, 0};        // world space, mid-way between surfaces
    Vec3 normal{0, 1, 0};       // unit, points from a towards b
    float penetration = 0.0f;   // positive when overlapping
    // Solver state
    float normalImpulse = 0.0f;
    float tangentImpulse[2] = {0, 0};
    float effectiveMassN = 0.0f;
    float effectiveMassT[2] = {0, 0};
    Vec3 rA{0, 0, 0};
    Vec3 rB{0, 0, 0};
    uint32_t featureId = 0;
};

// ---------------------------------------------------------------- raycast
struct RayHit {
    bool hit = false;
    float t = 0.0f;
    Vec3 point{0, 0, 0};
    Vec3 normal{0, 1, 0};
    BodyId body = INVALID_BODY;
    uint32_t userData = 0;
};

// ---------------------------------------------------------------- world
struct WorldConfig {
    Vec3 gravity{0.0f, -9.81f, 0.0f};
    int velocityIterations = 10;
    int positionIterations = 3;
    int substeps = 1;
    float sleepLinearThreshold = 0.045f;
    float sleepAngularThreshold = 0.08f;
    float timeToSleep = 0.55f;
    // Fraction of penetration resolved per position iteration.
    float baumgarte = 0.22f;
    float penetrationSlop = 0.0015f;
    // Contact persistence distance for warm starting.
    float warmStartDistance = 0.02f;
    bool enableSleeping = true;
};

class World {
public:
    explicit World(const WorldConfig& cfg = {});
    ~World();

    World(const World&) = delete;
    World& operator=(const World&) = delete;

    void setConfig(const WorldConfig& cfg) { cfg_ = cfg; }
    const WorldConfig& config() const { return cfg_; }

    // --- body management -------------------------------------------------
    BodyId addBody(const Body& body);
    void removeBody(BodyId id);
    Body* getBody(BodyId id);
    const Body* getBody(BodyId id) const;
    size_t bodyCount() const { return bodies_.size(); }
    // Iterate live bodies (slots may be empty after removal).
    const std::vector<Body>& bodies() const { return bodies_; }
    size_t activeBodyCount() const;

    // Recompute mass properties from shape + material density.
    static void computeMassProperties(Body& body);

    void setPosition(BodyId id, Vec3 p);
    void setOrientation(BodyId id, Quat q);
    void applyImpulse(BodyId id, Vec3 impulse, Vec3 worldPoint);
    void applyImpulseAtCenter(BodyId id, Vec3 impulse);
    void applyForce(BodyId id, Vec3 force);
    void applyTorque(BodyId id, Vec3 torque);
    void wake(BodyId id);

    // --- simulation ------------------------------------------------------
    void step(float dt);

    // --- queries ---------------------------------------------------------
    // Closest hit along the ray against all bodies (optionally filtered).
    RayHit raycast(Vec3 origin, Vec3 dir, float maxDist,
                   const std::function<bool(const Body&)>& filter = nullptr) const;
    // Convenience: true if anything blocks the segment.
    bool segmentBlocked(Vec3 a, Vec3 b,
                        const std::function<bool(const Body&)>& filter = nullptr) const;
    // All bodies whose world bounds overlap the sphere.
    std::vector<BodyId> querySphere(Vec3 center, float radius) const;

    const std::vector<Contact>& contacts() const { return contacts_; }
    // Number of contacts generated in the last step (before the solver).
    size_t lastContactCount() const { return lastContactCount_; }

    // --- character controller -------------------------------------------
    // Sweep a capsule through the static world using collide-and-slide.
    // Returns the resolved position; `groundNormalOut` is set when grounded.
    struct CharacterMoveResult {
        Vec3 position{0, 0, 0};
        bool grounded = false;
        Vec3 groundNormal{0, 1, 0};
        bool hitWall = false;
        bool hitCeiling = false;
    };
    CharacterMoveResult moveCapsule(Vec3 position, float radius, float halfHeight, Vec3 delta,
                                    float stepHeight = 0.35f, float maxSlopeCos = 0.5f) const;

private:
    friend struct Broadphase;
    void updateDerived(BodyId id);
    void updateAllDerived();
    void integrateVelocities(float dt);
    void integratePositions(float dt);
    void generateContacts();
    void solveVelocities();
    void solvePositions();
    void updateSleep(float dt);
    void broadphasePairs(std::vector<std::pair<BodyId, BodyId>>& out) const;

    WorldConfig cfg_;
    std::vector<Body> bodies_;
    std::vector<uint32_t> generations_;  // slot generation for id validation
    std::vector<BodyId> freeSlots_;
    std::vector<Contact> contacts_;
    std::vector<Contact> prevContacts_;
    size_t lastContactCount_ = 0;
    float accumulator_ = 0.0f;
};

// ---------------------------------------------------------------- collision
// Low-level narrow phase. Returns the number of contacts written (<= maxContacts).
// `normal` points from A towards B. `outPoint` is a representative world point.
int collideShapes(const Shape& sa, Vec3 pa, Quat qa, const Shape& sb, Vec3 pb, Quat qb,
                  Vec3* outNormal, float* outPenetration, Vec3* outPoint, int maxContacts = 4);

}  // namespace room2::phys
