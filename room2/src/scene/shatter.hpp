// room2 - glass fracture.
#pragma once

#include <cstdint>
#include <vector>

#include "core/math.hpp"
#include "core/rng.hpp"
#include "scene/builders.hpp"
#include "scene/scene.hpp"

namespace room2::scene {

// A single shard produced by the fracture, in world space.
struct ShardDesc {
    uint32_t mesh = kInvalidIndex;
    Vec3 position{0, 0, 0};
    Quat orientation{0, 0, 0, 1};
    Vec3 halfExtents{0.01f, 0.01f, 0.001f};   // box approximation for collision
    Vec3 linearVelocity{0, 0, 0};
    Vec3 angularVelocity{0, 0, 0};
    float mass = 0.005f;
};

struct ShatterParams {
    // The intact glass volume.
    Vec3 center{0, 0, 0};
    float radius = 0.04f;
    float height = 0.118f;
    float wallThickness = 0.0055f;
    // Impact point and direction in world space (direction points along the shot).
    Vec3 impactPoint{0, 0, 0};
    Vec3 impactDirection{0, 0, -1};
    float impactEnergy = 1.0f;
    int shardCount = 110;
    uint32_t seed = 12345;
    // Physics material.
    float density = 2500.0f;   // soda-lime glass
};

// Fractures the glass into `params.shardCount` shards. Meshes are appended to `scene`
// and the shard descriptions are returned for the physics world to consume. The
// distribution is physically motivated: the impact site produces many small, fast
// fragments while the far side breaks into fewer, larger pieces.
std::vector<ShardDesc> shatterGlass(Scene& scene, const SceneMaterials& materials,
                                    const ShatterParams& params);

// A rough estimate of the mass of one shard, for audio and impulse scaling.
float estimateShardMass(Vec3 halfExtents, float density);

}  // namespace room2::scene
