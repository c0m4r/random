#include "scene/shatter.hpp"

#include <algorithm>

#include "core/log.hpp"
#include "core/rng.hpp"
#include "procgen/mesh.hpp"

namespace room2::scene {

float estimateShardMass(Vec3 halfExtents, float density) {
    // Treat the shard as a box at 65% packing, which is close to a fractured wedge.
    const float volume = 8.0f * halfExtents.x * halfExtents.y * halfExtents.z * 0.65f;
    return std::max(volume * density, 1e-5f);
}

std::vector<ShardDesc> shatterGlass(Scene& scene, const SceneMaterials& materials,
                                    const ShatterParams& params) {
    std::vector<ShardDesc> shards;
    shards.reserve(static_cast<size_t>(params.shardCount));

    Rng rng(params.seed);
    const Vec3 toImpact = params.impactPoint - params.center;
    const float impactHeight = clamp(toImpact.y / std::max(params.height * 0.5f, 1e-4f), -1.0f, 1.0f);
    // Azimuth of the impact around the glass axis (atan2 over x/z).
    const float impactAzimuth = std::atan2(toImpact.x, toImpact.z);
    const Vec3 shotDir = normalize(params.impactDirection);
    const float energy = clamp(params.impactEnergy, 0.15f, 4.0f);

    for (int i = 0; i < params.shardCount; ++i) {
        // --- where on the shell this shard comes from -----------------------
        // Bias the sampling towards the impact azimuth and height so the hole is where
        // the bullet went in.
        float azimuth;
        float heightT;
        if (rng.chance(0.62f)) {
            // Near the impact: concentrated, small pieces.
            azimuth = impactAzimuth + rng.gaussian(0.0f, 0.85f);
            heightT = clamp(impactHeight + rng.gaussian(0.0f, 0.45f), -1.0f, 1.0f);
        } else {
            azimuth = rng.range(-PI, PI);
            heightT = rng.range(-1.0f, 1.0f);
        }
        const float y = params.center.y + heightT * params.height * 0.5f;
        const float rOuter = params.radius * (1.0f - 0.05f * std::fabs(heightT));
        const float r = rOuter - params.wallThickness * 0.5f;

        // Distance from the impact, used to size the piece.
        float dAz = azimuth - impactAzimuth;
        while (dAz > PI) dAz -= TWO_PI;
        while (dAz < -PI) dAz += TWO_PI;
        const float surfaceDistance =
            std::sqrt(dAz * dAz * r * r + (heightT - impactHeight) * (heightT - impactHeight) *
                                              params.height * params.height * 0.25f);
        // Small near the impact, growing with distance (a classic fracture gradient).
        const float sizeScale = clamp(0.28f + surfaceDistance / 0.16f, 0.28f, 1.8f);
        const float baseSize = rng.range(0.019f, 0.040f) * sizeScale;
        const float thickness = params.wallThickness * rng.range(0.75f, 1.15f);

        Vec3 position(params.center.x + std::sin(azimuth) * r, y,
                      params.center.z + std::cos(azimuth) * r);
        // The shard's flat face is tangent to the shell, so its normal points radially.
        const Vec3 outward(std::sin(azimuth), 0.0f, std::cos(azimuth));
        Quat orientation = quatBetween(Vec3(0, 0, 1), outward);
        // Random spin about the surface normal keeps the debris from looking stamped.
        orientation = orientation * quatFromAxisAngle(Vec3(0, 0, 1), rng.range(-PI, PI));

        const float size = std::max(baseSize, 0.008f);
        uint32_t mesh = buildGlassShard(scene, materials, Rng(params.seed + static_cast<uint32_t>(i) * 7919u),
                                        size, thickness);
        if (mesh == kInvalidIndex) continue;

        ShardDesc shard;
        shard.mesh = mesh;
        shard.position = position;
        shard.orientation = orientation;
        shard.halfExtents = Vec3(size * 0.5f, size * 0.32f, thickness * 0.5f);

        // --- velocity -------------------------------------------------------
        // Radial burst from the impact plus the bullet's momentum, scaled down for the
        // small fragments and biased away from the impact point.
        const Vec3 fromImpact = position - params.impactPoint;
        Vec3 burst = normalize(fromImpact);
        if (lengthSq(fromImpact) < 1e-8f) burst = outward;
        // Small fragments fly a little faster than large ones, but the range is bounded:
        // a real tumbler throws its contents across the table, not across the room.
        const float burstSpeed = clamp((0.9f + rng.range(0.0f, 1.1f)) * energy *
                                           (0.65f + 0.35f / sizeScale),
                                       0.25f, 2.9f);
        Vec3 velocity = burst * burstSpeed + shotDir * (0.55f * energy);
        velocity += Vec3(rng.symmetric(), rng.symmetric() * 0.45f + 0.35f, rng.symmetric()) * 0.55f;
        shard.linearVelocity = velocity;
        shard.angularVelocity = Vec3(rng.symmetric(), rng.symmetric(), rng.symmetric()) *
                                (8.0f + 14.0f * energy / std::max(sizeScale, 0.5f));
        shard.mass = estimateShardMass(shard.halfExtents, params.density);
        shards.push_back(shard);
    }

    R2_INFO("glass fractured into ", shards.size(), " shards (energy ", energy, ")");
    return shards;
}

}  // namespace room2::scene
