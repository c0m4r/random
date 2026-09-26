// room2 - procedural construction of the room contents.
#pragma once

#include <cstdint>
#include <vector>

#include "core/math.hpp"
#include "core/rng.hpp"
#include "scene/scene.hpp"

namespace room2::scene {

// Material slots resolved by the game before the builders run.
struct SceneMaterials {
    uint32_t wallPlaster = 0;
    uint32_t ceilingPlaster = 0;
    uint32_t floorConcrete = 0;
    uint32_t floorWood = 0;
    uint32_t skirting = 0;
    uint32_t tableTop = 0;
    uint32_t tableWood = 0;
    uint32_t glass = 0;
    uint32_t lampHousing = 0;
    uint32_t lampEmissive = 0;
    uint32_t gunSlide = 0;
    uint32_t gunFrame = 0;
    uint32_t gunGrip = 0;
    uint32_t gunBarrel = 0;
    uint32_t brass = 0;
    uint32_t fabric = 0;
};

// ---------------------------------------------------------------- room
struct RoomLayout {
    // Interior half extents measured from the room centre.
    Vec3 halfExtents{2.6f, 1.47f, 3.7f};
    float wallThickness = 0.14f;
    float skirtingHeight = 0.095f;
    float skirtingThickness = 0.018f;
    // Physical table placement.
    Vec3 tableCenter{-0.15f, 0.0f, -0.35f};
    Vec2 tableHalfExtents{0.72f, 0.42f};
    float tableHeight = 0.755f;
    float tableTopThickness = 0.042f;
    // Drinking glass, standing on the table.
    Vec3 glassPosition{-0.05f, 0.0f, -0.30f};
    float glassRadius = 0.040f;
    float glassHeight = 0.118f;
    float glassWallThickness = 0.0055f;
    // Ceiling lamp.
    Vec3 lampPosition{0.0f, 0.0f, -0.30f};
    float lampDropLength = 0.42f;
    float lampShadeRadius = 0.15f;
    float lampShadeHeight = 0.14f;
    float lampBulbRadius = 0.045f;

    // The room is centred on the origin, so the walkable floor is at -halfExtents.y.
    // Every furniture height below is measured UP FROM THE FLOOR, not from y = 0.
    float floorLevel() const { return -halfExtents.y; }
    float ceilingLevel() const { return halfExtents.y; }
    float ceilingHeight() const { return halfExtents.y * 2.0f; }
};

// A static box collider handed to the physics world.
struct StaticBox {
    Vec3 center{0, 0, 0};
    Vec3 halfExtents{0.5f, 0.5f, 0.5f};
    Quat rotation{0, 0, 0, 1};
};

// Everything the game needs to know about the constructed room.
struct RoomBuildResult {
    uint32_t roomMesh = kInvalidIndex;
    uint32_t roomInstance = kInvalidIndex;
    uint32_t tableMesh = kInvalidIndex;
    uint32_t tableInstance = kInvalidIndex;
    uint32_t glassMesh = kInvalidIndex;
    uint32_t glassInstance = kInvalidIndex;
    uint32_t glassInteriorMesh = kInvalidIndex;
    uint32_t glassInteriorInstance = kInvalidIndex;
    uint32_t lampMesh = kInvalidIndex;
    uint32_t lampInstance = kInvalidIndex;
    uint32_t lampGlowMesh = kInvalidIndex;
    uint32_t lampGlowInstance = kInvalidIndex;
    uint32_t lightIndex = kInvalidIndex;

    // World-space measurements used by gameplay and physics.
    Vec3 roomCenter{0, 0, 0};
    Vec3 roomHalfExtents{0, 0, 0};
    Vec3 tableCenter{0, 0, 0};
    Vec3 tableHalfExtents{0, 0, 0};
    Vec3 glassCenter{0, 0, 0};
    float glassRadius = 0.04f;
    float glassHeight = 0.118f;
    Vec3 lampPosition{0, 0, 0};
    float lampBulbRadius = 0.045f;
    Aabb roomBounds;
    // Static collision geometry for the room shell and furniture.
    std::vector<StaticBox> colliders;
};

// Appends the room's static box colliders (shell, skirting, table) to `out`.
void appendRoomColliders(const RoomLayout& layout, std::vector<StaticBox>& out);

// Builds the room shell, skirting boards, table, glass and ceiling lamp. Appends
// meshes and instances to `scene` and returns the handles the game needs.
RoomBuildResult buildRoom(Scene& scene, const SceneMaterials& materials,
                          const RoomLayout& layout);

// ---------------------------------------------------------------- weapon
// The HK USP is assembled from procedural primitives in a hierarchy of parts so it
// can be animated (slide travel, magazine, trigger, hammer).
enum class WeaponPart : uint32_t {
    Slide = 0,
    Barrel,
    Frame,
    Grip,
    Trigger,
    Hammer,
    Magazine,
    MagazineFloorplate,
    FrontSight,
    RearSight,
    SlideStop,
    SafetyLever,
    DecockLever,
    Extractor,
    GuideRod,
    RecoilSpring,
    Count
};

struct WeaponMeshSet {
    // Mesh indices per part, plus a merged "static" mesh for parts that never move
    // relative to the frame.
    uint32_t partMeshes[static_cast<uint32_t>(WeaponPart::Count)] = {};
    uint32_t count = 0;
    // Local-space bounds of the assembled pistol (for scaling / HUD placement).
    Aabb assembledBounds;
};

// Builds the pistol geometry into `scene` and returns the mesh handles.
WeaponMeshSet buildWeapon(Scene& scene, const SceneMaterials& materials);

// Builds a single brass cartridge case (ejected from the pistol).
uint32_t buildCartridgeCase(Scene& scene, const SceneMaterials& materials);

// Builds one convex glass shard. `seed` drives the random fracture pattern. The shard
// is generated around the origin with the given approximate size and a flat-ish
// wedge shape; physics approximates it with a box.
uint32_t buildGlassShard(Scene& scene, const SceneMaterials& materials, const Rng& rng,
                         float size, float thickness);

}  // namespace room2::scene
