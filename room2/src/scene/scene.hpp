// room2 - CPU-side scene description: materials, meshes, instances and lights.
// The renderer consumes this and owns all GPU resources.
#pragma once

#include <string>
#include <vector>

#include "core/math.hpp"
#include "procgen/mesh.hpp"

namespace room2::scene {

inline constexpr uint32_t kInvalidIndex = 0xFFFFFFFFu;

// Bindless texture slots. Slot 0 is always a 1x1 white texture and slot 1 a 1x1
// flat normal, so "no texture" can be expressed as a valid index.
inline constexpr uint32_t kTextureWhite = 0;
inline constexpr uint32_t kTextureFlatNormal = 1;
inline constexpr uint32_t kTextureBlack = 2;
inline constexpr uint32_t kReservedTextures = 3;

// ---------------------------------------------------------------- material
struct Material {
    std::string name;
    uint32_t baseColorTex = kTextureWhite;
    uint32_t normalTex = kTextureFlatNormal;
    uint32_t ormTex = kTextureWhite;
    uint32_t emissiveTex = kInvalidIndex;
    Vec4 baseColorFactor{1, 1, 1, 1};
    Vec3 emissiveFactor{0, 0, 0};
    float metallic = 0.0f;
    float roughness = 0.5f;
    float normalScale = 1.0f;
    float occlusionStrength = 1.0f;
    // Transmission / refraction (glass).
    float transmission = 0.0f;
    float ior = 1.5f;
    Vec3 attenuationColor{1, 1, 1};
    float attenuationDistance = 0.0f;   // 0 = no absorption
    float thickness = 0.0f;             // metres, used for the refraction offset
    float alpha = 1.0f;
    bool doubleSided = false;
    bool isEmissive() const { return maxComponent(emissiveFactor) > 1e-4f; }
    bool isTransmissive() const { return transmission > 0.5f; }
};

// ---------------------------------------------------------------- mesh
struct MeshRange {
    uint32_t firstIndex = 0;
    uint32_t indexCount = 0;
    int32_t vertexOffset = 0;
    uint32_t material = 0;
    Aabb bounds;
    Vec3 centroid{0, 0, 0};
};

struct Mesh {
    std::string name;
    std::vector<MeshRange> ranges;
    Aabb bounds;
    uint32_t lodFirstRange = 0;   // reserved for future LODs
};

// ---------------------------------------------------------------- instance
struct Instance {
    Mat4 transform;
    Mat3 normalMatrix;
    uint32_t mesh = 0;
    uint32_t materialOverride = kInvalidIndex;
    uint32_t userData = 0;
    Aabb worldBounds;
    bool castsShadow = true;
    bool visible = true;
    // View models are always drawn: a first-person weapon can straddle the frustum
    // plane when you look sharply up or down, and popping in and out reads as flicker.
    bool alwaysDraw = false;
    // Set for objects that move every frame (glass shards, casings, the weapon).
    bool dynamic = false;
    float sortBias = 0.0f;
};

// ---------------------------------------------------------------- lights
enum class LightType : uint32_t { Point = 0, Spot = 1, Directional = 2 };

struct Light {
    LightType type = LightType::Point;
    Vec3 position{0, 0, 0};
    Vec3 direction{0, -1, 0};
    Vec3 color{1, 1, 1};        // linear radiance tint
    float intensity = 1.0f;     // candela-like scale factor
    float radius = 0.05f;       // emissive sphere radius (soft shadow penumbra)
    float range = 20.0f;
    float innerConeCos = 0.9f;
    float outerConeCos = 0.8f;
    bool castsShadow = true;
    // Sampling data for the shadow cube (filled in by the renderer / room builder).
    Mat4 shadowViewProj[6];
    float shadowNear = 0.05f;
    float shadowFar = 30.0f;
    float shadowFovY = 90.0f * 3.14159265f / 180.0f;
};

// ---------------------------------------------------------------- environment
// A cheap analytic stand-in for the room's global illumination. `radianceCube` is a
// bindless index of a procedural cubemap used for specular ambient reflections.
struct Environment {
    Vec3 ceilingBounce{0.35f, 0.34f, 0.32f};
    Vec3 wallBounce{0.22f, 0.20f, 0.17f};
    Vec3 floorBounce{0.10f, 0.07f, 0.05f};
    Vec3 ambientTint{1.0f, 0.97f, 0.92f};
    float ambientIntensity = 0.35f;
    float exposure = 1.0f;
    uint32_t radianceCube = kInvalidIndex;
    float radianceCubeMipCount = 1.0f;
};

// ---------------------------------------------------------------- scene
class Scene {
public:
    // --- materials -------------------------------------------------------
    uint32_t addMaterial(const Material& material);
    Material& material(uint32_t index) { return materials_[index]; }
    const Material& material(uint32_t index) const { return materials_[index]; }
    uint32_t materialCount() const { return static_cast<uint32_t>(materials_.size()); }
    uint32_t findMaterial(const std::string& name) const;

    // --- meshes ----------------------------------------------------------
    // Appends `data` to the scene's shared geometry and returns a mesh index whose
    // ranges point into the shared vertex/index arrays.
    uint32_t addMesh(const procgen::MeshData& data, const std::string& name);
    // Reserves capacity in the shared geometry arrays (used so that runtime-generated
    // meshes such as glass shards can be appended after the initial upload).
    void reserveGeometry(size_t extraVertices, size_t extraIndices);
    Mesh& mesh(uint32_t index) { return meshes_[index]; }
    const Mesh& mesh(uint32_t index) const { return meshes_[index]; }
    uint32_t meshCount() const { return static_cast<uint32_t>(meshes_.size()); }

    const std::vector<Vertex>& vertices() const { return vertices_; }
    const std::vector<uint32_t>& indices() const { return indices_; }
    std::vector<Vertex>& vertices() { return vertices_; }
    std::vector<uint32_t>& indices() { return indices_; }

    // --- instances -------------------------------------------------------
    uint32_t addInstance(const Instance& instance);
    Instance& instance(uint32_t index) { return instances_[index]; }
    const Instance& instance(uint32_t index) const { return instances_[index]; }
    uint32_t instanceCount() const { return static_cast<uint32_t>(instances_.size()); }
    std::vector<Instance>& instances() { return instances_; }
    const std::vector<Instance>& instances() const { return instances_; }
    void removeInstance(uint32_t index);
    void updateInstanceTransform(uint32_t index, const Mat4& transform);

    // --- lights ----------------------------------------------------------
    uint32_t addLight(const Light& light);
    Light& light(uint32_t index) { return lights_[index]; }
    const std::vector<Light>& lights() const { return lights_; }
    std::vector<Light>& lights() { return lights_; }

    Environment& environment() { return environment_; }
    const Environment& environment() const { return environment_; }

    // Recomputes world bounds / normal matrices and the global bounds.
    void refresh();
    const Aabb& bounds() const { return bounds_; }

    // Statistics for the HUD / logs.
    struct Stats {
        uint32_t triangles = 0;
        uint32_t vertices = 0;
        uint32_t drawCalls = 0;
        uint32_t instances = 0;
        uint32_t materials = 0;
    };
    Stats stats() const;

    void clear();

private:
    std::vector<Material> materials_;
    std::vector<Mesh> meshes_;
    std::vector<Instance> instances_;
    std::vector<Light> lights_;
    std::vector<Vertex> vertices_;
    std::vector<uint32_t> indices_;
    size_t reservedVertices_ = 0;
    size_t reservedIndices_ = 0;
    Environment environment_{};
    Aabb bounds_{};
};

}  // namespace room2::scene
