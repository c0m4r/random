// room2 - procedural mesh construction.
// Every mesh in the project is generated at runtime from these primitives: no
// model files are loaded, which keeps the build self-contained.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/math.hpp"

namespace room2::procgen {

// A single drawable range inside a MeshData, referencing a material slot.
struct SubMesh {
    uint32_t indexOffset = 0;
    uint32_t indexCount = 0;
    uint32_t material = 0;
    Aabb bounds;
    std::string name;
};

struct MeshData {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<SubMesh> submeshes;
    Aabb bounds;

    void clear() {
        vertices.clear();
        indices.clear();
        submeshes.clear();
        bounds = Aabb{};
    }
    bool empty() const { return vertices.empty() || indices.empty(); }
    uint32_t vertexCount() const { return static_cast<uint32_t>(vertices.size()); }
    uint32_t indexCount() const { return static_cast<uint32_t>(indices.size()); }
    uint32_t triangleCount() const { return static_cast<uint32_t>(indices.size() / 3); }

    // Recomputes `bounds` from the vertex positions.
    void computeBounds();
    // Rebuilds vertex normals by area-weighted averaging of face normals, only
    // merging across edges whose dihedral angle is below `smoothAngleDeg`.
    void computeNormals(float smoothAngleDeg = 40.0f);
    // Builds per-vertex tangents from UVs (requires normals + UVs to be present).
    void computeTangents();
    // Applies a transform to positions (inverse-transpose to normals, upper 3x3 to tangents).
    void transform(const Mat4& m);
    // Appends `other`, offsetting its indices and submesh ranges by `materialOffset`.
    void append(const MeshData& other, uint32_t materialOffset = 0);
    // Merges every submesh into a single one using `material`.
    void mergeSubMeshes(uint32_t material = 0, const char* name = nullptr);
    // Closes a submesh starting at `firstIndex` with `material`.
    void addSubMesh(uint32_t firstIndex, uint32_t material, const char* name = nullptr);
    // Flips winding order and negates normals/tangents (for interior surfaces).
    void flipWinding();
    // Welds vertices that share position/normal/uv within a tolerance.
    void weld(float tolerance = 1e-5f);
};

// ---------------------------------------------------------------- primitives
// `uvScale` multiplies the generated texture coordinates; `material` is written
// directly into the submesh list.
MeshData makeBox(Vec3 halfExtents, Vec2 uvScale = Vec2(1, 1), uint32_t material = 0);
// A box whose faces point inwards (used for the room shell). UVs are in world units.
MeshData makeBoxInverted(Vec3 halfExtents, float uvScale = 1.0f, uint32_t material = 0);
MeshData makePlane(float sizeX, float sizeZ, int subdivX, int subdivZ, uint32_t material = 0);
MeshData makeUvSphere(float radius, int segments, int rings, uint32_t material = 0);
// Y-axis aligned cylinder centred on the origin.
MeshData makeCylinder(float radius, float height, int segments, bool capped, uint32_t material = 0);
MeshData makeCone(float radius, float height, int segments, uint32_t material = 0);
MeshData makeCapsule(float radius, float cylinderHeight, int segments, int rings,
                     uint32_t material = 0);
// Torus in the XZ plane.
MeshData makeTorus(float majorRadius, float minorRadius, int majorSegments, int minorSegments,
                   uint32_t material = 0);
// Surface of revolution around the Y axis from a 2D profile (x = radius, y = height).
// `closed` connects the last profile point back to the first (for solid profiles).
MeshData makeLathe(const std::vector<Vec2>& profile, int segments, uint32_t material = 0);
// Swaps the Y and Z axes so a Y-up primitive lies along Z.
MeshData makeRoundedBox(Vec3 halfExtents, float cornerRadius, int cornerSegments,
                        uint32_t material = 0);
// Screen-aligned fullscreen triangle in NDC, for post-processing passes.
MeshData makeFullscreenTriangle();

// ---------------------------------------------------------------- helpers
// Generates smooth normals for a lathe-ish surface by averaging around the seam.
void fixSeamNormals(MeshData& mesh, int segments);
// Scales all UVs by `s`.
void scaleUv(MeshData& mesh, Vec2 s);
// Offsets all UVs by `o`.
void offsetUv(MeshData& mesh, Vec2 o);
// Applies a box-projected (triplanar-ish) UV unwrap in world space; useful for
// architectural pieces where a clean unwrap is not otherwise available.
void boxProjectUv(MeshData& mesh, float scale = 1.0f);
// Bakes ambient-occlusion-like darkening into vertex colors? Not used; kept simple.

// ---------------------------------------------------------------- bounds helpers
Aabb computeBounds(const std::vector<Vertex>& vertices);

}  // namespace room2::procgen
