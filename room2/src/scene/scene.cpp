#include "scene/scene.hpp"

#include <algorithm>

#include "core/log.hpp"

namespace room2::scene {

uint32_t Scene::addMaterial(const Material& material) {
    materials_.push_back(material);
    return static_cast<uint32_t>(materials_.size() - 1);
}

uint32_t Scene::findMaterial(const std::string& name) const {
    for (size_t i = 0; i < materials_.size(); ++i)
        if (materials_[i].name == name) return static_cast<uint32_t>(i);
    return kInvalidIndex;
}

uint32_t Scene::addMesh(const procgen::MeshData& data, const std::string& name) {
    if (data.vertices.empty() || data.indices.empty()) {
        R2_WARN("Scene::addMesh('", name, "') called with empty geometry");
    }
    Mesh mesh;
    mesh.name = name;
    mesh.bounds = data.bounds;

    const int32_t vertexOffset = static_cast<int32_t>(vertices_.size());
    const uint32_t indexOffset = static_cast<uint32_t>(indices_.size());

    vertices_.insert(vertices_.end(), data.vertices.begin(), data.vertices.end());
    // Indices stay relative to this mesh's own vertex block; the draw call supplies
    // `firstVertex = vertexOffset`, so pre-offsetting here would double-apply it.
    indices_.insert(indices_.end(), data.indices.begin(), data.indices.end());

    if (data.submeshes.empty()) {
        MeshRange range;
        range.firstIndex = indexOffset;
        range.indexCount = static_cast<uint32_t>(data.indices.size());
        range.vertexOffset = vertexOffset;
        range.material = 0;
        range.bounds = data.bounds;
        range.centroid = data.bounds.center();
        mesh.ranges.push_back(range);
    } else {
        for (const auto& sub : data.submeshes) {
            if (sub.indexCount == 0) continue;
            MeshRange range;
            range.firstIndex = indexOffset + sub.indexOffset;
            range.indexCount = sub.indexCount;
            range.vertexOffset = vertexOffset;
            range.material = sub.material;
            // Sub-mesh bounds are not always provided; derive them from the vertices.
            Aabb b;
            const uint32_t end = std::min<uint32_t>(sub.indexOffset + sub.indexCount,
                                                    static_cast<uint32_t>(data.indices.size()));
            for (uint32_t i = sub.indexOffset; i < end; ++i) {
                const uint32_t vi = data.indices[i];
                if (vi < data.vertices.size()) b.expand(data.vertices[vi].position);
            }
            range.bounds = b.valid() ? b : data.bounds;
            range.centroid = range.bounds.center();
            mesh.ranges.push_back(range);
        }
    }
    if (mesh.ranges.empty()) {
        MeshRange range;
        range.firstIndex = indexOffset;
        range.indexCount = static_cast<uint32_t>(data.indices.size());
        range.vertexOffset = vertexOffset;
        range.bounds = data.bounds;
        range.centroid = data.bounds.center();
        mesh.ranges.push_back(range);
    }

    meshes_.push_back(std::move(mesh));
    return static_cast<uint32_t>(meshes_.size() - 1);
}

void Scene::reserveGeometry(size_t extraVertices, size_t extraIndices) {
    reservedVertices_ += extraVertices;
    reservedIndices_ += extraIndices;
    vertices_.reserve(vertices_.size() + extraVertices);
    indices_.reserve(indices_.size() + extraIndices);
}

uint32_t Scene::addInstance(const Instance& instance) {
    instances_.push_back(instance);
    return static_cast<uint32_t>(instances_.size() - 1);
}

void Scene::removeInstance(uint32_t index) {
    if (index >= instances_.size()) return;
    // Swap-and-pop keeps indices dense; callers must not hold stale indices.
    instances_[index] = instances_.back();
    instances_.pop_back();
}

void Scene::updateInstanceTransform(uint32_t index, const Mat4& transform) {
    if (index >= instances_.size()) return;
    Instance& inst = instances_[index];
    inst.transform = transform;
    inst.normalMatrix = inverse(mat3FromMat4Upper(transform));
    // Use the mesh's own local bounds; the instance's previous world bounds are only a
    // fallback for meshes that were never measured.
    Aabb local;
    if (inst.mesh < meshes_.size() && meshes_[inst.mesh].bounds.valid()) {
        local = meshes_[inst.mesh].bounds;
    } else {
        const Vec3 he = inst.worldBounds.valid() ? inst.worldBounds.extent()
                                                 : Vec3(0.05f, 0.05f, 0.05f);
        local.mn = -he;
        local.mx = he;
    }
    Aabb world;
    for (int i = 0; i < 8; ++i) {
        Vec3 corner{(i & 1) ? local.mx.x : local.mn.x, (i & 2) ? local.mx.y : local.mn.y,
                    (i & 4) ? local.mx.z : local.mn.z};
        world.expand(transformPoint(transform, corner));
    }
    inst.worldBounds = world;
}

uint32_t Scene::addLight(const Light& light) {
    lights_.push_back(light);
    return static_cast<uint32_t>(lights_.size() - 1);
}

void Scene::refresh() {
    bounds_ = Aabb{};
    for (auto& inst : instances_) {
        if (inst.mesh >= meshes_.size()) continue;
        const Mesh& m = meshes_[inst.mesh];
        inst.normalMatrix = inverse(mat3FromMat4Upper(inst.transform));
        Aabb world;
        for (int i = 0; i < 8; ++i) {
            Vec3 corner{(i & 1) ? m.bounds.mx.x : m.bounds.mn.x,
                        (i & 2) ? m.bounds.mx.y : m.bounds.mn.y,
                        (i & 4) ? m.bounds.mx.z : m.bounds.mn.z};
            world.expand(transformPoint(inst.transform, corner));
        }
        inst.worldBounds = world;
        if (inst.visible) bounds_.expand(world);
    }
}

Scene::Stats Scene::stats() const {
    Stats s;
    s.vertices = static_cast<uint32_t>(vertices_.size());
    s.triangles = static_cast<uint32_t>(indices_.size() / 3);
    s.instances = static_cast<uint32_t>(instances_.size());
    s.materials = static_cast<uint32_t>(materials_.size());
    for (const auto& inst : instances_) {
        if (inst.mesh >= meshes_.size()) continue;
        s.drawCalls += static_cast<uint32_t>(meshes_[inst.mesh].ranges.size());
    }
    return s;
}

void Scene::clear() {
    materials_.clear();
    meshes_.clear();
    instances_.clear();
    lights_.clear();
    vertices_.clear();
    indices_.clear();
    bounds_ = Aabb{};
}

}  // namespace room2::scene
