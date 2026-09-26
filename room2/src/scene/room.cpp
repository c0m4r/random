#include <algorithm>

#include "core/log.hpp"
#include "core/rng.hpp"
#include "procgen/mesh.hpp"
#include "scene/builders.hpp"

namespace room2::scene {
namespace {

using procgen::MeshData;

// ---------------------------------------------------------------- quad helper
// Emits a quad with world-space planar UVs so that tiling is continuous across the
// whole room regardless of how the surfaces are split up.
void addQuad(MeshData& mesh, const Vec3 corners[4], uint32_t material, float uvScale, int uvAxisA,
             int uvAxisB) {
    const Vec3 edge1 = corners[1] - corners[0];
    const Vec3 edge2 = corners[2] - corners[0];
    Vec3 normal = normalize(cross(edge1, edge2));
    if (lengthSq(normal) < 0.5f) normal = Vec3(0, 1, 0);

    const uint32_t firstIndex = static_cast<uint32_t>(mesh.indices.size());
    const uint32_t base = static_cast<uint32_t>(mesh.vertices.size());
    for (int i = 0; i < 4; ++i) {
        Vertex v{};
        v.position = corners[i];
        v.normal = normal;
        v.tangent = Vec4(normalize(std::fabs(normal.y) > 0.9f ? Vec3(1, 0, 0) : Vec3(0, 1, 0)), 1.0f);
        v.uv = Vec2(corners[i][uvAxisA], corners[i][uvAxisB]) * uvScale;
        v.uv1 = v.uv;
        mesh.vertices.push_back(v);
    }
    mesh.indices.insert(mesh.indices.end(), {base, base + 1, base + 2, base, base + 2, base + 3});
    // Every hand-built quad needs its own submesh range, otherwise Scene::addMesh
    // (which only walks the submesh list) would silently drop it.
    mesh.addSubMesh(firstIndex, material, nullptr);
}

void addBox(MeshData& mesh, Vec3 center, Vec3 halfExtents, uint32_t material, float uvScale = 2.0f,
            Quat rotation = Quat(0, 0, 0, 1)) {
    MeshData box = procgen::makeBox(halfExtents, Vec2(uvScale, uvScale), material);
    Mat4 transform = mat4TRS(center, rotation, Vec3(1, 1, 1));
    box.transform(transform);
    for (auto& sub : box.submeshes) sub.material = material;
    mesh.append(box);
}

// Reassigns every submesh material of a freshly built primitive.
void setMaterial(MeshData& mesh, uint32_t material) {
    for (auto& sub : mesh.submeshes) sub.material = material;
    if (mesh.submeshes.empty()) mesh.addSubMesh(0, material, nullptr);
}

void appendTransformed(MeshData& dst, const MeshData& src, const Mat4& transform,
                       uint32_t material, uint32_t materialOffset = 0) {
    MeshData copy = src;
    copy.transform(transform);
    for (auto& sub : copy.submeshes) sub.material = material;
    dst.append(copy, materialOffset);
}

}  // namespace

// ---------------------------------------------------------------- colliders
void appendRoomColliders(const RoomLayout& layout, std::vector<StaticBox>& out) {
    const Vec3 he = layout.halfExtents;
    const float t = layout.wallThickness;
    auto add = [&](Vec3 center, Vec3 half) {
        StaticBox box;
        box.center = center;
        box.halfExtents = half;
        out.push_back(box);
    };

    // Floor, ceiling and four walls (the collider shells sit just outside the visible
    // interior so the player never clips into them).
    add(Vec3(0, -he.y - t * 0.5f, 0), Vec3(he.x + t, t * 0.5f, he.z + t));
    add(Vec3(0, he.y + t * 0.5f, 0), Vec3(he.x + t, t * 0.5f, he.z + t));
    add(Vec3(-he.x - t * 0.5f, 0, 0), Vec3(t * 0.5f, he.y, he.z + t));
    add(Vec3(he.x + t * 0.5f, 0, 0), Vec3(t * 0.5f, he.y, he.z + t));
    add(Vec3(0, 0, -he.z - t * 0.5f), Vec3(he.x, he.y, t * 0.5f));
    add(Vec3(0, 0, he.z + t * 0.5f), Vec3(he.x, he.y, t * 0.5f));

    // Table: top slab and four legs, standing on the floor.
    const Vec3 tc = layout.tableCenter;
    const float floorY = layout.floorLevel();
    const float topHalfY = layout.tableTopThickness * 0.5f;
    add(Vec3(tc.x, floorY + layout.tableHeight - topHalfY, tc.z),
        Vec3(layout.tableHalfExtents.x, topHalfY, layout.tableHalfExtents.y));

    const float legTop = layout.tableHeight - layout.tableTopThickness;
    const float legHalf = 0.032f;
    const float insetX = layout.tableHalfExtents.x - 0.075f;
    const float insetZ = layout.tableHalfExtents.y - 0.065f;
    for (int sx = -1; sx <= 1; sx += 2) {
        for (int sz = -1; sz <= 1; sz += 2) {
            add(Vec3(tc.x + sx * insetX, floorY + legTop * 0.5f, tc.z + sz * insetZ),
                Vec3(legHalf, legTop * 0.5f, legHalf));
        }
    }
    // Apron rails just under the top.
    const float apronHalf = 0.02f;
    const float apronY = floorY + legTop - apronHalf;
    add(Vec3(tc.x, apronY, tc.z - layout.tableHalfExtents.y + 0.03f),
        Vec3(layout.tableHalfExtents.x - 0.05f, apronHalf, apronHalf));
    add(Vec3(tc.x, apronY, tc.z + layout.tableHalfExtents.y - 0.03f),
        Vec3(layout.tableHalfExtents.x - 0.05f, apronHalf, apronHalf));
    add(Vec3(tc.x - layout.tableHalfExtents.x + 0.03f, apronY, tc.z),
        Vec3(apronHalf, apronHalf, layout.tableHalfExtents.y - 0.05f));
    add(Vec3(tc.x + layout.tableHalfExtents.x - 0.03f, apronY, tc.z),
        Vec3(apronHalf, apronHalf, layout.tableHalfExtents.y - 0.05f));
}

// ---------------------------------------------------------------- room
RoomBuildResult buildRoom(Scene& scene, const SceneMaterials& materials,
                          const RoomLayout& layout) {
    RoomBuildResult result;
    const Vec3 he = layout.halfExtents;
    result.roomCenter = Vec3(0, 0, 0);
    result.roomHalfExtents = he;
    result.roomBounds = Aabb{-he, he};
    const float floorY = layout.floorLevel();
    result.tableCenter =
        Vec3(layout.tableCenter.x, floorY + layout.tableHeight, layout.tableCenter.z);
    result.tableHalfExtents =
        Vec3(layout.tableHalfExtents.x, layout.tableTopThickness * 0.5f, layout.tableHalfExtents.y);
    result.lampPosition = Vec3(layout.lampPosition.x,
                               layout.ceilingLevel() - layout.lampDropLength -
                                   layout.lampShadeHeight * 0.5f,
                               layout.lampPosition.z);
    result.lampBulbRadius = layout.lampBulbRadius;
    result.glassHeight = layout.glassHeight;
    result.glassRadius = layout.glassRadius;
    result.glassCenter = Vec3(layout.glassPosition.x,
                              floorY + layout.tableHeight + layout.glassHeight * 0.5f,
                              layout.glassPosition.z);

    Rng rng(20240926u);

    // ---------------- room shell ----------------
    {
        MeshData shell;
        const float uvWall = 1.0f / 3.2f;    // one texture repeat per 3.2 m
        const float uvFloor = 1.0f / 2.6f;
        const float uvCeiling = 1.0f / 3.0f;

        // Floor (normal +Y).
        const Vec3 floor[4] = {{-he.x, -he.y, -he.z},
                               {-he.x, -he.y, he.z},
                               {he.x, -he.y, he.z},
                               {he.x, -he.y, -he.z}};
        addQuad(shell, floor, materials.floorWood, uvFloor, 0, 2);

        // Ceiling (normal -Y).
        const Vec3 ceiling[4] = {{-he.x, he.y, -he.z},
                                 {he.x, he.y, -he.z},
                                 {he.x, he.y, he.z},
                                 {-he.x, he.y, he.z}};
        addQuad(shell, ceiling, materials.ceilingPlaster, uvCeiling, 0, 2);

        // Wall at -Z (normal +Z).
        const Vec3 wallBack[4] = {{-he.x, -he.y, -he.z},
                                  {he.x, -he.y, -he.z},
                                  {he.x, he.y, -he.z},
                                  {-he.x, he.y, -he.z}};
        addQuad(shell, wallBack, materials.wallPlaster, uvWall, 0, 1);

        // Wall at +Z (normal -Z).
        const Vec3 wallFront[4] = {{he.x, -he.y, he.z},
                                   {-he.x, -he.y, he.z},
                                   {-he.x, he.y, he.z},
                                   {he.x, he.y, he.z}};
        addQuad(shell, wallFront, materials.wallPlaster, uvWall, 0, 1);

        // Wall at -X (normal +X).
        const Vec3 wallLeft[4] = {{-he.x, -he.y, he.z},
                                  {-he.x, -he.y, -he.z},
                                  {-he.x, he.y, -he.z},
                                  {-he.x, he.y, he.z}};
        addQuad(shell, wallLeft, materials.wallPlaster, uvWall, 2, 1);

        // Wall at +X (normal -X).
        const Vec3 wallRight[4] = {{he.x, -he.y, -he.z},
                                   {he.x, -he.y, he.z},
                                   {he.x, he.y, he.z},
                                   {he.x, he.y, -he.z}};
        addQuad(shell, wallRight, materials.wallPlaster, uvWall, 2, 1);

        // Skirting boards around the base of every wall.
        const float sh = layout.skirtingHeight;
        const float st = layout.skirtingThickness;
        auto skirting = [&](Vec3 center, Vec3 half) {
            addBox(shell, center, half, materials.skirting, 1.0f);
        };
        skirting(Vec3(0, -he.y + sh * 0.5f, -he.z + st * 0.5f), Vec3(he.x, sh * 0.5f, st * 0.5f));
        skirting(Vec3(0, -he.y + sh * 0.5f, he.z - st * 0.5f), Vec3(he.x, sh * 0.5f, st * 0.5f));
        skirting(Vec3(-he.x + st * 0.5f, -he.y + sh * 0.5f, 0), Vec3(st * 0.5f, sh * 0.5f, he.z));
        skirting(Vec3(he.x - st * 0.5f, -he.y + sh * 0.5f, 0), Vec3(st * 0.5f, sh * 0.5f, he.z));
        // A simple cornice at the wall/ceiling junction.
        const float ch = 0.045f, ct = 0.03f;
        skirting(Vec3(0, he.y - ch * 0.5f, -he.z + ct * 0.5f), Vec3(he.x, ch * 0.5f, ct * 0.5f));
        skirting(Vec3(0, he.y - ch * 0.5f, he.z - ct * 0.5f), Vec3(he.x, ch * 0.5f, ct * 0.5f));
        skirting(Vec3(-he.x + ct * 0.5f, he.y - ch * 0.5f, 0), Vec3(ct * 0.5f, ch * 0.5f, he.z));
        skirting(Vec3(he.x - ct * 0.5f, he.y - ch * 0.5f, 0), Vec3(ct * 0.5f, ch * 0.5f, he.z));

        // A ceiling rose around the lamp flex.
        MeshData rose = procgen::makeCylinder(0.055f, 0.020f, 24, true, materials.ceilingPlaster);
        rose.transform(mat4TRS(Vec3(layout.lampPosition.x, he.y - 0.010f, layout.lampPosition.z),
                               Quat(0, 0, 0, 1), Vec3(1, 1, 1)));
        setMaterial(rose, materials.ceilingPlaster);
        shell.append(rose);

        shell.computeBounds();
        result.roomMesh = scene.addMesh(shell, "room_shell");
        Instance inst;
        inst.transform = mat4Identity();
        inst.mesh = result.roomMesh;
        inst.castsShadow = true;
        result.roomInstance = scene.addInstance(inst);
    }

    // ---------------- table ----------------
    {
        MeshData table;
        const Vec3 tc = layout.tableCenter;
        const float topY = floorY + layout.tableHeight - layout.tableTopThickness * 0.5f;
        const float hx = layout.tableHalfExtents.x;
        const float hz = layout.tableHalfExtents.y;

        // Top slab with a small chamfer achieved by stacking two boxes.
        addBox(table, Vec3(tc.x, topY, tc.z),
               Vec3(hx, layout.tableTopThickness * 0.5f, hz), materials.tableTop, 1.0f);
        addBox(table, Vec3(tc.x, topY - layout.tableTopThickness * 0.5f - 0.004f, tc.z),
               Vec3(hx - 0.006f, 0.004f, hz - 0.006f), materials.tableWood, 2.0f);

        // Apron rails.
        const float legTop = layout.tableHeight - layout.tableTopThickness;
        const float apronHalf = 0.020f;
        const float apronY = floorY + legTop - apronHalf;
        addBox(table, Vec3(tc.x, apronY, tc.z - hz + 0.030f), Vec3(hx - 0.05f, apronHalf, apronHalf),
               materials.tableWood, 2.0f);
        addBox(table, Vec3(tc.x, apronY, tc.z + hz - 0.030f), Vec3(hx - 0.05f, apronHalf, apronHalf),
               materials.tableWood, 2.0f);
        addBox(table, Vec3(tc.x - hx + 0.030f, apronY, tc.z), Vec3(apronHalf, apronHalf, hz - 0.05f),
               materials.tableWood, 2.0f);
        addBox(table, Vec3(tc.x + hx - 0.030f, apronY, tc.z), Vec3(apronHalf, apronHalf, hz - 0.05f),
               materials.tableWood, 2.0f);

        // Legs: a slightly tapered square section, built as a lathe for a turned look.
        const float insetX = hx - 0.075f;
        const float insetZ = hz - 0.065f;
        for (int sx = -1; sx <= 1; sx += 2) {
            for (int sz = -1; sz <= 1; sz += 2) {
                MeshData leg = procgen::makeLathe(
                    {
                        Vec2(0.040f, 0.000f),
                        Vec2(0.042f, 0.010f),
                        Vec2(0.033f, 0.045f),
                        Vec2(0.028f, 0.090f),
                        Vec2(0.030f, legTop - 0.070f),
                        Vec2(0.034f, legTop - 0.030f),
                        Vec2(0.036f, legTop),
                        Vec2(0.000f, legTop),
                        Vec2(0.000f, 0.000f),
                    },
                    20, materials.tableWood);
                leg.transform(mat4TRS(Vec3(tc.x + sx * insetX, floorY, tc.z + sz * insetZ),
                                      Quat(0, 0, 0, 1), Vec3(1, 1, 1)));
                setMaterial(leg, materials.tableWood);
                table.append(leg);
            }
        }
        table.computeBounds();
        result.tableMesh = scene.addMesh(table, "table");
        Instance inst;
        inst.transform = mat4Identity();
        inst.mesh = result.tableMesh;
        inst.castsShadow = true;
        result.tableInstance = scene.addInstance(inst);
    }

    // ---------------- drinking glass ----------------
    {
        const float R = layout.glassRadius;
        const float H = layout.glassHeight;
        const float wall = layout.glassWallThickness;
        const float innerR = R - wall;
        const float baseThickness = 0.014f;

        // Profile: bottom centre -> outer base -> outer wall -> rim -> inner wall ->
        // inner base -> inner centre. Revolved this gives a real tumbler with a thick
        // base and an open top.
        std::vector<Vec2> profile = {
            Vec2(0.0f, 0.0f),
            Vec2(R * 0.55f, 0.0f),
            Vec2(R * 0.88f, 0.0015f),
            Vec2(R * 0.99f, 0.006f),
            Vec2(R, 0.013f),
            Vec2(R, H * 0.45f),
            Vec2(R * 0.995f, H * 0.80f),
            Vec2(R * 0.985f, H - 0.004f),
            Vec2(R * 0.975f, H),
            Vec2(innerR * 0.99f, H),
            Vec2(innerR * 0.985f, H - 0.004f),
            Vec2(innerR, H * 0.80f),
            Vec2(innerR, baseThickness + 0.006f),
            Vec2(innerR * 0.92f, baseThickness + 0.002f),
            Vec2(innerR * 0.55f, baseThickness),
            Vec2(0.0f, baseThickness),
        };
        MeshData glass = procgen::makeLathe(profile, 64, materials.glass);
        setMaterial(glass, materials.glass);
        glass.computeBounds();
        result.glassMesh = scene.addMesh(glass, "drinking_glass");
        Instance inst;
        inst.transform = mat4TRS(Vec3(layout.glassPosition.x,
                                      floorY + layout.tableHeight, layout.glassPosition.z),
                                 Quat(0, 0, 0, 1), Vec3(1, 1, 1));
        inst.mesh = result.glassMesh;
        inst.castsShadow = false;   // glass caustics are not modelled; skip its shadow
        inst.dynamic = true;
        inst.userData = 1;
        result.glassInstance = scene.addInstance(inst);
    }

    // ---------------- ceiling lamp ----------------
    {
        MeshData lamp;
        const float ceilingY = he.y;
        const Vec3 lampBase(layout.lampPosition.x, ceilingY, layout.lampPosition.z);
        const float flexTop = ceilingY - 0.02f;
        const float shadeTopY = result.lampPosition.y + layout.lampShadeHeight * 0.5f;

        // Flex / cord.
        {
            MeshData cord = procgen::makeCylinder(0.0045f, flexTop - shadeTopY, 12, false,
                                                  materials.lampHousing);
            cord.transform(mat4TRS(Vec3(lampBase.x, (flexTop + shadeTopY) * 0.5f, lampBase.z),
                                   Quat(0, 0, 0, 1), Vec3(1, 1, 1)));
            setMaterial(cord, materials.lampHousing);
            lamp.append(cord);
        }
        // Shade: a truncated cone shell with visible thickness.
        {
            const float topR = layout.lampShadeRadius * 0.42f;
            const float botR = layout.lampShadeRadius;
            const float t = 0.004f;
            std::vector<Vec2> shadeProfile = {
                Vec2(topR, layout.lampShadeHeight * 0.5f),
                Vec2(botR, -layout.lampShadeHeight * 0.5f),
                Vec2(botR - t * 1.4f, -layout.lampShadeHeight * 0.5f),
                Vec2(topR - t, layout.lampShadeHeight * 0.5f),
                Vec2(topR, layout.lampShadeHeight * 0.5f),
            };
            MeshData shade = procgen::makeLathe(shadeProfile, 48, materials.lampHousing);
            setMaterial(shade, materials.lampHousing);
            shade.transform(mat4TRS(result.lampPosition, Quat(0, 0, 0, 1), Vec3(1, 1, 1)));
            lamp.append(shade);
        }
        // Housing ring at the top of the shade.
        {
            MeshData ring = procgen::makeCylinder(layout.lampShadeRadius * 0.44f, 0.014f, 32, true,
                                                  materials.lampHousing);
            ring.transform(mat4TRS(Vec3(result.lampPosition.x, shadeTopY, result.lampPosition.z),
                                   Quat(0, 0, 0, 1), Vec3(1, 1, 1)));
            setMaterial(ring, materials.lampHousing);
            lamp.append(ring);
        }
        lamp.computeBounds();
        result.lampMesh = scene.addMesh(lamp, "lamp");
        Instance inst;
        inst.transform = mat4Identity();
        inst.mesh = result.lampMesh;
        // The shade is thin fabric: a hard geometric silhouette from a light sitting a
        // few centimetres away reads as an artefact, so the transmitted light is
        // represented by the soft fill light instead and only the furniture casts
        // shadows into the room.
        inst.castsShadow = false;
        result.lampInstance = scene.addInstance(inst);

        // Emissive bulb.
        MeshData bulb = procgen::makeUvSphere(layout.lampBulbRadius, 24, 16, materials.lampEmissive);
        setMaterial(bulb, materials.lampEmissive);
        bulb.computeBounds();
        result.lampGlowMesh = scene.addMesh(bulb, "lamp_bulb");
        Instance bulbInst;
        // The bulb sits at the shade's aperture: a pendant directs light downwards and
        // outwards through the opening, not from deep inside the drum.
        // The filament sits just below the shade's aperture so the shade only blocks
        // upward light, which is what makes a pendant read correctly.
        const float bulbY = result.lampPosition.y - layout.lampShadeHeight * 0.5f - 0.035f;
        bulbInst.transform = mat4TRS(Vec3(result.lampPosition.x, bulbY, result.lampPosition.z),
                                     Quat(0, 0, 0, 1), Vec3(1, 1, 1));
        bulbInst.mesh = result.lampGlowMesh;
        bulbInst.castsShadow = false;
        result.lampGlowInstance = scene.addInstance(bulbInst);
    }

    // ---------------- lighting ----------------
    {
        Light main;
        main.type = LightType::Point;
        main.position = Vec3(result.lampPosition.x,
                             result.lampPosition.y - layout.lampShadeHeight * 0.5f - 0.038f,
                             result.lampPosition.z);
        main.color = Vec3(1.0f, 0.925f, 0.83f);   // warm incandescent
        main.intensity = 42.0f;
        main.radius = layout.lampBulbRadius * 1.15f;
        main.range = 16.0f;
        main.castsShadow = true;
        main.shadowNear = 0.08f;
        main.shadowFar = 9.0f;
        // The pendant directs light downwards and outwards through the shade's
        // aperture, so it is modelled as a wide spot: the ceiling above the lamp is
        // deliberately not lit by it (a separate soft fill stands in for the shade's
        // translucency). One perspective shadow map covers the whole room.
        main.type = LightType::Spot;
        main.direction = Vec3(0.0f, -1.0f, 0.0f);
        main.innerConeCos = std::cos(1.20f);   // ~69 degrees
        main.outerConeCos = std::cos(1.36f);   // ~78 degrees
        main.shadowNear = 0.12f;
        main.shadowFar = 9.5f;
        main.shadowFovY = 2.72f;               // 156 degrees, covers the room's corners
        const Mat4 shadowProj = mat4PerspectiveVk(main.shadowFovY, 1.0f, main.shadowNear,
                                                  main.shadowFar);
        const Mat4 shadowView =
            mat4LookAt(main.position, main.position + main.direction, Vec3(0.0f, 0.0f, -1.0f));
        for (int f = 0; f < 6; ++f) main.shadowViewProj[f] = shadowProj * shadowView;
        result.lightIndex = scene.addLight(main);

        // A weak upward spill standing in for the translucency of the fabric shade.
        Light spill;
        spill.type = LightType::Point;
        spill.position = Vec3(result.lampPosition.x, result.lampPosition.y + 0.14f,
                              result.lampPosition.z);
        spill.color = Vec3(1.0f, 0.90f, 0.76f);
        spill.intensity = 2.0f;
        spill.radius = layout.lampShadeRadius;
        spill.range = 6.0f;
        spill.castsShadow = false;
        scene.addLight(spill);
    }

    // ---------------- environment -----------------
    scene.environment().ceilingBounce = Vec3(0.46f, 0.45f, 0.42f);
    scene.environment().wallBounce = Vec3(0.34f, 0.32f, 0.28f);
    scene.environment().floorBounce = Vec3(0.15f, 0.10f, 0.065f);
    scene.environment().ambientIntensity = 0.42f;
    scene.environment().ambientTint = Vec3(1.0f, 0.965f, 0.91f);

    appendRoomColliders(layout, result.colliders);
    scene.refresh();
    return result;
}

}  // namespace room2::scene
