#include "game/game.hpp"

#include <algorithm>
#include <cstdio>

#include "core/log.hpp"
#include "core/rng.hpp"

namespace room2::game {
namespace {

constexpr float kPi = PI;

// Weapon handling timings, in seconds.
constexpr float kDrawTime = 0.62f;
constexpr float kHolsterTime = 0.45f;
constexpr float kReloadTime = 2.35f;
constexpr float kFireInterval = 0.155f;     // ~390 rpm effective
constexpr float kSlideCycleTime = 0.055f;
constexpr float kMuzzleFlashTime = 0.045f;

// Debris collision layer: fragments ignore each other but hit everything else.
constexpr uint32_t kShardGroup = 1u << 4;

Mat4 offsetMatrix(Vec3 translation, Vec3 eulerRadians) {
    return mat4TRS(translation, quatFromEuler(eulerRadians), Vec3(1, 1, 1));
}

}  // namespace

const char* weaponStateName(WeaponState state) {
    switch (state) {
        case WeaponState::Holstered: return "holstered";
        case WeaponState::Drawing: return "drawing";
        case WeaponState::Ready: return "ready";
        case WeaponState::Firing: return "firing";
        case WeaponState::Reloading: return "reloading";
        case WeaponState::Holstering: return "holstering";
    }
    return "?";
}

Game::~Game() { shutdown(); }

bool Game::init(scene::Scene& scene, render::Renderer& renderer, const GameConfig& config) {
    scene_ = &scene;
    renderer_ = &renderer;
    config_ = config;

    // ---------------- materials ----------------
    buildMaterials();

    // ---------------- room and furniture ----------------
    scene::RoomLayout layout;
    room_ = scene::buildRoom(scene, materials_, layout);

    // ---------------- physics world ----------------
    buildPhysicsWorld();

    // ---------------- player ----------------
    PlayerConfig playerConfig;
    const float floorY = room_.roomBounds.mn.y;
    player_.init(playerConfig, Vec3(1.15f, floorY + 0.02f, 2.35f), 0.0f);
    // Start looking straight at the glass so the demo (and the scripted self test) can
    // fire without any manual aiming.
    {
        const Vec3 eye = player_.eyePosition();
        const Vec3 toGlass = normalize(room_.glassCenter - eye);
        player_.setYaw(std::atan2(toGlass.x, -toGlass.z));
        player_.setPitch(std::asin(clamp(toGlass.y, -1.0f, 1.0f)));
    }
    camera_.setPerspective(70.0f * DEG2RAD, 16.0f / 9.0f, 0.05f, 60.0f);

    // ---------------- glass in the physics world ----------------
    {
        phys::Body body;
        body.shape = phys::Shape::makeCapsule(room_.glassRadius,
                                              std::max(room_.glassHeight * 0.5f - room_.glassRadius,
                                                       0.005f));
        body.position = Vec3(room_.glassCenter.x, room_.glassCenter.y, room_.glassCenter.z);
        body.flags = phys::BODY_STATIC;
        body.material.restitution = 0.08f;
        body.material.staticFriction = 0.5f;
        body.userData = 1;   // identifies the glass for raycasts
        glassBody_ = world_.addBody(body);
    }

    // ---------------- view-model fill ----------------
    // A short-range light rigidly attached to the camera. It only reaches the weapon and
    // the floor immediately around the player, which is the usual way a first-person
    // view model is kept readable without lighting the whole room.
    {
        scene::Light fill;
        fill.type = scene::LightType::Point;
        fill.position = player_.eyePosition();
        fill.color = Vec3(1.0f, 0.97f, 0.92f);
        fill.intensity = 0.85f;
        fill.radius = 0.12f;
        fill.range = 1.05f;
        fill.castsShadow = false;
        viewFillLight_ = scene.addLight(fill);
    }

    // ---------------- weapon ----------------
    weapon_ = scene::buildWeapon(scene, materials_);
    for (uint32_t i = 0; i < weapon_.count; ++i) {
        scene::Instance inst;
        inst.mesh = weapon_.partMeshes[i];
        inst.transform = mat4Identity();
        inst.castsShadow = false;   // the view model would otherwise shadow the room oddly
        inst.visible = false;       // hidden until drawn
        inst.dynamic = true;
        inst.alwaysDraw = true;
        weaponPartInstances_[i] = scene.addInstance(inst);
    }
    casingMesh_ = scene::buildCartridgeCase(scene, materials_);
    scene.refresh();

    // ---------------- audio ----------------
    if (config_.enableAudio) {
        if (!audio_.init(48000, 2, 64, config_.masterVolume)) {
            R2_WARN("audio unavailable: ", audio_.lastError());
        } else {
            audio_.play2D(audio::SoundId::RoomTone, {0.32f, 1.0f, 0.0f});
        }
    }

    if (config_.startWithWeaponDrawn) {
        weaponState_ = WeaponState::Drawing;
        stateTime_ = 0.0f;
    }
    setEvent("Click or press Q to draw the pistol", 4.0f);

    R2_INFO("game ready: ", dynamics_.size(), " dynamic bodies, weapon state ",
            weaponStateName(weaponState_));
    return true;
}

namespace {

// Creates a scene material from a procedural texture set and uploads its maps.
uint32_t addMaterial(scene::Scene& scene, render::Renderer& renderer, const char* name,
                     procgen::TextureSet set, Vec3 tint, float metallic, float roughness,
                     float normalScale) {
    scene::Material mat;
    mat.name = name;
    mat.baseColorTex = renderer.addTexture(set.baseColor, true, true, std::string(name) + "_bc");
    mat.normalTex = set.hasNormal
                        ? renderer.addTexture(set.normal, false, true, std::string(name) + "_n")
                        : scene::kTextureFlatNormal;
    mat.ormTex = set.hasOrm
                     ? renderer.addTexture(set.orm, false, true, std::string(name) + "_orm")
                     : scene::kTextureWhite;
    mat.baseColorFactor = Vec4(tint, 1.0f);
    mat.metallic = metallic;
    mat.roughness = roughness;
    mat.normalScale = normalScale;
    return scene.addMaterial(mat);
}

}  // namespace

void Game::buildMaterials() {
    using namespace room2::procgen;
    if (!scene_ || !renderer_) return;
    scene::Scene& scene = *scene_;
    render::Renderer& renderer = *renderer_;

    const uint32_t size = config_.textureSize;
    SurfaceParams sp;
    sp.seed = config_.seed;
    sp.texelsPerMetre = static_cast<float>(size) / 2.0f;

    scene::SceneMaterials& m = materials_;
    m.wallPlaster = addMaterial(scene, renderer, "wall_plaster", plasterWall(size, sp),
                                Vec3(1, 1, 1), 0.0f, 0.92f, 0.40f);
    m.ceilingPlaster = addMaterial(scene, renderer, "ceiling_plaster", paintedCeiling(size, sp),
                                   Vec3(1, 1, 1), 0.0f, 0.94f, 0.35f);
    m.floorWood = addMaterial(scene, renderer, "floor_wood", woodPlanks(size, sp),
                              Vec3(1.35f, 1.32f, 1.28f), 0.0f, 1.0f, 0.85f);
    m.floorConcrete = addMaterial(scene, renderer, "floor_concrete", concreteFloor(size, sp),
                                  Vec3(1, 1, 1), 0.0f, 1.0f, 1.0f);
    m.skirting = addMaterial(scene, renderer, "skirting", varnishedWood(size, sp),
                             Vec3(2.2f, 2.1f, 2.0f), 0.0f, 1.0f, 0.8f);
    m.tableTop = addMaterial(scene, renderer, "table_top", woodTableTop(size, sp),
                             Vec3(1.5f, 1.42f, 1.30f), 0.0f, 1.0f, 1.0f);
    m.tableWood = addMaterial(scene, renderer, "table_wood", varnishedWood(size, sp),
                              Vec3(3.4f, 3.2f, 3.0f), 0.0f, 1.0f, 0.8f);
    m.lampHousing = addMaterial(scene, renderer, "lamp_housing", paintedMetalWhite(size, sp),
                                Vec3(1, 1, 1), 0.0f, 1.0f, 0.6f);

    // Glass: transmissive, near-smooth.
    {
        TextureSet glassSet = glassSurface(512, sp);
        scene::Material glass;
        glass.name = "glass";
        glass.baseColorTex = renderer.addTexture(glassSet.baseColor, true, true, "glass_bc");
        glass.normalTex = glassSet.hasNormal
                              ? renderer.addTexture(glassSet.normal, false, true, "glass_n")
                              : scene::kTextureFlatNormal;
        glass.ormTex = glassSet.hasOrm
                           ? renderer.addTexture(glassSet.orm, false, true, "glass_orm")
                           : scene::kTextureWhite;
        glass.baseColorFactor = Vec4(0.97f, 0.98f, 0.98f, 1.0f);
        glass.metallic = 0.0f;
        glass.roughness = 0.045f;
        glass.normalScale = 0.25f;
        glass.transmission = 1.0f;
        glass.ior = 1.52f;
        glass.attenuationColor = Vec3(0.86f, 0.94f, 0.88f);
        glass.thickness = 0.006f;
        glass.alpha = 0.25f;
        glass.doubleSided = true;
        m.glass = scene.addMaterial(glass);
    }
    // Lamp diffuser: strongly emissive.
    {
        TextureSet lampSet = lampEmissive(256, sp);
        scene::Material lamp;
        lamp.name = "lamp_emissive";
        lamp.baseColorTex = renderer.addTexture(lampSet.baseColor, true, true, "lamp_bc");
        lamp.ormTex = lampSet.hasOrm ? renderer.addTexture(lampSet.orm, false, true, "lamp_orm")
                                     : scene::kTextureWhite;
        lamp.baseColorFactor = Vec4(1.0f, 0.96f, 0.90f, 1.0f);
        lamp.emissiveFactor = Vec3(14.0f, 12.4f, 10.2f);
        lamp.metallic = 0.0f;
        lamp.roughness = 0.6f;
        m.lampEmissive = scene.addMaterial(lamp);
    }

    // Blued steel is a dark metal; the tint lifts it just enough that the view model
    // still reads as steel in the dim lower part of the room.
    m.gunSlide = addMaterial(scene, renderer, "gun_slide", bluedSteel(size, sp),
                             Vec3(2.1f, 2.1f, 2.2f), 0.92f, 1.0f, 0.7f);
    m.gunFrame = addMaterial(scene, renderer, "gun_frame", blackPolymer(size, sp), Vec3(1, 1, 1),
                             0.0f, 1.0f, 1.0f);
    m.gunGrip = addMaterial(scene, renderer, "gun_grip", gripPanel(size, sp), Vec3(1, 1, 1), 0.0f,
                            1.0f, 1.0f);
    m.gunBarrel = addMaterial(scene, renderer, "gun_barrel", bluedSteel(size, sp),
                              Vec3(1.8f, 1.8f, 1.85f), 0.92f, 1.0f, 0.6f);
    m.brass = addMaterial(scene, renderer, "brass", brass(size, sp), Vec3(1, 1, 1), 1.0f, 1.0f,
                          0.5f);
    m.fabric = addMaterial(scene, renderer, "fabric", fabric(size, sp), Vec3(1, 1, 1), 0.0f, 1.0f,
                           1.0f);
    R2_INFO("materials: ", scene.materialCount(), " (", materials_.wallPlaster, "=",
            scene.material(materials_.wallPlaster).name, ")");
}

void Game::buildPhysicsWorld() {
    phys::WorldConfig physicsConfig;
    physicsConfig.gravity = Vec3(0, -9.81f, 0);
    physicsConfig.velocityIterations = 10;
    physicsConfig.positionIterations = 3;
    physicsConfig.substeps = 2;
    world_.setConfig(physicsConfig);

    for (const scene::StaticBox& box : room_.colliders) {
        phys::Body body;
        body.shape = phys::Shape::makeBox(box.halfExtents);
        body.position = box.center;
        body.orientation = box.rotation;
        body.flags = phys::BODY_STATIC;
        body.material.restitution = 0.02f;
        body.material.staticFriction = 0.62f;
        body.material.dynamicFriction = 0.5f;
        world_.addBody(body);
    }
}

Aabb Game::shardBounds() const {
    Aabb bounds;
    for (const auto& obj : dynamics_) {
        if (!obj.alive || !scene_) continue;
        const phys::Body* body = world_.getBody(obj.body);
        if (body) bounds.expand(body->position);
    }
    return bounds;
}

void Game::shardSpreadStats(float& meanDistance, float& maxDistance, float& maxSpeed) const {
    meanDistance = 0.0f;
    maxDistance = 0.0f;
    maxSpeed = 0.0f;
    int count = 0;
    double sum = 0.0;
    for (const auto& obj : dynamics_) {
        const phys::Body* body = world_.getBody(obj.body);
        if (!body || body->userData != 2) continue;
        const float d = distance(body->position, room_.glassCenter);
        sum += d;
        maxDistance = std::max(maxDistance, d);
        maxSpeed = std::max(maxSpeed, length(body->linearVelocity));
        ++count;
    }
    if (count > 0) meanDistance = static_cast<float>(sum / count);
}

void Game::shutdown() {
    audio_.shutdown();
    dynamics_.clear();
    scene_ = nullptr;
    renderer_ = nullptr;
}

void Game::setEvent(std::string text, float duration) {
    eventText_ = std::move(text);
    eventTimer_ = duration;
    lastEvent_ = eventText_;
}

void Game::playWeaponSound(audio::SoundId id, float volume, float pitch) {
    if (!audio_.isInitialised()) return;
    audio::PlayParams params;
    params.volume = volume;
    params.pitch = pitch;
    params.occlusion = 0.0f;
    audio_.play2D(id, params);
}

// ---------------------------------------------------------------- weapon transform
Mat4 Game::weaponTransform() const {
    const Vec3 eye = player_.eyePosition();
    // The offset is expressed in CAMERA-LOCAL space: +x right, +y up, -z forward.
    // (The weapon mesh has its muzzle along +z, so the root also carries a 180 degree
    // yaw that turns the model's +z into the camera's forward direction.)
    Vec3 offset(0.098f, -0.082f, -0.405f);
    float pitchExtra = 0.0f;   // local, relative to the camera
    float rollExtra = 0.0f;
    float yawExtra = 0.0f;

    // Draw / holster: the weapon rises from below the frame.
    const float drawT = clamp(drawProgress_, 0.0f, 1.0f);
    const float eased = drawT * drawT * (3.0f - 2.0f * drawT);
    offset.y -= (1.0f - eased) * 0.34f;
    offset.z += (1.0f - eased) * 0.06f;
    pitchExtra += (1.0f - eased) * -0.5f;

    // Holstering blends the weapon down and out.
    if (weaponState_ == WeaponState::Holstering) {
        const float t = clamp(stateTime_ / kHolsterTime, 0.0f, 1.0f);
        const float e = t * t;
        offset.y -= e * 0.34f;
        offset.z += e * 0.06f;
        pitchExtra -= e * 0.5f;
    }

    // Reload: the weapon tilts inboard and dips so the magazine well is visible.
    if (weaponState_ == WeaponState::Reloading) {
        const float tilt = std::sin(clamp(reloadPhase_, 0.0f, 1.0f) * kPi);
        offset.x -= tilt * 0.055f;
        offset.y -= tilt * 0.080f;
        offset.z += tilt * 0.050f;
        rollExtra += tilt * 0.55f;
        yawExtra -= tilt * 0.30f;
        pitchExtra += tilt * 0.24f;
    }

    // Head bob and inertial sway from the player controller.
    const Vec3 bob = player_.viewBob();
    const Vec2 sway = player_.viewSway();
    offset.x += bob.x + sway.x;
    offset.y += bob.y + sway.y;
    offset.z += bob.z;

    // Recoil is a LOCAL offset of the view model: the weapon kicks back towards the
    // shooter and rotates up while staying attached to the view.
    offset.z += recoilKick_;
    pitchExtra -= recoilPitch_;

    // The basis uses exactly the orientation the camera is using (including the view
    // recoil offsets), so the view model can never drift or wobble relative to the view.
    const float yaw = player_.yaw() + viewRecoilYaw_ + yawExtra;
    const float pitch = clamp(player_.pitch() + viewRecoilPitch_ + pitchExtra, -1.55f, 1.55f);
    const float cp = std::cos(pitch);
    const Vec3 forward(std::sin(yaw) * cp, std::sin(pitch), -std::cos(yaw) * cp);
    const Vec3 right = normalize(cross(forward, Vec3(0, 1, 0)));
    const Vec3 up = cross(right, forward);

    // Camera-local basis: +x right, +y up, +z backwards (so -z is forward).
    Mat4 cameraBasis;
    cameraBasis.c[0] = Vec4(right, 0.0f);
    cameraBasis.c[1] = Vec4(up, 0.0f);
    cameraBasis.c[2] = Vec4(-forward, 0.0f);
    cameraBasis.c[3] = Vec4(eye, 1.0f);

    // The pistol mesh points its muzzle along +z, so a half turn about Y aligns it with
    // the camera's forward direction. Roll (reload tilt) is applied in camera space.
    const Mat4 muzzleAlign = mat4FromQuat(quatFromAxisAngle(Vec3(0, 1, 0), PI));
    const Mat4 roll = mat4FromQuat(quatFromAxisAngle(Vec3(0, 0, 1), rollExtra));
    return cameraBasis * roll * mat4Translate(offset) * muzzleAlign;
}

// ---------------------------------------------------------------- dynamic objects
void Game::spawnCartridge(Vec3 position, Vec3 velocity) {
    if (casingMesh_ == scene::kInvalidIndex || !scene_) return;

    scene::Instance inst;
    inst.mesh = casingMesh_;
    inst.transform = mat4TRS(position, quatFromEuler(Vec3(0.4f, 0.9f, 0.2f)), Vec3(1, 1, 1));
    inst.castsShadow = false;
    inst.dynamic = true;
    const uint32_t instanceIndex = scene_->addInstance(inst);

    phys::Body body;
    body.shape = phys::Shape::makeCapsule(0.0062f, 0.0085f);
    body.position = position;
    body.orientation = quatFromEuler(Vec3(0.4f, 0.9f, 0.2f));
    body.linearVelocity = velocity;
    body.angularVelocity = Vec3(6.0f, 9.0f, 4.0f);
    body.material.restitution = 0.32f;
    body.material.density = 8500.0f;   // brass
    body.material.linearDamping = 0.05f;
    body.material.angularDamping = 0.22f;
    body.userData = 3;   // casing
    phys::World::computeMassProperties(body);
    const phys::BodyId id = world_.addBody(body);

    DynamicObject obj;
    obj.body = id;
    obj.instance = instanceIndex;
    dynamics_.push_back(obj);
}

void Game::updateDynamicObjects(float dt) {
    if (!scene_) return;
    int asleep = 0;
    int awake = 0;
    for (auto& obj : dynamics_) {
        if (!obj.alive) continue;
        const phys::Body* body = world_.getBody(obj.body);
        if (!body) {
            obj.alive = false;
            continue;
        }
        scene::Instance& inst = scene_->instance(obj.instance);
        inst.transform = mat4TRS(body->position, body->orientation, Vec3(1, 1, 1));
        inst.visible = true;
        scene_->updateInstanceTransform(obj.instance, inst.transform);

        if (body->isSleeping()) ++asleep;
        else ++awake;

        // Glass tinkles when a shard is still moving quickly and hitting something.
        obj.tinkleCooldown -= dt;
        const float speed = length(body->linearVelocity);
        if (obj.tinkleCooldown <= 0.0f && speed > 0.55f && audio_.isInitialised()) {
            const float volume = clamp(speed * 0.055f, 0.03f, 0.22f);
            audio::PlayParams params;
            params.volume = volume;
            params.pitch = 0.85f + 0.4f * std::fabs(std::sin(obj.body * 12.9898f));
            audio_.play(audio::SoundId::GlassTinkle, body->position, params);
            obj.tinkleCooldown = 0.22f;
        }
    }
    stats_.bodiesAsleep = asleep;
    stats_.bodiesAwake = awake;
}

void Game::updateWorldAudio(float dt) {
    if (!audio_.isInitialised()) return;
    audio_.setListener(player_.eyePosition(), player_.forward(), Vec3(0, 1, 0));
    audio_.update(dt);
}

// ---------------------------------------------------------------- firing
void Game::fire(const InputState& input, float dt) {
    (void)dt;
    if (stats_.roundsInMagazine <= 0) {
        playWeaponSound(audio::SoundId::DryFire, 0.7f, 1.0f);
        setEvent("Empty - press R to reload", 1.6f);
        return;
    }
    --stats_.roundsInMagazine;
    ++stats_.shotsFired;
    stats_.lastShotTime = simulationTime_;
    fireCooldown_ = kFireInterval;
    hudAmmoFlash_ = 0.25f;

    // --- weapon presentation ---
    slideOffset_ = 0.0f;
    slideVelocity_ = 0.0f;
    recoilKick_ = 0.032f;
    recoilPitch_ = 0.10f;
    viewRecoilPitch_ += 0.028f;
    viewRecoilYaw_ += (std::sin(simulationTime_ * 37.0f) * 0.5f + (input.mouseDeltaX > 0 ? 0.3f : -0.3f)) * 0.006f;
    muzzleFlash_ = kMuzzleFlashTime;
    triggerPull_ = 1.0f;
    hammerAngle_ = 0.0f;
    hammerCocked_ = false;
    fovRecoil_ = 1.0f;
    crosshairSpread_ = 1.7f;

    // --- sound ---
    playWeaponSound(audio::SoundId::Gunshot, 1.0f, 1.0f);
    playWeaponSound(audio::SoundId::SlideForward, 0.55f, 1.12f);

    // --- ballistics ---
    const Vec3 origin = player_.eyePosition();
    // A little dispersion so repeated shots are not pixel-identical.
    Rng rng(config_.seed + static_cast<uint32_t>(stats_.shotsFired) * 7919u);
    const float spread = 0.0022f;
    Vec3 direction = normalize(player_.forward() +
                               player_.right() * (rng.symmetric() * spread) +
                               player_.up() * (rng.symmetric() * spread));

    phys::RayHit hit = world_.raycast(origin, direction, 60.0f);
    if (hit.hit) {
        if (hit.userData == 1 && !stats_.glassBroken) {
            // ---- the glass shatters ----
            stats_.glassBroken = true;
            scene::ShatterParams params;
            params.center = room_.glassCenter;
            params.radius = room_.glassRadius;
            params.height = room_.glassHeight;
            params.wallThickness = 0.0055f;
            params.impactPoint = hit.point;
            params.impactDirection = direction;
            params.impactEnergy = 1.35f;
            params.shardCount = config_.shardCount;
            params.seed = config_.seed + 991u;
            std::vector<scene::ShardDesc> shards =
                scene::shatterGlass(*scene_, materials_, params);

            for (const scene::ShardDesc& shard : shards) {
                scene::Instance inst;
                inst.mesh = shard.mesh;
                inst.transform =
                    mat4TRS(shard.position, shard.orientation, Vec3(1, 1, 1));
                inst.castsShadow = false;
                inst.dynamic = true;
                const uint32_t instanceIndex = scene_->addInstance(inst);

                phys::Body body;
                body.shape = phys::Shape::makeBox(shard.halfExtents);
                body.position = shard.position;
                body.orientation = shard.orientation;
                body.linearVelocity = shard.linearVelocity;
                body.angularVelocity = shard.angularVelocity;
                body.material.density = params.density;
                // Glass on a hard floor is fairly lossy: a generous damping term keeps
                // the debris around the table instead of skittering into every corner.
                body.material.restitution = 0.18f;
                body.material.staticFriction = 0.58f;
                body.material.dynamicFriction = 0.46f;
                body.material.linearDamping = 0.42f;
                body.material.angularDamping = 0.55f;
                body.userData = 2;   // shard
                // Shards are spawned interpenetrating inside the glass shell, so they
                // must not collide with each other: the solver would otherwise blast
                // them apart. They still collide with the world and the furniture.
                body.collisionGroup = kShardGroup;
                body.collisionMask = ~kShardGroup;
                phys::World::computeMassProperties(body);
                const phys::BodyId id = world_.addBody(body);

                DynamicObject obj;
                obj.body = id;
                obj.instance = instanceIndex;
                obj.tinkleCooldown = 0.05f + 0.2f * std::fabs(std::sin(obj.body * 3.7f));
                dynamics_.push_back(obj);
            }
            stats_.shardCount = static_cast<int>(shards.size());

            // Remove the intact glass: hide it and drop its collider.
            scene::Instance& glass = scene_->instance(room_.glassInstance);
            glass.visible = false;
            glass.castsShadow = false;
            if (!glassBodyRemoved_) {
                world_.removeBody(glassBody_);
                glassBodyRemoved_ = true;
            }

            if (audio_.isInitialised()) {
                audio::PlayParams params2;
                params2.volume = 1.0f;
                params2.pitch = 1.0f;
                audio_.play(audio::SoundId::GlassShatter, hit.point, params2);
            }
            setEvent("Glass shattered", 2.5f);
            geometryDirty_ = true;
            R2_INFO("glass shattered by shot ", stats_.shotsFired, " at (", hit.point.x, ", ",
                    hit.point.y, ", ", hit.point.z, ")");
        } else {
            ++stats_.impacts;
            if (audio_.isInitialised()) {
                audio::PlayParams params;
                params.volume = 0.72f;
                params.pitch = 0.94f + 0.12f * std::fabs(std::sin(stats_.impacts * 2.3f));
                const bool wood = std::fabs(hit.normal.y) < 0.5f && hit.point.y < room_.tableCenter.y + 0.5f;
                audio_.play(wood ? audio::SoundId::ImpactWood : audio::SoundId::ImpactConcrete,
                            hit.point, params);
            }
        }
    }

    // Eject a casing to the right and slightly up.
    const Mat4 weapon = weaponTransform();
    // Model space: +z is the muzzle, +y up, so the ejection port is up and to the right
    // of the breech face, and casings leave to the right and slightly rearward.
    const Vec3 ejectPort = transformPoint(weapon, Vec3(-0.012f, 0.012f, 0.045f));
    const Vec3 ejectDir = normalize(transformDirection(weapon, Vec3(-0.75f, 0.42f, -0.12f)));
    spawnCartridge(ejectPort, ejectDir * 2.4f + Vec3(0, 0.35f, 0));
    if (audio_.isInitialised()) {
        audio::PlayParams params;
        params.volume = 0.5f;
        params.pitch = 1.0f;
        params.delaySeconds = 0.35f;
        audio_.play(audio::SoundId::ShellDrop, ejectPort, params);
    }
}

void Game::beginReload() {
    if (weaponState_ == WeaponState::Reloading || weaponState_ == WeaponState::Holstered ||
        weaponState_ == WeaponState::Holstering)
        return;
    if (stats_.roundsInMagazine >= 13 || stats_.roundsReserve <= 0) {
        setEvent(stats_.roundsReserve <= 0 ? "No spare magazines" : "Magazine full", 1.4f);
        return;
    }
    weaponState_ = WeaponState::Reloading;
    stateTime_ = 0.0f;
    reloadPhase_ = 0.0f;
    for (bool& played : reloadSoundsPlayed_) played = false;
    setEvent("Reloading", 2.4f);
}

void Game::updateWeapon(const InputState& input, float dt) {
    stateTime_ += dt;

    // ---- state transitions ----------------------------------------------
    switch (weaponState_) {
        case WeaponState::Holstered:
            drawProgress_ = 0.0f;
            // Both the trigger and Q draw the pistol.
            if (input.drawPressed || input.firePressed) {
                weaponState_ = WeaponState::Drawing;
                stateTime_ = 0.0f;
                playWeaponSound(audio::SoundId::WeaponDraw, 0.7f, 1.0f);
                playWeaponSound(audio::SoundId::SlideBack, 0.6f, 1.05f);
                setEvent("Pistol drawn", 1.4f);
            }
            break;
        case WeaponState::Drawing:
            drawProgress_ = clamp(stateTime_ / kDrawTime, 0.0f, 1.0f);
            if (stateTime_ >= kDrawTime) {
                weaponState_ = WeaponState::Ready;
                stateTime_ = 0.0f;
                playWeaponSound(audio::SoundId::SlideForward, 0.8f, 1.0f);
            }
            break;
        case WeaponState::Ready:
            drawProgress_ = 1.0f;
            // These are deliberately independent checks rather than an else-if chain:
            // holding the trigger while pressing R, or pressing Q while firing, should
            // not silently swallow the other action.
            if (input.drawPressed) {
                weaponState_ = WeaponState::Holstering;
                stateTime_ = 0.0f;
                playWeaponSound(audio::SoundId::WeaponHolster, 0.7f, 1.0f);
                setEvent("Pistol holstered", 1.2f);
            } else {
                if (input.reloadPressed) beginReload();
                // Holding the trigger keeps firing at the pistol's cyclic rate.
                if (weaponState_ == WeaponState::Ready && input.fireHeld &&
                    fireCooldown_ <= 0.0f) {
                    weaponState_ = WeaponState::Firing;
                    stateTime_ = 0.0f;
                    fire(input, dt);
                }
            }
            break;
        case WeaponState::Firing:
            drawProgress_ = 1.0f;
            if (stateTime_ >= 0.09f) {
                weaponState_ = WeaponState::Ready;
                stateTime_ = 0.0f;
            }
            break;
        case WeaponState::Reloading: {
            drawProgress_ = 1.0f;
            reloadPhase_ = clamp(stateTime_ / kReloadTime, 0.0f, 1.0f);
            // Stage 1: magazine release + magazine out.
            if (!reloadSoundsPlayed_[0] && stateTime_ > 0.18f) {
                reloadSoundsPlayed_[0] = true;
                playWeaponSound(audio::SoundId::MagRelease, 0.8f, 1.0f);
                playWeaponSound(audio::SoundId::MagOut, 0.8f, 1.0f);
            }
            // Stage 2: fresh magazine inserted and seated.
            if (!reloadSoundsPlayed_[1] && stateTime_ > 1.05f) {
                reloadSoundsPlayed_[1] = true;
                playWeaponSound(audio::SoundId::MagIn, 0.9f, 1.0f);
            }
            // Stage 3: slide released into battery.
            if (!reloadSoundsPlayed_[2] && stateTime_ > 1.62f) {
                reloadSoundsPlayed_[2] = true;
                playWeaponSound(audio::SoundId::SlideBack, 0.7f, 1.06f);
                playWeaponSound(audio::SoundId::SlideForward, 0.95f, 1.0f);
            }
            if (stateTime_ >= kReloadTime) {
                const int needed = 13 - stats_.roundsInMagazine;
                const int taken = std::min(needed, stats_.roundsReserve);
                stats_.roundsInMagazine += taken;
                stats_.roundsReserve -= taken;
                ++stats_.reloads;
                hammerCocked_ = true;
                weaponState_ = WeaponState::Ready;
                stateTime_ = 0.0f;
                reloadPhase_ = 0.0f;
                magazineDrop_ = 0.0f;
                setEvent("Reloaded", 1.2f);
            }
            break;
        }
        case WeaponState::Holstering:
            drawProgress_ = clamp(1.0f - stateTime_ / kHolsterTime, 0.0f, 1.0f);
            if (stateTime_ >= kHolsterTime) {
                weaponState_ = WeaponState::Holstered;
                stateTime_ = 0.0f;
                drawProgress_ = 0.0f;
            }
            break;
    }

    if (fireCooldown_ > 0.0f) fireCooldown_ -= dt;

    // ---- sub-part animation ---------------------------------------------
    // Slide: cycles back on firing, rides forward into battery.
    float slideTarget = 0.0f;
    if (muzzleFlash_ > 0.0f) slideTarget = 0.030f;
    if (weaponState_ == WeaponState::Reloading) {
        // The slide is racked during the last third of the reload.
        const float t = (stateTime_ - 1.55f) / 0.22f;
        if (t > 0.0f && t < 1.0f) slideTarget = std::sin(clamp(t, 0.0f, 1.0f) * kPi) * 0.032f;
    }
    const float stiffness = 950.0f;
    const float damping = 55.0f;
    slideVelocity_ += ((slideTarget - slideOffset_) * stiffness - slideVelocity_ * damping) * dt;
    slideOffset_ = clamp(slideOffset_ + slideVelocity_ * dt, 0.0f, 0.036f);

    // Trigger: pulled while firing, released afterwards.
    const float triggerTarget = (weaponState_ == WeaponState::Firing) ? 1.0f : 0.0f;
    triggerPull_ += (triggerTarget - triggerPull_) * dampf(26.0f, dt);

    // Hammer follows the slide / trigger.
    const float hammerTarget = hammerCocked_ ? 0.0f : 1.0f;
    hammerAngle_ += (hammerTarget - hammerAngle_) * dampf(30.0f, dt);

    // Recoil decay.
    recoilKick_ *= std::exp(-13.0f * dt);
    recoilPitch_ *= std::exp(-11.0f * dt);
    viewRecoilPitch_ *= std::exp(-7.5f * dt);
    viewRecoilYaw_ *= std::exp(-6.5f * dt);
    fovRecoil_ *= std::exp(-6.0f * dt);
    crosshairSpread_ += (1.0f - crosshairSpread_) * dampf(7.0f, dt);
    hudAmmoFlash_ = std::max(0.0f, hudAmmoFlash_ - dt);

    if (muzzleFlash_ > 0.0f) {
        muzzleFlash_ -= dt;
        muzzleFlashLightOn_ = true;
    } else {
        muzzleFlashLightOn_ = false;
    }

    // Magazine drops out and back in during the reload.
    if (weaponState_ == WeaponState::Reloading) {
        const float t = stateTime_;
        if (t < 0.45f) magazineDrop_ = smoothstepf(0.16f, 0.45f, t);
        else if (t < 1.0f) magazineDrop_ = 1.0f;
        else magazineDrop_ = 1.0f - smoothstepf(1.0f, 1.28f, t);
    } else {
        magazineDrop_ = 0.0f;
    }

    // ---- apply to the scene instances -----------------------------------
    if (!scene_) return;
    const bool visible = weaponState_ != WeaponState::Holstered || drawProgress_ > 0.01f;
    const Mat4 weapon = weaponTransform();
    for (uint32_t i = 0; i < weapon_.count; ++i) {
        Mat4 local = mat4Identity();
        switch (static_cast<scene::WeaponPart>(i)) {
            case scene::WeaponPart::Slide:
                local = mat4Translate(Vec3(0, 0, -slideOffset_));
                break;
            case scene::WeaponPart::Barrel:
                local = mat4Translate(Vec3(0, 0, -slideOffset_ * 0.12f));
                break;
            case scene::WeaponPart::Trigger:
                local = mat4TRS(Vec3(0, -0.0285f, 0.0105f),
                                quatFromAxisAngle(Vec3(1, 0, 0), triggerPull_ * 0.24f),
                                Vec3(1, 1, 1)) *
                        mat4Translate(Vec3(0, 0.0285f, -0.0105f));
                break;
            case scene::WeaponPart::Hammer:
                local = mat4TRS(Vec3(0, -0.0145f, -0.0125f),
                                quatFromAxisAngle(Vec3(1, 0, 0), -hammerAngle_ * 0.9f),
                                Vec3(1, 1, 1)) *
                        mat4Translate(Vec3(0, 0.0145f, 0.0125f));
                break;
            case scene::WeaponPart::Magazine:
            case scene::WeaponPart::MagazineFloorplate:
                local = mat4Translate(Vec3(0, -magazineDrop_ * 0.115f, 0));
                break;
            default:
                break;
        }
        // The slide/barrel also inherit a tiny amount of the frame's flex under recoil.
        const Mat4 world = weapon * local;
        scene::Instance& inst = scene_->instance(weaponPartInstances_[i]);
        inst.transform = world;
        inst.visible = visible;
        scene_->updateInstanceTransform(weaponPartInstances_[i], world);
    }
}

// ---------------------------------------------------------------- update
void Game::update(const InputState& input, float dt, render::UiContext& ui, uint32_t width,
                  uint32_t height) {
    lastInput_ = input;
    simulationTime_ += dt;
    stats_.simulationTime = simulationTime_;

    updateWeapon(input, dt);

    // View recoil is applied as an aim offset on top of the player's own look angles, so
    // the controller's state stays authoritative.
    player_.update(input, dt, world_);

    // Footsteps.
    footstepTimer_ -= dt;
    if (player_.takeFootstepCount() > 0 && footstepTimer_ <= 0.0f && audio_.isInitialised()) {
        audio::PlayParams params;
        params.volume = 0.42f;
        params.pitch = (stats_.shotsFired % 2 == 0) ? 1.02f : 0.95f;
        audio_.play(stats_.shotsFired % 2 == 0 ? audio::SoundId::FootstepAlt
                                               : audio::SoundId::Footstep,
                    player_.feetPosition(), params);
        footstepTimer_ = 0.14f;
    }

    // Physics.
    world_.step(dt);
    updateDynamicObjects(dt);
    updateWorldAudio(dt);

    // Keep the view-model fill attached to the camera.
    if (viewFillLight_ != scene::kInvalidIndex && scene_ &&
        viewFillLight_ < scene_->lights().size()) {
        scene_->light(viewFillLight_).position =
            player_.eyePosition() + player_.forward() * 0.22f + Vec3(0, -0.06f, 0);
    }

    // Camera follows the player.
    const float fov = (70.0f - fovRecoil_ * 1.6f) * DEG2RAD;
    camera_.setPerspective(fov, static_cast<float>(width) / std::max(1.0f, static_cast<float>(height)),
                           0.05f, 60.0f);
    camera_.setFromYawPitch(player_.eyePosition(), player_.yaw() + viewRecoilYaw_,
                            clamp(player_.pitch() + viewRecoilPitch_, -1.5f, 1.5f));
    camera_.setViewportSize(width, height);

    if (eventTimer_ > 0.0f) eventTimer_ -= dt;

    updateHud(ui, width, height);
}

void Game::updateHud(render::UiContext& ui, uint32_t width, uint32_t height) {
    ui.clear();
    const float w = static_cast<float>(width);
    const float h = static_cast<float>(height);
    // Keep the HUD the same physical size regardless of the render resolution.
    const float s = clamp(static_cast<float>(height) / 720.0f, 0.85f, 2.4f);
    const float pad = 26.0f * s;
    const Vec4 white(0.94f, 0.95f, 0.97f, 1.0f);
    const Vec4 dim(0.75f, 0.78f, 0.82f, 0.85f);
    const Vec4 accent(1.0f, 0.86f, 0.55f, 1.0f);

    // --- crosshair ---
    if (weaponState_ == WeaponState::Ready || weaponState_ == WeaponState::Firing) {
        const float radius = (9.0f + crosshairSpread_ * 8.0f) * s;
        const float gap = (2.5f + crosshairSpread_ * 2.5f) * s;
        ui.crosshair(Vec2(w * 0.5f, h * 0.5f), radius, gap, std::max(1.0f, 1.6f * s),
                     Vec4(0.95f, 0.96f, 0.98f, 0.85f));
    } else {
        ui.crosshair(Vec2(w * 0.5f, h * 0.5f), 6.0f * s, 3.0f * s, std::max(1.0f, 1.6f * s),
                     Vec4(0.9f, 0.92f, 0.95f, 0.35f));
    }

    // --- ammo readout ---
    const float ammoScale = 3.1f * s;
    const float reserveScale = 1.8f * s;
    char ammoText[64];
    char reserveText[64];
    std::snprintf(ammoText, sizeof(ammoText), "%d", stats_.roundsInMagazine);
    std::snprintf(reserveText, sizeof(reserveText), "/ %d", stats_.roundsReserve);
    const float ammoW = ui.textWidth(ammoText, ammoScale);
    const float reserveW = ui.textWidth(reserveText, reserveScale);
    const float groupRight = w - 26.0f * s;
    const float ammoBaseline = h - pad - 58.0f * s;
    const Vec4 ammoColor = hudAmmoFlash_ > 0.0f ? accent : white;
    ui.text(Vec2(groupRight - reserveW - ammoW - 10.0f * s, ammoBaseline), ammoText, ammoColor,
            ammoScale);
    ui.text(Vec2(groupRight - reserveW, ammoBaseline + ui.textHeight(ammoScale) * 0.66f),
            reserveText, dim, reserveScale);

    // --- state line ---
    char stateText[128];
    std::snprintf(stateText, sizeof(stateText), "%s", weaponStateName(weaponState_));
    ui.textRight(Vec2(groupRight, ammoBaseline + ui.textHeight(ammoScale) + 8.0f * s),
                 stateText, dim, 1.4f * s);

    // --- hints ---
    const char* hint = nullptr;
    if (weaponState_ == WeaponState::Holstered) {
        hint = "LMB DRAW      Q DRAW";
    } else if (stats_.glassBroken) {
        hint = "LMB FIRE      R RELOAD      Q HOLSTER";
    } else {
        hint = "LMB FIRE      R RELOAD      Q HOLSTER      TAB CURSOR      F5 INPUT";
    }
    ui.text(Vec2(pad, h - pad - 16.0f * s), hint, dim, 1.4f * s);

    // --- event banner ---
    if (eventTimer_ > 0.0f && !eventText_.empty()) {
        const float alpha = clamp(eventTimer_ * 2.0f, 0.0f, 1.0f);
        ui.text(Vec2(pad, pad + 96.0f * s), eventText_, Vec4(white.x, white.y, white.z, alpha), 1.9f * s);
    }

    // --- objective / status panel ---
    ui.text(Vec2(pad, pad),
            stats_.glassBroken ? "OBJECTIVE COMPLETE: GLASS SHATTERED"
                               : "OBJECTIVE: SHOOT THE GLASS",
            stats_.glassBroken ? Vec4(0.6f, 1.0f, 0.7f, 0.95f) : Vec4(0.95f, 0.9f, 0.7f, 0.9f),
            1.4f * s);

    if (stats_.shardCount > 0) {
        char shardText[96];
        std::snprintf(shardText, sizeof(shardText), "SHARDS %d   AWAKE %d   ASLEEP %d",
                      stats_.shardCount, stats_.bodiesAwake, stats_.bodiesAsleep);
        ui.text(Vec2(pad, pad + 22.0f * s), shardText, dim, 1.2f * s);
    }

    // --- live input diagnostics (F5) ---
    if (inputDebug_) {
        char buf[192];
        const InputState& i = lastInput_;
        std::snprintf(buf, sizeof(buf), "MOVE %c%c%c%c RUN %d JUMP %d CROUCH %d",
                      i.forward ? 'W' : '-', i.left ? 'A' : '-', i.back ? 'S' : '-',
                      i.right ? 'D' : '-', i.run ? 1 : 0, i.jump ? 1 : 0, i.crouch ? 1 : 0);
        ui.text(Vec2(pad, pad + 70.0f * s), buf, Vec4(0.7f, 1.0f, 0.8f, 0.9f), 1.2f * s);
        std::snprintf(buf, sizeof(buf), "MOUSE d(%+.0f,%+.0f)  FIRE %d/%d  RELOAD %d  DRAW %d",
                      i.mouseDeltaX, i.mouseDeltaY, i.firePressed ? 1 : 0, i.fireHeld ? 1 : 0,
                      i.reloadPressed ? 1 : 0, i.drawPressed ? 1 : 0);
        ui.text(Vec2(pad, pad + 88.0f * s), buf, Vec4(0.7f, 1.0f, 0.8f, 0.9f), 1.2f * s);
    }

    // --- performance ---
    const render::FrameStats& fs = renderer_ ? renderer_->stats() : render::FrameStats{};
    char perfText[160];
    std::snprintf(perfText, sizeof(perfText), "%.1f MS  DRAWS %u  TRIS %u", fs.cpuFrameMs,
                  fs.drawCalls, fs.triangles);
    ui.text(Vec2(pad, pad + 44.0f * s), perfText, Vec4(dim.x, dim.y, dim.z, 0.65f), 1.1f * s);
}

}  // namespace room2::game
