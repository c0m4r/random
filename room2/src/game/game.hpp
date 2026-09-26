// room2 - gameplay: player, weapon handling, ballistics and destruction.
#pragma once

#include <string>
#include <vector>

#include "audio/audio.hpp"
#include "game/player.hpp"
#include "physics/physics.hpp"
#include "render/renderer.hpp"
#include "render/ui.hpp"
#include "scene/builders.hpp"
#include "scene/scene.hpp"
#include "scene/shatter.hpp"

namespace room2::game {

enum class WeaponState : uint32_t {
    Holstered = 0,
    Drawing,
    Ready,
    Firing,
    Reloading,
    Holstering,
};

const char* weaponStateName(WeaponState state);

struct GameConfig {
    uint32_t seed = 20240926u;
    int shardCount = 110;
    bool enableAudio = true;
    float masterVolume = 0.8f;
    bool startWithWeaponDrawn = false;
    // Scripted self-test: plays a fixed sequence and records the outcome.
    bool selfTest = false;
    // Procedural texture resolution for the scene materials.
    uint32_t textureSize = 1024;
};

// A physics body per shard / casing, plus the scene instance that draws it.
struct DynamicObject {
    phys::BodyId body = phys::INVALID_BODY;
    uint32_t instance = scene::kInvalidIndex;
    bool alive = true;
    float tinkleCooldown = 0.0f;
};

struct GameStats {
    int shotsFired = 0;
    int roundsInMagazine = 13;
    int roundsReserve = 39;
    int reloads = 0;
    int impacts = 0;
    bool glassBroken = false;
    int shardCount = 0;
    int bodiesAsleep = 0;
    int bodiesAwake = 0;
    float simulationTime = 0.0f;
    float lastShotTime = -1.0f;
};

class Game {
public:
    Game() = default;
    ~Game();

    Game(const Game&) = delete;
    Game& operator=(const Game&) = delete;

    // Builds the scene contents. `renderer` is needed for texture uploads.
    bool init(scene::Scene& scene, render::Renderer& renderer, const GameConfig& config);
    void shutdown();

    // Advances one frame. `ui` receives the HUD.
    void update(const InputState& input, float dt, render::UiContext& ui, uint32_t width,
                uint32_t height);
    void updateHud(render::UiContext& ui, uint32_t width, uint32_t height);

    render::Camera& camera() { return camera_; }
    const render::Camera& camera() const { return camera_; }
    const GameStats& stats() const { return stats_; }
    const Player& player() const { return player_; }
    WeaponState weaponState() const { return weaponState_; }
    // F5 overlay: shows exactly what input the game is receiving.
    void setInputDebug(bool enabled) { inputDebug_ = enabled; }
    bool glassIntact() const { return !stats_.glassBroken; }
    audio::Engine& audioEngine() { return audio_; }
    const std::string& lastEvent() const { return lastEvent_; }
    // World-space bounds of every live shard/casing, for diagnostics.
    Aabb shardBounds() const;
    // True once after geometry has been appended (shards); the renderer must re-upload.
    bool consumeGeometryDirty() {
        const bool d = geometryDirty_;
        geometryDirty_ = false;
        return d;
    }
    // Mean and maximum distance of a shard from the glass, plus the fastest speed.
    void shardSpreadStats(float& meanDistance, float& maxDistance, float& maxSpeed) const;

    // Scene handles the renderer needs to know about.
    const scene::RoomBuildResult& room() const { return room_; }

private:
    void buildMaterials();
    void buildPhysicsWorld();
    void spawnCartridge(Vec3 position, Vec3 velocity);
    void fire(const InputState& input, float dt);
    void beginReload();
    void updateWeapon(const InputState& input, float dt);
    void updateDynamicObjects(float dt);
    void updateWorldAudio(float dt);
    Mat4 weaponTransform() const;
    void playWeaponSound(audio::SoundId id, float volume = 1.0f, float pitch = 1.0f);
    void setEvent(std::string text, float duration = 3.0f);

    scene::Scene* scene_ = nullptr;
    render::Renderer* renderer_ = nullptr;
    GameConfig config_{};
    scene::RoomBuildResult room_;
    scene::SceneMaterials materials_{};
    scene::WeaponMeshSet weapon_{};
    uint32_t casingMesh_ = scene::kInvalidIndex;

    audio::Engine audio_;
    phys::World world_;
    Player player_;
    render::Camera camera_;

    // Weapon animation state.
    WeaponState weaponState_ = WeaponState::Holstered;
    float stateTime_ = 0.0f;
    float drawProgress_ = 0.0f;
    float slideOffset_ = 0.0f;
    float slideVelocity_ = 0.0f;
    float triggerPull_ = 0.0f;
    float hammerAngle_ = 0.0f;
    float magazineDrop_ = 0.0f;
    float recoilKick_ = 0.0f;      // metres, weapon pushed back
    float recoilPitch_ = 0.0f;     // radians
    float muzzleFlash_ = 0.0f;
    bool  muzzleFlashLightOn_ = false;
    float reloadPhase_ = 0.0f;
    bool  reloadSoundsPlayed_[4] = {false, false, false, false};
    bool  hammerCocked_ = true;

    // Player-facing recoil accumulated onto the view.
    float viewRecoilPitch_ = 0.0f;
    float viewRecoilYaw_ = 0.0f;
    float fireCooldown_ = 0.0f;

    uint32_t weaponPartInstances_[static_cast<uint32_t>(scene::WeaponPart::Count)] = {};
    Vec3 weaponPartsBasePosition_[static_cast<uint32_t>(scene::WeaponPart::Count)] = {};

    GameStats stats_{};
    std::string lastEvent_;
    std::vector<DynamicObject> dynamics_;
    uint32_t glassInteriorInstance_ = scene::kInvalidIndex;
    phys::BodyId glassBody_ = phys::INVALID_BODY;
    bool glassBodyRemoved_ = false;

    // Presentation state.
    float fovRecoil_ = 0.0f;
    float crosshairSpread_ = 1.0f;
    float hudAmmoFlash_ = 0.0f;
    std::string eventText_;
    float eventTimer_ = 0.0f;
    float shardHitTimer_ = 0.0f;
    int lastReloadSoundStage_ = -1;
    float simulationTime_ = 0.0f;
    float footstepTimer_ = 0.0f;
    bool geometryDirty_ = false;
    bool inputDebug_ = false;
    InputState lastInput_{};
    uint32_t viewFillLight_ = scene::kInvalidIndex;
};

}  // namespace room2::game
