// room2 - first person player controller.
#pragma once

#include "core/math.hpp"
#include "physics/physics.hpp"

namespace room2::game {

struct InputState {
    bool forward = false;
    bool back = false;
    bool left = false;
    bool right = false;
    bool run = false;
    bool crouch = false;
    bool jump = false;
    float mouseDeltaX = 0.0f;   // pixels
    float mouseDeltaY = 0.0f;
    bool firePressed = false;
    bool fireHeld = false;
    bool reloadPressed = false;
    bool drawPressed = false;    // draw / holster toggle
    bool interactPressed = false;
};

struct PlayerConfig {
    float eyeHeight = 1.62f;       // above the feet
    float crouchEyeHeight = 1.05f;
    float radius = 0.30f;          // capsule radius
    float height = 1.78f;          // total capsule height standing
    float crouchHeight = 1.25f;
    float walkSpeed = 2.6f;        // m/s
    float runSpeed = 4.4f;
    float crouchSpeed = 1.3f;
    float acceleration = 42.0f;
    float friction = 12.0f;
    float jumpSpeed = 3.2f;
    float gravity = -18.0f;
    float mouseSensitivity = 0.0022f;   // radians per pixel
    float maxPitch = 1.50f;
    float stepHeight = 0.36f;
    float bobFrequency = 8.4f;
    float bobAmplitude = 0.009f;
    // Maximum view-model lag, in metres. Small: the pistol is only 19 cm long.
    float maxSway = 0.018f;
};

class Player {
public:
    void init(const PlayerConfig& config, Vec3 feetPosition, float yaw);
    // Advances the player. `world` provides the collision geometry.
    void update(const InputState& input, float dt, const phys::World& world);

    Vec3 eyePosition() const { return position_ + Vec3(0, currentEyeHeight(), 0); }
    Vec3 feetPosition() const { return position_; }
    Vec3 forward() const;
    Vec3 right() const;
    Vec3 up() const { return Vec3(0, 1, 0); }
    float yaw() const { return yaw_; }
    float pitch() const { return pitch_; }
    void setYaw(float yaw) { yaw_ = yaw; }
    void setPitch(float pitch) { pitch_ = clamp(pitch, -config_.maxPitch, config_.maxPitch); }
    void setPosition(Vec3 feet) { position_ = feet; }

    Vec3 velocity() const { return velocity_; }
    bool grounded() const { return grounded_; }
    bool crouching() const { return crouching_; }
    float speed() const { return length(Vec3(velocity_.x, 0, velocity_.z)); }
    float currentEyeHeight() const;

    // Accumulated view bob/sway offsets (metres) for the weapon view model.
    Vec3 viewBob() const { return bobOffset_; }
    Vec2 viewSway() const { return swayOffset_; }

    // Number of footsteps triggered since the last call (for audio).
    int takeFootstepCount() { const int n = footsteps_; footsteps_ = 0; return n; }
    // Set true while the player is moving on the ground, for footstep audio.
    bool wantsFootstep() const { return wantsFootstep_; }

private:
    PlayerConfig config_{};
    Vec3 position_{0, 0, 0};
    Vec3 velocity_{0, 0, 0};
    float yaw_ = 0.0f;
    float pitch_ = 0.0f;
    bool grounded_ = false;
    bool crouching_ = false;
    float eyeHeight_ = 1.62f;
    float bobPhase_ = 0.0f;
    Vec3 bobOffset_{0, 0, 0};
    Vec2 swayOffset_{0, 0};
    Vec2 swayVelocity_{0, 0};
    float stepDistance_ = 0.0f;
    int footsteps_ = 0;
    bool wantsFootstep_ = false;
    Vec3 lastGroundNormal_{0, 1, 0};
};

}  // namespace room2::game
