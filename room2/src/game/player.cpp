#include "game/player.hpp"

#include <algorithm>

namespace room2::game {

void Player::init(const PlayerConfig& config, Vec3 feetPosition, float yaw) {
    config_ = config;
    position_ = feetPosition;
    yaw_ = yaw;
    pitch_ = 0.0f;
    velocity_ = Vec3(0, 0, 0);
    grounded_ = false;
    crouching_ = false;
    eyeHeight_ = config_.eyeHeight;
}

float Player::currentEyeHeight() const {
    return lerpf(config_.crouchEyeHeight, config_.eyeHeight, eyeHeight_ / config_.eyeHeight);
}

Vec3 Player::forward() const {
    const float cp = std::cos(pitch_);
    return normalize(Vec3(std::sin(yaw_) * cp, std::sin(pitch_), -std::cos(yaw_) * cp));
}

Vec3 Player::right() const {
    return normalize(Vec3(std::cos(yaw_), 0.0f, std::sin(yaw_)));
}

void Player::update(const InputState& input, float dt, const phys::World& world) {
    dt = clamp(dt, 0.0f, 0.1f);

    // ---- look ------------------------------------------------------------
    yaw_ += input.mouseDeltaX * config_.mouseSensitivity;
    pitch_ = clamp(pitch_ - input.mouseDeltaY * config_.mouseSensitivity, -config_.maxPitch,
                   config_.maxPitch);
    if (yaw_ > PI) yaw_ -= TWO_PI;
    if (yaw_ < -PI) yaw_ += TWO_PI;

    // ---- crouch / stance -------------------------------------------------
    const bool wantCrouch = input.crouch;
    crouching_ = wantCrouch;
    const float targetEye = crouching_ ? config_.crouchEyeHeight : config_.eyeHeight;
    eyeHeight_ += (targetEye - eyeHeight_) * dampf(12.0f, dt);
    const float targetHeight = crouching_ ? config_.crouchHeight : config_.height;

    // ---- horizontal movement --------------------------------------------
    const Vec3 flatForward = normalize(Vec3(std::sin(yaw_), 0.0f, -std::cos(yaw_)));
    const Vec3 flatRight = right();
    Vec3 wish(0, 0, 0);
    if (input.forward) wish += flatForward;
    if (input.back) wish -= flatForward;
    if (input.right) wish += flatRight;
    if (input.left) wish -= flatRight;
    if (lengthSq(wish) > 1e-6f) wish = normalize(wish);

    float targetSpeed = config_.walkSpeed;
    if (crouching_) targetSpeed = config_.crouchSpeed;
    else if (input.run) targetSpeed = config_.runSpeed;

    // Accelerate towards the wish velocity, with ground friction when there is no input.
    const Vec3 desired = wish * targetSpeed;
    const float accel = grounded_ ? config_.acceleration : config_.acceleration * 0.28f;
    velocity_.x += (desired.x - velocity_.x) * clamp(accel * dt / std::max(targetSpeed, 1e-3f), 0.0f, 1.0f);
    velocity_.z += (desired.z - velocity_.z) * clamp(accel * dt / std::max(targetSpeed, 1e-3f), 0.0f, 1.0f);
    if (lengthSq(wish) < 1e-6f && grounded_) {
        const float damping = std::exp(-config_.friction * dt);
        velocity_.x *= damping;
        velocity_.z *= damping;
    }

    // ---- vertical --------------------------------------------------------
    velocity_.y += config_.gravity * dt;
    if (input.jump && grounded_ && !crouching_) {
        velocity_.y = config_.jumpSpeed;
        grounded_ = false;
    }
    if (velocity_.y < -32.0f) velocity_.y = -32.0f;

    // ---- collide and slide ----------------------------------------------
    const float radius = config_.radius;
    const float halfHeight = std::max(targetHeight * 0.5f - radius, 0.02f);
    const Vec3 delta = velocity_ * dt;
    phys::World::CharacterMoveResult move =
        world.moveCapsule(position_ + Vec3(0, radius + halfHeight, 0), radius, halfHeight, delta,
                          config_.stepHeight, 0.5f);

    // The controller returns the capsule centre; convert back to the feet position.
    const Vec3 newPosition = move.position - Vec3(0, radius + halfHeight, 0);
    const Vec3 actualDelta = newPosition - position_;
    position_ = newPosition;

    grounded_ = move.grounded;
    lastGroundNormal_ = move.groundNormal;
    if (grounded_ && velocity_.y < 0.0f) velocity_.y = 0.0f;
    if (move.hitCeiling && velocity_.y > 0.0f) velocity_.y = 0.0f;
    // Cancelled horizontal motion means we hit a wall: drop that velocity component so
    // the player does not keep accelerating into it.
    if (std::fabs(actualDelta.x) < std::fabs(delta.x) * 0.35f) velocity_.x *= 0.2f;
    if (std::fabs(actualDelta.z) < std::fabs(delta.z) * 0.35f) velocity_.z *= 0.2f;

    // ---- head bob and weapon sway ---------------------------------------
    const float planarSpeed = speed();
    if (grounded_ && planarSpeed > 0.25f) {
        const float rate = config_.bobFrequency * (crouching_ ? 0.72f : 1.0f) * (input.run ? 1.22f : 1.0f);
        bobPhase_ += rate * dt * clamp(planarSpeed / config_.walkSpeed, 0.4f, 1.6f);
        const float amplitude = config_.bobAmplitude * clamp(planarSpeed / config_.walkSpeed, 0.0f, 1.4f);
        bobOffset_ = Vec3(std::sin(bobPhase_) * amplitude * 0.85f,
                          -std::fabs(std::cos(bobPhase_)) * amplitude,
                          std::cos(bobPhase_ * 0.5f) * amplitude * 0.4f);
        // Footsteps land on the beat where the head is lowest.
        stepDistance_ += planarSpeed * dt;
        if (stepDistance_ > 0.72f) {
            stepDistance_ = 0.0f;
            ++footsteps_;
            wantsFootstep_ = true;
        }
    } else {
        bobOffset_ *= std::exp(-6.0f * dt);
        wantsFootstep_ = false;
    }

    // ---- weapon sway ------------------------------------------------------
    // The view model lags the view by an amount proportional to how fast you are
    // turning. The spring is integrated in closed form rather than with an explicit
    // Euler step: Euler is frame-rate dependent here and rings badly at high frame
    // rates, which shows up as the weapon visibly shaking.
    const float aimYaw = input.mouseDeltaX * config_.mouseSensitivity;
    const float aimPitch = -input.mouseDeltaY * config_.mouseSensitivity;
    const Vec2 swayTarget(clamp(-aimYaw * 1.5f, -config_.maxSway, config_.maxSway),
                          clamp(-aimPitch * 1.5f, -config_.maxSway, config_.maxSway));
    auto criticallyDampedStep = [dt](float& x, float& v, float target) {
        const float omega = 13.0f;             // response rate, rad/s
        const float d0 = x - target;
        const float b = v + omega * d0;
        const float decay = std::exp(-omega * dt);
        const float d1 = (d0 + b * dt) * decay;
        v = (b - omega * (d0 + b * dt)) * decay;
        x = target + d1;
    };
    criticallyDampedStep(swayOffset_.x, swayVelocity_.x, swayTarget.x);
    criticallyDampedStep(swayOffset_.y, swayVelocity_.y, swayTarget.y);
}

}  // namespace room2::game
