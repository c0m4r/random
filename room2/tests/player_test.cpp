// room2 - regression test for the first person controller's view-model motion.
//
// The weapon used to visibly shake: the sway spring was integrated with an explicit
// Euler step whose position update was scaled by a constant 60, which makes the damping
// ratio frame-rate dependent (it rings badly above ~100 fps). This test pins the
// behaviour down at several frame rates.
#include <cmath>
#include <cstdio>
#include <vector>

#include "game/player.hpp"
#include "physics/physics.hpp"

using namespace room2;

namespace {

int g_failures = 0;

void check(bool condition, const char* what, const char* detail = "") {
    if (condition) {
        std::printf("  [PASS] %s %s\n", what, detail);
    } else {
        std::printf("  [FAIL] %s %s\n", what, detail);
        ++g_failures;
    }
}

// phys::World is non-copyable (its scratch buffers are reused), so tests construct one
// in place rather than returning it by value.
void addFloor(phys::World& world) {
    phys::Body floor;
    floor.shape = phys::Shape::makeBox(Vec3(20.0f, 0.5f, 20.0f));
    floor.position = Vec3(0, -0.5f, 0);
    floor.flags = phys::BODY_STATIC;
    world.addBody(floor);
}

phys::WorldConfig testConfig() {
    phys::WorldConfig cfg;
    cfg.substeps = 1;
    return cfg;
}

// Flicks the view once, then measures the decay of the view-model sway.
struct SwayResult {
    float peak = 0.0f;
    float finalMagnitude = 0.0f;
    int signChanges = 0;
    float limitViolation = 0.0f;
};

SwayResult runSway(float fps, float maxSway) {
    phys::World world(testConfig());
    addFloor(world);
    game::Player player;
    game::PlayerConfig pc;
    player.init(pc, Vec3(0, 0.02f, 0), 0.0f);

    const float dt = 1.0f / fps;
    // Settle on the ground first.
    game::InputState idle;
    for (int i = 0; i < 120; ++i) player.update(idle, dt, world);

    SwayResult result;
    float last = 0.0f;
    int lastSign = 0;
    // One frame of fast mouse movement, then nothing.
    game::InputState flick;
    flick.mouseDeltaX = 60.0f;
    flick.mouseDeltaY = -40.0f;
    player.update(flick, dt, world);

    for (int i = 0; i < static_cast<int>(fps * 2.0f); ++i) {
        game::InputState none;
        player.update(none, dt, world);
        const Vec2 sway = player.viewSway();
        const float magnitude = std::sqrt(sway.x * sway.x + sway.y * sway.y);
        result.peak = std::max(result.peak, magnitude);
        result.limitViolation =
            std::max(result.limitViolation,
                     std::max(std::fabs(sway.x), std::fabs(sway.y)) - maxSway);
        const int sign = sway.x > 1e-5f ? 1 : (sway.x < -1e-5f ? -1 : 0);
        if (sign != 0 && lastSign != 0 && sign != lastSign) ++result.signChanges;
        if (sign != 0) lastSign = sign;
        last = magnitude;
    }
    result.finalMagnitude = last;
    return result;
}

}  // namespace

int main() {
    std::printf("room2 player / view-model tests\n");

    // A view-model sway must stay inside its configured limit at every frame rate.
    std::printf("-- sway stays within its limit --\n");
    for (float fps : {30.0f, 60.0f, 90.0f, 144.0f, 240.0f}) {
        const game::PlayerConfig pc;
        const SwayResult r = runSway(fps, pc.maxSway);
        char detail[128];
        std::snprintf(detail, sizeof(detail), "(%.0f fps: peak %.4f m, overshoot %.5f m)", fps,
                      r.peak, r.limitViolation);
        check(r.limitViolation < 1e-4f, "sway never exceeds maxSway", detail);
    }

    // The spring must be critically damped: a single flick should not ring. The old
    // frame-rate-dependent integrator changed sign many times at high frame rates.
    std::printf("-- sway is critically damped (no ringing) --\n");
    for (float fps : {30.0f, 60.0f, 90.0f, 144.0f, 240.0f}) {
        const game::PlayerConfig pc;
        const SwayResult r = runSway(fps, pc.maxSway);
        char detail[128];
        std::snprintf(detail, sizeof(detail), "(%.0f fps: %d direction changes)", fps,
                      r.signChanges);
        check(r.signChanges <= 1, "a single flick does not oscillate", detail);
    }

    // And it must actually settle, at every frame rate.
    std::printf("-- sway settles --\n");
    for (float fps : {30.0f, 60.0f, 90.0f, 144.0f, 240.0f}) {
        const game::PlayerConfig pc;
        const SwayResult r = runSway(fps, pc.maxSway);
        char detail[128];
        std::snprintf(detail, sizeof(detail), "(%.0f fps: residual %.6f m after 2 s)", fps,
                      r.finalMagnitude);
        check(r.finalMagnitude < 1e-4f, "sway decays to rest", detail);
    }

    // The same input must produce the same sway regardless of frame rate, within a
    // small tolerance: the whole point of integrating the spring in closed form.
    std::printf("-- frame-rate independence --\n");
    {
        const game::PlayerConfig pc;
        const SwayResult a = runSway(60.0f, pc.maxSway);
        const SwayResult b = runSway(240.0f, pc.maxSway);
        const float delta = std::fabs(a.peak - b.peak);
        char detail[128];
        std::snprintf(detail, sizeof(detail), "(peak 60 fps %.5f vs 240 fps %.5f, delta %.5f)", a.peak,
                      b.peak, delta);
        check(delta < 0.004f, "peak sway is frame-rate independent", detail);
    }

    // Head bob must not drift while standing still.
    std::printf("-- idle head bob decays --\n");
    {
        phys::World world(testConfig());
        addFloor(world);
        game::Player player;
        game::PlayerConfig pc;
        player.init(pc, Vec3(0, 0.02f, 0), 0.0f);
        game::InputState in;
        in.forward = true;
        for (int i = 0; i < 120; ++i) player.update(in, 1.0f / 60.0f, world);
        const float moving = length(player.viewBob());
        in.forward = false;
        for (int i = 0; i < 180; ++i) player.update(in, 1.0f / 60.0f, world);
        const float idle = length(player.viewBob());
        char detail[128];
        std::snprintf(detail, sizeof(detail), "(bob while moving %.5f m -> idle %.6f m)", moving,
                      idle);
        check(moving > 0.001f && idle < 0.0005f, "bob stops when the player stops", detail);
    }

    std::printf("-- movement, look and fire are independent --\n");
    {
        phys::World world(testConfig());
        addFloor(world);
        game::Player player;
        game::PlayerConfig pc;
        const Vec3 start(0, 0.02f, 0);
        player.init(pc, start, 0.0f);
        game::InputState in;
        in.forward = true;
        in.fireHeld = true;
        in.reloadPressed = true;
        in.mouseDeltaX = 8.0f;
        in.mouseDeltaY = -3.0f;
        for (int i = 0; i < 60; ++i) player.update(in, 1.0f / 60.0f, world);
        const float moved = distance(player.feetPosition(), start);
        const bool looked = std::fabs(player.yaw()) > 0.01f && std::fabs(player.pitch()) > 0.01f;
        char detail[160];
        std::snprintf(detail, sizeof(detail), "(moved %.3f m while looking, yaw %.3f pitch %.3f)",
                      moved, player.yaw(), player.pitch());
        check(moved > 0.5f && looked, "moving, looking and firing happen together", detail);
    }

    std::printf("=====================================================================\n");
    if (g_failures == 0) {
        std::printf("all player checks passed\n");
        return 0;
    }
    std::printf("%d check(s) FAILED\n", g_failures);
    return 1;
}
