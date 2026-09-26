#!/usr/bin/env bash
# Builds room2, runs every module test suite, then verifies the renderer and the
# gameplay loop. On a machine without a usable GPU it falls back to the vendored
# software Vulkan driver and validation layers in .tools/.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BUILD_DIR="${BUILD_DIR:-build}"
RES="${RES:-960x540}"
TEX="${TEX:-512}"
SHADOW="${SHADOW:-1024}"
PROBE="${PROBE:-64}"
SHOTS="${SHOTS:-shots}"
FAILURES=0

step() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
fail() { printf '\033[31mFAILED: %s\033[0m\n' "$*"; FAILURES=$((FAILURES + 1)); }
ok()   { printf '\033[32mok: %s\033[0m\n' "$*"; }

# ---------------------------------------------------------------- GPU selection
GPU_ARGS=()
if [ ! -d /dev/dri ] && [ ! -e /dev/nvidia0 ]; then
    if [ -f "$ROOT/.tools/env.sh" ]; then
        echo "no GPU device nodes found - using the vendored software Vulkan driver"
        # shellcheck disable=SC1091
        source "$ROOT/.tools/env.sh"
        GPU_ARGS=(--no-validation)
    else
        fail "no GPU and no vendored software driver; run scripts/fetch_tools.sh first"
        exit 1
    fi
fi

# ---------------------------------------------------------------- build
step "configure and build"
cmake -S . -B "$BUILD_DIR" -G Ninja -DCMAKE_BUILD_TYPE=Release >/dev/null || { fail "cmake configure"; exit 1; }
cmake --build "$BUILD_DIR" >/dev/null || { fail "build"; exit 1; }
ok "built $BUILD_DIR/room2"

# ---------------------------------------------------------------- module tests
step "module unit tests"
if (cd "$BUILD_DIR" && ctest --output-on-failure); then
    ok "all module tests passed"
else
    fail "module tests"
fi

W="${RES%x*}"
H="${RES#*x}"

# ---------------------------------------------------------------- validation-clean render
step "offscreen render with validation enabled"
mkdir -p "$SHOTS"
RENDER_LOG="$BUILD_DIR/verify_render.log"
"$BUILD_DIR/room2" --headless --screenshot "$SHOTS/verify_room.png" \
    --width "$W" --height "$H" --texture-size "$TEX" --shadow-resolution "$SHADOW" \
    --probe-size "$PROBE" --probe-samples 32 --frames 12 >"$RENDER_LOG" 2>&1
if grep -q "SCREENSHOT" "$RENDER_LOG"; then
    grep -E "SCREENSHOT|mean luma|frames=" "$RENDER_LOG" | sed 's/^/    /'
    ok "rendered $SHOTS/verify_room.png"
else
    fail "render produced no screenshot"
    tail -20 "$RENDER_LOG"
fi
if grep -qE "^\[ERROR\].*\[vk" "$RENDER_LOG"; then
    fail "validation errors were reported:"
    grep -E "^\[ERROR\].*\[vk" "$RENDER_LOG" | head -10 | sed 's/^/    /'
else
    ok "no Vulkan validation errors"
fi

# ---------------------------------------------------------------- windowed present path
# The swapchain/present path is only exercised when a window exists. This caught a real
# hang (the renderer signalled its own semaphore while present waited on the swapchain's)
# that the headless render could never have found, so it is a permanent check.
step "windowed render + present"
WINDOWED_LOG="$BUILD_DIR/verify_windowed.log"
if [ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]; then
    "$BUILD_DIR/room2" --windowed-shot "$SHOTS/verify_windowed.png" --frames 30 \
        --width 640 --height 360 --texture-size "$TEX" --shadow-resolution "$SHADOW" \
        --probe-size "$PROBE" --probe-samples 8 "${GPU_ARGS[@]}" >"$WINDOWED_LOG" 2>&1
    if grep -q "WINDOWED SHOT" "$WINDOWED_LOG"; then
        grep -E "windowed run complete|WINDOWED SHOT" "$WINDOWED_LOG" | sed 's/^/    /'
        ok "30 frames rendered and presented"
    else
        fail "windowed run did not present any frame (hang?)"
        tail -20 "$WINDOWED_LOG" | sed 's/^/    /'
    fi
    if grep -qE "^\[ERROR\].*\[vk" "$WINDOWED_LOG"; then
        fail "validation errors on the windowed path"
        grep -E "^\[ERROR\].*\[vk" "$WINDOWED_LOG" | head -5 | sed 's/^/    /'
    else
        ok "windowed path ran validation-clean"
    fi
else
    echo "    no display available - skipping the windowed present check"
fi

# ---------------------------------------------------------------- gameplay self test
step "scripted gameplay self test"
SELFTEST_LOG="$BUILD_DIR/verify_selftest.log"
ROOM2_AUDIO_NULL_DEVICE=1 "$BUILD_DIR/room2" --headless --selftest 9 \
    --screenshot "$SHOTS/verify_selftest.png" --width "$W" --height "$H" \
    --texture-size "$TEX" --shadow-resolution "$SHADOW" --probe-size "$PROBE" \
    --probe-samples 16 >"$SELFTEST_LOG" 2>&1
sed -n '/SELFTEST/,$p' "$SELFTEST_LOG" | sed 's/^/    /'
if grep -q "glass broken      : YES" "$SELFTEST_LOG" && \
   grep -q "reloads           : 1" "$SELFTEST_LOG"; then
    ok "pistol drawn, glass shattered and reloaded"
else
    fail "self test did not complete the expected sequence"
fi
if grep -qE "^\[ERROR\].*\[vk" "$SELFTEST_LOG"; then
    fail "validation errors during the self test"
else
    ok "self test ran validation-clean"
fi

# ---------------------------------------------------------------- summary
step "summary"
if [ "$FAILURES" -eq 0 ]; then
    printf '\033[32mALL CHECKS PASSED\033[0m\n'
    exit 0
fi
printf '\033[31m%d CHECK(S) FAILED\033[0m\n' "$FAILURES"
exit 1
