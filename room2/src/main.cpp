// room2 - application entry point.
#include <SDL2/SDL.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <string>
#include <vector>

#include "audio/audio.hpp"
#include "core/log.hpp"
#include "core/png.hpp"
#include "core/rng.hpp"
#include "gfx/device.hpp"
#include "gfx/swapchain.hpp"
#include "game/game.hpp"
#include "physics/physics.hpp"
#include "render/ui.hpp"
#include "procgen/mesh.hpp"
#include "procgen/texture.hpp"
#include "render/renderer.hpp"
#include "scene/builders.hpp"
#include "scene/scene.hpp"

namespace {

struct Options {
    bool headless = false;
    bool windowed = true;
    bool validation = true;
    bool vsync = true;
    uint32_t width = 1280;
    uint32_t height = 720;
    float renderScale = 1.0f;
    std::string screenshotPath;
    std::string dumpTexturesDir;
    std::string dumpAudioDir;
    int frames = 0;
    int selftestSeconds = 0;
    uint32_t seed = 20240926u;
    int textureSize = 1024;
    int shadowResolution = 1024;
    int probeSize = 64;
    int probeSamples = 64;
    bool noBloom = false;
    bool noSsao = false;
    bool noTaa = false;
    bool noShadows = false;
    bool listDevices = false;
    float exposure = 1.6f;
    std::string dumpTargetsDir;
    std::string windowedShotPath;
    float masterVolume = 0.8f;
};

void printUsage() {
    std::printf(
        "room2 - photorealistic Vulkan first person demo\n"
        "usage: room2 [options]\n"
        "  --width N --height N       window / offscreen resolution (default 1280x720)\n"
        "  --render-scale F           internal resolution multiplier (default 1.0)\n"
        "  --headless                 no window; render offscreen\n"
        "  --screenshot PATH          render and write a PNG, then exit\n"
        "  --frames N                 number of frames to render for --screenshot (default 24)\n"
        "  --selftest SECONDS         run a scripted gameplay session and report results\n"
        "  --dump-textures DIR        write every generated texture as a PNG and exit\n"
        "  --dump-audio DIR           write the synthesised sound bank as WAVs and exit\n"
        "  --dump-targets DIR         write every render target as a PNG for debugging\n"
        "  --windowed-shot FILE       open a window, render --frames frames, capture and exit\n"
        "  --texture-size N           procedural texture resolution (default 1024)\n"
        "  --shadow-resolution N      cube shadow map face size (default 1024)\n"
        "  --probe-size N             environment probe face size (default 64)\n"
        "  --probe-samples N          GGX samples per probe texel (default 64)\n"
        "  --exposure F               tonemap exposure (default 1.0)\n"
        "  --no-vsync --no-validation --no-bloom --no-ssao --no-taa --no-shadows\n"
        "  --seed N                   deterministic content seed\n"
        "  --list-devices             enumerate Vulkan devices and exit\n"
        "  --help\n");
}

bool parseArgs(int argc, char** argv, Options& out) {
    auto need = [&](int& i) -> const char* {
        if (i + 1 >= argc) {
            R2_ERROR("option ", argv[i], " requires a value");
            return nullptr;
        }
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--help" || a == "-h") {
            printUsage();
            return false;
        } else if (a == "--headless") {
            out.headless = true;
            out.windowed = false;
        } else if (a == "--width") {
            const char* v = need(i);
            if (!v) return false;
            out.width = static_cast<uint32_t>(std::atoi(v));
        } else if (a == "--height") {
            const char* v = need(i);
            if (!v) return false;
            out.height = static_cast<uint32_t>(std::atoi(v));
        } else if (a == "--render-scale") {
            const char* v = need(i);
            if (!v) return false;
            out.renderScale = static_cast<float>(std::atof(v));
        } else if (a == "--screenshot") {
            const char* v = need(i);
            if (!v) return false;
            out.screenshotPath = v;
            out.headless = true;
            out.windowed = false;
        } else if (a == "--frames") {
            const char* v = need(i);
            if (!v) return false;
            out.frames = std::atoi(v);
        } else if (a == "--selftest") {
            const char* v = need(i);
            if (!v) return false;
            out.selftestSeconds = std::atoi(v);
            out.headless = true;
            out.windowed = false;
        } else if (a == "--dump-textures") {
            const char* v = need(i);
            if (!v) return false;
            out.dumpTexturesDir = v;
        } else if (a == "--dump-audio") {
            const char* v = need(i);
            if (!v) return false;
            out.dumpAudioDir = v;
        } else if (a == "--texture-size") {
            const char* v = need(i);
            if (!v) return false;
            out.textureSize = std::atoi(v);
        } else if (a == "--shadow-resolution") {
            const char* v = need(i);
            if (!v) return false;
            out.shadowResolution = std::atoi(v);
        } else if (a == "--probe-size") {
            const char* v = need(i);
            if (!v) return false;
            out.probeSize = std::atoi(v);
        } else if (a == "--probe-samples") {
            const char* v = need(i);
            if (!v) return false;
            out.probeSamples = std::atoi(v);
        } else if (a == "--exposure") {
            const char* v = need(i);
            if (!v) return false;
            out.exposure = static_cast<float>(std::atof(v));
        } else if (a == "--seed") {
            const char* v = need(i);
            if (!v) return false;
            out.seed = static_cast<uint32_t>(std::strtoul(v, nullptr, 10));
        } else if (a == "--no-vsync") {
            out.vsync = false;
        } else if (a == "--no-validation") {
            out.validation = false;
        } else if (a == "--no-bloom") {
            out.noBloom = true;
        } else if (a == "--no-ssao") {
            out.noSsao = true;
        } else if (a == "--no-taa") {
            out.noTaa = true;
        } else if (a == "--no-shadows") {
            out.noShadows = true;
        } else if (a == "--windowed-shot") {
            const char* v = need(i);
            if (!v) return false;
            out.windowedShotPath = v;
            out.headless = false;
            out.windowed = true;
        } else if (a == "--dump-targets") {
            const char* v = need(i);
            if (!v) return false;
            out.dumpTargetsDir = v;
            out.headless = true;
            out.windowed = false;
        } else if (a == "--list-devices") {
            out.listDevices = true;
        } else if (a == "--verbose") {
            room2::logSetLevel(room2::LogLevel::Debug);
        } else {
            R2_ERROR("unknown option '", a, "' (try --help)");
            return false;
        }
    }
    return true;
}

using namespace room2;

bool mReloadHeld = false;
bool mDrawHeld = false;
bool mFireHeld = false;

// Builds a scene material from a procedural texture set.
uint32_t addMaterialFromSet(scene::Scene& scene, render::Renderer& renderer, const char* name,
                            procgen::TextureSet set, Vec3 baseColorTint, float metallic,
                            float roughness, float normalScale = 1.0f) {
    scene::Material mat;
    mat.name = name;
    mat.baseColorTex = renderer.addTexture(set.baseColor, true, true, std::string(name) + "_bc");
    mat.normalTex = set.hasNormal
                        ? renderer.addTexture(set.normal, false, true, std::string(name) + "_n")
                        : scene::kTextureFlatNormal;
    mat.ormTex = set.hasOrm
                     ? renderer.addTexture(set.orm, false, true, std::string(name) + "_orm")
                     : scene::kTextureWhite;
    mat.baseColorFactor = Vec4(baseColorTint, 1.0f);
    mat.metallic = metallic;
    mat.roughness = roughness;
    mat.normalScale = normalScale;
    return scene.addMaterial(mat);
}

int dumpTextures(const Options& opt) {
    const uint32_t size = static_cast<uint32_t>(opt.textureSize);
    procgen::SurfaceParams params;
    params.seed = opt.seed;
    params.texelsPerMetre = static_cast<float>(size) / 2.0f;

    struct Entry {
        const char* name;
        procgen::TextureSet (*fn)(uint32_t, const procgen::SurfaceParams&);
    };
    const Entry entries[] = {
        {"plaster_wall", procgen::plasterWall},
        {"painted_ceiling", procgen::paintedCeiling},
        {"concrete_floor", procgen::concreteFloor},
        {"wood_planks", procgen::woodPlanks},
        {"wood_table_top", procgen::woodTableTop},
        {"varnished_wood", procgen::varnishedWood},
        {"blued_steel", procgen::bluedSteel},
        {"black_polymer", procgen::blackPolymer},
        {"grip_panel", procgen::gripPanel},
        {"brass", procgen::brass},
        {"glass_surface", procgen::glassSurface},
        {"painted_metal_white", procgen::paintedMetalWhite},
        {"lamp_emissive", procgen::lampEmissive},
        {"fabric", procgen::fabric},
        {"cardboard", procgen::cardboard},
    };
    int written = 0;
    for (const auto& e : entries) {
        procgen::TextureSet set = e.fn(size, params);
        const std::string base = std::string(opt.dumpTexturesDir) + "/" + e.name;
        if (procgen::writePng(base + "_basecolor.png", set.baseColor, true)) ++written;
        if (set.hasNormal && procgen::writePng(base + "_normal.png", set.normal, false))
            ++written;
        if (set.hasOrm && procgen::writePng(base + "_orm.png", set.orm, false)) ++written;
        R2_INFO("texture ", e.name, ": ", set.baseColor.width, "x", set.baseColor.height);
    }
    std::printf("WROTE %d texture PNGs to %s\n", written, opt.dumpTexturesDir.c_str());
    return written > 0 ? 0 : 1;
}

int dumpAudio(const Options& opt) {
    audio::Engine engine;
    if (!engine.init()) {
        R2_ERROR("audio engine failed to initialise: ", engine.lastError());
        return 1;
    }
    const int count = engine.dumpWavFiles(opt.dumpAudioDir);
    std::printf("WROTE %d WAV files to %s\n", count, opt.dumpAudioDir.c_str());
    engine.shutdown();
    return count > 0 ? 0 : 1;
}

int runApp(const Options& opt) {
    if (!opt.dumpTexturesDir.empty()) return dumpTextures(opt);
    if (!opt.dumpAudioDir.empty()) return dumpAudio(opt);

    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS | SDL_INIT_TIMER | SDL_INIT_AUDIO) != 0) {
        R2_ERROR("SDL_Init failed: ", SDL_GetError());
        return 1;
    }

    SDL_Window* window = nullptr;
    if (!opt.headless) {
        SDL_WindowFlags flags = static_cast<SDL_WindowFlags>(
            SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI);
        window = SDL_CreateWindow("room2", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                  static_cast<int>(opt.width), static_cast<int>(opt.height), flags);
        if (!window) {
            R2_ERROR("SDL_CreateWindow failed: ", SDL_GetError());
            SDL_Quit();
            return 1;
        }
    }

    gfx::Device device;
    gfx::DeviceConfig deviceConfig;
    deviceConfig.enableValidation = opt.validation;
    deviceConfig.headless = opt.headless;
    deviceConfig.appName = "room2";
    if (!device.init(window, deviceConfig)) {
        R2_ERROR("failed to initialise the Vulkan device");
        if (window) SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    gfx::Swapchain swapchain;
    if (window) {
        if (!swapchain.init(device, opt.width, opt.height, opt.vsync)) {
            R2_ERROR("failed to create the swapchain");
            device.shutdown();
            SDL_DestroyWindow(window);
            SDL_Quit();
            return 1;
        }
    }

    render::RendererConfig renderConfig;
    renderConfig.width = opt.width;
    renderConfig.height = opt.height;
    renderConfig.renderScale = opt.renderScale;
    renderConfig.headless = opt.headless;
    renderConfig.shadowResolution = opt.shadowResolution;
    renderConfig.envProbeSize = opt.probeSize;
    renderConfig.envProbeSamples = opt.probeSamples;
    renderConfig.enableBloom = !opt.noBloom;
    renderConfig.enableSSAO = !opt.noSsao;
    renderConfig.enableTAA = !opt.noTaa;
    renderConfig.enableShadows = !opt.noShadows;
    renderConfig.enableValidation = opt.validation;
    renderConfig.verboseStats = room2::logGetLevel() <= room2::LogLevel::Debug;

    render::Renderer renderer;
    if (!renderer.init(device, window ? &swapchain : nullptr, renderConfig)) {
        R2_ERROR("failed to initialise the renderer");
        device.shutdown();
        if (window) SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    render::UiContext ui;
    if (!ui.init(device)) {
        R2_ERROR("failed to create the UI font atlas");
        renderer.shutdown();
        device.shutdown();
        if (window) SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }
    renderer.setUiAtlas(ui.atlas());

    scene::Scene demoScene;
    game::Game theGame;
    game::GameConfig gameConfig;
    gameConfig.seed = opt.seed;
    gameConfig.masterVolume = opt.masterVolume;
    gameConfig.selfTest = !opt.selftestSeconds;
    gameConfig.textureSize = static_cast<uint32_t>(opt.textureSize);
    if (!theGame.init(demoScene, renderer, gameConfig)) {
        R2_ERROR("failed to build the game scene");
        renderer.shutdown();
        device.shutdown();
        if (window) SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }
    renderer.uploadScene(demoScene);

    const scene::Scene::Stats sceneStats = demoScene.stats();
    R2_INFO("scene: ", sceneStats.triangles, " triangles, ", sceneStats.vertices,
            " vertices, ", sceneStats.drawCalls, " draw calls, ", sceneStats.materials,
            " materials");

    int exitCode = 0;
    std::vector<uint8_t> pixels;
    uint32_t outW = 0, outH = 0;

    if (!opt.screenshotPath.empty() || opt.selftestSeconds > 0) {
        // ---- scripted run: no window, deterministic input ----
        const int fps = 60;
        const float dt = 1.0f / fps;
        int frames = opt.frames > 0 ? opt.frames : 90;
        if (opt.selftestSeconds > 0) frames = opt.selftestSeconds * fps;

        game::InputState in;
        bool firedOnce = false;
        int reloadAt = static_cast<int>(frames * 0.72f);
        int shatterFrame = -1;
        std::vector<uint8_t> last;

        const auto start = std::chrono::high_resolution_clock::now();
        for (int f = 0; f < frames; ++f) {
            in = game::InputState{};
            // Script: draw, aim, fire at the glass, then reload.
            if (f == 2) in.drawPressed = true;
            // The draw animation takes ~0.62 s (37 frames), so fire once it is complete.
            if (f == 55) {
                in.firePressed = true;
                in.fireHeld = true;
            }
            if (f == reloadAt) in.reloadPressed = true;

            theGame.update(in, dt, ui, renderer.renderWidth(), renderer.renderHeight());
            // New geometry (glass shards) is appended at runtime and must reach the GPU.
            if (theGame.consumeGeometryDirty()) renderer.uploadScene(demoScene);
            const render::UiDrawList& list = ui.list();

            render::Camera cam = theGame.camera();
            const Vec2 jitter = render::haltonJitter(static_cast<uint32_t>(f),
                                                     renderer.renderWidth(),
                                                     renderer.renderHeight());
            cam.setJitter(renderer.configFlag("taa") ? jitter : Vec2(0, 0));
            cam.setViewportSize(renderer.renderWidth(), renderer.renderHeight());

            if (!renderer.beginFrame()) continue;
            renderer.render(demoScene, cam, dt, &list, opt.exposure);
            renderer.endFrame();
            if (theGame.stats().glassBroken && shatterFrame < 0) shatterFrame = f;
            (void)firedOnce;
        }
        device.waitIdle();
        renderer.capture(pixels, outW, outH);
        renderer.renderOffscreen(demoScene, theGame.camera(), dt, &ui.list(), opt.exposure, 1,
                                 last, outW, outH);
        const auto endTime = std::chrono::high_resolution_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(endTime - start).count();
        pixels = last;

        const game::GameStats& gs = theGame.stats();
        if (!opt.screenshotPath.empty()) {
            int nonBlack = 0;
            double sum = 0.0;
            for (size_t i = 0; i + 3 < pixels.size(); i += 4) {
                const int luma = pixels[i] + pixels[i + 1] + pixels[i + 2];
                if (luma > 12) ++nonBlack;
                sum += luma / 3.0;
            }
            if (!png::writeRgba(opt.screenshotPath, outW, outH, pixels.data())) {
                R2_ERROR("failed to write ", opt.screenshotPath);
                exitCode = 1;
            } else {
                std::printf(
                    "SCREENSHOT %s  %ux%u\n"
                    "  frames=%d total=%.1fms avg=%.2fms/frame\n"
                    "  mean luma=%.2f  non-black pixels=%.1f%%\n",
                    opt.screenshotPath.c_str(), outW, outH, frames, ms, ms / frames,
                    sum / static_cast<double>(pixels.size() / 4),
                    100.0 * nonBlack / static_cast<double>(pixels.size() / 4));
            }
        }
        if (opt.selftestSeconds > 0) {
            {
                const scene::Scene::Stats st2 = demoScene.stats();
                R2_INFO("post-run scene: ", st2.instances, " instances, ", st2.drawCalls,
                        " draw calls");
                for (uint32_t i = 0; i < 6 && i < demoScene.instanceCount(); ++i) {
                    const scene::Instance& inst = demoScene.instance(i);
                    R2_INFO("  instance ", i, " mesh=", inst.mesh, " visible=", inst.visible,
                            " bounds=(",
                            inst.worldBounds.valid() ? inst.worldBounds.mn.x : -999.0f, ",",
                            inst.worldBounds.valid() ? inst.worldBounds.mn.y : -999.0f, ",",
                            inst.worldBounds.valid() ? inst.worldBounds.mn.z : -999.0f, ")-(",
                            inst.worldBounds.valid() ? inst.worldBounds.mx.x : -999.0f, ",",
                            inst.worldBounds.valid() ? inst.worldBounds.mx.y : -999.0f, ",",
                            inst.worldBounds.valid() ? inst.worldBounds.mx.z : -999.0f, ")");
                }
            }
            std::printf(
                "SELFTEST (%d frames, %.2f s simulated)\n"
                "  weapon state      : %s\n"
                "  shots fired       : %d\n"
                "  magazine / reserve: %d / %d\n"
                "  reloads           : %d\n"
                "  impacts           : %d\n"
                "  glass broken      : %s at frame %d\n"
                "  shards spawned    : %d\n"
                "  shards awake/asleep: %d / %d\n"
                "  audio device      : %s\n"
                "  last event        : %s\n"
                "  shard spread (m)  : %.3f x %.3f x %.3f\n"
                "  shard distance    : mean %.3f m, max %.3f m, fastest %.2f m/s\n"
                "  final frame luma  : %.2f\n",
                frames, frames * dt, game::weaponStateName(theGame.weaponState()),
                gs.shotsFired, gs.roundsInMagazine, gs.roundsReserve, gs.reloads, gs.impacts,
                gs.glassBroken ? "YES" : "NO", shatterFrame, gs.shardCount, gs.bodiesAwake,
                gs.bodiesAsleep,
                theGame.audioEngine().isInitialised() ? "open" : "unavailable",
                theGame.lastEvent().c_str(),
                [&]() {
                    const Aabb b = theGame.shardBounds();
                    return b.valid() ? (b.mx.x - b.mn.x) : 0.0f;
                }(),
                [&]() {
                    const Aabb b = theGame.shardBounds();
                    return b.valid() ? (b.mx.y - b.mn.y) : 0.0f;
                }(),
                [&]() {
                    const Aabb b = theGame.shardBounds();
                    return b.valid() ? (b.mx.z - b.mn.z) : 0.0f;
                }(),
                [&]() { float m = 0, x = 0, s = 0; theGame.shardSpreadStats(m, x, s); return m; }(),
                [&]() { float m = 0, x = 0, s = 0; theGame.shardSpreadStats(m, x, s); return x; }(),
                [&]() { float m = 0, x = 0, s = 0; theGame.shardSpreadStats(m, x, s); return s; }(),
                [&]() {
                    double s2 = 0.0;
                    for (size_t i = 0; i + 3 < pixels.size(); i += 4)
                        s2 += (pixels[i] + pixels[i + 1] + pixels[i + 2]) / 3.0;
                    return pixels.empty() ? 0.0 : s2 / static_cast<double>(pixels.size() / 4);
                }());
            if (!gs.glassBroken) exitCode = 1;
            if (gs.shotsFired < 1) exitCode = 1;
        }
    } else {
        // ---- interactive loop ----
        bool running = true;
        double lastTime = static_cast<double>(SDL_GetPerformanceCounter()) /
                          static_cast<double>(SDL_GetPerformanceFrequency());
        // Mouse look is grabbed immediately: requiring a click to "capture" the mouse
        // made the first click ambiguous with the fire button, and made it feel as
        // though looking, moving and shooting could not be combined.
        bool mouseLook = true;
        int pendingMouseX = 0;
        int pendingMouseY = 0;
        bool leftButtonDown = false;
        bool inputDebug = false;
        SDL_RaiseWindow(window);
        SDL_SetRelativeMouseMode(SDL_TRUE);

        // Bounded run: --frames N makes the windowed loop exit after N presented frames,
        // which is how the swapchain/present path is verified automatically.
        const int windowedFrameLimit = opt.frames > 0 ? opt.frames : 0;
        int presentedFrames = 0;
        const bool captureWindow = !opt.windowedShotPath.empty();
        std::vector<uint8_t> windowShot;
        uint32_t windowShotW = 0, windowShotH = 0;

        while (running) {
            if (windowedFrameLimit > 0 && presentedFrames >= windowedFrameLimit) running = false;
            if (!running) break;

            // Movement, look and weapon input are all sampled every frame from
            // independent sources, so any combination can be active at once.
            SDL_Event event;
            while (SDL_PollEvent(&event)) {
                switch (event.type) {
                    case SDL_QUIT:
                        running = false;
                        break;
                    case SDL_KEYDOWN:
                        if (event.key.repeat) break;
                        switch (event.key.keysym.sym) {
                            case SDLK_ESCAPE: running = false; break;
                            case SDLK_F1:
                                renderer.setConfigFlag("ssao", !renderer.configFlag("ssao"));
                                break;
                            case SDLK_F2:
                                renderer.setConfigFlag("bloom", !renderer.configFlag("bloom"));
                                break;
                            case SDLK_F3:
                                renderer.setConfigFlag("taa", !renderer.configFlag("taa"));
                                break;
                            case SDLK_F4:
                                renderer.setConfigFlag("shadows", !renderer.configFlag("shadows"));
                                break;
                            case SDLK_F5:
                                inputDebug = !inputDebug;
                                theGame.setInputDebug(inputDebug);
                                break;
                            case SDLK_TAB:
                                mouseLook = !mouseLook;
                                SDL_SetRelativeMouseMode(mouseLook ? SDL_TRUE : SDL_FALSE);
                                break;
                            default: break;
                        }
                        break;
                    case SDL_MOUSEBUTTONDOWN:
                        if (event.button.button == SDL_BUTTON_LEFT) leftButtonDown = true;
                        // Clicking the window re-grabs the mouse if it was released.
                        if (!mouseLook) {
                            mouseLook = true;
                            SDL_SetRelativeMouseMode(SDL_TRUE);
                        }
                        break;
                    case SDL_MOUSEBUTTONUP:
                        if (event.button.button == SDL_BUTTON_LEFT) leftButtonDown = false;
                        break;
                    case SDL_MOUSEMOTION:
                        if (mouseLook) {
                            pendingMouseX += event.motion.xrel;
                            pendingMouseY += event.motion.yrel;
                        }
                        break;
                    case SDL_WINDOWEVENT:
                        if (event.window.event == SDL_WINDOWEVENT_SIZE_CHANGED) {
                            swapchain.setNeedsRecreate();
                        } else if (event.window.event == SDL_WINDOWEVENT_FOCUS_GAINED) {
                            // Losing focus drops the grab; restore it on return so the
                            // game is immediately playable again.
                            SDL_SetRelativeMouseMode(mouseLook ? SDL_TRUE : SDL_FALSE);
                        } else if (event.window.event == SDL_WINDOWEVENT_FOCUS_LOST) {
                            leftButtonDown = false;
                        }
                        break;
                    default:
                        break;
                }
            }

            // Injected keyboard/mouse state (the self test drives the same fields).
            const Uint8* keys = SDL_GetKeyboardState(nullptr);
            game::InputState in;
            in.forward = keys[SDL_SCANCODE_W] != 0;
            in.back = keys[SDL_SCANCODE_S] != 0;
            in.left = keys[SDL_SCANCODE_A] != 0;
            in.right = keys[SDL_SCANCODE_D] != 0;
            in.run = keys[SDL_SCANCODE_LSHIFT] != 0;
            in.crouch = keys[SDL_SCANCODE_LCTRL] != 0;
            in.jump = keys[SDL_SCANCODE_SPACE] != 0;
            in.reloadPressed = keys[SDL_SCANCODE_R] != 0 && !mReloadHeld;
            in.drawPressed = keys[SDL_SCANCODE_Q] != 0 && !mDrawHeld;
            mReloadHeld = keys[SDL_SCANCODE_R] != 0;
            mDrawHeld = keys[SDL_SCANCODE_Q] != 0;
            in.fireHeld = leftButtonDown;
            in.firePressed = leftButtonDown && !mFireHeld;
            mFireHeld = leftButtonDown;
            in.mouseDeltaX = static_cast<float>(pendingMouseX);
            in.mouseDeltaY = static_cast<float>(pendingMouseY);
            pendingMouseX = 0;
            pendingMouseY = 0;

            const double now = static_cast<double>(SDL_GetPerformanceCounter()) /
                               static_cast<double>(SDL_GetPerformanceFrequency());
            const float dt = static_cast<float>(std::min(now - lastTime, 0.05));
            lastTime = now;

            theGame.update(in, dt, ui, renderer.renderWidth(), renderer.renderHeight());
            if (theGame.consumeGeometryDirty()) renderer.uploadScene(demoScene);
            render::Camera cam = theGame.camera();
            const Vec2 jitter = render::haltonJitter(renderer.stats().frameIndex,
                                                     renderer.renderWidth(),
                                                     renderer.renderHeight());
            cam.setJitter(renderer.configFlag("taa") ? jitter : Vec2(0, 0));

            if (captureWindow && presentedFrames == windowedFrameLimit - 1) {
                renderer.requestCapture();
            }
            if (!renderer.beginFrame()) {
                if (swapchain.needsRecreate()) swapchain.recreate(0, 0, opt.vsync);
                // Never spin: a failed acquire should not burn the CPU or freeze the UI.
                SDL_Delay(2);
                continue;
            }
            renderer.render(demoScene, cam, dt, &ui.list(), opt.exposure);
            renderer.endFrame();
            ++presentedFrames;
            if (captureWindow && presentedFrames == windowedFrameLimit) {
                renderer.capture(windowShot, windowShotW, windowShotH);
            }
        }
        if (windowedFrameLimit > 0) {
            R2_INFO("windowed run complete: ", presentedFrames, " frames presented");
        }
        if (captureWindow && !windowShot.empty()) {
            if (png::writeRgba(opt.windowedShotPath, windowShotW, windowShotH, windowShot.data())) {
                std::printf("WINDOWED SHOT %s  %ux%u  (%d frames presented)\n",
                            opt.windowedShotPath.c_str(), windowShotW, windowShotH,
                            presentedFrames);
            } else {
                R2_ERROR("failed to write ", opt.windowedShotPath);
                exitCode = 1;
            }
        }
    }

    device.waitIdle();
    theGame.shutdown();
    ui.shutdown();
    renderer.shutdown();
    swapchain.shutdown();
    device.shutdown();
    if (window) SDL_DestroyWindow(window);
    SDL_Quit();
    return exitCode;
}

}  // namespace

int main(int argc, char** argv) {
    room2::logSetLevel(room2::LogLevel::Info);
    Options opt;
    if (!parseArgs(argc, argv, opt)) return 1;
    return runApp(opt);
}
