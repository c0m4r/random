// room2 - procedural audio engine.
// All sounds are synthesised at startup from DSP primitives; the project ships no
// audio assets. Also provides a small room-reverb bus and 3D positional playback.
#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include "../core/math.hpp"

namespace room2::audio {

// ---------------------------------------------------------------- sound bank
enum class SoundId : uint32_t {
    Gunshot = 0,        // full HK USP discharge: crack + body + room tail
    GunshotTail,        // late reverb tail only (for layering / distance)
    DryFire,            // hammer/striker fall on empty chamber
    SafetyClick,
    MagRelease,         // magazine catch pressed
    MagOut,             // magazine sliding out of the magwell
    MagIn,              // magazine inserted and seated
    SlideBack,          // slide racked rearwards
    SlideForward,       // slide slamming into battery
    ShellDrop,          // spent casing hitting the floor
    ShellBounce,        // small secondary casing bounce
    GlassShatter,       // full breaking-glass event
    GlassTinkle,        // a single small shard landing
    GlassStress,        // short high-frequency crackle as the glass cracks
    ImpactConcrete,     // bullet hitting plaster/concrete
    ImpactWood,
    ImpactMetal,
    Footstep,           // generic footstep (variations selected at build time)
    FootstepAlt,
    WeaponDraw,         // holster -> hands
    WeaponHolster,
    Cloth,
    RoomTone,           // looping HVAC / electrical hum bed
    EarRing,            // post-shot tinnitus
    UiBeep,
    Count
};

inline constexpr uint32_t kSoundCount = static_cast<uint32_t>(SoundId::Count);

// ---------------------------------------------------------------- voice
using VoiceId = int32_t;
inline constexpr VoiceId INVALID_VOICE = -1;

struct PlayParams {
    float volume = 1.0f;    // linear gain, 1.0 == nominal
    float pitch = 1.0f;     // playback rate multiplier (also shifts formants)
    float reverbSend = -1.0f;  // <0 -> use the per-sound default
    float occlusion = 0.0f;    // 0 = clear line of sight, 1 = fully blocked
    float delaySeconds = 0.0f; // schedule the voice slightly in the future
};

// ---------------------------------------------------------------- engine
class Engine {
public:
    // Opaque implementation state.  Declared here (rather than in the private
    // section) only so the render plumbing below can name it; the definition
    // lives entirely in audio.cpp and is not part of the public API.
    struct Impl;

    Engine();
    ~Engine();

    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    // Initialises SDL audio and synthesises the whole sound bank.
    // Returns false if no audio device could be opened (the game keeps running
    // silently in that case).
    bool init(int sampleRate = 48000, int channels = 2, int maxVoices = 64,
              float masterVolume = 0.8f);
    void shutdown();
    bool isInitialised() const { return initialised_; }
    const std::string& lastError() const { return lastError_; }

    // --- listener / mixing ------------------------------------------------
    void setListener(Vec3 position, Vec3 forward, Vec3 up);
    void setMasterVolume(float v);
    float masterVolume() const { return masterVolume_; }
    // Applies the room reverb model (call once per frame; safe to call when
    // audio is unavailable).
    void update(float dt);
    // Number of voices currently producing sound.
    int activeVoiceCount() const;

    // --- playback ---------------------------------------------------------
    VoiceId play(SoundId id, Vec3 worldPosition, const PlayParams& params = {});
    VoiceId play2D(SoundId id, const PlayParams& params = {});
    void stop(VoiceId voice, float fadeSeconds = 0.02f);
    void stopAll();
    bool isPlaying(VoiceId voice) const;

    // --- synthesis access (offline tests / custom sounds) -----------------
    // Raw mono sample data for a sound, at the engine sample rate.
    const std::vector<float>& soundSamples(SoundId id) const;
    int sampleRate() const { return sampleRate_; }
    // Renders the whole bank to 16-bit stereo WAV files (used by the --dump-audio
    // verification mode). Returns the number of files written.
    int dumpWavFiles(const std::string& directory) const;

    // Offline render of a mix (no device required) - renders `frames` stereo
    // frames into `out` using exactly the same mixing path as the live device.
    // `out` must have room for frames*2 floats.
    void renderOffline(float* out, int frames);

    // --- diagnostics ------------------------------------------------------
    // Peak absolute sample observed since the last call (per channel).
    Vec2 takePeakLevel();

private:
    // Render plumbing (additive; implementation detail of audio.cpp).
    static void audioCallbackThunk(void* userdata, uint8_t* stream, int len);
    // `what`: 0 = play, 1 = stop one voice, 2 = stop every voice.
    bool pushCommand(int what, SoundId sound, const PlayParams& params, const Vec3& position,
                     float gain, float pan, float itd, float airCutoff, float reverbSend,
                     bool useEar, uint32_t voiceHandle);
    void renderBlock(float* out, int frames);
    bool initialised_ = false;
    int sampleRate_ = 48000;
    int channels_ = 2;
    float masterVolume_ = 0.8f;
    std::string lastError_;
    Impl* impl_ = nullptr;
};

// ---------------------------------------------------------------- helpers
// Fills `out` with `seconds` of the given sound, mixed to mono at `sampleRate`.
// Requires Engine::init() to have run (so the bank exists).
std::vector<float> renderSoundToBuffer(SoundId id, float seconds, int sampleRate);

// Number of SDL audio callbacks entered since start-up.  Stays 0 when the engine
// is in null-device mode, which makes "no audio callback runs" directly testable.
uint64_t audioCallbackCount();

}  // namespace room2::audio
