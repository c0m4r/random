// room2 - audio engine verification.
//
// Build & run in ONE bash call (/tmp does not persist on this machine):
//   g++ -std=c++20 -O2 -Wall -Wextra $(pkg-config --cflags sdl2) -I src
//       tests/audio_test.cpp src/audio/audio.cpp -o build/audio_test -lSDL2 && build/audio_test
//
// Everything asserted here is required by the audio module contract:
//   1. init() succeeds with no audio device available (null-device mode)
//   2. every SoundId synthesises a finite, non-empty, sensibly long buffer
//      with a usable peak and RMS
//   3. an offline stereo mix of several positioned sounds stays finite,
//      non-silent and inside [-1,1]
//   4. constant-power panning actually puts energy in the correct channel
//   5. the distance model makes a near source louder than a far one
//   6. dumpWavFiles() writes every bank sound as a 16-bit stereo WAV
// plus the supporting behaviours the game relies on (delay scheduling,
// occlusion, deterministic seeded synthesis, seamless RoomTone loop,
// voice stealing, timing independence of the mixing path).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "audio/audio.hpp"
#include "core/math.hpp"

using namespace room2;
using namespace room2::audio;

// ------------------------------------------------------------------- harness
namespace {

int g_checks = 0;
int g_failures = 0;

void check(bool cond, const std::string& what) {
    ++g_checks;
    if (cond) {
        std::printf("  [ ok ] %s\n", what.c_str());
    } else {
        ++g_failures;
        std::printf("  [FAIL] %s\n", what.c_str());
    }
}

struct Stats {
    int frames = 0;
    float seconds = 0.0f;
    float peak = 0.0f;
    float rms = 0.0f;
    bool finite = true;
};

Stats analyse(const std::vector<float>& b, int sampleRate) {
    Stats s;
    s.frames = int(b.size());
    s.seconds = sampleRate > 0 ? float(b.size()) / float(sampleRate) : 0.0f;
    if (b.empty()) {
        s.finite = true;
        return s;
    }
    double acc = 0.0;
    for (float v : b) {
        if (!std::isfinite(v)) s.finite = false;
        float a = std::fabs(v);
        s.peak = std::max(s.peak, a);
        acc += double(v) * double(v);
    }
    s.rms = float(std::sqrt(acc / double(b.size())));
    return s;
}

const char* soundName(SoundId id) {
    static const char* names[kSoundCount] = {
        "Gunshot",     "GunshotTail", "DryFire",      "SafetyClick",  "MagRelease",
        "MagOut",      "MagIn",       "SlideBack",    "SlideForward", "ShellDrop",
        "ShellBounce", "GlassShatter", "GlassTinkle", "GlassStress",  "ImpactConcrete",
        "ImpactWood",  "ImpactMetal", "Footstep",     "FootstepAlt",  "WeaponDraw",
        "WeaponHolster", "Cloth",     "RoomTone",     "EarRing",      "UiBeep"};
    return names[uint32_t(id) < kSoundCount ? uint32_t(id) : 0];
}

// Per-sound minimum duration (seconds) and the reason for that bound.  These are
// deliberately conservative lower bounds that still catch a sound collapsing to
// a click (or being truncated before its designed tail):
//   Gunshot        1.00 - the room tail alone must last 0.8-1.2 s
//   GunshotTail    0.90 - the same tail without the direct sound
//   DryFire        0.05 - sharp click plus a short metallic ring
//   SafetyClick    0.03 - the tightest, shortest click in the bank
//   MagRelease     0.05 - click plus a small spring snick
//   MagOut         0.30 - a slide-out scrape (~0.24 s) plus the clack
//   MagIn          0.25 - hollow thunk plus the seat click and frame ring
//   SlideBack      0.28 - racking travel (~0.25 s) plus the rear hard stop
//   SlideForward   0.20 - scrape, slam and the recoil-spring ring
//   ShellDrop      0.80 - primary hit plus three decaying bounces
//   ShellBounce    0.20 - one bounce plus its spin-down
//   GlassShatter   1.10 - the shower and skittering span 0.6-1.2 s
//   GlassTinkle    0.15 - a small shard's fast inharmonic ping
//   GlassStress    0.04 - a very short crackle, not a discrete event
//   ImpactConcrete 0.35 - grit plus a dust tail
//   ImpactWood     0.25 - dry debris tail (shorter than concrete)
//   ImpactMetal    0.55 - the long inharmonic metal ring
//   Footstep       0.15 - scuff, thud and settling grains
//   FootstepAlt    0.18 - the variant is longer and later
//   WeaponDraw     0.45 - leather creak through to the grip slap
//   WeaponHolster  0.38 - the holstering gesture
//   Cloth          0.20 - a short rustle gesture
//   RoomTone       3.50 - the loop bed is authored at 4 s
//   EarRing        2.00 - a slow tinnitus decay
//   UiBeep         0.10 - a short clean blip
const float kMinDuration[kSoundCount] = {1.00f, 0.90f, 0.05f, 0.03f, 0.05f, 0.30f, 0.25f, 0.28f,
                                         0.20f, 0.80f, 0.20f, 1.10f, 0.15f, 0.04f, 0.35f, 0.25f,
                                         0.55f, 0.15f, 0.18f, 0.45f, 0.38f, 0.20f, 3.50f, 2.00f,
                                         0.10f};

float energy(const std::vector<float>& interleaved, int channel) {
    double acc = 0.0;
    for (size_t i = size_t(channel); i < interleaved.size(); i += 2) {
        acc += double(interleaved[i]) * double(interleaved[i]);
    }
    return float(acc);
}

// Mean absolute first difference, normalised: a cheap brightness proxy used to
// verify that occlusion really darkens a source.
float brightness(const std::vector<float>& interleaved) {
    double acc = 0.0;
    size_t n = interleaved.size() / 2;
    if (n < 2) return 0.0f;
    for (size_t i = 1; i < n; ++i) {
        acc += std::fabs(double(interleaved[2 * i]) - double(interleaved[2 * i - 2]));
    }
    return float(acc / double(n - 1));
}

struct WavInfo {
    bool ok = false;
    int channels = 0;
    int bits = 0;
    int sampleRate = 0;
    int frames = 0;
    float peak = 0.0f;
    float rms = 0.0f;
    uint64_t bytes = 0;
};

// Minimal RIFF reader used to prove dumpWavFiles() output really is a valid
// 16-bit stereo WAV at the engine rate (independent of the writer).
WavInfo readWav(const std::string& path) {
    WavInfo w;
    std::ifstream f(path, std::ios::binary);
    if (!f) return w;
    std::error_code ec;
    w.bytes = uint64_t(std::filesystem::file_size(path, ec));
    char riff[4], wave[4];
    uint32_t riffSize = 0;
    f.read(riff, 4);
    f.read(reinterpret_cast<char*>(&riffSize), 4);
    f.read(wave, 4);
    if (std::memcmp(riff, "RIFF", 4) != 0 || std::memcmp(wave, "WAVE", 4) != 0) return w;
    // Walk the chunk list rather than assuming a fixed 44-byte header: an
    // odd-sized data chunk forces a pad byte, and a writer may emit 'fact'.
    uint16_t audioFmt = 0, channels = 0, bits = 0, blockAlign = 0;
    uint32_t rate = 0, byteRate = 0, dataSize = 0;
    std::streampos dataPos = 0;
    bool sawFmt = false;
    while (f && f.peek() != EOF) {
        char id[4];
        uint32_t size = 0;
        f.read(id, 4);
        f.read(reinterpret_cast<char*>(&size), 4);
        if (!f) break;
        if (std::memcmp(id, "fmt ", 4) == 0) {
            f.read(reinterpret_cast<char*>(&audioFmt), 2);
            f.read(reinterpret_cast<char*>(&channels), 2);
            f.read(reinterpret_cast<char*>(&rate), 4);
            f.read(reinterpret_cast<char*>(&byteRate), 4);
            f.read(reinterpret_cast<char*>(&blockAlign), 2);
            f.read(reinterpret_cast<char*>(&bits), 2);
            sawFmt = true;
            if (size > 16) f.seekg(std::streamoff(size) - 16, std::ios::cur);
        } else if (std::memcmp(id, "data", 4) == 0) {
            dataSize = size;
            dataPos = f.tellg();
            break;
        } else {
            f.seekg(std::streamoff(size + (size & 1u)), std::ios::cur);
        }
    }
    const bool headerOk = sawFmt && dataSize > 0 && audioFmt == 1 && bits == 16 &&
                          channels == 2 && blockAlign == 4 && byteRate == rate * 4;
    if (!headerOk) return w;
    f.clear();
    f.seekg(dataPos);
    w.channels = channels;
    w.bits = bits;
    w.sampleRate = int(rate);
    w.frames = int(dataSize) / int(blockAlign);
    std::vector<int16_t> pcm(size_t(w.frames) * channels);
    f.read(reinterpret_cast<char*>(pcm.data()), std::streamsize(pcm.size() * sizeof(int16_t)));
    if (f.gcount() != std::streamsize(pcm.size() * sizeof(int16_t))) return w;
    double acc = 0.0;
    for (int16_t s : pcm) {
        float v = float(s) / 32767.0f;
        w.peak = std::max(w.peak, std::fabs(v));
        acc += double(v) * double(v);
    }
    w.rms = pcm.empty() ? 0.0f : float(std::sqrt(acc / double(pcm.size())));
    w.ok = true;
    return w;
}

}  // namespace

int main() {
    std::printf("=== room2 audio test ===\n");

    // -----------------------------------------------------------------------
    // 1. init with no audio device
    // -----------------------------------------------------------------------
    // The engine is asked for the null-device path explicitly.  This machine has
    // no /dev/snd but does run a PipeWire server, which SDL will happily connect
    // to; forcing the path makes the whole suite deterministic (nothing drains
    // the command ring behind our back) and exercises exactly the "no device"
    // behaviour the module contract requires.
    setenv("ROOM2_AUDIO_NULL_DEVICE", "1", 1);
    std::printf("\n[1] Engine::init() with no usable audio device\n");
    std::printf("  /dev/snd present: %s\n", std::filesystem::exists("/dev/snd") ? "yes" : "no");
    Engine engine;
    const bool ok = engine.init(48000, 2, 64, 0.8f);
    check(ok, "init() returned true (must succeed with no audio device)");
    check(engine.isInitialised(), "isInitialised() == true");
    check(engine.sampleRate() == 48000, "sampleRate() == 48000");
    {
        std::printf("  device mode : null device (no audio callback runs)\n");
        std::printf("  lastError() : \"%s\"\n", engine.lastError().c_str());
        check(!engine.lastError().empty(), "null-device mode recorded a reason in lastError()");
        check(audioCallbackCount() == 0, "no SDL audio callback has run");
        check(engine.activeVoiceCount() == 0, "activeVoiceCount() == 0 before any play()");
    }
    // A second init() must be idempotent.
    check(engine.init(48000, 2, 64, 0.8f), "second init() is a no-op and returns true");
    // Exercise the mixer, then re-check that still nothing ran on a device
    // thread: update()/renderOffline() must be the only render path.
    engine.play2D(SoundId::UiBeep);
    engine.update(0.05f);
    check(audioCallbackCount() == 0, "still no audio callback after update()/play()");
    engine.stopAll();
    engine.update(0.05f);

    // -----------------------------------------------------------------------
    // 2. every SoundId synthesises a usable buffer
    // -----------------------------------------------------------------------
    std::printf("\n[2] sound bank: %u sounds\n", kSoundCount);
    std::printf("  %-15s %8s %8s %8s %8s  %s\n", "sound", "frames", "sec", "peak", "rms", "result");
    struct Row {
        const char* name;
        float sec, peak, rms;
    };
    std::vector<Row> table;
    table.reserve(kSoundCount);
    const int sr = engine.sampleRate();
    for (uint32_t i = 0; i < kSoundCount; ++i) {
        const SoundId id = SoundId(i);
        const std::vector<float>& b = engine.soundSamples(id);
        const Stats s = analyse(b, sr);
        const bool nonEmpty = !b.empty();
        const bool finite = s.finite;
        const bool peakOk = s.peak > 0.02f && s.peak <= 1.05f;
        const bool rmsOk = s.rms > 0.0005f;
        const bool durOk = s.seconds >= kMinDuration[i];
        const bool passed = nonEmpty && finite && peakOk && rmsOk && durOk;
        std::printf("  %-15s %8d %8.3f %8.4f %8.5f  %s\n", soundName(id), s.frames, s.seconds, s.peak,
                    s.rms, passed ? "ok" : "FAIL");
        table.push_back({soundName(id), s.seconds, s.peak, s.rms});
        check(nonEmpty, std::string(soundName(id)) + ": buffer is non-empty");
        check(finite, std::string(soundName(id)) + ": all samples finite (no NaN/Inf)");
        check(peakOk, std::string(soundName(id)) + ": peak in (0.02, 1.05]");
        check(rmsOk, std::string(soundName(id)) + ": RMS > 0.0005");
        check(durOk, std::string(soundName(id)) + ": duration >= " +
                         std::to_string(kMinDuration[i]).substr(0, 4) + " s");
    }
    // The gunshot must peak close to full scale per the sound design.
    {
        const Stats g = analyse(engine.soundSamples(SoundId::Gunshot), sr);
        check(g.peak >= 0.90f && g.peak <= 1.0f, "Gunshot peaks near full scale (0.90..1.00)");
    }
    // Deterministic seeded synthesis: the same query twice is bit-identical.
    {
        const std::vector<float> a = renderSoundToBuffer(SoundId::GlassShatter, 1.0f, sr);
        const std::vector<float> b = renderSoundToBuffer(SoundId::GlassShatter, 1.0f, sr);
        check(a == b, "renderSoundToBuffer() is deterministic (seeded PRNG, no rand())");
    }
    // renderSoundToBuffer() honours the requested length and resamples.
    {
        const std::vector<float> half = renderSoundToBuffer(SoundId::Gunshot, 0.5f, 24000);
        const Stats s = analyse(half, 24000);
        check(s.frames == 12000, "renderSoundToBuffer(0.5 s @24k) produced 12000 frames");
        check(s.seconds > 0.49f && s.seconds < 0.51f, "resampled duration is 0.5 s");
        check(s.finite && s.peak > 0.02f && s.peak <= 1.05f, "resampled buffer stays valid");
        const std::vector<float> up = renderSoundToBuffer(SoundId::Gunshot, 0.25f, 96000);
        check(up.size() == 24000, "renderSoundToBuffer(0.25 s @96k) produced 24000 frames");
    }
    // RoomTone must loop without a click: the wrap-around slew is bounded.
    {
        const std::vector<float>& rt = engine.soundSamples(SoundId::RoomTone);
        float maxStep = 0.0f;
        for (size_t i = 1; i < rt.size(); ++i) maxStep = std::max(maxStep, std::fabs(rt[i] - rt[i - 1]));
        const float seam = std::fabs(rt.front() - rt.back());
        std::printf("  RoomTone loop seam discontinuity: %.6f (max in-loop step %.6f)\n", seam, maxStep);
        // A click at the loop point would show up as a wrap-around step much
        // larger than the signal's own sample-to-sample slew.
        check(seam <= maxStep, "RoomTone seam step is no larger than the in-loop max");
    }

    // -----------------------------------------------------------------------
    // 3. offline render of a positioned mix
    // -----------------------------------------------------------------------
    std::printf("\n[3] renderOffline(2 s) with several positioned sounds\n");
    engine.setListener(Vec3(0, 1.6f, 0), Vec3(0, 0, -1), Vec3(0, 1, 0));
    const int frames = 2 * sr;
    std::vector<float> mix(size_t(frames) * 2, 0.0f);
    engine.play(SoundId::Gunshot, Vec3(0.4f, 1.4f, -1.2f));
    engine.play(SoundId::GlassShatter, Vec3(-0.6f, 1.0f, -2.4f), PlayParams{0.9f, 1.0f, -1.0f, 0.0f, 0.05f});
    engine.play(SoundId::ShellDrop, Vec3(1.5f, 0.02f, -0.5f));
    engine.play(SoundId::Footstep, Vec3(0.0f, 0.0f, -0.3f));
    engine.play2D(SoundId::RoomTone, PlayParams{0.5f, 1.0f, -1.0f, 0.0f, 0.0f});
    engine.play2D(SoundId::EarRing, PlayParams{0.4f, 1.0f, -1.0f, 0.0f, 0.3f});
    engine.renderOffline(mix.data(), frames);
    {
        const Stats s = analyse(mix, sr);
        float maxAbs = 0.0f;
        for (float v : mix) maxAbs = std::max(maxAbs, std::fabs(v));
        std::printf("  mix peak %.4f  rms %.5f  frames %d\n", s.peak, s.rms, s.frames / 2);
        check(s.finite, "offline mix is finite");
        check(s.rms > 0.0005f, "offline mix is non-silent");
        check(maxAbs <= 1.0f, "offline mix never clips (|sample| <= 1.0)");
        check(s.peak > 0.05f, "offline mix has real level");
        check(engine.takePeakLevel().x >= 0.0f, "takePeakLevel() returns a sane value");
        const Vec2 pk = engine.takePeakLevel();
        check(pk.x == 0.0f && pk.y == 0.0f, "takePeakLevel() resets the accumulator");
    }
    // activeVoiceCount / isPlaying should show the voices were running.
    engine.stopAll();
    engine.update(0.2f);  // flush the stop commands through the same path
    check(engine.activeVoiceCount() == 0, "stopAll() retires every voice");

    // -----------------------------------------------------------------------
    // 4. panning
    // -----------------------------------------------------------------------
    std::printf("\n[4] constant-power panning\n");
    {
        auto renderOne = [&](Vec3 pos, float seconds) {
            std::vector<float> out(size_t(seconds * float(sr)) * 2, 0.0f);
            engine.play(SoundId::SlideForward, pos, PlayParams{1.0f, 1.0f, 0.0f, 0.0f, 0.0f});
            engine.renderOffline(out.data(), int(out.size() / 2));
            engine.stopAll();
            engine.update(0.1f);
            return out;
        };
        const std::vector<float> left = renderOne(Vec3(-3.0f, 1.6f, 0.0f), 0.4f);
        const std::vector<float> right = renderOne(Vec3(3.0f, 1.6f, 0.0f), 0.4f);
        const float le = energy(left, 0), re = energy(left, 1);
        const float le2 = energy(right, 0), re2 = energy(right, 1);
        std::printf("  hard left : L=%.6f R=%.6f  (L/R = %.2f)\n", le, re, re > 0 ? le / re : 0.0f);
        std::printf("  hard right: L=%.6f R=%.6f  (R/L = %.2f)\n", le2, re2, le2 > 0 ? re2 / le2 : 0.0f);
        check(le > re * 2.0f, "hard-left source: left channel energy >> right channel energy");
        check(re2 > le2 * 2.0f, "hard-right source: right channel energy >> left channel energy");
        check(le > 0.0f && re2 > 0.0f, "both panned renders are non-silent");
    }

    // -----------------------------------------------------------------------
    // 5. distance model
    // -----------------------------------------------------------------------
    std::printf("\n[5] inverse-distance attenuation\n");
    {
        auto renderAt = [&](float dist, float occlusion) {
            std::vector<float> out(size_t(0.6f * float(sr)) * 2, 0.0f);
            engine.play(SoundId::GunshotTail, Vec3(dist, 1.6f, 0.0f),
                        PlayParams{1.0f, 1.0f, -1.0f, occlusion, 0.0f});
            engine.renderOffline(out.data(), int(out.size() / 2));
            engine.stopAll();
            engine.update(0.1f);
            return out;
        };
        const std::vector<float> near = renderAt(1.0f, 0.0f);
        const std::vector<float> far = renderAt(20.0f, 0.0f);
        const Stats sn = analyse(near, sr);
        const Stats sf = analyse(far, sr);
        std::printf("  near (1 m) rms %.6f   far (20 m) rms %.6f   ratio %.1fx\n", sn.rms, sf.rms,
                    sf.rms > 0 ? sn.rms / sf.rms : 0.0f);
        check(sn.rms > sf.rms, "a 1 m source is louder than the same source at 20 m");
        check(sn.rms > sf.rms * 3.0f, "the distance falloff is substantial (>3x)");
        check(sf.rms > 0.0f, "the far source is still audible");

        // Occlusion: same geometry, but blocked -> quieter and darker.
        const std::vector<float> blocked = renderAt(3.0f, 1.0f);
        const std::vector<float> clear = renderAt(3.0f, 0.0f);
        const Stats sb = analyse(blocked, sr);
        const Stats sc = analyse(clear, sr);
        std::printf("  occlusion: clear rms %.6f (brightness %.6f) vs blocked rms %.6f "
                    "(brightness %.6f)\n",
                    sc.rms, brightness(clear), sb.rms, brightness(blocked));
        check(sb.rms < sc.rms, "occlusion reduces level");
        check(brightness(blocked) <= brightness(clear), "occlusion low-passes (darkens) the source");
    }

    // -----------------------------------------------------------------------
    // 6. scheduling, voice stealing, timing
    // -----------------------------------------------------------------------
    std::printf("\n[6] scheduling, stealing, timing\n");
    {
        // delaySeconds.  A fresh engine is used so that no reverb tail from an
        // earlier block can leak into the "this must be silent" window.
        Engine sched;
        check(sched.init(48000, 2, 16, 0.8f), "scheduling engine initialised");
        sched.setListener(Vec3(0, 1.6f, 0), Vec3(0, 0, -1), Vec3(0, 1, 0));
        std::vector<float> early(size_t(0.05f * float(sr)) * 2, 0.0f);
        sched.play(SoundId::Gunshot, Vec3(0, 1.6f, 0), PlayParams{1.0f, 1.0f, -1.0f, 0.0f, 0.30f});
        sched.renderOffline(early.data(), int(early.size() / 2));
        const Stats se = analyse(early, sr);
        std::vector<float> later(size_t(0.5f * float(sr)) * 2, 0.0f);
        sched.renderOffline(later.data(), int(later.size() / 2));
        const Stats sl = analyse(later, sr);
        std::printf("  delayed voice: first 50 ms rms %.7g -> next 500 ms rms %.6f\n", se.rms, sl.rms);
        check(se.rms < 1e-9f, "delaySeconds=0.3 produces silence before the scheduled start");
        check(sl.rms > 0.001f, "the delayed voice does sound once its delay elapses");
        // A shorter delay must start correspondingly earlier.
        Engine sched2;
        sched2.init(48000, 2, 16, 0.8f);
        sched2.setListener(Vec3(0, 0, 0), Vec3(0, 0, -1), Vec3(0, 1, 0));
        std::vector<float> win(size_t(0.20f * float(sr)) * 2, 0.0f);
        sched2.play(SoundId::Gunshot, Vec3(0, 0, -1), PlayParams{1.0f, 1.0f, -1.0f, 0.0f, 0.05f});
        sched2.renderOffline(win.data(), int(win.size() / 2));
        check(analyse(win, sr).rms > 0.001f, "a 50 ms delay starts well inside a 200 ms window");
        sched.shutdown();
        sched2.shutdown();
    }
    {
        // update(dt) must advance the mix even without a device: a long sound
        // started now must still be reporting as playing a moment later, and
        // must have retired after its full length.
        engine.play(SoundId::Gunshot, Vec3(0, 0, -1));
        engine.update(0.016f);
        check(engine.activeVoiceCount() == 1, "update() starts a voice in null-device mode");
        // update() takes a frame delta (clamped to 0.25 s internally), so drive
        // it the way a game loop would until the 1.28 s gunshot has finished.
        for (int i = 0; i < 120; ++i) engine.update(1.0f / 60.0f);
        check(engine.activeVoiceCount() == 0, "update() retires the voice after its duration");
    }
    {
        // Voice stealing: maxVoices=4, ask for 8 -> calls still succeed and the
        // pool stays saturated instead of dropping the new sounds.
        Engine small;
        check(small.init(48000, 2, 4, 0.8f), "small engine (maxVoices=4) initialised");
        small.setListener(Vec3(0, 0, 0), Vec3(0, 0, -1), Vec3(0, 1, 0));
        std::vector<VoiceId> ids;
        for (int i = 0; i < 8; ++i) {
            ids.push_back(small.play(SoundId::Gunshot, Vec3(0, 0, -1),
                                     PlayParams{0.1f * float(i + 1), 1.0f, -1.0f, 0.0f, 0.0f}));
        }
        bool allValid = true;
        for (VoiceId v : ids) allValid = allValid && v != INVALID_VOICE;
        check(allValid, "play() returns a valid VoiceId even when all voices are busy");
        std::vector<float> buf(size_t(0.25f * float(sr)) * 2, 0.0f);
        small.renderOffline(buf.data(), int(buf.size() / 2));
        std::printf("  active voices after 8 plays into a 4-voice pool: %d\n", small.activeVoiceCount());
        check(small.activeVoiceCount() == 4, "the pool saturates at maxVoices (voice stealing worked)");
        const Stats s = analyse(buf, sr);
        check(s.finite && s.rms > 0.0005f, "the stolen-voice mix is finite and audible");
        check(s.peak <= 1.0f, "the stolen-voice mix does not clip");
        small.shutdown();
    }

    // -----------------------------------------------------------------------
    // 7. WAV dump
    // -----------------------------------------------------------------------
    std::printf("\n[7] dumpWavFiles()\n");
    const std::string dir = "/home/c0m4r/ai/random/room2/build/audio_dump";
    std::filesystem::remove_all(dir);
    const int written = engine.dumpWavFiles(dir);
    uint64_t total = 0;
    int counted = 0;
    for (const auto& e : std::filesystem::directory_iterator(dir)) {
        if (e.is_regular_file()) {
            total += uint64_t(e.file_size());
            ++counted;
        }
    }
    std::printf("  files written: %d (directory contains %d)  total size: %llu bytes (%.1f KiB)\n",
                written, counted, (unsigned long long)total, double(total) / 1024.0);
    check(written == int(kSoundCount), "one WAV per SoundId");
    check(counted == written, "every reported file exists on disk");
    check(total > 0, "total size is non-zero");
    {
        // Decode two files back and report their peak/RMS as durable evidence.
        for (const char* name : {"gunshot", "glass_shatter"}) {
            const std::string path = dir + "/" + name + ".wav";
            const WavInfo w = readWav(path);
            std::printf("  %-14s %s: %d Hz, %d ch, %d bit, %d frames, peak %.4f, rms %.5f, %llu bytes\n",
                        name, w.ok ? "valid" : "INVALID", w.sampleRate, w.channels, w.bits, w.frames,
                        w.peak, w.rms, (unsigned long long)w.bytes);
            check(w.ok, std::string(name) + ".wav parses as 16-bit stereo PCM");
            check(w.sampleRate == sr, std::string(name) + ".wav sample rate matches the engine");
            check(w.frames > 0 && w.peak > 0.05f && w.peak <= 1.0f,
                  std::string(name) + ".wav has real, non-clipping content");
        }
        const std::string rtPath = dir + "/room_tone.wav";
        const WavInfo rt = readWav(rtPath);
        check(rt.ok && rt.frames == 4 * sr, "room_tone.wav is the full 4 s loop");
    }

    // -----------------------------------------------------------------------
    // 8. shutdown safety
    // -----------------------------------------------------------------------
    std::printf("\n[8] shutdown / teardown\n");
    engine.shutdown();
    check(!engine.isInitialised(), "shutdown() clears isInitialised()");
    engine.update(0.016f);                                 // must be a safe no-op
    check(engine.play(SoundId::UiBeep, Vec3(0, 0, -1)) == INVALID_VOICE,
          "play() after shutdown() is rejected");
    engine.stopAll();                                      // must not crash
    engine.setListener(Vec3(0, 0, 0), Vec3(0, 0, -1), Vec3(0, 1, 0));
    check(engine.dumpWavFiles("/home/c0m4r/ai/random/room2/build/audio_dump_after_shutdown") ==
              int(kSoundCount),
          "the bank survives shutdown() for offline dumping");
    check(engine.init(48000, 2, 64, 0.8f), "re-init() after shutdown() succeeds");
    engine.shutdown();

    std::printf("\n=== %d checks, %d failures ===\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
