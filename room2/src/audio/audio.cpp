// room2 - procedural audio engine (implementation).
//
// Everything in this file is synthesised from DSP primitives at runtime: the
// project ships no audio assets and downloads nothing.  The synthesis toolkit
// (noise generators, envelopes, one-pole/biquad/SVF filters, modal resonators,
// Schroeder reverb, soft clipper, DC blocker) lives in the anonymous namespace
// at the top; the concrete sound design for every room2::audio::SoundId follows
// it; the mixer / 3D positional playback engine is at the bottom.
//
// THREADING MODEL (documented per the task requirements)
// -----------------------------------------------------
// The SDL audio callback runs on a separate thread.  Rather than guarding the
// whole mixer with a mutex (which would block the audio thread and cannot be
// held across an allocation), the engine uses a lock-free SPSC command ring:
//
//   * game thread : play()/stop()/stopAll()/setListener()/setMasterVolume()
//                   only ever *push* a POD command into the ring.  Pushing
//                   never allocates, never locks and never blocks; if the ring
//                   is momentarily full a short bounded spin is used and the
//                   command is finally dropped (audio is best-effort).
//   * audio thread: the render path drains the ring at the start of every
//                   block, then mixes.  Drain and mix touch only pre-allocated
//                   state and never allocate.
//
// The ring is a classic Lamport SPSC queue; head_/tail_ are std::atomic with
// acquire/release ordering, and head_ is mirrored in a producer-private cached
// counter so the producer does not touch the consumer-owned cache line on
// every push.
//
// All spatialisation is computed on the *game* thread at play() time (pan,
// inverse-distance attenuation, air-absorption cutoff, occlusion low-pass,
// reverb send, inter-channel delay) and shipped inside the command, so the
// audio thread never reads listener state and the two threads share nothing
// mutable except the ring.
//
// The voice pool is only mutated by whichever thread currently owns the render
// path (the SDL callback when a device is open, otherwise the game thread
// inside update()/renderOffline()).  VoiceId handles are compared against the
// slot's stored handle so a stale handle can never address a recycled slot.
// Diagnostics the game thread may read at any time (peak meters, active voice
// count, per-voice "playing" bits) are plain atomics written by the render
// thread; they are advisory and never drive mixer state.

#include "audio.hpp"

#include <SDL.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <new>
#include <string>
#include <thread>
#include <vector>

namespace room2::audio {
namespace {

// ===========================================================================
// 1. deterministic PRNG + noise generators
// ===========================================================================

// Small xorshift32 PRNG.  Deterministic and explicitly seeded, so the whole
// bank is bit-identical from run to run (no rand(), no std::random_device).
class Rng {
public:
    explicit Rng(uint32_t seed = 0x1234567u) { reseed(seed); }
    void reseed(uint32_t seed) {
        state_ = seed ? seed : 0x9E3779B9u;  // xorshift must not be seeded with 0
        for (int i = 0; i < 4; ++i) nextU32();
    }
    uint32_t nextU32() {
        uint32_t x = state_;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        state_ = x;
        return x;
    }
    float nextUnit() { return float(nextU32() >> 8) * (1.0f / 16777216.0f); }  // [0,1)
    float nextBipolar() { return nextUnit() * 2.0f - 1.0f; }                    // [-1,1)
    float nextRange(float lo, float hi) { return lo + (hi - lo) * nextUnit(); }
    // Approximately gaussian (sum of 3 uniforms), unit-ish variance.
    float nextGaussian() { return ((nextUnit() + nextUnit() + nextUnit()) - 1.5f) * 1.1547f; }
    bool chance(float p) { return nextUnit() < p; }
    // Log-uniform: perceptually even spread of e.g. resonant frequencies.
    float nextLogRange(float lo, float hi) {
        return std::exp(lerpf(std::log(lo), std::log(hi), nextUnit()));
    }

private:
    uint32_t state_ = 0x1234567u;
};

inline float whiteNoise(Rng& r) { return r.nextBipolar(); }

// Paul Kellet's economical pink-noise filter (-3 dB/octave).
class PinkNoise {
public:
    float process(Rng& r) {
        float w = whiteNoise(r);
        b0_ = 0.99886f * b0_ + w * 0.0555179f;
        b1_ = 0.99332f * b1_ + w * 0.0750759f;
        b2_ = 0.96900f * b2_ + w * 0.1538520f;
        b3_ = 0.86650f * b3_ + w * 0.3104856f;
        b4_ = 0.55000f * b4_ + w * 0.5329522f;
        b5_ = -0.7616f * b5_ - w * 0.0168980f;
        float out = b0_ + b1_ + b2_ + b3_ + b4_ + b5_ + b6_ + w * 0.5362f;
        b6_ = w * 0.115926f;
        return out * 0.34f;
    }

private:
    float b0_ = 0, b1_ = 0, b2_ = 0, b3_ = 0, b4_ = 0, b5_ = 0, b6_ = 0;
};

// Leaky-integrated white noise (-6 dB/octave), clamped to roughly unit range.
class BrownNoise {
public:
    float process(Rng& r) {
        state_ = (state_ + 0.026f * whiteNoise(r)) * 0.9985f;
        return clamp(state_ * 9.0f, -1.0f, 1.0f);
    }

private:
    float state_ = 0.0f;
};

// ===========================================================================
// 2. envelopes
// ===========================================================================

// Exponential decay envelope.  `t60` is the time to fall 60 dB; the value is
// exactly 1 at t=0 and snaps to 0 once it is inaudible, so buffers always end
// in true silence.
inline float expDecay(float t, float t60) {
    if (t <= 0.0f) return 1.0f;
    float a = 6.9077553f / std::max(t60, 1e-5f);  // ln(1000)
    float v = std::exp(-a * t);
    return v < 1e-6f ? 0.0f : v;
}

// Exponential decay with an explicit linear attack ramp (seconds).
inline float expDecayA(float t, float attack, float t60) {
    if (t <= 0.0f) return 0.0f;
    float e = expDecay(t, t60);
    return attack <= 0.0f ? e : e * clamp(t / attack, 0.0f, 1.0f);
}

// Raised-cosine (equal-power-ish) fade pair; a+b == 1 across the whole span.
inline float fadeIn(int i, int n) {
    if (n <= 1) return 1.0f;
    float x = float(i) / float(n - 1);
    return 0.5f - 0.5f * std::cos(PI * x);
}
inline float fadeOut(int i, int n) { return fadeIn(n - 1 - i, n); }

// Classic ADSR.  Segment based, sample-accurate, no allocation.
class Adsr {
public:
    void set(float attack, float decay, float sustain, float release) {
        a_ = std::max(attack, 0.0f);
        d_ = std::max(decay, 0.0f);
        s_ = clamp(sustain, 0.0f, 1.0f);
        r_ = std::max(release, 0.0f);
    }
    void gate(bool on) {
        if (on == gate_) return;
        gate_ = on;
        if (on) {
            stage_ = Stage::Attack;
        } else if (stage_ != Stage::Idle) {
            releaseStart_ = value_;
            stage_ = Stage::Release;
        }
    }
    float process(float dt) {
        switch (stage_) {
            case Stage::Idle: return 0.0f;
            case Stage::Attack:
                value_ += a_ > 0.0f ? dt / a_ : 1.0f;
                if (value_ >= 1.0f) { value_ = 1.0f; stage_ = Stage::Decay; }
                break;
            case Stage::Decay:
                value_ -= d_ > 0.0f ? dt / d_ : 1.0f;
                if (value_ <= s_) { value_ = s_; stage_ = Stage::Sustain; }
                break;
            case Stage::Sustain: value_ = s_; break;
            case Stage::Release:
                value_ -= r_ > 0.0f ? (releaseStart_ * dt / r_) : 1.0f;
                if (value_ <= 0.0f) { value_ = 0.0f; stage_ = Stage::Idle; }
                break;
        }
        return value_;
    }
    float value() const { return value_; }
    bool done() const { return stage_ == Stage::Idle; }

private:
    enum class Stage { Idle, Attack, Decay, Sustain, Release };
    Stage stage_ = Stage::Idle;
    bool gate_ = false;
    float a_ = 0.005f, d_ = 0.05f, s_ = 0.7f, r_ = 0.1f;
    float value_ = 0.0f, releaseStart_ = 0.0f;
};

// ===========================================================================
// 3. filters
// ===========================================================================

struct OnePoleLP {
    float z = 0.0f;
    void reset() { z = 0.0f; }
    float process(float x, float cutoff, int sr) {
        float a = 1.0f - std::exp(-TWO_PI * clamp(cutoff, 1.0f, 0.45f * float(sr)) / float(sr));
        z += a * (x - z);
        return z;
    }
};

struct OnePoleHP {
    float z = 0.0f;
    void reset() { z = 0.0f; }
    float process(float x, float cutoff, int sr) {
        float a = 1.0f - std::exp(-TWO_PI * clamp(cutoff, 1.0f, 0.45f * float(sr)) / float(sr));
        z += a * (x - z);
        return x - z;
    }
};

// Chamberlin state-variable filter: simultaneous LP/BP/HP outputs with an
// adjustable Q.  Lightly oversampled (2x) to stay stable at high cutoffs.
struct Svf {
    float lp = 0.0f, bp = 0.0f;
    void reset() { lp = bp = 0.0f; }
    void process(float x, float cutoff, float q, int sr) {
        float f = 2.0f * std::sin(PI * clamp(cutoff, 20.0f, 0.45f * float(sr)) / float(sr));
        float damp = clamp(1.0f / std::max(q, 0.05f), 0.02f, 2.0f);
        for (int i = 0; i < 2; ++i) {  // 2x oversampled
            lp += f * bp;
            float hp = x - lp - damp * bp;
            bp += f * hp;
        }
    }
    float lowpass() const { return lp; }
    float bandpass() const { return bp; }
    float highpass(float in) const { return in - lp - bp; }
};

// Transposed-direct-form-II biquad (RBJ cookbook coefficients).
struct Biquad {
    float b0 = 1.0f, b1 = 0.0f, b2 = 0.0f, a1 = 0.0f, a2 = 0.0f;
    float z1 = 0.0f, z2 = 0.0f;
    void reset() { z1 = z2 = 0.0f; }
    float process(float x) {
        float y = b0 * x + z1;
        z1 = b1 * x - a1 * y + z2;
        z2 = b2 * x - a2 * y;
        return y;
    }
    void setLowpass(float fc, float q, int sr) {
        float w = TWO_PI * clamp(fc, 10.0f, 0.45f * float(sr)) / float(sr);
        float cw = std::cos(w), sw = std::sin(w);
        float alpha = sw / (2.0f * std::max(q, 0.05f));
        float a0 = 1.0f + alpha;
        b0 = (1.0f - cw) * 0.5f / a0;
        b1 = (1.0f - cw) / a0;
        b2 = b0;
        a1 = -2.0f * cw / a0;
        a2 = (1.0f - alpha) / a0;
    }
    void setHighpass(float fc, float q, int sr) {
        float w = TWO_PI * clamp(fc, 10.0f, 0.45f * float(sr)) / float(sr);
        float cw = std::cos(w), sw = std::sin(w);
        float alpha = sw / (2.0f * std::max(q, 0.05f));
        float a0 = 1.0f + alpha;
        b0 = (1.0f + cw) * 0.5f / a0;
        b1 = -(1.0f + cw) / a0;
        b2 = b0;
        a1 = -2.0f * cw / a0;
        a2 = (1.0f - alpha) / a0;
    }
    void setBandpass(float fc, float q, int sr) {  // 0 dB peak gain
        float w = TWO_PI * clamp(fc, 10.0f, 0.45f * float(sr)) / float(sr);
        float cw = std::cos(w), sw = std::sin(w);
        float alpha = sw / (2.0f * std::max(q, 0.05f));
        float a0 = 1.0f + alpha;
        b0 = alpha / a0;
        b1 = 0.0f;
        b2 = -alpha / a0;
        a1 = -2.0f * cw / a0;
        a2 = (1.0f - alpha) / a0;
    }
    void setPeak(float fc, float q, float gainDb, int sr) {
        float w = TWO_PI * clamp(fc, 10.0f, 0.45f * float(sr)) / float(sr);
        float cw = std::cos(w), sw = std::sin(w);
        float alpha = sw / (2.0f * std::max(q, 0.05f));
        float A = std::pow(10.0f, gainDb / 40.0f);
        float a0 = 1.0f + alpha / A;
        b0 = (1.0f + alpha * A) / a0;
        b1 = -2.0f * cw / a0;
        b2 = (1.0f - alpha * A) / a0;
        a1 = -2.0f * cw / a0;
        a2 = (1.0f - alpha / A) / a0;
    }
};

// One-pole DC blocker (also removes the sub-audible wander of noise gens).
struct DcBlocker {
    float x1 = 0.0f, y1 = 0.0f;
    void reset() { x1 = y1 = 0.0f; }
    float process(float x, float r = 0.9995f) {
        float y = x - x1 + r * y1;
        x1 = x;
        y1 = y;
        return y;
    }
};

// ===========================================================================
// 4. modal (waveguide-ish) resonator bank - the metal / glass voice
// ===========================================================================

// A bank of independent inharmonic partials.  Each partial is a two-pole
// resonator fed by an excitation burst, with its own decay rate.  Ratio
// inharmonicity + independent decay rates + detuned beating copies are what
// separate a convincing glass/metal ring from a sine blip.
struct ModeSpec {
    float ratio;   // frequency multiplier relative to the fundamental
    float gain;    // linear amplitude
    float t60;     // decay time to -60 dB (seconds)
    float detune;  // fractional detune of the second (beating) copy
};

// Renders one partial (plus its detuned twin) additively into out.
inline void renderModalPartial(std::vector<float>& out, int sr, float freq, float gain, float t60,
                               float detune, int onset) {
    int n = int(out.size());
    if (freq <= 1.0f || freq >= 0.45f * float(sr) || gain <= 0.0f || onset >= n) return;
    float amp = gain * (1.0f - std::exp(-6.9077553f / std::max(t60, 1e-4f) / float(sr)));
    float f2 = freq * (1.0f + detune);
    float a1 = std::exp(-6.9077553f / (t60 * float(sr)));
    float a2 = std::exp(-6.9077553f / (t60 * 1.13f * float(sr)));
    float c1 = 2.0f * a1 * std::cos(TWO_PI * freq / float(sr));
    float c2 = 2.0f * a2 * std::cos(TWO_PI * f2 / float(sr));
    float y1 = 0.0f, ym1 = 0.0f, y2 = 0.0f, ym2 = 0.0f;
    int end = std::min(n, onset + int(t60 * 4.0f * float(sr)) + 8);
    for (int i = std::max(onset, 0); i < end; ++i) {
        float y = amp + c1 * y1 - a1 * a1 * ym1;
        ym1 = y1;
        y1 = y;
        float z = amp + c2 * y2 - a2 * a2 * ym2;
        ym2 = y2;
        y2 = z;
        out[size_t(i)] += 0.5f * (y + z);
    }
}

// Adds an inharmonic modal bank excited at `onset`.
inline void addModalBank(std::vector<float>& out, int sr, float f0, const ModeSpec* modes, int count,
                         float gain, float t60Scale, float detuneAmount, int onset) {
    for (int m = 0; m < count; ++m) {
        renderModalPartial(out, sr, f0 * modes[m].ratio, gain * modes[m].gain,
                           modes[m].t60 * t60Scale, modes[m].detune * detuneAmount, onset);
    }
}

// ===========================================================================
// 5. delay line, comb / allpass, Schroeder reverb, early reflections
// ===========================================================================

class DelayLine {
public:
    void init(int maxSamples) {
        buf_.assign(size_t(std::max(maxSamples, 4)), 0.0f);
        w_ = 0;
    }
    void clear() { std::fill(buf_.begin(), buf_.end(), 0.0f); }
    // Fractional read (linear interpolation) `d` samples behind the write head.
    float read(float d) const {
        int n = int(buf_.size());
        if (n == 0) return 0.0f;
        d = clamp(d, 1.0f, float(n - 2));
        float rp = float(w_) - d;
        while (rp < 0.0f) rp += float(n);
        int i0 = int(rp) % n;
        float fr = rp - std::floor(rp);
        int i1 = (i0 + 1) % n;
        return buf_[size_t(i0)] * (1.0f - fr) + buf_[size_t(i1)] * fr;
    }
    void write(float v) {
        if (buf_.empty()) return;
        buf_[size_t(w_)] = v;
        w_ = (w_ + 1) % int(buf_.size());
    }
    int size() const { return int(buf_.size()); }

private:
    std::vector<float> buf_;
    int w_ = 0;
};

// Lowpass-damped feedback comb (Schroeder/Freeverb style).
struct DampedComb {
    DelayLine dl;
    float store = 0.0f;
    float feedback = 0.84f;
    float damp1 = 0.25f, damp2 = 0.75f;
    float process(float x, float delaySamples) {
        float y = dl.read(delaySamples);
        store = y * damp2 + store * damp1;
        dl.write(x + store * feedback);
        return y;
    }
};

// Allpass with independent input and feedback coefficients.  This form lets us
// keep a short, perceptually dense diffusion stage while still solving the
// coefficients so the stage decays -60 dB in the requested reverb time.
struct Allpass {
    DelayLine dl;
    float g1 = 0.5f, g2 = 0.5f;
    float process(float x, float delaySamples) {
        float d = dl.read(delaySamples);
        float v = x + g1 * d;
        dl.write(v);
        return d - g2 * v;
    }
};

// Schroeder reverb: 4 parallel damped combs -> 2 series allpasses per channel,
// with decorrelated tunings.  Tuned for the scene's room: ~6 x 3 x 8 m hard
// plaster, i.e. bright, dense, short predelay and a ~1 s decay.
class SchroederReverb {
public:
    void init(int sr, float rt60, float damping, float sizeScale) {
        sr_ = sr;
        static const int baseComb[4] = {1116, 1188, 1277, 1356};  // ~23-28 ms at 48 kHz
        static const int baseAp[2] = {556, 441};                  // ~11.6 / 9.2 ms
        rt60 = std::max(rt60, 0.05f);
        for (int ch = 0; ch < 2; ++ch) {
            for (int c = 0; c < 4; ++c) {
                int d = std::max(int(float(baseComb[c] + ch * 23) * sizeScale), 16);
                combs_[ch][c].dl.init(d + 4);
                combDelay_[ch][c] = float(d);
                combs_[ch][c].feedback =
                    clamp(std::pow(10.0f, -3.0f * float(d) / rt60 / float(sr)), 0.05f, 0.9955f);
                combs_[ch][c].damp1 = clamp(damping, 0.0f, 0.95f);
                combs_[ch][c].damp2 = 1.0f - combs_[ch][c].damp1;
            }
            for (int a = 0; a < 2; ++a) {
                int d = std::max(int(float(baseAp[a] + ch * 17) * sizeScale), 8);
                allpass_[ch][a].dl.init(d + 4);
                apDelay_[ch][a] = float(d);
                // Allpass stage decay: solve |g1*g2| = 10^(-3*D/RT60).
                float target = std::pow(10.0f, -3.0f * float(d) / rt60 / float(sr));
                allpass_[ch][a].g1 = clamp(std::sqrt(target), 0.0f, 0.9f);
                allpass_[ch][a].g2 = allpass_[ch][a].g1;
            }
        }
    }
    void reset() {
        for (int ch = 0; ch < 2; ++ch) {
            for (int c = 0; c < 4; ++c) {
                combs_[ch][c].dl.clear();
                combs_[ch][c].store = 0.0f;
            }
            for (int a = 0; a < 2; ++a) allpass_[ch][a].dl.clear();
        }
    }
    // Wet output gain.  Each damped comb has a DC gain near 1/(1-feedback) ~ 12,
    // so the comb sum is scaled by 1/4 * 1/12 to bring the steady-state gain back
    // to roughly unity.  Without this the reverb return swamps the dry signal and
    // the per-sound send has almost no audible effect (the combs reach full level
    // even for a very small send).
    static constexpr float kWetGain = 0.021f;

    void process(float inL, float inR, float& outL, float& outR) {
        float in[2] = {inL, inR};
        float o[2] = {0.0f, 0.0f};
        for (int ch = 0; ch < 2; ++ch) {
            float acc = 0.0f;
            for (int c = 0; c < 4; ++c) acc += combs_[ch][c].process(in[ch], combDelay_[ch][c]);
            acc *= 0.25f * kWetGain;
            for (int a = 0; a < 2; ++a) acc = allpass_[ch][a].process(acc, apDelay_[ch][a]);
            o[ch] = acc;
        }
        outL = o[0];
        outR = o[1];
    }
    void processBlock(float* buf, int frames) {  // in-place stereo
        for (int i = 0; i < frames; ++i) {
            float l, r;
            process(buf[2 * i], buf[2 * i + 1], l, r);
            buf[2 * i] = l;
            buf[2 * i + 1] = r;
        }
    }

private:
    int sr_ = 48000;
    DampedComb combs_[2][4];
    Allpass allpass_[2][2];
    float combDelay_[2][4] = {};
    float apDelay_[2][2] = {};
};

// Sparse early-reflection network: ten decorrelated taps per channel, matching
// the first-order reflections of a 6 x 3 x 8 m box.
class EarlyReflections {
public:
    void init(int sr, float preDelaySeconds) {
        sr_ = sr;
        static const float taps[10] = {0.0053f, 0.0081f, 0.0114f, 0.0149f, 0.0188f,
                                       0.0231f, 0.0277f, 0.0329f, 0.0386f, 0.0447f};
        static const float gains[10] = {0.62f, 0.55f, 0.48f, 0.42f, 0.37f,
                                        0.33f, 0.29f, 0.25f, 0.22f, 0.19f};
        pre_ = int(preDelaySeconds * float(sr));
        for (int ch = 0; ch < 2; ++ch) {
            for (int t = 0; t < 10; ++t) {
                // Mild inter-channel decorrelation; hard plaster keeps the
                // highs alive, so only a gentle top-end tilt is applied.
                float jitter =
                    1.0f + (ch == 0 ? 0.0f : 0.031f) + float(t) * (ch == 0 ? 0.004f : -0.006f);
                int d = std::max(int(taps[t] * jitter * float(sr)) + pre_, 4);
                dl_[ch][t].init(d + 4);
                delay_[ch][t] = float(d);
                gain_[ch][t] = gains[t] * (ch == 0 ? 1.0f : 0.93f);
            }
            lp_[ch].reset();
        }
    }
    void reset() {
        for (int ch = 0; ch < 2; ++ch) {
            for (int t = 0; t < 10; ++t) dl_[ch][t].clear();
            lp_[ch].reset();
        }
    }
    void process(float inL, float inR, float& outL, float& outR) {
        float in[2] = {inL, inR};
        float o[2] = {0.0f, 0.0f};
        for (int ch = 0; ch < 2; ++ch) {
            float acc = 0.0f;
            for (int t = 0; t < 10; ++t) {
                acc += dl_[ch][t].read(delay_[ch][t]) * gain_[ch][t];
                dl_[ch][t].write(in[ch]);
            }
            float dark = lp_[ch].process(acc, 14000.0f, sr_);
            o[ch] = acc * 0.95f + dark * 0.05f;
        }
        outL = o[0];
        outR = o[1];
    }
    void processBlock(float* buf, int frames) {
        for (int i = 0; i < frames; ++i) {
            float l, r;
            process(buf[2 * i], buf[2 * i + 1], l, r);
            buf[2 * i] = l;
            buf[2 * i + 1] = r;
        }
    }

private:
    int sr_ = 48000, pre_ = 0;
    DelayLine dl_[2][10];
    float gain_[2][10] = {};
    float delay_[2][10] = {};
    OnePoleLP lp_[2];
};

// ===========================================================================
// 6. saturation / utility processing
// ===========================================================================

// Smooth odd-symmetric soft clipper.  tanh is bounded by 1, so callers can rely
// on the output never leaving [-1,1].  Used as the final safety net on the mix
// bus, where driving it is intended.
inline float softClip(float x) {
    if (x <= -3.0f) return -1.0f;
    if (x >= 3.0f) return 1.0f;
    return std::tanh(x);
}
// Odd-symmetric saturating knee: fully linear below `knee`, then a smooth
// compressive curve asymptoting at knee + 1/3.  Unlike driving tanh hard, this
// keeps a normalised peak essentially intact (a sample at ~knee comes out within
// ~2 % of where it went in) while making it impossible to exceed 1.0.  That
// matters here: a bank sound is normalised to ~0.97 and must stay there.
inline float softKnee(float x, float knee) {
    float a = std::fabs(x);
    if (a <= knee) return x;
    float over = a - knee;
    return signf(x) * (knee + over / (1.0f + 3.0f * over));
}

// Scales a buffer so its peak is `target`, then applies the soft knee so the
// result can never reach full scale.  Returns the peak before normalisation.
// Every bank sound is finalised through this, which is what guarantees both the
// "peak near full scale" and the "must not clip" requirements at once.
inline float normalizeBuffer(std::vector<float>& b, float target) {
    float peak = 0.0f;
    for (float v : b)
        if (std::isfinite(v)) peak = std::max(peak, std::fabs(v));
    if (peak < 1e-9f) return 0.0f;
    const float g = target / peak;
    for (float& v : b) v = softKnee(v * g, target * 0.97f);
    return peak;
}

// DC blocker, run twice with the filter state pre-settled.  A single cold pass
// leaves a one-sample transient at the head of the buffer (and would break a
// seamless loop), so the first pass only warms the state up.
inline void applyDcBlock(std::vector<float>& b) {
    if (b.empty()) return;
    DcBlocker dc;
    for (float v : b) dc.process(v);  // warm-up pass, result discarded
    for (float& v : b) v = dc.process(v);
}

// Trims trailing near-silence so bank entries are as short as they can be while
// still decaying naturally.  Keeps at least `minSamples`.
inline void trimTail(std::vector<float>& b, float threshold, int minSamples) {
    if (b.empty()) return;
    int floorIdx = std::min(std::max(minSamples, 0), int(b.size()) - 1);
    int last = int(b.size()) - 1;
    while (last > floorIdx && std::fabs(b[size_t(last)]) < threshold) --last;
    int newLen = std::min(last + 1 + 240, int(b.size()));
    b.resize(size_t(std::max(newLen, floorIdx + 1)));
}

// ===========================================================================
// 7. synthesis primitives built on the toolkit above
// ===========================================================================

// A short burst of band-passed noise: the workhorse transient primitive.
inline void addNoiseBurst(std::vector<float>& out, Rng& rng, int sr, float start, float t60,
                          float attack, float hp, float lp, float bp, float q, float amp) {
    int n = int(out.size());
    int i0 = std::max(int(start * float(sr)), 0);
    if (i0 >= n) return;
    int len = std::min(int(t60 * 4.5f * float(sr)) + 64, n - i0);
    if (len <= 0) return;
    Biquad bhp, blp, bbp;
    bhp.setHighpass(hp, 0.707f, sr);
    blp.setLowpass(lp, 0.707f, sr);
    bool useBp = bp > 20.0f;
    if (useBp) bbp.setBandpass(bp, std::max(q, 0.2f), sr);
    float inv = 1.0f / float(sr);
    for (int i = 0; i < len; ++i) {
        float t = float(i) * inv;
        float x = bhp.process(whiteNoise(rng));
        x = blp.process(x);
        if (useBp) x = bbp.process(x) * 2.2f;
        out[size_t(i0 + i)] += x * amp * expDecayA(t, attack, t60);
    }
}

// Swept sine ("thump"/"blast" primitive): f0 -> f1 exponentially over `sweep`.
inline void addSineSweep(std::vector<float>& out, int sr, float start, float f0, float f1,
                         float sweep, float t60, float attack, float amp) {
    int n = int(out.size());
    int i0 = std::max(int(start * float(sr)), 0);
    if (i0 >= n) return;
    int len = std::min(int(t60 * 4.5f * float(sr)) + 64, n - i0);
    if (len <= 0) return;
    float phase = 0.0f, inv = 1.0f / float(sr);
    float k = std::log(std::max(f1, 1.0f) / std::max(f0, 1.0f)) / std::max(sweep, 1e-4f);
    for (int i = 0; i < len; ++i) {
        float t = float(i) * inv;
        float f = f0 * std::exp(k * t);
        phase += TWO_PI * f * inv;
        if (phase > TWO_PI) phase -= TWO_PI;
        out[size_t(i0 + i)] += std::sin(phase) * amp * expDecayA(t, attack, t60);
    }
}

// Amplitude-modulated filtered noise: the friction / scrape / rustle primitive.
// `fric` is the stick-slip modulation rate in Hz, `depth` its depth (0..1).
inline void addScrape(std::vector<float>& out, Rng& rng, int sr, float start, float dur, float hp,
                      float lp, float bp, float q, float attack, float release, float amp,
                      float fric, float depth) {
    int n = int(out.size());
    int i0 = std::max(int(start * float(sr)), 0);
    if (i0 >= n) return;
    int len = std::min(int(dur * float(sr)), n - i0);
    if (len <= 0) return;
    Biquad bhp, blp, bbp;
    bhp.setHighpass(hp, 0.707f, sr);
    blp.setLowpass(lp, 0.707f, sr);
    bool useBp = bp > 20.0f;
    if (useBp) bbp.setBandpass(bp, std::max(q, 0.2f), sr);
    float inv = 1.0f / float(sr);
    for (int i = 0; i < len; ++i) {
        float t = float(i) * inv;
        float env = expDecayA(t, attack, std::max(dur, 1e-3f) * 0.42f);
        if (release > 0.0f) env *= clamp((dur - t) / release, 0.0f, 1.0f);
        float x = bhp.process(whiteNoise(rng));
        x = blp.process(x);
        if (useBp) x = bbp.process(x) * 2.0f;
        float mod = 1.0f - depth + depth * (0.5f + 0.5f * std::sin(TWO_PI * fric * t));
        out[size_t(i0 + i)] += x * amp * env * mod;
    }
}

// Adds a decaying sinusoidal partial with optional beating detune.
inline void addTone(std::vector<float>& out, int sr, float start, float freq, float t60,
                    float attack, float amp, float detune) {
    int n = int(out.size());
    int i0 = std::max(int(start * float(sr)), 0);
    if (i0 >= n || freq <= 0.0f || freq >= 0.45f * float(sr)) return;
    int len = std::min(int(t60 * 4.5f * float(sr)) + 64, n - i0);
    if (len <= 0) return;
    float inv = 1.0f / float(sr);
    float ph1 = 0.0f, ph2 = 0.0f;
    float f2 = freq * (1.0f + detune);
    for (int i = 0; i < len; ++i) {
        float t = float(i) * inv;
        float env = expDecayA(t, attack, t60);
        ph1 += TWO_PI * freq * inv;
        ph2 += TWO_PI * f2 * inv;
        if (ph1 > TWO_PI) ph1 -= TWO_PI;
        if (ph2 > TWO_PI) ph2 -= TWO_PI;
        float s = detune > 0.0f ? 0.5f * (std::sin(ph1) + std::sin(ph2)) : std::sin(ph1);
        out[size_t(i0 + i)] += s * amp * env;
    }
}

// Adds one stereo shard-like impact: noise onset + inharmonic glass modes.
// `pan` in [-1,1] is folded to mono with constant-power channel weights, so the
// shower keeps a sense of width even though a bank entry is mono.
inline void addShard(std::vector<float>& out, Rng& rng, int sr, float start, float f0, float t60,
                     float brightness, float gain, float pan) {
    int n = int(out.size());
    int i0 = std::max(int(start * float(sr)), 0);
    if (i0 >= n) return;
    float gl = std::sqrt(0.5f * (1.0f - pan));
    float gr = std::sqrt(0.5f * (1.0f + pan));
    std::vector<float> tmp(size_t(n), 0.0f);
    addNoiseBurst(tmp, rng, sr, start, 0.0016f, 0.0002f, 3800.0f * brightness, 19000.0f, 0.0f, 1.0f,
                  gain * 0.55f);
    static const ModeSpec modes[6] = {{1.000f, 1.00f, 0.30f, 0.0025f},  {2.414f, 0.62f, 0.20f, 0.0031f},
                                      {3.917f, 0.44f, 0.14f, 0.0042f},  {5.731f, 0.30f, 0.10f, 0.0055f},
                                      {8.113f, 0.20f, 0.07f, 0.0071f},  {11.417f, 0.12f, 0.05f, 0.0090f}};
    addModalBank(tmp, sr, f0, modes, 6, gain, t60, 1.0f, i0);
    float fold = (gl + gr) * 0.70710678f;
    for (int i = i0; i < n; ++i) out[size_t(i)] += tmp[size_t(i)] * fold;
}

// ===========================================================================
// 8. room tail rendering
// ===========================================================================

// A bright, hard-plaster room response.  `direct` is the dry source material;
// the return value is the early-reflection + late-reverb part alone.
std::vector<float> renderRoomTail(const std::vector<float>& direct, int sr, float rt60,
                                  float predelay, float earlyMix, float tailMix) {
    const int n = int(direct.size());
    std::vector<float> out(size_t(n), 0.0f);
    EarlyReflections er;
    er.init(sr, predelay);
    SchroederReverb rev;
    rev.init(sr, rt60, 0.20f, 1.0f);
    for (int i = 0; i < n; ++i) {
        float in = direct[size_t(i)];
        float l = 0.0f, r = 0.0f, rl = 0.0f, rr = 0.0f;
        er.process(in, in, l, r);
        rev.process(in, in, rl, rr);
        out[size_t(i)] += (l + r) * 0.5f * earlyMix + (rl + rr) * 0.5f * tailMix;
    }
    return out;
}

// ===========================================================================
// 9. the individual sound designs
// ===========================================================================

std::vector<float> synthGunshot(int sr, bool tailOnly) {
    const int n = int(1.28f * float(sr));
    std::vector<float> dry(size_t(n), 0.0f);
    // The direct sound is always synthesised: it is what excites the room, so
    // tail-only mode renders the very same event and then returns the tail
    // alone (that is exactly what "the same room tail, no direct sound" means).
    {
        Rng rng(0xA17B00C5u);
        // (a) ~1 ms mechanical click: primer / sear / breech transient.
        Biquad hp1, hp2;
        hp1.setHighpass(1200.0f, 0.707f, sr);
        hp2.setHighpass(4000.0f, 0.707f, sr);
        int clickLen = int(0.0012f * float(sr));
        for (int i = 0; i < clickLen; ++i) {
            float t = float(i) / float(sr);
            float x = whiteNoise(rng) + 0.4f * rng.nextGaussian();
            x = hp1.process(x) * 0.7f + hp2.process(x) * 0.6f;
            dry[size_t(i)] += x * expDecay(t, 0.00055f);
        }
        // (b) bright supersonic "crack": fast-decaying high-passed noise.
        addNoiseBurst(dry, rng, sr, 0.0f, 0.0075f, 0.00008f, 2600.0f, 19000.0f, 0.0f, 1.0f, 1.15f);
        addNoiseBurst(dry, rng, sr, 0.0004f, 0.022f, 0.0002f, 6000.0f, 19000.0f, 7500.0f, 0.7f, 0.55f);
        // (c) mid "body" of the report: band-passed noise, ~60-120 ms decay.
        addNoiseBurst(dry, rng, sr, 0.0006f, 0.095f, 0.0012f, 300.0f, 6500.0f, 950.0f, 0.9f, 0.95f);
        addNoiseBurst(dry, rng, sr, 0.0008f, 0.075f, 0.0010f, 500.0f, 5000.0f, 1900.0f, 1.4f, 0.55f);
        // (d) muzzle-blast thump: 130 Hz -> 45 Hz sweep plus a 60 Hz tail.
        addSineSweep(dry, sr, 0.0f, 130.0f, 45.0f, 0.085f, 0.055f, 0.0015f, 1.0f);
        addSineSweep(dry, sr, 0.0f, 95.0f, 52.0f, 0.13f, 0.115f, 0.0025f, 0.55f);
        addNoiseBurst(dry, rng, sr, 0.0f, 0.05f, 0.0015f, 45.0f, 340.0f, 110.0f, 1.1f, 0.7f);
        // Slide/barrel mechanical resonance: a short metallic edge so the shot is
        // not purely noise-based.
        static const ModeSpec gunModes[5] = {{1.0f, 1.0f, 0.035f, 0.004f},
                                             {2.73f, 0.55f, 0.025f, 0.005f},
                                             {4.61f, 0.34f, 0.017f, 0.006f},
                                             {7.09f, 0.20f, 0.011f, 0.008f},
                                             {9.83f, 0.12f, 0.008f, 0.010f}};
        addModalBank(dry, sr, 720.0f, gunModes, 5, 0.35f, 1.0f, 1.0f, 0);
        applyDcBlock(dry);
    }
    // (e) dense early reflections + late reverb: ~1 s, hard and bright.
    std::vector<float> tail = renderRoomTail(dry, sr, 1.02f, 0.0045f, 0.62f, 1.0f);
    // A second, lighter pass over the tail thickens the late field (the room
    // keeps re-exciting itself), which is what makes a small hard room dense
    // rather than a single slap.
    std::vector<float> tail2 = renderRoomTail(tail, sr, 0.95f, 0.0f, 0.10f, 0.22f);
    for (size_t i = 0; i < tail.size(); ++i) tail[i] += tail2[i] * 0.55f;

    std::vector<float> out;
    if (tailOnly) {
        out = std::move(tail);
        normalizeBuffer(out, 0.94f);
    } else {
        out.assign(size_t(n), 0.0f);
        for (size_t i = 0; i < out.size(); ++i) out[i] = dry[i] + tail[i] * 0.92f;
        normalizeBuffer(out, 0.97f);  // ~full scale, knee-bounded well under 1.0
    }
    return out;
}

// Common metallic click builder: noise transient + inharmonic modal bank.
std::vector<float> synthMetalClick(int sr, float length, float f0, float decayScale, float noiseHp,
                                   float noiseLp, float noiseT60, float noiseGain, float modalGain,
                                   uint32_t seed) {
    Rng rng(seed);
    std::vector<float> out(size_t(int(length * float(sr))), 0.0f);
    addNoiseBurst(out, rng, sr, 0.0f, noiseT60, 0.00012f, noiseHp, noiseLp, 0.0f, 1.0f, noiseGain);
    static const ModeSpec modes[6] = {{1.000f, 1.00f, 0.09f, 0.006f},  {2.381f, 0.66f, 0.062f, 0.008f},
                                      {3.803f, 0.46f, 0.041f, 0.010f}, {5.917f, 0.30f, 0.027f, 0.013f},
                                      {8.211f, 0.19f, 0.018f, 0.016f}, {12.037f, 0.11f, 0.012f, 0.020f}};
    addModalBank(out, sr, f0, modes, 6, modalGain, decayScale, 1.0f, 0);
    applyDcBlock(out);
    normalizeBuffer(out, 0.95f);
    return out;
}

std::vector<float> synthDryFire(int sr) {
    Rng rng(0x51D3A001u);
    const int n = int(0.22f * float(sr));
    std::vector<float> out(size_t(n), 0.0f);
    // Striker falling on an empty chamber: dry, snappy, slightly metallic.
    Biquad hp, bp;
    hp.setHighpass(900.0f, 0.707f, sr);
    bp.setBandpass(3100.0f, 0.9f, sr);
    int clickLen = int(0.012f * float(sr));
    for (int i = 0; i < clickLen; ++i) {
        float t = float(i) / float(sr);
        float x = hp.process(whiteNoise(rng));
        x = x * 0.6f + bp.process(x) * 1.6f;
        out[size_t(i)] += x * expDecay(t, 0.0035f);
    }
    addNoiseBurst(out, rng, sr, 0.0004f, 0.028f, 0.0002f, 1500.0f, 12000.0f, 4200.0f, 0.8f, 0.5f);
    // Firing-pin channel / barrel-hood ring: short, inharmonic, with a clear
    // high-mode ping that separates it from SafetyClick.
    static const ModeSpec modes[6] = {{1.000f, 1.00f, 0.075f, 0.007f},  {2.147f, 0.72f, 0.052f, 0.009f},
                                      {3.591f, 0.50f, 0.034f, 0.012f},  {5.237f, 0.34f, 0.022f, 0.015f},
                                      {7.683f, 0.22f, 0.014f, 0.019f},  {10.417f, 0.13f, 0.009f, 0.024f}};
    addModalBank(out, sr, 1150.0f, modes, 6, 0.42f, 1.0f, 1.0f, 0);
    applyDcBlock(out);
    normalizeBuffer(out, 0.93f);
    trimTail(out, 2e-4f, int(0.03f * float(sr)));
    return out;
}

std::vector<float> synthSafetyClick(int sr) {
    // Small, tight, higher pitched and shorter than DryFire.
    std::vector<float> out = synthMetalClick(sr, 0.085f, 2450.0f, 0.55f, 2600.0f, 16000.0f, 0.0075f,
                                             0.85f, 0.35f, 0x5AFE7001u);
    trimTail(out, 2e-4f, int(0.012f * float(sr)));
    return out;
}

std::vector<float> synthMagRelease(int sr) {
    // Lower, softer and slightly longer than the safety: a bigger button with
    // more spring behind it.
    Rng rng(0x4A9BE002u);
    const int n = int(0.14f * float(sr));
    std::vector<float> out(size_t(n), 0.0f);
    addNoiseBurst(out, rng, sr, 0.0f, 0.011f, 0.0003f, 700.0f, 7000.0f, 0.0f, 1.0f, 0.9f);
    static const ModeSpec modes[5] = {{1.0f, 1.0f, 0.06f, 0.006f},
                                      {2.29f, 0.6f, 0.04f, 0.008f},
                                      {3.94f, 0.34f, 0.025f, 0.011f},
                                      {6.41f, 0.19f, 0.016f, 0.014f},
                                      {9.77f, 0.10f, 0.010f, 0.018f}};
    addModalBank(out, sr, 1080.0f, modes, 5, 0.45f, 1.0f, 1.0f, 0);
    addTone(out, sr, 0.012f, 3300.0f, 0.012f, 0.0004f, 0.10f, 0.02f);  // tiny spring snick
    applyDcBlock(out);
    normalizeBuffer(out, 0.90f);
    trimTail(out, 2e-4f, int(0.02f * float(sr)));
    return out;
}

std::vector<float> synthMagOut(int sr) {
    Rng rng(0x3A17C003u);
    const int n = int(0.46f * float(sr));
    std::vector<float> out(size_t(n), 0.0f);
    // The catch releases (MagRelease's gesture continues into the slide-out).
    addNoiseBurst(out, rng, sr, 0.0f, 0.010f, 0.0003f, 800.0f, 8000.0f, 0.0f, 1.0f, 0.75f);
    // Magazine sliding out of the magwell: metal-on-metal friction whose
    // brightness and level rise as it clears and starts to fall free.
    Biquad bp, hp;
    bp.setBandpass(2100.0f, 1.3f, sr);
    hp.setHighpass(900.0f, 0.707f, sr);
    int slideLen = std::min(int(0.24f * float(sr)), n);
    for (int i = 0; i < slideLen; ++i) {
        float t = float(i) / float(sr);
        float u = float(i) / float(slideLen);
        float env = 0.35f + 0.65f * std::sin(PI * clamp(u * 0.85f + 0.1f, 0.0f, 1.0f));
        float x = hp.process(whiteNoise(rng));
        x = bp.process(x) * 2.4f;
        // Stick-slip: two incommensurate rates read as sliding metal rather than
        // as filtered hiss.
        float mod = 0.55f + 0.45f * std::sin(TWO_PI * (17.0f + 30.0f * u) * t) *
                               std::sin(TWO_PI * 3.1f * t + 0.7f);
        out[size_t(i)] += x * env * mod * 0.55f;
    }
    // It clears the well and clacks lightly against the frame.
    addNoiseBurst(out, rng, sr, 0.235f, 0.020f, 0.0004f, 500.0f, 9000.0f, 1800.0f, 0.8f, 0.55f);
    static const ModeSpec modes[5] = {{1.0f, 1.0f, 0.11f, 0.006f},
                                      {2.61f, 0.62f, 0.07f, 0.008f},
                                      {4.13f, 0.40f, 0.045f, 0.011f},
                                      {6.87f, 0.24f, 0.028f, 0.014f},
                                      {10.29f, 0.13f, 0.018f, 0.018f}};
    addModalBank(out, sr, 1650.0f, modes, 5, 0.30f, 1.0f, 1.0f, int(0.235f * float(sr)));
    // Light clatter as it tumbles away.
    for (int k = 0; k < 3; ++k) {
        float when = 0.285f + float(k) * rng.nextRange(0.035f, 0.065f);
        addNoiseBurst(out, rng, sr, when, 0.006f + 0.004f * float(k), 0.0003f, 1200.0f, 12000.0f, 0.0f,
                      1.0f, 0.22f / (1.0f + 0.6f * float(k)));
    }
    applyDcBlock(out);
    normalizeBuffer(out, 0.92f);
    return out;
}

std::vector<float> synthMagIn(int sr) {
    Rng rng(0x2B6D4004u);
    const int n = int(0.36f * float(sr));
    std::vector<float> out(size_t(n), 0.0f);
    // (1) Hollow "thunk" as the magwell swallows the magazine body.
    addSineSweep(out, sr, 0.0f, 240.0f, 86.0f, 0.03f, 0.055f, 0.0018f, 0.85f);
    addNoiseBurst(out, rng, sr, 0.0f, 0.045f, 0.0015f, 90.0f, 1200.0f, 300.0f, 1.1f, 0.75f);
    // Guide rails rattling against the magazine as it travels.
    for (int k = 0; k < 4; ++k) {
        float when = 0.035f + float(k) * 0.021f;
        addNoiseBurst(out, rng, sr, when, 0.004f, 0.0002f, 2200.0f, 14000.0f, 4200.0f, 1.2f,
                      0.16f / (1.0f + 0.35f * float(k)));
    }
    // (2) Crisp metallic seat click as the catch snaps into the notch.
    const float seat = 0.145f;
    addNoiseBurst(out, rng, sr, seat, 0.009f, 0.0002f, 1500.0f, 16000.0f, 0.0f, 1.0f, 0.95f);
    static const ModeSpec modes[5] = {{1.0f, 1.0f, 0.10f, 0.007f},
                                      {2.43f, 0.68f, 0.066f, 0.009f},
                                      {3.87f, 0.44f, 0.042f, 0.012f},
                                      {6.31f, 0.26f, 0.026f, 0.015f},
                                      {9.53f, 0.14f, 0.016f, 0.019f}};
    addModalBank(out, sr, 1420.0f, modes, 5, 0.42f, 1.0f, 1.0f, int(seat * float(sr)));
    // The polymer frame rings faintly right after seating.
    addTone(out, sr, seat + 0.002f, 430.0f, 0.07f, 0.001f, 0.16f, 0.004f);
    applyDcBlock(out);
    normalizeBuffer(out, 0.95f);
    return out;
}

std::vector<float> synthSlideBack(int sr) {
    Rng rng(0x1C0DE005u);
    const int n = int(0.40f * float(sr));
    std::vector<float> out(size_t(n), 0.0f);
    const float travel = 0.245f;
    const int travelLen = std::min(int(travel * float(sr)), n);
    // Rising-tension scrape: the band-pass tracks upward as the recoil spring
    // compresses and the rails load up.
    Svf svf;
    const int steps = 64;
    for (int i = 0; i < travelLen; ++i) {
        float t = float(i) / float(sr);
        float u = float(i) / float(travelLen);
        float env = std::pow(clamp(u * 1.25f, 0.0f, 1.0f), 1.4f) * (0.35f + 0.65f * u);
        int band = int(u * float(steps - 1));
        float fc = lerpf(900.0f, 4200.0f, float(band) / float(steps - 1));
        float x = whiteNoise(rng);
        svf.process(x, fc, 1.6f, sr);
        float mod = 0.6f + 0.4f * std::sin(TWO_PI * (23.0f + 90.0f * u) * t);
        out[size_t(i)] += (svf.bandpass() * 2.6f + svf.highpass(x) * 0.35f) * env * mod * 0.65f;
    }
    // Recoil spring compressing: a light metallic "zzzip" under the scrape.
    for (int k = 0; k < 5; ++k) {
        float when = 0.03f + float(k) * 0.042f;
        addNoiseBurst(out, rng, sr, when, 0.012f, 0.0004f, 3000.0f, 15000.0f, 6200.0f, 2.0f,
                      0.12f / (1.0f + 0.25f * float(k)));
    }
    // Hard stop: the slide hits the rear of its travel.
    const float stop = travel + 0.004f;
    addNoiseBurst(out, rng, sr, stop, 0.014f, 0.0002f, 600.0f, 12000.0f, 1800.0f, 0.9f, 1.0f);
    static const ModeSpec modes[6] = {{1.0f, 1.0f, 0.10f, 0.008f},  {2.19f, 0.70f, 0.066f, 0.010f},
                                      {3.47f, 0.48f, 0.042f, 0.013f}, {5.81f, 0.30f, 0.026f, 0.016f},
                                      {8.41f, 0.18f, 0.016f, 0.020f}, {11.9f, 0.10f, 0.010f, 0.025f}};
    addModalBank(out, sr, 620.0f, modes, 6, 0.5f, 1.0f, 1.0f, int(stop * float(sr)));
    applyDcBlock(out);
    normalizeBuffer(out, 0.95f);
    return out;
}

std::vector<float> synthSlideForward(int sr) {
    Rng rng(0x0BA77006u);
    const int n = int(0.34f * float(sr));
    std::vector<float> out(size_t(n), 0.0f);
    // Very short forward scrape: the slide is already moving fast.
    addScrape(out, rng, sr, 0.0f, 0.045f, 700.0f, 13000.0f, 2600.0f, 1.5f, 0.006f, 0.012f, 0.32f, 40.0f,
              0.4f);
    // Slam into battery: heavy, bright, hard metal-on-metal.
    const float hit = 0.042f;
    addNoiseBurst(out, rng, sr, hit, 0.013f, 0.00012f, 400.0f, 19000.0f, 0.0f, 1.0f, 1.15f);
    addNoiseBurst(out, rng, sr, hit, 0.030f, 0.0003f, 1500.0f, 16000.0f, 3400.0f, 0.9f, 0.7f);
    addSineSweep(out, sr, hit, 320.0f, 120.0f, 0.02f, 0.038f, 0.0006f, 0.75f);
    static const ModeSpec modes[6] = {{1.0f, 1.0f, 0.17f, 0.007f},  {2.08f, 0.74f, 0.115f, 0.009f},
                                      {3.29f, 0.52f, 0.075f, 0.012f}, {4.93f, 0.35f, 0.048f, 0.014f},
                                      {7.37f, 0.21f, 0.030f, 0.018f}, {10.61f, 0.12f, 0.019f, 0.022f}};
    addModalBank(out, sr, 880.0f, modes, 6, 0.62f, 1.0f, 1.0f, int(hit * float(sr)));
    addTone(out, sr, hit + 0.006f, 2150.0f, 0.075f, 0.002f, 0.09f, 0.010f);  // recoil spring ring
    applyDcBlock(out);
    normalizeBuffer(out, 0.96f);
    return out;
}

// One brass casing impact (shared by ShellDrop and ShellBounce).
void addCasingHit(std::vector<float>& out, Rng& rng, int sr, float start, float gain, float pitch) {
    int onset = int(start * float(sr));
    if (onset < 0 || onset >= int(out.size())) return;
    addNoiseBurst(out, rng, sr, start, 0.0035f, 0.00015f, 2500.0f, 19000.0f, 0.0f, 1.0f, gain * 0.9f);
    static const ModeSpec modes[5] = {{1.0f, 1.0f, 0.085f, 0.008f},
                                      {2.67f, 0.60f, 0.055f, 0.010f},
                                      {4.41f, 0.38f, 0.034f, 0.013f},
                                      {6.93f, 0.22f, 0.021f, 0.017f},
                                      {10.87f, 0.12f, 0.013f, 0.022f}};
    addModalBank(out, sr, 2750.0f * pitch, modes, 5, gain * 0.55f, 1.0f, 1.0f, onset);
    addTone(out, sr, start, 5850.0f * pitch, 0.020f, 0.0002f, gain * 0.16f, 0.006f);
}

std::vector<float> synthShellDrop(int sr) {
    Rng rng(0x7E550007u);
    const int n = int(1.05f * float(sr));
    std::vector<float> out(size_t(n), 0.0f);
    addCasingHit(out, rng, sr, 0.0f, 1.0f, 1.0f);  // primary landing
    addScrape(out, rng, sr, 0.02f, 0.20f, 1400.0f, 15000.0f, 4300.0f, 2.2f, 0.02f, 0.05f, 0.055f, 30.0f,
              0.7f);
    // Two secondary bounces: quieter and brighter (they land on an edge).
    addCasingHit(out, rng, sr, 0.215f, 0.42f, 1.21f);
    addCasingHit(out, rng, sr, 0.408f, 0.19f, 1.44f);
    addScrape(out, rng, sr, 0.24f, 0.34f, 1800.0f, 16000.0f, 5200.0f, 2.4f, 0.02f, 0.08f, 0.035f, 42.0f,
              0.8f);
    // Final settle: the casing rocks to rest on the floor.
    addCasingHit(out, rng, sr, 0.63f, 0.075f, 1.62f);
    addScrape(out, rng, sr, 0.64f, 0.24f, 2200.0f, 16000.0f, 5600.0f, 2.6f, 0.03f, 0.14f, 0.020f, 55.0f,
              0.9f);
    applyDcBlock(out);
    normalizeBuffer(out, 0.88f);
    return out;
}

std::vector<float> synthShellBounce(int sr) {
    // A single, small, bright secondary bounce with a hint of spin-down.
    Rng rng(0x7E550008u);
    const int n = int(0.30f * float(sr));
    std::vector<float> out(size_t(n), 0.0f);
    addCasingHit(out, rng, sr, 0.0f, 0.72f, 1.34f);
    addScrape(out, rng, sr, 0.008f, 0.16f, 2000.0f, 16000.0f, 5400.0f, 2.4f, 0.01f, 0.05f, 0.05f, 48.0f,
              0.85f);
    addCasingHit(out, rng, sr, 0.145f, 0.22f, 1.58f);
    applyDcBlock(out);
    normalizeBuffer(out, 0.82f);
    return out;
}

std::vector<float> synthGlassShatter(int sr) {
    Rng rng(0x61A55009u);
    const int n = int(1.30f * float(sr));
    std::vector<float> out(size_t(n), 0.0f);
    // (1) Initial sharp crack: extremely bright, very short.
    addNoiseBurst(out, rng, sr, 0.0f, 0.0018f, 0.00006f, 5000.0f, 19500.0f, 9000.0f, 0.6f, 1.0f);
    addNoiseBurst(out, rng, sr, 0.0002f, 0.020f, 0.0002f, 2200.0f, 19000.0f, 0.0f, 1.0f, 0.85f);
    // (2) The vessel letting go: broadband burst over ~60-90 ms.
    addNoiseBurst(out, rng, sr, 0.0008f, 0.055f, 0.0006f, 900.0f, 18000.0f, 3200.0f, 0.7f, 0.60f);
    addNoiseBurst(out, rng, sr, 0.002f, 0.090f, 0.0012f, 300.0f, 9000.0f, 1500.0f, 0.8f, 0.30f);
    // Modal ring of the large remaining pieces, right at the break.
    static const ModeSpec bigModes[6] = {{1.0f, 1.0f, 0.24f, 0.004f},  {2.31f, 0.60f, 0.16f, 0.006f},
                                         {3.72f, 0.42f, 0.11f, 0.008f}, {5.44f, 0.28f, 0.07f, 0.011f},
                                         {7.91f, 0.17f, 0.045f, 0.014f}, {11.03f, 0.10f, 0.03f, 0.018f}};
    addModalBank(out, sr, 1980.0f, bigModes, 6, 0.22f, 1.0f, 1.0f, int(0.001f * float(sr)));

    // (3) Stochastic shower of individually synthesised shards.  Deterministic
    //     (seeded) onsets, pitches, decays and pan positions; the onset
    //     distribution is fast-then-slow, as a real vessel lets go.
    const int shardCount = 58;
    for (int k = 0; k < shardCount; ++k) {
        float u = rng.nextUnit();
        float when = std::pow(u, 1.85f) * 0.62f + rng.nextRange(0.0f, 0.02f);
        float size = std::pow(1.0f - u, 2.0f);  // big pieces land first
        float f0 = rng.nextLogRange(1500.0f, 7800.0f) * (1.0f - 0.35f * size);
        float t60 = rng.nextRange(0.035f, 0.075f) + 0.16f * size * size;
        float gain = rng.nextRange(0.16f, 0.45f) * (0.45f + 0.95f * size);
        float pan = clamp(rng.nextGaussian() * 0.55f, -1.0f, 1.0f);
        addShard(out, rng, sr, when, f0, t60, rng.nextRange(0.85f, 1.25f), gain, pan);
    }
    // A few larger late pieces (the base and heavier fragments falling).
    for (int k = 0; k < 4; ++k) {
        float when = 0.60f + float(k) * rng.nextRange(0.09f, 0.14f);
        addShard(out, rng, sr, when, rng.nextLogRange(900.0f, 1900.0f), rng.nextRange(0.15f, 0.30f),
                 rng.nextRange(0.7f, 1.0f), rng.nextRange(0.10f, 0.22f),
                 clamp(rng.nextGaussian() * 0.7f, -1.0f, 1.0f));
    }
    // (4) Fragments skittering across the table and floor: a dense stream of
    //     tiny high-frequency grains with irregular, thinning gaps.
    float t = 0.05f;
    while (t < 0.95f) {
        float rate = lerpf(1.0f, 0.25f, clamp(t / 0.95f, 0.0f, 1.0f));
        if (rng.chance(0.72f * rate)) {
            addScrape(out, rng, sr, t, rng.nextRange(0.006f, 0.022f), 2600.0f, 19000.0f,
                      rng.nextRange(4200.0f, 9000.0f), 2.6f, 0.0003f, 0.004f,
                      rng.nextRange(0.05f, 0.16f) * rate, rng.nextRange(60.0f, 220.0f), 0.7f);
        }
        if (rng.chance(0.22f)) {  // isolated tick of a fragment settling
            addShard(out, rng, sr, t, rng.nextLogRange(2200.0f, 8200.0f), rng.nextRange(0.01f, 0.03f),
                     1.1f, rng.nextRange(0.03f, 0.10f), clamp(rng.nextGaussian(), -1.0f, 1.0f));
        }
        t += rng.nextRange(0.006f, 0.026f);
    }
    // (5) Floor impacts of the big falling fragments: duller and lower, the only
    //     part of the event that is not bright.
    for (int k = 0; k < 5; ++k) {
        float when = rng.nextRange(0.42f, 0.92f);
        addShard(out, rng, sr, when, rng.nextLogRange(600.0f, 1400.0f), rng.nextRange(0.06f, 0.13f),
                 0.75f, rng.nextRange(0.08f, 0.18f), clamp(rng.nextGaussian() * 0.8f, -1.0f, 1.0f));
    }
    applyDcBlock(out);
    normalizeBuffer(out, 0.96f);
    return out;
}

std::vector<float> synthGlassTinkle(int sr) {
    // One small shard landing: bright, fast, inharmonically shimmering through
    // beat-detuned partials with independent decay rates.
    Rng rng(0x71A4C00Au);
    const int n = int(0.34f * float(sr));
    std::vector<float> out(size_t(n), 0.0f);
    addNoiseBurst(out, rng, sr, 0.0f, 0.0022f, 0.00008f, 4200.0f, 19500.0f, 0.0f, 1.0f, 0.55f);
    static const ModeSpec modes[6] = {{1.0f, 1.0f, 0.20f, 0.0032f},  {2.44f, 0.64f, 0.13f, 0.0041f},
                                      {3.86f, 0.45f, 0.085f, 0.0053f}, {5.72f, 0.30f, 0.055f, 0.0069f},
                                      {8.29f, 0.18f, 0.034f, 0.0088f}, {12.13f, 0.10f, 0.022f, 0.0113f}};
    addModalBank(out, sr, 4180.0f, modes, 6, 0.55f, 1.0f, 1.0f, 0);
    applyDcBlock(out);
    normalizeBuffer(out, 0.90f);
    return out;
}

std::vector<float> synthGlassStress(int sr) {
    // A very short high-frequency crackle: several tiny brittle-failure grains
    // inside a ~40 ms window, bright and narrow-band.
    Rng rng(0x51A5500Bu);
    const int n = int(0.085f * float(sr));
    std::vector<float> out(size_t(n), 0.0f);
    const int grains = 14;
    for (int k = 0; k < grains; ++k) {
        float when = float(k) * rng.nextRange(0.0008f, 0.0034f);
        addNoiseBurst(out, rng, sr, when, rng.nextRange(0.0007f, 0.0022f), 0.00004f,
                      rng.nextRange(4000.0f, 6500.0f), 19500.0f, rng.nextRange(6500.0f, 11000.0f), 2.2f,
                      rng.nextRange(0.35f, 0.9f));
    }
    static const ModeSpec modes[4] = {{1.0f, 1.0f, 0.030f, 0.004f},
                                      {2.57f, 0.5f, 0.019f, 0.006f},
                                      {4.83f, 0.3f, 0.012f, 0.008f},
                                      {8.11f, 0.16f, 0.008f, 0.011f}};
    addModalBank(out, sr, 6100.0f, modes, 4, 0.22f, 1.0f, 1.0f, 0);
    applyDcBlock(out);
    normalizeBuffer(out, 0.85f);
    trimTail(out, 2e-4f, int(0.012f * float(sr)));
    return out;
}

std::vector<float> synthImpactConcrete(int sr) {
    // Gritty, dusty "spat": sharp transient, granular mid band, dust tail.
    Rng rng(0x1C0C0C0Cu);
    const int n = int(0.50f * float(sr));
    std::vector<float> out(size_t(n), 0.0f);
    addNoiseBurst(out, rng, sr, 0.0f, 0.0012f, 0.00006f, 3000.0f, 19000.0f, 0.0f, 1.0f, 1.0f);
    addNoiseBurst(out, rng, sr, 0.0f, 0.045f, 0.0004f, 600.0f, 12000.0f, 1800.0f, 0.8f, 0.85f);
    // Grit: individually placed grains (aggregate tearing out of the wall).
    for (int k = 0; k < 26; ++k) {
        float when = rng.nextRange(0.0f, 0.075f);
        addNoiseBurst(out, rng, sr, when, rng.nextRange(0.0014f, 0.006f), 0.00008f,
                      rng.nextRange(700.0f, 2400.0f), 16000.0f, rng.nextRange(1400.0f, 3600.0f), 1.6f,
                      rng.nextRange(0.10f, 0.35f));
    }
    // Low-frequency punch into the wall (the round dumping its energy).
    addSineSweep(out, sr, 0.0f, 200.0f, 90.0f, 0.02f, 0.045f, 0.0006f, 0.8f);
    // Dust / debris settling: a diffuse low-passed noise tail.
    Biquad lp;
    lp.setLowpass(2200.0f, 0.7f, sr);
    const int dustStart = int(0.03f * float(sr));
    for (int i = dustStart; i < n; ++i) {
        float t = float(i - dustStart) / float(sr);
        out[size_t(i)] += lp.process(whiteNoise(rng)) * expDecayA(t, 0.004f, 0.16f) * 0.22f;
    }
    applyDcBlock(out);
    normalizeBuffer(out, 0.93f);
    return out;
}

std::vector<float> synthImpactWood(int sr) {
    // Drier, hollow, lower: a tabletop takes the round with a woody "tok".
    Rng rng(0x0D0D000Du);
    const int n = int(0.38f * float(sr));
    std::vector<float> out(size_t(n), 0.0f);
    addNoiseBurst(out, rng, sr, 0.0f, 0.0022f, 0.0001f, 1800.0f, 16000.0f, 3600.0f, 1.1f, 0.8f);
    addNoiseBurst(out, rng, sr, 0.0f, 0.020f, 0.0004f, 300.0f, 6000.0f, 900.0f, 1.0f, 0.55f);
    // Hollow body resonance of the tabletop: few partials, strong fundamental and
    // an audible but quick wooden pitch.
    static const ModeSpec modes[5] = {{1.0f, 1.0f, 0.13f, 0.005f},
                                      {2.76f, 0.55f, 0.075f, 0.007f},
                                      {4.19f, 0.34f, 0.045f, 0.010f},
                                      {6.64f, 0.19f, 0.026f, 0.013f},
                                      {9.37f, 0.10f, 0.015f, 0.017f}};
    addModalBank(out, sr, 320.0f, modes, 5, 0.85f, 1.0f, 1.0f, 0);
    addTone(out, sr, 0.0f, 168.0f, 0.11f, 0.0008f, 0.35f, 0.006f);
    // Splinters: a few dry high ticks in the first 50 ms.
    for (int k = 0; k < 10; ++k) {
        float when = rng.nextRange(0.0f, 0.05f);
        addNoiseBurst(out, rng, sr, when, rng.nextRange(0.001f, 0.004f), 0.00008f, 1600.0f, 15000.0f,
                      rng.nextRange(2200.0f, 5200.0f), 1.8f, rng.nextRange(0.08f, 0.26f));
    }
    // Short dry debris tail (much less dust than concrete).
    Biquad lp;
    lp.setLowpass(1100.0f, 0.7f, sr);
    const int tailStart = int(0.02f * float(sr));
    for (int i = tailStart; i < n; ++i) {
        float t = float(i - tailStart) / float(sr);
        out[size_t(i)] += lp.process(whiteNoise(rng)) * expDecayA(t, 0.003f, 0.09f) * 0.18f;
    }
    applyDcBlock(out);
    normalizeBuffer(out, 0.92f);
    return out;
}

std::vector<float> synthImpactMetal(int sr) {
    // Bright ringing with a long decay and strong inharmonic partials.
    Rng rng(0x17E7A10Du);
    const int n = int(0.85f * float(sr));
    std::vector<float> out(size_t(n), 0.0f);
    addNoiseBurst(out, rng, sr, 0.0f, 0.0009f, 0.00004f, 4000.0f, 19500.0f, 0.0f, 1.0f, 1.0f);
    addNoiseBurst(out, rng, sr, 0.0f, 0.030f, 0.0003f, 2000.0f, 18000.0f, 6000.0f, 0.9f, 0.6f);
    // Inharmonic sheet-metal modes: non-integer ratios, independent decays, and
    // one deliberately strong partial at 3.83x for the characteristic clang.
    static const ModeSpec modes[6] = {{1.0f, 0.85f, 0.42f, 0.0021f}, {1.87f, 0.60f, 0.30f, 0.0028f},
                                      {3.83f, 1.00f, 0.52f, 0.0034f}, {5.19f, 0.55f, 0.26f, 0.0044f},
                                      {7.41f, 0.36f, 0.17f, 0.0059f}, {11.29f, 0.20f, 0.10f, 0.0077f}};
    addModalBank(out, sr, 1180.0f, modes, 6, 0.5f, 1.0f, 1.0f, 0);
    addTone(out, sr, 0.0f, 232.0f, 0.24f, 0.0006f, 0.20f, 0.004f);  // panel "bong"
    applyDcBlock(out);
    normalizeBuffer(out, 0.94f);
    return out;
}

std::vector<float> synthFootstep(int sr, bool alt) {
    // Shoe on a hard floor: broadband scuff + light low thud.  The two variants
    // differ in level, timing, spectral tilt and grain texture so alternating
    // them does not sound like a repeated sample.
    Rng rng(alt ? 0xF0075711u : 0xF0075710u);
    const int n = int((alt ? 0.26f : 0.22f) * float(sr));
    std::vector<float> out(size_t(n), 0.0f);
    const float heel = alt ? 0.006f : 0.0f;  // the alt foot lands slightly later
    const float level = alt ? 0.88f : 1.0f;
    addScrape(out, rng, sr, heel, 0.045f, alt ? 700.0f : 500.0f, alt ? 14000.0f : 11000.0f,
              alt ? 2600.0f : 1900.0f, alt ? 1.1f : 1.5f, 0.0006f, 0.012f, 0.55f * level,
              alt ? 90.0f : 60.0f, 0.5f);
    // Body weight: a soft low thud (not a click) plus a hint of sole squeak.
    addNoiseBurst(out, rng, sr, heel + 0.0008f, alt ? 0.035f : 0.045f, 0.0022f, 60.0f, 420.0f, 150.0f,
                  1.0f, (alt ? 0.55f : 0.72f) * level);
    addSineSweep(out, sr, heel + 0.0008f, alt ? 96.0f : 84.0f, alt ? 58.0f : 52.0f, 0.02f, 0.032f,
                 0.0025f, (alt ? 0.40f : 0.52f) * level);
    // Sole/frame settling micro-grains in the first 30 ms.
    const int grains = alt ? 5 : 8;
    for (int k = 0; k < grains; ++k) {
        float when = heel + rng.nextRange(0.001f, 0.030f);
        addNoiseBurst(out, rng, sr, when, rng.nextRange(0.0015f, 0.005f), 0.0001f,
                      rng.nextRange(900.0f, 2600.0f), 15000.0f, rng.nextRange(2000.0f, 5200.0f), 1.4f,
                      rng.nextRange(0.05f, 0.20f) * level);
    }
    // A whisper of room so a footstep does not sound anechoic.
    std::vector<float> tail = renderRoomTail(out, sr, 0.55f, 0.004f, 0.28f, 0.30f);
    for (size_t i = 0; i < out.size(); ++i) out[i] = out[i] * 0.94f + tail[i] * 0.30f;
    applyDcBlock(out);
    normalizeBuffer(out, alt ? 0.86f : 0.92f);
    return out;
}

std::vector<float> synthWeaponMove(int sr, bool draw) {
    // Cloth/leather movement plus polymer-on-metal contact.  "Draw" pulls the
    // pistol up and out (rising friction, ending in the grip slap); "holster"
    // pushes it back down (falling friction, softer seating).
    Rng rng(draw ? 0x0D0A0001u : 0x0D0A0002u);
    const int n = int((draw ? 0.55f : 0.48f) * float(sr));
    std::vector<float> out(size_t(n), 0.0f);
    const float dur = draw ? 0.30f : 0.26f;
    // Leather creak: a narrow, frequency-wandering resonance with stick-slip.
    Svf svf;
    const int creakLen = std::min(int(dur * float(sr)), n);
    for (int i = 0; i < creakLen; ++i) {
        float t = float(i) / float(sr);
        float u = float(i) / float(creakLen);
        float env = std::sin(PI * clamp(u * 0.9f + 0.05f, 0.0f, 1.0f));
        float fc = draw ? lerpf(420.0f, 900.0f, u) : lerpf(880.0f, 400.0f, u);
        svf.process(whiteNoise(rng), fc, 5.5f, sr);
        float mod = 0.35f + 0.65f * std::pow(std::fabs(std::sin(TWO_PI * (7.0f + 13.0f * u) * t)), 1.6f);
        out[size_t(i)] += svf.bandpass() * 3.2f * env * mod * (draw ? 0.30f : 0.24f);
    }
    // Cloth/fabric rustle over the movement.
    addScrape(out, rng, sr, 0.01f, dur * 0.85f, 1200.0f, 12000.0f, 3400.0f, 0.9f, 0.02f, 0.10f,
              draw ? 0.30f : 0.24f, 26.0f, 0.65f);
    addScrape(out, rng, sr, 0.04f, dur * 0.70f, 2200.0f, 16000.0f, 5600.0f, 1.1f, 0.03f, 0.12f,
              draw ? 0.17f : 0.14f, 41.0f, 0.75f);
    // Polymer-on-metal contact: the frame scraping the holster shell...
    addScrape(out, rng, sr, draw ? 0.10f : 0.06f, dur * 0.6f, 1800.0f, 15000.0f, 4600.0f, 2.0f, 0.008f,
              0.05f, draw ? 0.22f : 0.18f, 60.0f, 0.6f);
    // ...then a positive clack as it clears (draw) or seats (holster).
    const float contact = draw ? dur * 0.78f : dur * 0.72f;
    const float clackGain = draw ? 1.0f : 0.72f;
    addNoiseBurst(out, rng, sr, contact, draw ? 0.012f : 0.016f, 0.0002f, 900.0f, 15000.0f, 2600.0f,
                  1.0f, 0.55f * clackGain);
    static const ModeSpec modes[5] = {{1.0f, 1.0f, 0.075f, 0.006f},
                                      {2.37f, 0.6f, 0.048f, 0.008f},
                                      {4.11f, 0.38f, 0.030f, 0.011f},
                                      {6.59f, 0.22f, 0.019f, 0.014f},
                                      {10.03f, 0.12f, 0.012f, 0.018f}};
    addModalBank(out, sr, draw ? 1450.0f : 1150.0f, modes, 5, 0.34f * clackGain, 1.0f, 1.0f,
                 int(contact * float(sr)));
    // Grip slap into the palm (draw only): a small, tight thud.
    if (draw) {
        addNoiseBurst(out, rng, sr, 0.40f, 0.030f, 0.0015f, 80.0f, 2600.0f, 700.0f, 0.9f, 0.42f);
        addSineSweep(out, sr, 0.40f, 210.0f, 110.0f, 0.02f, 0.035f, 0.002f, 0.35f);
    }
    applyDcBlock(out);
    normalizeBuffer(out, draw ? 0.90f : 0.87f);
    return out;
}

std::vector<float> synthCloth(int sr) {
    // Short fabric rustle: granular, high-passed, soft attack, no percussive
    // transient at all.
    Rng rng(0xC1074001u);
    const int n = int(0.28f * float(sr));
    std::vector<float> out(size_t(n), 0.0f);
    for (int k = 0; k < 40; ++k) {
        float when = rng.nextRange(0.0f, 0.20f);
        addNoiseBurst(out, rng, sr, when, rng.nextRange(0.004f, 0.016f), rng.nextRange(0.002f, 0.010f),
                      rng.nextRange(900.0f, 2000.0f), 15000.0f, rng.nextRange(2200.0f, 6500.0f), 1.0f,
                      rng.nextRange(0.10f, 0.32f));
    }
    // Slow swell so the grains read as a single gesture.
    for (int i = 0; i < n; ++i) {
        float u = float(i) / float(std::max(n - 1, 1));
        out[size_t(i)] *= std::sin(PI * clamp(u, 0.0f, 1.0f)) * 1.6f + 0.15f;
    }
    Biquad hp;
    hp.setHighpass(700.0f, 0.707f, sr);
    for (float& v : out) v = hp.process(v);
    applyDcBlock(out);
    normalizeBuffer(out, 0.80f);
    return out;
}

std::vector<float> synthRoomTone(int sr) {
    // Seamless-looping HVAC / electrical bed.  Every periodic component is built
    // from a frequency that completes a whole number of cycles inside the loop,
    // and the noise component is cross-faded head-to-tail, so the buffer wraps
    // with no discontinuity (and therefore no click).
    const float seconds = 4.0f;
    const int len = int(seconds * float(sr));
    const int fade = int(0.30f * float(sr));
    std::vector<float> buf(size_t(len + fade), 0.0f);
    Rng rng(0x4007A0E1u);

    // Mains hum at 50 Hz with harmonics (the electrical part of the bed).
    auto humCycle = [&](float freq, float amp, float phase) {
        int cycles = std::max(int(std::lround(freq * seconds)), 1);
        float f = float(cycles) / seconds;
        for (int i = 0; i < len; ++i) {
            float t = float(i) / float(sr);
            buf[size_t(i)] += amp * std::sin(TWO_PI * f * t + phase);
        }
    };
    humCycle(50.0f, 1.0f, 0.0f);
    humCycle(100.0f, 0.42f, 0.7f);
    humCycle(150.0f, 0.20f, 1.9f);
    humCycle(200.0f, 0.11f, 2.6f);
    humCycle(250.0f, 0.055f, 0.3f);
    // A second, slightly detuned hum source beats slowly against the first (two
    // mains-fed devices); still cycle-exact for the loop length.
    humCycle(50.25f, 0.30f, 1.1f);

    // Noise bed (air handling) with a slow, loop-periodic amplitude modulation.
    //
    // IIR filtering makes a circular signal awkward: a filter carries state, so
    // even a periodic input does not give a periodic output unless the states
    // are already in their steady cycle.  The construction used here runs the
    // filters over TWO loop periods and discards the first, so by the time the
    // kept period starts every state has converged to its periodic value.  The
    // noise bed is then genuinely circular: buf[0] and buf[len] are (to well
    // below the noise floor) the same sample.
    Biquad lp, bp, hp;
    lp.setLowpass(7000.0f, 0.707f, sr);
    bp.setBandpass(120.0f, 1.0f, sr);
    hp.setHighpass(28.0f, 0.707f, sr);
    {
        const int total = len + fade;
        std::vector<float> rumble(static_cast<size_t>(total), 0.0f);
        std::vector<float> air(static_cast<size_t>(total), 0.0f);
        const int warm = total;  // one discarded loop period settles the states
        for (int i = 0; i < warm + total; ++i) {
            float x = lp.process(whiteNoise(rng));
            float r = bp.process(x) * 3.0f;
            float a = hp.process(x) * 0.9f;
            if (i >= warm) {
                rumble[size_t(i - warm)] = r;
                air[size_t(i - warm)] = a;
            }
        }
        const int modCycles = 3;  // 0.75 Hz, cycle-exact for the loop length
        for (int i = 0; i < total; ++i) {
            float t = float(i) / float(sr);
            float m = 0.72f + 0.28f * std::sin(TWO_PI * float(modCycles) / seconds * t + 0.4f);
            buf[size_t(i)] += (rumble[size_t(i)] * 1.1f + air[size_t(i)] * 0.30f) * m;
        }
    }

    // Fold the overrun back over the head so the buffer loops seamlessly.
    //
    // The blend fades the *tail* in and the *head* out (not the reverse):
    //   buf[0]    == tail[len]  -> continuous with buf[len-1] == tail[len-1]
    //   buf[fade] == head[fade] -> continuous with the untouched region
    for (int i = 0; i < fade; ++i) {
        float w = fadeIn(i, fade);  // 1 at the seam, 0 at the end of the fade
        buf[size_t(i)] = buf[size_t(i)] * (1.0f - w) + buf[size_t(len + i)] * w;
    }
    buf.resize(size_t(len));
    // Remove any residual offset by subtracting the exact mean.  A DC blocker
    // (an IIR with a cold start) would itself introduce a transient at the loop
    // point, which is precisely what must not happen here.
    double mean = 0.0;
    for (float v : buf) mean += double(v);
    mean /= double(len);
    for (float& v : buf) v -= float(mean);
    normalizeBuffer(buf, 0.55f);  // ambient bed: present but never dominant
    return buf;
}

std::vector<float> synthEarRing(int sr) {
    // Post-shot tinnitus: a pure high sine with a very slow decay and a gentle
    // beating detune that makes it feel like it is inside the head.
    const float dur = 3.0f;
    const int n = int(dur * float(sr));
    std::vector<float> out(size_t(n), 0.0f);
    // 4.4 kHz beating at ~3.5 Hz, plus quieter 6.2 / 3.3 kHz partials so it never
    // reads as a single oscillator.
    addTone(out, sr, 0.0f, 4400.0f, 2.10f, 0.010f, 0.90f, 0.0008f);
    addTone(out, sr, 0.0f, 6200.0f, 1.55f, 0.014f, 0.32f, 0.0011f);
    addTone(out, sr, 0.0f, 3300.0f, 2.30f, 0.012f, 0.16f, 0.0014f);
    // Fade the very end to zero so the sound never ends on a step.
    const int fadeN = std::min(int(0.06f * float(sr)), n);
    for (int i = 0; i < fadeN; ++i) out[size_t(n - 1 - i)] *= fadeOut(i, fadeN);
    applyDcBlock(out);
    normalizeBuffer(out, 0.78f);
    return out;
}

std::vector<float> synthUiBeep(int sr) {
    // Clean short sine blip with a click-free raised-cosine envelope.
    const float dur = 0.13f;
    const int n = int(dur * float(sr));
    std::vector<float> out(size_t(n), 0.0f);
    addTone(out, sr, 0.0f, 880.0f, 0.055f, 0.0f, 1.0f, 0.0f);
    addTone(out, sr, 0.0f, 1760.0f, 0.030f, 0.0f, 0.22f, 0.0f);
    // Raised-cosine window over the whole buffer: starts and ends at exactly 0.
    const int attackN = std::max(n / 10, 1);
    const int releaseN = std::max(int(0.055f * float(sr)), 1);
    for (int i = 0; i < n; ++i) {
        float w = clamp(float(i) / float(attackN), 0.0f, 1.0f) *
                  clamp(float(n - 1 - i) / float(releaseN), 0.0f, 1.0f);
        out[size_t(i)] *= w * w * (3.0f - 2.0f * w);
    }
    applyDcBlock(out);
    normalizeBuffer(out, 0.72f);
    return out;
}

// ===========================================================================
// 10. the bank
// ===========================================================================

struct SoundEntry {
    std::vector<float> samples;
    float defaultReverbSend = 0.25f;
    bool looping = false;
};

// Per-sound default reverb send: how much of this sound feeds the room bus.
// Ambient / UI material stays dry; the gunshot and glass drive the room hard.
const float kDefaultSend[kSoundCount] = {
    0.35f,  // Gunshot
    0.20f,  // GunshotTail
    0.18f,  // DryFire
    0.12f,  // SafetyClick
    0.12f,  // MagRelease
    0.16f,  // MagOut
    0.18f,  // MagIn
    0.16f,  // SlideBack
    0.18f,  // SlideForward
    0.22f,  // ShellDrop
    0.22f,  // ShellBounce
    0.30f,  // GlassShatter
    0.26f,  // GlassTinkle
    0.14f,  // GlassStress
    0.26f,  // ImpactConcrete
    0.22f,  // ImpactWood
    0.28f,  // ImpactMetal
    0.20f,  // Footstep
    0.20f,  // FootstepAlt
    0.16f,  // WeaponDraw
    0.16f,  // WeaponHolster
    0.14f,  // Cloth
    0.00f,  // RoomTone
    0.00f,  // EarRing
    0.00f,  // UiBeep
};

std::vector<SoundEntry> buildBank(int sr) {
    std::vector<SoundEntry> bank(kSoundCount);
    auto set = [&](SoundId id, std::vector<float> s) {
        SoundEntry& e = bank[size_t(id)];
        e.samples = std::move(s);
        e.defaultReverbSend = kDefaultSend[size_t(id)];
        e.looping = (id == SoundId::RoomTone);
    };
    set(SoundId::Gunshot, synthGunshot(sr, false));
    set(SoundId::GunshotTail, synthGunshot(sr, true));
    set(SoundId::DryFire, synthDryFire(sr));
    set(SoundId::SafetyClick, synthSafetyClick(sr));
    set(SoundId::MagRelease, synthMagRelease(sr));
    set(SoundId::MagOut, synthMagOut(sr));
    set(SoundId::MagIn, synthMagIn(sr));
    set(SoundId::SlideBack, synthSlideBack(sr));
    set(SoundId::SlideForward, synthSlideForward(sr));
    set(SoundId::ShellDrop, synthShellDrop(sr));
    set(SoundId::ShellBounce, synthShellBounce(sr));
    set(SoundId::GlassShatter, synthGlassShatter(sr));
    set(SoundId::GlassTinkle, synthGlassTinkle(sr));
    set(SoundId::GlassStress, synthGlassStress(sr));
    set(SoundId::ImpactConcrete, synthImpactConcrete(sr));
    set(SoundId::ImpactWood, synthImpactWood(sr));
    set(SoundId::ImpactMetal, synthImpactMetal(sr));
    set(SoundId::Footstep, synthFootstep(sr, false));
    set(SoundId::FootstepAlt, synthFootstep(sr, true));
    set(SoundId::WeaponDraw, synthWeaponMove(sr, true));
    set(SoundId::WeaponHolster, synthWeaponMove(sr, false));
    set(SoundId::Cloth, synthCloth(sr));
    set(SoundId::RoomTone, synthRoomTone(sr));
    set(SoundId::EarRing, synthEarRing(sr));
    set(SoundId::UiBeep, synthUiBeep(sr));
    return bank;
}

// Bank used by the free function renderSoundToBuffer() (kept separate from the
// engine's own copy so the helper works with or without a live Engine).  The
// function-local static is initialised exactly once, thread-safely.
// Lazily built bank backing renderSoundToBuffer().  It is deliberately kept out
// of the Engine so the helper works with no Engine alive; an explicit teardown
// is registered so the memory is released at exit (rather than being reported as
// a leak) and so no other static can observe a half-destroyed bank.
std::vector<SoundEntry>& sharedBankStorage() {
    static std::vector<SoundEntry>* bank = new std::vector<SoundEntry>(buildBank(48000));
    return *bank;
}

const std::vector<SoundEntry>& sharedBank() { return sharedBankStorage(); }

struct SharedBankTeardown {
    ~SharedBankTeardown() {
        delete &sharedBankStorage();
    }
};
const SharedBankTeardown g_sharedBankTeardown{};

}  // namespace   (end of the synthesis toolkit / sound-bank helpers)

// ===========================================================================
// 11. lock-free SPSC command ring
// ===========================================================================

enum class CmdType : uint32_t { Play, Stop, StopAll, Listener, MasterVolume };

struct Command {
    CmdType type = CmdType::Play;
    uint32_t id = 0;   // voice handle (rendered by the producer)
    uint32_t gen = 0;  // matches Impl::generation; stale commands are dropped
    SoundId sound = SoundId::Count;
    PlayParams params;
    Vec3 position{};
    // Fully pre-computed spatial terms (the game thread owns the listener; the
    // audio thread must never read listener state directly).
    float gain = 1.0f;
    float pan = 0.0f;
    float itd = 0.0f;
    float airCutoff = 20000.0f;
    float reverbSend = 0.0f;
    bool useEar = false;  // 2D: ignore distance / pan / air absorption
};

class CommandRing {
public:
    void init(size_t capacityPow2) {
        buf_.assign(capacityPow2, Command{});
        mask_ = capacityPow2 - 1;
        head_.store(0, std::memory_order_relaxed);
        tail_.store(0, std::memory_order_relaxed);
        cachedTail_ = 0;
    }
    // Producer side (game thread).  Returns false when the ring is full.
    bool push(const Command& c) {
        size_t h = head_.load(std::memory_order_relaxed);
        if (h - cachedTail_ >= mask_ + 1) {
            cachedTail_ = tail_.load(std::memory_order_acquire);
            if (h - cachedTail_ >= mask_ + 1) return false;
        }
        buf_[h & mask_] = c;
        head_.store(h + 1, std::memory_order_release);
        return true;
    }
    // Consumer side (audio thread / offline render path).
    bool pop(Command& out) {
        size_t t = tail_.load(std::memory_order_relaxed);
        if (t == head_.load(std::memory_order_acquire)) return false;
        out = buf_[t & mask_];
        tail_.store(t + 1, std::memory_order_release);
        return true;
    }

private:
    std::vector<Command> buf_;
    size_t mask_ = 0;
    size_t cachedTail_ = 0;
    alignas(64) std::atomic<size_t> head_{0};
    alignas(64) std::atomic<size_t> tail_{0};
};

// ===========================================================================
// 12. mixer state
// ===========================================================================

constexpr int kItdMaxSamples = 64;  // 1.33 ms at 48 kHz: room for the 0.7 ms ITD

// Incremented by the device callback; exposed through audioCallbackCount().
std::atomic<uint64_t> g_callbackCount{0};

struct ListenerState {
    Vec3 position{};
    Vec3 forward{0, 0, -1};
    Vec3 up{0, 1, 0};
    Vec3 right{1, 0, 0};
};

struct VoiceSlot {
    // Atomic diagnostics only; the render thread owns everything else.
    std::atomic<bool> playing{false};  // handle is valid and still sounding
    std::atomic<float> gain{0.0f};     // last applied linear gain (for stealing)
    // Render-thread-owned state.
    uint32_t handle = 0;
    SoundId sampleId = SoundId::Count;
    int startDelay = 0;  // samples until this voice begins (delaySeconds)
    float pos = 0.0f;
    float step = 1.0f;
    float vol = 1.0f;
    float gl = 0.0f, gr = 0.0f;
    float send = 0.0f;
    float airCutoff = 20000.0f;
    float fadeGain = 1.0f, fadeStep = 0.0f;
    bool fading = false;
    bool useEar = false;
    float lpL = 0.0f, lpR = 0.0f;
    // Small inter-channel delay (<0.7 ms): the far ear's channel is delayed.
    std::vector<float> itdL, itdR;
    int itdWrite = 0;
    float itdDelay = 0.0f;
    bool itdLeft = true;  // which channel is the far (delayed) ear
};

struct Engine::Impl {
    std::vector<SoundEntry> bank;
    int sampleRate = 48000;
    int channels = 2;
    int maxVoices = 64;
    float masterVolume = 0.8f;

    ListenerState listener;  // game-thread copy used to spatialise play() calls
    std::atomic<float> masterGain{0.8f};

    std::vector<VoiceSlot> voices;
    CommandRing ring;

    std::atomic<uint32_t> nextHandle{1};
    std::atomic<uint32_t> generation{0};
    std::atomic<int> activeVoices{0};
    std::atomic<float> peakL{0.0f};
    std::atomic<float> peakR{0.0f};

    // Reverb bus: a dedicated stereo Schroeder network tuned to the room.
    SchroederReverb reverb;
    EarlyReflections early;
    std::vector<float> wetScratch;  // stereo interleaved, allocated up front
    std::vector<float> dryScratch;  // stereo interleaved, allocated up front

    bool hasDevice = false;
    SDL_AudioDeviceID device = 0;
};

// ---- spatialisation (game thread) -----------------------------------------

// Constant-power pan from the listener's right vector, inverse-distance
// attenuation with a 2 m reference, air absorption, occlusion and a
// distance-dependent reverb send.
void computeSpatial(const ListenerState& L, Vec3 pos, const PlayParams& p, float master, int sr,
                    float defaultSend, float& outGain, float& outPan, float& outItd, float& outCutoff,
                    float& outSend) {
    Vec3 d = pos - L.position;
    float dist = length(d);
    Vec3 dir = dist > EPS ? d / dist : Vec3(0, 0, 0);

    // Inverse-distance attenuation: unity inside the 2 m reference distance,
    // strict 1/d beyond it, and never evaluated at exactly zero distance.
    const float ref = 2.0f;
    float dEff = std::max(dist, 1.0f);
    float atten = ref / std::max(dEff, ref);
    if (dist < ref) atten = 1.0f - 0.15f * (1.0f - dist / ref);  // gentle near field

    // Air absorption: the further away, the darker and slightly quieter.
    float cutoff = lerpf(20000.0f, 1650.0f, clamp((dist - 1.0f) / 28.0f, 0.0f, 1.0f));
    float absGain = lerpf(1.0f, 0.72f, clamp(dist / 40.0f, 0.0f, 1.0f));

    // Head shadow: sources behind the listener are a little darker and quieter.
    float front = dot(dir, L.forward);
    float behind = 1.0f - 0.22f * clamp(-front, 0.0f, 1.0f);

    // Occlusion: progressive low-pass plus gain reduction.
    float occ = clamp(p.occlusion, 0.0f, 1.0f);
    cutoff = lerpf(cutoff, std::min(cutoff, 620.0f), occ);
    float occGain = lerpf(1.0f, 0.30f, occ);

    outGain = atten * absGain * behind * occGain * p.volume * master;

    float pan = clamp(dot(dir, L.right), -1.0f, 1.0f);
    outPan = pan;
    // Inter-channel delay: the far ear lags the near ear by up to ~0.65 ms.
    outItd = pan * 0.65f * 1e-3f * float(sr);
    outCutoff = clamp(cutoff, 200.0f, 20000.0f);
    // The wet send rises with distance (more indirect energy arrives) and with
    // occlusion; an explicit PlayParams::reverbSend overrides the default.
    float send = p.reverbSend >= 0.0f ? p.reverbSend : defaultSend;
    outSend = clamp(send * lerpf(0.85f, 1.55f, clamp(dist / 25.0f, 0.0f, 1.0f)) *
                        lerpf(1.0f, 1.35f, occ),
                    0.0f, 1.5f);
}

// ---- id / handle helpers --------------------------------------------------

// Handles are allocated on the game thread so play() can return one immediately;
// the slot index is derived from the handle so a stale handle can never address
// a recycled slot.
inline int slotForHandle(uint32_t h, int maxVoices) {
    return int((h - 1) % uint32_t(std::max(maxVoices, 1)));
}

// ---- render path ----------------------------------------------------------

inline float readSample(const std::vector<float>& b, float d) {
    int n = int(b.size());
    int i0 = int(d);
    if (i0 < 0) i0 = 0;
    if (i0 >= n) return 0.0f;
    float fr = d - float(i0);
    int i1 = (i0 + 1 < n) ? i0 + 1 : i0;
    return b[size_t(i0)] * (1.0f - fr) + b[size_t(i1)] * fr;
}

// Reads the fractional delay tap in the ITD ring (write index is the newest
// sample), so a tap of `1 + d` samples back yields the delayed channel.
inline float itdRead(const std::vector<float>& b, int write, int delaySamples) {
    int n = int(b.size());
    if (n <= 0) return 0.0f;
    delaySamples = std::min(std::max(delaySamples, 1), n - 1);
    int idx = (write - delaySamples + n) % n;
    return b[size_t(idx)];
}

// Chooses a slot for a new voice: a free slot if one exists, otherwise the
// quietest (ties broken by oldest) voice, so a new sound is never dropped just
// because the pool is momentarily full.
int allocateSlot(Engine::Impl& im, uint32_t handle) {
    int n = im.maxVoices;
    if (n <= 0) return -1;
    int preferred = slotForHandle(handle, n);
    for (int i = 0; i < n; ++i) {
        int idx = (preferred + i) % n;
        if (!im.voices[size_t(idx)].playing.load(std::memory_order_relaxed)) return idx;
    }
    int best = preferred;
    float bestGain = 1e30f;
    for (int i = 0; i < n; ++i) {
        int idx = (preferred + i) % n;
        float g = im.voices[size_t(idx)].gain.load(std::memory_order_relaxed);
        if (g < bestGain - 1e-6f) {
            bestGain = g;
            best = idx;
        }
    }
    return best;
}

void startVoice(Engine::Impl& im, const Command& c) {
    if (uint32_t(c.sound) >= kSoundCount || im.bank.empty()) return;
    const SoundEntry& e = im.bank[size_t(c.sound)];
    if (e.samples.empty()) return;
    int idx = allocateSlot(im, c.id);
    if (idx < 0) return;
    VoiceSlot& v = im.voices[size_t(idx)];
    v.handle = c.id;
    v.sampleId = c.sound;
    v.pos = 0.0f;
    v.step = std::max(c.params.pitch, 0.01f);
    v.vol = c.params.volume;
    v.fading = false;
    v.fadeGain = 1.0f;
    v.fadeStep = 0.0f;
    float pan = clamp(c.pan, -1.0f, 1.0f);
    if (c.useEar) {
        // 2D: centred, equal power, no head shadow.
        v.gl = v.gr = 0.70710678f;
    } else {
        // Constant-power pan: gainL = cos(theta), gainR = sin(theta).
        float angle = (pan * 0.5f + 0.5f) * HALF_PI;
        v.gl = std::cos(angle);
        v.gr = std::sin(angle);
    }
    v.gl *= c.gain;
    v.gr *= c.gain;
    v.send = c.reverbSend;
    v.airCutoff = c.airCutoff;
    v.useEar = c.useEar;
    v.lpL = v.lpR = 0.0f;
    v.itdDelay = std::min(std::fabs(c.itd), float(kItdMaxSamples - 2));
    v.itdLeft = c.itd <= 0.0f;  // source on the left -> left ear is near
    v.itdWrite = 0;
    std::fill(v.itdL.begin(), v.itdL.end(), 0.0f);
    std::fill(v.itdR.begin(), v.itdR.end(), 0.0f);
    v.startDelay = int(c.params.delaySeconds * float(im.sampleRate) + 0.5f);
    v.gain.store(0.5f * (v.gl + v.gr), std::memory_order_relaxed);
    v.playing.store(true, std::memory_order_relaxed);
}

void applyStop(VoiceSlot& v, float fadeSeconds) {
    if (fadeSeconds <= 1e-4f) {
        v.playing.store(false, std::memory_order_relaxed);
        v.fading = false;
        v.fadeGain = 0.0f;
        return;
    }
    v.fading = true;
    v.fadeStep = 1.0f / std::max(fadeSeconds, 1e-3f);
}

void drainCommands(Engine::Impl& im) {
    Command c;
    while (im.ring.pop(c)) {
        // Drop anything issued before the most recent stopAll().
        if (c.gen != im.generation.load(std::memory_order_acquire)) continue;
        switch (c.type) {
            case CmdType::Play: startVoice(im, c); break;
            case CmdType::Stop: {
                if (c.sound == SoundId::Count) {  // stopAll(): fade every voice
                    for (auto& v : im.voices) {
                        if (v.playing.load(std::memory_order_relaxed)) applyStop(v, c.gain);
                    }
                } else {
                    const uint32_t h = c.id;
                    VoiceSlot& v = im.voices[size_t(slotForHandle(h, im.maxVoices))];
                    if (v.handle == h && v.playing.load(std::memory_order_relaxed)) {
                        applyStop(v, c.gain);
                    }
                }
                break;
            }
            case CmdType::StopAll:
                for (auto& v : im.voices) {
                    v.playing.store(false, std::memory_order_relaxed);
                    v.gain.store(0.0f, std::memory_order_relaxed);
                }
                break;
            case CmdType::MasterVolume: im.masterGain.store(c.gain, std::memory_order_relaxed); break;
            case CmdType::Listener: break;  // spatial terms are pre-computed
        }
    }
}

// The one mixing path, shared by the SDL callback, by update() in null-device
// mode and by renderOffline().  Never allocates, never locks, never blocks.
void Engine::renderBlock(float* out, int frames) {
    Engine::Impl& im = *impl_;
    if (frames <= 0) return;
    const int sr = im.sampleRate;
    const int channels = im.channels;
    drainCommands(im);

    const size_t need = size_t(frames) * 2;
    // Scratch is pre-allocated by init(); this only ever grows if a caller asks
    // for a larger block than the engine reserved, which cannot happen on the
    // audio thread because the device block size is fixed at init().
    if (im.dryScratch.size() < need) im.dryScratch.resize(need, 0.0f);
    if (im.wetScratch.size() < need) im.wetScratch.resize(need, 0.0f);
    float* dry = im.dryScratch.data();
    float* wet = im.wetScratch.data();
    std::fill(dry, dry + need, 0.0f);
    std::fill(wet, wet + need, 0.0f);

    // ---- voices ----------------------------------------------------------
    for (int vi = 0; vi < im.maxVoices; ++vi) {
        VoiceSlot& v = im.voices[size_t(vi)];
        if (!v.playing.load(std::memory_order_relaxed)) continue;
        if (uint32_t(v.sampleId) >= kSoundCount) {
            v.playing.store(false, std::memory_order_relaxed);
            continue;
        }
        const std::vector<float>& src = im.bank[size_t(v.sampleId)].samples;
        const int srcLen = int(src.size());
        if (srcLen <= 0) {
            v.playing.store(false, std::memory_order_relaxed);
            continue;
        }
        const float lpCoef =
            1.0f - std::exp(-TWO_PI * clamp(v.airCutoff, 200.0f, 0.45f * float(sr)) / float(sr));
        bool finished = false;
        for (int i = 0; i < frames; ++i) {
            if (v.startDelay > 0) {
                --v.startDelay;
                continue;  // scheduled in the future
            }
            if (v.pos >= float(srcLen)) {
                finished = true;
                break;
            }
            float s = readSample(src, v.pos);
            v.pos += v.step;

            if (v.fading) {
                v.fadeGain -= v.fadeStep / float(sr);
                if (v.fadeGain <= 0.0f) {
                    v.fadeGain = 0.0f;
                    finished = true;
                    break;
                }
            }
            const float g = v.fadeGain;
            float l = s * v.gl * g;
            float r = s * v.gr * g;

            // Air absorption / occlusion: one-pole low-pass per channel.
            v.lpL += lpCoef * (l - v.lpL);
            v.lpR += lpCoef * (r - v.lpR);
            l = v.lpL;
            r = v.lpR;

            // Inter-channel delay (<0.7 ms) into the two short ring lines.
            v.itdL[size_t(v.itdWrite)] = l;
            v.itdR[size_t(v.itdWrite)] = r;
            const int d = int(v.itdDelay);
            float lOut = l, rOut = r;
            if (d > 0) {
                if (v.itdLeft) {
                    lOut = itdRead(v.itdL, v.itdWrite, 1 + d);  // left ear is far
                } else {
                    rOut = itdRead(v.itdR, v.itdWrite, 1 + d);  // right ear is far
                }
            }
            v.itdWrite = (v.itdWrite + 1) % kItdMaxSamples;

            dry[2 * i] += lOut;
            dry[2 * i + 1] += rOut;
            const float mono = (lOut + rOut) * 0.5f;
            wet[2 * i] += mono * v.send;
            wet[2 * i + 1] += mono * v.send;
        }
        v.gain.store(0.5f * (std::fabs(v.gl) + std::fabs(v.gr)) * v.fadeGain,
                     std::memory_order_relaxed);
        if (finished) {
            v.playing.store(false, std::memory_order_relaxed);
            v.gain.store(0.0f, std::memory_order_relaxed);
        }
    }

    // ---- room reverb bus (early reflections + late tail) ------------------
    float rl = 0.0f, rr = 0.0f;
    for (int i = 0; i < frames; ++i) {
        float el = 0.0f, er = 0.0f, wl = 0.0f, wr = 0.0f;
        im.early.process(wet[2 * i], wet[2 * i + 1], el, er);
        im.reverb.process(wet[2 * i], wet[2 * i + 1], wl, wr);
        rl = el * 0.35f + wl;
        rr = er * 0.35f + wr;
        wet[2 * i] = rl;
        wet[2 * i + 1] = rr;
    }

    // ---- master bus ------------------------------------------------------
    const float master = im.masterGain.load(std::memory_order_relaxed);
    float peakL = 0.0f, peakR = 0.0f;
    for (int i = 0; i < frames; ++i) {
        float l = (dry[2 * i] + wet[2 * i]) * master;
        float r = (dry[2 * i + 1] + wet[2 * i + 1]) * master;
        // Soft saturation instead of hard clipping: the mix can approach full
        // scale without ever exceeding it and without harsh clipping artefacts.
        l = clamp(softClip(l * 0.8f) * 1.25f, -1.0f, 1.0f);
        r = clamp(softClip(r * 0.8f) * 1.25f, -1.0f, 1.0f);
        peakL = std::max(peakL, std::fabs(l));
        peakR = std::max(peakR, std::fabs(r));
        if (channels == 2) {
            out[2 * i] = l;
            out[2 * i + 1] = r;
        } else {
            out[i] = 0.5f * (l + r);
        }
    }

    // ---- diagnostics -----------------------------------------------------
    if (peakL > im.peakL.load(std::memory_order_relaxed)) {
        im.peakL.store(peakL, std::memory_order_relaxed);
    }
    if (peakR > im.peakR.load(std::memory_order_relaxed)) {
        im.peakR.store(peakR, std::memory_order_relaxed);
    }
    int active = 0;
    for (const auto& v : im.voices) {
        if (v.playing.load(std::memory_order_relaxed)) ++active;
    }
    im.activeVoices.store(active, std::memory_order_relaxed);
}

namespace {

// ---- wav writer -----------------------------------------------------------

bool writeWav16(const std::string& path, const float* mono, int frames, int sampleRate) {
    if (frames <= 0) return false;
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    const int outCh = 2;  // dumpWavFiles always writes 16-bit stereo
    const uint32_t dataBytes = uint32_t(frames) * uint32_t(outCh) * 2u;
    auto u32 = [&](uint32_t v) { f.write(reinterpret_cast<const char*>(&v), 4); };
    auto u16 = [&](uint16_t v) { f.write(reinterpret_cast<const char*>(&v), 2); };
    f.write("RIFF", 4);
    u32(36u + dataBytes);
    f.write("WAVE", 4);
    f.write("fmt ", 4);
    u32(16);
    u16(1);  // PCM
    u16(uint16_t(outCh));
    u32(uint32_t(sampleRate));
    u32(uint32_t(sampleRate) * uint32_t(outCh) * 2u);
    u16(uint16_t(outCh * 2));
    u16(16);
    f.write("data", 4);
    u32(dataBytes);
    std::vector<int16_t> pcm(size_t(frames) * size_t(outCh));
    for (int i = 0; i < frames; ++i) {
        float v = clamp(mono[i], -1.0f, 1.0f);
        int16_t s = int16_t(std::lround(v * 32767.0f));
        pcm[size_t(i) * 2 + 0] = s;
        pcm[size_t(i) * 2 + 1] = s;
    }
    f.write(reinterpret_cast<const char*>(pcm.data()), std::streamsize(pcm.size() * sizeof(int16_t)));
    return f.good();
}

const char* kSoundNames[kSoundCount] = {
    "gunshot",      "gunshot_tail", "dry_fire",      "safety_click", "mag_release",
    "mag_out",      "mag_in",       "slide_back",    "slide_forward", "shell_drop",
    "shell_bounce", "glass_shatter", "glass_tinkle", "glass_stress",  "impact_concrete",
    "impact_wood",  "impact_metal", "footstep",      "footstep_alt",  "weapon_draw",
    "weapon_holster", "cloth",      "room_tone",     "ear_ring",      "ui_beep"};

}  // namespace

// ===========================================================================
// 13. Engine
// ===========================================================================

Engine::Engine() {
    impl_ = new (std::nothrow) Impl();
    if (impl_) {
        impl_->ring.init(1024);
        impl_->listener.position = Vec3(0, 0, 0);
    }
}

Engine::~Engine() {
    shutdown();
    delete impl_;
    impl_ = nullptr;
}

bool Engine::init(int sampleRate, int channels, int maxVoices, float masterVolume) {
    if (initialised_) return true;
    if (!impl_) {
        lastError_ = "audio: out of memory";
        return false;
    }
    sampleRate_ = clamp(sampleRate, 8000, 192000);
    channels_ = clamp(channels, 1, 2);
    masterVolume_ = clamp(masterVolume, 0.0f, 1.0f);
    const int voices = clamp(maxVoices, 1, 256);

    // --- SDL audio --------------------------------------------------------
    std::string deviceError;
    if (SDL_WasInit(SDL_INIT_AUDIO) == 0) {
        if (SDL_InitSubSystem(SDL_INIT_AUDIO) != 0) {
            deviceError = std::string("SDL_InitSubSystem(audio) failed: ") + SDL_GetError();
        }
    }
    SDL_AudioSpec want{};
    want.freq = sampleRate_;
    want.format = AUDIO_F32SYS;
    want.channels = Uint8(channels_);
    want.samples = 512;  // ~10.7 ms at 48 kHz: low latency, no underrun risk
    want.callback = &Engine::audioCallbackThunk;
    want.userdata = this;

    // An explicit opt-out keeps the null-device path testable on machines that
    // do have a working server (and makes offline verification deterministic,
    // since a live callback would consume commands behind the caller's back).
    bool forceNull = false;
    if (const char* env = std::getenv("ROOM2_AUDIO_NULL_DEVICE")) {
        forceNull = (env[0] == '1' || env[0] == 'y' || env[0] == 'Y');
    }
    if (forceNull) deviceError = "ROOM2_AUDIO_NULL_DEVICE requested null-device mode";

    SDL_AudioSpec have{};
    if (deviceError.empty()) {
        impl_->device = SDL_OpenAudioDevice(nullptr, 0, &want, &have, 0);
        if (impl_->device == 0) {
            deviceError = std::string("SDL_OpenAudioDevice failed: ") + SDL_GetError();
        } else if (const char* drv = SDL_GetCurrentAudioDriver()) {
            // SDL's "dummy"/"disk" back ends accept a device but have nowhere to
            // play it.  Treating them as silent keeps the engine from spinning a
            // callback thread that produces inaudible output (and from writing
            // files with the disk driver).
            if (std::strcmp(drv, "dummy") == 0 || std::strcmp(drv, "disk") == 0) {
                SDL_CloseAudioDevice(impl_->device);
                impl_->device = 0;
                deviceError = std::string("SDL audio driver '") + drv + "' has no real output";
            }
        }
    }
    if (impl_->device != 0) {
        impl_->hasDevice = true;
        if (have.freq > 0) sampleRate_ = have.freq;
        if (have.channels > 0) channels_ = have.channels > 2 ? 2 : int(have.channels);
    } else {
        // NULL-DEVICE MODE: no usable output device (headless machine, no
        // /dev/snd, dummy driver only, or an explicit opt-out).  init() still
        // succeeds, the whole software mixing graph and the sound bank are fully
        // functional through update()/renderOffline(), and no audio callback
        // ever runs.
        impl_->hasDevice = false;
        lastError_ = deviceError.empty()
                         ? std::string("audio: no output device; null-device mode")
                         : deviceError + " [null-device mode]";
    }

    // --- DSP graph --------------------------------------------------------
    impl_->sampleRate = sampleRate_;
    impl_->channels = channels_;
    impl_->maxVoices = voices;
    impl_->masterVolume = masterVolume_;
    impl_->masterGain.store(masterVolume_, std::memory_order_relaxed);
    // Room preset: ~6 x 3 x 8 m hard plaster -> bright, dense, ~1 s RT60.
    impl_->reverb.init(sampleRate_, 1.05f, 0.20f, 1.0f);
    impl_->early.init(sampleRate_, 0.004f);

    // VoiceSlot holds std::atomic members, so it is neither copyable nor
    // movable; build the pool at its final size in one go.
    impl_->voices = std::vector<VoiceSlot>(size_t(voices));
    for (auto& v : impl_->voices) {
        v.itdL.assign(kItdMaxSamples, 0.0f);
        v.itdR.assign(kItdMaxSamples, 0.0f);
        v.playing.store(false, std::memory_order_relaxed);
        v.gain.store(0.0f, std::memory_order_relaxed);
    }
    impl_->dryScratch.assign(size_t(8192) * 2, 0.0f);
    impl_->wetScratch.assign(size_t(8192) * 2, 0.0f);

    // --- sound bank (always built, device or not) -------------------------
    impl_->bank = buildBank(sampleRate_);  // each synth already DC-blocks itself

    impl_->nextHandle.store(1, std::memory_order_relaxed);
    impl_->generation.store(0, std::memory_order_relaxed);
    impl_->activeVoices.store(0, std::memory_order_relaxed);
    impl_->peakL.store(0.0f, std::memory_order_relaxed);
    impl_->peakR.store(0.0f, std::memory_order_relaxed);
    initialised_ = true;

    if (impl_->hasDevice) SDL_PauseAudioDevice(impl_->device, 0);  // start the callback
    return true;
}

void Engine::shutdown() {
    if (!impl_) return;
    if (impl_->device != 0) {
        SDL_CloseAudioDevice(impl_->device);
        impl_->device = 0;
    }
    impl_->hasDevice = false;
    for (auto& v : impl_->voices) {
        v.playing.store(false, std::memory_order_relaxed);
        v.gain.store(0.0f, std::memory_order_relaxed);
    }
    impl_->activeVoices.store(0, std::memory_order_relaxed);
    initialised_ = false;
    // The bank is intentionally retained: soundSamples() references stay valid
    // and a later init()/renderSoundToBuffer() stays cheap.
}

void Engine::setListener(Vec3 position, Vec3 forward, Vec3 up) {
    if (!impl_) return;
    ListenerState L;
    L.position = position;
    L.forward = lengthSq(forward) > EPS ? normalize(forward) : Vec3(0, 0, -1);
    Vec3 u = lengthSq(up) > EPS ? normalize(up) : Vec3(0, 1, 0);
    // Right-handed basis; falls back to a world-axis-derived vector when forward
    // and up are (nearly) parallel.
    Vec3 r = cross(L.forward, u);
    if (lengthSq(r) < 1e-6f) {
        Vec3 alt = std::fabs(L.forward.y) > 0.9f ? Vec3(0, 0, 1) : Vec3(0, 1, 0);
        r = cross(L.forward, alt);
    }
    L.right = normalize(r);
    L.up = normalize(cross(L.right, L.forward));
    impl_->listener = L;

    PlayParams p;
    pushCommand(3, SoundId::Count, p, position, 0.0f, 0.0f, 0.0f, 20000.0f, 0.0f, false, 0);
}

void Engine::setMasterVolume(float v) {
    masterVolume_ = clamp(v, 0.0f, 1.0f);
    if (!impl_) return;
    impl_->masterVolume = masterVolume_;
    impl_->masterGain.store(masterVolume_, std::memory_order_relaxed);
    PlayParams p;
    pushCommand(4, SoundId::Count, p, Vec3(0, 0, 0), masterVolume_, 0.0f, 0.0f, 20000.0f, 0.0f, false,
                0);
}

void Engine::update(float dt) {
    if (!impl_ || !initialised_) return;
    dt = clamp(dt, 0.0f, 0.25f);
    if (dt <= 0.0f) return;
    int frames = int(dt * float(sampleRate_) + 0.5f);
    if (frames <= 0) return;
    // Render into the engine scratch and discard: scheduling, voice lifetimes
    // and the reverb tail all advance at the correct rate even when no device is
    // running (null-device mode).
    int remaining = frames;
    while (remaining > 0) {
        const int block = std::min(remaining, 1024);
        renderBlock(impl_->dryScratch.data(), block);
        remaining -= block;
    }
}

int Engine::activeVoiceCount() const {
    return impl_ ? impl_->activeVoices.load(std::memory_order_relaxed) : 0;
}

VoiceId Engine::play(SoundId id, Vec3 worldPosition, const PlayParams& params) {
    if (!impl_ || !initialised_ || uint32_t(id) >= kSoundCount) return INVALID_VOICE;
    PlayParams p = params;
    p.volume = clamp(params.volume, 0.0f, 8.0f);
    p.pitch = clamp(params.pitch, 0.05f, 8.0f);
    p.delaySeconds = clamp(params.delaySeconds, 0.0f, 30.0f);
    float gain = 1.0f, pan = 0.0f, itd = 0.0f, cutoff = 20000.0f, send = 0.0f;
    computeSpatial(impl_->listener, worldPosition, p,
                   impl_->masterGain.load(std::memory_order_relaxed), impl_->sampleRate,
                   impl_->bank[size_t(id)].defaultReverbSend, gain, pan, itd, cutoff, send);
    const uint32_t handle = impl_->nextHandle.fetch_add(1, std::memory_order_relaxed);
    pushCommand(0, id, p, worldPosition, gain, pan, itd, cutoff, send, false, handle);
    return VoiceId(handle);
}

VoiceId Engine::play2D(SoundId id, const PlayParams& params) {
    if (!impl_ || !initialised_ || uint32_t(id) >= kSoundCount) return INVALID_VOICE;
    PlayParams p = params;
    p.volume = clamp(params.volume, 0.0f, 8.0f);
    p.pitch = clamp(params.pitch, 0.05f, 8.0f);
    p.delaySeconds = clamp(params.delaySeconds, 0.0f, 30.0f);
    const float gain = p.volume * impl_->masterGain.load(std::memory_order_relaxed);
    const float send = params.reverbSend >= 0.0f ? params.reverbSend
                                                  : impl_->bank[size_t(id)].defaultReverbSend;
    const uint32_t handle = impl_->nextHandle.fetch_add(1, std::memory_order_relaxed);
    pushCommand(0, id, p, Vec3(0, 0, 0), gain, 0.0f, 0.0f, 20000.0f, clamp(send, 0.0f, 1.5f), true,
                handle);
    return VoiceId(handle);
}

void Engine::stop(VoiceId voice, float fadeSeconds) {
    if (!impl_ || voice == INVALID_VOICE) return;
    PlayParams p;
    // For a stop command the fade time travels in `gain` (see drainCommands).
    pushCommand(1, SoundId::Count, p, Vec3(0, 0, 0), clamp(fadeSeconds, 0.0f, 5.0f), 0.0f, 0.0f,
                20000.0f, 0.0f, false, uint32_t(voice));
}

void Engine::stopAll() {
    if (!impl_) return;
    // Bumping the generation invalidates every Play/Stop command queued before
    // this call, so nothing can sneak in behind the stop.
    impl_->generation.fetch_add(1, std::memory_order_acq_rel);
    PlayParams p;
    if (!pushCommand(2, SoundId::Count, p, Vec3(0, 0, 0), 0.01f, 0.0f, 0.0f, 20000.0f, 0.0f, false,
                     0)) {
        // Ring full: fall back to an immediate hard reset of every voice.
        for (auto& v : impl_->voices) v.playing.store(false, std::memory_order_relaxed);
    }
}

bool Engine::isPlaying(VoiceId voice) const {
    if (!impl_ || voice == INVALID_VOICE) return false;
    const uint32_t h = uint32_t(voice);
    const VoiceSlot& v = impl_->voices[size_t(slotForHandle(h, impl_->maxVoices))];
    return v.playing.load(std::memory_order_relaxed) && v.handle == h;
}

const std::vector<float>& Engine::soundSamples(SoundId id) const {
    static const std::vector<float> kEmpty;
    if (uint32_t(id) >= kSoundCount) return kEmpty;
    if (impl_ && !impl_->bank.empty()) return impl_->bank[size_t(id)].samples;
    return sharedBank()[size_t(id)].samples;
}

int Engine::dumpWavFiles(const std::string& directory) const {
    if (!impl_) return 0;
    std::error_code ec;
    std::filesystem::create_directories(directory, ec);
    if (ec) return 0;
    const std::vector<SoundEntry>& bank = impl_->bank.empty() ? sharedBank() : impl_->bank;
    int written = 0;
    for (uint32_t i = 0; i < kSoundCount; ++i) {
        const SoundEntry& e = bank[size_t(i)];
        if (e.samples.empty()) continue;
        std::string path = directory;
        if (!path.empty() && path.back() != '/') path += '/';
        path += kSoundNames[i];
        path += ".wav";
        if (writeWav16(path, e.samples.data(), int(e.samples.size()), sampleRate_)) ++written;
    }
    return written;
}

void Engine::renderOffline(float* out, int frames) {
    if (!out || frames <= 0) return;
    if (!impl_) {
        std::memset(out, 0, size_t(frames) * 2 * sizeof(float));
        return;
    }
    int done = 0;
    while (done < frames) {
        const int block = std::min(frames - done, 1024);
        renderBlock(out + size_t(done) * 2, block);
        done += block;
    }
}

Vec2 Engine::takePeakLevel() {
    if (!impl_) return Vec2(0, 0);
    return Vec2(impl_->peakL.exchange(0.0f, std::memory_order_relaxed),
                impl_->peakR.exchange(0.0f, std::memory_order_relaxed));
}

void Engine::audioCallbackThunk(void* userdata, Uint8* stream, int len) {
    Engine* self = static_cast<Engine*>(userdata);
    if (!self || !self->impl_) {
        std::memset(stream, 0, size_t(len));
        return;
    }
    Engine::Impl& im = *self->impl_;
    g_callbackCount.fetch_add(1, std::memory_order_relaxed);
    const int ch = std::max(im.channels, 1);
    float* out = reinterpret_cast<float*>(stream);
    const int totalFrames = len / int(sizeof(float)) / ch;
    int done = 0;
    while (done < totalFrames) {
        const int block = std::min(totalFrames - done, 1024);
        self->renderBlock(out + size_t(done) * size_t(ch), block);
        done += block;
    }
}

bool Engine::pushCommand(int what, SoundId sound, const PlayParams& params, const Vec3& position,
                         float gain, float pan, float itd, float airCutoff, float reverbSend,
                         bool useEar, uint32_t voiceHandle) {
    if (!impl_) return false;
    Command cmd;
    switch (what) {
        case 0: cmd.type = CmdType::Play; break;
        case 1: cmd.type = CmdType::Stop; break;
        case 3: cmd.type = CmdType::Listener; break;
        case 4: cmd.type = CmdType::MasterVolume; break;
        default: cmd.type = CmdType::StopAll; break;
    }
    cmd.sound = sound;
    cmd.params = params;
    cmd.position = position;
    cmd.gain = gain;
    cmd.pan = pan;
    cmd.itd = itd;
    cmd.airCutoff = airCutoff;
    cmd.reverbSend = reverbSend;
    cmd.useEar = useEar;
    cmd.id = voiceHandle;
    // Stamped with the generation the producer observed: a command issued
    // before a stopAll() is dropped by the consumer instead of being applied
    // after the reset.  For a Play the handle was allocated by the caller.
    cmd.gen = impl_->generation.load(std::memory_order_acquire);
    // The producer never blocks: a short bounded spin covers a momentarily full
    // ring (only possible if the game thread enqueues >1000 commands between two
    // callbacks).  Dropping is the correct last resort for audio.
    for (int attempt = 0; attempt < 8; ++attempt) {
        if (impl_->ring.push(cmd)) return true;
        std::this_thread::yield();
    }
    return false;
}

// ===========================================================================
// 14. free functions
// ===========================================================================

uint64_t audioCallbackCount() {
    // Number of SDL audio callbacks entered since start-up.  Zero proves the
    // engine is running in null-device mode (or that the device never started).
    return g_callbackCount.load(std::memory_order_relaxed);
}

std::vector<float> renderSoundToBuffer(SoundId id, float seconds, int sampleRate) {
    std::vector<float> out;
    if (sampleRate <= 0 || seconds <= 0.0f) return out;
    if (uint32_t(id) >= kSoundCount) return out;
    const std::vector<float>& src = sharedBank()[size_t(id)].samples;
    const int sr = sampleRate;
    const int frames = int(seconds * float(sr));
    out.assign(size_t(frames), 0.0f);
    if (src.empty() || frames <= 0) return out;

    if (sr == 48000) {
        const int n = std::min(frames, int(src.size()));
        std::copy(src.begin(), src.begin() + n, out.begin());
        return out;
    }
    // Resample with linear interpolation (upsampling) or a box average
    // (downsampling, which avoids aliasing the bright transients).
    const double ratio = 48000.0 / double(sr);
    if (ratio <= 1.0) {
        for (int i = 0; i < frames; ++i) {
            double sp = double(i) * ratio;
            int i0 = int(sp);
            if (i0 + 1 >= int(src.size())) break;
            float fr = float(sp - double(i0));
            out[size_t(i)] = src[size_t(i0)] * (1.0f - fr) + src[size_t(i0 + 1)] * fr;
        }
    } else {
        const int tap = int(ratio);
        for (int i = 0; i < frames; ++i) {
            const size_t begin = size_t(double(i) * ratio);
            const size_t end = std::min(begin + size_t(tap), src.size());
            if (begin >= src.size()) break;
            float acc = 0.0f;
            for (size_t k = begin; k < end; ++k) acc += src[k];
            out[size_t(i)] = acc / float(std::max<size_t>(end - begin, 1));
        }
    }
    return out;
}

}  // namespace room2::audio
