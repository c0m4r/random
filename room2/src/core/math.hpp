// room2 - self-contained linear algebra (no external math dependency).
// Convention: right-handed world space, Y up, column-major matrices (GLSL/Vulkan friendly),
// depth range [0,1] for Vulkan clip space.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>

namespace room2 {

inline constexpr float PI      = 3.14159265358979323846f;
inline constexpr float TWO_PI  = 6.28318530717958647692f;
inline constexpr float HALF_PI = 1.57079632679489661923f;
inline constexpr float DEG2RAD = PI / 180.0f;
inline constexpr float RAD2DEG = 180.0f / PI;
inline constexpr float EPS     = 1e-6f;

template <typename T> inline T min2(T a, T b) { return a < b ? a : b; }
template <typename T> inline T max2(T a, T b) { return a > b ? a : b; }
template <typename T> inline T clamp(T v, T lo, T hi) { return v < lo ? lo : (v > hi ? hi : v); }
inline float lerpf(float a, float b, float t) { return a + (b - a) * t; }
inline float smoothstepf(float e0, float e1, float x) {
    float t = clamp((x - e0) / (e1 - e0 + EPS), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}
inline float signf(float x) { return x < 0.0f ? -1.0f : 1.0f; }
// Frame-rate independent exponential smoothing factor.
inline float dampf(float rate, float dt) { return 1.0f - std::exp(-rate * dt); }

// ---------------------------------------------------------------- Vec2
struct Vec2 {
    float x = 0, y = 0;
    Vec2() = default;
    Vec2(float a, float b) : x(a), y(b) {}
    explicit Vec2(float s) : x(s), y(s) {}
    float&       operator[](int i)       { return (&x)[i]; }
    const float& operator[](int i) const { return (&x)[i]; }
};
inline Vec2 operator+(Vec2 a, Vec2 b) { return {a.x + b.x, a.y + b.y}; }
inline Vec2 operator-(Vec2 a, Vec2 b) { return {a.x - b.x, a.y - b.y}; }
inline Vec2 operator*(Vec2 a, float s) { return {a.x * s, a.y * s}; }
inline Vec2 operator*(float s, Vec2 a) { return {a.x * s, a.y * s}; }
inline Vec2 operator*(Vec2 a, Vec2 b) { return {a.x * b.x, a.y * b.y}; }
inline Vec2 operator/(Vec2 a, float s) { return {a.x / s, a.y / s}; }
inline Vec2 operator-(Vec2 a) { return {-a.x, -a.y}; }
inline Vec2& operator+=(Vec2& a, Vec2 b) { a.x += b.x; a.y += b.y; return a; }
inline Vec2& operator-=(Vec2& a, Vec2 b) { a.x -= b.x; a.y -= b.y; return a; }
inline Vec2& operator*=(Vec2& a, float s) { a.x *= s; a.y *= s; return a; }
inline float dot(Vec2 a, Vec2 b) { return a.x * b.x + a.y * b.y; }
inline float length(Vec2 a) { return std::sqrt(dot(a, a)); }
inline Vec2 normalize(Vec2 a) { float l = length(a); return l > EPS ? a / l : Vec2(0, 0); }
inline Vec2 lerp(Vec2 a, Vec2 b, float t) { return a + (b - a) * t; }

// ---------------------------------------------------------------- Vec3
struct Vec3 {
    float x = 0, y = 0, z = 0;
    Vec3() = default;
    Vec3(float a, float b, float c) : x(a), y(b), z(c) {}
    explicit Vec3(float s) : x(s), y(s), z(s) {}
    float&       operator[](int i)       { return (&x)[i]; }
    const float& operator[](int i) const { return (&x)[i]; }
};
inline Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline Vec3 operator*(Vec3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
inline Vec3 operator*(float s, Vec3 a) { return {a.x * s, a.y * s, a.z * s}; }
inline Vec3 operator*(Vec3 a, Vec3 b) { return {a.x * b.x, a.y * b.y, a.z * b.z}; }
inline Vec3 operator/(Vec3 a, float s) { return {a.x / s, a.y / s, a.z / s}; }
inline Vec3 operator/(Vec3 a, Vec3 b) { return {a.x / b.x, a.y / b.y, a.z / b.z}; }
inline Vec3 operator-(Vec3 a) { return {-a.x, -a.y, -a.z}; }
inline Vec3& operator+=(Vec3& a, Vec3 b) { a.x += b.x; a.y += b.y; a.z += b.z; return a; }
inline Vec3& operator-=(Vec3& a, Vec3 b) { a.x -= b.x; a.y -= b.y; a.z -= b.z; return a; }
inline Vec3& operator*=(Vec3& a, float s) { a.x *= s; a.y *= s; a.z *= s; return a; }
inline Vec3& operator/=(Vec3& a, float s) { a.x /= s; a.y /= s; a.z /= s; return a; }
inline Vec3& operator*=(Vec3& a, Vec3 b) { a.x *= b.x; a.y *= b.y; a.z *= b.z; return a; }
inline bool operator==(Vec3 a, Vec3 b) { return a.x == b.x && a.y == b.y && a.z == b.z; }
inline float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3 cross(Vec3 a, Vec3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline float length(Vec3 a) { return std::sqrt(dot(a, a)); }
inline float lengthSq(Vec3 a) { return dot(a, a); }
inline float distance(Vec3 a, Vec3 b) { return length(a - b); }
inline Vec3 normalize(Vec3 a) { float l = length(a); return l > EPS ? a / l : Vec3(0, 0, 0); }
inline Vec3 lerp(Vec3 a, Vec3 b, float t) { return a + (b - a) * t; }
inline Vec3 minv(Vec3 a, Vec3 b) { return {min2(a.x, b.x), min2(a.y, b.y), min2(a.z, b.z)}; }
inline Vec3 maxv(Vec3 a, Vec3 b) { return {max2(a.x, b.x), max2(a.y, b.y), max2(a.z, b.z)}; }
inline Vec3 clampv(Vec3 v, float lo, float hi) {
    return {clamp(v.x, lo, hi), clamp(v.y, lo, hi), clamp(v.z, lo, hi)};
}
inline Vec3 absv(Vec3 v) { return {std::fabs(v.x), std::fabs(v.y), std::fabs(v.z)}; }
inline Vec3 reflect(Vec3 i, Vec3 n) { return i - n * (2.0f * dot(n, i)); }
inline Vec3 refract(Vec3 i, Vec3 n, float eta) {
    float c = dot(n, i);
    float k = 1.0f - eta * eta * (1.0f - c * c);
    return k < 0.0f ? Vec3(0, 0, 0) : i * eta - n * (eta * c + std::sqrt(k));
}
inline Vec3 faceforward(Vec3 n, Vec3 i, Vec3 nref) { return dot(nref, i) < 0.0f ? n : -n; }
inline Vec3 powv(Vec3 a, float e) {
    return {std::pow(a.x, e), std::pow(a.y, e), std::pow(a.z, e)};
}
inline float maxComponent(Vec3 v) { return max2(v.x, max2(v.y, v.z)); }
inline float minComponent(Vec3 v) { return min2(v.x, min2(v.y, v.z)); }
inline Vec3 mixv(Vec3 a, Vec3 b, Vec3 t) { return a + (b - a) * t; }

// ---------------------------------------------------------------- Vec4
struct Vec4 {
    float x = 0, y = 0, z = 0, w = 0;
    Vec4() = default;
    Vec4(float a, float b, float c, float d) : x(a), y(b), z(c), w(d) {}
    Vec4(Vec3 v, float d) : x(v.x), y(v.y), z(v.z), w(d) {}
    explicit Vec4(float s) : x(s), y(s), z(s), w(s) {}
    Vec3 xyz() const { return {x, y, z}; }
    float&       operator[](int i)       { return (&x)[i]; }
    const float& operator[](int i) const { return (&x)[i]; }
};
inline Vec4 operator+(Vec4 a, Vec4 b) { return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w}; }
inline Vec4 operator-(Vec4 a, Vec4 b) { return {a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w}; }
inline Vec4 operator*(Vec4 a, float s) { return {a.x * s, a.y * s, a.z * s, a.w * s}; }
inline Vec4 operator*(float s, Vec4 a) { return a * s; }
inline Vec4 operator*(Vec4 a, Vec4 b) { return {a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w}; }
inline Vec4 operator/(Vec4 a, float s) { return {a.x / s, a.y / s, a.z / s, a.w / s}; }
inline Vec4 operator-(Vec4 a) { return {-a.x, -a.y, -a.z, -a.w}; }
inline Vec4& operator+=(Vec4& a, Vec4 b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; return a; }
inline Vec4& operator-=(Vec4& a, Vec4 b) { a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w; return a; }
inline Vec4& operator*=(Vec4& a, float s) { a.x *= s; a.y *= s; a.z *= s; a.w *= s; return a; }
inline Vec4& operator/=(Vec4& a, float s) { a.x /= s; a.y /= s; a.z /= s; a.w /= s; return a; }
inline Vec4& operator*=(Vec4& a, Vec4 b) { a.x *= b.x; a.y *= b.y; a.z *= b.z; a.w *= b.w; return a; }
inline bool operator==(Vec4 a, Vec4 b) {
    return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}
inline float dot(Vec4 a, Vec4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }
inline Vec4 lerp(Vec4 a, Vec4 b, float t) { return a + (b - a) * t; }

// ---------------------------------------------------------------- Mat3 (column-major)
struct Mat3 {
    // m[col][row]
    Vec3 c[3] = {Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1)};
    Mat3() = default;
    Vec3&       operator[](int i)       { return c[i]; }
    const Vec3& operator[](int i) const { return c[i]; }
};
inline Vec3 operator*(const Mat3& m, Vec3 v) {
    return m.c[0] * v.x + m.c[1] * v.y + m.c[2] * v.z;
}
inline Mat3 operator*(const Mat3& a, const Mat3& b) {
    Mat3 r;
    for (int i = 0; i < 3; ++i) r.c[i] = a * b.c[i];
    return r;
}
inline Mat3 transpose(const Mat3& m) {
    Mat3 r;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) r.c[i][j] = m.c[j][i];
    return r;
}
inline float determinant(const Mat3& m) {
    return dot(m.c[0], cross(m.c[1], m.c[2]));
}
inline Mat3 inverse(const Mat3& m) {
    float d = determinant(m);
    if (std::fabs(d) < 1e-12f) return Mat3{};
    float id = 1.0f / d;
    Mat3 r;
    r.c[0] = cross(m.c[1], m.c[2]) * id;
    r.c[1] = cross(m.c[2], m.c[0]) * id;
    r.c[2] = cross(m.c[0], m.c[1]) * id;
    return transpose(r);
}
// Build a Mat3 from columns.
inline Mat3 mat3FromCols(Vec3 a, Vec3 b, Vec3 c) {
    Mat3 r; r.c[0] = a; r.c[1] = b; r.c[2] = c; return r;
}
inline Mat3 mat3Diagonal(Vec3 d) {
    Mat3 r;
    r.c[0] = {d.x, 0, 0}; r.c[1] = {0, d.y, 0}; r.c[2] = {0, 0, d.z};
    return r;
}
// Skew-symmetric cross product matrix: cross(a,b) == skew(a) * b
inline Mat3 skew(Vec3 a) {
    return mat3FromCols(Vec3(0, a.z, -a.y), Vec3(-a.z, 0, a.x), Vec3(a.y, -a.x, 0));
}
// Outer product a * b^T
inline Mat3 outer(Vec3 a, Vec3 b) {
    return mat3FromCols(a * b.x, a * b.y, a * b.z);
}

// ---------------------------------------------------------------- Quat (x,y,z,w)
struct Quat {
    float x = 0, y = 0, z = 0, w = 1;
    Quat() = default;
    Quat(float a, float b, float c, float d) : x(a), y(b), z(c), w(d) {}
    Quat(Vec3 v, float s) : x(v.x), y(v.y), z(v.z), w(s) {}
    Vec3 xyz() const { return {x, y, z}; }
};
inline Quat operator*(Quat a, Quat b) {
    return {a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
            a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
            a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
            a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z};
}
inline Quat operator*(Quat q, float s) { return {q.x * s, q.y * s, q.z * s, q.w * s}; }
inline Quat operator+(Quat a, Quat b) { return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w}; }
inline Quat operator-(Quat a, Quat b) { return {a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w}; }
inline Quat operator-(Quat a) { return {-a.x, -a.y, -a.z, -a.w}; }
inline float dot(Quat a, Quat b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }
inline float length(Quat q) { return std::sqrt(dot(q, q)); }
inline Quat normalize(Quat q) { float l = length(q); return l > EPS ? q * (1.0f / l) : Quat(0, 0, 0, 1); }
inline Quat conjugate(Quat q) { return {-q.x, -q.y, -q.z, q.w}; }
inline Quat inverse(Quat q) { return conjugate(q) * (1.0f / std::max(dot(q, q), EPS)); }
inline Quat lerp(Quat a, Quat b, float t) { return normalize(a + (b - a) * t); }
inline Quat slerp(Quat a, Quat b, float t) {
    float d = dot(a, b);
    if (d < 0.0f) { b = -b; d = -d; }
    if (d > 0.9995f) return normalize(lerp(a, b, t));
    float theta = std::acos(clamp(d, -1.0f, 1.0f));
    float st = std::sin(theta);
    float wa = std::sin((1.0f - t) * theta) / st;
    float wb = std::sin(t * theta) / st;
    return normalize(a * wa + b * wb);
}
// Rotate vector v by quaternion q.
inline Vec3 rotate(Quat q, Vec3 v) {
    Vec3 u = q.xyz();
    return u * (2.0f * dot(u, v)) + v * (q.w * q.w - dot(u, u)) + cross(u, v) * (2.0f * q.w);
}
inline Quat quatFromAxisAngle(Vec3 axis, float radians) {
    float l = length(axis);
    if (l < EPS) return Quat(0, 0, 0, 1);
    float h = radians * 0.5f;
    float s = std::sin(h) / l;
    return {axis.x * s, axis.y * s, axis.z * s, std::cos(h)};
}
inline Quat quatFromEuler(Vec3 eulerRad) {  // applied Y * X * Z (yaw, pitch, roll)
    Quat qy = quatFromAxisAngle({0, 1, 0}, eulerRad.y);
    Quat qx = quatFromAxisAngle({1, 0, 0}, eulerRad.x);
    Quat qz = quatFromAxisAngle({0, 0, 1}, eulerRad.z);
    return qy * qx * qz;
}
inline Mat3 mat3FromQuat(Quat q) {
    float xx = q.x * q.x, yy = q.y * q.y, zz = q.z * q.z;
    float xy = q.x * q.y, xz = q.x * q.z, yz = q.y * q.z;
    float wx = q.w * q.x, wy = q.w * q.y, wz = q.w * q.z;
    Mat3 r;
    r.c[0] = {1.0f - 2.0f * (yy + zz), 2.0f * (xy + wz), 2.0f * (xz - wy)};
    r.c[1] = {2.0f * (xy - wz), 1.0f - 2.0f * (xx + zz), 2.0f * (yz + wx)};
    r.c[2] = {2.0f * (xz + wy), 2.0f * (yz - wx), 1.0f - 2.0f * (xx + yy)};
    return r;
}
inline Quat quatFromMat3(const Mat3& m) {
    float t = m.c[0][0] + m.c[1][1] + m.c[2][2];
    Quat q;
    if (t > 0.0f) {
        float s = std::sqrt(t + 1.0f) * 2.0f;
        q.w = 0.25f * s;
        q.x = (m.c[1][2] - m.c[2][1]) / s;
        q.y = (m.c[2][0] - m.c[0][2]) / s;
        q.z = (m.c[0][1] - m.c[1][0]) / s;
    } else if (m.c[0][0] > m.c[1][1] && m.c[0][0] > m.c[2][2]) {
        float s = std::sqrt(1.0f + m.c[0][0] - m.c[1][1] - m.c[2][2]) * 2.0f;
        q.w = (m.c[1][2] - m.c[2][1]) / s;
        q.x = 0.25f * s;
        q.y = (m.c[1][0] + m.c[0][1]) / s;
        q.z = (m.c[2][0] + m.c[0][2]) / s;
    } else if (m.c[1][1] > m.c[2][2]) {
        float s = std::sqrt(1.0f + m.c[1][1] - m.c[0][0] - m.c[2][2]) * 2.0f;
        q.w = (m.c[2][0] - m.c[0][2]) / s;
        q.x = (m.c[1][0] + m.c[0][1]) / s;
        q.y = 0.25f * s;
        q.z = (m.c[2][1] + m.c[1][2]) / s;
    } else {
        float s = std::sqrt(1.0f + m.c[2][2] - m.c[0][0] - m.c[1][1]) * 2.0f;
        q.w = (m.c[0][1] - m.c[1][0]) / s;
        q.x = (m.c[2][0] + m.c[0][2]) / s;
        q.y = (m.c[2][1] + m.c[1][2]) / s;
        q.z = 0.25f * s;
    }
    return normalize(q);
}
// Shortest-arc rotation taking unit vector `from` to unit vector `to`.
inline Quat quatBetween(Vec3 from, Vec3 to) {
    from = normalize(from);
    to = normalize(to);
    float d = clamp(dot(from, to), -1.0f, 1.0f);
    if (d > 0.999999f) return Quat(0, 0, 0, 1);
    if (d < -0.999999f) {
        Vec3 axis = cross(Vec3(1, 0, 0), from);
        if (lengthSq(axis) < 1e-8f) axis = cross(Vec3(0, 1, 0), from);
        return quatFromAxisAngle(axis, PI);
    }
    Vec3 c = cross(from, to);
    return normalize(Quat(c.x, c.y, c.z, 1.0f + d));
}

// ---------------------------------------------------------------- Mat4 (column-major)
// m[col][row]; element (row r, col c) == m[c][r]. Uploaded directly to GLSL mat4.
struct Mat4 {
    Vec4 c[4] = {Vec4(1, 0, 0, 0), Vec4(0, 1, 0, 0), Vec4(0, 0, 1, 0), Vec4(0, 0, 0, 1)};
    Mat4() = default;
    Vec4&       operator[](int i)       { return c[i]; }
    const Vec4& operator[](int i) const { return c[i]; }
    float*       data()       { return &c[0].x; }
    const float* data() const { return &c[0].x; }
};
inline Mat4 operator*(const Mat4& a, const Mat4& b) {
    Mat4 r;
    for (int i = 0; i < 4; ++i) {
        Vec4 col = b.c[i];
        r.c[i] = a.c[0] * col.x + a.c[1] * col.y + a.c[2] * col.z + a.c[3] * col.w;
    }
    return r;
}
inline Vec4 operator*(const Mat4& m, Vec4 v) {
    return m.c[0] * v.x + m.c[1] * v.y + m.c[2] * v.z + m.c[3] * v.w;
}
inline Vec3 transformPoint(const Mat4& m, Vec3 p) {
    Vec4 r = m * Vec4(p, 1.0f);
    return {r.x, r.y, r.z};
}
inline Vec3 transformDirection(const Mat4& m, Vec3 d) {
    Vec4 r = m * Vec4(d, 0.0f);
    return {r.x, r.y, r.z};
}
inline Mat4 mat4Identity() { return Mat4{}; }
inline Mat4 mat4Translate(Vec3 t) {
    Mat4 m;
    m.c[3] = Vec4(t, 1.0f);
    return m;
}
inline Mat4 mat4Scale(Vec3 s) {
    Mat4 m;
    m.c[0] = {s.x, 0, 0, 0};
    m.c[1] = {0, s.y, 0, 0};
    m.c[2] = {0, 0, s.z, 0};
    return m;
}
inline Mat4 mat4FromMat3(const Mat3& r) {
    Mat4 m;
    m.c[0] = Vec4(r.c[0], 0.0f);
    m.c[1] = Vec4(r.c[1], 0.0f);
    m.c[2] = Vec4(r.c[2], 0.0f);
    return m;
}
inline Mat4 mat4FromQuat(Quat q) { return mat4FromMat3(mat3FromQuat(q)); }
inline Mat4 mat4TRS(Vec3 t, Quat r, Vec3 s) {
    Mat4 m = mat4FromQuat(r);
    m.c[0] *= s.x;
    m.c[1] *= s.y;
    m.c[2] *= s.z;
    m.c[3] = Vec4(t, 1.0f);
    return m;
}
inline Mat4 transpose(const Mat4& m) {
    Mat4 r;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) r.c[i][j] = m.c[j][i];
    return r;
}
// General 4x4 inverse. The cofactor expansion is written on a row-major copy for
// clarity, then transposed back into this type's column-major storage.
inline Mat4 inverse(const Mat4& mat) {
    double m[16];
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) m[r * 4 + c] = static_cast<double>(mat.c[c][r]);

    double inv[16];
    inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] +
             m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
    inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] -
             m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
    inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] +
             m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
    inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] -
              m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
    inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] -
             m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
    inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] +
             m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
    inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] -
             m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
    inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] +
              m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
    inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] +
             m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
    inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] -
             m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
    inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] +
              m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
    inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] -
              m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
    inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] -
             m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
    inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] +
             m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
    inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] -
              m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
    inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] +
              m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];

    const double det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
    if (std::fabs(det) < 1e-18) return Mat4{};
    const double invDet = 1.0 / det;

    Mat4 r;
    for (int row = 0; row < 4; ++row)
        for (int col = 0; col < 4; ++col)
            r.c[col][row] = static_cast<float>(inv[row * 4 + col] * invDet);
    return r;
}

inline Mat3 mat3FromMat4Upper(const Mat4& m) {
    Mat3 r;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) r.c[i][j] = m.c[i][j];
    return r;
}

// Right-handed look-at view matrix.
inline Mat4 mat4LookAt(Vec3 eye, Vec3 center, Vec3 up) {
    Vec3 f = normalize(center - eye);
    Vec3 s = normalize(cross(f, up));
    Vec3 u = cross(s, f);
    Mat4 m;
    m.c[0] = Vec4(s.x, u.x, -f.x, 0.0f);
    m.c[1] = Vec4(s.y, u.y, -f.y, 0.0f);
    m.c[2] = Vec4(s.z, u.z, -f.z, 0.0f);
    m.c[3] = Vec4(-dot(s, eye), -dot(u, eye), dot(f, eye), 1.0f);
    return m;
}

// Vulkan perspective projection, depth in [0,1], Y flipped for Vulkan clip space.
inline Mat4 mat4PerspectiveVk(float fovYRad, float aspect, float zNear, float zFar) {
    float t = std::tan(fovYRad * 0.5f);
    Mat4 m;
    m.c[0] = {1.0f / (aspect * t), 0, 0, 0};
    m.c[1] = {0, -1.0f / t, 0, 0};               // flip Y: Vulkan has +Y down in NDC
    m.c[2] = {0, 0, zFar / (zNear - zFar), -1.0f};
    m.c[3] = {0, 0, (zFar * zNear) / (zNear - zFar), 0};
    return m;
}

// Same projection without the Y flip. Required when rendering cubemap faces: the
// standard cube-map face convention already accounts for the flipped image axis, so
// applying the flip as well would mirror every face.
inline Mat4 mat4PerspectiveVkCubeFace(float fovYRad, float zNear, float zFar) {
    float t = std::tan(fovYRad * 0.5f);
    Mat4 m;
    m.c[0] = {1.0f / t, 0, 0, 0};
    m.c[1] = {0, 1.0f / t, 0, 0};
    m.c[2] = {0, 0, zFar / (zNear - zFar), -1.0f};
    m.c[3] = {0, 0, (zFar * zNear) / (zNear - zFar), 0};
    return m;
}

// Orthographic projection for shadow maps, depth in [0,1], Y flipped for Vulkan.
inline Mat4 mat4OrthoVk(float l, float r, float b, float t, float n, float f) {
    Mat4 m;
    m.c[0] = {2.0f / (r - l), 0, 0, 0};
    m.c[1] = {0, -2.0f / (t - b), 0, 0};
    m.c[2] = {0, 0, 1.0f / (n - f), 0};
    m.c[3] = {-(r + l) / (r - l), (t + b) / (t - b), n / (n - f), 1.0f};
    return m;
}

inline Mat4 mat4InfinitePerspectiveVk(float fovYRad, float aspect, float zNear) {
    float t = std::tan(fovYRad * 0.5f);
    Mat4 m;
    m.c[0] = {1.0f / (aspect * t), 0, 0, 0};
    m.c[1] = {0, -1.0f / t, 0, 0};
    m.c[2] = {0, 0, -1.0f, -1.0f};
    m.c[3] = {0, 0, -zNear, 0};
    return m;
}

// ---------------------------------------------------------------- misc helpers
struct Ray {
    Vec3 origin;
    Vec3 dir;  // expected normalized
};
inline Vec3 rayAt(const Ray& r, float t) { return r.origin + r.dir * t; }

struct Aabb {
    Vec3 mn{1e30f, 1e30f, 1e30f};
    Vec3 mx{-1e30f, -1e30f, -1e30f};
    void expand(Vec3 p) { mn = minv(mn, p); mx = maxv(mx, p); }
    void expand(const Aabb& o) { mn = minv(mn, o.mn); mx = maxv(mx, o.mx); }
    Vec3 center() const { return (mn + mx) * 0.5f; }
    Vec3 extent() const { return (mx - mn) * 0.5f; }
    bool valid() const { return mx.x >= mn.x && mx.y >= mn.y && mx.z >= mn.z; }
    bool overlaps(const Aabb& o) const {
        return mn.x <= o.mx.x && mx.x >= o.mn.x && mn.y <= o.mx.y && mx.y >= o.mn.y &&
               mn.z <= o.mx.z && mx.z >= o.mn.z;
    }
    bool contains(Vec3 p) const {
        return p.x >= mn.x && p.x <= mx.x && p.y >= mn.y && p.y <= mx.y && p.z >= mn.z &&
               p.z <= mx.z;
    }
};

// Ray/AABB slab test; returns true and sets tHit to the near hit distance.
inline bool rayAabb(const Ray& r, const Aabb& b, float& tHit) {
    float t0 = -1e30f, t1 = 1e30f;
    for (int i = 0; i < 3; ++i) {
        float d = r.dir[i];
        if (std::fabs(d) < 1e-9f) {
            if (r.origin[i] < b.mn[i] || r.origin[i] > b.mx[i]) return false;
        } else {
            float inv = 1.0f / d;
            float ta = (b.mn[i] - r.origin[i]) * inv;
            float tb = (b.mx[i] - r.origin[i]) * inv;
            if (ta > tb) std::swap(ta, tb);
            t0 = max2(t0, ta);
            t1 = min2(t1, tb);
            if (t0 > t1) return false;
        }
    }
    tHit = t0 >= 0.0f ? t0 : t1;
    return t1 >= 0.0f;
}

// Moller-Trumbore ray/triangle intersection (both faces).
inline bool rayTriangle(const Ray& r, Vec3 v0, Vec3 v1, Vec3 v2, float& t, float* uOut = nullptr,
                        float* vOut = nullptr) {
    Vec3 e1 = v1 - v0, e2 = v2 - v0;
    Vec3 p = cross(r.dir, e2);
    float det = dot(e1, p);
    if (std::fabs(det) < 1e-12f) return false;
    float inv = 1.0f / det;
    Vec3 tv = r.origin - v0;
    float u = dot(tv, p) * inv;
    if (u < -1e-6f || u > 1.0f + 1e-6f) return false;
    Vec3 q = cross(tv, e1);
    float v = dot(r.dir, q) * inv;
    if (v < -1e-6f || u + v > 1.0f + 1e-6f) return false;
    float tt = dot(e2, q) * inv;
    if (tt < 1e-5f) return false;
    t = tt;
    if (uOut) *uOut = u;
    if (vOut) *vOut = v;
    return true;
}

// Ray/sphere intersection; returns nearest positive t.
inline bool raySphere(const Ray& r, Vec3 center, float radius, float& t) {
    Vec3 oc = r.origin - center;
    float b = dot(oc, r.dir);
    float c = dot(oc, oc) - radius * radius;
    float disc = b * b - c;
    if (disc < 0.0f) return false;
    float sq = std::sqrt(disc);
    float t0 = -b - sq;
    float t1 = -b + sq;
    t = t0 >= 1e-5f ? t0 : t1;
    return t >= 1e-5f;
}

// ---------------------------------------------------------------- packing for GPU
struct Vertex {
    Vec3 position;
    Vec3 normal;
    Vec4 tangent;   // xyz = tangent, w = bitangent sign
    Vec2 uv;
    Vec2 uv1;       // secondary UV (lightmap / detail)
};

// ---------------------------------------------------------------- color helpers
inline Vec3 srgbToLinear(Vec3 c) {
    auto f = [](float v) {
        return v <= 0.04045f ? v / 12.92f : std::pow((v + 0.055f) / 1.055f, 2.4f);
    };
    return {f(c.x), f(c.y), f(c.z)};
}
inline Vec3 linearToSrgb(Vec3 c) {
    auto f = [](float v) {
        return v <= 0.0031308f ? v * 12.92f : 1.055f * std::pow(v, 1.0f / 2.4f) - 0.055f;
    };
    return {f(c.x), f(c.y), f(c.z)};
}
inline Vec3 hexColor(uint32_t rgb) {
    return {float((rgb >> 16) & 0xFF) / 255.0f, float((rgb >> 8) & 0xFF) / 255.0f,
            float(rgb & 0xFF) / 255.0f};
}

}  // namespace room2
