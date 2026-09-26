// room2 - first person camera with frustum culling and TAA jitter support.
#pragma once

#include <array>

#include "core/math.hpp"

namespace room2::render {

struct Frustum {
    // Planes point inwards: a point p is inside when dot(n, p) + d >= 0 for all planes.
    struct Plane {
        Vec3 n{0, 1, 0};
        float d = 0.0f;
        float distance(Vec3 p) const { return dot(n, p) + d; }
    };
    std::array<Plane, 6> planes{};

    bool containsPoint(Vec3 p) const {
        for (const auto& pl : planes)
            if (pl.distance(p) < 0.0f) return false;
        return true;
    }
    // Conservative AABB test (positive vertex).
    bool intersectsAabb(const Aabb& box) const {
        for (const auto& pl : planes) {
            Vec3 positive{pl.n.x >= 0.0f ? box.mx.x : box.mn.x, pl.n.y >= 0.0f ? box.mx.y : box.mn.y,
                          pl.n.z >= 0.0f ? box.mx.z : box.mn.z};
            if (pl.distance(positive) < 0.0f) return false;
        }
        return true;
    }
    bool intersectsSphere(Vec3 center, float radius) const {
        for (const auto& pl : planes)
            if (pl.distance(center) < -radius) return false;
        return true;
    }
};

class Camera {
public:
    void setPerspective(float fovYRadians, float aspect, float zNear, float zFar) {
        fovY_ = fovYRadians;
        aspect_ = aspect;
        zNear_ = zNear;
        zFar_ = zFar;
        projDirty_ = true;
    }
    void setView(Vec3 eye, Vec3 forward, Vec3 up) {
        position_ = eye;
        forward_ = normalize(forward);
        right_ = normalize(cross(forward_, up));
        up_ = cross(right_, forward_);
        viewDirty_ = true;
    }
    // Builds the view basis from yaw/pitch (radians) with no roll.
    void setFromYawPitch(Vec3 eye, float yaw, float pitch) {
        forward_ = normalize(Vec3(std::cos(pitch) * std::sin(yaw), std::sin(pitch),
                                  -std::cos(pitch) * std::cos(yaw)));
        right_ = normalize(cross(forward_, Vec3(0, 1, 0)));
        up_ = cross(right_, forward_);
        position_ = eye;
        viewDirty_ = true;
    }

    const Mat4& view() const {
        if (viewDirty_) {
            view_ = mat4LookAt(position_, position_ + forward_, up_);
            viewDirty_ = false;
            frustumDirty_ = true;
        }
        return view_;
    }
    const Mat4& projection() const {
        if (projDirty_) {
            proj_ = mat4PerspectiveVk(fovY_, aspect_, zNear_, zFar_);
            projDirty_ = false;
            frustumDirty_ = true;
        }
        return proj_;
    }
    Mat4 viewProjection() const { return projection() * view(); }
    // View-projection with a sub-pixel jitter applied for temporal antialiasing.
    Mat4 jitteredViewProjection() const {
        Mat4 p = projection();
        p.c[2][0] += jitter_.x * 2.0f / static_cast<float>(viewportWidth_);
        p.c[2][1] += jitter_.y * 2.0f / static_cast<float>(viewportHeight_);
        return p * view();
    }
    Mat4 projectionUnjittered() const { return projection(); }

    void setJitter(Vec2 pixels) { jitter_ = pixels; }
    Vec2 jitter() const { return jitter_; }
    void setViewportSize(uint32_t w, uint32_t h) {
        viewportWidth_ = w;
        viewportHeight_ = h;
    }

    Vec3 position() const { return position_; }
    Vec3 forward() const { return forward_; }
    Vec3 right() const { return right_; }
    Vec3 up() const { return up_; }
    float fovY() const { return fovY_; }
    float aspect() const { return aspect_; }
    float zNear() const { return zNear_; }
    float zFar() const { return zFar_; }

    const Frustum& frustum() const {
        if (frustumDirty_) {
            buildFrustum();
            frustumDirty_ = false;
        }
        return frustum_;
    }

    // Ray through a normalised device coordinate in [-1,1] (x right, y up).
    Ray rayFromNdc(Vec2 ndc) const {
        float tanHalf = std::tan(fovY_ * 0.5f);
        Vec3 dir = normalize(forward_ + right_ * (ndc.x * tanHalf * aspect_) + up_ * (ndc.y * tanHalf));
        return {position_, dir};
    }
    // Approximate world size of one pixel at the given distance.
    float pixelWorldSize(float distance) const {
        return 2.0f * std::tan(fovY_ * 0.5f) * distance / static_cast<float>(viewportHeight_);
    }

private:
    void buildFrustum() const {
        const Mat4 vp = viewProjection();
        // Rows of vp (row i, column j) == vp.c[j][i].
        auto row = [&vp](int i) { return Vec4(vp.c[0][i], vp.c[1][i], vp.c[2][i], vp.c[3][i]); };
        const Vec4 r0 = row(0), r1 = row(1), r2 = row(2), r3 = row(3);
        // Vulkan clip space: -w <= x <= w, -w <= y <= w, 0 <= z <= w.
        Vec4 planes[6] = {
            r3 + r0,  // left
            r3 - r0,  // right
            r3 + r1,  // bottom
            r3 - r1,  // top
            r2,       // near (z >= 0)
            r3 - r2,  // far  (z <= w)
        };
        for (int i = 0; i < 6; ++i) {
            Vec3 n{planes[i].x, planes[i].y, planes[i].z};
            float len = length(n);
            if (len < 1e-9f) len = 1.0f;
            frustum_.planes[i].n = n / len;
            frustum_.planes[i].d = planes[i].w / len;
        }
    }

    Vec3 position_{0, 0, 0};
    Vec3 forward_{0, 0, -1};
    Vec3 right_{1, 0, 0};
    Vec3 up_{0, 1, 0};
    float fovY_ = 60.0f * DEG2RAD;
    float aspect_ = 16.0f / 9.0f;
    float zNear_ = 0.02f;
    float zFar_ = 200.0f;
    Vec2 jitter_{0, 0};
    uint32_t viewportWidth_ = 1920;
    uint32_t viewportHeight_ = 1080;

    mutable Mat4 view_;
    mutable Mat4 proj_;
    mutable Frustum frustum_;
    mutable bool viewDirty_ = true;
    mutable bool projDirty_ = true;
    mutable bool frustumDirty_ = true;
};

}  // namespace room2::render
