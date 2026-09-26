package main

import "math"

// Vec2 is a 2D vector.
type Vec2 struct{ X, Y float64 }

// Vec3 is a 3D vector / point.
type Vec3 struct{ X, Y, Z float64 }

func V3(x, y, z float64) Vec3 { return Vec3{x, y, z} }

func (a Vec3) Add(b Vec3) Vec3      { return Vec3{a.X + b.X, a.Y + b.Y, a.Z + b.Z} }
func (a Vec3) Sub(b Vec3) Vec3      { return Vec3{a.X - b.X, a.Y - b.Y, a.Z - b.Z} }
func (a Vec3) Mul(s float64) Vec3   { return Vec3{a.X * s, a.Y * s, a.Z * s} }
func (a Vec3) MulV(b Vec3) Vec3     { return Vec3{a.X * b.X, a.Y * b.Y, a.Z * b.Z} }
func (a Vec3) Div(s float64) Vec3   { return Vec3{a.X / s, a.Y / s, a.Z / s} }
func (a Vec3) Neg() Vec3            { return Vec3{-a.X, -a.Y, -a.Z} }
func (a Vec3) Dot(b Vec3) float64   { return a.X*b.X + a.Y*b.Y + a.Z*b.Z }
func (a Vec3) LenSq() float64       { return a.Dot(a) }
func (a Vec3) Len() float64         { return math.Sqrt(a.Dot(a)) }
func (a Vec3) Dist(b Vec3) float64  { return a.Sub(b).Len() }
func (a Vec3) Lerp(b Vec3, t float64) Vec3 {
	return Vec3{a.X + (b.X-a.X)*t, a.Y + (b.Y-a.Y)*t, a.Z + (b.Z-a.Z)*t}
}

func (a Vec3) Cross(b Vec3) Vec3 {
	return Vec3{
		a.Y*b.Z - a.Z*b.Y,
		a.Z*b.X - a.X*b.Z,
		a.X*b.Y - a.Y*b.X,
	}
}

func (a Vec3) Normalize() Vec3 {
	l := a.Len()
	if l < 1e-12 {
		return Vec3{0, 0, 0}
	}
	return a.Div(l)
}

// Clamp limits v to [lo, hi].
func Clamp(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// Clamp01 limits v to [0, 1].
func Clamp01(v float64) float64 { return Clamp(v, 0, 1) }

// Lerp interpolates between a and b.
func Lerp(a, b, t float64) float64 { return a + (b-a)*t }

// Smoothstep is the classic hermite ramp.
func Smoothstep(t float64) float64 {
	t = Clamp01(t)
	return t * t * (3 - 2*t)
}

// EaseOutCubic decelerating ramp.
func EaseOutCubic(t float64) float64 {
	t = Clamp01(t)
	u := 1 - t
	return 1 - u*u*u
}

// EaseInOutCubic is a smooth accelerate/decelerate ramp.
func EaseInOutCubic(t float64) float64 {
	t = Clamp01(t)
	if t < 0.5 {
		return 4 * t * t * t
	}
	u := -2*t + 2
	return 1 - u*u*u/2
}

// Degrees converts degrees to radians.
func Degrees(d float64) float64 { return d * math.Pi / 180 }

// Mat4 is a column-major 4x4 matrix (m[col*4+row]), compatible with OpenGL layout.
type Mat4 [16]float64

// Identity4 returns the identity matrix.
func Identity4() Mat4 {
	return Mat4{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}
}

// Mul returns a*b (apply b first, then a).
func (a Mat4) Mul(b Mat4) Mat4 {
	var o Mat4
	for c := 0; c < 4; c++ {
		for r := 0; r < 4; r++ {
			var s float64
			for k := 0; k < 4; k++ {
				s += a[k*4+r] * b[c*4+k]
			}
			o[c*4+r] = s
		}
	}
	return o
}

// MulVec4 transforms a 4-component vector.
func (m Mat4) MulVec4(v [4]float64) [4]float64 {
	var o [4]float64
	for r := 0; r < 4; r++ {
		o[r] = m[r]*v[0] + m[4+r]*v[1] + m[8+r]*v[2] + m[12+r]*v[3]
	}
	return o
}

// MulPoint transforms a point (w=1).
func (m Mat4) MulPoint(p Vec3) Vec3 {
	o := m.MulVec4([4]float64{p.X, p.Y, p.Z, 1})
	return Vec3{o[0], o[1], o[2]}
}

// MulDir transforms a direction (w=0).
func (m Mat4) MulDir(d Vec3) Vec3 {
	o := m.MulVec4([4]float64{d.X, d.Y, d.Z, 0})
	return Vec3{o[0], o[1], o[2]}
}

// Perspective builds a right-handed perspective projection matrix looking down -Z.
func Perspective(fovyRad, aspect, near, far float64) Mat4 {
	f := 1 / math.Tan(fovyRad/2)
	var m Mat4
	m[0] = f / aspect
	m[5] = f
	m[10] = (far + near) / (near - far)
	m[11] = -1
	m[14] = 2 * far * near / (near - far)
	return m
}

// LookAt builds a right-handed view matrix.
func LookAt(eye, center, up Vec3) Mat4 {
	f := center.Sub(eye).Normalize()
	s := f.Cross(up).Normalize()
	u := s.Cross(f)
	return Mat4{
		s.X, u.X, -f.X, 0,
		s.Y, u.Y, -f.Y, 0,
		s.Z, u.Z, -f.Z, 0,
		-s.Dot(eye), -u.Dot(eye), f.Dot(eye), 1,
	}
}

// Translate returns a translation matrix.
func Translate(t Vec3) Mat4 {
	m := Identity4()
	m[12], m[13], m[14] = t.X, t.Y, t.Z
	return m
}

// Scale returns a scaling matrix.
func Scale(s Vec3) Mat4 {
	m := Identity4()
	m[0], m[5], m[10] = s.X, s.Y, s.Z
	return m
}

// RotX returns a rotation about X.
func RotX(a float64) Mat4 {
	c, s := math.Cos(a), math.Sin(a)
	m := Identity4()
	m[5], m[6], m[9], m[10] = c, s, -s, c
	return m
}

// RotY returns a rotation about Y.
func RotY(a float64) Mat4 {
	c, s := math.Cos(a), math.Sin(a)
	m := Identity4()
	m[0], m[2], m[8], m[10] = c, -s, s, c
	return m
}

// RotZ returns a rotation about Z.
func RotZ(a float64) Mat4 {
	c, s := math.Cos(a), math.Sin(a)
	m := Identity4()
	m[0], m[1], m[4], m[5] = c, s, -s, c
	return m
}

// Euler builds a YXZ rotation matrix (yaw, pitch, roll).
func Euler(yaw, pitch, roll float64) Mat4 {
	return RotY(yaw).Mul(RotX(pitch)).Mul(RotZ(roll))
}

// TRS composes translate*rotate*scale.
func TRS(t Vec3, yaw, pitch, roll float64, s Vec3) Mat4 {
	return Translate(t).Mul(Euler(yaw, pitch, roll)).Mul(Scale(s))
}

// AABB is an axis-aligned box used for collision and hitscan tests.
type AABB struct {
	Min, Max Vec3
}

// Box builds an AABB from center and half extents.
func Box(center, half Vec3) AABB {
	return AABB{Min: center.Sub(half), Max: center.Add(half)}
}

func (b AABB) Center() Vec3 { return b.Min.Add(b.Max).Mul(0.5) }
func (b AABB) Size() Vec3   { return b.Max.Sub(b.Min) }

// Contains reports whether p is inside the box.
func (b AABB) Contains(p Vec3) bool {
	return p.X >= b.Min.X && p.X <= b.Max.X &&
		p.Y >= b.Min.Y && p.Y <= b.Max.Y &&
		p.Z >= b.Min.Z && p.Z <= b.Max.Z
}

// RayHit describes a ray/geometry intersection.
type RayHit struct {
	Hit  bool
	T    float64
	P    Vec3
	N    Vec3
	Kind int // hitKind* constant
}

// hit kinds
const (
	hitNone = iota
	hitWall
	hitFloor
	hitCeiling
	hitTable
	hitProp
	hitGlass
	hitWater
	hitShard
)

// rayAABB returns the entry distance of a ray into an AABB (slab method).
func rayAABB(ro, rd Vec3, b AABB) (float64, bool) {
	tmin := math.Inf(-1)
	tmax := math.Inf(1)
	for axis := 0; axis < 3; axis++ {
		var o, d, lo, hi float64
		switch axis {
		case 0:
			o, d, lo, hi = ro.X, rd.X, b.Min.X, b.Max.X
		case 1:
			o, d, lo, hi = ro.Y, rd.Y, b.Min.Y, b.Max.Y
		default:
			o, d, lo, hi = ro.Z, rd.Z, b.Min.Z, b.Max.Z
		}
		if math.Abs(d) < 1e-9 {
			if o < lo || o > hi {
				return 0, false
			}
			continue
		}
		inv := 1 / d
		t1 := (lo - o) * inv
		t2 := (hi - o) * inv
		if t1 > t2 {
			t1, t2 = t2, t1
		}
		if t1 > tmin {
			tmin = t1
		}
		if t2 < tmax {
			tmax = t2
		}
		if tmin > tmax {
			return 0, false
		}
	}
	if tmax < 0 {
		return 0, false
	}
	if tmin < 0 {
		return 0, true
	}
	return tmin, true
}

// aabbNormal returns the outward normal of the face closest to the hit point.
func aabbNormal(b AABB, p Vec3) Vec3 {
	c := b.Center()
	d := p.Sub(c)
	e := b.Size().Mul(0.5)
	ax, ay, az := math.Abs(d.X)/math.Max(e.X, 1e-9), math.Abs(d.Y)/math.Max(e.Y, 1e-9), math.Abs(d.Z)/math.Max(e.Z, 1e-9)
	switch {
	case ax >= ay && ax >= az:
		return V3(math.Copysign(1, d.X), 0, 0)
	case ay >= az:
		return V3(0, math.Copysign(1, d.Y), 0)
	default:
		return V3(0, 0, math.Copysign(1, d.Z))
	}
}

// rayCylinder intersects a ray with a Y-axis cylinder between y0 and y1.
// Returns entry distance and the surface normal at the hit.
func rayCylinder(ro, rd Vec3, cx, cz, radius, y0, y1 float64) (float64, Vec3, bool) {
	ox, oz := ro.X-cx, ro.Z-cz
	a := rd.X*rd.X + rd.Z*rd.Z
	if a < 1e-12 {
		return 0, Vec3{}, false
	}
	b := 2 * (ox*rd.X + oz*rd.Z)
	c := ox*ox + oz*oz - radius*radius
	disc := b*b - 4*a*c
	if disc < 0 {
		return 0, Vec3{}, false
	}
	sq := math.Sqrt(disc)
	for _, t := range [2]float64{(-b - sq) / (2 * a), (-b + sq) / (2 * a)} {
		if t <= 1e-4 {
			continue
		}
		y := ro.Y + rd.Y*t
		if y < y0 || y > y1 {
			continue
		}
		p := ro.Add(rd.Mul(t))
		n := V3(p.X-cx, 0, p.Z-cz).Normalize()
		return t, n, true
	}
	return 0, Vec3{}, false
}

// segAABB reports whether the segment a→b intersects the box.
func segAABB(a, b Vec3, box AABB) bool {
	d := b.Sub(a)
	_, ok := rayAABB(a, d, box)
	if !ok {
		return false
	}
	t, ok := rayAABB(a, d, box)
	return ok && t <= 1.0
}
