package main

import "math"

// Procedural texture ids used by the shader.
const (
	TexFlat = iota
	TexPlanks
	TexPlaster
	TexWood
	TexRug
	TexCanvasArt
	TexGlow
	TexTile
	TexMetal
	TexPaper
)

// Material describes a surface for the software shader.
type Material struct {
	Albedo      Vec3
	Emissive    Vec3
	Spec        float64 // specular strength 0..1
	Shininess   float64 // Blinn-Phong exponent
	Alpha       float64 // 1 = opaque
	Tex         int
	DoubleSided bool
	Unlit       bool
	Additive    bool
	Refl        float64 // fake reflection amount (used by glass)
}

// Opaque reports whether the material can go through the depth-writing pass.
func (m Material) Opaque() bool { return m.Alpha >= 0.999 }

// Vertex is a mesh vertex. A is a per-vertex opacity multiplier where 0 means
// "opaque" (the default), so only fading sprites need to set it.
type Vertex struct {
	Pos Vec3
	Nrm Vec3
	U   float64
	V   float64
	A   float64
}

// Mesh is an indexed triangle list.
type Mesh struct {
	Verts []Vertex
	Idx   []uint32
}

func newMesh() *Mesh { return &Mesh{} }

// AddTri appends a triangle, computing the face normal when it is missing. When
// explicit normals are supplied the winding is flipped if needed so the
// geometric (culling) orientation always agrees with the shading normal.
func (m *Mesh) AddTri(a, b, c Vertex) {
	geo := b.Pos.Sub(a.Pos).Cross(c.Pos.Sub(a.Pos))
	if geo.LenSq() > 1e-18 {
		ref := a.Nrm
		if ref.LenSq() < 1e-18 {
			ref = b.Nrm
		}
		if ref.LenSq() < 1e-18 {
			ref = c.Nrm
		}
		if ref.LenSq() > 1e-18 && ref.Normalize().Dot(geo.Normalize()) < -0.3 {
			b, c = c, b
			geo = geo.Neg()
		}
	}
	n := a.Nrm
	if n.LenSq() < 1e-12 {
		n = geo.Normalize()
	}
	if a.Nrm.LenSq() < 1e-12 {
		a.Nrm = n
	}
	if b.Nrm.LenSq() < 1e-12 {
		b.Nrm = n
	}
	if c.Nrm.LenSq() < 1e-12 {
		c.Nrm = n
	}
	base := uint32(len(m.Verts))
	m.Verts = append(m.Verts, a, b, c)
	m.Idx = append(m.Idx, base, base+1, base+2)
}

// AddQuad appends a quad (a,b,c,d in CCW order) as two triangles.
func (m *Mesh) AddQuad(a, b, c, d Vertex) {
	m.AddTri(a, b, c)
	m.AddTri(a, c, d)
}

// VertexCount reports the number of vertices.
func (m *Mesh) VertexCount() int { return len(m.Verts) }

// TriCount reports the number of triangles.
func (m *Mesh) TriCount() int { return len(m.Idx) / 3 }

// Transform applies a rigid transform (rotation+translation) in place.
func (m *Mesh) Transform(mat Mat4) *Mesh {
	for i := range m.Verts {
		v := &m.Verts[i]
		v.Pos = mat.MulPoint(v.Pos)
		v.Nrm = mat.MulDir(v.Nrm).Normalize()
	}
	return m
}

// Transformed returns a transformed copy.
func (m *Mesh) Transformed(mat Mat4) *Mesh {
	out := &Mesh{Verts: make([]Vertex, len(m.Verts)), Idx: append([]uint32(nil), m.Idx...)}
	copy(out.Verts, m.Verts)
	return out.Transform(mat)
}

// Append merges another mesh into this one.
func (m *Mesh) Append(o *Mesh) *Mesh {
	base := uint32(len(m.Verts))
	m.Verts = append(m.Verts, o.Verts...)
	for _, i := range o.Idx {
		m.Idx = append(m.Idx, i+base)
	}
	return m
}

// Bounds returns the AABB of the mesh.
func (m *Mesh) Bounds() AABB {
	if len(m.Verts) == 0 {
		return AABB{}
	}
	b := AABB{Min: m.Verts[0].Pos, Max: m.Verts[0].Pos}
	for _, v := range m.Verts[1:] {
		b.Min.X = math.Min(b.Min.X, v.Pos.X)
		b.Min.Y = math.Min(b.Min.Y, v.Pos.Y)
		b.Min.Z = math.Min(b.Min.Z, v.Pos.Z)
		b.Max.X = math.Max(b.Max.X, v.Pos.X)
		b.Max.Y = math.Max(b.Max.Y, v.Pos.Y)
		b.Max.Z = math.Max(b.Max.Z, v.Pos.Z)
	}
	return b
}

// quadUV builds a vertex with an explicit normal and uv.
func qv(p Vec3, n Vec3, u, v float64) Vertex { return Vertex{Pos: p, Nrm: n, U: u, V: v, A: 1} }

// AddQuadN appends a quad with explicit vertices and per-corner UVs.
func (m *Mesh) AddQuadN(a, b, c, d Vec3, n Vec3, uv [4]Vec2) {
	m.AddQuad(
		qv(a, n, uv[0].X, uv[0].Y),
		qv(b, n, uv[1].X, uv[1].Y),
		qv(c, n, uv[2].X, uv[2].Y),
		qv(d, n, uv[3].X, uv[3].Y),
	)
}

// AddBoxUV appends an axis-aligned box whose UVs are the two tangent world axes
// scaled by 1/uvScale (so tiling is uniform across the whole scene).
func (m *Mesh) AddBoxUV(b AABB, uvScale float64) {
	s := 1 / uvScale
	min, max := b.Min, b.Max
	// -Z face
	m.AddQuadN(
		V3(min.X, min.Y, min.Z), V3(max.X, min.Y, min.Z), V3(max.X, max.Y, min.Z), V3(min.X, max.Y, min.Z),
		V3(0, 0, -1), [4]Vec2{{min.X * s, min.Y * s}, {max.X * s, min.Y * s}, {max.X * s, max.Y * s}, {min.X * s, max.Y * s}})
	// +Z face
	m.AddQuadN(
		V3(max.X, min.Y, max.Z), V3(min.X, min.Y, max.Z), V3(min.X, max.Y, max.Z), V3(max.X, max.Y, max.Z),
		V3(0, 0, 1), [4]Vec2{{max.X * s, min.Y * s}, {min.X * s, min.Y * s}, {min.X * s, max.Y * s}, {max.X * s, max.Y * s}})
	// -X face
	m.AddQuadN(
		V3(min.X, min.Y, max.Z), V3(min.X, min.Y, min.Z), V3(min.X, max.Y, min.Z), V3(min.X, max.Y, max.Z),
		V3(-1, 0, 0), [4]Vec2{{max.Z * s, min.Y * s}, {min.Z * s, min.Y * s}, {min.Z * s, max.Y * s}, {max.Z * s, max.Y * s}})
	// +X face
	m.AddQuadN(
		V3(max.X, min.Y, min.Z), V3(max.X, min.Y, max.Z), V3(max.X, max.Y, max.Z), V3(max.X, max.Y, min.Z),
		V3(1, 0, 0), [4]Vec2{{min.Z * s, min.Y * s}, {max.Z * s, min.Y * s}, {max.Z * s, max.Y * s}, {min.Z * s, max.Y * s}})
	// +Y face (up)
	m.AddQuadN(
		V3(min.X, max.Y, min.Z), V3(max.X, max.Y, min.Z), V3(max.X, max.Y, max.Z), V3(min.X, max.Y, max.Z),
		V3(0, 1, 0), [4]Vec2{{min.X * s, min.Z * s}, {max.X * s, min.Z * s}, {max.X * s, max.Z * s}, {min.X * s, max.Z * s}})
	// -Y face (down)
	m.AddQuadN(
		V3(min.X, min.Y, max.Z), V3(max.X, min.Y, max.Z), V3(max.X, min.Y, min.Z), V3(min.X, min.Y, min.Z),
		V3(0, -1, 0), [4]Vec2{{min.X * s, max.Z * s}, {max.X * s, max.Z * s}, {max.X * s, min.Z * s}, {min.X * s, min.Z * s}})
}

// AddBox adds an axis-aligned box with a default UV scale.
func (m *Mesh) AddBox(b AABB) { m.AddBoxUV(b, 1) }

// AddBoxAt adds a box from center and half extents.
func (m *Mesh) AddBoxAt(center, half Vec3, uvScale float64) {
	m.AddBoxUV(Box(center, half), uvScale)
}

// AddPlaneY appends a horizontal quad facing up (+Y) or down.
func (m *Mesh) AddPlaneY(center Vec3, hx, hz float64, up bool, uvScale float64) {
	s := 1 / uvScale
	a := center.Add(V3(-hx, 0, -hz))
	b := center.Add(V3(hx, 0, -hz))
	c := center.Add(V3(hx, 0, hz))
	d := center.Add(V3(-hx, 0, hz))
	n := V3(0, 1, 0)
	if up {
		m.AddQuadN(a, b, c, d, n, [4]Vec2{{a.X * s, a.Z * s}, {b.X * s, b.Z * s}, {c.X * s, c.Z * s}, {d.X * s, d.Z * s}})
	} else {
		n = V3(0, -1, 0)
		m.AddQuadN(d, c, b, a, n, [4]Vec2{{d.X * s, d.Z * s}, {c.X * s, c.Z * s}, {b.X * s, b.Z * s}, {a.X * s, a.Z * s}})
	}
}

// LathePoint is one sample of a surface-of-revolution profile.
type LathePoint struct {
	R, Y float64
}

// AddLathe revolves a profile (in the XY half-plane, x=r>=0) around the Y axis.
// Normals are smooth, derived from the profile tangent. u wraps 0..1 around the
// axis, v maps 0..1 along the profile.
func (m *Mesh) AddLathe(profile []LathePoint, segments int, vScale float64) {
	n := len(profile)
	if n < 2 || segments < 3 {
		return
	}
	// Precompute normals in the (r,y) plane.
	type pn struct {
		r, y   float64
		nr, ny float64
	}
	pts := make([]pn, n)
	for i := 0; i < n; i++ {
		var dr, dy float64
		if i == 0 {
			dr = profile[1].R - profile[0].R
			dy = profile[1].Y - profile[0].Y
		} else if i == n-1 {
			dr = profile[n-1].R - profile[n-2].R
			dy = profile[n-1].Y - profile[n-2].Y
		} else {
			dr = profile[i+1].R - profile[i-1].R
			dy = profile[i+1].Y - profile[i-1].Y
		}
		nr, ny := dy, -dr
		l := math.Hypot(nr, ny)
		if l < 1e-12 {
			nr, ny = 1, 0
		} else {
			nr, ny = nr/l, ny/l
		}
		// Keep normals pointing away from the axis for outward-facing profiles.
		pts[i] = pn{profile[i].R, profile[i].Y, nr, ny}
	}
	// Arc-length parameterisation for v.
	vl := make([]float64, n)
	for i := 1; i < n; i++ {
		vl[i] = vl[i-1] + math.Hypot(profile[i].R-profile[i-1].R, profile[i].Y-profile[i-1].Y)
	}
	total := vl[n-1]
	if total < 1e-9 {
		total = 1
	}

	for s := 0; s < segments; s++ {
		a0 := 2 * math.Pi * float64(s) / float64(segments)
		a1 := 2 * math.Pi * float64(s+1) / float64(segments)
		c0, s0 := math.Cos(a0), math.Sin(a0)
		c1, s1 := math.Cos(a1), math.Sin(a1)
		u0 := float64(s) / float64(segments)
		u1 := float64(s+1) / float64(segments)
		for i := 0; i < n-1; i++ {
			p0, p1 := pts[i], pts[i+1]
			v0 := vl[i] / total * vScale
			v1 := vl[i+1] / total * vScale
			A := qv(V3(p0.r*c0, p0.y, p0.r*s0), V3(p0.nr*c0, p0.ny, p0.nr*s0), u0, v0)
			B := qv(V3(p1.r*c0, p1.y, p1.r*s0), V3(p1.nr*c0, p1.ny, p1.nr*s0), u0, v1)
			C := qv(V3(p1.r*c1, p1.y, p1.r*s1), V3(p1.nr*c1, p1.ny, p1.nr*s1), u1, v1)
			D := qv(V3(p0.r*c1, p0.y, p0.r*s1), V3(p0.nr*c1, p0.ny, p0.nr*s1), u1, v0)
			m.AddQuad(A, B, C, D)
		}
	}
}

// AddDiscY appends a flat disc at height y facing up (or down), centred on (cx,cz).
func (m *Mesh) AddDiscY(cx, cz, y, radius float64, segments int, up bool) {
	n := V3(0, 1, 0)
	if !up {
		n = V3(0, -1, 0)
	}
	c := qv(V3(cx, y, cz), n, 0.5, 0.5)
	for s := 0; s < segments; s++ {
		a0 := 2 * math.Pi * float64(s) / float64(segments)
		a1 := 2 * math.Pi * float64(s+1) / float64(segments)
		p0 := V3(cx+radius*math.Cos(a0), y, cz+radius*math.Sin(a0))
		p1 := V3(cx+radius*math.Cos(a1), y, cz+radius*math.Sin(a1))
		v0 := qv(p0, n, 0.5+0.5*math.Cos(a0), 0.5+0.5*math.Sin(a0))
		v1 := qv(p1, n, 0.5+0.5*math.Cos(a1), 0.5+0.5*math.Sin(a1))
		if up {
			m.AddTri(c, v0, v1)
		} else {
			m.AddTri(c, v1, v0)
		}
	}
}

// AddRingY appends a flat annulus at height y (used for puddles and decals).
func (m *Mesh) AddRingY(cx, cz, y, r0, r1 float64, segments int, up bool) {
	n := V3(0, 1, 0)
	if !up {
		n = V3(0, -1, 0)
	}
	for s := 0; s < segments; s++ {
		a0 := 2 * math.Pi * float64(s) / float64(segments)
		a1 := 2 * math.Pi * float64(s+1) / float64(segments)
		c0, s0 := math.Cos(a0), math.Sin(a0)
		c1, s1 := math.Cos(a1), math.Sin(a1)
		A := qv(V3(cx+r0*c0, y, cz+r0*s0), n, 0, 0)
		B := qv(V3(cx+r1*c0, y, cz+r1*s0), n, 1, 0)
		C := qv(V3(cx+r1*c1, y, cz+r1*s1), n, 1, 1)
		D := qv(V3(cx+r0*c1, y, cz+r0*s1), n, 0, 1)
		if up {
			m.AddQuad(A, B, C, D)
		} else {
			m.AddQuad(A, D, C, B)
		}
	}
}

// AddSphere appends a UV sphere.
func (m *Mesh) AddSphere(center Vec3, radius float64, seg, rings int) {
	for i := 0; i < rings; i++ {
		phi0 := math.Pi * float64(i) / float64(rings)
		phi1 := math.Pi * float64(i+1) / float64(rings)
		for j := 0; j < seg; j++ {
			th0 := 2 * math.Pi * float64(j) / float64(seg)
			th1 := 2 * math.Pi * float64(j+1) / float64(seg)
			p := func(phi, th float64) Vertex {
				n := V3(math.Sin(phi)*math.Cos(th), math.Cos(phi), math.Sin(phi)*math.Sin(th))
				return Vertex{Pos: center.Add(n.Mul(radius)), Nrm: n, U: th / (2 * math.Pi), V: phi / math.Pi}
			}
			m.AddQuad(p(phi0, th0), p(phi1, th0), p(phi1, th1), p(phi0, th1))
		}
	}
}

// AddBillboard appends a camera-facing quad (used for particles, flashes, decals).
func (m *Mesh) AddBillboard(center, right, up Vec3, hw, hh float64, uvScale float64) {
	n := right.Cross(up).Normalize()
	a := center.Sub(right.Mul(hw)).Sub(up.Mul(hh))
	b := center.Add(right.Mul(hw)).Sub(up.Mul(hh))
	c := center.Add(right.Mul(hw)).Add(up.Mul(hh))
	d := center.Sub(right.Mul(hw)).Add(up.Mul(hh))
	uv := [4]Vec2{{0, 1}, {uvScale, 1}, {uvScale, 0}, {0, 0}}
	m.AddQuadN(a, b, c, d, n, uv)
}

// AddBulletHole appends a small dark decal quad plus a ring, oriented by n.
func (m *Mesh) AddBulletHole(p, n Vec3, size float64) {
	// Build an orthonormal basis around the surface normal.
	t := V3(0, 1, 0)
	if math.Abs(n.Y) > 0.9 {
		t = V3(1, 0, 0)
	}
	r := n.Cross(t).Normalize()
	u := r.Cross(n).Normalize()
	origin := p.Add(n.Mul(0.004))
	m.AddQuadN(
		origin.Sub(r.Mul(size)).Sub(u.Mul(size)),
		origin.Add(r.Mul(size)).Sub(u.Mul(size)),
		origin.Add(r.Mul(size)).Add(u.Mul(size)),
		origin.Sub(r.Mul(size)).Add(u.Mul(size)),
		n, [4]Vec2{{0, 1}, {1, 1}, {1, 0}, {0, 0}},
	)
}
