package main

import "math"

// Particle kinds.
const (
	pWater = iota
	pSparkle
	pDust
	pSpark
	pCasing
	pSmoke
)

// Particle is a single billboard sprite with simple physics.
type Particle struct {
	pos, vel Vec3
	life     float64
	maxLife  float64
	size     float64
	grow     float64
	kind     int
	drag     float64
	gravity  float64
	bounce   bool
	stuck    bool
	dead     bool
	spin     float64
	ang      float64
}

// Shard is one piece of broken glass.
type Shard struct {
	mesh   *Mesh
	pos    Vec3
	vel    Vec3
	euler  Vec3
	angVel Vec3
	life   float64
	radius float64
	dead   bool
	bounds AABB
}

// Bounds returns the shard's world-space AABB.
func (s *Shard) Bounds() AABB { return s.bounds }

// Model returns the shard transform.
func (s *Shard) Model() Mat4 {
	return TRS(s.pos, s.euler.X, s.euler.Y, s.euler.Z, V3(1, 1, 1))
}

// ---------------------------------------------------------------------------
// Shattering
// ---------------------------------------------------------------------------

// Shatter replaces the intact glass with a cloud of physical shards.
// impact is the world-space hit point, dir the shot direction.
func (w *World) Shatter(impact, dir Vec3) {
	if w.glassBroken {
		return
	}
	w.glassBroken = true

	// Local-space impact (relative to the glass origin) drives the burst.
	local := impact.Sub(w.glassBase)
	radial := V3(local.X, 0, local.Z)
	if radial.Len() < 1e-4 {
		radial = V3(1, 0, 0)
	}
	radial = radial.Normalize()

	shards := buildShards()
	for i, s := range shards {
		// Base velocity: outward from the impact line, plus the bullet's push.
		away := s.pos.Sub(local).Normalize()
		spread := 0.55 + hash1(float64(i)*3.1)*0.9
		v := away.Mul(spread * 1.05)
		v = v.Add(radial.Mul(0.75 + hash1(float64(i)*7.7)*0.85))
		v = v.Add(dir.Mul(1.15 + hash1(float64(i)*2.3)*1.05))
		v.Y += 0.55 + hash1(float64(i)*5.9)*1.35
		s.vel = v
		s.pos = s.pos.Add(w.glassBase)

		s.angVel = V3((hash1(float64(i)*11.3)-0.5)*16, (hash1(float64(i)*13.7)-0.5)*16, (hash1(float64(i)*17.1)-0.5)*16)
		s.euler = V3(hash1(float64(i)*19.3)*6.28, hash1(float64(i)*23.9)*6.28, hash1(float64(i)*29.5)*6.28)
		s.life = 9.5 + hash1(float64(i)*31.1)*3
		s.recomputeBounds()
		w.shards = append(w.shards, s)
	}

	// Water splash.
	level := w.waterLevel
	for i := 0; i < 90; i++ {
		a := hash1(float64(i)*4.7) * 2 * math.Pi
		rr := math.Sqrt(hash1(float64(i)*9.1)) * 0.026
		p := w.glassBase.Add(V3(math.Cos(a)*rr, 0.02+hash1(float64(i)*6.1)*(level-0.02), math.Sin(a)*rr))
		v := V3(math.Cos(a)*(0.6+hash1(float64(i)*3.3)*1.9), 0.9+hash1(float64(i)*8.9)*2.6, math.Sin(a)*(0.6+hash1(float64(i)*5.5)*1.9))
		v = v.Add(dir.Mul(1.4 + hash1(float64(i)*12.3)*1.6))
		v = v.Add(V3(radial.X, 0, radial.Z).Mul(0.8))
		w.particles = append(w.particles, &Particle{
			pos: p, vel: v, life: 1.6 + hash1(float64(i)*2.1)*1.4, maxLife: 3,
			size: 0.006 + hash1(float64(i)*7.3)*0.008, kind: pWater, gravity: -9.0, bounce: true, drag: 0.25,
		})
	}
	// A fine mist that hangs for a moment.
	for i := 0; i < 26; i++ {
		a := hash1(float64(i)*3.9) * 2 * math.Pi
		p := w.glassBase.Add(V3(math.Cos(a)*0.02, 0.05+hash1(float64(i)*4.4)*0.09, math.Sin(a)*0.02))
		w.particles = append(w.particles, &Particle{
			pos: p, vel: V3(math.Cos(a)*0.7, 0.5+hash1(float64(i)*6.6)*0.9, math.Sin(a)*0.7),
			life: 0.9 + hash1(float64(i)*8.2)*0.9, maxLife: 1.8,
			size: 0.007, grow: 0.016, kind: pSmoke, gravity: -0.4, drag: 1.6,
		})
	}
	// Glass glitter.
	for i := 0; i < 40; i++ {
		away := V3(hash1(float64(i)*1.7)-0.5, hash1(float64(i)*2.9)*0.9, hash1(float64(i)*3.7)-0.5).Normalize()
		w.particles = append(w.particles, &Particle{
			pos: impact.Add(away.Mul(0.02)),
			vel: away.Mul(1.5 + hash1(float64(i)*4.3)*3.5),
			life: 0.35 + hash1(float64(i)*5.1)*0.7, maxLife: 1.1,
			size: 0.004 + hash1(float64(i)*6.7)*0.004, kind: pSparkle, gravity: -6, drag: 0.8,
		})
	}

	// The puddle starts growing immediately.
	w.puddleR = 0.02
}

// buildShards slices the glass wall and base into jittered solid pieces.
// Shard geometry is centred on each piece's centroid (local to the glass).
func buildShards() []*Shard {
	const sectors = 14
	const bands = 3
	const th = glassWall

	// Wall profile (r,y), densely sampled by arc length for jittered bands.
	curve := densify(glassWallProfile(), 24)

	// Grid of jittered angles and arc positions.
	angles := make([]float64, sectors+1)
	for i := 0; i <= sectors; i++ {
		j := 0.0
		if i > 0 && i < sectors {
			j = (hash2(float64(i), 1.7) - 0.5) * (2 * math.Pi / float64(sectors)) * 0.55
		}
		angles[i] = 2*math.Pi*float64(i)/float64(sectors) + j
	}
	ss := make([]float64, bands+1)
	for j := 0; j <= bands; j++ {
		jj := 0.0
		if j > 0 && j < bands {
			jj = (hash2(float64(j), 9.3) - 0.5) * (1.0 / float64(bands)) * 0.5
		}
		ss[j] = Clamp01(float64(j)/float64(bands) + jj)
	}

	surf := func(a, s float64) pt3 {
		lp := sampleCurve(curve, s)
		return pt3{lp.R * math.Cos(a), lp.Y, lp.R * math.Sin(a)}
	}
	inner := func(p pt3) pt3 {
		r := math.Hypot(p.x, p.z)
		if r < 1e-6 {
			return p
		}
		nr := math.Max(r-th, 0.001)
		k := nr / r
		return pt3{p.x * k, p.y, p.z * k}
	}

	var out []*Shard
	for i := 0; i < sectors; i++ {
		for j := 0; j < bands; j++ {
			a0, a1 := angles[i], angles[i+1]
			s0, s1 := ss[j], ss[j+1]
			A := surf(a0, s0)
			B := surf(a1, s0)
			C := surf(a1, s1)
			D := surf(a0, s1)
			m := buildShardMesh(A, B, C, D, inner)
			if m != nil {
				out = append(out, m)
			}
		}
	}
	// Base: pie wedges of the bottom.
	const baseWedges = 6
	innerR := glassRBottom - glassWall
	for i := 0; i < baseWedges; i++ {
		a0 := 2 * math.Pi * float64(i) / baseWedges
		a1 := 2 * math.Pi * float64(i+1) / baseWedges
		h := glassBaseY - 0.0015
		A := pt3{innerR * math.Cos(a0), 0, innerR * math.Sin(a0)}
		B := pt3{innerR * math.Cos(a1), 0, innerR * math.Sin(a1)}
		C := pt3{(glassRBottom + 0.0002) * math.Cos(a1), h, (glassRBottom + 0.0002) * math.Sin(a1)}
		D := pt3{(glassRBottom + 0.0002) * math.Cos(a0), h, (glassRBottom + 0.0002) * math.Sin(a0)}
		E := pt3{0, 0, 0}
		F := pt3{0, h, 0}
		m := buildWedgeShard(A, B, C, D, E, F)
		if m != nil {
			out = append(out, m)
		}
	}
	return out
}

// densify resamples a profile polyline into n points by arc length.
func densify(p []LathePoint, n int) []LathePoint {
	lens := make([]float64, len(p))
	for i := 1; i < len(p); i++ {
		lens[i] = lens[i-1] + math.Hypot(p[i].R-p[i-1].R, p[i].Y-p[i-1].Y)
	}
	total := lens[len(p)-1]
	out := make([]LathePoint, n)
	for k := 0; k < n; k++ {
		s := total * float64(k) / float64(n-1)
		idx := 0
		for idx < len(p)-2 && lens[idx+1] < s {
			idx++
		}
		seg := lens[idx+1] - lens[idx]
		t := 0.0
		if seg > 1e-9 {
			t = (s - lens[idx]) / seg
		}
		out[k] = LathePoint{R: Lerp(p[idx].R, p[idx+1].R, t), Y: Lerp(p[idx].Y, p[idx+1].Y, t)}
	}
	return out
}

// sampleCurve samples a dense polyline by normalised arc length.
func sampleCurve(c []LathePoint, s float64) LathePoint {
	s = Clamp01(s)
	f := s * float64(len(c)-1)
	i := int(f)
	if i >= len(c)-1 {
		return c[len(c)-1]
	}
	t := f - float64(i)
	return LathePoint{R: Lerp(c[i].R, c[i+1].R, t), Y: Lerp(c[i].Y, c[i+1].Y, t)}
}

type pt3 struct{ x, y, z float64 }

func (p pt3) v() Vec3 { return V3(p.x, p.y, p.z) }

// buildShardMesh builds a curved hexahedral shard from four outer-surface
// corners and returns it centred on its centroid.
func buildShardMesh(A, B, C, D pt3, inner func(pt3) pt3) *Shard {
	A2, B2, C2, D2 := inner(A), inner(B), inner(C), inner(D)
	return shardFromPoints(A.v(), B.v(), C.v(), D.v(), A2.v(), B2.v(), C2.v(), D2.v())
}

func buildWedgeShard(A, B, C, D, E, F pt3) *Shard {
	// A,B inner bottom edge; D,C outer top edge; E apex bottom; F apex top.
	return shardFromPoints(
		A.v(), B.v(), C.v(), D.v(),
		E.v(), E.v(), F.v(), F.v(),
	)
}

// shardFromPoints welds eight corners into a closed solid, centres it and
// returns a Shard carrying only the mesh.
func shardFromPoints(p ...Vec3) *Shard {
	if len(p) != 8 {
		return nil
	}
	c := Vec3{}
	for _, v := range p {
		c = c.Add(v)
	}
	c = c.Div(8)
	m := newMesh()
	o := make([]Vec3, 8)
	for i, v := range p {
		o[i] = v.Sub(c)
	}
	addFace := func(a, b, cc, d int, flip bool) {
		var va, vb, vc, vd Vec3
		if flip {
			va, vb, vc, vd = o[a], o[d], o[cc], o[b]
		} else {
			va, vb, vc, vd = o[a], o[b], o[cc], o[d]
		}
		n := vb.Sub(va).Cross(vc.Sub(va)).Normalize()
		m.AddQuadN(va, vb, vc, vd, n, [4]Vec2{{0, 0}, {1, 0}, {1, 1}, {0, 1}})
	}
	// 0..3 outer (A,B,C,D), 4..7 inner (A2,B2,C2,D2) matching indices.
	addFace(0, 1, 2, 3, false) // outer
	addFace(4, 5, 6, 7, true)  // inner
	addFace(0, 1, 5, 4, true)  // bottom edge
	addFace(1, 2, 6, 5, true)  // side
	addFace(2, 3, 7, 6, true)  // top edge
	addFace(3, 0, 4, 7, true)  // side
	if m.TriCount() == 0 {
		return nil
	}
	radius := 0.0
	for _, v := range m.Verts {
		radius = math.Max(radius, v.Pos.Len())
	}
	return &Shard{mesh: m, pos: c, radius: radius}
}

// recomputeBounds refreshes the world-space AABB of a shard.
func (s *Shard) recomputeBounds() {
	r := s.radius
	s.bounds = Box(s.pos, V3(r, r, r))
}

// ---------------------------------------------------------------------------
// Simulation
// ---------------------------------------------------------------------------

// Update advances shards, particles and the water puddle.
func (w *World) Update(dt float64) {
	if w.puddleR > 0 && w.puddleR < 0.155 {
		w.puddleR = math.Min(0.155, w.puddleR+dt*0.06)
	}
	w.updateShards(dt)
	w.updateParticles(dt)
}

func (w *World) updateShards(dt float64) {
	solids := w.shardColliders()
	alive := w.shards[:0]
	for _, s := range w.shards {
		if s.dead {
			continue
		}
		s.life -= dt
		if s.life <= 0 {
			continue
		}
		s.vel.Y -= 9.81 * dt
		s.vel = s.vel.Mul(1 - math.Min(1, 0.7*dt))
		s.pos = s.pos.Add(s.vel.Mul(dt))
		s.euler = s.euler.Add(s.angVel.Mul(dt))
		s.angVel = s.angVel.Mul(1 - math.Min(1, 1.4*dt))

		// Floor, walls, table and other furniture.
		if s.pos.Y < 0.002+s.radius*0.4 {
			w.bounceShardFloor(s, 0.002)
		}
		const m = 0.06
		if s.pos.X < roomMinX+m+s.radius*0.5 {
			s.pos.X = roomMinX + m + s.radius*0.5
			s.vel.X = math.Abs(s.vel.X) * 0.35
			s.angVel = s.angVel.Mul(0.7)
		}
		if s.pos.X > roomMaxX-m-s.radius*0.5 {
			s.pos.X = roomMaxX - m - s.radius*0.5
			s.vel.X = -math.Abs(s.vel.X) * 0.35
			s.angVel = s.angVel.Mul(0.7)
		}
		if s.pos.Z < roomMinZ+m+s.radius*0.5 {
			s.pos.Z = roomMinZ + m + s.radius*0.5
			s.vel.Z = math.Abs(s.vel.Z) * 0.35
			s.angVel = s.angVel.Mul(0.7)
		}
		if s.pos.Z > roomMaxZ-m-s.radius*0.5 {
			s.pos.Z = roomMaxZ - m - s.radius*0.5
			s.vel.Z = -math.Abs(s.vel.Z) * 0.35
			s.angVel = s.angVel.Mul(0.7)
		}
		for _, b := range solids {
			if collideSphereBox(s, b) {
				s.angVel = s.angVel.Mul(0.75)
			}
		}
		if s.pos.Y > roomH-0.1 {
			s.pos.Y = roomH - 0.1
			s.vel.Y = -math.Abs(s.vel.Y) * 0.3
		}
		s.recomputeBounds()
		alive = append(alive, s)
	}
	w.shards = alive
}

func (w *World) bounceShardFloor(s *Shard, y float64) {
	s.pos.Y = y + s.radius*0.4
	if s.vel.Y < 0 {
		s.vel.Y = -s.vel.Y * 0.30
		if s.vel.Y < 0.35 {
			s.vel.Y = 0
		}
	}
	s.vel.X *= 0.72
	s.vel.Z *= 0.72
	s.angVel = s.angVel.Mul(0.55)
}

// shardColliders returns the boxes shards can bounce off.
func (w *World) shardColliders() []AABB {
	out := make([]AABB, 0, len(w.boxes))
	for _, b := range w.boxes {
		// Ignore thin/vertical slivers that would trap shards.
		if b.Max.Y-b.Min.Y < 0.01 {
			continue
		}
		out = append(out, b)
	}
	return out
}

// collideSphereBox pushes a shard out of a box and reflects its velocity.
func collideSphereBox(s *Shard, b AABB) bool {
	r := s.radius * 0.45
	q := V3(
		Clamp(s.pos.X, b.Min.X, b.Max.X),
		Clamp(s.pos.Y, b.Min.Y, b.Max.Y),
		Clamp(s.pos.Z, b.Min.Z, b.Max.Z),
	)
	d := s.pos.Sub(q)
	if d.LenSq() > r*r {
		// Centre may be inside the box: handle that case explicitly.
		if !b.Contains(s.pos) {
			return false
		}
	}
	n := d
	if n.LenSq() < 1e-9 {
		// Inside: push out along the axis of least penetration.
		c := b.Center()
		e := b.Size().Mul(0.5)
		dd := s.pos.Sub(c)
		ax, ay, az := math.Abs(dd.X)-e.X, math.Abs(dd.Y)-e.Y, math.Abs(dd.Z)-e.Z
		switch {
		case ax >= ay && ax >= az:
			n = V3(math.Copysign(1, dd.X), 0, 0)
			s.pos.X = c.X + math.Copysign(e.X+r, dd.X)
		case ay >= az:
			n = V3(0, math.Copysign(1, dd.Y), 0)
			s.pos.Y = c.Y + math.Copysign(e.Y+r, dd.Y)
		default:
			n = V3(0, 0, math.Copysign(1, dd.Z))
			s.pos.Z = c.Z + math.Copysign(e.Z+r, dd.Z)
		}
	} else {
		n = n.Normalize()
		s.pos = q.Add(n.Mul(r))
	}
	vn := s.vel.Dot(n)
	if vn < 0 {
		t := s.vel.Sub(n.Mul(vn))
		s.vel = t.Mul(0.74).Add(n.Mul(-vn * 0.28))
	}
	return true
}

func (w *World) updateParticles(dt float64) {
	alive := w.particles[:0]
	for _, p := range w.particles {
		if p.dead {
			continue
		}
		p.life -= dt
		if p.life <= 0 {
			continue
		}
		if p.stuck {
			alive = append(alive, p)
			continue
		}
		p.vel.Y += p.gravity * dt
		if p.drag > 0 {
			p.vel = p.vel.Mul(1 - math.Min(1, p.drag*dt))
		}
		p.ang += p.spin * dt
		newPos := p.pos.Add(p.vel.Mul(dt))

		if p.kind == pCasing || p.kind == pWater {
			// Bounce off the floor and the table top.
			floor := 0.001 + p.size*0.4
			if newPos.Y < floor {
				newPos.Y = floor
				if p.vel.Y < 0 {
					p.vel.Y = -p.vel.Y * 0.32
					if p.vel.Y < 0.25 {
						p.vel.Y = 0
					}
				}
				p.vel.X *= 0.7
				p.vel.Z *= 0.7
				if p.kind == pWater && p.vel.Len() < 0.35 {
					p.vel = Vec3{}
					p.life = math.Min(p.life, 1.4)
				}
			}
			for _, b := range w.boxes {
				if b.Min.Y > 0.2 && newPos.Y-p.size < b.Max.Y && newPos.Y+p.size > b.Min.Y &&
					newPos.X > b.Min.X-0.01 && newPos.X < b.Max.X+0.01 &&
					newPos.Z > b.Min.Z-0.01 && newPos.Z < b.Max.Z+0.01 {
					if p.vel.Y < 0 {
						newPos.Y = b.Max.Y + p.size*0.6
						p.vel.Y = -p.vel.Y * 0.3
						p.vel.X *= 0.7
						p.vel.Z *= 0.7
						if p.kind == pWater {
							p.life = math.Min(p.life, 0.9)
						}
					}
				}
			}
		}
		p.pos = newPos
		alive = append(alive, p)
	}
	w.particles = alive
}

// ---------------------------------------------------------------------------
// Particles rendering
// ---------------------------------------------------------------------------

// SubmitDynamic queues shards and particles for the frame.
func (w *World) SubmitDynamic(r *Renderer) {
	cam := r.Camera()
	right := cam.right()
	up := right.Cross(cam.forward()).Normalize()

	// Shards (translucent, sorted with the rest of the blended geometry).
	for _, s := range w.shards {
		if s.dead {
			continue
		}
		a := 0.62
		if s.life < 1.6 {
			a *= Clamp01(s.life / 1.6)
		}
		r.Submit(s.mesh, s.Model(), Material{
			Albedo: V3(0.70, 0.82, 0.86), Spec: 1.0, Shininess: 220,
			Alpha: a, DoubleSided: true, Refl: 0.62,
		})
	}

	// Group particles by kind so each batch shares a material.
	groups := map[int]*Mesh{}
	for _, p := range w.particles {
		if p.dead {
			continue
		}
		lifeT := Clamp01(p.life / math.Max(p.maxLife, 1e-3))
		sz := p.size
		if p.grow > 0 {
			sz += p.grow * (1 - lifeT)
		}
		m := groups[p.kind]
		if m == nil {
			m = newMesh()
			groups[p.kind] = m
		}
		alpha := lifeT
		switch p.kind {
		case pWater:
			alpha = Clamp01(lifeT * 1.6)
		case pSparkle:
			alpha = Clamp01(lifeT * 2.2)
		case pSpark:
			alpha = Clamp01(lifeT * 2.6)
		case pSmoke:
			alpha = (1 - lifeT) * 0.32
		}
		if alpha <= 0.01 {
			continue
		}
		start := len(m.Verts)
		m.AddBillboard(p.pos, right, up, sz, sz, 1)
		for i := start; i < len(m.Verts); i++ {
			m.Verts[i].A = alpha
		}
	}

	for kind, m := range groups {
		if len(m.Idx) == 0 {
			continue
		}
		var mat Material
		switch kind {
		case pWater:
			mat = Material{Albedo: V3(0.62, 0.80, 0.92), Unlit: true, Alpha: 0.8, DoubleSided: true}
		case pSparkle:
			mat = Material{Albedo: V3(0.85, 0.95, 1.0), Unlit: true, Alpha: 0.9, Additive: true, DoubleSided: true}
		case pSpark:
			mat = Material{Albedo: V3(1.6, 1.05, 0.45), Unlit: true, Alpha: 1, Additive: true, DoubleSided: true}
		case pSmoke:
			mat = Material{Albedo: V3(0.30, 0.32, 0.34), Unlit: true, Alpha: 0.30, Additive: true, DoubleSided: true}
		case pCasing:
			mat = Material{Albedo: V3(0.95, 0.72, 0.30), Spec: 0.9, Shininess: 96, Alpha: 1, DoubleSided: true}
		default:
			mat = Material{Albedo: V3(0.8, 0.8, 0.8), Unlit: true, Alpha: 0.6, DoubleSided: true}
		}
		r.Submit(m, Identity4(), mat)
	}
}

// SpawnImpact adds sparks and dust where a bullet struck a hard surface.
func (w *World) SpawnImpact(p, n Vec3, soft bool) {
	for i := 0; i < 14; i++ {
		d := n.Add(V3(hash1(float64(i)*3.1)-0.5, hash1(float64(i)*5.3)-0.5, hash1(float64(i)*7.7)-0.5).Mul(1.3)).Normalize()
		w.particles = append(w.particles, &Particle{
			pos: p.Add(n.Mul(0.005)), vel: d.Mul(1.2 + hash1(float64(i)*2.7)*3.2),
			life: 0.16 + hash1(float64(i)*4.9)*0.3, maxLife: 0.5,
			size: 0.0035 + hash1(float64(i)*6.1)*0.003, kind: pSpark, gravity: -7, drag: 1.6,
		})
	}
	nDust := 10
	if soft {
		nDust = 18
	}
	for i := 0; i < nDust; i++ {
		d := n.Add(V3(hash1(float64(i)*1.3)-0.5, hash1(float64(i)*2.9)-0.5+0.3, hash1(float64(i)*8.1)-0.5).Mul(1.4)).Normalize()
		w.particles = append(w.particles, &Particle{
			pos: p.Add(n.Mul(0.01)), vel: d.Mul(0.5 + hash1(float64(i)*3.3)*1.4),
			life: 0.5 + hash1(float64(i)*5.7)*0.8, maxLife: 1.3,
			size: 0.005 + hash1(float64(i)*7.9)*0.005, grow: 0.014, kind: pDust, gravity: -0.5, drag: 2.1,
		})
	}
}

// SpawnCasing ejects a spent case from the pistol.
func (w *World) SpawnCasing(pos, vel Vec3) {
	w.particles = append(w.particles, &Particle{
		pos: pos, vel: vel, life: 3.4, maxLife: 3.4, size: 0.008, kind: pCasing,
		gravity: -9.0, bounce: true, drag: 0.4, spin: 14,
	})
}

// SpawnMuzzleSmoke puffs a little smoke from the muzzle.
func (w *World) SpawnMuzzleSmoke(pos, dir Vec3) {
	for i := 0; i < 8; i++ {
		d := dir.Mul(0.6).Add(V3(hash1(float64(i)*2.1)-0.5, hash1(float64(i)*3.7)+0.25, hash1(float64(i)*4.3)-0.5).Mul(0.5))
		w.particles = append(w.particles, &Particle{
			pos: pos, vel: d.Mul(0.5 + hash1(float64(i)*6.7)*0.9),
			life: 0.4 + hash1(float64(i)*8.3)*0.6, maxLife: 1.0,
			size: 0.006, grow: 0.022, kind: pSmoke, gravity: 0.5, drag: 1.9,
		})
	}
}
