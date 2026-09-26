package main

import (
	"math"
	"runtime"
	"sort"
	"sync"
)

// ---------------------------------------------------------------------------
// Noise / procedural texture helpers (all cheap: a few hashes and lerps)
// ---------------------------------------------------------------------------

// ihash is a fast 64-bit integer hash (splitmix-style finaliser).
func ihash(x, y int64) uint64 {
	h := uint64(x)*0x9E3779B97F4A7C15 ^ uint64(y)*0xC2B2AE3D27D4EB4F
	h ^= h >> 30
	h *= 0xBF58476D1CE4E5B9
	h ^= h >> 27
	h *= 0x94D049BB133111EB
	h ^= h >> 31
	return h
}

// hash1 returns a pseudo-random value in [0,1) for a scalar.
func hash1(n float64) float64 {
	return float64(ihash(int64(math.Floor(n)), 0x5bf03635)>>11) / float64(uint64(1)<<53)
}

// hash2 returns a pseudo-random value in [0,1) for a lattice cell.
func hash2(x, y float64) float64 {
	return float64(ihash(int64(math.Floor(x)), int64(math.Floor(y)))>>11) / float64(uint64(1)<<53)
}

const inv53 = 1.0 / float64(uint64(1)<<53)

// h2i hashes an integer lattice cell into [0,1).
func h2i(x, y int64) float64 { return float64(ihash(x, y)>>11) * inv53 }

func vnoise(x, y float64) float64 {
	xi, yi := math.Floor(x), math.Floor(y)
	xf, yf := x-xi, y-yi
	u := xf * xf * (3 - 2*xf)
	v := yf * yf * (3 - 2*yf)
	ix, iy := int64(xi), int64(yi)
	a := h2i(ix, iy)
	b := h2i(ix+1, iy)
	c := h2i(ix, iy+1)
	d := h2i(ix+1, iy+1)
	return (a + (b-a)*u) + ((c + (d-c)*u)-(a + (b-a)*u))*v
}

func fbm(x, y float64, oct int) float64 {
	sum, amp, freq, norm := 0.0, 0.5, 1.0, 0.0
	for i := 0; i < oct; i++ {
		sum += amp * vnoise(x*freq, y*freq)
		norm += amp
		amp *= 0.5
		freq *= 2.03
	}
	return sum / norm
}

var (
	colFloorA  = V3(0.42, 0.27, 0.155)
	colFloorB  = V3(0.30, 0.185, 0.105)
	colPlaster = V3(0.80, 0.775, 0.735)
	colWoodA   = V3(0.40, 0.235, 0.125)
	colWoodB   = V3(0.265, 0.145, 0.072)
)

// TexPlanks returns a wood floor plank pattern. u,v are metres.
func texPlanks(u, v float64) Vec3 {
	const plankW = 0.20
	const plankL = 1.35
	row := math.Floor(v / plankW)
	fy := v/plankW - row
	off := hash1(row*3.77) * 0.9
	uu := u/plankL + off
	col := math.Floor(uu)
	fx := uu - col
	h := hash2(row*1.7, col*2.3)
	base := colFloorA.Lerp(colFloorB, h)
	// grain: fine streaks along the plank direction
	g := vnoise(uu*38, row*17.0+fy*3.0)
	base = base.Mul(1 + (g-0.5)*0.28)
	// stagger joints
	if math.Abs(fx-0.5) > 0.494 {
		base = base.Mul(0.55)
	}
	// gap between planks
	edge := math.Min(fy, 1-fy)
	if edge < 0.035 {
		base = base.Mul(0.45 + edge*10)
	}
	return base
}

// TexPlaster is a subtly mottled painted wall. u,v are metres.
func texPlaster(u, v float64) Vec3 {
	n := vnoise(u*2.1, v*2.1)
	g := vnoise(u*0.55, v*0.42)
	c := colPlaster.Mul(0.90 + n*0.16)
	c = c.Mul(0.96 + g*0.09)
	return c
}

// TexWood is the table top: warm timber with elliptical grain. u,v are metres.
func texWood(u, v float64) Vec3 {
	rings := math.Sin((v*9.0+vnoise(u*3.1, v*3.1)*4.5)*math.Pi) * 0.5
	c := colWoodA.Lerp(colWoodB, Clamp01(rings*0.5+0.5))
	c = c.Mul(1 + (vnoise(u*26, v*5.2)-0.5)*0.22)
	return c
}

// TexRug is a woven rug. u,v are metres.
func texRug(u, v float64) Vec3 {
	border := math.Min(math.Min(u+1.3, 1.3-u), math.Min(v+0.95, 0.95-v))
	base := V3(0.34, 0.10, 0.095)
	pattern := V3(0.52, 0.30, 0.14)
	if border < 0.12 {
		base = V3(0.20, 0.055, 0.055)
		if border > 0.085 {
			base = pattern
		}
	}
	check := math.Mod(math.Floor(u*14)+math.Floor(v*14), 2)
	if border >= 0.12 && check < 0.5 {
		base = base.Lerp(pattern, 0.30)
	}
	n := fbm(u*30, v*30, 2)
	return base.Mul(0.85 + n*0.30)
}

// TexCanvasArt is a small abstract painting. u,v in 0..1.
func texCanvasArt(u, v float64) Vec3 {
	sky := V3(0.62, 0.72, 0.85).Lerp(V3(0.95, 0.80, 0.58), Clamp01(v*1.1))
	blob := fbm(u*3.4+1.7, v*3.4, 3)
	col := sky.Lerp(V3(0.20, 0.28, 0.42), Clamp01((blob-0.42)*2.6))
	band := math.Abs(v-0.62)
	if band < 0.05 {
		col = col.Lerp(V3(0.85, 0.35, 0.22), 0.75*(1-band/0.05))
	}
	return col.Mul(0.95 + fbm(u*60, v*60, 2)*0.10)
}

// TexGlow is the bright window pane.
func texGlow(u, v float64) Vec3 {
	t := Clamp01(v)
	sky := V3(0.70, 0.83, 1.02).Lerp(V3(1.10, 1.08, 1.00), t)
	// faint gradient plus a soft cloud
	sky = sky.Mul(1 + (vnoise(u*2.2, v*2.2)-0.5)*0.10)
	return sky.Mul(1.35)
}

// TexMetal is brushed gunmetal.
func texMetal(u, v float64) Vec3 {
	n := vnoise(u*160, v*26)
	return V3(0.30, 0.31, 0.34).Mul(0.88 + n*0.24)
}

// TexTile is a ceramic tile (used sparingly, e.g. a small tray).
func texTile(u, v float64) Vec3 {
	fx := math.Abs(u - math.Floor(u) - 0.5)
	fy := math.Abs(v - math.Floor(v) - 0.5)
	c := V3(0.80, 0.82, 0.84)
	if math.Max(fx, fy) > 0.47 {
		c = V3(0.45, 0.46, 0.48)
	}
	return c.Mul(0.96 + vnoise(u*20, v*20)*0.08)
}

// ---------------------------------------------------------------------------
// Lighting
// ---------------------------------------------------------------------------

// PointLight is a positional light with smooth distance falloff.
type PointLight struct {
	Pos     Vec3
	Color   Vec3
	Range   float64
	Power   float64
	Enabled bool
}

// DirLight is a directional light (sun through the window).
type DirLight struct {
	Dir     Vec3 // direction the light travels
	Color   Vec3
	Enabled bool
}

// ---------------------------------------------------------------------------
// Geometry submission
// ---------------------------------------------------------------------------

// DrawItem is one mesh instance queued for a frame.
type DrawItem struct {
	mesh  *Mesh
	model Mat4
	mat   Material
	off   int // vertex offset into the frame's transformed vertex arena
	ord   int
}

// xvert is a transformed vertex in screen space.
type xvert struct {
	wp    Vec3
	vp    Vec3 // view space
	n     Vec3 // world normal
	u, v  float64
	alpha float64
	sx    float64
	sy    float64
	iw    float64 // 1 / view depth
	depth float64 // positive view depth
	front bool
}

// groundPoly is a flat translucent/additive polygon lying on a horizontal plane.
type groundPoly struct {
	pts []Vec2
	y   float64
	mat Material
}

// Camera is the view used for a frame.
type Camera struct {
	Pos    Vec3
	Yaw    float64
	Pitch  float64
	Roll   float64
	FovY   float64
	Near   float64
	Far    float64
	Aspect float64
	view   Mat4
	proj   Mat4
}

func (c *Camera) right() Vec3 {
	return V3(math.Cos(c.Yaw), 0, math.Sin(c.Yaw))
}

// Rotation returns the camera's world orientation matrix (yaw, pitch, roll).
func (c *Camera) Rotation() Mat4 {
	return RotY(-c.Yaw).Mul(RotX(c.Pitch)).Mul(RotZ(c.Roll))
}

func (c *Camera) forward() Vec3 {
	cp := math.Cos(c.Pitch)
	return V3(math.Sin(c.Yaw)*cp, math.Sin(c.Pitch), -math.Cos(c.Yaw)*cp)
}

func (c *Camera) upVec() Vec3 {
	return c.right().Cross(c.forward()).Normalize()
}

// ViewDir returns the world-space direction from a point to the camera.
func (c *Camera) ViewDir(p Vec3) Vec3 { return c.Pos.Sub(p).Normalize() }

// Renderer is the software rasterizer.
type Renderer struct {
	W, H    int
	Color   []uint32 // packed R|G<<8|B<<16|A<<24
	Depth   []float32
	bloom   []Vec3
	bloomT  []Vec3
	bloomW  int
	bloomH  int
	ambTop  Vec3
	ambBot  Vec3
	exposed float64

	cam    Camera
	items  []DrawItem
	shads  []groundPoly
	lights []PointLight
	dir    DirLight

	arena []xvert
	tris  []rtri
	bands int

	tone [toneLUTSize]float64

	BloomOn bool
	Stats   FrameStats
}

// FrameStats carries per-frame timing/counters for the HUD.
type FrameStats struct {
	Shaded  int
	Tris    int
	Items   int
	MsShade float64
}

// rtri is a projected triangle ready to rasterize.
type rtri struct {
	v       [3]xvert
	mat     *Material
	ord     int
	area    float64
	zmin    float64
	specLvl int
}

// specLevel maps a Blinn-Phong exponent to a number of squarings (k^(2^lvl)).
func specLevel(shininess float64) int {
	if shininess <= 2 {
		return 1
	}
	lvl := int(math.Round(math.Log2(shininess)))
	if lvl < 1 {
		lvl = 1
	}
	if lvl > 8 {
		lvl = 8
	}
	return lvl
}

// NewRenderer allocates a renderer for the given resolution.
func NewRenderer(w, h int, bloom bool) *Renderer {
	r := &Renderer{
		BloomOn: bloom,
		ambTop:  V3(0.118, 0.125, 0.148),
		ambBot:  V3(0.032, 0.030, 0.029),
		exposed: 0.98,
		bands:   runtime.NumCPU(),
	}
	r.Resize(w, h)
	return r
}

// Resize (re)allocates the framebuffers.
func (r *Renderer) Resize(w, h int) {
	if w < 16 {
		w = 16
	}
	if h < 16 {
		h = 16
	}
	if r.W == w && r.H == h && r.Color != nil {
		return
	}
	r.W, r.H = w, h
	r.Color = make([]uint32, w*h)
	r.Depth = make([]float32, w*h)
	r.bloomW, r.bloomH = (w+3)/4, (h+3)/4
	r.bloom = make([]Vec3, r.bloomW*r.bloomH)
	r.bloomT = make([]Vec3, r.bloomW*r.bloomH)
	r.buildLUT()
	r.arena = make([]xvert, 0, 65536)
	r.tris = make([]rtri, 0, 32768)
	if r.bands < 1 {
		r.bands = 1
	}
	if r.bands > h {
		r.bands = h
	}
}

// toneLUTSize is the resolution of the linear→display lookup table.
const toneLUTSize = 4096

// toneMax is the linear value mapped to the top of the table.
const toneMax = 8.0

func (r *Renderer) buildLUT() {
	for i := 0; i < toneLUTSize; i++ {
		x := float64(i) / float64(toneLUTSize-1) * toneMax
		r.tone[i] = math.Pow(1-math.Exp(-x*r.exposed), 1/2.2)
	}
}

// SetCamera installs the frame camera (also computes view/proj matrices).
func (r *Renderer) SetCamera(c Camera) {
	if c.FovY <= 0 {
		c.FovY = Degrees(70)
	}
	if c.Near <= 0 {
		c.Near = 0.02
	}
	if c.Far <= 0 {
		c.Far = 60
	}
	if c.Aspect <= 0 {
		c.Aspect = float64(r.W) / float64(r.H)
	}
	c.view = LookAt(c.Pos, c.Pos.Add(c.forward()), V3(0, 1, 0))
	if c.Roll != 0 {
		c.view = c.view.Mul(RotZ(c.Roll))
	}
	c.proj = Perspective(c.FovY, c.Aspect, c.Near, c.Far)
	r.cam = c
}

// Camera returns the current frame camera.
func (r *Renderer) Camera() *Camera { return &r.cam }

// SetAmbient sets the hemisphere ambient colours.
func (r *Renderer) SetAmbient(top, bottom Vec3) { r.ambTop, r.ambBot = top, bottom }

// SetLights installs the point lights for this frame.
func (r *Renderer) SetLights(l []PointLight, d DirLight) {
	r.lights = append(r.lights[:0], l...)
	r.dir = d
}

// Reset begins a new frame.
func (r *Renderer) Reset(sky Vec3) {
	r.items = r.items[:0]
	r.shads = r.shads[:0]
	r.arena = r.arena[:0]
	r.tris = r.tris[:0]
	r.Stats = FrameStats{}
	c := packColor(sky)
	inf := float32(math.Inf(1))
	r.par(r.H, func(y int) {
		row := y * r.W
		for i := row; i < row+r.W; i++ {
			r.Color[i] = c
			r.Depth[i] = inf
		}
	})
}

func packColor(c Vec3) uint32 {
	cr := uint32(Clamp(c.X, 0, 1)*255 + 0.5)
	cg := uint32(Clamp(c.Y, 0, 1)*255 + 0.5)
	cb := uint32(Clamp(c.Z, 0, 1)*255 + 0.5)
	return cr | cg<<8 | cb<<16 | 0xFF<<24
}

func (r *Renderer) itemOrder() int { return len(r.items) }

// Submit queues a mesh instance.
func (r *Renderer) Submit(m *Mesh, model Mat4, mat Material) {
	if m == nil || len(m.Idx) == 0 {
		return
	}
	if mat.Alpha <= 0 {
		mat.Alpha = 1
	}
	r.items = append(r.items, DrawItem{mesh: m, model: model, mat: mat, ord: len(r.items)})
}

// AddGroundPoly queues a flat polygon lying on the horizontal plane y (metres).
func (r *Renderer) AddGroundPoly(pts []Vec2, y float64, mat Material) {
	if len(pts) < 3 {
		return
	}
	r.shads = append(r.shads, groundPoly{pts: append([]Vec2(nil), pts...), y: y, mat: mat})
}

// AddBoxShadow projects an AABB onto the ground along the light direction.
func (r *Renderer) AddBoxShadow(b AABB, lightDir Vec3, groundY, alpha float64) {
	if lightDir.Y >= -1e-3 {
		return
	}
	var pts []Vec2
	for i := 0; i < 8; i++ {
		p := Vec3{b.Min.X, b.Min.Y, b.Min.Z}
		if i&1 != 0 {
			p.X = b.Max.X
		}
		if i&2 != 0 {
			p.Y = b.Max.Y
		}
		if i&4 != 0 {
			p.Z = b.Max.Z
		}
		t := (p.Y - groundY) / lightDir.Y
		q := p.Sub(lightDir.Mul(t))
		pts = append(pts, Vec2{q.X, q.Z})
	}
	hull := convexHull2D(pts)
	r.AddGroundPoly(hull, groundY, Material{Albedo: V3(0, 0, 0), Alpha: alpha, Unlit: true})
}

// convexHull2D returns the convex hull of pts (monotone chain, CCW).
func convexHull2D(pts []Vec2) []Vec2 {
	if len(pts) < 3 {
		return pts
	}
	sort.Slice(pts, func(i, j int) bool {
		if pts[i].X == pts[j].X {
			return pts[i].Y < pts[j].Y
		}
		return pts[i].X < pts[j].X
	})
	cross := func(o, a, b Vec2) float64 {
		return (a.X-o.X)*(b.Y-o.Y) - (a.Y-o.Y)*(b.X-o.X)
	}
	var hull []Vec2
	for _, p := range pts {
		for len(hull) >= 2 && cross(hull[len(hull)-2], hull[len(hull)-1], p) <= 0 {
			hull = hull[:len(hull)-1]
		}
		hull = append(hull, p)
	}
	lower := len(hull) + 1
	for i := len(pts) - 2; i >= 0; i-- {
		p := pts[i]
		for len(hull) >= lower && cross(hull[len(hull)-2], hull[len(hull)-1], p) <= 0 {
			hull = hull[:len(hull)-1]
		}
		hull = append(hull, p)
	}
	if len(hull) > 1 {
		hull = hull[:len(hull)-1]
	}
	return hull
}

// project fills screen-space fields of an xvert.
func (r *Renderer) project(v *xvert) {
	clip := r.cam.proj.MulVec4([4]float64{v.vp.X, v.vp.Y, v.vp.Z, 1})
	invW := 1 / clip[3]
	ndcX := clip[0] * invW
	ndcY := clip[1] * invW
	v.sx = (ndcX*0.5 + 0.5) * float64(r.W)
	v.sy = (0.5 - ndcY*0.5) * float64(r.H)
	v.depth = -v.vp.Z
	v.iw = 1 / math.Max(v.depth, 1e-6)
	v.front = v.depth > r.cam.Near
}

// transform fills the arena for all queued items.
func (r *Renderer) transform() {
	view := r.cam.view
	near := r.cam.Near
	for i := range r.items {
		it := &r.items[i]
		it.off = len(r.arena)
		n := len(it.mesh.Verts)
		start := len(r.arena)
		r.arena = append(r.arena, make([]xvert, n)...)
		dst := r.arena[start:]
		anyFront, anyBehind := false, false
		for j := range it.mesh.Verts {
			sv := &it.mesh.Verts[j]
			wp := it.model.MulPoint(sv.Pos)
			vp := view.MulPoint(wp)
			a := sv.A
			if a == 0 {
				a = 1
			}
			xv := xvert{wp: wp, vp: vp, n: it.model.MulDir(sv.Nrm).Normalize(), u: sv.U, v: sv.V, alpha: a}
			dst[j] = xv
			if -vp.Z > near {
				anyFront = true
			} else {
				anyBehind = true
			}
		}
		if anyFront && !anyBehind {
			for j := range dst {
				r.project(&dst[j])
			}
		} else if anyFront {
			for j := range dst {
				if -dst[j].vp.Z > near {
					r.project(&dst[j])
				}
			}
		}
	}
}

// lerpX interpolates two xverts by t (used by near-plane clipping).
func lerpX(a, b xvert, t float64) xvert {
	return xvert{
		wp:    a.wp.Lerp(b.wp, t),
		vp:    a.vp.Lerp(b.vp, t),
		n:     a.n.Lerp(b.n, t),
		u:     Lerp(a.u, b.u, t),
		v:     Lerp(a.v, b.v, t),
		alpha: Lerp(a.alpha, b.alpha, t),
	}
}

// Flush rasterizes everything queued.
func (r *Renderer) Flush() {
	r.transform()

	// Build triangle list (near-plane clipped), split into opaque/transparent.
	near := r.cam.Near
	var opaqueTris, blendTris []rtri
	matOf := func(it *DrawItem) *Material { return &r.items[it.ord].mat }

	for i := range r.items {
		it := &r.items[i]
		verts := r.arena[it.off : it.off+len(it.mesh.Verts)]
		mat := matOf(it)
		blend := !mat.Opaque() || mat.Additive
		for t := 0; t+2 < len(it.mesh.Idx); t += 3 {
			a := verts[it.mesh.Idx[t]]
			b := verts[it.mesh.Idx[t+1]]
			c := verts[it.mesh.Idx[t+2]]
			front := a.front && b.front && c.front
			if front {
				rt := makeTri(a, b, c, mat, it.ord)
				if rt.area == 0 {
					continue
				}
				if blend {
					blendTris = append(blendTris, rt)
				} else {
					opaqueTris = append(opaqueTris, rt)
				}
				continue
			}
			if !a.front && !b.front && !c.front {
				continue
			}
			// Clip against the near plane and fan-triangulate.
			poly := clipNear([]xvert{a, b, c}, near)
			for k := 1; k+1 < len(poly); k++ {
				for j := range poly {
					if !poly[j].front {
						r.project(&poly[j])
					}
				}
				rt := makeTri(poly[0], poly[k], poly[k+1], mat, it.ord)
				if rt.area == 0 {
					continue
				}
				if blend {
					blendTris = append(blendTris, rt)
				} else {
					opaqueTris = append(opaqueTris, rt)
				}
			}
		}
	}

	r.Stats.Tris = len(opaqueTris) + len(blendTris)
	r.Stats.Items = len(r.items)

	// Ground shadows go into the blend list, drawn after opaque geometry.
	for _, sp := range r.shads {
		base := len(r.arena)
		for _, p := range sp.pts {
			wp := V3(p.X, sp.y, p.Y)
			vp := r.cam.view.MulPoint(wp)
			xv := xvert{wp: wp, vp: vp, n: V3(0, 1, 0), alpha: 1}
			r.arena = append(r.arena, xv)
			if -vp.Z > near {
				r.project(&r.arena[len(r.arena)-1])
			}
		}
		spMat := sp.mat
		for k := 1; k+1 < len(sp.pts); k++ {
			a, b, c := r.arena[base], r.arena[base+k], r.arena[base+k+1]
			if !(a.front && b.front && c.front) {
				continue
			}
			rt := makeTri(a, b, c, &spMat, -1)
			if rt.area != 0 {
				blendTris = append(blendTris, rt)
			}
		}
	}

	// Opaque geometry is drawn front-to-back so the depth test rejects the
	// hidden pixels before they are shaded.
	sort.Slice(opaqueTris, func(i, j int) bool { return opaqueTris[i].zmin < opaqueTris[j].zmin })
	r.rasterBands(opaqueTris)
	// Transparent: sort back-to-front so blending is correct.
	sort.SliceStable(blendTris, func(i, j int) bool { return blendTris[i].zmin > blendTris[j].zmin })
	r.rasterBands(blendTris)

	if r.BloomOn {
		r.applyBloom()
	}
}

func clipNear(poly []xvert, near float64) []xvert {
	var out []xvert
	for i := range poly {
		cur := poly[i]
		nxt := poly[(i+1)%len(poly)]
		curIn := -cur.vp.Z > near
		nxtIn := -nxt.vp.Z > near
		if curIn {
			out = append(out, cur)
		}
		if curIn != nxtIn {
			d0 := -cur.vp.Z - near
			d1 := -nxt.vp.Z - near
			t := d0 / (d0 - d1)
			out = append(out, lerpX(cur, nxt, t))
		}
	}
	return out
}

func (r *Renderer) makeTriPub(a, b, c xvert, mat *Material, ord int) rtri {
	return makeTri(a, b, c, mat, ord)
}

func makeTri(a, b, c xvert, mat *Material, ord int) rtri {
	area := (b.sx-a.sx)*(c.sy-a.sy) - (c.sx-a.sx)*(b.sy-a.sy)
	zmin := math.Min(a.depth, math.Min(b.depth, c.depth))
	lvl := 0
	if mat.Spec > 0 {
		lvl = specLevel(mat.Shininess)
	}
	return rtri{v: [3]xvert{a, b, c}, mat: mat, ord: ord, area: area, zmin: zmin, specLvl: lvl}
}

// ---------------------------------------------------------------------------
// Rasterization
// ---------------------------------------------------------------------------

func (r *Renderer) rasterBands(tris []rtri) {
	if len(tris) == 0 {
		return
	}
	bands := r.bands
	if bands > r.H {
		bands = r.H
	}
	if bands <= 1 {
		r.rasterRange(tris, 0, r.H)
		return
	}
	var wg sync.WaitGroup
	h := r.H
	for b := 0; b < bands; b++ {
		y0 := b * h / bands
		y1 := (b + 1) * h / bands
		wg.Add(1)
		go func(y0, y1 int) {
			defer wg.Done()
			r.rasterRange(tris, y0, y1)
		}(y0, y1)
	}
	wg.Wait()
}

func (r *Renderer) rasterRange(tris []rtri, y0, y1 int) {
	W := r.W
	shaded := 0
	for i := range tris {
		t := &tris[i]
		a, b, c := &t.v[0], &t.v[1], &t.v[2]
		minXf := math.Min(a.sx, math.Min(b.sx, c.sx))
		maxXf := math.Max(a.sx, math.Max(b.sx, c.sx))
		minYf := math.Min(a.sy, math.Min(b.sy, c.sy))
		maxYf := math.Max(a.sy, math.Max(b.sy, c.sy))
		if maxYf < float64(y0) || minYf > float64(y1) {
			continue
		}
		// Backface culling (front faces are clockwise in screen space).
		if !t.mat.DoubleSided && t.area >= 0 {
			continue
		}
		x0 := int(math.Max(math.Floor(minXf), 0))
		x1 := int(math.Min(math.Ceil(maxXf), float64(W-1)))
		py0 := int(math.Max(math.Floor(minYf), float64(y0)))
		py1 := int(math.Min(math.Ceil(maxYf), float64(y1-1)))
		if x1 < x0 || py1 < py0 {
			continue
		}
		invArea := 1 / t.area
		// Barycentric gradients (w0,w1,w2) per horizontal pixel step.
		dw0dx := (b.sy - c.sy) * invArea
		dw1dx := (c.sy - a.sy) * invArea
		dw2dx := (a.sy - b.sy) * invArea

		mat := t.mat
		specLvl := t.specLvl
		blend := !mat.Opaque() || mat.Additive
		for y := py0; y <= py1; y++ {
			px := float64(x0) + 0.5
			py := float64(y) + 0.5
			w0 := ((b.sy-c.sy)*(px-c.sx) + (c.sx-b.sx)*(py-c.sy)) * invArea
			w1 := ((c.sy-a.sy)*(px-c.sx) + (a.sx-c.sx)*(py-c.sy)) * invArea
			w2 := 1 - w0 - w1
			row := y * W
			for x := x0; x <= x1; x++ {
				if w0 >= 0 && w1 >= 0 && w2 >= 0 {
					iw := w0*a.iw + w1*b.iw + w2*c.iw
					if iw > 0 {
						depth := 1 / iw
						idx := row + x
						if float32(depth) < r.Depth[idx] {
							att := 1 / iw
							p := V3(
								(w0*a.wp.X*a.iw+w1*b.wp.X*b.iw+w2*c.wp.X*c.iw)*att,
								(w0*a.wp.Y*a.iw+w1*b.wp.Y*b.iw+w2*c.wp.Y*c.iw)*att,
								(w0*a.wp.Z*a.iw+w1*b.wp.Z*b.iw+w2*c.wp.Z*c.iw)*att)
							n := V3(
								(w0*a.n.X*a.iw+w1*b.n.X*b.iw+w2*c.n.X*c.iw)*att,
								(w0*a.n.Y*a.iw+w1*b.n.Y*b.iw+w2*c.n.Y*c.iw)*att,
								(w0*a.n.Z*a.iw+w1*b.n.Z*b.iw+w2*c.n.Z*c.iw)*att)
							nl := n.Len()
							if nl > 1e-9 {
								n = n.Div(nl)
							}
							u := (w0*a.u*a.iw + w1*b.u*b.iw + w2*c.u*c.iw) * att
							v := (w0*a.v*a.iw + w1*b.v*b.iw + w2*c.v*c.iw) * att
							alpha := (w0*a.alpha*a.iw + w1*b.alpha*b.iw + w2*c.alpha*c.iw) * att
							col := r.shade(mat, p, n, u, v, specLvl)
							shaded++
							af := mat.Alpha * alpha
							if blend {
								// Blend in display space, like a 2D compositor would.
								e := r.encode(col)
								if mat.Additive {
									r.Color[idx] = addColor(r.Color[idx], e, Clamp01(af))
								} else {
									if af > 0.999 {
										r.Depth[idx] = float32(depth)
									}
									r.Color[idx] = mixColor(r.Color[idx], e, Clamp01(af))
								}
							} else {
								r.Color[idx] = r.tonemap(col)
								r.Depth[idx] = float32(depth)
							}
						}
					}
				}
				w0 += dw0dx
				w1 += dw1dx
				w2 += dw2dx
			}
		}
	}
	shadeMu.Lock()
	r.Stats.Shaded += shaded
	shadeMu.Unlock()
}

var shadeMu sync.Mutex

// enc maps one linear channel to display space through the tone LUT.
func (r *Renderer) enc(x float64) float64 {
	if x <= 0 {
		return 0
	}
	if x >= toneMax {
		return r.tone[toneLUTSize-1]
	}
	return r.tone[int(x*(toneLUTSize-1)/toneMax)]
}

// encode applies exposure, an exponential roll-off and gamma, returning a
// display-referred colour in 0..1.
func (r *Renderer) encode(c Vec3) Vec3 {
	return V3(r.enc(c.X), r.enc(c.Y), r.enc(c.Z))
}

// tonemap applies exposure, an exponential roll-off and gamma (via LUT).
func (r *Renderer) tonemap(c Vec3) uint32 {
	e := r.encode(c)
	return pack255(e.X*255, e.Y*255, e.Z*255)
}

// addColor adds a display-space colour to a packed pixel.
func addColor(dst uint32, c Vec3, a float64) uint32 {
	dr := float64(dst&0xFF) + c.X*a*255
	dg := float64((dst>>8)&0xFF) + c.Y*a*255
	db := float64((dst>>16)&0xFF) + c.Z*a*255
	return pack255(dr, dg, db)
}

// mixColor alpha-blends a display-space colour over a packed pixel.
func mixColor(dst uint32, c Vec3, a float64) uint32 {
	dr := float64(dst & 0xFF)
	dg := float64((dst >> 8) & 0xFF)
	db := float64((dst >> 16) & 0xFF)
	dr += (c.X*255 - dr) * a
	dg += (c.Y*255 - dg) * a
	db += (c.Z*255 - db) * a
	return pack255(dr, dg, db)
}

func pack255(r, g, b float64) uint32 {
	cr := uint32(Clamp(r, 0, 255))
	cg := uint32(Clamp(g, 0, 255))
	cb := uint32(Clamp(b, 0, 255))
	return cr | cg<<8 | cb<<16 | 0xFF<<24
}

// shade evaluates the lighting model for one pixel. It is written with scalar
// arithmetic because it is the innermost loop of the whole renderer.
func (r *Renderer) shade(mat *Material, p, n Vec3, u, v float64, specLvl int) Vec3 {
	ar, ag, ab := mat.Albedo.X, mat.Albedo.Y, mat.Albedo.Z
	switch mat.Tex {
	case TexPlanks:
		t := texPlanks(u, v)
		ar, ag, ab = ar*t.X, ag*t.Y, ab*t.Z
	case TexPlaster:
		t := texPlaster(u, v)
		ar, ag, ab = ar*t.X, ag*t.Y, ab*t.Z
	case TexWood:
		t := texWood(u, v)
		ar, ag, ab = ar*t.X, ag*t.Y, ab*t.Z
	case TexRug:
		t := texRug(u, v)
		ar, ag, ab = ar*t.X, ag*t.Y, ab*t.Z
	case TexCanvasArt:
		t := texCanvasArt(u, v)
		ar, ag, ab = ar*t.X, ag*t.Y, ab*t.Z
	case TexGlow:
		t := texGlow(u, v)
		ar, ag, ab = ar*t.X, ag*t.Y, ab*t.Z
	case TexMetal:
		t := texMetal(u, v)
		ar, ag, ab = ar*t.X, ag*t.Y, ab*t.Z
	case TexTile:
		t := texTile(u, v)
		ar, ag, ab = ar*t.X, ag*t.Y, ab*t.Z
	}
	if mat.Unlit {
		return V3(ar+mat.Emissive.X, ag+mat.Emissive.Y, ab+mat.Emissive.Z)
	}

	// Direction from the surface to the eye.
	ex := r.cam.Pos.X - p.X
	ey := r.cam.Pos.Y - p.Y
	ez := r.cam.Pos.Z - p.Z
	el2 := ex*ex + ey*ey + ez*ez
	if el2 > 1e-12 {
		ei := 1 / math.Sqrt(el2)
		ex, ey, ez = ex*ei, ey*ei, ez*ei
	}
	nx, ny, nz := n.X, n.Y, n.Z
	if mat.DoubleSided && nx*ex+ny*ey+nz*ez < 0 {
		nx, ny, nz = -nx, -ny, -nz
	}

	// Hemisphere ambient.
	t := ny*0.5 + 0.5
	ambR := r.ambTop.X*t + r.ambBot.X*(1-t)
	ambG := r.ambTop.Y*t + r.ambBot.Y*(1-t)
	ambB := r.ambTop.Z*t + r.ambBot.Z*(1-t)
	cr, cg, cb := ar*ambR, ag*ambG, ab*ambB

	spec := mat.Spec

	// Directional light (daylight through the window).
	if r.dir.Enabled {
		lx, ly, lz := -r.dir.Dir.X, -r.dir.Dir.Y, -r.dir.Dir.Z
		ll := math.Sqrt(lx*lx + ly*ly + lz*lz)
		if ll > 1e-9 {
			li := 1 / ll
			lx, ly, lz = lx*li, ly*li, lz*li
		}
		if ndl := nx*lx + ny*ly + nz*lz; ndl > 0 {
			dc := r.dir.Color
			cr += ar * dc.X * ndl
			cg += ag * dc.Y * ndl
			cb += ab * dc.Z * ndl
			if spec > 0 {
				hx, hy, hz := lx+ex, ly+ey, lz+ez
				hl2 := hx*hx + hy*hy + hz*hz
				if hl2 > 1e-12 {
					k := (nx*hx + ny*hy + nz*hz) / math.Sqrt(hl2)
					if k > 0 {
						sp := spec * specPowLvl(k, specLvl)
						cr += dc.X * sp
						cg += dc.Y * sp
						cb += dc.Z * sp
					}
				}
			}
		}
	}

	// Point lights.
	for i := range r.lights {
		L := &r.lights[i]
		if !L.Enabled {
			continue
		}
		dx := L.Pos.X - p.X
		dy := L.Pos.Y - p.Y
		dz := L.Pos.Z - p.Z
		d2 := dx*dx + dy*dy + dz*dz
		rng2 := L.Range * L.Range
		if d2 > rng2*1.6 {
			continue
		}
		dd := math.Sqrt(d2)
		if dd < 1e-6 {
			dd = 1e-6
		}
		invd := 1 / dd
		lx, ly, lz := dx*invd, dy*invd, dz*invd
		ndl := nx*lx + ny*ly + nz*lz
		if ndl <= 0 {
			continue
		}
		att := L.Power / (1 + d2/rng2)
		w := 1 - Clamp01((dd-0.15*L.Range)/(1.45*L.Range))
		att *= w * w
		if att <= 0.0005 {
			continue
		}
		f := ndl * att
		lc := L.Color
		cr += ar * lc.X * f
		cg += ag * lc.Y * f
		cb += ab * lc.Z * f
		if spec > 0 {
			hx, hy, hz := lx+ex, ly+ey, lz+ez
			hl2 := hx*hx + hy*hy + hz*hz
			if hl2 > 1e-12 {
				k := (nx*hx + ny*hy + nz*hz) / math.Sqrt(hl2)
				if k > 0 {
					sp := spec * att * specPowLvl(k, specLvl)
					cr += lc.X * sp
					cg += lc.Y * sp
					cb += lc.Z * sp
				}
			}
		}
	}

	em := mat.Emissive
	cr += em.X
	cg += em.Y
	cb += em.Z

	// Fresnel-ish sheen for glass.
	if mat.Refl > 0 {
		f := 1 - Clamp01(nx*ex+ny*ey+nz*ez)
		f = f * f * f * f * mat.Refl
		cr += 0.55 * f
		cg += 0.62 * f
		cb += 0.72 * f
	}
	return V3(cr, cg, cb)
}

// specPowLvl computes k^(2^lvl) by progressive squaring (no log/exp in the loop).
func specPowLvl(k float64, lvl int) float64 {
	k2 := k * k
	switch lvl {
	case 0, 1:
		return k2
	case 2:
		return k2 * k2
	case 3:
		k4 := k2 * k2
		return k4 * k4
	case 4:
		k4 := k2 * k2
		k8 := k4 * k4
		return k8 * k8
	case 5:
		k4 := k2 * k2
		k8 := k4 * k4
		return k8 * k8 * k8 * k8
	case 6:
		k4 := k2 * k2
		k8 := k4 * k4
		k16 := k8 * k8
		return k16 * k16
	case 7:
		k4 := k2 * k2
		k8 := k4 * k4
		k16 := k8 * k8
		return k16 * k16 * k16 * k16
	default:
		k4 := k2 * k2
		k8 := k4 * k4
		k16 := k8 * k8
		k32 := k16 * k16
		return k32 * k32
	}
}

// ---------------------------------------------------------------------------
// Bloom
// ---------------------------------------------------------------------------

func (r *Renderer) applyBloom() {
	bw, bh := r.bloomW, r.bloomH
	if bw < 2 || bh < 2 {
		return
	}
	tmp := r.bloomT

	// Bright pass with a 2x2 subsample of each 4x4 block.
	r.par(bh, func(y int) {
		py := y * 4
		row0, row1 := 0, 0
		if py < r.H {
			row0 = py * r.W
		}
		if py+1 < r.H {
			row1 = (py + 1) * r.W
		}
		for x := 0; x < bw; x++ {
			px := x * 4
			var sr, sg, sb float64
			n := 0.0
			if row0 != 0 && px < r.W {
				c := r.Color[row0+px]
				sr += float64(c & 0xFF)
				sg += float64((c >> 8) & 0xFF)
				sb += float64((c >> 16) & 0xFF)
				n++
			}
			if row1 != 0 && px+1 < r.W {
				c := r.Color[row1+px+1]
				sr += float64(c & 0xFF)
				sg += float64((c >> 8) & 0xFF)
				sb += float64((c >> 16) & 0xFF)
				n++
			}
			if n == 0 {
				r.bloom[y*bw+x] = Vec3{}
				continue
			}
			sr /= n * 255
			sg /= n * 255
			sb /= n * 255
			mx := math.Max(sr, math.Max(sg, sb))
			k := (mx - 0.70) * 3.6
			if k <= 0 {
				r.bloom[y*bw+x] = Vec3{}
			} else {
				if k > 1 {
					k = 1
				}
				r.bloom[y*bw+x] = V3(sr*k, sg*k, sb*k)
			}
		}
	})

	// Two 5-tap separable blur passes.
	blur := func(src, dst []Vec3, horizontal bool) {
		r.par(bh, func(y int) {
			for x := 0; x < bw; x++ {
				var sr, sg, sb, wsum float64
				for k := -2; k <= 2; k++ {
					xx, yy := x, y
					if horizontal {
						xx = x + k
					} else {
						yy = y + k
					}
					if xx < 0 || xx >= bw || yy < 0 || yy >= bh {
						continue
					}
					w := 3.0 - math.Abs(float64(k))
					c := src[yy*bw+xx]
					sr += c.X * w
					sg += c.Y * w
					sb += c.Z * w
					wsum += w
				}
				if wsum > 0 {
					dst[y*bw+x] = V3(sr/wsum, sg/wsum, sb/wsum)
				}
			}
		})
	}
	blur(r.bloom, tmp, true)
	blur(tmp, r.bloom, false)

	// Additive upsample back into the frame (scalar maths, 4 taps).
	r.par(r.H, func(y int) {
		sy := float64(y)*0.25 - 0.5
		y0 := int(math.Floor(sy))
		fy := sy - float64(y0)
		yb0 := int(Clamp(float64(y0), 0, float64(bh-1)))
		yb1 := int(Clamp(float64(y0+1), 0, float64(bh-1)))
		b0 := yb0 * bw
		b1 := yb1 * bw
		w00 := (1 - fy)
		w1 := fy
		row := y * r.W
		for x := 0; x < r.W; x++ {
			sx := float64(x)*0.25 - 0.5
			x0 := int(math.Floor(sx))
			fx := sx - float64(x0)
			xa := int(Clamp(float64(x0), 0, float64(bw-1)))
			xb := int(Clamp(float64(x0+1), 0, float64(bw-1)))
			wa, wb := 1-fx, fx
			c00 := r.bloom[b0+xa]
			c01 := r.bloom[b0+xb]
			c10 := r.bloom[b1+xa]
			c11 := r.bloom[b1+xb]
			w0a := wa * w00
			w0b := wb * w00
			w1a := wa * w1
			w1b := wb * w1
			br := (c00.X*w0a + c01.X*w0b + c10.X*w1a + c11.X*w1b) * 0.62
			bg := (c00.Y*w0a + c01.Y*w0b + c10.Y*w1a + c11.Y*w1b) * 0.62
			bb := (c00.Z*w0a + c01.Z*w0b + c10.Z*w1a + c11.Z*w1b) * 0.62
			if br+bg+bb <= 0.001 {
				continue
			}
			idx := row + x
			c := r.Color[idx]
			dr := float64(c&0xFF) + br*255
			dg := float64((c>>8)&0xFF) + bg*255
			db := float64((c>>16)&0xFF) + bb*255
			r.Color[idx] = pack255(dr, dg, db)
		}
	})
}

// par runs f(i) for i in [0,n) across the configured number of workers.
func (r *Renderer) par(n int, f func(i int)) {
	if n <= 0 {
		return
	}
	workers := r.bands
	if workers > n {
		workers = n
	}
	if workers <= 1 {
		for i := 0; i < n; i++ {
			f(i)
		}
		return
	}
	var wg sync.WaitGroup
	chunk := (n + workers - 1) / workers
	for b := 0; b < n; b += chunk {
		hi := b + chunk
		if hi > n {
			hi = n
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			for i := lo; i < hi; i++ {
				f(i)
			}
		}(b, hi)
	}
	wg.Wait()
}

// CloneColor returns a copy of the framebuffer (safe to hand to another goroutine).
func (r *Renderer) CloneColor() []uint32 {
	out := make([]uint32, len(r.Color))
	copy(out, r.Color)
	return out
}

// ColorBytes converts the framebuffer to RGBA8 bytes.
func (r *Renderer) ColorBytes(dst []byte) []byte {
	need := len(r.Color) * 4
	if cap(dst) < need {
		dst = make([]byte, need)
	}
	dst = dst[:need]
	r.par(r.H, func(y int) {
		base := y * r.W
		o := base * 4
		for i := base; i < base+r.W; i++ {
			c := r.Color[i]
			dst[o+0] = byte(c)
			dst[o+1] = byte(c >> 8)
			dst[o+2] = byte(c >> 16)
			dst[o+3] = 0xFF
			o += 4
		}
	})
	return dst
}
