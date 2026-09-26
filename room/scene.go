package main

import "math"

// Room dimensions (metres, +Y up). The player starts at +Z looking toward -Z.
const (
	roomMinX = -2.85
	roomMaxX = 2.85
	roomMinZ = -3.25
	roomMaxZ = 3.25
	roomH    = 2.62

	tableTopY  = 0.75
	tableThick = 0.045

	// Drinking glass (a tumbler): outer dimensions and wall thickness.
	glassH       = 0.150
	glassRBottom = 0.036
	glassRTop    = 0.0425
	glassWall    = 0.0018
	glassBaseY   = 0.017
	glassR       = glassRTop // collision/raycast radius
)

// geoGroup is one material batch of static geometry.
type geoGroup struct {
	mat  Material
	mesh *Mesh
}

// GeoBuilder batches static geometry by material.
type GeoBuilder struct {
	groups []geoGroup
	cur    int
	have   bool
}

// Use starts (or reuses) the batch for a material and returns its mesh.
func (g *GeoBuilder) Use(mat Material) *Mesh {
	if g.have && g.groups[g.cur].mat == mat {
		return g.groups[g.cur].mesh
	}
	for i := range g.groups {
		if g.groups[i].mat == mat {
			g.cur, g.have = i, true
			return g.groups[i].mesh
		}
	}
	g.groups = append(g.groups, geoGroup{mat: mat, mesh: newMesh()})
	g.cur, g.have = len(g.groups)-1, true
	return g.groups[g.cur].mesh
}

// World holds the room, its props and all mutable simulation state.
type World struct {
	static  []geoGroup
	glass   *Mesh
	water   *Mesh
	decal   *Mesh
	decalPt []decalRec

	glassBase   Vec3 // world position of the glass origin (bottom centre)
	waterLevel  float64
	puddleR     float64
	puddleMesh  *Mesh
	glassBroken bool

	// collision / raycast boxes for props and furniture
	boxes []AABB

	// dynamic state
	shards    []*Shard
	particles []*Particle
	lights    []PointLight
	dir       DirLight

	// scratch meshes rebuilt each frame
	dynMesh   *Mesh
	sparkMesh *Mesh
}

type decalRec struct {
	p, n Vec3
	size float64
	rot  float64
}

// NewWorld builds the room and its contents.
func NewWorld() *World {
	w := &World{}
	w.buildRoom()
	w.buildProps()
	w.buildGlass()
	w.glassBase = V3(-0.26, tableTopY+tableThick*0.5+0.002, -1.02)
	w.waterLevel = 0.098
	w.decal = newMesh()
	w.dynMesh = newMesh()
	w.sparkMesh = newMesh()
	w.puddleMesh = newMesh()
	w.dir = DirLight{Dir: V3(-1, -0.62, 0.3).Normalize(), Color: V3(0.64, 0.62, 0.57), Enabled: true}
	w.ceilingLamp()
	return w
}

// ---------------------------------------------------------------------------
// Materials
// ---------------------------------------------------------------------------

func matFloor() Material {
	return Material{Albedo: V3(1, 1, 1), Spec: 0.14, Shininess: 24, Alpha: 1, Tex: TexPlanks}
}

func matWall() Material {
	return Material{Albedo: V3(1, 1, 1), Spec: 0.05, Shininess: 8, Alpha: 1, Tex: TexPlaster}
}

func matCeiling() Material {
	return Material{Albedo: V3(0.94, 0.93, 0.90), Spec: 0.03, Shininess: 8, Alpha: 1, Tex: TexPlaster}
}

func matWood() Material {
	return Material{Albedo: V3(1, 1, 1), Spec: 0.22, Shininess: 32, Alpha: 1, Tex: TexWood}
}

func matDarkWood() Material {
	return Material{Albedo: V3(0.62, 0.55, 0.48), Spec: 0.20, Shininess: 28, Alpha: 1, Tex: TexWood}
}

func matTrim() Material {
	return Material{Albedo: V3(0.86, 0.85, 0.82), Spec: 0.10, Shininess: 16, Alpha: 1}
}

func matRug() Material {
	return Material{Albedo: V3(1, 1, 1), Spec: 0.02, Shininess: 6, Alpha: 1, Tex: TexRug}
}

func matGlass() Material {
	return Material{
		Albedo: V3(0.80, 0.88, 0.90), Spec: 0.85, Shininess: 128,
		Alpha: 0.24, DoubleSided: true, Refl: 0.55,
	}
}

func matWater() Material {
	return Material{
		Albedo: V3(0.42, 0.66, 0.78), Spec: 0.75, Shininess: 96,
		Alpha: 0.62, DoubleSided: true, Refl: 0.30,
	}
}

func matMetalGun() Material {
	return Material{Albedo: V3(0.55, 0.56, 0.60), Spec: 0.55, Shininess: 64, Alpha: 1, Tex: TexMetal}
}

func matPolymer() Material {
	return Material{Albedo: V3(0.16, 0.16, 0.18), Spec: 0.22, Shininess: 24, Alpha: 1}
}

// ---------------------------------------------------------------------------
// Room shell
// ---------------------------------------------------------------------------

func (w *World) buildRoom() {
	gb := &GeoBuilder{}

	floor := gb.Use(matFloor())
	floor.AddPlaneY(V3(0, 0, 0), (roomMaxX-roomMinX)/2, (roomMaxZ-roomMinZ)/2, true, 1)
	ceil := gb.Use(matCeiling())
	ceil.AddPlaneY(V3(0, roomH, 0), (roomMaxX-roomMinX)/2, (roomMaxZ-roomMinZ)/2, false, 1.6)

	wall := gb.Use(matWall())
	const t = 0.06
	// -Z and +Z walls
	wall.AddBoxUV(AABB{Min: V3(roomMinX-t, -0.1, roomMinZ-t), Max: V3(roomMaxX+t, roomH, roomMinZ)}, 1)
	wall.AddBoxUV(AABB{Min: V3(roomMinX-t, -0.1, roomMaxZ), Max: V3(roomMaxX+t, roomH, roomMaxZ+t)}, 1)
	// -X wall
	wall.AddBoxUV(AABB{Min: V3(roomMinX-t, -0.1, roomMinZ-t), Max: V3(roomMinX, roomH, roomMaxZ+t)}, 1)

	// +X wall with a window opening (z in [-1.55, 0.15], y in [0.95, 2.15])
	const wz0, wz1, wy0, wy1 = -1.55, 0.15, 0.95, 2.15
	wall.AddBoxUV(AABB{Min: V3(roomMaxX, -0.1, roomMinZ-t), Max: V3(roomMaxX+t, roomH, wz0)}, 1)
	wall.AddBoxUV(AABB{Min: V3(roomMaxX, -0.1, wz1), Max: V3(roomMaxX+t, roomH, roomMaxZ+t)}, 1)
	wall.AddBoxUV(AABB{Min: V3(roomMaxX, -0.1, wz0), Max: V3(roomMaxX+t, wy0, wz1)}, 1)
	wall.AddBoxUV(AABB{Min: V3(roomMaxX, wy1, wz0), Max: V3(roomMaxX+t, roomH, wz1)}, 1)

	// Window reveal (the inside of the opening, so it does not look paper thin).
	wall.AddBoxUV(AABB{Min: V3(roomMaxX, wy1, wz0 - 0.02), Max: V3(roomMaxX+0.10, wy1 + 0.05, wz1 + 0.02)}, 1)
	wall.AddBoxUV(AABB{Min: V3(roomMaxX, wy0 - 0.05, wz0 - 0.02), Max: V3(roomMaxX+0.10, wy0, wz1 + 0.02)}, 1)
	wall.AddBoxUV(AABB{Min: V3(roomMaxX, wy0 - 0.05, wz0 - 0.02), Max: V3(roomMaxX+0.10, wy1 + 0.05, wz0)}, 1)
	wall.AddBoxUV(AABB{Min: V3(roomMaxX, wy0 - 0.05, wz1), Max: V3(roomMaxX+0.10, wy1 + 0.05, wz1 + 0.02)}, 1)

	// Baseboards.
	trim := gb.Use(matTrim())
	const bb, bbt = 0.085, 0.018
	trim.AddBoxUV(AABB{Min: V3(roomMinX, 0, roomMinZ), Max: V3(roomMaxX, bb, roomMinZ+bbt)}, 1)
	trim.AddBoxUV(AABB{Min: V3(roomMinX, 0, roomMaxZ-bbt), Max: V3(roomMaxX, bb, roomMaxZ)}, 1)
	trim.AddBoxUV(AABB{Min: V3(roomMinX, 0, roomMinZ), Max: V3(roomMinX+bbt, bb, roomMaxZ)}, 1)
	trim.AddBoxUV(AABB{Min: V3(roomMaxX-bbt, 0, roomMinZ), Max: V3(roomMaxX, bb, roomMaxZ)}, 1)

	// Window frame + bright pane.
	trim.AddBoxUV(AABB{Min: V3(roomMaxX - 0.03, wy0 - 0.02, wz0 - 0.04), Max: V3(roomMaxX + 0.02, wy0 + 0.03, wz1 + 0.04)}, 1)
	trim.AddBoxUV(AABB{Min: V3(roomMaxX - 0.03, wy1 - 0.03, wz0 - 0.04), Max: V3(roomMaxX + 0.02, wy1 + 0.02, wz1 + 0.04)}, 1)
	trim.AddBoxUV(AABB{Min: V3(roomMaxX - 0.03, wy0, wz0 - 0.04), Max: V3(roomMaxX + 0.02, wy1, wz0 + 0.01)}, 1)
	trim.AddBoxUV(AABB{Min: V3(roomMaxX - 0.03, wy0, wz1 - 0.01), Max: V3(roomMaxX + 0.02, wy1, wz1 + 0.04)}, 1)
	trim.AddBoxUV(AABB{Min: V3(roomMaxX - 0.02, wy0, -0.72), Max: V3(roomMaxX + 0.01, wy1, -0.68)}, 1) // mullion

	pane := gb.Use(Material{Albedo: V3(1, 1, 1), Tex: TexGlow, Unlit: true, Emissive: V3(0.02, 0.02, 0.02)})
	pane.AddBoxUV(AABB{Min: V3(roomMaxX + 0.02, wy0, wz0), Max: V3(roomMaxX + 0.045, wy1, wz1)}, 1.2)

	// Door on the back wall.
	door := gb.Use(Material{Albedo: V3(0.72, 0.71, 0.70), Spec: 0.14, Shininess: 20, Alpha: 1})
	const dx0, dx1, dy1 = 1.05, 1.98, 2.06
	door.AddBoxUV(AABB{Min: V3(dx0, 0, roomMinZ), Max: V3(dx1, dy1, roomMinZ+0.045)}, 1)
	// Casing: four thin rails around the door, not a slab across the wall.
	const cw = 0.055
	trim.AddBoxUV(AABB{Min: V3(dx0 - cw, 0, roomMinZ), Max: V3(dx0, dy1 + cw, roomMinZ+0.022)}, 1)
	trim.AddBoxUV(AABB{Min: V3(dx1, 0, roomMinZ), Max: V3(dx1 + cw, dy1 + cw, roomMinZ+0.022)}, 1)
	trim.AddBoxUV(AABB{Min: V3(dx0 - cw, dy1, roomMinZ), Max: V3(dx1 + cw, dy1 + cw, roomMinZ+0.022)}, 1)
	panel := gb.Use(Material{Albedo: V3(0.63, 0.62, 0.61), Spec: 0.14, Shininess: 20, Alpha: 1})
	for _, py := range [2][2]float64{{0.18, 0.92}, {1.14, 1.88}} {
		panel.AddBoxUV(AABB{Min: V3(dx0 + 0.14, py[0], roomMinZ + 0.045), Max: V3(dx1 - 0.14, py[1], roomMinZ + 0.053)}, 1)
	}
	knob := gb.Use(Material{Albedo: V3(0.82, 0.72, 0.35), Spec: 0.85, Shininess: 96, Alpha: 1})
	knob.AddSphere(V3(dx0+0.10, 1.02, roomMinZ+0.075), 0.026, 12, 8)

	w.static = gb.groups
}

// ---------------------------------------------------------------------------
// Furniture and props
// ---------------------------------------------------------------------------

func (w *World) buildProps() {
	gb := &GeoBuilder{}
	wood := gb.Use(matWood())

	tc := V3(0, 0, -1.02) // table centre
	thx, thz := 0.82, 0.46
	top := AABB{Min: V3(tc.X-thx, tableTopY-tableThick, tc.Z-thz), Max: V3(tc.X+thx, tableTopY, tc.Z+thz)}
	wood.AddBoxUV(top, 1)
	w.boxes = append(w.boxes, top)

	// Apron rails under the top.
	ap := 0.055
	wood.AddBoxUV(AABB{Min: V3(tc.X-thx+0.06, tableTopY-0.16, tc.Z-thz+0.05), Max: V3(tc.X+thx-0.06, tableTopY-tableThick, tc.Z-thz+0.05+ap)}, 1)
	wood.AddBoxUV(AABB{Min: V3(tc.X-thx+0.06, tableTopY-0.16, tc.Z+thz-0.05-ap), Max: V3(tc.X+thx-0.06, tableTopY-tableThick, tc.Z+thz-0.05)}, 1)
	wood.AddBoxUV(AABB{Min: V3(tc.X-thx+0.05, tableTopY-0.16, tc.Z-thz+0.06), Max: V3(tc.X-thx+0.05+ap, tableTopY-tableThick, tc.Z+thz-0.06)}, 1)
	wood.AddBoxUV(AABB{Min: V3(tc.X+thx-0.05-ap, tableTopY-0.16, tc.Z-thz+0.06), Max: V3(tc.X+thx-0.05, tableTopY-tableThick, tc.Z+thz-0.06)}, 1)

	// Legs.
	const legH = 0.035
	for _, sx := range []float64{-1, 1} {
		for _, sz := range []float64{-1, 1} {
			lx := tc.X + sx*(thx-0.075)
			lz := tc.Z + sz*(thz-0.075)
			leg := AABB{Min: V3(lx-legH, 0, lz-legH), Max: V3(lx+legH, tableTopY-tableThick, lz+legH)}
			wood.AddBoxUV(leg, 1)
			w.boxes = append(w.boxes, leg)
		}
	}

	// Chair facing the table.
	chair := V3(1.28, 0, -0.52)
	seatY := 0.45
	seat := AABB{Min: V3(chair.X-0.22, seatY-0.035, chair.Z-0.22), Max: V3(chair.X+0.22, seatY, chair.Z+0.22)}
	wood.AddBoxUV(seat, 1)
	w.boxes = append(w.boxes, seat)
	for _, sx := range []float64{-1, 1} {
		for _, sz := range []float64{-1, 1} {
			lx := chair.X + sx*0.18
			lz := chair.Z + sz*0.18
			leg := AABB{Min: V3(lx-0.022, 0, lz-0.022), Max: V3(lx+0.022, seatY-0.035, lz+0.022)}
			wood.AddBoxUV(leg, 1)
			w.boxes = append(w.boxes, leg)
		}
	}
	// Backrest (on the far side from the table).
	bz := chair.Z + 0.20
	post := AABB{Min: V3(chair.X-0.20, seatY, bz-0.025), Max: V3(chair.X-0.16, seatY+0.46, bz+0.025)}
	wood.AddBoxUV(post, 1)
	post2 := AABB{Min: V3(chair.X+0.16, seatY, bz-0.025), Max: V3(chair.X+0.20, seatY+0.46, bz+0.025)}
	wood.AddBoxUV(post2, 1)
	slat := AABB{Min: V3(chair.X-0.18, seatY+0.30, bz-0.02), Max: V3(chair.X+0.18, seatY+0.42, bz+0.02)}
	wood.AddBoxUV(slat, 1)
	w.boxes = append(w.boxes, post, post2, slat)

	// Bookshelf against the -X wall.
	shelf := gb.Use(matDarkWood())
	sx0, sx1 := roomMinX, roomMinX+0.30
	sz0, sz1 := -2.45, -1.15
	sh := 1.72
	shelf.AddBoxUV(AABB{Min: V3(sx0, 0, sz0), Max: V3(sx1, sh, sz0+0.03)}, 1)
	shelf.AddBoxUV(AABB{Min: V3(sx0, 0, sz1-0.03), Max: V3(sx1, sh, sz1)}, 1)
	shelf.AddBoxUV(AABB{Min: V3(sx0, sh-0.04, sz0), Max: V3(sx1, sh, sz1)}, 1)
	shelf.AddBoxUV(AABB{Min: V3(sx0, 0, sz0), Max: V3(sx1, 0.06, sz1)}, 1)
	for _, y := range []float64{0.42, 0.86, 1.30, sh - 0.05} {
		shelf.AddBoxUV(AABB{Min: V3(sx0, y, sz0), Max: V3(sx1, y+0.03, sz1)}, 1)
	}
	bookCols := []Vec3{
		{0.42, 0.13, 0.11}, {0.13, 0.22, 0.36}, {0.16, 0.29, 0.20},
		{0.48, 0.34, 0.12}, {0.30, 0.16, 0.30}, {0.55, 0.50, 0.44},
	}
	bookMats := make([]Material, len(bookCols))
	for i, c := range bookCols {
		bookMats[i] = Material{Albedo: c, Spec: 0.10, Shininess: 14, Alpha: 1}
	}
	for _, y := range []float64{0.45, 0.89, 1.33} {
		bz := sz0 + 0.08
		for i := 0; i < 9; i++ {
			h := 0.17 + hash2(float64(i), y*10)*0.075
			th := 0.022 + hash2(float64(i)*2.3, y*7)*0.022
			if bz+th > sz1-0.05 {
				break
			}
			tone := int(hash2(float64(i)*5.1, y*3.7) * float64(len(bookMats)))
			if tone >= len(bookMats) {
				tone = len(bookMats) - 1
			}
			b := AABB{Min: V3(sx0+0.05, y+0.03, bz), Max: V3(sx1-0.02, y+0.03+h, bz+th)}
			gb.Use(bookMats[tone]).AddBoxUV(b, 1)
			bz += th + 0.004
		}
	}

	// Picture frame on the back wall.
	frame := gb.Use(matDarkWood())
	fx0, fx1, fy0, fy1 := -0.72, 0.18, 1.42, 2.06
	const ft = 0.05
	frame.AddBoxUV(AABB{Min: V3(fx0, fy0, roomMinZ), Max: V3(fx1, fy0+ft, roomMinZ+0.035)}, 1)
	frame.AddBoxUV(AABB{Min: V3(fx0, fy1-ft, roomMinZ), Max: V3(fx1, fy1, roomMinZ+0.035)}, 1)
	frame.AddBoxUV(AABB{Min: V3(fx0, fy0, roomMinZ), Max: V3(fx0+ft, fy1, roomMinZ+0.035)}, 1)
	frame.AddBoxUV(AABB{Min: V3(fx1-ft, fy0, roomMinZ), Max: V3(fx1, fy1, roomMinZ+0.035)}, 1)
	art := gb.Use(Material{Albedo: V3(1, 1, 1), Spec: 0.06, Shininess: 10, Alpha: 1, Tex: TexCanvasArt})
	az := roomMinZ + 0.012
	art.AddQuadN(
		V3(fx1-ft, fy0+ft, az), V3(fx0+ft, fy0+ft, az), V3(fx0+ft, fy1-ft, az), V3(fx1-ft, fy1-ft, az),
		V3(0, 0, 1), [4]Vec2{{1, 0}, {0, 0}, {0, 1}, {1, 1}})

	// Rug on the floor: a rounded rectangle triangle-fanned from its centre.
	rug := gb.Use(matRug())
	rugC := V3(0.0, 0.008, 0.75)
	center := qv(rugC, V3(0, 1, 0), 0, 0)
	prev := Vec2{}
	for i := 0; i <= 48; i++ {
		a := 2 * math.Pi * float64(i) / 48
		cx, cz := math.Cos(a), math.Sin(a)
		rx, rz := 1.35, 0.95
		k := 1 / math.Max(math.Abs(cx)/rx, math.Abs(cz)/rz)
		cur := Vec2{cx * k, cz * k}
		if i > 0 {
			p0 := qv(rugC.Add(V3(prev.X, 0, prev.Y)), V3(0, 1, 0), prev.X, prev.Y)
			p1 := qv(rugC.Add(V3(cur.X, 0, cur.Y)), V3(0, 1, 0), cur.X, cur.Y)
			rug.AddTri(center, p0, p1)
		}
		prev = cur
	}

	// Notebook on the table (a small prop next to the glass).
	paper := gb.Use(Material{Albedo: V3(0.86, 0.84, 0.78), Spec: 0.10, Shininess: 16, Alpha: 1})
	nb := AABB{Min: V3(0.30, tableTopY, -1.28), Max: V3(0.62, tableTopY+0.028, -1.05)}
	paper.AddBoxUV(nb, 1)
	w.boxes = append(w.boxes, nb)
	cover := gb.Use(Material{Albedo: V3(0.22, 0.28, 0.42), Spec: 0.12, Shininess: 20, Alpha: 1})
	cover.AddBoxUV(AABB{Min: V3(0.295, tableTopY, -1.285), Max: V3(0.625, tableTopY+0.006, -1.045)}, 1)
	pen := gb.Use(Material{Albedo: V3(0.10, 0.10, 0.12), Spec: 0.45, Shininess: 48, Alpha: 1})
	pen.AddBoxUV(AABB{Min: V3(0.34, tableTopY+0.028, -1.20), Max: V3(0.58, tableTopY+0.040, -1.185)}, 1)
	w.boxes = append(w.boxes, AABB{Min: V3(0.34, tableTopY, -1.21), Max: V3(0.58, tableTopY+0.04, -1.18)})

	w.static = append(w.static, gb.groups...)
}

// buildGlass builds the drinking glass and the water inside it (local space,
// origin at the bottom centre).
func (w *World) buildGlass() {
	profile := glassProfile()
	g := newMesh()
	g.AddLathe(profile, 26, 1)
	w.glass = g

	// Water: a tapered plug matching the inner wall, capped on top.
	innerR := func(y float64) float64 {
		t := Clamp01((y - glassBaseY) / (glassH - glassBaseY))
		return glassRBottom - glassWall + t*((glassRTop-glassWall)-(glassRBottom-glassWall))
	}
	wm := newMesh()
	top := w.waterLevel
	prof := []LathePoint{
		{0, glassBaseY},
		{innerR(glassBaseY) - 0.0012, glassBaseY},
		{innerR(top) - 0.0012, top},
		{0, top},
	}
	// Build the side wall + bottom, then the top cap.
	side := newMesh()
	side.AddLathe(prof, 24, 1)
	wm.Append(side)
	wm.AddDiscY(0, 0, top, innerR(top)-0.0012, 24, true)
	w.water = wm
}

// glassWallProfile returns the outer wall polyline of the glass (r, y), which
// the shatter code slices into pieces.
func glassWallProfile() []LathePoint {
	return []LathePoint{
		{glassRBottom, 0},
		{glassRBottom, 0.007},
		{glassRTop, glassH},
	}
}

// glassProfile returns the lathe profile of the drinking glass (r, y).
func glassProfile() []LathePoint {
	return []LathePoint{
		{0.0000, 0.0000},
		{glassRBottom, 0.0000},
		{glassRBottom, 0.0070},
		{glassRTop, glassH},
		{glassRTop - glassWall, glassH},
		{glassRBottom - glassWall, glassBaseY},
		{glassRBottom - glassWall, glassBaseY - 0.0015},
		{0.0000, glassBaseY - 0.0015},
	}
}

// ceilingLamp adds the ceiling fixture (mesh + point light).
func (w *World) ceilingLamp() {
	gb := &GeoBuilder{}
	shade := gb.Use(Material{Albedo: V3(0.92, 0.90, 0.86), Spec: 0.30, Shininess: 40, Alpha: 1, DoubleSided: true})
	prof := []LathePoint{
		{0.001, 0.10},
		{0.16, 0.0},
		{0.165, 0.0},
		{0.02, 0.115},
		{0.0, 0.115},
	}
	sm := newMesh()
	sm.AddLathe(prof, 20, 1)
	sm.Transform(Translate(V3(0, roomH-0.16, 0)))
	shade.Append(sm)

	bulb := gb.Use(Material{Albedo: V3(1.0, 0.92, 0.72), Unlit: true, Emissive: V3(1.5, 1.32, 0.95)})
	bm := newMesh()
	bm.AddSphere(V3(0, roomH-0.19, 0), 0.045, 12, 8)
	bulb.Append(bm)
	// A little ceiling rose.
	trim := gb.Use(matTrim())
	trim.AddBoxUV(AABB{Min: V3(-0.06, roomH-0.03, -0.06), Max: V3(0.06, roomH, 0.06)}, 1)

	w.static = append(w.static, gb.groups...)
	w.lights = []PointLight{
		{Pos: V3(0, roomH-0.22, 0), Color: V3(1.0, 0.87, 0.70), Range: 5.4, Power: 1.05, Enabled: true},
		{Pos: V3(2.25, 1.50, -0.7), Color: V3(0.62, 0.70, 0.88), Range: 3.2, Power: 0.30, Enabled: true},
	}
}

// ---------------------------------------------------------------------------
// Rendering the world
// ---------------------------------------------------------------------------

// SubmitStatic queues the room and props, plus their ground shadows.
func (w *World) SubmitStatic(r *Renderer) {
	for i := range w.static {
		r.Submit(w.static[i].mesh, Identity4(), w.static[i].mat)
	}
	// Ground shadows projected along the sun direction.
	for _, b := range w.boxes {
		r.AddBoxShadow(b, w.dir.Dir, 0.004, 0.34)
	}
	// Daylight pool where the window throws light onto the floor.
	w.addWindowPool(r)

	if !w.glassBroken {
		r.Submit(w.glass, Translate(w.glassBase), matGlass())
		r.Submit(w.water, Translate(w.glassBase), matWater())
	} else {
		if w.puddleR > 0.001 {
			pm := newMesh()
			pm.AddDiscY(w.glassBase.X, w.glassBase.Z, tableTopY+0.006, w.puddleR, 20, true)
			r.Submit(pm, Identity4(), Material{Albedo: V3(0.30, 0.50, 0.62), Spec: 0.85, Shininess: 128, Alpha: 0.55, Refl: 0.35})
		}
	}
	if len(w.decalPt) > 0 {
		m := newMesh()
		for _, d := range w.decalPt {
			m.AddBulletHole(d.p, d.n, d.size)
		}
		r.Submit(m, Identity4(), Material{Albedo: V3(0.05, 0.045, 0.04), Spec: 0.0, Alpha: 1, Unlit: true})
	}
}

// addWindowPool projects the window opening onto the floor along the sun.
func (w *World) addWindowPool(r *Renderer) {
	const wz0, wz1, wy0, wy1 = -1.55, 0.15, 0.95, 2.15
	var pts []Vec2
	for _, y := range []float64{wy0, wy1} {
		for _, z := range []float64{wz0, wz1} {
			p := V3(roomMaxX, y, z)
			t := (p.Y - 0.005) / w.dir.Dir.Y
			q := p.Sub(w.dir.Dir.Mul(t))
			pts = append(pts, Vec2{q.X, q.Z})
		}
	}
	hull := convexHull2D(pts)
	mat := Material{Albedo: V3(0.22, 0.20, 0.17), Unlit: true, Alpha: 0.5, Additive: true}
	r.AddGroundPoly(hull, 0.0035, mat)
}

// ---------------------------------------------------------------------------
// Hit testing
// ---------------------------------------------------------------------------

// Raycast returns the closest hit against the room and props.
func (w *World) Raycast(ro, rd Vec3) RayHit {
	best := RayHit{Kind: hitNone, T: math.Inf(1)}

	consider := func(t float64, p, n Vec3, kind int) {
		if t > 1e-4 && t < best.T {
			best = RayHit{Hit: true, T: t, P: p, N: n, Kind: kind}
		}
	}

	// Floor / ceiling.
	if rd.Y < -1e-9 {
		t := (0 - ro.Y) / rd.Y
		p := ro.Add(rd.Mul(t))
		if p.X > roomMinX && p.X < roomMaxX && p.Z > roomMinZ && p.Z < roomMaxZ {
			consider(t, p, V3(0, 1, 0), hitFloor)
		}
	} else if rd.Y > 1e-9 {
		t := (roomH - ro.Y) / rd.Y
		p := ro.Add(rd.Mul(t))
		if p.X > roomMinX && p.X < roomMaxX && p.Z > roomMinZ && p.Z < roomMaxZ {
			consider(t, p, V3(0, -1, 0), hitCeiling)
		}
	}
	// Walls (including the window reveal boxes so shots can hit the frame).
	wallBoxes := []AABB{
		{Min: V3(roomMinX - 0.06, -0.1, roomMinZ - 0.06), Max: V3(roomMaxX + 0.06, roomH, roomMinZ)},
		{Min: V3(roomMinX - 0.06, -0.1, roomMaxZ), Max: V3(roomMaxX + 0.06, roomH, roomMaxZ + 0.06)},
		{Min: V3(roomMinX - 0.06, -0.1, roomMinZ - 0.06), Max: V3(roomMinX, roomH, roomMaxZ + 0.06)},
		{Min: V3(roomMaxX, -0.1, roomMinZ - 0.06), Max: V3(roomMaxX + 0.06, roomH, -1.55)},
		{Min: V3(roomMaxX, -0.1, 0.15), Max: V3(roomMaxX + 0.06, roomH, roomMaxZ + 0.06)},
		{Min: V3(roomMaxX, -0.1, -1.55), Max: V3(roomMaxX + 0.06, 0.95, 0.15)},
		{Min: V3(roomMaxX, 2.15, -1.55), Max: V3(roomMaxX + 0.06, roomH, 0.15)},
		// The window pane stops bullets (and takes a bullet hole) instead of
		// letting shots escape the room through the opening.
		{Min: V3(roomMaxX + 0.02, 0.95, -1.55), Max: V3(roomMaxX + 0.045, 2.15, 0.15)},
	}
	for _, b := range wallBoxes {
		if t, ok := rayAABB(ro, rd, b); ok {
			p := ro.Add(rd.Mul(t))
			consider(t, p, aabbNormal(b, p), hitWall)
		}
	}
	for _, b := range w.boxes {
		if t, ok := rayAABB(ro, rd, b); ok {
			p := ro.Add(rd.Mul(t))
			consider(t, p, aabbNormal(b, p), hitProp)
		}
	}
	// Glass and water.
	if !w.glassBroken {
		if t, n, ok := rayCylinder(ro, rd, w.glassBase.X, w.glassBase.Z, glassR, w.glassBase.Y, w.glassBase.Y+glassH); ok {
			consider(t, ro.Add(rd.Mul(t)), n, hitGlass)
		}
		if t, n, ok := rayCylinder(ro, rd, w.glassBase.X, w.glassBase.Z, 0.030, w.glassBase.Y+0.015, w.glassBase.Y+w.waterLevel); ok {
			consider(t, ro.Add(rd.Mul(t)), n, hitWater)
		}
	}
	// Shards are shootable once airborne (a nice touch when sweeping the room).
	for _, s := range w.shards {
		if s.dead {
			continue
		}
		b := s.Bounds()
		if t, ok := rayAABB(ro, rd, b); ok {
			p := ro.Add(rd.Mul(t))
			consider(t, p, aabbNormal(b, p), hitShard)
		}
	}
	return best
}

// Solid reports whether a cylinder at (x,z) with the given radius overlaps world geometry.
func (w *World) Solid(x, z, radius, y, height float64) bool {
	if x-radius < roomMinX+0.02 || x+radius > roomMaxX-0.02 ||
		z-radius < roomMinZ+0.02 || z+radius > roomMaxZ-0.02 {
		return true
	}
	playerBox := AABB{Min: V3(x - radius, y, z - radius), Max: V3(x + radius, y + height, z + radius)}
	for _, b := range w.boxes {
		if b.Min.Y > playerBox.Max.Y || b.Max.Y < playerBox.Min.Y {
			continue
		}
		if playerBox.Min.X < b.Max.X && playerBox.Max.X > b.Min.X &&
			playerBox.Min.Z < b.Max.Z && playerBox.Max.Z > b.Min.Z {
			return true
		}
	}
	return false
}
