package main

import "math"

// GunState is the pistol's state machine.
type GunState int

// Gun states.
const (
	GunHolstered GunState = iota
	GunDrawing
	GunReady
	GunHolstering
	GunReloading
)

func (g GunState) String() string {
	switch g {
	case GunHolstered:
		return "holstered"
	case GunDrawing:
		return "drawing"
	case GunReady:
		return "ready"
	case GunHolstering:
		return "holstering"
	case GunReloading:
		return "reloading"
	}
	return "unknown"
}

// Input is a decoded input frame from the client.
type Input struct {
	Keys     uint32
	DX, DY   float64
	FireDown bool
	Events   []string
}

// Input bit assignments (must match PROTOCOL.md).
const (
	keyW = 1 << iota
	keyA
	keyS
	keyD
	keyShift
	keySpace
	keyQ
	keyR
	keyE
	keyUp
	keyLeft
	keyDown
	keyRight
)

// Player holds the camera, movement and weapon state.
type Player struct {
	pos     Vec3 // feet
	vel     Vec3
	Yaw     float64
	Pitch   float64
	eyeH    float64
	onAir   bool
	bob     float64
	bobAmt  float64
	shake   float64
	shakeT  float64

	gun       GunState
	gunT      float64 // 0 = holstered, 1 = fully drawn
	drawT     float64
	ammo      int
	mag       int
	reloadT   float64
	recoilPos float64
	recoilVel float64
	recoilRot float64
	rotVel    float64
	camKick   float64
	muzzleT   float64
	flashSeed float64
	spread    float64

	Shots int
	Hits  int

	gunMesh    *Mesh
	gunBatches []geoGroup
	flashMesh  *Mesh
	viewH      int
	tracers   []tracer
	lastFire  float64

	events []string
}

type tracer struct {
	from, to Vec3
	t        float64
	life     float64
}

// NewPlayer creates the player at the spawn point.
func NewPlayer() *Player {
	p := &Player{
		pos:   V3(0.25, 0, 1.75),
		Yaw:   0,
		Pitch: -0.10,
		eyeH:  1.62,
		mag:   8,
		ammo:  8,
	}
	p.buildGun()
	return p
}

// Eye returns the camera position.
func (p *Player) Eye() Vec3 {
	bob := math.Sin(p.bob) * 0.018 * p.bobAmt
	side := math.Cos(p.bob*0.5) * 0.010 * p.bobAmt
	r := V3(math.Cos(p.Yaw), 0, math.Sin(p.Yaw))
	return p.pos.Add(V3(r.X*side, p.eyeH+bob, r.Z*side))
}

// Forward returns the aim direction.
func (p *Player) Forward() Vec3 {
	cp := math.Cos(p.Pitch)
	return V3(math.Sin(p.Yaw)*cp, math.Sin(p.Pitch), -math.Cos(p.Yaw)*cp)
}

// Camera builds the frame camera for the renderer.
func (p *Player) Camera(aspect float64) Camera {
	return Camera{
		Pos:    p.Eye(),
		Yaw:    p.Yaw,
		Pitch:  p.Pitch + p.camKick,
		Roll:   p.shake * 0.035 * math.Sin(p.shakeT*47),
		FovY:   Degrees(72),
		Near:   0.02,
		Far:    80,
		Aspect: aspect,
	}
}

// ---------------------------------------------------------------------------
// Gun viewmodel
// ---------------------------------------------------------------------------

func (p *Player) buildGun() {
	gb := &GeoBuilder{}
	metal := gb.Use(matMetalGun())
	poly := gb.Use(matPolymer())

	// Slide + frame, muzzle toward +Z.
	metal.AddBoxUV(AABB{Min: V3(-0.0135, 0.028, 0.000), Max: V3(0.0135, 0.063, 0.190)}, 1)
	metal.AddBoxUV(AABB{Min: V3(-0.010, 0.048, 0.190), Max: V3(0.010, 0.060, 0.205)}, 1)
	// Serrations at the rear of the slide.
	for i := 0; i < 5; i++ {
		z := 0.008 + float64(i)*0.009
		metal.AddBoxUV(AABB{Min: V3(-0.0138, 0.030, z), Max: V3(0.0138, 0.061, z+0.004)}, 1)
	}
	// Frame + dust cover.
	poly.AddBoxUV(AABB{Min: V3(-0.012, 0.010, 0.008), Max: V3(0.012, 0.030, 0.176)}, 1)
	// Trigger guard.
	poly.AddBoxUV(AABB{Min: V3(-0.006, -0.020, 0.062), Max: V3(0.006, 0.014, 0.074)}, 1)
	poly.AddBoxUV(AABB{Min: V3(-0.006, -0.020, 0.020), Max: V3(0.006, 0.014, 0.032)}, 1)
	poly.AddBoxUV(AABB{Min: V3(-0.006, -0.032, 0.020), Max: V3(0.006, -0.018, 0.074)}, 1)
	metal.AddBoxUV(AABB{Min: V3(-0.004, -0.018, 0.036), Max: V3(0.004, 0.008, 0.044)}, 1)
	// Grip, raked back.
	grip := newMesh()
	grip.AddBoxUV(AABB{Min: V3(-0.0125, -0.062, -0.026), Max: V3(0.0125, 0.062, 0.026)}, 1)
	grip.Transform(RotX(0.30))
	grip.Transform(Translate(V3(0, -0.030, 0.004)))
	poly.Append(grip)
	// Magazine floor plate.
	plate := newMesh()
	plate.AddBoxUV(AABB{Min: V3(-0.013, -0.008, -0.028), Max: V3(0.013, 0.008, 0.028)}, 1)
	plate.Transform(RotX(0.30))
	plate.Transform(Translate(V3(0, -0.098, -0.023)))
	metal.Append(plate)
	// Sights.
	metal.AddBoxUV(AABB{Min: V3(-0.0025, 0.063, 0.168), Max: V3(0.0025, 0.070, 0.176)}, 1)
	metal.AddBoxUV(AABB{Min: V3(-0.008, 0.063, 0.006), Max: V3(-0.003, 0.070, 0.014)}, 1)
	metal.AddBoxUV(AABB{Min: V3(0.003, 0.063, 0.006), Max: V3(0.008, 0.070, 0.014)}, 1)
	// Hammer.
	metal.AddBoxUV(AABB{Min: V3(-0.005, 0.050, -0.014), Max: V3(0.005, 0.062, -0.002)}, 1)
	// Ejection port.
	dark := gb.Use(Material{Albedo: V3(0.045, 0.045, 0.05), Spec: 0.3, Shininess: 30, Alpha: 1})
	dark.AddBoxUV(AABB{Min: V3(0.008, 0.042, 0.070), Max: V3(0.0142, 0.058, 0.116)}, 1)
	// Bore.
	barrel := newMesh()
	barrel.AddLathe([]LathePoint{{0, 0}, {0.0075, 0}, {0.0075, 0.004}, {0.0035, 0.004}, {0.0035, 0.012}, {0, 0.012}}, 14, 1)
	barrel.Transform(RotX(math.Pi / 2))
	barrel.Transform(Translate(V3(0, 0.046, 0.186)))
	dark.Append(barrel)

	// Reorient the pistol into camera-local space: muzzle toward -Z (the
	// direction the camera looks), +X = left, +Y = up.
	for i := range gb.groups {
		gb.groups[i].mesh.Transform(RotY(math.Pi))
	}
	p.gunMesh = newMesh()
	for i := range gb.groups {
		p.gunMesh.Append(gb.groups[i].mesh)
	}
	// Materials differ per batch, so keep the batches alongside the merged mesh.
	p.gunBatches = gb.groups

	// Muzzle flash: two crossed additive quads plus a bright core.
	fm := newMesh()
	q1 := newMesh()
	q1.AddBillboard(V3(0, 0, -0.026), V3(1, 0, 0), V3(0, 1, 0), 0.030, 0.017, 1)
	q2 := newMesh()
	q2.AddBillboard(V3(0, 0, -0.026), V3(0, 1, 0), V3(0.35, 0, 1).Normalize(), 0.030, 0.017, 1)
	fm.Append(q1)
	fm.Append(q2)
	fm.AddSphere(V3(0, 0, -0.012), 0.010, 8, 6)
	p.flashMesh = fm
}

// ---------------------------------------------------------------------------
// Update
// ---------------------------------------------------------------------------

// Update advances the player and returns sound events.
func (p *Player) Update(dt float64, w *World, in Input) []string {
	p.events = p.events[:0]

	// Mouse look.
	p.Yaw += in.DX * 0.0022
	p.Pitch -= in.DY * 0.0022
	p.Pitch = Clamp(p.Pitch, -1.45, 1.45)

	for _, e := range in.Events {
		switch e {
		case "fireDown":
			p.triggerDown(w)
		case "toggle":
			p.toggleGun()
		case "reload":
			p.startReload()
		case "start":
			if p.gun == GunHolstered || p.gun == GunHolstering {
				p.drawGun()
			}
		}
	}

	// Movement.
	dir := V3(0, 0, 0)
	fwd := V3(math.Sin(p.Yaw), 0, -math.Cos(p.Yaw))
	right := V3(math.Cos(p.Yaw), 0, math.Sin(p.Yaw))
	if in.Keys&(keyW|keyUp) != 0 {
		dir = dir.Add(fwd)
	}
	if in.Keys&(keyS|keyDown) != 0 {
		dir = dir.Sub(fwd)
	}
	if in.Keys&(keyD|keyRight) != 0 {
		dir = dir.Add(right)
	}
	if in.Keys&(keyA|keyLeft) != 0 {
		dir = dir.Sub(right)
	}
	speed := 3.05
	if in.Keys&keyShift != 0 {
		speed = 4.5
	}
	if p.gun == GunReady || p.gun == GunDrawing {
		speed *= 0.94
	}
	if dir.LenSq() > 1e-6 {
		dir = dir.Normalize()
		accel := 26.0
		if !p.onAir {
			accel = 42.0
		}
		p.vel.X += dir.X * accel * dt
		p.vel.Z += dir.Z * accel * dt
	} else if !p.onAir {
		// Ground friction.
		f := math.Max(0, 1-11*dt)
		p.vel.X *= f
		p.vel.Z *= f
	}
	// Clamp horizontal speed.
	hs := math.Hypot(p.vel.X, p.vel.Z)
	if hs > speed {
		k := speed / hs
		p.vel.X *= k
		p.vel.Z *= k
	}

	// Gravity + jump.
	p.vel.Y -= 13.5 * dt
	if in.Keys&keySpace != 0 && !p.onAir {
		p.vel.Y = 3.5
		p.onAir = true
		p.events = append(p.events, "jump")
	}

	// Integrate with axis-separated collision so we slide along walls.
	const radius = 0.28
	step := func(dx, dz float64) {
		if dx != 0 && !w.Solid(p.pos.X+dx, p.pos.Z, radius, p.pos.Y, 1.75) {
			p.pos.X += dx
		} else if dx != 0 {
			p.vel.X = 0
		}
		if dz != 0 && !w.Solid(p.pos.X, p.pos.Z+dz, radius, p.pos.Y, 1.75) {
			p.pos.Z += dz
		} else if dz != 0 {
			p.vel.Z = 0
		}
	}
	step(p.vel.X*dt, 0)
	step(0, p.vel.Z*dt)
	p.pos.Y += p.vel.Y * dt
	if p.pos.Y <= 0 {
		p.pos.Y = 0
		if p.vel.Y < -3.5 {
			p.events = append(p.events, "land")
		}
		p.vel.Y = 0
		p.onAir = false
	}

	// Head bob.
	spd := math.Hypot(p.vel.X, p.vel.Z)
	target := Clamp01(spd / 3.4)
	p.bobAmt += (target - p.bobAmt) * math.Min(1, dt*7)
	p.bob += dt * (5.2 + spd*1.15)
	if p.onAir {
		p.bobAmt *= 0.4
	}

	// Weapon state machine.
	switch p.gun {
	case GunDrawing:
		p.drawT += dt / 0.42
		if p.drawT >= 1 {
			p.drawT = 1
			p.gun = GunReady
		}
	case GunHolstering:
		p.drawT -= dt / 0.36
		if p.drawT <= 0 {
			p.drawT = 0
			p.gun = GunHolstered
		}
	case GunReloading:
		p.reloadT -= dt
		if p.reloadT <= 0 {
			p.ammo = p.mag
			p.gun = GunReady
			p.events = append(p.events, "reloadDone")
		}
	}
	p.gunT = EaseInOutCubic(p.drawT)

	// Recoil springs (weapon + camera).
	spring := func(x, v float64, k, d float64) (float64, float64) {
		a := -k*x - d*v
		v += a * dt
		return x + v*dt, v
	}
	p.recoilPos, p.recoilVel = spring(p.recoilPos, p.recoilVel, 190, 19)
	p.recoilRot, p.rotVel = spring(p.recoilRot, p.rotVel, 210, 20)
	p.camKick = Lerp(p.camKick, 0, math.Min(1, dt*7.5))
	p.shake = Lerp(p.shake, 0, math.Min(1, dt*6))
	p.shakeT += dt

	if p.muzzleT > 0 {
		p.muzzleT -= dt
	}

	// Crosshair spread: base + movement + recoil.
	targetSpread := 2.6 + p.bobAmt*6.5 + math.Abs(p.recoilPos)*16 + p.shake*9
	p.spread = Lerp(p.spread, targetSpread, math.Min(1, dt*12))

	// Tracers fade.
	keep := p.tracers[:0]
	for i := range p.tracers {
		p.tracers[i].t += dt
		if p.tracers[i].t < p.tracers[i].life {
			keep = append(keep, p.tracers[i])
		}
	}
	p.tracers = keep

	p.lastFire += dt
	return p.events
}

func (p *Player) drawGun() {
	if p.gun == GunReady || p.gun == GunDrawing {
		return
	}
	p.gun = GunDrawing
	if p.drawT <= 0 {
		p.drawT = 0
	}
	p.events = append(p.events, "draw")
}

func (p *Player) holsterGun() {
	if p.gun == GunHolstered || p.gun == GunHolstering {
		return
	}
	p.gun = GunHolstering
	p.events = append(p.events, "holster")
}

func (p *Player) toggleGun() {
	if p.gun == GunReady || p.gun == GunDrawing || p.gun == GunReloading {
		p.holsterGun()
	} else {
		p.drawGun()
	}
}

func (p *Player) startReload() {
	if p.gun != GunReady || p.ammo == p.mag {
		return
	}
	p.gun = GunReloading
	p.reloadT = 1.25
	p.events = append(p.events, "reload")
}

// triggerDown fires the pistol (drawing it first if it is still holstered).
func (p *Player) triggerDown(w *World) {
	switch p.gun {
	case GunHolstered, GunHolstering:
		p.drawGun()
		return
	case GunDrawing, GunReloading:
		return
	}
	if p.ammo <= 0 {
		p.events = append(p.events, "empty")
		p.startReload()
		return
	}
	p.fire(w)
}

func (p *Player) fire(w *World) {
	// Accuracy is sampled *before* this shot's own recoil widens the cone.
	coneSpread := p.spread

	p.ammo--
	p.Shots++
	p.lastFire = 0
	p.muzzleT = 0.055
	p.flashSeed = hash1(float64(p.Shots) * 7.31)

	// Recoil impulses.
	p.recoilVel += 3.4
	p.rotVel += 9.0
	p.camKick += 0.030
	p.shake = math.Min(1.0, p.shake+0.85)
	p.spread = math.Min(20, p.spread+6)

	eye := p.Eye()
	dir := p.Forward()

	// Spread cone (in radians) derived from the crosshair size in pixels.
	h := p.viewH
	if h < 16 {
		h = 540
	}
	ang := (coneSpread / (float64(h) / 2)) * math.Tan(Degrees(72)/2)
	if ang > 0 {
		a := hash1(float64(p.Shots)*1.7) * 2 * math.Pi
		r := math.Sqrt(hash1(float64(p.Shots)*3.9)) * ang
		right := V3(math.Cos(p.Yaw), 0, math.Sin(p.Yaw))
		dir = dir.Add(right.Mul(math.Cos(a) * r)).Add(V3(0, 1, 0).Mul(math.Sin(a) * r)).Normalize()
	}

	muzzle := p.muzzleWorld()
	hit := w.Raycast(eye, dir)
	end := eye.Add(dir.Mul(40))
	if hit.Hit {
		end = hit.P
	}

	p.events = append(p.events, "shot")
	w.SpawnMuzzleSmoke(muzzle, dir)
	casingVel := V3(math.Cos(p.Yaw), 0, math.Sin(p.Yaw)).Mul(1.7 + hash1(float64(p.Shots)*5.1)*0.9)
	casingVel = casingVel.Add(V3(0, 1.5+hash1(float64(p.Shots)*2.3)*0.8, 0))
	eject := p.GunModel().MulPoint(V3(-0.012, 0.052, -0.075))
	w.SpawnCasing(eject, casingVel)

	switch {
	case !hit.Hit:
		// Nothing hit: the tracer just flies off.
	case hit.Kind == hitGlass || hit.Kind == hitWater:
		w.Shatter(hit.P, dir)
		p.Hits++
		p.events = append(p.events, "shatter", "hit")
	case hit.Kind == hitShard:
		w.SpawnImpact(hit.P, hit.N, false)
		p.Hits++
		p.events = append(p.events, "impact", "hit")
	default:
		soft := hit.Kind == hitProp
		w.SpawnImpact(hit.P, hit.N, soft)
		if len(w.decalPt) < 40 {
			w.decalPt = append(w.decalPt, decalRec{p: hit.P, n: hit.N, size: 0.012})
		}
		p.events = append(p.events, "impact")
		if hit.Kind == hitWall || hit.Kind == hitCeiling {
			p.events = append(p.events, "ricochet")
		}
	}

	p.tracers = append(p.tracers, tracer{from: muzzle.Add(dir.Mul(1.1)), to: end, life: 0.042})
	if p.ammo == 0 {
		p.startReload()
	}
}

// muzzleWorld returns the world position of the muzzle.
func (p *Player) muzzleWorld() Vec3 {
	m := p.GunModel()
	return m.MulPoint(V3(0, 0.046, -0.205))
}

// GunModel returns the viewmodel transform in world space.
func (p *Player) GunModel() Mat4 {
	cam := p.Camera(1)
	base := Translate(cam.Pos).Mul(cam.Rotation())
	t := p.gunT
	// Draw/holster: drop the pistol below the view and tilt it.
	drop := (1 - t) * 0.55
	roll := (1 - t) * 0.6
	yaw := -0.05 + (1-t)*0.35
	pitch := 0.02 + (1-t)*0.5

	recoilBack := p.recoilPos * 0.034
	recoilUp := p.recoilRot * 0.022

	// Idle sway from walking.
	sway := math.Sin(p.bob) * 0.012 * p.bobAmt
	sway2 := math.Cos(p.bob*0.5) * 0.010 * p.bobAmt

	// The camera looks down -Z in its own frame, so the viewmodel sits at -Z;
	// recoil pushes it back towards the shooter (+Z). A longer distance plus a
	// slight scale-up flattens the perspective the way a viewmodel FOV does.
	off := V3(0.255+sway2, -0.205+sway-drop+recoilUp, -0.50+recoilBack)
	return base.Mul(Translate(off)).Mul(Euler(yaw, pitch, roll)).Mul(Scale(V3(1.18, 1.18, 1.18)))
}

// SubmitGun queues the viewmodel, muzzle flash and tracers.
func (p *Player) SubmitGun(r *Renderer, w *World) {
	if p.gun == GunHolstered && p.gunT <= 0.001 {
		return
	}
	model := p.GunModel()
	for i := range p.gunBatches {
		r.Submit(p.gunBatches[i].mesh, model, p.gunBatches[i].mat)
	}
	if p.muzzleT > 0 {
		k := Clamp01(p.muzzleT / 0.055)
		s := 0.75 + hash1(p.flashSeed*31.7)*0.55
		m := model.Mul(Translate(V3(0, 0.046, -0.212))).Mul(Scale(V3(s, s, s*(1.1+k*0.5))))
		r.Submit(p.flashMesh, m, Material{Albedo: V3(2.2, 1.55, 0.65), Unlit: true, Additive: true, Alpha: k, DoubleSided: true})
	}
	// Tracers: camera-facing ribbons.
	cam := r.Camera()
	right := cam.right()
	for _, tr := range p.tracers {
		a := 1 - tr.t/tr.life
		m := newMesh()
		d := tr.to.Sub(tr.from)
		wd := right.Mul(0.0035)
		m.AddQuadN(
			tr.from.Sub(wd), tr.from.Add(wd), tr.to.Add(wd), tr.to.Sub(wd),
			d.Cross(right).Normalize(),
			[4]Vec2{{0, 0}, {1, 0}, {1, 1}, {0, 1}})
		r.Submit(m, Identity4(), Material{Albedo: V3(1.5, 1.15, 0.6), Unlit: true, Additive: true, Alpha: a * 0.5, DoubleSided: true})
	}
}

// ViewmodelLight returns a light rigged to the camera so the pistol reads well.
func (p *Player) ViewmodelLight() PointLight {
	cam := p.Camera(1)
	f := p.Forward()
	r := V3(math.Cos(p.Yaw), 0, math.Sin(p.Yaw))
	pos := cam.Pos.Add(f.Mul(0.25)).Add(r.Mul(0.25)).Add(V3(0, 0.30, 0))
	return PointLight{Pos: pos, Color: V3(1.0, 0.95, 0.88), Range: 1.5, Power: 0.55, Enabled: true}
}

// FlashLight is the transient light emitted by a shot.
func (p *Player) FlashLight() PointLight {
	cam := p.Camera(1)
	f := p.Forward()
	k := Clamp01(p.muzzleT / 0.055)
	return PointLight{
		Pos:     cam.Pos.Add(f.Mul(0.45)),
		Color:   V3(1.25, 0.88, 0.48),
		Range:   3.6,
		Power:   2.1 * k * k,
		Enabled: p.muzzleT > 0,
	}
}

// Hint returns the status line shown in the HUD.
func (p *Player) Hint(w *World) string {
	switch p.gun {
	case GunHolstered:
		return "Press Q (or right-click) to draw the pistol"
	case GunDrawing:
		return "Drawing…"
	case GunReloading:
		return "Reloading…"
	case GunHolstering:
		return "Holstering…"
	}
	if w.glassBroken {
		return "Glass destroyed — nice shot. Press F to reset the room"
	}
	return "Aim at the glass on the table and fire"
}
