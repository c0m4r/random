package main

import (
	"math"
	"testing"
)

// stepPlayer advances the player in fixed substeps like the server loop does.
func stepPlayer(p *Player, w *World, seconds float64, in Input) []string {
	var evs []string
	n := int(seconds / simStep)
	for i := 0; i < n; i++ {
		evs = append(evs, p.Update(simStep, w, in)...)
		w.Update(simStep)
		in.DX, in.DY = 0, 0
		in.Events = nil
	}
	return evs
}

func TestDrawFireShatter(t *testing.T) {
	w := NewWorld()
	p := NewPlayer()
	if p.gun != GunHolstered {
		t.Fatalf("initial gun state %v", p.gun)
	}
	// Clicking while holstered draws instead of firing.
	evs := stepPlayer(p, w, simStep*1.5, Input{Events: []string{"start"}})
	if p.gun != GunDrawing {
		t.Fatalf("gun = %v after draw, want drawing", p.gun)
	}
	if !has(evs, "draw") {
		t.Errorf("no draw event: %v", evs)
	}
	stepPlayer(p, w, 0.6, Input{})
	if p.gun != GunReady {
		t.Fatalf("gun = %v after draw animation, want ready", p.gun)
	}

	// Aim at the glass and fire.
	eye := p.Eye()
	target := w.glassBase.Add(V3(0, glassH*0.5, 0))
	d := target.Sub(eye)
	wantYaw := math.Atan2(d.X, -d.Z)
	wantPitch := math.Atan2(d.Y, math.Hypot(d.X, d.Z))
	p.Yaw = wantYaw
	p.Pitch = wantPitch
	p.spread = 0 // deterministic: test the hitscan, not the accuracy cone
	stepPlayer(p, w, simStep*1.5, Input{Events: []string{"fireDown"}})
	if !w.glassBroken {
		hit := w.Raycast(eye, p.Forward())
		t.Fatalf("glass intact after firing; raycast kind=%d t=%.3f shots=%d ammo=%d gun=%v",
			hit.Kind, hit.T, p.Shots, p.ammo, p.gun)
	}
	if p.Shots != 1 || p.Hits != 1 {
		t.Errorf("shots/hits = %d/%d, want 1/1", p.Shots, p.Hits)
	}
	if p.ammo != 7 {
		t.Errorf("ammo = %d, want 7", p.ammo)
	}
	if len(w.shards) < 20 {
		t.Errorf("only %d shards spawned", len(w.shards))
	}
	if len(w.particles) < 50 {
		t.Errorf("only %d particles spawned", len(w.particles))
	}

	// Shards must settle inside the room and eventually disappear.
	for i := 0; i < 40; i++ {
		stepPlayer(p, w, 0.1, Input{})
		for _, s := range w.shards {
			if s.pos.X < roomMinX-0.5 || s.pos.X > roomMaxX+0.5 ||
				s.pos.Z < roomMinZ-0.5 || s.pos.Z > roomMaxZ+0.5 || s.pos.Y < -0.5 {
				t.Fatalf("shard escaped the room: %v", s.pos)
			}
		}
	}
	if len(w.shards) == 0 {
		t.Errorf("all shards vanished within 4s (expected a slow fade)")
	}
}

func TestReloadAndEmpty(t *testing.T) {
	w := NewWorld()
	p := NewPlayer()
	p.gun = GunReady
	p.drawT, p.gunT = 1, 1
	for i := 0; i < 8; i++ {
		stepPlayer(p, w, 0.25, Input{Events: []string{"fireDown"}})
	}
	st := p.gun
	if p.ammo != 0 && st != GunReloading {
		t.Fatalf("after 8 shots: ammo=%d gun=%v", p.ammo, st)
	}
	// Firing on empty must not fire but should trigger a reload.
	before := p.Shots
	evs := stepPlayer(p, w, 0.05, Input{Events: []string{"fireDown"}})
	if p.Shots != before && p.ammo == 0 {
		t.Errorf("fired with an empty magazine")
	}
	if !has(evs, "empty") && p.gun != GunReloading {
		t.Errorf("empty trigger produced neither empty event nor reload: %v", evs)
	}
	stepPlayer(p, w, 1.6, Input{})
	if p.ammo != p.mag {
		t.Errorf("ammo after reload = %d, want %d", p.ammo, p.mag)
	}
	if p.gun != GunReady {
		t.Errorf("gun = %v after reload", p.gun)
	}
}

func TestPlayerStaysInRoom(t *testing.T) {
	w := NewWorld()
	p := NewPlayer()
	for _, dir := range []uint32{keyW, keyS, keyA, keyD, keyW | keyA, keyS | keyD, keyW | keyD} {
		p.pos = V3(0.25, 0, 1.75)
		p.vel = Vec3{}
		stepPlayer(p, w, 3.0, Input{Keys: dir | keyShift})
		if p.pos.X < roomMinX-0.01 || p.pos.X > roomMaxX+0.01 ||
			p.pos.Z < roomMinZ-0.01 || p.pos.Z > roomMaxZ+0.01 {
			t.Fatalf("keys %b pushed the player out of the room: %v", dir, p.pos)
		}
	}
	// Sprinting into the table must not walk through it.
	p.pos = V3(0, 0, 0.4)
	p.Yaw = 0
	p.vel = Vec3{}
	stepPlayer(p, w, 2.0, Input{Keys: keyW | keyShift})
	if p.pos.Z < -0.35 {
		t.Errorf("player walked through the table to z=%.2f", p.pos.Z)
	}
}

func TestPistolIsInFrontOfCamera(t *testing.T) {
	p := NewPlayer()
	p.gun = GunReady
	p.drawT, p.gunT = 1, 1
	cam := p.Camera(16.0 / 9.0)
	m := p.GunModel()
	muzzle := m.MulPoint(V3(0, 0.046, -0.205))
	// The muzzle must be in front of the camera and roughly where it aims.
	fwd := p.Forward()
	v := muzzle.Sub(cam.Pos)
	if v.Len() < 0.2 || v.Len() > 1.2 {
		t.Errorf("muzzle distance %.2f m is implausible", v.Len())
	}
	if v.Normalize().Dot(fwd) < 0.75 {
		t.Errorf("muzzle points away from the aim direction (dot=%.2f)", v.Normalize().Dot(fwd))
	}
	// A shot from the spawn point must hit the table or the room, not nothing.
	hit := NewWorld().Raycast(cam.Pos, fwd)
	if !hit.Hit {
		t.Errorf("aiming forward from the spawn hits nothing")
	}
}
