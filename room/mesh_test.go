package main

import (
	"math"
	"testing"
)

// windingReport counts triangles whose geometric orientation disagrees with
// their shading normal. Such a triangle is either culled when it should be
// visible or lit from the wrong side, so this must always be zero.
func windingReport(m *Mesh) (bad, checked int) {
	for i := 0; i+2 < len(m.Idx); i += 3 {
		a, b, c := m.Verts[m.Idx[i]], m.Verts[m.Idx[i+1]], m.Verts[m.Idx[i+2]]
		geo := b.Pos.Sub(a.Pos).Cross(c.Pos.Sub(a.Pos))
		if geo.LenSq() < 1e-18 {
			continue
		}
		checked++
		ref := a.Nrm.Add(b.Nrm).Add(c.Nrm)
		if ref.LenSq() > 1e-18 && ref.Normalize().Dot(geo.Normalize()) < -0.3 {
			bad++
		}
	}
	return bad, checked
}

func itoa(v int) string {
	if v == 0 {
		return "0"
	}
	var b []byte
	for v > 0 {
		b = append([]byte{byte('0' + v%10)}, b...)
		v /= 10
	}
	return string(b)
}

func checkMesh(t *testing.T, name string, m *Mesh) {
	t.Helper()
	bad, checked := windingReport(m)
	if checked == 0 {
		t.Errorf("%s: no valid triangles", name)
	}
	if bad != 0 {
		t.Errorf("%s: %d of %d triangles wound against their normals", name, bad, checked)
	}
}

func TestPrimitiveWinding(t *testing.T) {
	m := newMesh()
	m.AddBoxUV(AABB{Min: V3(-1, -1, -1), Max: V3(1, 1, 1)}, 1)
	checkMesh(t, "box", m)

	m = newMesh()
	m.AddLathe(glassProfile(), 24, 1)
	checkMesh(t, "lathe(glass)", m)

	m = newMesh()
	m.AddLathe([]LathePoint{{0, 0}, {0.03, 0}, {0.034, 0.13}, {0.031, 0.13}, {0.028, 0.015}, {0, 0.015}}, 24, 1)
	checkMesh(t, "lathe(closed cup)", m)

	m = newMesh()
	m.AddDiscY(0, 0, 0, 1, 20, true)
	checkMesh(t, "disc up", m)
	m = newMesh()
	m.AddDiscY(0, 0, 0, 1, 20, false)
	checkMesh(t, "disc down", m)

	m = newMesh()
	m.AddRingY(0, 0, 0, 0.5, 1, 16, true)
	checkMesh(t, "ring up", m)

	m = newMesh()
	m.AddSphere(V3(0, 0, 0), 1, 12, 8)
	checkMesh(t, "sphere", m)

	m = newMesh()
	m.AddBillboard(V3(0, 0, 0), V3(1, 0, 0), V3(0, 1, 0), 1, 1, 1)
	checkMesh(t, "billboard", m)

	m = newMesh()
	m.AddBulletHole(V3(0, 0, 0), V3(0, 0, 1), 0.1)
	checkMesh(t, "bullet hole", m)

	m = newMesh()
	m.AddPlaneY(V3(0, 0, 0), 2, 3, true, 1)
	checkMesh(t, "plane up", m)
	m = newMesh()
	m.AddPlaneY(V3(0, 0, 0), 2, 3, false, 1)
	checkMesh(t, "plane down", m)
}

// The room must be built from surfaces that are all visible from inside it.
// Double-sided materials (the lamp shade) may legitimately be wound either way.
func TestRoomShellIsClosedAndFacingInward(t *testing.T) {
	w := NewWorld()
	var all Mesh
	for _, g := range w.static {
		all.Append(g.mesh)
		if g.mat.DoubleSided {
			continue
		}
		checkMesh(t, "room group (tex="+itoa(g.mat.Tex)+")", g.mesh)
	}
	b := all.Bounds()
	if b.Min.X > roomMinX+0.01 || b.Max.X < roomMaxX-0.01 ||
		b.Min.Z > roomMinZ+0.01 || b.Max.Z < roomMaxZ-0.01 {
		t.Errorf("room shell does not enclose the room: %+v", b)
	}
	if b.Max.Y < roomH-0.01 {
		t.Errorf("room has no ceiling (max y = %.2f)", b.Max.Y)
	}

	// A camera in the middle of the room must see geometry in every direction.
	p := NewPlayer()
	p.pos = V3(0.25, 0, 1.75)
	r := NewRenderer(160, 90, false)
	for _, yaw := range []float64{0, math.Pi / 2, math.Pi, -math.Pi / 2} {
		for _, pitch := range []float64{-0.5, 0, 0.5} {
			p.Yaw, p.Pitch = yaw, pitch
			r.SetCamera(p.Camera(16.0 / 9.0))
			r.Reset(V3(0, 0, 0))
			r.SetLights(w.lights, w.dir)
			w.SubmitStatic(r)
			r.Flush()
			if r.Stats.Shaded < r.W*r.H/4 {
				t.Errorf("looking yaw=%.2f pitch=%.2f the room covers only %d of %d pixels",
					yaw, pitch, r.Stats.Shaded, r.W*r.H)
			}
		}
	}
}

// Every solid prop must be reachable by bullets and must block the player.
func TestPropsAreRaycastableAndSolid(t *testing.T) {
	w := NewWorld()
	// The table top must stop a downward ray.
	hit := w.Raycast(V3(0, 1.6, -1.0), V3(0, -1, 0))
	if !hit.Hit || hit.P.X != 0 || math.Abs(hit.P.Y-tableTopY) > 0.12 {
		t.Errorf("ray onto the table hit kind=%d at %v", hit.Kind, hit.P)
	}
	// The walls must stop shots.
	for i, dir := range []Vec3{{1, 0, 0}, {-1, 0, 0}, {0, 0, 1}, {0, 0, -1}, {0, 1, 0}} {
		if h := w.Raycast(V3(0, 1.3, 0), dir); !h.Hit {
			t.Errorf("shot %d into %v hit nothing", i, dir)
		}
	}
	// The glass must be hit from the spawn position when aiming at it.
	p := NewPlayer()
	d := w.glassBase.Add(V3(0, glassH/2, 0)).Sub(p.Eye())
	if h := w.Raycast(p.Eye(), d.Normalize()); h.Kind != hitGlass {
		t.Errorf("aiming at the glass from spawn hit kind=%d instead", h.Kind)
	}
	// The table must block the player.
	if !w.Solid(0, -1.0, 0.28, 0, 1.75) {
		t.Errorf("the table does not block the player")
	}
	if w.Solid(0.25, 1.75, 0.28, 0, 1.75) {
		t.Errorf("the spawn point is inside solid geometry")
	}
}
