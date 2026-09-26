package main

import (
	"testing"
	"time"
)

func TestPhaseTiming(t *testing.T) {
	g := NewGame(960, 540, true)
	g.player.gun = GunReady
	g.player.drawT, g.player.gunT = 1, 1
	g.Draw()
	w, p, r := g.world, g.player, g.render
	n := 20
	var tReset, tSubmit, tFlush, tBytes time.Duration
	for i := 0; i < n; i++ {
		t0 := time.Now()
		r.SetCamera(p.Camera(float64(r.W) / float64(r.H)))
		r.Reset(V3(0.05, 0.055, 0.065))
		r.SetLights(w.lights, w.dir)
		t1 := time.Now()
		w.SubmitStatic(r)
		w.SubmitDynamic(r)
		p.SubmitGun(r, w)
		t2 := time.Now()
		r.Flush()
		t3 := time.Now()
		buf := make([]byte, r.W*r.H*4)
		r.ColorBytes(buf)
		t4 := time.Now()
		tReset += t1.Sub(t0)
		tSubmit += t2.Sub(t1)
		tFlush += t3.Sub(t2)
		tBytes += t4.Sub(t3)
	}
	f := func(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 / float64(n) }
	t.Logf("reset=%.2fms submit=%.2fms flush(raster+bloom)=%.2fms bytes=%.2fms total=%.2fms",
		f(tReset), f(tSubmit), f(tFlush), f(tBytes), f(tReset+tSubmit+tFlush+tBytes))
	t.Logf("shaded=%d of %d px (%.2fx)", r.Stats.Shaded, r.W*r.H, float64(r.Stats.Shaded)/float64(r.W*r.H))
}
