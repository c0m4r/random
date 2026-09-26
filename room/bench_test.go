package main

import (
	"testing"
	"time"
)

func benchRender(b *testing.B, w, h int, bloom bool, shatter bool) {
	g := NewGame(w, h, bloom)
	g.player.gun = GunReady
	g.player.drawT, g.player.gunT = 1, 1
	if shatter {
		hp := g.world.glassBase
		g.world.Shatter(hp, g.player.Forward())
		for i := 0; i < 40; i++ {
			g.Step(simStep, Input{})
		}
	}
	g.Draw() // warm up
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		g.Draw()
	}
}

func BenchmarkRender960(b *testing.B)  { benchRender(b, 960, 540, true, false) }
func BenchmarkRender1280(b *testing.B) { benchRender(b, 1280, 720, true, false) }
func BenchmarkRenderShards(b *testing.B) { benchRender(b, 960, 540, true, true) }

func TestFrameTiming(t *testing.T) {
	for _, cfg := range []struct {
		w, h  int
		bloom bool
	}{
		{960, 540, false}, {960, 540, true}, {1280, 720, true},
	} {
		g := NewGame(cfg.w, cfg.h, cfg.bloom)
		g.player.gun = GunReady
		g.player.drawT, g.player.gunT = 1, 1
		g.Draw()
		n := 12
		start := time.Now()
		for i := 0; i < n; i++ {
			g.Draw()
		}
		ms := float64(time.Since(start).Microseconds()) / 1000 / float64(n)
		t.Logf("%dx%d bloom=%-5v -> %.2f ms/frame (%.0f fps), %d shaded px, %d tris",
			cfg.w, cfg.h, cfg.bloom, ms, 1000/ms, g.render.Stats.Shaded, g.render.Stats.Tris)
	}
}
