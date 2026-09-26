// Command room3d renders a small 3D room — with a table, a drinking glass and a
// pistol — entirely in Go with a hand-written software rasterizer, and streams
// the frames to a browser over a WebSocket. Click to lock the pointer, press Q
// (or right-click) to draw the pistol and shoot the glass.
package main

import (
	"bytes"
	"embed"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"image"
	"image/png"
	"log"
	"math"
	"net/http"
	"os"
	"sync"
	"time"

	"github.com/gorilla/websocket"
)

//go:embed all:web
var webFS embed.FS

const (
	frameMagic   = "RGB1"
	frameHdrSize = 16
	simStep      = 1.0 / 120.0
)

// debugTrace enables a verbose wire/input trace (ROOM3D_DEBUG=1).
var debugTrace = os.Getenv("ROOM3D_DEBUG") != ""

var (
	flagAddr  = flag.String("addr", "127.0.0.1:8787", "HTTP listen address")
	flagW     = flag.Int("w", 960, "default render width")
	flagH     = flag.Int("h", 540, "default render height")
	flagMaxW  = flag.Int("maxw", 1280, "maximum render width")
	flagMaxH  = flag.Int("maxh", 720, "maximum render height")
	flagBloom = flag.Bool("bloom", true, "enable bloom post-processing")
	flagShots = flag.String("shots", "", "render verification PNGs into this directory and exit")
)

// ---------------------------------------------------------------------------
// Game state (one instance per connected client)
// ---------------------------------------------------------------------------

// Game bundles a world, a player and a renderer.
type Game struct {
	world  *World
	player *Player
	render *Renderer
	acc        float64
	pendDX     float64
	pendDY     float64
	pendEvents []string
	time       float64

	frameSeq  uint32
	frameBuf  []byte
	fpsEMA    float64
	msEMA     float64
	lastStats time.Time
	statsN    int
	statsMs   float64
	events    []string
}

// NewGame creates a fresh room for one client.
func NewGame(w, h int, bloom bool) *Game {
	g := &Game{
		world:  NewWorld(),
		player: NewPlayer(),
		render: NewRenderer(w, h, bloom),
	}
	g.player.viewH = h
	return g
}

// Reset restores the room to its initial state (the F key).
func (g *Game) Reset() {
	w := g.world
	w.glassBroken = false
	w.shards = nil
	w.particles = nil
	w.decalPt = nil
	w.puddleR = 0
	g.player.ammo = g.player.mag
	if g.player.gun == GunReloading {
		g.player.gun = GunReady
	}
	g.events = append(g.events, "reset")
}

// Step advances the simulation by dt seconds.
func (g *Game) Step(dt float64, in Input) {
	if dt > 0.10 {
		dt = 0.10
	}
	g.time += dt
	ev := g.player.Update(dt, g.world, in)
	g.world.Update(dt)
	g.events = append(g.events, ev...)
}

// Advance runs the simulation in fixed substeps. Discrete input (mouse deltas and
// one-shot events) is buffered until a substep actually consumes it, so a frame
// that lands between two substeps can never swallow a click.
func (g *Game) Advance(dt float64, in Input) {
	if dt > 0.25 {
		dt = 0.25
	}
	g.pendEvents = append(g.pendEvents, in.Events...)
	g.pendDX += in.DX
	g.pendDY += in.DY
	g.acc += dt
	n := 0
	for g.acc >= simStep && n < 40 {
		sub := Input{Keys: in.Keys, DX: g.pendDX, DY: g.pendDY, Events: g.pendEvents}
		g.Step(simStep, sub)
		g.pendDX, g.pendDY = 0, 0
		g.pendEvents = nil
		g.acc -= simStep
		n++
	}
	if n == 40 {
		g.acc = 0
	}
}

// Draw renders one frame of the current state.
func (g *Game) Draw() {
	r := g.render
	w := g.world
	p := g.player
	p.viewH = r.H

	cam := p.Camera(float64(r.W) / float64(r.H))
	r.SetCamera(cam)

	// Sky/background colour (only visible if the room ever leaks).
	r.Reset(V3(0.05, 0.055, 0.065))

	lights := make([]PointLight, 0, len(w.lights)+2)
	lights = append(lights, w.lights...)
	lights = append(lights, p.ViewmodelLight())
	if p.muzzleT > 0 {
		lights = append(lights, p.FlashLight())
	}
	r.SetLights(lights, w.dir)
	r.SetAmbient(V3(0.16, 0.165, 0.19), V3(0.055, 0.052, 0.05))

	w.SubmitStatic(r)
	w.SubmitDynamic(r)
	p.SubmitGun(r, w)
	r.Flush()
}

// FrameMessage serialises the framebuffer with its 16-byte header.
func (g *Game) FrameMessage() []byte {
	r := g.render
	need := frameHdrSize + r.W*r.H*4
	if cap(g.frameBuf) < need {
		g.frameBuf = make([]byte, need)
	}
	buf := g.frameBuf[:need]
	copy(buf[0:4], frameMagic)
	binary.LittleEndian.PutUint32(buf[4:], uint32(r.W))
	binary.LittleEndian.PutUint32(buf[8:], uint32(r.H))
	g.frameSeq++
	binary.LittleEndian.PutUint32(buf[12:], g.frameSeq)
	r.ColorBytes(buf[frameHdrSize:])
	return buf
}

// ---------------------------------------------------------------------------
// Wire messages
// ---------------------------------------------------------------------------

type inMsg struct {
	T    string   `json:"t"`
	W    int      `json:"w"`
	H    int      `json:"h"`
	K    uint32   `json:"k"`
	DX   float64  `json:"dx"`
	DY   float64  `json:"dy"`
	Ev   []string `json:"ev"`
	Fire *bool    `json:"fire"`
}

type helloMsg struct {
	T   string `json:"t"`
	W   int    `json:"w"`
	H   int    `json:"h"`
	Fov int    `json:"fov"`
	Mag int    `json:"mag"`
}

type hudMsg struct {
	T         string  `json:"t"`
	Gun       string  `json:"gun"`
	Ammo      int     `json:"ammo"`
	Mag       int     `json:"mag"`
	Spread    float64 `json:"spread"`
	FPS       float64 `json:"fps"`
	Ms        float64 `json:"ms"`
	Shots     int     `json:"shots"`
	Hits      int     `json:"hits"`
	Shattered bool    `json:"shattered"`
	Locked    bool    `json:"locked"`
	Hint      string  `json:"hint"`
}

type evsMsg struct {
	T string   `json:"t"`
	E []string `json:"e"`
}

// clientState is the shared input mailbox between the reader and the game loop.
type clientState struct {
	mu      sync.Mutex
	keys    uint32
	dx, dy  float64
	events  []string
	fire    bool
	locked  bool
	hello   bool
	wantW   int
	wantH   int
	resize  bool
	stopped bool
	acks    chan struct{}
}

func (c *clientState) push(m inMsg) {
	c.mu.Lock()
	defer c.mu.Unlock()
	switch m.T {
	case "hello":
		c.hello = true
		c.wantW, c.wantH = m.W, m.H
		c.resize = true
	case "resize":
		c.wantW, c.wantH = m.W, m.H
		c.resize = true
	case "in":
		c.keys = m.K
		c.dx += m.DX
		c.dy += m.DY
		for _, e := range m.Ev {
			if e == "start" {
				c.locked = true
			}
			if e == "stop" {
				c.locked = false
			}
		}
		if len(m.Ev) > 0 {
			c.events = append(c.events, m.Ev...)
		}
		if m.Fire != nil {
			c.fire = *m.Fire
		}
	case "ack":
		select {
		case c.acks <- struct{}{}:
		default:
		}
	case "stop":
		c.locked = false
		c.keys = 0
		c.fire = false
	}
}

// Take drains the accumulated input.
func (c *clientState) Take() (Input, bool, [2]int, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	in := Input{Keys: c.keys, DX: c.dx, DY: c.dy, Events: c.events}
	c.dx, c.dy = 0, 0
	c.events = nil
	resize := c.resize
	wh := [2]int{c.wantW, c.wantH}
	locked := c.locked
	c.resize = false
	return in, resize && c.hello, wh, locked
}

// ---------------------------------------------------------------------------
// HTTP / WebSocket
// ---------------------------------------------------------------------------

var upgrader = websocket.Upgrader{
	ReadBufferSize:  1 << 16,
	WriteBufferSize: 1 << 16,
	CheckOrigin:     func(r *http.Request) bool { return true },
}

func main() {
	flag.Parse()
	log.SetFlags(log.Ltime | log.Lmsgprefix)
	log.SetPrefix("[room3d] ")

	if *flagShots != "" {
		if err := renderShots(*flagShots); err != nil {
			log.Fatalf("shots: %v", err)
		}
		return
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/ws", handleWS)
	mux.HandleFunc("/", handleStatic)

	srv := &http.Server{
		Addr:              *flagAddr,
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
	}
	url := fmt.Sprintf("http://%s/", *flagAddr)
	log.Printf("room3d ready — open %s", url)
	log.Printf("software renderer, %dx%d default (max %dx%d), bloom=%v",
		*flagW, *flagH, *flagMaxW, *flagMaxH, *flagBloom)
	if err := srv.ListenAndServe(); err != nil {
		log.Fatalf("listen: %v", err)
	}
}

func handleStatic(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path == "/favicon.ico" {
		w.WriteHeader(http.StatusNoContent)
		return
	}
	name := "web/index.html"
	switch r.URL.Path {
	case "/", "/index.html":
		name = "web/index.html"
	case "/app.js":
		name = "web/app.js"
	case "/style.css":
		name = "web/style.css"
	default:
		http.NotFound(w, r)
		return
	}
	data, err := webFS.ReadFile(name)
	if err != nil {
		http.Error(w, "not found", http.StatusNotFound)
		return
	}
	switch {
	case len(name) > 4 && name[len(name)-4:] == ".css":
		w.Header().Set("Content-Type", "text/css; charset=utf-8")
	case len(name) > 3 && name[len(name)-3:] == ".js":
		w.Header().Set("Content-Type", "text/javascript; charset=utf-8")
	default:
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
	}
	w.Header().Set("Cache-Control", "no-store")
	_, _ = w.Write(data)
}

func clampResolution(w, h int) (int, int) {
	if w <= 0 || h <= 0 {
		w, h = *flagW, *flagH
	}
	aspect := float64(w) / float64(h)
	w, h = w/2*2, h/2*2
	if w > *flagMaxW {
		w = *flagMaxW / 2 * 2
		h = int(float64(w)/aspect) / 2 * 2
	}
	if h > *flagMaxH {
		h = *flagMaxH / 2 * 2
		w = int(float64(h)*aspect) / 2 * 2
	}
	if w < 320 {
		w = 320
	}
	if h < 200 {
		h = 200
	}
	return w, h
}

func handleWS(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("upgrade: %v", err)
		return
	}
	defer conn.Close()
	conn.SetReadLimit(1 << 20)

	cw, ch := clampResolution(*flagW, *flagH)
	g := NewGame(cw, ch, *flagBloom)
	cs := &clientState{acks: make(chan struct{}, 1)}
	done := make(chan struct{})
	var closeOnce sync.Once

	// Reader goroutine.
	go func() {
		defer closeOnce.Do(func() { close(done) })
		conn.SetReadDeadline(time.Now().Add(120 * time.Second))
		conn.SetPongHandler(func(string) error {
			return conn.SetReadDeadline(time.Now().Add(120 * time.Second))
		})
		for {
			_, data, err := conn.ReadMessage()
			if err != nil {
				if debugTrace {
					log.Printf("reader exit: %v", err)
				}
				return
			}
			if debugTrace {
				log.Printf("recv %s", string(data))
			}
			var m inMsg
			if err := json.Unmarshal(data, &m); err != nil {
				continue
			}
			if m.T == "in" {
				m.DX = Clamp(m.DX, -600, 600)
				m.DY = Clamp(m.DY, -600, 600)
			}
			cs.push(m)
		}
	}()

	var writeMu sync.Mutex
	send := func(mt int, data []byte) error {
		writeMu.Lock()
		defer writeMu.Unlock()
		_ = conn.SetWriteDeadline(time.Now().Add(4 * time.Second))
		return conn.WriteMessage(mt, data)
	}

	// Handshake.
	_ = send(websocket.TextMessage, mustJSON(helloMsg{T: "hello", W: g.render.W, H: g.render.H, Fov: 72, Mag: g.player.mag}))

	simTick := time.NewTicker(8 * time.Millisecond)
	defer simTick.Stop()
	hudTick := time.NewTicker(80 * time.Millisecond)
	defer hudTick.Stop()
	pingTick := time.NewTicker(25 * time.Second)
	defer pingTick.Stop()

	last := time.Now()
	framePending := false
	// Adaptive resolution: keep the frame time near the target by trading
	// pixels, between a floor and whatever the client asked for.
	wantW, wantH := cw, ch
	renderW, renderH := cw, ch
	autoScale := 1.0
	lastAdapt := time.Now()
	locked := false

	for {
		select {
		case <-done:
			return
		case <-simTick.C:
		case <-hudTick.C:
			_ = send(websocket.TextMessage, mustJSON(hudMsg{
				T: "hud", Gun: g.player.gun.String(), Ammo: g.player.ammo, Mag: g.player.mag,
				Spread: g.player.spread, FPS: g.fpsEMA, Ms: g.msEMA,
				Shots: g.player.Shots, Hits: g.player.Hits,
				Shattered: g.world.glassBroken, Locked: locked, Hint: g.player.Hint(g.world),
			}))
			continue
		case <-pingTick.C:
			_ = send(websocket.PingMessage, nil)
			continue
		case <-cs.acks:
			framePending = false
		}

		now := time.Now()
		dt := now.Sub(last).Seconds()
		last = now

		var lockedNow bool
		in, resize, wh, lockedNow := cs.Take()
		locked = lockedNow
		if resize {
			wantW, wantH = clampResolution(wh[0], wh[1])
			renderW, renderH = pickResolution(wantW, wantH, autoScale)
			if renderW != g.render.W || renderH != g.render.H {
				g.render.Resize(renderW, renderH)
				g.player.viewH = renderH
				_ = send(websocket.TextMessage, mustJSON(helloMsg{T: "hello", W: renderW, H: renderH, Fov: 72, Mag: g.player.mag}))
			}
		}
		for _, e := range in.Events {
			if e == "reset" {
				g.Reset()
			}
			if e == "stop" {
				g.player.holsterGun()
			}
		}
		if debugTrace && (len(in.Events) > 0 || in.DX != 0 || in.DY != 0) {
			log.Printf("input ev=%v dx=%.1f dy=%.1f | gun=%v ammo=%d shots=%d",
				in.Events, in.DX, in.DY, g.player.gun, g.player.ammo, g.player.Shots)
		}
		g.Advance(dt, in)
		if len(g.events) > 0 {
			if debugTrace {
				log.Printf("  -> events %v", g.events)
			}
			_ = send(websocket.TextMessage, mustJSON(evsMsg{T: "evs", E: g.events}))
			g.events = g.events[:0]
		}

		if framePending {
			continue
		}
		t0 := time.Now()
		g.Draw()
		if err := send(websocket.BinaryMessage, g.FrameMessage()); err != nil {
			return
		}
		ms := float64(time.Since(t0).Microseconds()) / 1000
		g.statsMs += ms
		g.statsN++
		if time.Since(g.lastStats) > 500*time.Millisecond {
			el := time.Since(g.lastStats).Seconds()
			g.fpsEMA = float64(g.statsN) / el
			g.msEMA = g.statsMs / float64(g.statsN)
			g.statsN, g.statsMs = 0, 0
			g.lastStats = time.Now()

			// Retune the render resolution towards ~20 ms per frame.
			if time.Since(lastAdapt) > 1500*time.Millisecond && g.msEMA > 3 {
				lastAdapt = time.Now()
				switch {
				case g.msEMA > 26:
					autoScale *= math.Sqrt(20.0 / g.msEMA)
				case g.msEMA < 11:
					autoScale *= 1.08
				}
				autoScale = Clamp(autoScale, 0.12, 1.0)
				nw, nh := pickResolution(wantW, wantH, autoScale)
				if nw != g.render.W || nh != g.render.H {
					g.render.Resize(nw, nh)
					g.player.viewH = nh
					_ = send(websocket.TextMessage, mustJSON(helloMsg{T: "hello", W: nw, H: nh, Fov: 72, Mag: g.player.mag}))
				}
			}
		}
		framePending = true
	}
}

// pickResolution scales the client's requested size by the adaptive factor and
// clamps it to sane bounds while preserving the aspect ratio.
func pickResolution(wantW, wantH int, scale float64) (int, int) {
	if wantW <= 0 || wantH <= 0 {
		wantW, wantH = *flagW, *flagH
	}
	w := int(float64(wantW)*scale) &^ 1
	h := int(float64(wantH)*scale) &^ 1
	// Floor: below this a software-rendered frame looks too soft to be useful.
	if w < 640 {
		w = 640
	}
	if h < 360 {
		h = 360
	}
	if w > wantW {
		w = wantW &^ 1
	}
	if h > wantH {
		h = wantH &^ 1
	}
	return w, h
}

func mustJSON(v any) []byte {
	b, err := json.Marshal(v)
	if err != nil {
		return []byte("{}")
	}
	return b
}

// ---------------------------------------------------------------------------
// Offline verification renders (-shots)
// ---------------------------------------------------------------------------

type shotPose struct {
	name        string
	pos         Vec3
	yaw, pitch  float64
	gun         bool
	fire        bool
	simSeconds  float64
	shatterWith *Vec3
	tracerAge   float64 // if >0, keep the tracer alive for this long
}

func renderShots(dir string) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	aim := func(from Vec3, to Vec3) (float64, float64) {
		d := to.Sub(from)
		yaw := math.Atan2(d.X, -d.Z)
		pitch := math.Atan2(d.Y, math.Hypot(d.X, d.Z))
		return yaw, pitch
	}
	glassPos := V3(-0.26, tableTopY+0.02, -1.02)
	eyeHigh := V3(0.25, 1.62, 1.75)
	y1, p1 := aim(eyeHigh, glassPos)
	eyeClose := V3(-0.10, 1.28, 0.35)
	y2, p2 := aim(eyeClose, glassPos.Add(V3(0, 0.05, 0)))
	eyeLow := V3(-0.05, 1.15, -0.05)
	y3, p3 := aim(eyeLow, glassPos)

	poses := []shotPose{
		{name: "01-entry", pos: V3(0.25, 0, 1.75), yaw: y1, pitch: p1, gun: true},
		{name: "02-closeup", pos: V3(-0.10, 0, 0.35), yaw: y2, pitch: p2, gun: true},
		{name: "03-drawn", pos: V3(0.25, 0, 1.55), yaw: y1, pitch: p1, gun: true, fire: true, simSeconds: 0.012, tracerAge: 0.012},
		{name: "04-shattered", pos: V3(0.10, 0, 1.05), yaw: y3, pitch: p3, gun: true, shatterWith: &glassPos, simSeconds: 0.20},
		{name: "05-shards", pos: V3(-0.30, 0, 0.85), yaw: y3 + 0.05, pitch: p3 - 0.06, gun: false, shatterWith: &glassPos, simSeconds: 1.15},
		{name: "06-window", pos: V3(0.0, 0, 0.4), yaw: 1.45, pitch: -0.05, gun: false},
		{name: "07-backwall", pos: V3(0.0, 0, -0.4), yaw: 3.14159, pitch: -0.02, gun: false},
		{name: "08-floor", pos: V3(0.4, 0, 1.2), yaw: 0.35, pitch: -0.55, gun: false},
	}
	w, h := 1280, 720
	g := NewGame(w, h, *flagBloom)
	for _, pose := range poses {
		// Fresh room for every pose.
		g = NewGame(w, h, *flagBloom)
		g.player.pos = pose.pos
		g.player.Yaw = pose.yaw
		g.player.Pitch = pose.pitch
		if pose.gun {
			g.player.gun = GunReady
			g.player.drawT = 1
			g.player.gunT = 1
		}
		if pose.fire {
			g.player.triggerDown(g.world)
		}
		if pose.shatterWith != nil {
			d := pose.shatterWith.Sub(g.player.Eye()).Normalize()
			g.world.Shatter(*pose.shatterWith, d)
		}
		n := int(pose.simSeconds / simStep)
		for i := 0; i < n; i++ {
			g.Step(simStep, Input{})
		}
		g.Draw()
		if err := writePNG(dir+"/"+pose.name+".png", g); err != nil {
			return err
		}
		log.Printf("wrote %s/%s.png  (%d tris, %d shaded px)", dir, pose.name, g.render.Stats.Tris, g.render.Stats.Shaded)
	}
	return nil
}

func writePNG(path string, g *Game) error {
	r := g.render
	img := image.NewRGBA(image.Rect(0, 0, r.W, r.H))
	for i, c := range r.Color {
		img.Pix[i*4+0] = byte(c)
		img.Pix[i*4+1] = byte(c >> 8)
		img.Pix[i*4+2] = byte(c >> 16)
		img.Pix[i*4+3] = 0xFF
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		return err
	}
	return os.WriteFile(path, buf.Bytes(), 0o644)
}

var _ = binary.LittleEndian
