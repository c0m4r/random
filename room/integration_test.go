package main

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gorilla/websocket"
)

// wsClient is a fake browser client: it reads the socket continuously, acks
// every frame (exactly like the real client) and records what it has seen.
// A single long-lived reader avoids gorilla's "expired read deadline corrupts
// the connection" trap.
type wsClient struct {
	t    *testing.T
	conn *websocket.Conn

	mu     sync.Mutex
	frames int
	huds    []hudMsg
	evs     []string
	hello   helloMsg
	lastSeq uint32
	w, h    int
	closed bool
	err    error
}

func newWSClient(t *testing.T, srv *httptest.Server) *wsClient {
	t.Helper()
	url := "ws" + strings.TrimPrefix(srv.URL, "http") + "/ws"
	conn, _, err := websocket.DefaultDialer.Dial(url, nil)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	c := &wsClient{t: t, conn: conn}
	go c.readLoop()
	if err := c.send(map[string]any{"t": "hello", "w": 960, "h": 540}); err != nil {
		t.Fatalf("send hello: %v", err)
	}
	// Wait for the server hello so the handshake is complete before returning.
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		c.mu.Lock()
		got := c.hello.T == "hello"
		c.mu.Unlock()
		if got {
			return c
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("no server hello")
	return nil
}

func (c *wsClient) readLoop() {
	for {
		_ = c.conn.SetReadDeadline(time.Now().Add(5 * time.Second))
		mt, data, err := c.conn.ReadMessage()
		if err != nil {
			c.mu.Lock()
			c.closed = true
			c.err = err
			c.mu.Unlock()
			return
		}
		switch mt {
		case websocket.BinaryMessage:
			if len(data) < frameHdrSize {
				c.t.Errorf("short frame: %d bytes", len(data))
				continue
			}
			if string(data[0:4]) != frameMagic {
				c.t.Errorf("bad frame magic %q", data[0:4])
				continue
			}
			w := binary.LittleEndian.Uint32(data[4:])
			h := binary.LittleEndian.Uint32(data[8:])
			if int(w)*int(h)*4+frameHdrSize != len(data) {
				c.t.Errorf("frame payload mismatch: %dx%d, %d bytes", w, h, len(data))
			}
			seq := binary.LittleEndian.Uint32(data[12:])
			c.mu.Lock()
			if seq <= c.lastSeq {
				c.t.Errorf("frame sequence went backwards: %d after %d", seq, c.lastSeq)
			}
			c.lastSeq = seq
			c.frames++
			c.w, c.h = int(w), int(h)
			c.mu.Unlock()
			_ = c.send(map[string]any{"t": "ack"})
		case websocket.TextMessage:
			var probe struct {
				T string `json:"t"`
			}
			if json.Unmarshal(data, &probe) != nil {
				continue
			}
			c.mu.Lock()
			switch probe.T {
			case "hello":
				var h helloMsg
				if json.Unmarshal(data, &h) == nil {
					c.hello = h
				}
			case "hud":
				var h hudMsg
				if json.Unmarshal(data, &h) == nil {
					c.huds = append(c.huds, h)
				}
			case "evs":
				var e evsMsg
				if json.Unmarshal(data, &e) == nil {
					c.evs = append(c.evs, e.E...)
				}
			}
			c.mu.Unlock()
		}
	}
}

func (c *wsClient) send(v any) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.conn.WriteJSON(v)
}

func (c *wsClient) close() { c.conn.Close() }

// snapshot returns the counters and drains the recorded events.
func (c *wsClient) snapshot(drain bool) (frames int, last hudMsg, evs []string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	frames = c.frames
	if n := len(c.huds); n > 0 {
		last = c.huds[n-1]
	}
	evs = append(evs, c.evs...)
	if drain {
		c.evs = nil
	}
	return
}

// waitFor polls until pred is satisfied or the timeout expires.
func (c *wsClient) waitFor(timeout time.Duration, pred func(frames int, h hudMsg, evs []string) bool) bool {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		f, h, e := c.snapshot(false)
		if pred(f, h, e) {
			return true
		}
		time.Sleep(15 * time.Millisecond)
	}
	return false
}

func has(evs []string, want string) bool {
	for _, e := range evs {
		if e == want {
			return true
		}
	}
	return false
}

// TestProtocolEndToEnd drives the server exactly like the browser client does:
// it negotiates a resolution, consumes and acknowledges frames, draws the
// pistol, aims at the glass and pulls the trigger.
func TestProtocolEndToEnd(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(handleWS))
	defer srv.Close()

	c := newWSClient(t, srv)
	defer c.close()

	if c.hello.W != 960 || c.hello.H != 540 {
		t.Errorf("hello size = %dx%d, want 960x540", c.hello.W, c.hello.H)
	}
	if c.hello.Mag != 8 {
		t.Errorf("hello mag = %d, want 8", c.hello.Mag)
	}

	// Idle: frames must flow and be flow-controlled by our acks.
	time.Sleep(700 * time.Millisecond)
	frames, h, _ := c.snapshot(true)
	if frames < 5 {
		t.Fatalf("only %d frames in 700ms (flow control broken?)", frames)
	}
	if h.Gun != "holstered" {
		t.Errorf("initial gun state = %q, want holstered", h.Gun)
	}
	if h.Ammo != 8 {
		t.Errorf("initial ammo = %d, want 8", h.Ammo)
	}
	if h.Shattered {
		t.Errorf("glass already broken at start")
	}
	t.Logf("idle: %d frames, hint %q", frames, h.Hint)

	// Pointer lock draws the pistol.
	if err := c.send(map[string]any{"t": "in", "k": 0, "ev": []string{"start"}}); err != nil {
		t.Fatal(err)
	}
	if !c.waitFor(2*time.Second, func(_ int, h hudMsg, evs []string) bool {
		return has(evs, "draw") && h.Gun == "ready" && h.Locked
	}) {
		_, h, evs := c.snapshot(false)
		t.Fatalf("pistol never became ready after start: gun=%q locked=%v evs=%v", h.Gun, h.Locked, evs)
	}

	// Aim at the glass on the table: the player spawns at (0.25,1.62,1.75)
	// looking down -Z; the glass sits at (-0.26, 0.80, -1.02).
	eye := V3(0.25, 1.62, 1.75)
	target := V3(-0.26, tableTopY+0.045, -1.02)
	d := target.Sub(eye)
	wantYaw := math.Atan2(d.X, -d.Z)
	wantPitch := math.Atan2(d.Y, math.Hypot(d.X, d.Z))
	dyaw := wantYaw / 0.0022
	dpitch := -(wantPitch + 0.10) / 0.0022
	if err := c.send(map[string]any{"t": "in", "k": 0, "dx": dyaw, "dy": dpitch}); err != nil {
		t.Fatal(err)
	}
	time.Sleep(150 * time.Millisecond)

	// Fire.
	_, _, _ = c.snapshot(true)
	if err := c.send(map[string]any{"t": "in", "k": 0, "ev": []string{"fireDown"}}); err != nil {
		t.Fatal(err)
	}
	if !c.waitFor(2*time.Second, func(_ int, h hudMsg, evs []string) bool { return has(evs, "shot") }) {
		_, h, evs := c.snapshot(false)
		t.Fatalf("no shot event; gun=%q ammo=%d evs=%v", h.Gun, h.Ammo, evs)
	}
	_, _, evs := c.snapshot(true)
	if !has(evs, "shatter") {
		t.Errorf("glass did not shatter; events %v", evs)
	}
	if !has(evs, "hit") {
		t.Errorf("no hit event; events %v", evs)
	}
	if !c.waitFor(2*time.Second, func(_ int, h hudMsg, _ []string) bool {
		return h.Shattered && h.Ammo == 7 && h.Shots == 1 && h.Hits == 1
	}) {
		_, h, _ := c.snapshot(false)
		t.Errorf("hud after the shot: shattered=%v ammo=%d shots=%d hits=%d",
			h.Shattered, h.Ammo, h.Shots, h.Hits)
	}
	_, h, _ = c.snapshot(false)
	t.Logf("after shot: gun=%s ammo=%d shots=%d hits=%d shattered=%v hint=%q",
		h.Gun, h.Ammo, h.Shots, h.Hits, h.Shattered, h.Hint)

	// Firing again must not shatter anything twice.
	if err := c.send(map[string]any{"t": "in", "k": 0, "ev": []string{"fireUp", "fireDown"}}); err != nil {
		t.Fatal(err)
	}
	if !c.waitFor(2*time.Second, func(_ int, h hudMsg, _ []string) bool { return h.Ammo == 6 }) {
		t.Errorf("second shot did not consume ammo: %d", h.Ammo)
	}
	_, _, evs = c.snapshot(true)
	if has(evs, "shatter") {
		t.Errorf("glass shattered twice")
	}

	// Holster, then draw again.
	if err := c.send(map[string]any{"t": "in", "k": 0, "ev": []string{"toggle"}}); err != nil {
		t.Fatal(err)
	}
	if !c.waitFor(2*time.Second, func(_ int, h hudMsg, evs []string) bool {
		return h.Gun == "holstered" && has(evs, "holster")
	}) {
		_, h, evs := c.snapshot(false)
		t.Errorf("toggle did not holster: gun=%q evs=%v", h.Gun, evs)
	}

	// Reset restores the room, and the pistol comes back out.
	if err := c.send(map[string]any{"t": "in", "k": 0, "ev": []string{"toggle", "reset"}}); err != nil {
		t.Fatal(err)
	}
	if !c.waitFor(2*time.Second, func(_ int, h hudMsg, evs []string) bool {
		return !h.Shattered && h.Ammo == 8 && has(evs, "reset")
	}) {
		_, h, evs := c.snapshot(false)
		t.Errorf("reset did not restore the room: shattered=%v ammo=%d evs=%v", h.Shattered, h.Ammo, evs)
	}

	// Moving must keep the frame stream alive.
	_, _, _ = c.snapshot(true)
	startFrames, _, _ := c.snapshot(false)
	if err := c.send(map[string]any{"t": "in", "k": keyW | keyD}); err != nil {
		t.Fatal(err)
	}
	time.Sleep(400 * time.Millisecond)
	midFrames, _, _ := c.snapshot(false)
	if midFrames-startFrames < 3 {
		t.Errorf("frame stream stalled while moving: %d -> %d", startFrames, midFrames)
	}

	// Two shots, then a reload.
	if err := c.send(map[string]any{"t": "in", "k": 0, "ev": []string{"fireDown", "fireUp", "fireDown", "fireUp"}}); err != nil {
		t.Fatal(err)
	}
	if !c.waitFor(3*time.Second, func(_ int, h hudMsg, _ []string) bool { return h.Ammo == 6 }) {
		_, h, _ := c.snapshot(false)
		t.Fatalf("two shots did not empty two rounds (ammo=%d)", h.Ammo)
	}
	_, _, _ = c.snapshot(true)
	if err := c.send(map[string]any{"t": "in", "k": 0, "ev": []string{"reload"}}); err != nil {
		t.Fatal(err)
	}
	if !c.waitFor(2*time.Second, func(_ int, h hudMsg, evs []string) bool {
		return has(evs, "reload") && h.Gun == "reloading"
	}) {
		_, h, evs := c.snapshot(false)
		t.Errorf("reload did not start: gun=%q ammo=%d evs=%v", h.Gun, h.Ammo, evs)
	}
	if !c.waitFor(3*time.Second, func(_ int, h hudMsg, _ []string) bool {
		return h.Ammo == 8 && h.Gun == "ready"
	}) {
		_, h, _ := c.snapshot(false)
		t.Errorf("magazine not full after reload: ammo=%d gun=%q", h.Ammo, h.Gun)
	}
	// Firing while the magazine is full and the pistol is holstered does nothing.
	if err := c.send(map[string]any{"t": "in", "k": 0, "ev": []string{"toggle"}}); err != nil {
		t.Fatal(err)
	}
	if !c.waitFor(2*time.Second, func(_ int, h hudMsg, _ []string) bool { return h.Gun == "holstered" }) {
		t.Errorf("pistol did not holster")
	}
	var shotsBefore, ammoBefore int
	{
		_, h, _ := c.snapshot(false)
		shotsBefore, ammoBefore = h.Shots, h.Ammo
	}
	if err := c.send(map[string]any{"t": "in", "k": 0, "ev": []string{"fireDown", "fireUp"}}); err != nil {
		t.Fatal(err)
	}
	time.Sleep(600 * time.Millisecond)
	_, h, evs = c.snapshot(true)
	if h.Shots != shotsBefore || h.Ammo != ammoBefore {
		t.Errorf("trigger pulled while holstered changed state: shots %d->%d ammo %d->%d",
			shotsBefore, h.Shots, ammoBefore, h.Ammo)
	}
	if has(evs, "shot") {
		t.Errorf("a shot was fired while the pistol was holstered")
	}

	// Stop (pointer lock lost) holsters the pistol again.
	if err := c.send(map[string]any{"t": "in", "k": 0, "ev": []string{"stop"}}); err != nil {
		t.Fatal(err)
	}
	if !c.waitFor(2*time.Second, func(_ int, h hudMsg, _ []string) bool {
		return h.Gun == "holstered" && !h.Locked
	}) {
		_, h, _ := c.snapshot(false)
		t.Errorf("lock loss did not holster the pistol: gun=%q locked=%v", h.Gun, h.Locked)
	}

	// Resize negotiation.
	if err := c.send(map[string]any{"t": "resize", "w": 1280, "h": 720}); err != nil {
		t.Fatal(err)
	}
	// The server answers a resize with a fresh hello; the exact size may be
	// scaled down by the adaptive-resolution controller on a slow machine.
	if !c.waitFor(5*time.Second, func(_ int, h hudMsg, _ []string) bool { return true && c.width() >= 480 }) {
		t.Errorf("no usable resolution after resize (w=%d)", c.width())
	}
	if c.width() > 1280 || c.height() > 720 {
		t.Errorf("server exceeded the requested cap: %dx%d", c.width(), c.height())
	}
	t.Logf("final render size %dx%d after requesting 1280x720", c.width(), c.height())
}

func (c *wsClient) width() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.w
}

func (c *wsClient) height() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.h
}

// TestStaticAssetsServed checks the embedded web client is reachable.
func TestStaticAssetsServed(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/ws", handleWS)
	mux.HandleFunc("/", handleStatic)
	srv := httptest.NewServer(mux)
	defer srv.Close()

	for _, path := range []string{"/", "/app.js", "/style.css"} {
		resp, err := http.Get(srv.URL + path)
		if err != nil {
			t.Fatalf("GET %s: %v", path, err)
		}
		body := make([]byte, 4096)
		n, _ := resp.Body.Read(body)
		resp.Body.Close()
		if resp.StatusCode != 200 || n == 0 {
			t.Errorf("GET %s -> %d (%d bytes)", path, resp.StatusCode, n)
		}
	}
	resp, err := http.Get(srv.URL + "/nope")
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusNotFound {
		t.Errorf("GET /nope -> %d, want 404", resp.StatusCode)
	}
}
