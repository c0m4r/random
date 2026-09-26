package main

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/gorilla/websocket"
)

// cdp is a minimal Chrome DevTools Protocol client (no external deps).
type cdp struct {
	conn *websocket.Conn
	id   int
}

type cdpMsg struct {
	ID     int             `json:"id"`
	Method string          `json:"method"`
	Params json.RawMessage `json:"params"`
	Result json.RawMessage `json:"result"`
	Error  *struct {
		Message string `json:"message"`
	} `json:"error"`
}

func dialCDP(url string) (*cdp, error) {
	conn, _, err := websocket.DefaultDialer.Dial(url, nil)
	if err != nil {
		return nil, err
	}
	conn.SetReadDeadline(time.Now().Add(60 * time.Second))
	return &cdp{conn: conn}, nil
}

func (c *cdp) call(method string, params any, out any) error {
	c.id++
	req := map[string]any{"id": c.id, "method": method}
	if params != nil {
		req["params"] = params
	}
	if err := c.conn.WriteJSON(req); err != nil {
		return err
	}
	for {
		_, data, err := c.conn.ReadMessage()
		if err != nil {
			return err
		}
		var m cdpMsg
		if json.Unmarshal(data, &m) != nil {
			continue
		}
		if m.ID != c.id {
			continue // event notification
		}
		if m.Error != nil {
			return fmt.Errorf("%s: %s", method, m.Error.Message)
		}
		if out != nil && len(m.Result) > 0 {
			return json.Unmarshal(m.Result, out)
		}
		return nil
	}
}

// eval evaluates a JS expression and decodes its JSON result.
func (c *cdp) eval(expr string, out any) error {
	var res struct {
		Result struct {
			Type  string          `json:"type"`
			Value json.RawMessage `json:"value"`
		} `json:"result"`
		ExceptionDetails *struct {
		Text string `json:"text"`
		} `json:"exceptionDetails"`
	}
	if err := c.call("Runtime.evaluate", map[string]any{
		"expression": expr, "returnByValue": true, "awaitPromise": true,
	}, &res); err != nil {
		return err
	}
	if res.ExceptionDetails != nil {
		return fmt.Errorf("js: %s", res.ExceptionDetails.Text)
	}
	if out == nil || len(res.Result.Value) == 0 {
		return nil
	}
	// The page returns JSON.stringify(...) strings; unwrap them.
	if res.Result.Type == "string" {
		var inner string
		if err := json.Unmarshal(res.Result.Value, &inner); err != nil {
			return err
		}
		return json.Unmarshal([]byte(inner), out)
	}
	return json.Unmarshal(res.Result.Value, out)
}

func (c *cdp) click(x, y float64, button string) error {
	for _, typ := range []string{"mousePressed", "mouseReleased"} {
		if err := c.call("Input.dispatchMouseEvent", map[string]any{
			"type": typ, "x": x, "y": y, "button": button, "clickCount": 1,
			"buttons": map[string]int{"left": 1, "right": 2}[button],
		}, nil); err != nil {
			return err
		}
		time.Sleep(30 * time.Millisecond)
	}
	return nil
}

func (c *cdp) key(code, key string, down bool) error {
	typ := "keyUp"
	if down {
		typ = "keyDown"
	}
	return c.call("Input.dispatchKeyEvent", map[string]any{
		"type": typ, "code": code, "key": key, "windowsVirtualKeyCode": int(key[0]),
	}, nil)
}

// TestBrowserClient drives the real web client in headless Chromium: it checks
// that frames are painted, the HUD updates, the pistol can be drawn and the
// glass can be shot through the actual UI.
func TestBrowserClient(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping browser test in short mode")
	}
	chrome, err := exec.LookPath("chromium")
	if err != nil {
		if chrome, err = exec.LookPath("google-chrome"); err != nil {
			if chrome, err = exec.LookPath("chromium-browser"); err != nil {
				t.Skip("no chromium/chrome found")
			}
		}
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/ws", handleWS)
	mux.HandleFunc("/", handleStatic)
	srv := httptest.NewServer(mux)
	defer srv.Close()

	// A free port for the DevTools endpoint.
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	debugPort := l.Addr().(*net.TCPAddr).Port
	l.Close()

	profile, err := os.MkdirTemp("", "room3d-chrome")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(profile)

	cmd := exec.Command(chrome,
		"--headless=new", "--disable-gpu", "--no-sandbox", "--disable-dev-shm-usage",
		"--no-first-run", "--no-default-browser-check", "--mute-audio",
		"--hide-scrollbars", "--window-size=1280,720",
		fmt.Sprintf("--remote-debugging-port=%d", debugPort),
		"--user-data-dir="+profile,
		"about:blank",
	)
	var chromeLog bytes.Buffer
	cmd.Stderr = &chromeLog
	cmd.Stdout = &chromeLog
	if err := cmd.Start(); err != nil {
		t.Fatalf("start chromium: %v", err)
	}
	defer func() {
		_ = cmd.Process.Kill()
		_, _ = cmd.Process.Wait()
	}()

	// Wait for DevTools and find the page target.
	var wsURL string
	deadline := time.Now().Add(30 * time.Second)
	for time.Now().Before(deadline) && wsURL == "" {
		resp, err := http.Get(fmt.Sprintf("http://127.0.0.1:%d/json/list", debugPort))
		if err == nil {
			var targets []struct {
				Type                 string `json:"type"`
				WebSocketDebuggerURL string `json:"webSocketDebuggerUrl"`
			}
			_ = json.NewDecoder(resp.Body).Decode(&targets)
			resp.Body.Close()
			for _, tg := range targets {
				if tg.Type == "page" {
					wsURL = tg.WebSocketDebuggerURL
				}
			}
		}
		if wsURL == "" {
			time.Sleep(200 * time.Millisecond)
		}
	}
	if wsURL == "" {
		t.Fatalf("chromium DevTools never came up:\n%s", chromeLog.String())
	}
	c, err := dialCDP(wsURL)
	if err != nil {
		t.Fatalf("cdp dial: %v", err)
	}
	defer c.conn.Close()

	if err := c.call("Page.enable", nil, nil); err != nil {
		t.Fatal(err)
	}
	if err := c.call("Runtime.enable", nil, nil); err != nil {
		t.Fatal(err)
	}
	if err := c.call("Page.navigate", map[string]any{"url": srv.URL + "/"}, nil); err != nil {
		t.Fatal(err)
	}
	// Give the client time to connect, negotiate and paint a few frames.
	deadline = time.Now().Add(20 * time.Second)
	var metrics struct {
		Frames float64 `json:"frames"`
		Fill   float64 `json:"fill"`
		CW     int     `json:"cw"`
		CH     int     `json:"ch"`
		CSSW   float64 `json:"cssw"`
		CSSH   float64 `json:"cssh"`
		InnerW int     `json:"iw"`
		InnerH int     `json:"ih"`
		Status string  `json:"status"`
		Hint   string  `json:"hint"`
		Title  string  `json:"title"`
	}
	for time.Now().Before(deadline) {
		if err := c.eval(`(() => {
			const cv = document.getElementById('screen');
			let fill = 0;
			if (cv && cv.width) {
				const d = cv.getContext('2d').getImageData(0, 0, cv.width, cv.height).data;
				let lit = 0, total = 0;
				for (let i = 0; i < d.length; i += 4 * 97) { total++; if (d[i]+d[i+1]+d[i+2] > 30) lit++; }
				fill = total ? lit / total : 0;
			}
			return JSON.stringify({
				frames: fill > 0.5 ? 99 : 0, fill: fill,
				cw: cv ? cv.width : 0, ch: cv ? cv.height : 0,
				cssw: cv ? cv.clientWidth : 0, cssh: cv ? cv.clientHeight : 0,
				iw: innerWidth, ih: innerHeight,
				status: (document.getElementById('stats')||{}).textContent || '',
				hint: (document.getElementById('hint')||{}).textContent || '',
				title: (document.getElementById('overlay-title')||{}).textContent || ''
			});
		})()`, &metrics); err != nil {
			t.Fatalf("eval metrics: %v", err)
		}
		if metrics.Frames > 3 {
			break
		}
		time.Sleep(400 * time.Millisecond)
	}
	parseFPS := func(s string) float64 {
		var f float64
		if _, err := fmt.Sscanf(strings.TrimPrefix(s, "fps "), "%f", &f); err != nil {
			return 0
		}
		return f
	}
	t.Logf("client: canvas %dx%d (css %.0fx%.0f) in %dx%d window, painted fill=%.0f%%",
		metrics.CW, metrics.CH, metrics.CSSW, metrics.CSSH, metrics.InnerW, metrics.InnerH, metrics.Fill*100)
	t.Logf("overlay=%q stats=%q hint=%q", metrics.Title, metrics.Status, metrics.Hint)
	if metrics.CW == 0 || metrics.CH == 0 {
		t.Fatalf("client never sized the canvas")
	}
	if metrics.Fill < 0.5 {
		t.Errorf("canvas looks blank: only %.0f%% of sampled pixels are non-black", metrics.Fill*100)
	}

	if metrics.CW < 640 || metrics.CH < 360 {
		t.Errorf("canvas backing store too small: %dx%d", metrics.CW, metrics.CH)
	}
	if metrics.CW > 1280 || metrics.CH > 720 {
		t.Errorf("canvas backing store above the cap: %dx%d", metrics.CW, metrics.CH)
	}
	// The canvas must keep the render aspect ratio inside the window.
	sr := float64(metrics.CW) / float64(metrics.CH)
	cr := metrics.CSSW / metrics.CSSH
	if d := sr - cr; d > 0.02 || d < -0.02 {
		t.Errorf("canvas is distorted: render aspect %.3f vs css aspect %.3f", sr, cr)
	}

	// Click to lock the pointer and start playing.
	cx, cy := float64(metrics.InnerW)/2, float64(metrics.InnerH)/2

	type hudState struct {
		Hint         string `json:"hint"`
		Stats        string `json:"stats"`
		Pips         int    `json:"pips"`
		Locked       bool   `json:"locked"`
		OverlayShown bool   `json:"overlayShown"`
	}
	readHUD := func() hudState {
		var v hudState
		if err := c.eval(`(() => {
			const pips = document.querySelectorAll('#ammo .pip, #ammo i, #ammo span');
			let on = 0;
			pips.forEach(p => { if (!/off/.test(p.className)) on++; });
			return JSON.stringify({
				hint: (document.getElementById('hint')||{}).textContent || '',
				stats: (document.getElementById('stats')||{}).textContent || '',
				pips: on,
				locked: document.pointerLockElement === document.getElementById('screen'),
				overlayShown: getComputedStyle(document.getElementById('overlay')).display !== 'none'
			});
		})()`, &v); err != nil {
			t.Logf("readHUD eval: %v", err)
		}
		return v
	}
	if err := c.click(cx, cy, "left"); err != nil {
		t.Fatal(err)
	}
	time.Sleep(900 * time.Millisecond)
	h := readHUD()
	t.Logf("after click: locked=%v overlayShown=%v hint=%q stats=%q", h.Locked, h.OverlayShown, h.Hint, h.Stats)
	if !h.Locked {
		t.Errorf("pointer lock was not acquired by clicking the page")
	}
	if h.OverlayShown {
		t.Errorf("control overlay is still on screen while locked")
	}
	// The server's fps counter needs a moment to produce its first sample.
	fps := 0.0
	fpsDeadline := time.Now().Add(6 * time.Second)
	for time.Now().Before(fpsDeadline) {
		if fps = parseFPS(readHUD().Stats); fps > 5 {
			break
		}
		time.Sleep(300 * time.Millisecond)
	}
	t.Logf("client renders at %.1f fps", fps)
	if fps < 20 {
		t.Errorf("client is rendering too slowly: %.1f fps", fps)
	}

	// The lock should have drawn the pistol; Q holsters and redraws it.
	_ = c.key("KeyQ", "q", true)
	time.Sleep(120 * time.Millisecond)
	_ = c.key("KeyQ", "q", false)
	time.Sleep(400 * time.Millisecond)
	h = readHUD()
	t.Logf("after Q: hint=%q", h.Hint)
	_ = c.key("KeyQ", "q", true)
	time.Sleep(120 * time.Millisecond)
	_ = c.key("KeyQ", "q", false)
	time.Sleep(800 * time.Millisecond)
	h = readHUD()
	t.Logf("after Q again: hint=%q ammo pips=%d (loaded)", h.Hint, h.Pips)
	if h.Pips < 6 {
		t.Errorf("ammo pips look wrong after drawing: %d lit", h.Pips)
	}

	// Aim at the glass using real mouse movement, then fire.
	// From the spawn the glass is ~11 degrees left and ~16 degrees down.
	if err := c.call("Input.dispatchMouseEvent", map[string]any{
		"type": "mouseMoved", "x": cx, "y": cy, "button": "none", "buttons": 0,
	}, nil); err != nil {
		t.Fatal(err)
	}
	// movementX/Y are the deltas between successive events, so the pointer has
	// to keep moving for the client to accumulate anything.
	for i := 1; i <= 24; i++ {
		_ = c.call("Input.dispatchMouseEvent", map[string]any{
			"type": "mouseMoved", "x": cx - 3.5*float64(i), "y": cy + 3.6*float64(i),
			"button": "none", "buttons": 0,
		}, nil)
		time.Sleep(10 * time.Millisecond)
	}
	time.Sleep(250 * time.Millisecond)
	for i := 0; i < 2; i++ {
		if err := c.click(cx, cy, "left"); err != nil {
			t.Fatal(err)
		}
		time.Sleep(500 * time.Millisecond)
	}
	_ = c.click(cx, cy, "left")
	time.Sleep(800 * time.Millisecond)
	h = readHUD()
	t.Logf("after shooting: hint=%q stats=%q pips=%d", h.Hint, h.Stats, h.Pips)
	if !strings.Contains(h.Stats, "shots 3") && !strings.Contains(h.Stats, "hits 1") && !strings.Contains(h.Stats, "hits 2") {
		t.Errorf("shot counters did not advance: %q", h.Stats)
	}
	if !strings.Contains(strings.ToLower(h.Hint), "glass") {
		t.Errorf("HUD hint never mentions the glass: %q", h.Hint)
	}

	// Screenshot for visual inspection.
	var shot struct {
		Data string `json:"data"`
	}
	if err := c.call("Page.captureScreenshot", map[string]any{"format": "png"}, &shot); err != nil {
		t.Fatalf("capture: %v", err)
	}
	raw, err := decodeBase64(shot.Data)
	if err != nil {
		t.Fatalf("decode screenshot: %v", err)
	}
	out := filepath.Join(".", "browser-play.png")
	if err := os.WriteFile(out, raw, 0o644); err != nil {
		t.Fatal(err)
	}
	if cfg, _, err := image.DecodeConfig(bytes.NewReader(raw)); err == nil {
		t.Logf("wrote %s (%dx%d, %d bytes)", out, cfg.Width, cfg.Height, len(raw))
	}
}

func decodeBase64(s string) ([]byte, error) {
	return base64.StdEncoding.DecodeString(s)
}
