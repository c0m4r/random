# Wire protocol — Room3D (Go software-rendered FPS) ↔ browser client

The Go server does **all** 3D: it rasterizes a 24-bit framebuffer and streams raw RGBA
frames over a WebSocket. The browser is a dumb display + input device (plus WebAudio SFX).

Endpoint: `ws://<host>/ws` (page is served from `/`).

## Server → Client

### 1. Binary frames

One binary WebSocket message per frame:

```
offset  size  content
0       4     magic ASCII "RGB1"
4       4     uint32 LE width   (pixels)
8       4     uint32 LE height  (pixels)
12      4     uint32 LE frame sequence number (monotonic, starts at 1)
16      w*h*4 raw RGBA8 pixels, row-major, top-left origin, NOT premultiplied
```

Client paints it with:

```js
const px = new Uint8ClampedArray(buf, 16, w * h * 4);
ctx.putImageData(new ImageData(px, w, h), 0, 0);
```

A new frame is only sent after the client has `ack`ed the previous one (see below);
the server never has more than one frame in flight.

### 2. Text (JSON) messages

Sent in this order on connect: `hello`, then `hud` at ~12 Hz, plus `evs` batches.

```json
{"t":"hello","w":960,"h":540,"fov":72,"mag":8}
```
`speed` is not sent; movement is server-side. Sent once right after the client's `hello`.

```json
{"t":"hud","gun":"holstered","ammo":8,"mag":8,"spread":2.6,"fps":58,"ms":9.1,
 "shots":0,"hits":0,"shattered":false,"locked":true,"hint":"..."}
```
* `gun` ∈ `holstered | drawing | ready | firing | reloading | holstering`
* `ammo` rounds left in magazine, `mag` magazine size
* `spread` current crosshair spread in **pixels at render resolution** (crosshair gap)
* `fps`, `ms` render stats for the on-screen counter
* `shots`, `hits` counters, `shattered` true once the glass is broken
* `locked` pointer-lock state as tracked from the client's `start`/`stop` events
* `hint` short human-readable status line the client may show (already localized to English)

```json
{"t":"evs","e":["shot","shatter","impact","reload","draw","holster","empty","splash","ricochet","reset","hit"]}
```
Sound/effect triggers. Zero or more per frame. Known values:
`shot, shatter, impact, ricochet, splash, reload, draw, holster, empty, hit, reset`.

## Client → Server (text JSON, at most one message per animation frame)

```json
{"t":"hello","w":1280,"h":720}
```
Ask for a render resolution (server clamps to ≤1280×720, keeps the aspect and even
numbers). The server replies with its own `hello` carrying the size it will actually
render, which may be *smaller* than requested: the renderer auto-tunes its resolution
to hold roughly 20 ms per frame, and announces every change with another `hello`.
Clients must therefore treat a server `hello` (and the frame header) as authoritative
at any time, not just during the handshake.

```json
{"t":"in","k":0,"dx":0.0,"dy":0.0,"ev":["fireDown"]}
```
* `k` — bitmask of **held** keys:
  `1`=W `2`=A `4`=S `8`=D `16`=Shift `32`=Space `64`=Q `128`=R `256`=E `512`=arrowUp `1024`=arrowLeft `2048`=arrowDown `4096`=arrowRight
* `dx`,`dy` — accumulated mouse movement since the previous `in` message (CSS pixels, raw
  `movementX/movementY` sums). The accumulator is reset to 0 by the client after sending.
* `ev` — discrete one-shot actions, any of:
  `fireDown`, `fireUp`, `reload`, `toggle`, `reset`, `start`, `stop`
  (`start`/`stop` = pointer lock gained/lost).

Send this every `requestAnimationFrame` **and** immediately on any discrete event, even if
nothing changed. Never let it exceed ~120 messages/second.

```json
{"t":"ack"}
```
Sent immediately after `putImageData` for a received frame. This is the flow-control
signal: the server renders the next frame only after an `ack`.

```json
{"t":"resize","w":1024,"h":576}
```
Sent (debounced ~150 ms) when the canvas/backing size wants to change.

## Client behaviour requirements

* Fullscreen `<canvas>`; the render resolution comes from the server (letterbox with CSS if
  the aspect differs — use `object-fit: contain`-like centering, black background).
* Click on the page → `requestPointerLock()`. Lost lock → overlay "Click to play" +
  `{"t":"stop"}`. Gain → `{"t":"start"}`.
* Keys are captured only while pointer-locked; `Escape` is never intercepted (browser
  unlocks the pointer for us). Prevent `contextmenu` on the canvas.
* Right mouse button = the same action as `Q` (`toggle`: draw/holster the pistol).
* Losing the pointer lock must send `ev:["stop"]` (the server holsters the pistol); gaining
  it sends `ev:["start"]` (the server draws the pistol).
* Left mouse button: on `mousedown` send `ev:["fireDown"]`, on `mouseup` send `ev:["fireUp"]`.
* Reconnect automatically (1 s backoff) if the socket closes; re-send `hello`.
* WebAudio synth for `evs` (no external assets, no CDN): gunshot (noise burst + fast decay),
  glass shatter (bright noise + several detuned partials), impact/ricochet (short click/ping),
  splash (filtered noise blip), reload (two mechanical clicks), draw/holster (short swoosh),
  empty (dry click), hit (subtle tick). Resume the AudioContext on the first user gesture.
* HUD (DOM, not canvas): crosshair with a gap driven by `spread` (4 ticks + centre dot),
  ammo pips from `ammo`/`mag`, small stats line (`fps`, `ms`, `shots`, `hits`), a status
  line from `hint`, and a hitmarker flash on `hit`. Crosshair must be exactly centred.
* No frameworks, no build step, no external network requests: plain `index.html`,
  `style.css`, `app.js`.
