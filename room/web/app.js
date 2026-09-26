/* =====================================================================
 * Room3D — browser client for the Go software renderer.
 *
 * The server owns all 3D work and streams raw RGBA frames over /ws; this
 * file is a well-behaved display + input device (plus a WebAudio synth).
 * Message names, field names, key bitmask values and event names below are
 * protocol constants taken verbatim from PROTOCOL.md.
 *
 * Plain script: no modules, no build step, no external requests.
 * ===================================================================== */
(function () {
  'use strict';

  /* ================================================================== *
   * Protocol constants
   * ================================================================== */

  var WS_URL = (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws';

  var FRAME_MAGIC = 0x31424752;   // bytes 'R','G','B','1' read as uint32 LE
  var FRAME_HEADER = 16;          // magic u32 | width u32 | height u32 | seq u32
  var MAX_DIM = 8192;             // sanity bound for server-supplied dimensions

  // {"t":"in","k":...} — bitmask of *held* keys.
  var KEY_BITS = {
    KeyW: 1, KeyA: 2, KeyS: 4, KeyD: 8,
    ShiftLeft: 16, ShiftRight: 16,
    Space: 32, KeyQ: 64, KeyR: 128, KeyE: 256,
    ArrowUp: 512, ArrowLeft: 1024, ArrowDown: 2048, ArrowRight: 4096
  };

  var MAX_RENDER_W = 1280;        // server clamps to <= 1280x720, keeps aspect
  var MAX_RENDER_H = 720;
  var RESIZE_DEBOUNCE_MS = 150;   // "debounced ~150 ms"
  var RECONNECT_MS = 1000;        // "reconnect automatically (1 s backoff)"
  var MIN_IN_INTERVAL_MS = 1000 / 120;  // never exceed ~120 "in" messages/second
  var MAX_MOUSE_DELTA = 200;      // per-event clamp (tab-switch / lock spikes)
  var MAX_MOUSE_ACC = 4000;       // per-frame accumulator clamp
  var MAX_EV_QUEUE = 32;
  var KEY_SCROLL_CODES = { Space: 1, ArrowUp: 1, ArrowLeft: 1, ArrowDown: 1, ArrowRight: 1 };
  var HIT_FLASH_MS = 220;
  var MSG_ACK = '{"t":"ack"}';

  /* ================================================================== *
   * DOM
   * ================================================================== */

  var canvas = document.getElementById('screen');
  var ctx = canvas.getContext('2d', { alpha: false });

  var hudEl = document.getElementById('hud');
  var statsEl = document.getElementById('stats');
  var crosshairEl = document.getElementById('crosshair');
  var hitmarkerEl = document.getElementById('hitmarker');
  var ammoEl = document.getElementById('ammo');
  var hintEl = document.getElementById('hint');
  var overlayEl = document.getElementById('overlay');
  var titleEl = document.getElementById('overlay-title');
  var subEl = document.getElementById('overlay-sub');

  /* ================================================================== *
   * State
   * ================================================================== */

  var state = {
    ws: null,
    open: false,          // socket open
    firstFrame: false,    // at least one frame painted on this connection
    locked: false,        // pointer lock held
    helloDone: false,     // server hello received on this connection
    everConnected: false,
    reqW: 0, reqH: 0,     // last resolution we asked for
    srvW: 0, srvH: 0,     // authoritative resolution from the server hello
    seq: 0                // last frame sequence number
  };

  var reconnectTimer = 0;
  var resizeTimer = 0;
  var hitTimer = 0;

  var cssW = 0, cssH = 0;      // canvas CSS (letterboxed) size
  var cssScale = 1;            // canvas CSS pixels per render pixel

  // Cached ImageData wrapper. ImageData keeps the Uint8ClampedArray we hand it
  // (no pixel copy), so this cache only avoids re-wrapping the *same* buffer;
  // each WebSocket message owns a fresh ArrayBuffer, hence a fresh tiny wrapper.
  var imgData = null, imgW = 0, imgH = 0, imgBuf = null;

  var nowMs = (window.performance && performance.now)
    ? function () { return performance.now(); }
    : function () { return Date.now(); };

  function isFiniteNum(v) { return typeof v === 'number' && isFinite(v); }
  function fin(v) { return isFiniteNum(v) ? v : null; }
  function clamp(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }

  /* ================================================================== *
   * Canvas sizing: backing store = server resolution, CSS = letterbox
   * ================================================================== */

  function layout() {
    var rw = canvas.width, rh = canvas.height;
    if (!(rw > 0) || !(rh > 0)) return;

    var vw = Math.max(1, window.innerWidth || 0);
    var vh = Math.max(1, window.innerHeight || 0);
    var s = Math.min(vw / rw, vh / rh);
    if (!isFinite(s) || s <= 0) s = 1;

    var w = clamp(Math.round(rw * s), 1, vw);
    var h = clamp(Math.round(rh * s), 1, vh);
    if (w !== cssW || h !== cssH) {
      canvas.style.width = w + 'px';
      canvas.style.height = h + 'px';
      cssW = w;
      cssH = h;
    }
    cssScale = w / rw;   // spread is quoted in render pixels -> CSS pixels
    applySpread();
  }

  // Ask for a resolution that matches the window's CSS pixel size, aspect kept,
  // capped at 1280x720 (server rounds to even numbers).
  function desiredResolution() {
    var vw = Math.max(1, window.innerWidth || 0);
    var vh = Math.max(1, window.innerHeight || 0);
    var s = Math.min(1, MAX_RENDER_W / vw, MAX_RENDER_H / vh);
    if (!isFinite(s) || s <= 0) s = 1;
    var w = Math.floor(vw * s);
    var h = Math.floor(vh * s);
    w -= w & 1;
    h -= h & 1;
    return { w: clamp(w, 2, MAX_RENDER_W), h: clamp(h, 2, MAX_RENDER_H) };
  }

  // Adopt the server's resolution as the canvas backing size.
  function applyServerSize(w, h) {
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
      imgData = null; imgW = 0; imgH = 0; imgBuf = null;  // size changed
      cssW = 0; cssH = 0;                                 // force CSS recompute
    }
    layout();
  }

  /* ================================================================== *
   * WebSocket
   * ================================================================== */

  function sendText(text) {
    if (!state.open || !state.ws || state.ws.readyState !== 1) return false;
    try {
      state.ws.send(text);
      return true;
    } catch (err) {
      return false;
    }
  }

  function connect() {
    if (state.ws && (state.ws.readyState === 0 || state.ws.readyState === 1)) return;
    if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = 0; }

    var ws;
    try {
      ws = new WebSocket(WS_URL);
    } catch (err) {
      scheduleReconnect();
      return;
    }
    ws.binaryType = 'arraybuffer';   // frames must arrive as ArrayBuffer
    state.ws = ws;

    ws.onopen = function () {
      if (state.ws !== ws) return;
      state.open = true;
      state.helloDone = false;
      state.firstFrame = false;
      state.everConnected = true;
      sendHello();                   // fresh hello on every (re)connect
      updateOverlay();
    };

    ws.onmessage = function (ev) {
      if (state.ws !== ws) return;
      var data = ev.data;
      if (typeof data === 'string') {
        onTextMessage(data);
      } else if (data instanceof ArrayBuffer) {
        onFrame(data);
      }
      // anything else (Blob, etc.) is ignored: binaryType is set to arraybuffer
    };

    ws.onerror = function () {
      if (state.ws !== ws) return;
      try { ws.close(); } catch (err) { /* onclose handles the rest */ }
    };

    ws.onclose = function () {
      if (state.ws !== ws) return;
      onDisconnected();
    };
  }

  function onDisconnected() {
    state.open = false;
    state.helloDone = false;
    state.firstFrame = false;
    state.ws = null;
    releaseInput();
    if (state.locked && document.exitPointerLock) document.exitPointerLock();
    updateOverlay();
    scheduleReconnect();
  }

  function scheduleReconnect() {
    if (reconnectTimer) return;
    reconnectTimer = setTimeout(function () {
      reconnectTimer = 0;
      connect();
    }, RECONNECT_MS);
  }

  function sendHello() {
    var d = desiredResolution();
    state.reqW = d.w;
    state.reqH = d.h;
    sendText('{"t":"hello","w":' + d.w + ',"h":' + d.h + '}');
  }

  function maybeSendResize() {
    if (!state.open || !state.helloDone) return;   // hello carries the first size
    var d = desiredResolution();
    if (d.w === state.reqW && d.h === state.reqH) return;
    if (d.w === state.srvW && d.h === state.srvH) return;
    state.reqW = d.w;
    state.reqH = d.h;
    sendText('{"t":"resize","w":' + d.w + ',"h":' + d.h + '}');
  }

  /* ================================================================== *
   * Server -> client messages
   * ================================================================== */

  function onTextMessage(text) {
    if (!text) return;
    var m;
    try {
      m = JSON.parse(text);
    } catch (err) {
      return;                                  // malformed JSON: ignore quietly
    }
    if (!m || typeof m !== 'object') return;
    switch (m.t) {
      case 'hello': onHello(m); break;
      case 'hud': onHud(m); break;
      case 'evs': onEvents(m.e); break;
      default: break;                          // unknown type: ignore
    }
  }

  function onHello(m) {
    var w = fin(m.w), h = fin(m.h);
    if (w === null || h === null) return;
    w = Math.floor(w); h = Math.floor(h);
    if (w < 1 || h < 1 || w > MAX_DIM || h > MAX_DIM) return;
    state.helloDone = true;
    state.srvW = w;
    state.srvH = h;
    applyServerSize(w, h);
    // The window may have changed during the handshake.
    if (resizeTimer) { clearTimeout(resizeTimer); }
    resizeTimer = setTimeout(function () { resizeTimer = 0; maybeSendResize(); }, RESIZE_DEBOUNCE_MS);
  }

  function onFrame(buf) {
    var len = buf.byteLength;
    if (len < FRAME_HEADER) return;            // zero-length / truncated header
    if (!ctx) return;

    var view = new DataView(buf);
    if (view.getUint32(0, true) !== FRAME_MAGIC) return;   // not an "RGB1" frame

    var w = view.getUint32(4, true);
    var h = view.getUint32(8, true);
    var seq = view.getUint32(12, true);
    if (w === 0 || h === 0 || w > MAX_DIM || h > MAX_DIM) return;
    if (len < FRAME_HEADER + w * h * 4) return;            // short payload

    if (canvas.width !== w || canvas.height !== h) applyServerSize(w, h);

    // Zero-copy view straight into the received buffer (PROTOCOL.md §1).
    var px = new Uint8ClampedArray(buf, FRAME_HEADER, w * h * 4);
    ctx.putImageData(imageDataFor(px, w, h, buf), 0, 0);
    sendText(MSG_ACK);                         // flow control: ack immediately

    state.seq = seq;   // informational: a frame is never dropped, every one is acked
    if (!state.firstFrame) {
      state.firstFrame = true;
      updateOverlay();
    }
  }

  function imageDataFor(px, w, h, buf) {
    if (imgData !== null && imgW === w && imgH === h && imgBuf === buf) return imgData;
    imgData = new ImageData(px, w, h);
    imgW = w;
    imgH = h;
    imgBuf = buf;
    return imgData;
  }

  function onEvents(list) {
    if (!list || typeof list.length !== 'number') return;
    for (var i = 0; i < list.length; i++) {
      var name = list[i];
      if (typeof name !== 'string') continue;
      if (name === 'hit') flashHitmarker();
      Audio.play(name);                        // unknown names are ignored
    }
  }

  /* ================================================================== *
   * HUD
   * ================================================================== */

  var hud = {
    ammo: -1, mag: -1, spread: -1,
    fps: null, ms: null, shots: null, hits: null,
    hint: null
  };
  var spreadCss = -1;
  var pipCount = -1;

  function onHud(m) {
    var mag = fin(m.mag);
    if (mag !== null) {
      mag = clamp(Math.floor(mag), 0, 64);
      if (mag !== hud.mag) { hud.mag = mag; buildPips(mag); }
    }

    var ammo = fin(m.ammo);
    if (ammo !== null) {
      ammo = clamp(Math.floor(ammo), 0, hud.mag > 0 ? hud.mag : 64);
      if (ammo !== hud.ammo) { hud.ammo = ammo; paintPips(); }
    }

    var spread = fin(m.spread);
    if (spread !== null && spread !== hud.spread) {
      hud.spread = spread;
      applySpread();
    }

    var fps = fin(m.fps), ms = fin(m.ms), shots = fin(m.shots), hits = fin(m.hits);
    if ((fps !== null && fps !== hud.fps) || (ms !== null && ms !== hud.ms) ||
        (shots !== null && shots !== hud.shots) || (hits !== null && hits !== hud.hits)) {
      // Absent / non-numeric fields keep their last known value.
      if (fps !== null) hud.fps = fps;
      if (ms !== null) hud.ms = ms;
      if (shots !== null) hud.shots = shots;
      if (hits !== null) hud.hits = hits;
      statsEl.textContent =
        'fps ' + (hud.fps === null ? '--' : hud.fps.toFixed(0)) +
        ' \u00b7 ms ' + (hud.ms === null ? '--' : hud.ms.toFixed(1)) +
        ' \u00b7 shots ' + (hud.shots === null ? 0 : hud.shots) +
        ' \u00b7 hits ' + (hud.hits === null ? 0 : hud.hits);
    }

    if (typeof m.hint === 'string' && m.hint !== hud.hint) {
      hud.hint = m.hint;
      hintEl.textContent = m.hint;
    }
  }

  function buildPips(mag) {
    if (mag === pipCount) return;
    pipCount = mag;
    ammoEl.textContent = '';
    if (mag <= 0) { paintPips(); return; }
    var frag = document.createDocumentFragment();
    for (var i = 0; i < mag; i++) {
      var pip = document.createElement('span');
      pip.className = 'pip';
      frag.appendChild(pip);
    }
    var count = document.createElement('span');
    count.className = 'count';
    frag.appendChild(count);
    ammoEl.appendChild(frag);
    paintPips();
  }

  function paintPips() {
    var pips = ammoEl.children;
    var ammo = hud.ammo < 0 ? pipCount : hud.ammo;
    for (var i = 0; i < pips.length; i++) {
      var el = pips[i];
      if (el.className !== 'pip' && el.className !== 'pip off') continue;
      var on = i < ammo;
      if (on === (el.className === 'pip')) continue;
      el.className = on ? 'pip' : 'pip off';
    }
    var count = ammoEl.lastElementChild;
    if (count && count.className === 'count') {
      count.textContent = (hud.ammo < 0 ? '\u2013' : hud.ammo) + ' / ' + (pipCount < 0 ? 0 : pipCount);
    }
  }

  // `spread` is quoted in pixels at the render resolution: convert to CSS px.
  function applySpread() {
    var gap = (hud.spread > 0 ? hud.spread : 0) * cssScale;
    gap = clamp(gap, 0, 400);
    var rounded = Math.round(gap * 100) / 100;
    if (rounded === spreadCss) return;
    spreadCss = rounded;
    crosshairEl.style.setProperty('--gap', rounded + 'px');
  }

  function flashHitmarker() {
    if (!state.locked) return;
    hitmarkerEl.classList.remove('on');
    void hitmarkerEl.offsetWidth;              // restart the CSS animation
    hitmarkerEl.classList.add('on');
    if (hitTimer) clearTimeout(hitTimer);
    hitTimer = setTimeout(function () {
      hitTimer = 0;
      hitmarkerEl.classList.remove('on');
    }, HIT_FLASH_MS);
  }

  function updateOverlay() {
    var ready = state.firstFrame;
    var showOverlay = !ready || !state.locked;
    if (showOverlay) overlayEl.classList.remove('hidden');
    else overlayEl.classList.add('hidden');
    overlayEl.classList.toggle('ready', ready);

    titleEl.textContent = ready ? 'CLICK TO PLAY' : 'CONNECTING\u2026';
    subEl.textContent = ready
      ? 'pointer lock is required for mouse look'
      : (state.everConnected ? 'link lost \u2014 reconnecting\u2026' : 'connecting to renderer\u2026');

    hudEl.classList.toggle('idle', !state.locked);
  }

  /* ================================================================== *
   * Input
   * ================================================================== */

  var keysDown = Object.create(null);
  var keyMask = 0;
  var dxAcc = 0, dyAcc = 0;
  var evQueue = [];
  var lastInAt = -1e9;
  var firing = false;

  function clearKeys() {
    keysDown = Object.create(null);
    keyMask = 0;
  }

  function round2(v) { return Math.round(v * 100) / 100; }

  // One "in" message per animation frame, plus immediately on discrete events,
  // throttled to <= ~120 messages/second. The accumulators are only reset once
  // the message actually went out. `force` bypasses the throttle for the rare
  // state-critical sends (lock change, blur) that must not wait for a rAF.
  function flushInput(force) {
    if (!state.open || !state.ws || state.ws.readyState !== 1) {
      dxAcc = 0; dyAcc = 0; evQueue.length = 0;
      return;
    }
    var now = nowMs();
    if (!force && now - lastInAt < MIN_IN_INTERVAL_MS) return;   // rAF will pick it up

    var msg = '{"t":"in","k":' + keyMask +
      ',"dx":' + round2(dxAcc) + ',"dy":' + round2(dyAcc);
    if (evQueue.length) msg += ',"ev":["' + evQueue.join('","') + '"]';
    msg += '}';

    if (!sendText(msg)) return;                        // keep state, retry later
    lastInAt = now;
    dxAcc = 0;
    dyAcc = 0;
    evQueue.length = 0;
  }

  function pushEvent(name) {
    if (!state.open) return;
    if (evQueue.length >= MAX_EV_QUEUE) evQueue.shift();
    evQueue.push(name);
  }

  function queueEvent(name) {
    pushEvent(name);
    flushInput();
  }

  /* ---- pointer lock ---- */

  var lockSupported = !!canvas.requestPointerLock;

  function requestLock() {
    if (!lockSupported || state.locked || !state.firstFrame) return;
    var p;
    try {
      p = canvas.requestPointerLock({ unadjustedMovement: true });
    } catch (err) {
      try { canvas.requestPointerLock(); } catch (err2) { /* denied */ }
      return;
    }
    if (p && typeof p.catch === 'function') {
      p.catch(function () {
        // Raw (unadjusted) movement unsupported or denied: plain lock retry.
        try { canvas.requestPointerLock(); } catch (err3) { /* denied */ }
      });
    }
  }

  function onLockChange() {
    var locked = lockSupported && document.pointerLockElement === canvas;
    if (locked === state.locked) return;
    state.locked = locked;
    document.body.classList.toggle('locked', locked);
    dxAcc = 0;
    dyAcc = 0;

    if (locked) {
      Audio.resume();                 // first user gesture unlocks WebAudio
      pushEvent('start');
    } else {
      if (firing) { firing = false; pushEvent('fireUp'); }
      clearKeys();
      pushEvent('stop');
    }
    flushInput(true);                 // one message carries the whole transition
    updateOverlay();
  }

  function releaseInput() {
    clearKeys();
    dxAcc = 0;
    dyAcc = 0;
    if (firing) { firing = false; pushEvent('fireUp'); }
    flushInput(true);                 // rAF may already be paused (blur / hidden)
  }

  /* ---- mouse ---- */

  function onMouseMove(e) {
    if (!state.locked) return;
    var mx = e.movementX || 0;
    var my = e.movementY || 0;
    dxAcc += clamp(mx, -MAX_MOUSE_DELTA, MAX_MOUSE_DELTA);
    dyAcc += clamp(my, -MAX_MOUSE_DELTA, MAX_MOUSE_DELTA);
    dxAcc = clamp(dxAcc, -MAX_MOUSE_ACC, MAX_MOUSE_ACC);
    dyAcc = clamp(dyAcc, -MAX_MOUSE_ACC, MAX_MOUSE_ACC);
  }

  function onMouseDown(e) {
    if (!state.locked) return;                 // the click handler locks instead
    if (e.button === 1) e.preventDefault();    // no middle-click autoscroll
    if (e.button === 0) {
      firing = true;
      queueEvent('fireDown');
    } else if (e.button === 2) {
      queueEvent('toggle');                    // right button == Q
    }
  }

  function onMouseUp(e) {
    if (!state.locked) return;
    if (e.button === 0 && firing) {
      firing = false;
      queueEvent('fireUp');
    }
  }

  function onContextMenu(e) {
    e.preventDefault();                        // right button is "toggle"
  }

  function onClick() {
    if (!state.locked) requestLock();
  }

  /* ---- keyboard ---- */

  function onKeyDown(e) {
    if (e.code === 'Escape') return;            // never intercepted
    if (!state.locked) return;                  // keys only while locked
    var bit = KEY_BITS[e.code];
    if (bit) {
      if (KEY_SCROLL_CODES[e.code]) e.preventDefault();   // no page scrolling
      if (!keysDown[e.code]) {
        keysDown[e.code] = bit;
        keyMask |= bit;
      }
      if (!e.repeat) {
        if (e.code === 'KeyQ') queueEvent('toggle');
        else if (e.code === 'KeyR') queueEvent('reload');
      }
    } else if (e.code === 'KeyF') {
      if (!e.repeat) queueEvent('reset');
    }
  }

  function onKeyUp(e) {
    if (e.code === 'Escape') return;
    var bit = keysDown[e.code];
    if (!bit) return;
    delete keysDown[e.code];
    keyMask &= ~bit;
  }

  /* ================================================================== *
   * Audio — everything is synthesized, no asset files
   * ================================================================== */

  var Audio = (function () {
    var SOUND_MIN_GAP = {
      shot: 0.05, shatter: 0.12, impact: 0.03, ricochet: 0.05, splash: 0.05,
      reload: 0.16, draw: 0.16, holster: 0.16, empty: 0.08, hit: 0.035, reset: 0.4
    };

    var actx = null;
    var master = null;
    var noiseBuf = null;
    var last = Object.create(null);
    var enabled = true;

    function init() {
      if (actx || !enabled) return actx;
      var AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) { enabled = false; return null; }
      try {
        actx = new AC();
      } catch (err) {
        enabled = false;
        return null;
      }

      // Soft limiter so stacked voices never clip.
      var comp = actx.createDynamicsCompressor();
      comp.threshold.value = -10;
      comp.knee.value = 8;
      comp.ratio.value = 6;
      comp.attack.value = 0.004;
      comp.release.value = 0.18;

      master = actx.createGain();
      master.gain.value = 0.5;
      master.connect(comp);
      comp.connect(actx.destination);

      // One shared noise buffer for every noise-based voice.
      var sr = actx.sampleRate;
      var n = Math.max(1, Math.floor(sr * 0.5));
      noiseBuf = actx.createBuffer(1, n, sr);
      var d = noiseBuf.getChannelData(0);
      for (var i = 0; i < n; i++) d[i] = Math.random() * 2 - 1;

      return actx;
    }

    function resume() {
      var c = init();
      if (c && c.state === 'suspended' && c.resume) {
        var p = c.resume();
        if (p && typeof p.catch === 'function') p.catch(function () {});
      }
    }

    // Filtered noise burst with an exponential decay tail.
    function noise(t0, dur, peak, type, f0, f1, q) {
      var src = actx.createBufferSource();
      src.buffer = noiseBuf;
      src.loop = true;

      var flt = actx.createBiquadFilter();
      flt.type = type;
      flt.Q.value = q;
      flt.frequency.setValueAtTime(Math.max(20, f0), t0);
      if (f1 && f1 !== f0) {
        flt.frequency.exponentialRampToValueAtTime(Math.max(20, f1), t0 + dur);
      }

      var g = actx.createGain();
      g.gain.setValueAtTime(0.0001, t0);
      g.gain.linearRampToValueAtTime(peak, t0 + Math.min(0.006, dur * 0.25));
      g.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);

      src.connect(flt);
      flt.connect(g);
      g.connect(master);
      src.start(t0, Math.random() * 0.4);
      src.stop(t0 + dur + 0.02);
      src.onended = function () {
        try { src.disconnect(); flt.disconnect(); g.disconnect(); } catch (err) {}
      };
    }

    // Oscillator voice with an optional pitch sweep.
    function tone(t0, dur, peak, type, f0, f1) {
      var osc = actx.createOscillator();
      osc.type = type;
      osc.frequency.setValueAtTime(Math.max(20, f0), t0);
      if (f1 && f1 !== f0) {
        osc.frequency.exponentialRampToValueAtTime(Math.max(20, f1), t0 + dur);
      }

      var g = actx.createGain();
      g.gain.setValueAtTime(0.0001, t0);
      g.gain.linearRampToValueAtTime(peak, t0 + Math.min(0.004, dur * 0.15));
      g.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);

      osc.connect(g);
      g.connect(master);
      osc.start(t0);
      osc.stop(t0 + dur + 0.02);
      osc.onended = function () {
        try { osc.disconnect(); g.disconnect(); } catch (err) {}
      };
    }

    function click(t0, peak, freq) {
      noise(t0, 0.022, peak, 'bandpass', freq, freq * 0.75, 3);
    }

    var VOICES = {
      // Punchy: crack + swept lowpass body + short thump, then a room tail.
      shot: function (t) {
        noise(t, 0.055, 0.5, 'highpass', 2400, 3200, 0.7);
        noise(t, 0.26, 0.85, 'lowpass', 6000, 260, 0.9);
        tone(t, 0.20, 0.7, 'sine', 130, 42);
        noise(t + 0.012, 0.32, 0.09, 'lowpass', 900, 280, 0.7);
      },
      // Bright, shard-like: noise burst + detuned high partials.
      shatter: function (t) {
        noise(t, 0.13, 0.5, 'highpass', 3200, 5200, 0.8);
        noise(t + 0.008, 0.5, 0.15, 'bandpass', 4200, 1700, 1.2);
        for (var i = 0; i < 7; i++) {
          var f = 1700 + Math.random() * 5300;
          var dt = t + Math.random() * 0.09;
          tone(dt, 0.09 + Math.random() * 0.42, 0.05 + Math.random() * 0.05,
            (i & 1) ? 'triangle' : 'sine', f, f * (0.9 + Math.random() * 0.12));
        }
      },
      impact: function (t) {
        click(t, 0.42, 1500);
        tone(t, 0.07, 0.28, 'sine', 240, 90);
      },
      ricochet: function (t) {
        tone(t, 0.34, 0.2, 'triangle', 2600, 620);
        tone(t + 0.006, 0.30, 0.11, 'sine', 3150, 880);
        noise(t, 0.05, 0.14, 'highpass', 3000, 4200, 0.7);
      },
      splash: function (t) {
        noise(t, 0.22, 0.4, 'bandpass', 1400, 260, 1.1);
        tone(t, 0.18, 0.15, 'sine', 380, 120);
      },
      // Two mechanical clicks (plus the magazine seating).
      reload: function (t) {
        click(t, 0.34, 1900);
        tone(t, 0.05, 0.09, 'square', 1200, 700);
        click(t + 0.13, 0.3, 1300);
        click(t + 0.27, 0.36, 2400);
        tone(t + 0.27, 0.06, 0.08, 'triangle', 900, 1500);
      },
      draw: function (t) {
        noise(t, 0.16, 0.2, 'bandpass', 500, 2600, 1.2);
        tone(t + 0.04, 0.09, 0.07, 'triangle', 700, 1400);
      },
      holster: function (t) {
        noise(t, 0.16, 0.18, 'bandpass', 2400, 420, 1.2);
        tone(t, 0.08, 0.06, 'triangle', 1200, 620);
      },
      empty: function (t) {
        click(t, 0.3, 2600);
        tone(t, 0.03, 0.09, 'square', 1500, 900);
      },
      hit: function (t) {
        tone(t, 0.05, 0.1, 'sine', 1900, 1500);
        noise(t, 0.012, 0.11, 'highpass', 4200, 5200, 0.7);
      },
      reset: function (t) {
        noise(t, 0.28, 0.2, 'bandpass', 350, 2200, 1.0);
        tone(t, 0.3, 0.13, 'sine', 240, 120);
      }
    };

    function play(name) {
      var voice = VOICES[name];
      if (!voice || !enabled) return;
      var c = init();
      if (!c) return;
      if (c.state === 'suspended') resume();

      var t = c.currentTime + 0.001;
      var gap = SOUND_MIN_GAP[name] || 0.03;
      var prev = last[name];
      if (prev !== undefined && t - prev < gap) return;   // retrigger guard
      last[name] = t;
      voice(t);
    }

    return { play: play, resume: resume };
  })();

  /* ================================================================== *
   * Wiring
   * ================================================================== */

  function onResize() {
    layout();                                  // instant letterbox update
    if (resizeTimer) clearTimeout(resizeTimer);
    resizeTimer = setTimeout(function () {
      resizeTimer = 0;
      maybeSendResize();
    }, RESIZE_DEBOUNCE_MS);
  }

  function onBlur() {
    if (state.locked) releaseInput();
    else { dxAcc = 0; dyAcc = 0; }
  }

  function onVisibility() {
    if (document.hidden) {
      dxAcc = 0;
      dyAcc = 0;
      if (state.locked) releaseInput();
    }
  }

  function tick() {
    requestAnimationFrame(tick);
    if (state.open) flushInput();
  }

  document.addEventListener('mousemove', onMouseMove, { passive: true });
  document.addEventListener('mousedown', onMouseDown, false);
  document.addEventListener('mouseup', onMouseUp, false);
  document.addEventListener('click', onClick, false);
  document.addEventListener('contextmenu', onContextMenu, false);
  document.addEventListener('keydown', onKeyDown, false);
  document.addEventListener('keyup', onKeyUp, false);
  document.addEventListener('pointerlockchange', onLockChange, false);
  document.addEventListener('pointerlockerror', function () { /* silent */ }, false);
  window.addEventListener('resize', onResize, { passive: true });
  window.addEventListener('blur', onBlur, false);
  document.addEventListener('visibilitychange', onVisibility, false);

  layout();
  updateOverlay();
  connect();
  requestAnimationFrame(tick);
})();
