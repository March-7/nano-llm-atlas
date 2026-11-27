import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
  arrangeWorld,
  localView,
  modelOrigin,
  overviewPitch,
  overviewYaw,
  toLocal,
  toWorld,
  worldInsets,
} from '../src/detail/atlas-world.js';
import { basis, cameraMap, cellPoint, intersects, screenBox } from '../src/detail/tensor-paint.js';
import { mountTensorScene } from '../src/detail/tensor-scene.js';
import { classicLayout } from '../src/detail/classic-layout.js';
import { modernExample } from '../src/detail/modern.js';

const entries = ['gpt2-small', 'nanogpt', 'gpt2-xl', 'gpt3', 'llama', 'qwen'].map((id, i) => ({
  id,
  layout: i < 4 ? classicLayout(id) : modernExample(id).layout,
}));
function close(actual, expected, epsilon = 1e-7) {
  assert.equal(actual.length, expected.length);
  actual.forEach((v, i) => assert.ok(Math.abs(v - expected[i]) < epsilon, v + ' ≠ ' + expected[i]));
}

test('the six real geometries fit below the toolbar in one wide or narrow world', () => {
  for (const [w, h, columns] of [
    [1152, 740, 3],
    [390, 650, 2],
    [768, 490, 3],
  ]) {
    const world = arrangeWorld(entries, w, h);
    const camera = cameraMap(basis(overviewYaw, overviewPitch), [0, 0, 0], 1, w, h);
    const rectangles = [...world.values()].map((m) => screenBox(m.layout.bounds, localView(m, camera)));
    assert.equal(rectangles.length, 6);
    rectangles.forEach(([l, t, r, b], i) => {
      assert.ok(l >= 0 && r <= w && t - 20 >= worldInsets(w, h).top - 3 && b <= h);
      for (const other of rectangles.slice(i + 1)) assert.equal(intersects([l, t, r, b], other), false);
      if (i >= columns) assert.ok(t > rectangles[i - columns][3]);
    });
  }
});

test('zoomed cell and whole-atlas projections share the same camera and preserve each model’s aspect ratios', () => {
  const world = arrangeWorld(entries, 1152, 740);
  for (const m of world.values()) {
    const n = m.layout.layers.at(-1).nodes().at(-1);
    const p = cellPoint(n, n.cx - 1, n.cy - 1, n.cz - 1);
    const q = p.map((x, i) => x + [7, 5, 9][i]);
    close(toLocal(m, toWorld(m, p)), p);
    for (const [yaw, pitch] of [
      [overviewYaw, overviewPitch],
      [-1.2, 0.6],
      [2.4, -0.25],
    ]) {
      // A camera centered on one individual scalar still projects another model
      // with the exact same eye, scale and origin as the full world.
      const axes = basis(yaw, pitch),
        a = toWorld(m, p, axes),
        b = toWorld(m, q, axes);
      close(toLocal(m, a, axes), p);
      assert.ok(Math.abs(Math.hypot(...b.map((v, i) => v - a[i])) / m.unit - Math.sqrt(155)) < 1e-7);
      const camera = cameraMap(axes, a, 25 / (m.unit * m.layout.cell), 1152, 740);
      close(localView(m, camera).project(q), camera.project(b), 1e-6);
      for (const other of world.values()) {
        close(
          localView(other, camera).project(other.localCenter),
          camera.project(modelOrigin(other, camera)),
          1e-5,
        );
      }
    }
  }
});

const orbitAngles = [
  [overviewYaw, overviewPitch],
  [1.08, 0.38], // Llama whole model, wheel out, then horizontal + diagonal middle drag
  [Math.PI / 2, 0],
  [Math.PI, 0.61],
  [-2.2, -0.8],
  [-Math.PI / 2, 1.18],
  [2 * Math.PI + overviewYaw, overviewPitch],
];

test('all six model-space verticals stay parallel throughout a full yaw and diagonal orbit', () => {
  const world = arrangeWorld(entries, 990, 850);
  for (const [yaw, pitch] of orbitAngles) {
    const camera = cameraMap(basis(yaw, pitch), [0, 0, 0], 1, 990, 850);
    for (const m of world.values()) {
      const v = localView(m, camera),
        c = m.localCenter;
      const start = v.project(c),
        end = v.project([c[0], c[1] + 1 / m.unit, c[2]]);
      assert.ok(Math.abs(end[0] - start[0]) < 1e-7, m.id + ' has a diagonal residual axis at yaw ' + yaw);
      assert.ok(Math.abs(end[1] - start[1] - Math.cos(pitch)) < 1e-7);
    }
  }
});

test('comparison positions and pan survive a full turn while tensor depth still rotates', () => {
  for (const [w, h] of [
    [990, 850],
    [390, 650],
  ]) {
    const world = arrangeWorld(entries, w, h);
    const neutral = cameraMap(basis(overviewYaw, overviewPitch), [0, 0, 0], 0.91, w, h);
    const baseline = [...world.values()].map((m) => localView(m, neutral).project(m.localCenter));
    let depthMoved = false;
    for (const [yaw, pitch] of orbitAngles) {
      const axes = basis(yaw, pitch);
      const pan = axes.right.map((val, k) => val * 23 + axes.down[k] * 19);
      const camera = cameraMap(axes, pan, 0.91, w, h);
      const rectangles = [];
      [...world.values()].forEach((m, i) => {
        const v = localView(m, camera),
          c = m.localCenter;
        const p = v.project(c);
        close(p, [baseline[i][0] - 0.91 * 23, baseline[i][1] - 0.91 * 19]);
        const rect = screenBox(m.layout.bounds, v);
        for (const other of rectangles)
          assert.equal(intersects(rect, other), false, m.id + ' overlaps at yaw ' + yaw);
        rectangles.push(rect);
        const alongZ = v.project([c[0], c[1], c[2] + 15 / m.unit])[0] - p[0];
        if (Math.abs(alongZ + 0.91 * 15 * Math.sin(overviewYaw)) > 3) depthMoved = true;
      });
    }
    assert.ok(depthMoved, 'tensor depth must respond to real camera motion');
  }
});

test('Llama fit, two wheel events and middle-drag keep six labels visible and preserve the wheel angle', (t) => {
  // Exercise the real scene input handlers and projection. A no-op raster
  // context avoids a GPU/browser dependency for camera regressions.
  const w = 980,
    h = 856,
    listeners = new Map(),
    frames = new Map();
  let nextFrame = 0,
    zoom = 0,
    markers = [],
    active = null;
  const context = new Proxy(
    { measureText: (s) => ({ width: s.length * 5 }) },
    {
      get: (obj, k) => (k in obj ? obj[k] : () => {}),
    },
  );
  const surface = () => ({
    style: {},
    width: w,
    height: h,
    getContext: () => context,
    hasAttribute: () => true,
    getBoundingClientRect: () => ({ left: 0, top: 0, width: w, height: h }),
    addEventListener: (name, fn) => listeners.set(name, fn),
    setPointerCapture() {},
    hasPointerCapture: () => false,
  });
  const restore = [];
  function replace(name, value) {
    const saved = Object.getOwnPropertyDescriptor(globalThis, name);
    Object.defineProperty(globalThis, name, { configurable: true, value });
    restore.push(() => (saved ? Object.defineProperty(globalThis, name, saved) : delete globalThis[name]));
  }
  replace('devicePixelRatio', 1);
  replace('window', { matchMedia: () => ({ matches: true }), addEventListener() {} });
  replace('document', { createElement: surface });
  replace(
    'ResizeObserver',
    class {
      observe() {}
      disconnect() {}
    },
  );
  replace('requestAnimationFrame', (f) => {
    frames.set(++nextFrame, f);
    return nextFrame;
  });
  replace('cancelAnimationFrame', (id) => frames.delete(id));
  const scene = mountTensorScene(surface(), {
    onZoom: (value) => {
      zoom = value;
    },
    onCell: (hit) => {
      active = hit;
    },
    onWorldLabels: (list) => {
      markers = list;
    },
  });
  t.after(() => {
    scene.destroy();
    restore.reverse().forEach((f) => f());
  });
  function render() {
    let i = 0;
    while (frames.size) {
      assert.ok(++i < 8, 'camera should settle without an idle rendering loop');
      const work = [...frames.values()];
      frames.clear();
      work.forEach((f) => f(performance.now()));
    }
  }
  function drag(dx, dy) {
    let prevented = false;
    const e = {
      button: 1,
      pointerId: 1,
      pointerType: 'mouse',
      clientX: 520,
      clientY: 390,
      preventDefault() {
        prevented = true;
      },
    };
    listeners.get('pointerdown')(e);
    assert.ok(prevented, 'middle press must suppress native autoscroll');
    listeners.get('pointermove')({ ...e, clientX: e.clientX + dx, clientY: e.clientY + dy });
    listeners.get('pointerup')({
      ...e,
      type: 'pointerup',
      clientX: e.clientX + dx,
      clientY: e.clientY + dy,
    });
    render();
  }
  const llama = entries.find((e) => e.id === 'llama').layout;
  const tensor = llama.layers[0].nodes().find((n) => n.op === 'attnMtx');
  function projectedAxes() {
    const p = scene.inspectCell(tensor, 0, 0, 0);
    const x = scene.inspectCell(tensor, 3, 0, 0);
    const y = scene.inspectCell(tensor, 0, 3, 0);
    assert.equal(active.node.id, tensor.id);
    return [x.screenX - p.screenX, x.screenY - p.screenY, y.screenX - p.screenX, y.screenY - p.screenY];
  }
  function twoWheels() {
    for (let i = 0; i < 2; i++) {
      listeners.get('wheel')({
        clientX: w * 0.54,
        clientY: h * 0.48,
        deltaY: 210,
        deltaMode: 0,
        preventDefault() {},
      });
      render();
    }
  }
  scene.setWorld(entries);
  scene.setLayout(llama);
  scene.fitModel();
  render();
  const initialZoom = zoom,
    initialVectors = projectedAxes();
  twoWheels();
  assert.ok(zoom > 0.7 && zoom <= 1);
  close(
    projectedAxes(),
    initialVectors.map((v) => (v * zoom) / initialZoom),
  );
  const before = new Map(markers.map((m) => [m.id, m.x]));
  assert.equal(before.size, 6);
  const vectors = projectedAxes();
  drag(-123.333, 32);
  const rotated = projectedAxes();
  assert.ok(Math.abs(rotated[0] - vectors[0]) > 0.001);
  for (const marker of markers) {
    assert.ok(marker.y >= worldInsets(w, h).top + 8 && marker.y < h - 12);
    close([marker.x], [before.get(marker.id)]);
  }
  assert.equal(markers.length, 6);
  scene.pan(9, -12);
  render();
  const shifted = markers.map((m) => m.x);
  drag(-213, 90);
  close(
    markers.map((m) => m.x),
    shifted,
  );
  // Return to a non-home angle before zooming. A reverse wheel must never
  // undo this user rotation to fit the layout.
  scene.fitModel();
  drag(-123.333, 32);
  const beforeZoom = zoom,
    beforeVectors = projectedAxes();
  twoWheels();
  close(
    projectedAxes(),
    beforeVectors.map((v) => (v * zoom) / beforeZoom),
  );
  // Enter a tensor close-up first: crossing back into atlas scale must never
  // interpolate the projection angle, even when the zoom changes by orders of magnitude.
  scene.focus('ln1.lnResid');
  const closeUpZoom = zoom;
  const closeUpAxes = projectedAxes();
  let pairs = 0;
  while (zoom > 0.85 && pairs++ < 16) {
    twoWheels();
    close(
      projectedAxes(),
      closeUpAxes.map((v) => (v * zoom) / closeUpZoom),
    );
  }
  assert.ok(pairs > 1 && zoom <= 0.85);
  let auxiliaryCancelled = false;
  listeners.get('auxclick')({
    button: 1,
    preventDefault() {
      auxiliaryCancelled = true;
    },
  });
  assert.ok(auxiliaryCancelled);
});
