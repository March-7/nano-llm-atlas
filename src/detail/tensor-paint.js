/* Model-space y increases downwards. A face is parameterized in real tensor
 * indices, so grid drawing, picking and highlighted slices use the same map. */
export const palette = {
  w: { front: [91, 90, 102], edge: '#545465', dot: [77, 76, 174], flow: '111,108,194' },
  i: { front: [86, 92, 85], edge: '#4b584e', dot: [54, 146, 59], flow: '98,173,92' },
  a: { front: [100, 95, 77], edge: '#686546', dot: [188, 153, 55], flow: '173,145,68' },
};
export const clamp = (n, lo, hi) => Math.max(lo, Math.min(hi, n));
export const center = (n) => [n.x + n.dx / 2, n.y + n.dy / 2, n.z + n.dz / 2];
export const box = (n) => [n.x, n.y, n.z, n.x + n.dx, n.y + n.dy, n.z + n.dz];
const distantInk = { w: [77, 76, 155], i: [58, 113, 81], a: [151, 119, 58] };
const pointKeys = ['x', 'y', 'z'],
  lengths = ['dx', 'dy', 'dz'],
  counts = ['cx', 'cy', 'cz'];
const planeAxes = [
  [2, 1],
  [0, 2],
  [0, 1],
];
const rgb = (a) => 'rgb(' + a.join(',') + ')';
export const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

export function basis(yaw, pitch) {
  const s = Math.sin(yaw),
    c = Math.cos(yaw),
    p = Math.sin(pitch),
    q = Math.cos(pitch);
  return { right: [c, 0, -s], down: [s * p, q, c * p], eye: [s * q, -p, c * q] };
}

export function cameraMap(ax, target, scale, w, h) {
  const { right, down } = ax;
  const project = (p) => {
    const x = p[0] - target[0],
      y = p[1] - target[1],
      z = p[2] - target[2];
    return [
      w / 2 + (right[0] * x + right[1] * y + right[2] * z) * scale,
      h / 2 + (down[0] * x + down[1] * y + down[2] * z) * scale,
    ];
  };
  return { ...ax, target, scale, w, h, project };
}

export function screenBox(b, view) {
  const mid = [(b[0] + b[3]) / 2, (b[1] + b[4]) / 2, (b[2] + b[5]) / 2];
  const half = [(b[3] - b[0]) / 2, (b[4] - b[1]) / 2, (b[5] - b[2]) / 2];
  const [x, y] = view.project(mid);
  const span = (v) =>
    (Math.abs(v[0]) * half[0] + Math.abs(v[1]) * half[1] + Math.abs(v[2]) * half[2]) * view.scale;
  const rx = span(view.right),
    ry = span(view.down);
  return [x - rx, y - ry, x + rx, y + ry];
}
export const intersects = (a, b) => a[2] >= b[0] && a[0] <= b[2] && a[3] >= b[1] && a[1] <= b[3];

export function planes(node, view) {
  const result = [];
  for (let normal = 0; normal < 3; normal++) {
    if (Math.abs(view.eye[normal]) < 0.001) continue;
    const [u, v] = planeAxes[normal];
    const high = view.eye[normal] > 0;
    const origin = [node.x, node.y, node.z];
    if (high) origin[normal] += node[lengths[normal]];
    const unitU = node[lengths[u]] / node[counts[u]],
      unitV = node[lengths[v]] / node[counts[v]];
    const du = [view.scale * view.right[u] * unitU, view.scale * view.down[u] * unitU];
    const dv = [view.scale * view.right[v] * unitV, view.scale * view.down[v] * unitV];
    const det = du[0] * dv[1] - du[1] * dv[0];
    if (Math.abs(det) < 1e-14) continue;
    const base = view.project(origin);
    result.push({ node, normal, u, v, high, base, du, dv, det, nu: node[counts[u]], nv: node[counts[v]] });
  }
  return result;
}
const screenAt = (f, u, v) => [
  f.base[0] + f.du[0] * u + f.dv[0] * v,
  f.base[1] + f.du[1] * u + f.dv[1] * v,
];
function uv(f, x, y) {
  const dx = x - f.base[0],
    dy = y - f.base[1];
  return [(dx * f.dv[1] - dy * f.dv[0]) / f.det, (dy * f.du[0] - dx * f.du[1]) / f.det];
}
export function pickFace(f, x, y, view) {
  const [u, v] = uv(f, x, y);
  if (u < -1e-8 || u > f.nu + 1e-8 || v < -1e-8 || v > f.nv + 1e-8) return null;
  const n = f.node,
    pos = [n.x, n.y, n.z],
    idx = [0, 0, 0];
  idx[f.u] = clamp(Math.floor(u), 0, f.nu - 1);
  idx[f.v] = clamp(Math.floor(v), 0, f.nv - 1);
  idx[f.normal] = f.high ? n[counts[f.normal]] - 1 : 0;
  pos[f.u] += (u * n[lengths[f.u]]) / f.nu;
  pos[f.v] += (v * n[lengths[f.v]]) / f.nv;
  if (f.high) pos[f.normal] += n[lengths[f.normal]];
  return {
    node: n.whole ?? n,
    x: idx[0],
    y: idx[1],
    z: idx[2] + (n.sliceZ ?? 0),
    distance: dot(pos, view.eye),
  };
}
export function cellPoint(n, x, y, z) {
  const idx = [x, y, z];
  return idx.map((a, i) => n[pointKeys[i]] + ((a + 0.5) * n[lengths[i]]) / n[counts[i]]);
}
function viewportRange(f, view) {
  const samples = [
    [0, 0],
    [view.w, 0],
    [view.w, view.h],
    [0, view.h],
  ].map(([x, y]) => uv(f, x, y));
  return [
    clamp(Math.floor(Math.min(...samples.map((p) => p[0]))) - 1, 0, f.nu),
    clamp(Math.floor(Math.min(...samples.map((p) => p[1]))) - 1, 0, f.nv),
    clamp(Math.ceil(Math.max(...samples.map((p) => p[0]))) + 1, 0, f.nu),
    clamp(Math.ceil(Math.max(...samples.map((p) => p[1]))) + 1, 0, f.nv),
  ];
}
function rect(ctx, f, x0, y0, x1, y1) {
  ctx.beginPath();
  ctx.moveTo(...screenAt(f, x0, y0));
  ctx.lineTo(...screenAt(f, x1, y0));
  ctx.lineTo(...screenAt(f, x1, y1));
  ctx.lineTo(...screenAt(f, x0, y1));
  ctx.closePath();
}
function drawGrid(ctx, f, r, step, color, width) {
  if (step <= 0) return;
  const [x0, y0, x1, y1] = r;
  ctx.beginPath();
  for (let x = Math.ceil(x0 / step) * step; x <= x1; x += step) {
    ctx.moveTo(...screenAt(f, x, y0));
    ctx.lineTo(...screenAt(f, x, y1));
  }
  for (let y = Math.ceil(y0 / step) * step; y <= y1; y += step) {
    ctx.moveTo(...screenAt(f, x0, y));
    ctx.lineTo(...screenAt(f, x1, y));
  }
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.stroke();
}

function gain(n, shape) {
  if (n.valueScale != null) return n.valueScale;
  if (n.op === 'idxObj') return 1 / (shape?.vocabSize || 3);
  if (n.op === 'lmHeadWeight') return 5;
  if (n.op === 'attnMtxSm') return 2;
  if (/lnSigma/.test(n.op)) return 0.5;
  if (/Embed|lnAgg/.test(n.op) || n.op === 'residual0') return 10;
  if (n.kind === 'w' && !/lnMu/.test(n.op)) {
    const d = shape?.C || Math.max(n.cx, n.cy);
    return /^(q|k|v|qkv)(Weight|Bias)Block$/.test(n.op) ? d * 0.25 : d * 0.5;
  }
  return 1;
}
// Canvas backing stores vary with zoom and DPR. Limit their total pixels,
// including the larger high-magnification tiles (about 24 MB of RGBA data).
export class TileCache extends Map {
  pixels = 0;
  set(key, bitmap) {
    const size = bitmap.width * bitmap.height;
    while (this.size && this.pixels + size > 6_000_000) {
      const oldest = this.keys().next().value,
        old = this.get(oldest);
      this.pixels -= old.width * old.height;
      this.delete(oldest);
    }
    this.pixels += size;
    return super.set(key, bitmap);
  }
  clear() {
    super.clear();
    this.pixels = 0;
  }
}
const chunk = 16;
function tileFor(f, tileU, tileV, sample, tileCache, pixels, shape) {
  const n = f.node;
  const key =
    n.id + '/' + (n.sliceZ ?? 'all') + '/' + [f.normal, Number(f.high), tileU, tileV, pixels].join('/');
  if (tileCache.has(key)) return tileCache.get(key);
  const rows = Math.min(chunk, f.nv - tileV),
    cols = Math.min(chunk, f.nu - tileU);
  const bitmap = document.createElement('canvas');
  bitmap.width = cols * pixels;
  bitmap.height = rows * pixels;
  const ctx = bitmap.getContext('2d'),
    tone = palette[n.kind],
    front = tone.front;
  const shade = f.normal === 2 ? 1 : f.normal === 1 ? 1.12 : 0.79;
  const idx = [0, 0, 0],
    k = gain(n, shape);
  idx[f.normal] = f.high ? n[counts[f.normal]] - 1 : 0;
  const base = front.map((a) => Math.round(a * shade));
  const other = rgb(base.map((a) => a - 5));
  ctx.fillStyle = rgb(base);
  ctx.fillRect(0, 0, bitmap.width, bitmap.height);
  const colors = [];
  for (let i = 0; i <= 32; i++) {
    const t = (i - 16) / 16;
    colors.push(rgb(base.map((v, j) => Math.round(v + Math.abs(t) * ((t > 0 ? tone.dot[j] : 7) - v)))));
  }
  for (let row = 0; row < rows; row++) {
    idx[f.v] = tileV + row;
    for (let col = 0; col < cols; col++) {
      idx[f.u] = tileU + col;
      const px = col * pixels,
        py = row * pixels;
      if ((idx[0] + idx[1] + idx[2]) % 2) {
        ctx.fillStyle = other;
        ctx.fillRect(px, py, pixels, pixels);
      }
      const val = sample.value(n.whole ?? n, idx[0], idx[1], idx[2] + (n.sliceZ ?? 0));
      if (typeof val !== 'number' || Number.isNaN(val)) continue;
      const bucket = Math.round(clamp(val * k, -1, 1) * 16) + 16;
      ctx.fillStyle = colors[bucket];
      ctx.beginPath();
      ctx.arc(px + pixels / 2, py + pixels / 2, pixels * 0.304, 0, 2 * Math.PI);
      ctx.fill();
    }
  }
  tileCache.set(key, bitmap);
  return bitmap;
}
function valueTiles(ctx, f, r, sample, cache, pixelSpacing, shape) {
  const pixels = pixelSpacing > 14 ? 18 : 7;
  const [x0, y0, x1, y1] = r;
  ctx.save();
  ctx.transform(f.du[0], f.du[1], f.dv[0], f.dv[1], ...f.base);
  for (let y = Math.floor(y0 / chunk) * chunk; y < y1; y += chunk) {
    for (let x = Math.floor(x0 / chunk) * chunk; x < x1; x += chunk) {
      const tile = tileFor(f, x, y, sample, cache, pixels, shape);
      ctx.drawImage(tile, x, y, tile.width / pixels, tile.height / pixels);
    }
  }
  ctx.restore();
}

export function paintNode(
  ctx,
  n,
  view,
  { sample = null, cache, shape, selected, dependency, flat = false, roleTint = 0 } = {},
) {
  const faces = planes(n, view);
  for (const f of faces) {
    const tone = palette[n.kind],
      shade = f.normal === 2 ? 1 : f.normal === 1 ? 1.14 : 0.78;
    rect(ctx, f, 0, 0, f.nu, f.nv);
    ctx.fillStyle = rgb(
      tone.front.map((a, i) => Math.round((a + roleTint * (distantInk[n.kind][i] - a)) * shade)),
    );
    ctx.fill();
    if (flat) continue;
    const perp = Math.min(Math.abs(f.det) / Math.hypot(...f.du), Math.abs(f.det) / Math.hypot(...f.dv));
    const totalWidth = Math.hypot(...f.du) * f.nu,
      totalHeight = Math.hypot(...f.dv) * f.nv;
    if (totalWidth < 1.3 || totalHeight < 1.3) continue;
    const range = viewportRange(f, view);
    if (range[2] <= range[0] || range[3] <= range[1]) continue;
    if (sample && perp >= 1.98) valueTiles(ctx, f, range, sample, cache, perp, shape);
    if (perp >= 3.0) drawGrid(ctx, f, range, 1, 'rgba(34,37,42,0.26)', 0.42);
    let step = 16;
    while (step * perp < 5) step *= 16;
    drawGrid(ctx, f, range, step, 'rgba(227,227,225,0.37)', 0.64);
    if (step * 16 * perp < 1600) drawGrid(ctx, f, range, step * 16, 'rgba(239,235,220,0.53)', 0.9);
    rect(ctx, f, 0, 0, f.nu, f.nv);
    ctx.strokeStyle = selected
      ? '#e3cd60'
      : dependency
        ? n.kind === 'w'
          ? '#7270c8'
          : '#68a565'
        : tone.edge;
    ctx.lineWidth = selected || dependency ? 1.45 : 0.66;
    ctx.stroke();
  }
  return faces;
}

export function paintSlice(ctx, node, indices, view, { color = '#eac553', length, axis = -1 } = {}) {
  const lines = [],
    p = cellPoint(node, ...indices);
  for (const f of planes(node, view)) {
    const minU = clamp(indices[f.u], 0, f.nu - 1),
      minV = clamp(indices[f.v], 0, f.nv - 1);
    const maxU = axis === f.u ? Math.min(f.nu, minU + length) : minU + 1;
    const maxV = axis === f.v ? Math.min(f.nv, minV + length) : minV + 1;
    const onSurface = indices[f.normal] === (f.high ? node[counts[f.normal]] - 1 : 0);
    if (!onSurface && axis !== f.normal) continue;
    rect(ctx, f, minU, minV, maxU, maxV);
    ctx.save();
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.34;
    ctx.fill();
    ctx.restore();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.4;
    ctx.stroke();
    lines.push(f);
  }
  if (!lines.length) {
    // Internal cells and reduction slices project to a full 3D box too. Draw
    // its actual outline even when none of its faces is on the tensor exterior.
    const local = { cx: 1, cy: 1, cz: 1 };
    for (let k = 0; k < 3; k++) {
      const unit = node[lengths[k]] / node[counts[k]];
      local[pointKeys[k]] = node[pointKeys[k]] + indices[k] * unit;
      local[lengths[k]] = unit * (k === axis ? length : 1);
    }
    for (const f of planes(local, view)) {
      rect(ctx, f, 0, 0, 1, 1);
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.25;
      ctx.stroke();
    }
  }
  if ((view.scale * node.dx) / node.cx < 3) {
    const [x, y] = view.project(p);
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.15;
    ctx.strokeRect(x - 2, y - 2, 4, 4);
  }
  return view.project(p);
}
