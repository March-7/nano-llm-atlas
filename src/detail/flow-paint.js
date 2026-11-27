// All animation positions originate in tensor-local world coordinates. The
// caller owns the clock, camera and playback state, including scrubbing.
import { cellPoint, clamp, dot, paintSlice } from './tensor-paint.js';

const axes = [
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1],
];
const origins = ['x', 'y', 'z'];
const sizes = ['dx', 'dy', 'dz'];
const counts = ['cx', 'cy', 'cz'];
const materials = {
  w: { rim: '#524c91', side: '#7771be', front: '#afa4f3', top: '#e2dbff', seam: '#655f9e' },
  i: { rim: '#28654d', side: '#4b9370', front: '#91d4a5', top: '#d6f4ce', seam: '#528b68' },
  a: { rim: '#855b28', side: '#bd8a48', front: '#edbf74', top: '#fff0ba', seam: '#9e733d' },
};
const blend = (a, b, t) => a + (b - a) * t;
const mix = (a, b, t) => a.map((v, i) => blend(v, b[i], t));
const add = (p, v, k) => p.map((n, i) => n + v[i] * k);
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const normalized = (v) => {
  const length = Math.hypot(...v) || 1;
  return v.map((a) => a / length);
};
function smooth(x) {
  const t = clamp(x, 0, 1);
  return t * t * (3 - 2 * t);
}
function windowTime(p, start, end) {
  return clamp((p - start) / (end - start), 0, 1);
}

function validSpan(raw) {
  const n = raw?.node;
  if (
    !n ||
    raw.indices?.length !== 3 ||
    !Object.hasOwn(materials, n.kind) ||
    ![-1, 0, 1, 2].includes(raw.axis)
  )
    return null;
  if (
    !origins.every((k) => Number.isFinite(n[k])) ||
    !counts.every((k) => Number.isSafeInteger(n[k]) && n[k] > 0)
  )
    return null;
  const indices = Array.from(raw.indices, Math.floor);
  if (!indices.every(Number.isSafeInteger)) return null;
  const unit = sizes.map((k, i) => n[k] / n[counts[i]]);
  if (!unit.every((v) => Number.isFinite(v) && v > 0)) return null;
  const axis = raw.axis;
  let length = 1;
  if (axis >= 0) {
    const len = Math.floor(raw.length);
    if (!Number.isSafeInteger(len) || len < 1) return null;
    const last = Math.min(n[counts[axis]], indices[axis] + len);
    indices[axis] = Math.max(0, indices[axis]);
    length = last - indices[axis];
    if (length < 1) return null;
  }
  if (indices.some((v, i) => v < 0 || v >= n[counts[i]])) return null;
  return { node: n, indices, axis, length, unit };
}

function indexAt(s, offset) {
  const idx = [...s.indices];
  if (s.axis >= 0) idx[s.axis] += offset;
  return idx;
}
function at(s, offset) {
  return cellPoint(s.node, ...indexAt(s, offset));
}

/** Tight AABB of the addressed cells, in the same units as layout.bounds.
 * Out-of-range slices from padded timesteps may have an empty intersection. */
export function spanBounds(raw) {
  const s = validSpan(raw);
  if (!s) return null;
  const lo = s.indices.map((v, i) => s.node[origins[i]] + v * s.unit[i]);
  return [...lo, ...lo.map((v, i) => v + (i === s.axis ? s.length : 1) * s.unit[i])];
}

// Choose actual neighboring cells for a carried strip. At low magnification
// separated strips sample occupied intervals across the reduction, including
// both endpoints. Drawing work depends on the display budget, not tensor size.
function segment(s, k, groups) {
  const begin = Math.floor((k * s.length) / groups);
  const end = Math.floor(((k + 1) * s.length) / groups);
  const length = Math.max(1, Math.min(5, end - begin));
  const start =
    groups === 1 ? begin : Math.round(blend(begin, Math.max(begin, end - length), k / (groups - 1)));
  return { start, length, middle: start + (length - 1) / 2 };
}
function inputsOf(raw) {
  if (!raw?.length) return [];
  const lanes = [];
  const limit = Math.min(6, raw.length);
  for (let i = 0; i < limit; i++) {
    const src = validSpan(raw[Math.round((i * (raw.length - 1)) / Math.max(1, limit - 1))]);
    if (src) lanes.push(src);
  }
  return lanes;
}

function inFrame(p, view, margin = 16) {
  return (
    Number.isFinite(p[0]) &&
    Number.isFinite(p[1]) &&
    p[0] >= -margin &&
    p[0] <= view.w + margin &&
    p[1] >= -margin &&
    p[1] <= view.h + margin
  );
}
function lineInterval(a, b, view, pad) {
  let near = 0,
    far = 1;
  for (let k = 0; k < 2; k++) {
    const max = (k ? view.h : view.w) + pad,
      min = -pad,
      diff = b[k] - a[k];
    if (Math.abs(diff) < 0.000001) {
      if (a[k] < min || a[k] > max) return null;
    } else {
      const u = (min - a[k]) / diff,
        v = (max - a[k]) / diff;
      near = Math.max(near, Math.min(u, v));
      far = Math.min(far, Math.max(u, v));
      if (far < near) return null;
    }
  }
  return [near, far];
}
function visibleInterval(s, start, length, view) {
  const a = view.project(at(s, start)),
    b = view.project(at(s, start + length - 1));
  const interval = lineInterval(a, b, view, Math.max(6, Math.max(...s.unit) * view.scale * 1.8));
  if (!interval) return null;
  const [near, far] = interval;
  const first = Math.max(0, Math.floor(near * (length - 1)) - 1);
  const last = Math.min(length - 1, Math.ceil(far * (length - 1)) + 1);
  return [start + first, last - first + 1];
}
function band(ctx, s, start, length, view, ink, alpha) {
  if (!length || alpha <= 0.008) return;
  const visible = visibleInterval(s, start, length, view);
  if (!visible) return;
  // The clipped slab has its own endcaps, including for a row embedded deep
  // inside a parent matrix. Drawing the parent's far face would mark a
  // different coordinate and can be millions of cells outside the viewport.
  const idx = indexAt(s, visible[0]),
    slab = {};
  for (let i = 0; i < 3; i++) {
    slab[origins[i]] = s.node[origins[i]] + idx[i] * s.unit[i];
    slab[counts[i]] = i === s.axis ? visible[1] : 1;
    slab[sizes[i]] = s.unit[i] * slab[counts[i]];
  }
  ctx.save();
  ctx.globalAlpha *= alpha;
  paintSlice(ctx, slab, [0, 0, 0], view, { color: ink, axis: s.axis, length: visible[1] });
  ctx.restore();
}

// Favor the free face of a thin matrix. A depth slice can instead peel off its
// side. The chosen direction is fixed in the model as the viewer orbits.
function liftAxis(s) {
  const possible = [2, 1, 0].filter((i) => i !== s.axis);
  const a = possible.reduce((best, next) => (s.node[sizes[next]] < s.node[sizes[best]] ? next : best));
  const slip = s.axis === 0 ? [0, -0.84, 0.14] : [-0.68, -0.36, 0.15];
  return normalized(axes[a].map((v, i) => v * (a === 1 ? -1 : 1) + slip[i]));
}
function curve(a, b, liftA, liftB, height, t) {
  // A cubic with controls pulled away from the two tensor surfaces.
  const q = 1 - t;
  return a.map(
    (v, i) =>
      q * q * (1 + 2 * t) * v +
      t * t * (3 - 2 * t) * b[i] +
      3 * t * q * height * (q * liftA[i] + t * liftB[i]),
  );
}

// Generate all three visible planes of a compact, possibly turning prism.
function prism(ctx, view, item) {
  const { point, vectors, widths, ink, divisions, values, alpha } = item;
  const screen = view.project(point);
  const radius = Math.max(...widths) * view.scale;
  if (!inFrame(screen, view, radius + 4) || !Number.isFinite(radius) || alpha <= 0.009) return;
  const half = vectors.map((v, i) => v.map((q) => (q * widths[i]) / 2));
  const corners = [];
  for (let mask = 0; mask < 8; mask++) {
    const pt = point.map(
      (v, k) =>
        v +
        (mask & 1 ? 1 : -1) * half[0][k] +
        (mask & 2 ? 1 : -1) * half[1][k] +
        (mask & 4 ? 1 : -1) * half[2][k],
    );
    corners.push(view.project(pt));
  }
  ctx.save();
  ctx.globalAlpha *= alpha;
  const faces = [
    { a: 0, low: [0, 2, 6, 4], high: [1, 3, 7, 5] },
    { a: 1, low: [0, 1, 5, 4], high: [2, 3, 7, 6] },
    { a: 2, low: [0, 1, 3, 2], high: [4, 5, 7, 6] },
  ];
  let detailFace = null,
    detailArea = 0;
  for (const f of faces) {
    const facing = dot(vectors[f.a], view.eye);
    if (Math.abs(facing) < 0.003) continue;
    const face = (facing > 0 ? f.high : f.low).map((i) => corners[i]);
    const [a, b, , d] = face;
    const area = Math.abs((b[0] - a[0]) * (d[1] - a[1]) - (b[1] - a[1]) * (d[0] - a[0]));
    if (area < 0.25) continue;
    // Fixed world-space light distinguishes the narrow machined edge from the
    // broad cell face at every camera angle, including the reverse of a slab.
    const n = vectors[f.a].map((v) => v * Math.sign(facing));
    const shade = dot(n, [-0.28, -0.72, 0.63]);
    ctx.beginPath();
    ctx.moveTo(...a);
    ctx.lineTo(...b);
    ctx.lineTo(...face[2]);
    ctx.lineTo(...d);
    ctx.closePath();
    ctx.fillStyle = shade > 0.66 ? ink.top : shade > 0.08 ? ink.front : ink.side;
    ctx.fill();
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = 'rgba(252,252,239,0.84)';
    ctx.stroke();
    ctx.lineWidth = 1.0;
    ctx.strokeStyle = ink.rim;
    ctx.stroke();
    if (f.a === 0) continue;
    if (area > detailArea) {
      detailArea = area;
      detailFace = face;
    }
    // f.a ≠ 0 gives a face whose first edge follows the length of the strip.
    if (divisions > 1 && Math.hypot(b[0] - a[0], b[1] - a[1]) >= divisions * 2.25) {
      ctx.beginPath();
      for (let j = 1; j < divisions; j++) {
        const u = j / divisions;
        ctx.moveTo(blend(a[0], b[0], u), blend(a[1], b[1], u));
        ctx.lineTo(blend(d[0], face[2][0], u), blend(d[1], face[2][1], u));
      }
      ctx.strokeStyle = ink.seam;
      ctx.lineWidth = 0.72;
      ctx.stroke();
    }
  }
  if (detailFace && item.readIndex != null && detailArea > 18) {
    const [a, b, c, d] = detailFace;
    const low = item.readIndex / divisions,
      high = (item.readIndex + 1) / divisions;
    ctx.beginPath();
    ctx.moveTo(...mix(a, b, low));
    ctx.lineTo(...mix(a, b, high));
    ctx.lineTo(...mix(d, c, high));
    ctx.lineTo(...mix(d, c, low));
    ctx.closePath();
    ctx.fillStyle = 'rgba(255,240,148,0.78)';
    ctx.fill();
    ctx.strokeStyle = '#a27c31';
    ctx.lineWidth = 0.9;
    ctx.stroke();
  }
  if (detailFace && values && detailArea > 36) {
    const [a, b, c, d] = detailFace;
    const edge = Math.hypot(b[0] - a[0], b[1] - a[1]) / divisions;
    const across = Math.hypot(d[0] - a[0], d[1] - a[1]);
    const radius = Math.min(edge, across) * 0.23;
    if (radius > 0.7)
      for (let j = 0; j < values.length; j++) {
        const v = values[j];
        if (typeof v !== 'number' || !Number.isFinite(v)) continue;
        const u = (j + 0.5) / values.length;
        const x = (blend(a[0], b[0], u) + blend(d[0], c[0], u)) / 2,
          y = (blend(a[1], b[1], u) + blend(d[1], c[1], u)) / 2;
        ctx.fillStyle = v < 0 ? ink.rim : ink.top;
        ctx.beginPath();
        ctx.arc(
          x,
          y,
          Math.min(2.4, radius) * (0.5 + 0.5 * Math.sqrt(Math.abs(v) / (0.1 + Math.abs(v)))),
          0,
          Math.PI * 2,
        );
        ctx.fill();
      }
  }
  ctx.restore();
}

// Rotate a lifted strip from its source axis toward the receiving row. This
// joint rotates as a rigid frame; a dot-product strip contracts along its length.
function frame(from, to, t) {
  const direction = normalized(mix(axes[from], axes[to], smooth(t)));
  const guide = from !== to ? axes[3 - from - to] : axes[(from + 2) % 3];
  return [direction, normalized(cross(guide, direction)), guide];
}
function valuesFor(s, start, length, sample) {
  if (!sample?.value) return null;
  return Array.from({ length }, (_, i) => sample.value(s.node, ...indexAt(s, start + i)));
}
function queueItem(queue, point, vectors, widths, ink, divisions, alpha, values) {
  queue.push({ point, vectors, widths, ink, divisions, alpha, values });
}
function stageGap(from, to, unit, view) {
  const distance = Math.hypot(...to.map((v, i) => v - from[i]));
  const screenLimit = (Math.min(view.w, view.h) * 0.16) / view.scale;
  return Math.min(Math.max(unit * 3.2, distance * 0.19), screenLimit);
}
function writePoint(out, offset, direction, gap) {
  return add(at(out, offset), direction, gap);
}
function guide(ctx, view, a, b, dirA, dirB, lift, material, t) {
  const lo = Math.max(0, t - 0.19),
    steps = 5;
  const points = Array.from({ length: steps + 1 }, (_, i) =>
    view.project(curve(a, b, dirA, dirB, lift, blend(lo, t, i / steps))),
  );
  ctx.save();
  ctx.globalAlpha *= 0.42;
  ctx.strokeStyle = material.seam;
  ctx.lineWidth = 1.15;
  ctx.beginPath();
  for (let i = 1; i < points.length; i++) {
    const a = points[i - 1],
      b = points[i],
      segment = lineInterval(a, b, view, 4);
    if (!segment) continue;
    ctx.moveTo(...mix(a, b, segment[0]));
    ctx.lineTo(...mix(a, b, segment[1]));
  }
  ctx.stroke();
  ctx.restore();
}

// Lift the currently addressed contiguous band as a gridded daughter piece.
// Small registration lines show exactly where both ends left the parent row.
function liftedRead(ctx, view, s, head, lift, queue, sample) {
  if (lift <= 0.005) return;
  const direction = s.axis < 0 ? 0 : s.axis;
  const cellSize = Math.max(0.000001, s.unit[direction] * view.scale);
  const lengthLimit = Math.min(112, view.w * 0.32, view.h * 0.4);
  const capacity = Math.min(12, Math.max(1, Math.floor(lengthLimit / Math.max(cellSize, 10))));
  const length = Math.min(s.length, capacity);
  const start = clamp(head - Math.floor((length - 1) / 2), 0, s.length - length);
  const center = at(s, start + (length - 1) / 2),
    normal = liftAxis(s);
  const height = (lift * clamp(Math.min(...s.unit) * view.scale * 6.7, 23, 50)) / view.scale;
  const point = add(center, normal, height);
  const ink = materials[s.node.kind],
    actualWidth = Math.min(...s.unit);
  const cell = blend(
    s.unit[direction],
    clamp(cellSize, s.length === 1 ? 15.5 : 9.8, 22) / view.scale,
    lift,
  );
  const width = blend(
    actualWidth,
    clamp(actualWidth * view.scale, s.length === 1 ? 14 : 10.8, 18) / view.scale,
    lift,
  );
  const widths = [cell * length, width, width * 0.82];
  if (!inFrame(view.project(point), view, Math.max(...widths) * view.scale + 4)) return;
  ctx.save();
  ctx.strokeStyle = ink.rim;
  ctx.lineWidth = 1.05;
  ctx.globalAlpha *= lift * 0.85;
  ctx.beginPath();
  for (const sign of [-1, 1]) {
    const base = add(center, axes[direction], 0.5 * sign * s.unit[direction] * length);
    const edge = add(point, axes[direction], 0.5 * sign * cell * length);
    ctx.moveTo(...view.project(base));
    ctx.lineTo(...view.project(edge));
  }
  ctx.stroke();
  ctx.restore();
  queue.push({
    point,
    vectors: frame(direction, direction, 0),
    widths,
    ink,
    divisions: length,
    alpha: Math.min(1, lift * 3.3),
    values: valuesFor(s, start, length, sample),
    readIndex: head - start,
  });
}

/** Number of addressed cells in a read-in-place phase, including its head. */
export function readCount(progress, length, lane = 0, total = 1) {
  if (!Number.isFinite(progress) || !Number.isSafeInteger(length) || length < 1) return 0;
  const delay = total > 1 ? (0.105 * lane) / (total - 1) : 0;
  const scan = smooth(windowTime(progress, 0.07 + delay, 0.81 + delay));
  return 1 + Math.min(length - 1, Math.floor(scan * length));
}
function readInPlace(ctx, view, s, p, lane, total, queue, sample) {
  const delay = total > 1 ? (0.105 * lane) / (total - 1) : 0;
  const count = readCount(p, s.length, lane, total),
    head = count - 1,
    ink = materials[s.node.kind];
  band(ctx, s, 0, s.length, view, ink.rim, 0.78);
  band(ctx, s, 0, count, view, ink.front, 0.97);
  band(ctx, s, head, 1, view, ink.top, 1);
  const lift =
    smooth(windowTime(p, 0.02 + delay, 0.2 + delay)) * (1 - smooth(windowTime(p, 0.78 + delay, 1)));
  liftedRead(ctx, view, s, head, lift, queue, sample);
}
function sameSpan(a, b) {
  return (
    a.node.id === b.node.id &&
    a.length === b.length &&
    (a.length === 1 || a.axis === b.axis) &&
    a.indices.every((v, i) => v === b.indices[i])
  );
}

function isReadPhase(out, inputs) {
  return !inputs.length || out.node.kind === 'w' || inputs.some((s) => sameSpan(s, out));
}
function transferDelay(length, lane, total, allScalars) {
  if (allScalars && total > 1) return (lane / (total - 1)) * 0.32;
  return length === 1 && total > 1 ? 0.36 : Math.min(lane * 0.012, 0.055);
}
function transferCount(progress, length, delay) {
  const scan = windowTime(progress, 0.018 + delay, 0.44 + delay);
  return 1 + Math.min(length - 1, Math.floor(scan * length));
}

/** First source's actual read-head count for the stage's running equation. */
export function phaseReadCount(phase, progress) {
  if (!Number.isSafeInteger(phase?.readoutLength) || phase.readoutLength < 1 || !Number.isFinite(progress))
    return 0;
  const out = validSpan(phase.output);
  if (!out) return 0;
  const inputs = inputsOf(phase.sources),
    source = inputs[0] ?? out;
  const allScalars = inputs.every((s) => s.length === 1);
  const count = isReadPhase(out, inputs)
    ? readCount(progress, source.length, 0, Math.max(1, inputs.length))
    : transferCount(progress, source.length, transferDelay(source.length, 0, inputs.length, allScalars));
  return Math.min(phase.readoutLength, count);
}

function transportGuide(ctx, view, a, b, normal, receiving, lift, ink, alpha) {
  if (alpha < 0.025) return;
  const begin = view.project(a),
    end = view.project(b),
    margin = Math.max(view.w, view.h) * 0.8;
  if (!inFrame(begin, view, margin) || !inFrame(end, view, margin)) return;
  ctx.save();
  ctx.strokeStyle = ink.side;
  ctx.globalAlpha *= 0.28 * alpha;
  ctx.lineWidth = 3.1;
  ctx.beginPath();
  ctx.moveTo(...begin);
  ctx.bezierCurveTo(
    ...view.project(add(a, normal, lift)),
    ...view.project(add(b, receiving, lift)),
    ...end,
  );
  ctx.stroke();
  ctx.restore();
}
function readLane(ctx, view, s, out, lane, total, p, receiving, gap, queue, sample, allScalars) {
  const material = materials[s.node.kind];
  const totalGroups = Math.min(s.length, total > 3 ? 5 : 8);
  const delay = transferDelay(s.length, lane, total, allScalars);
  const count = transferCount(p, s.length, delay),
    head = count - 1;
  const peel =
    smooth(windowTime(p, 0.015 + delay, 0.11 + delay)) *
    (1 - smooth(windowTime(p, 0.4 + delay, 0.56 + delay)));
  if (s.length > 1) liftedRead(ctx, view, s, head, peel, queue, sample);
  const decay = 1 - smooth(windowTime(p, 0.66, 0.95)) * 0.6;
  band(ctx, s, 0, s.length, view, material.rim, 0.48 * decay);
  band(ctx, s, 0, count, view, material.front, 0.93 * decay);
  if (p < 0.65) band(ctx, s, head, 1, view, material.top, 1);
  const from = s.axis < 0 ? (out.axis < 0 ? 0 : out.axis) : s.axis;
  const to = out.axis < 0 ? from : out.axis;
  const normal = liftAxis(s);
  const start = at(s, (s.length - 1) / 2),
    end = writePoint(out, (out.length - 1) / 2, receiving, gap);
  const guiding = smooth(windowTime(p, 0.04, 0.15)) * (1 - smooth(windowTime(p, 0.67, 0.84)));
  transportGuide(
    ctx,
    view,
    start,
    end,
    normal,
    receiving,
    stageGap(start, end, Math.min(...s.unit), view),
    material,
    guiding,
  );
  for (let j = 0; j < totalGroups; j++) {
    const position = totalGroups === 1 ? 0 : j / (totalGroups - 1);
    const departure = 0.06 + position * 0.4 + delay;
    const t = windowTime(p, departure, departure + (allScalars && total === 1 ? 0.54 : 0.26));
    if (t <= 0 || t >= 1) continue;
    const seg = segment(s, j, totalGroups);
    const a = at(s, seg.middle);
    const destOffset = out.length > 1 ? position * (out.length - 1) : 0;
    const b = writePoint(out, destOffset, receiving, gap);
    const lift = stageGap(a, b, Math.min(...s.unit), view);
    const point = curve(a, b, normal, receiving, lift, smooth(t));
    const inkFade = 1 - smooth(windowTime(t, 0.82, 1));
    const magnify = smooth(windowTime(t, 0, 0.22)) * (1 - smooth(windowTime(t, 0.86, 1)));
    const actualCell = s.unit[from];
    const readable = clamp(actualCell * view.scale, 8.5, 14) / view.scale;
    const size = blend(actualCell, readable, magnify);
    const narrow = Math.min(...s.unit);
    const width = blend(narrow, clamp(narrow * view.scale, 7.0, 12) / view.scale, magnify);
    const contraction = out.length === 1 ? 1 - 0.77 * smooth(windowTime(t, 0.45, 1)) : 1;
    const widths = [size * seg.length * contraction, width, width * blend(1, 0.63, magnify)];
    if (!inFrame(view.project(point), view, Math.max(...widths) * view.scale + 5)) continue;
    guide(ctx, view, a, b, normal, receiving, lift, material, smooth(t));
    if (t < 0.26) {
      const origin = view.project(a),
        lifted = view.project(point);
      const distance = Math.hypot(lifted[0] - origin[0], lifted[1] - origin[1]);
      if (distance > 1.2 && distance < Math.min(view.w, view.h) * 0.32) {
        ctx.save();
        ctx.globalAlpha *= 0.48 * (1 - t / 0.26);
        ctx.strokeStyle = material.rim;
        ctx.lineWidth = 0.85;
        ctx.beginPath();
        ctx.moveTo(...origin);
        ctx.lineTo(...lifted);
        ctx.stroke();
        ctx.restore();
      }
    }
    queueItem(
      queue,
      point,
      frame(from, to, windowTime(t, 0.2, 0.78)),
      widths,
      material,
      seg.length,
      inkFade,
      valuesFor(s, seg.start, seg.length, sample),
    );
  }
}

function writeResult(ctx, view, out, p, receiving, gap, queue, sample) {
  const material = materials[out.node.kind],
    direction = out.axis < 0 ? 0 : out.axis;
  // Hold the accumulated result directly above its true destination. Output
  // cells start settling only after the incoming strips have finished reading.
  const groups = Math.min(out.length, 10);
  const make = windowTime(p, 0.35, 0.77);
  const written = windowTime(p, 0.77, 0.985);
  const filled = Math.min(out.length, Math.floor(written * out.length));
  for (let j = 0; j < groups; j++) {
    const u = groups === 1 ? 0 : j / (groups - 1);
    const seg = segment(out, j, groups);
    const landed = 0.77 + (0.215 * (seg.start + seg.length)) / out.length;
    const drop = windowTime(p, landed - 0.105, landed);
    if (make <= u * 0.6 || drop >= 1) continue;
    const pos = at(out, seg.middle);
    const point = add(pos, receiving, gap * (1 - smooth(drop)));
    const actual = out.unit[direction],
      narrow = Math.min(...out.unit),
      elevated = 1 - smooth(drop);
    const width = blend(narrow, clamp(narrow * view.scale, 7.3, 11.5) / view.scale, elevated);
    const cell = blend(actual, clamp(actual * view.scale, 8.4, 14) / view.scale, elevated);
    const alpha = smooth((make - u * 0.6) / 0.24) * blend(0.69, 1, drop);
    const size = [cell * seg.length, width, width * blend(1, 0.68, elevated)];
    if (inFrame(view.project(point), view, Math.max(...size) * view.scale + 2)) {
      queueItem(
        queue,
        point,
        frame(direction, direction, 0),
        size,
        material,
        seg.length,
        alpha,
        valuesFor(out, seg.start, seg.length, make >= 0.98 || drop > 0 ? sample : null),
      );
    }
  }
  band(ctx, out, 0, out.length, view, material.rim, 0.7);
  band(ctx, out, 0, filled, view, material.front, 1);
  if (filled > 0 && filled < out.length) band(ctx, out, filled - 1, 1, view, '#f0d57e', 1);
  if (p > 0.925) band(ctx, out, 0, out.length, view, material.top, smooth(windowTime(p, 0.925, 1)) * 0.7);
}

/**
 * Paint one frame of a calculation on the existing tensor canvas. Progress is
 * normalized to this phase. sample.value(node,x,y,z) is optional and read only
 * for drawn packet cells. Each strip keeps real source indices at either end.
 */
export function paintComputation(ctx, view, phase, progress, { sample } = {}) {
  const out = validSpan(phase?.output);
  if (
    !out ||
    !view ||
    !Number.isFinite(view.scale) ||
    !(view.scale > 0) ||
    !(view.w > 0) ||
    !(view.h > 0) ||
    !Number.isFinite(progress)
  )
    return;
  const p = clamp(progress, 0, 1),
    inputs = inputsOf(phase.sources),
    receiving = liftAxis(out);
  const dest = at(out, (out.length - 1) / 2);
  const gap =
    Math.max(
      22 / view.scale,
      ...inputs.map((s) => stageGap(at(s, (s.length - 1) / 2), dest, Math.min(...out.unit), view)),
    ) * 0.88;
  const queue = [];
  ctx.save();
  ctx.beginPath();
  ctx.rect(0, 0, view.w, view.h);
  ctx.clip();
  // Read-only frames may use their source as the phase's landing location:
  // a checkpoint coefficient, a lookup or a temporary reduction on a row.
  if (isReadPhase(out, inputs)) {
    const reads = inputs.length ? inputs : [out];
    reads.forEach((s, i) => readInPlace(ctx, view, s, p, i, reads.length, queue, sample));
    if (!reads.some((s) => sameSpan(s, out))) readInPlace(ctx, view, out, p, 0, 1, queue, sample);
  } else {
    const allScalars = inputs.every((s) => s.length === 1);
    inputs.forEach((s, i) =>
      readLane(ctx, view, s, out, i, inputs.length, p, receiving, gap, queue, sample, allScalars),
    );
    writeResult(ctx, view, out, p, receiving, gap, queue, sample);
  }
  queue.sort((a, b) => dot(a.point, view.eye) - dot(b.point, view.eye));
  for (const item of queue) prism(ctx, view, item);
  ctx.restore();
}
