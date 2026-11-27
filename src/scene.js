import { buildArchitecture, inks } from './geometry.js';

const background = [17, 20, 37];
const clamp = (v, low, high) => Math.max(low, Math.min(high, v));
const mix = (a, b, t) => a.map((v, i) => Math.round(v + (b[i] - v) * t));
const color = (v, alpha = 1) => 'rgba(' + v.join(',') + ',' + alpha + ')';
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

const surfaces = [
  { axis: 2, sign: 1, face: 'front', corners: [7, 6, 5, 4], cells: [0, 1], light: 0.75 },
  { axis: 2, sign: -1, face: 'back', corners: [2, 3, 0, 1], cells: [0, 1], light: 0.6 },
  { axis: 0, sign: -1, face: 'left', corners: [3, 7, 4, 0], cells: [2, 1], light: 0.54 },
  { axis: 0, sign: 1, face: 'right', corners: [6, 2, 1, 5], cells: [2, 1], light: 0.62 },
  { axis: 1, sign: 1, face: 'top', corners: [3, 2, 6, 7], cells: [0, 2], light: 0.9 },
];

function corners(part) {
  const [x0, y0, z0] = part.min;
  const [x1, y1, z1] = part.max;
  return [
    [x0, y0, z0],
    [x1, y0, z0],
    [x1, y1, z0],
    [x0, y1, z0],
    [x0, y0, z1],
    [x1, y0, z1],
    [x1, y1, z1],
    [x0, y1, z1],
  ];
}

function camera(yaw, pitch) {
  const sin = Math.sin(yaw),
    cos = Math.cos(yaw);
  const up = Math.sin(pitch),
    across = Math.cos(pitch);
  return {
    right: [cos, 0, -sin],
    down: [sin * up, -across, cos * up],
    eye: [sin * across, up, cos * across],
  };
}

function bounds(points) {
  let left = Infinity,
    right = -Infinity,
    top = Infinity,
    bottom = -Infinity;
  for (const [px, py] of points) {
    left = Math.min(left, px);
    right = Math.max(right, px);
    top = Math.min(top, py);
    bottom = Math.max(bottom, py);
  }
  return { left, right, top, bottom };
}

function tracePolygon(ctx, vertices) {
  ctx.beginPath();
  ctx.moveTo(...vertices[0]);
  for (let i = 1; i < vertices.length; i++) ctx.lineTo(...vertices[i]);
  ctx.closePath();
}

function contains(vertices, x, y) {
  let positive = false,
    negative = false;
  for (let i = 0; i < vertices.length; i++) {
    const a = vertices[i],
      b = vertices[(i + 1) % vertices.length];
    const side = (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0]);
    positive ||= side > 0.1;
    negative ||= side < -0.1;
    if (positive && negative) return false;
  }
  return true;
}

function overlaps(a, b, margin = 0) {
  return (
    a.left < b.right + margin &&
    a.right > b.left - margin &&
    a.top < b.bottom + margin &&
    a.bottom > b.top - margin
  );
}

// Order opaque, axis-aligned tensors by a separating world-space plane.
// Sorting centers alone can put a narrow foreground head behind a larger matrix.
function depthOrder(items, eye) {
  const remaining = items.map((item, index) => ({ item, index, before: 0, after: [] }));
  for (let i = 0; i < remaining.length; i++) {
    const a = remaining[i];
    for (let j = i + 1; j < remaining.length; j++) {
      const b = remaining[j];
      if (!overlaps(a.item.bounds, b.item.bounds)) continue;
      let order = 0,
        conflict = false;
      for (let k = 0; k < 3; k++) {
        if (Math.abs(eye[k]) < 0.0001) continue;
        let sign = 0;
        if (a.item.part.max[k] <= b.item.part.min[k] + 0.0001) sign = Math.sign(eye[k]);
        else if (b.item.part.max[k] <= a.item.part.min[k] + 0.0001) sign = -Math.sign(eye[k]);
        if (sign && order && sign !== order) conflict = true;
        if (sign) order = sign;
      }
      if (conflict || !order) continue;
      const [far, near] = order > 0 ? [a, b] : [b, a];
      far.after.push(near);
      near.before++;
    }
  }
  const waiting = new Set(remaining);
  const sorted = [];
  while (waiting.size) {
    let next, fallback;
    for (const node of waiting) {
      if (!fallback || node.item.depth < fallback.item.depth) fallback = node;
      if (!node.before && (!next || node.item.depth < next.item.depth)) next = node;
    }
    next ??= fallback;
    sorted.push(next.item);
    waiting.delete(next);
    for (const child of next.after) child.before--;
  }
  return sorted;
}

function tileTexture(part, face, textureCache) {
  const columns = part.grid[face.cells[0]],
    rows = part.grid[face.cells[1]];
  const key = [part.id, part.stage, columns, rows, face.face, part.finish].join('/');
  if (textureCache.has(key)) return textureCache.get(key);
  const pixel = 12;
  const texture = document.createElement('canvas');
  texture.width = columns * pixel;
  texture.height = rows * pixel;
  const g = texture.getContext('2d');
  const ink = inks[part.swatch ?? part.stage];
  const seam = mix(background, ink, face.light * 0.41);
  g.fillStyle = color(seam);
  g.fillRect(0, 0, texture.width, texture.height);

  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < columns; col++) {
      const noise = Math.sin(col * 31.17 + row * 19.71 + part.seed * 8.23) * 1657.367;
      const variance = noise - Math.floor(noise);
      const motif = (2 * col + 3 * row + part.seed) % 11 === 0;
      const masked =
        part.finish === 'causal' &&
        face.cells[1] === 1 &&
        face.axis === 2 &&
        (face.sign === 1 ? col > row : columns - 1 - col > row);
      let shade = face.light * (0.72 + 0.25 * variance + (motif ? 0.14 : 0));
      if (part.finish === 'state' && (row + col + part.seed) % 5 < 2) shade *= 1.1;
      if (part.finish === 'ribbon') shade *= 0.89;
      if (masked) shade = 0.145 + 0.055 * variance;

      const px = col * pixel,
        py = row * pixel;
      const cell = mix(background, ink, Math.min(1, shade));
      g.fillStyle = color(cell);
      g.fillRect(px + 1, py + 1, pixel - 1.25, pixel - 1.25);
      if (!masked) {
        g.fillStyle = color(mix(cell, [250, 247, 255], 0.14));
        g.fillRect(px + 1, py + 1, pixel - 1.25, 0.8);
      }
    }
  }
  textureCache.set(key, texture);
  return texture;
}

function drawFace(ctx, part, face, screen, cache, active, dim, labels) {
  const quad = face.corners.map((i) => screen[i]);
  const [a, b, , d] = quad;
  const ink = inks[part.swatch ?? part.stage];
  tracePolygon(ctx, quad);
  if (part.finish === 'board' || part.finish === 'joint') {
    const level = part.finish === 'joint' ? 0.64 : face.axis === 1 ? 0.065 : 0.15;
    ctx.fillStyle = color(mix(background, ink, level));
    ctx.fill();
    ctx.lineWidth = part.finish === 'board' ? 0.7 : 0.8;
    ctx.strokeStyle = color(ink, part.finish === 'board' ? 0.34 : 0.7);
    ctx.stroke();
    if (face.axis === 1 && part.finish === 'board') boardDetails(ctx, a, b, d, ink);
    return quad;
  }
  const texture = tileTexture(part, face, cache);
  ctx.save();
  ctx.transform(
    (b[0] - a[0]) / texture.width,
    (b[1] - a[1]) / texture.width,
    (d[0] - a[0]) / texture.height,
    (d[1] - a[1]) / texture.height,
    a[0],
    a[1],
  );
  ctx.drawImage(texture, 0, 0);
  ctx.restore();

  tracePolygon(ctx, quad);
  if (dim || active) {
    ctx.fillStyle = active ? 'rgba(235,234,255,0.055)' : 'rgba(15,18,35,0.11)';
    ctx.fill();
  }
  ctx.strokeStyle = color(
    mix(ink, [255, 250, 255], active ? 0.65 : 0.3),
    active ? 0.97 : dim ? 0.53 : 0.75,
  );
  ctx.lineWidth = active ? 1.05 : 0.72;
  ctx.stroke();

  if (labels && part.print && face.face === 'front') {
    const w = part.size[0],
      h = part.size[1];
    const tagHeight = Math.min(0.34, h * 0.47, w * 0.55);
    const tagWidth = Math.min(w * 0.9, tagHeight * (0.67 + part.print.length * 0.51));
    if ((Math.hypot(d[0] - a[0], d[1] - a[1]) * tagHeight) / h > 6) {
      ctx.save();
      // Use model-space dimensions for engraving so the type follows the plane.
      ctx.transform((b[0] - a[0]) / w, (b[1] - a[1]) / w, (d[0] - a[0]) / h, (d[1] - a[1]) / h, a[0], a[1]);
      const middleY = h * (part.printY ?? 0.5);
      const tagX = (w - tagWidth) / 2,
        tagY = middleY - tagHeight / 2;
      ctx.fillStyle = color(mix(background, ink, 0.13));
      ctx.fillRect(tagX, tagY, tagWidth, tagHeight);
      ctx.fillStyle = color(mix(ink, [255, 255, 255], 0.51));
      ctx.font = '600 ' + tagHeight * 0.74 + 'px ui-monospace, SFMono-Regular, monospace';
      ctx.textBaseline = 'middle';
      ctx.textAlign = 'center';
      ctx.fillText(part.print, w / 2, middleY + tagHeight * 0.04, tagWidth * 0.92);
      ctx.restore();
    }
  }
  return quad;
}

function boardDetails(ctx, a, b, d, ink) {
  const at = (u, v) => [
    a[0] + u * (b[0] - a[0]) + v * (d[0] - a[0]),
    a[1] + u * (b[1] - a[1]) + v * (d[1] - a[1]),
  ];
  ctx.strokeStyle = color(ink, 0.26);
  ctx.lineWidth = 0.64;
  for (const v of [0.12, 0.88]) {
    ctx.beginPath();
    ctx.moveTo(...at(0.055, v));
    ctx.lineTo(...at(0.945, v));
    ctx.stroke();
  }
  for (const u of [0.06, 0.94]) {
    for (const v of [0.12, 0.88]) {
      const p = at(u, v);
      ctx.fillStyle = color(ink, 0.62);
      ctx.fillRect(p[0] - 0.8, p[1] - 0.8, 1.6, 1.6);
    }
  }
}

function drawTrace(ctx, trace, project, selected, annotate) {
  const points = trace.points.map(project);
  const ink = inks[trace.stage],
    standard = trace.stage === 'bus';
  ctx.beginPath();
  ctx.moveTo(...points[0]);
  for (const p of points.slice(1)) ctx.lineTo(...p);
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  ctx.lineWidth = trace.feedback ? 2.35 : trace.skip ? 1.7 : standard ? 2 : 1.6;
  ctx.strokeStyle = color(background, 0.98);
  ctx.stroke();
  ctx.lineWidth = trace.feedback ? 1.45 : trace.skip ? 1.0 : standard ? 1.2 : 1.06;
  ctx.strokeStyle = color(ink, trace.stage === selected ? 0.94 : trace.skip ? 0.56 : 0.74);
  ctx.stroke();
  if (trace.skip || trace.feedback) {
    const index = trace.feedback ? 3 : 2;
    const a = points[index],
      b = points[index + 1];
    const dx = b[0] - a[0],
      dy = b[1] - a[1],
      length = Math.hypot(dx, dy);
    if (length < 10) return;
    const arrowAt = trace.annotate ? 0.79 : 0.5;
    const x = a[0] + dx * arrowAt,
      y = a[1] + dy * arrowAt;
    const ux = dx / length,
      uy = dy / length;
    ctx.beginPath();
    ctx.moveTo(x + ux * 3.2, y + uy * 3.2);
    ctx.lineTo(x - ux * 2.3 - uy * 1.8, y - uy * 2.3 + ux * 1.8);
    ctx.lineTo(x - ux * 2.3 + uy * 1.8, y - uy * 2.3 - ux * 1.8);
    ctx.closePath();
    ctx.fillStyle = color(ink, 0.9);
    ctx.fill();
    if (trace.annotate && annotate && length > 86) {
      const angle = Math.atan2(dy, dx);
      const readable = Math.abs(angle) > Math.PI / 2 ? angle + Math.PI : angle;
      ctx.save();
      ctx.translate(a[0] + dx * 0.54, a[1] + dy * 0.54);
      ctx.rotate(readable);
      ctx.font = '8.5px ui-monospace, SFMono-Regular, Menlo, monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      const size = ctx.measureText('residual').width + 7;
      ctx.fillStyle = color(background, 0.96);
      ctx.fillRect(-size / 2, -4.7, size, 9.4);
      ctx.fillStyle = color(ink, 0.9);
      ctx.fillText('residual', 0, 0.4);
      ctx.restore();
    }
  }
}

function stationLabels(ctx, stations, project, width, height, chosen, hovered, solids) {
  const small = width < 475;
  const font = small ? 9 : 10.5;
  const gap = small ? 10 : 15;
  ctx.font = '500 ' + font + 'px ui-monospace, SFMono-Regular, Menlo, monospace';
  const occupied = [];
  for (const station of stations) {
    const [ax, ay] = project(station.anchor);
    if (ax < -10 || ax > width + 10 || ay < -6 || ay > height + 6) continue;
    const title = small ? station.short : station.title;
    const w = Math.ceil(ctx.measureText(title).width) + 16;
    const h = font + 8;
    const side = station.side;
    let best,
      bestCost = Infinity;
    const solidList = solids.filter((p) => p.part.finish !== 'board' && p.part.finish !== 'joint');
    for (const sign of [side, -side]) {
      for (const distance of [gap, gap + 14, gap + 31]) {
        for (const offset of [0, -w * 0.34, w * 0.34, -w * 0.65, w * 0.65]) {
          const left = clamp(ax - w / 2 + offset, 3, Math.max(3, width - w - 3));
          const top = clamp(ay + sign * distance - (sign < 0 ? h : 0), 3, Math.max(3, height - h - 3));
          const rect = { left, right: left + w, top, bottom: top + h, stage: station.stage };
          const overlapLabels = occupied.filter((prev) => overlaps(prev, rect, 3)).length;
          const overlapParts = solidList.filter((p) => overlaps(p.bounds, rect, -1)).length;
          const cost =
            overlapLabels * 1000 +
            overlapParts * 53 +
            Math.abs(offset) * 0.7 +
            distance +
            (sign === side ? 0 : 20);
          if (cost < bestCost) {
            bestCost = cost;
            best = rect;
          }
        }
      }
    }
    const rect = best;
    occupied.push(rect);
    const ink = inks[station.stage];
    const targetX = clamp(ax, rect.left + 5, rect.right - 5);
    const targetY = ay > rect.bottom ? rect.bottom : rect.top;
    const elbowY = targetY + (ay - targetY) * 0.26;
    ctx.strokeStyle = color(ink, hovered === station.stage || chosen === station.stage ? 0.88 : 0.57);
    ctx.lineWidth = 0.81;
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(ax, elbowY);
    ctx.lineTo(targetX, targetY);
    ctx.stroke();
    ctx.fillStyle = color(ink, 0.94);
    ctx.beginPath();
    ctx.arc(ax, ay, 1.7, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = 'rgba(20,23,42,0.93)';
    ctx.beginPath();
    ctx.roundRect(rect.left, rect.top, w, h, 3);
    ctx.fill();
    const prominent = hovered === station.stage || chosen === station.stage;
    ctx.strokeStyle = color(ink, prominent ? 0.7 : 0.3);
    ctx.lineWidth = 0.7;
    ctx.stroke();
    ctx.fillStyle = color(mix(ink, [250, 250, 255], 0.26));
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(title, (rect.left + rect.right) / 2, (rect.top + rect.bottom) / 2 + 0.4);
  }
  return occupied;
}

function orientation(ctx, view, height) {
  const cx = 27,
    cy = height - 25;
  const axes = [
    { axis: 0, letter: 'X', ink: [204, 148, 153] },
    { axis: 1, letter: 'Y', ink: [141, 197, 176] },
    { axis: 2, letter: 'Z', ink: [145, 164, 215] },
  ].sort((a, b) => view.eye[a.axis] - view.eye[b.axis]);
  ctx.font = '8px ui-monospace, SFMono-Regular, Menlo, monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  for (const { axis, letter, ink } of axes) {
    const dx = view.right[axis],
      dy = view.down[axis];
    const tx = cx + dx * 14,
      ty = cy + dy * 14;
    ctx.strokeStyle = color(ink, 0.8);
    ctx.lineWidth = 1.15;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(tx, ty);
    ctx.stroke();
    ctx.fillStyle = color(ink, 0.94);
    ctx.fillText(letter, cx + dx * 20, cy + dy * 20);
  }
  ctx.fillStyle = 'rgb(194,197,215)';
  ctx.beginPath();
  ctx.arc(cx, cy, 1.65, 0, Math.PI * 2);
  ctx.fill();
}

/**
 * Attach a dependency-free, orthographic 3D tensor viewer. Orbit deltas use CSS
 * pixels, zoom factors are multiplicative. Geometry is only drawn on changes.
 */
export function mountScene(canvas, { onSelect, onZoom, onHover } = {}) {
  const ctx = canvas.getContext('2d');
  const controller = new AbortController();
  const signal = controller.signal;
  let model = 'nanogpt',
    selected = null,
    hovered = null;
  let labels = true,
    spread = 1,
    targetSpread = 1,
    geometry;
  let width = 0,
    height = 0,
    ratio = 1,
    frame = 0,
    previousTime = 0;
  let yaw,
    pitch,
    zoom = 1,
    panX = 0,
    panY = 0,
    customCamera = false;
  let faceHits = [],
    labelHits = [];
  const textures = new Map(),
    pointers = new Map();
  let pinchDistance = 0,
    pinchMid = null;
  const still = window.matchMedia('(prefers-reduced-motion: reduce)');
  canvas.style.touchAction = 'none';
  canvas.style.cursor = 'grab';
  if (!canvas.hasAttribute('tabindex')) canvas.tabIndex = 0;

  function rebuild() {
    geometry = buildArchitecture(model, spread);
    for (const part of geometry.parts) part.corners = corners(part);
    geometry.vertices = geometry.parts
      .flatMap((p) => p.corners)
      .concat(geometry.traces.flatMap((t) => t.points));
  }
  function home() {
    const narrow = width / Math.max(height, 1) < 1.38;
    return { yaw: narrow ? -0.91 : -0.56, pitch: narrow ? 0.66 : 0.53 };
  }
  function reset() {
    ({ yaw, pitch } = home());
    customCamera = false;
    zoom = 1;
    panX = 0;
    panY = 0;
    onZoom?.(zoom);
    requestDraw();
  }
  function fit(view) {
    const b = bounds(geometry.vertices.map((p) => [dot(p, view.right), dot(p, view.down)]));
    const gutterX = Math.min(52, Math.max(13, width * 0.065));
    const gutterY = Math.min(67, Math.max(25, height * 0.14));
    const scale = Math.min(
      (width - 2 * gutterX) / Math.max(1, b.right - b.left),
      (height - 2 * gutterY) / Math.max(1, b.bottom - b.top),
    );
    return { b, scale: Math.max(1, scale) };
  }
  function render(time) {
    frame = 0;
    if (!ctx || !width || !height) return;
    if (spread !== targetSpread) {
      const dt = Math.min(0.06, Math.max(0.008, (time - previousTime) / 1000));
      spread += (targetSpread - spread) * (1 - Math.exp(-13 * dt));
      if (still.matches || Math.abs(spread - targetSpread) < 0.001) spread = targetSpread;
      rebuild();
    }
    previousTime = time;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);
    ctx.imageSmoothingQuality = 'high';
    const view = camera(yaw, pitch);
    const { b, scale: fitted } = fit(view);
    const defaultAngles = home();
    const baseFit = customCamera ? fit(camera(defaultAngles.yaw, defaultAngles.pitch)).scale : fitted;
    const scale = Math.min(fitted, baseFit * 1.13) * zoom;
    const ox = width / 2 - ((b.left + b.right) / 2) * scale + panX;
    const oy = height * 0.52 - ((b.top + b.bottom) / 2) * scale + panY;
    const project = (p) => [ox + dot(p, view.right) * scale, oy + dot(p, view.down) * scale];
    const visibleFaces = surfaces.filter((face) => view.eye[face.axis] * face.sign > 0.005);
    const prepared = geometry.parts.map((part) => {
      const screen = part.corners.map(project);
      return { part, screen, depth: dot(part.center, view.eye), bounds: bounds(screen) };
    });

    // Soft contact light stays on the ground beneath each module.
    for (const { part } of prepared.filter((item) => item.part.finish === 'board')) {
      const [cx, cy] = project([part.center[0], -0.25, part.center[2]]);
      const radius = Math.max(1, part.size[0] * scale * 0.75);
      ctx.save();
      ctx.translate(cx, cy + 2);
      ctx.scale(1, 0.43);
      const halo = ctx.createRadialGradient(0, 0, 0, 0, 0, radius);
      halo.addColorStop(0, color(inks[part.stage], 0.105));
      halo.addColorStop(1, color(inks[part.stage], 0));
      ctx.fillStyle = halo;
      ctx.fillRect(-radius, -radius, radius * 2, radius * 2);
      ctx.restore();
    }
    faceHits = [];
    function paint(item) {
      const { part } = item;
      const active = part.stage !== 'bus' && (part.stage === hovered || part.stage === selected);
      const dim = !active && !!(selected || hovered) && part.finish !== 'board';
      for (const face of visibleFaces) {
        const quad = drawFace(ctx, part, face, item.screen, textures, active, dim, labels);
        if (part.stage !== 'bus') faceHits.push({ quad, bounds: item.bounds, stage: part.stage });
      }
    }
    const boards = prepared.filter((item) => item.part.finish === 'board');
    depthOrder(boards, view.eye).forEach(paint);
    for (const trace of geometry.traces) drawTrace(ctx, trace, project, selected, labels && width >= 475);
    depthOrder(
      prepared.filter((item) => item.part.finish !== 'board'),
      view.eye,
    ).forEach(paint);
    labelHits = labels
      ? stationLabels(ctx, geometry.stations, project, width, height, selected, hovered, prepared)
      : [];
    orientation(ctx, view, height);
    if (spread !== targetSpread) requestDraw();
  }
  function requestDraw() {
    if (!frame && !signal.aborted) frame = requestAnimationFrame(render);
  }
  function resize() {
    const box = canvas.getBoundingClientRect();
    width = box.width;
    height = box.height;
    ratio = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.round(width * ratio)),
      h = Math.max(1, Math.round(height * ratio));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    if (!customCamera) ({ yaw, pitch } = home());
    requestDraw();
  }
  function setModel(next) {
    if (!['nanogpt', 'llama', 'qwen'].includes(next) || next === model) return;
    model = next;
    selected = null;
    hover(null);
    textures.clear();
    faceHits = [];
    labelHits = [];
    rebuild();
    reset();
  }
  function select(stageId) {
    selected = geometry.stations.some((s) => s.stage === stageId) ? stageId : null;
    requestDraw();
  }
  function orbit(dx, dy) {
    customCamera = true;
    yaw -= dx * 0.0071;
    pitch = clamp(pitch + dy * 0.0055, 0.1, 1.31);
    requestDraw();
  }
  function zoomBy(factor, anchor) {
    if (!(factor > 0) || !Number.isFinite(factor)) return;
    const next = clamp(zoom * factor, 0.55, 2.75);
    if (anchor && zoom !== next) {
      const k = next / zoom;
      panX = anchor[0] - width / 2 - (anchor[0] - width / 2 - panX) * k;
      panY = anchor[1] - height * 0.52 - (anchor[1] - height * 0.52 - panY) * k;
    }
    zoom = next;
    onZoom?.(zoom);
    requestDraw();
  }
  function atPoint(px, py) {
    for (const rect of labelHits) {
      if (px >= rect.left && px <= rect.right && py >= rect.top && py <= rect.bottom) return rect.stage;
    }
    for (let i = faceHits.length - 1; i >= 0; i--) {
      const { quad, bounds: b, stage } = faceHits[i];
      if (px >= b.left && px <= b.right && py >= b.top && py <= b.bottom && contains(quad, px, py))
        return stage;
    }
    return null;
  }
  function hover(stage) {
    if (stage !== hovered) {
      hovered = stage;
      onHover?.(stage);
      requestDraw();
    }
    if (!pointers.size) canvas.style.cursor = stage ? 'pointer' : 'grab';
  }
  function local(e) {
    const rect = canvas.getBoundingClientRect();
    return [e.clientX - rect.left, e.clientY - rect.top];
  }
  function pinch() {
    const [a, b] = [...pointers.values()];
    return { distance: Math.hypot(a.x - b.x, a.y - b.y), mid: [(a.x + b.x) / 2, (a.y + b.y) / 2] };
  }
  function pointerDown(e) {
    if (e.button !== 0 && e.button !== 2) return;
    const [x, y] = local(e);
    const pointer = { x, y, startX: x, startY: y, moved: false, panning: e.shiftKey || e.button === 2 };
    pointers.set(e.pointerId, pointer);
    canvas.setPointerCapture(e.pointerId);
    if (pointers.size >= 2) {
      for (const p of pointers.values()) p.moved = true;
      ({ distance: pinchDistance, mid: pinchMid } = pinch());
    }
    canvas.style.cursor = 'grabbing';
  }
  function pointerMove(e) {
    const [x, y] = local(e),
      p = pointers.get(e.pointerId);
    if (!p) {
      if (e.pointerType !== 'touch') hover(atPoint(x, y));
      return;
    }
    const dx = x - p.x,
      dy = y - p.y;
    p.x = x;
    p.y = y;
    if (!p.moved && Math.hypot(x - p.startX, y - p.startY) > 4) p.moved = true;
    if (!p.moved) return;
    hover(null);
    if (pointers.size > 1) {
      const { distance, mid } = pinch();
      if (pinchMid) {
        panX += mid[0] - pinchMid[0];
        panY += mid[1] - pinchMid[1];
      }
      if (distance > 2 && pinchDistance > 2) zoomBy(distance / pinchDistance, mid);
      pinchDistance = distance;
      pinchMid = mid;
    } else if (p.panning) {
      panX += dx;
      panY += dy;
      requestDraw();
    } else {
      orbit(dx, dy);
    }
  }
  function pointerUp(e) {
    const p = pointers.get(e.pointerId);
    if (!p) return;
    pointers.delete(e.pointerId);
    if (canvas.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId);
    if (pointers.size < 2) {
      pinchMid = null;
      pinchDistance = 0;
    }
    if (!p.moved && !p.panning && e.type === 'pointerup') {
      const stage = atPoint(...local(e));
      if (stage) {
        select(stage);
        onSelect?.(stage);
      }
    }
    if (!pointers.size) hover(e.pointerType === 'touch' ? null : atPoint(...local(e)));
  }
  function keyDown(e) {
    const amount = e.shiftKey ? 80 : 35;
    switch (e.key) {
      case 'ArrowLeft':
        orbit(-amount, 0);
        break;
      case 'ArrowRight':
        orbit(amount, 0);
        break;
      case 'ArrowUp':
        orbit(0, -amount);
        break;
      case 'ArrowDown':
        orbit(0, amount);
        break;
      case '+':
      case '=':
        zoomBy(1.15);
        break;
      case '-':
      case '_':
        zoomBy(1 / 1.15);
        break;
      case 'Home':
      case '0':
        reset();
        break;
      default:
        return;
    }
    e.preventDefault();
    e.stopPropagation();
  }
  function wheel(e) {
    e.preventDefault();
    const unit = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? height : 1;
    zoomBy(Math.exp(-clamp(e.deltaY * unit, -230, 230) * 0.0015), local(e));
  }
  const on = (event, fn, options = {}) => canvas.addEventListener(event, fn, { signal, ...options });
  on('pointerdown', pointerDown);
  on('pointermove', pointerMove);
  on('pointerup', pointerUp);
  on('pointercancel', pointerUp);
  on('lostpointercapture', pointerUp);
  on('pointerleave', () => {
    if (!pointers.size) hover(null);
  });
  on('contextmenu', (e) => e.preventDefault());
  on('wheel', wheel, { passive: false });
  on('keydown', keyDown);
  const observer = new ResizeObserver(resize);
  observer.observe(canvas);
  window.addEventListener('resize', resize, { signal });
  rebuild();
  resize();

  return {
    setModel,
    select,
    orbit,
    zoomBy,
    reset,
    setLabels(visible) {
      labels = !!visible;
      requestDraw();
    },
    setExploded(open) {
      targetSpread = open ? 1 : 0;
      requestDraw();
    },
    destroy() {
      controller.abort();
      observer.disconnect();
      cancelAnimationFrame(frame);
      textures.clear();
      pointers.clear();
    },
  };
}
