import {
  arrangeWorld,
  localView,
  overviewPitch,
  overviewYaw,
  toLocal,
  toWorld,
  worldInsets,
} from './atlas-world.js?v=3.1';
import {
  TileCache,
  basis,
  box,
  cameraMap,
  cellPoint,
  center,
  clamp,
  dot,
  intersects,
  paintNode,
  paintSlice,
  palette,
  pickFace,
  planes,
  screenBox,
} from './tensor-paint.js?v=3.1';

const HOME_YAW = overviewYaw,
  HOME_PITCH = overviewPitch;
const dims = ['cx', 'cy', 'cz'];
const colors = { w: '#5652b9', i: '#47944f', a: '#b08e26' };

function placeArrow(ctx, a, b, dirA, dirB, rgb, width, selected = false) {
  const dx = b[0] - a[0],
    dy = b[1] - a[1],
    length = Math.hypot(dx, dy);
  if (length < 4 || length > 9000) return;
  const unit = (v) => {
    const d = Math.hypot(...v) || 1;
    return [v[0] / d, v[1] / d];
  };
  const u = unit(dirA),
    v = unit(dirB);
  const handle = Math.min(92, length * 0.43);
  const head = Math.min(length * 0.35, 3.9 + width * 1.18);
  const end = [b[0] - v[0] * head, b[1] - v[1] * head];
  ctx.strokeStyle = 'rgba(' + rgb + ',' + (selected ? 0.88 : 0.37) + ')';
  ctx.fillStyle = ctx.strokeStyle;
  ctx.lineWidth = width;
  ctx.lineCap = 'butt';
  ctx.beginPath();
  ctx.moveTo(...a);
  ctx.bezierCurveTo(
    a[0] + u[0] * handle,
    a[1] + u[1] * handle,
    end[0] - v[0] * handle,
    end[1] - v[1] * handle,
    ...end,
  );
  ctx.stroke();
  const r = Math.max(2.1, width * 0.9);
  ctx.beginPath();
  ctx.moveTo(...b);
  ctx.lineTo(end[0] - v[1] * r, end[1] + v[0] * r);
  ctx.lineTo(end[0] + v[1] * r, end[1] - v[0] * r);
  ctx.closePath();
  ctx.fill();
}

function connect(ctx, source, dest, view, cell) {
  const a = center(source),
    b = center(dest);
  const aa = box(source),
    bb = box(dest);
  const gaps = a.map((v, i) => Math.max(bb[i] - aa[i + 3], aa[i] - bb[i + 3]));
  let from = gaps.indexOf(Math.max(...gaps)),
    to = from;
  // A residual branch leaves the bottom of a layer and enters the normalizer
  // from the side. The straight stream remains visible beside the branch.
  if (source.op?.includes('Residual') && dest.stage === 'norm' && gaps[1] > 0 && gaps[0] > 0) {
    from = 1;
    to = 0;
  }
  if (dest.op === 'mlpFc' && source.op?.includes('lnResid') && gaps[1] > 0 && gaps[0] > 0) {
    from = 1;
    to = 0;
  }
  const signA = Math.sign(b[from] - a[from]) || 1,
    signB = Math.sign(b[to] - a[to]) || 1;
  a[from] = signA > 0 ? aa[from + 3] : aa[from];
  b[to] = signB > 0 ? bb[to] : bb[to + 3];
  const p = view.project(a),
    q = view.project(b),
    padding = clamp(cell * view.scale * 0.5, 0.8, 5);
  const da = [view.right[from] * signA, view.down[from] * signA];
  const db = [view.right[to] * signB, view.down[to] * signB];
  p[0] += da[0] * padding;
  p[1] += da[1] * padding;
  q[0] -= db[0] * padding;
  q[1] -= db[1] * padding;
  const thick = clamp(cell * view.scale * (source.kind === 'a' ? 1.0 : 2.8), 0.85, 15);
  placeArrow(ctx, p, q, da, db, palette[source.kind].flow, thick);
}

function tag(ctx, name, x, y, height, area, used, force = false) {
  if (!name || x < -100 || x > area[2] + 100 || y < -2 || y > area[3] + 13) return;
  const size = height - 4;
  ctx.font = '450 ' + size + 'px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
  const width = ctx.measureText(name).width + 6;
  const left = clamp(x - width / 2, 2, Math.max(2, area[2] - width - 2));
  const top = Math.max(2, y - height - 1);
  const r = [left, top, left + width, top + height];
  if (!force && used.some((prev) => intersects(prev, r))) return;
  used.push(r);
  ctx.fillStyle = 'rgba(15,17,17,0.94)';
  ctx.beginPath();
  ctx.roundRect(left, top, width, height, 2);
  ctx.fill();
  ctx.fillStyle = '#f9faf8';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(name, left + width / 2, top + height / 2 + 0.3);
}

function compareDepth(a, b, view) {
  // Disjoint solids with overlapping projections have the same ordering on
  // every separating axis. This also handles a long matrix beside a thin bias.
  if (intersects(a.rect, b.rect)) {
    let sign = 0,
      conflict = false;
    for (let i = 0; i < 3; i++) {
      if (Math.abs(view.eye[i]) < 0.0001) continue;
      const k =
        a.box[i + 3] <= b.box[i]
          ? -Math.sign(view.eye[i])
          : b.box[i + 3] <= a.box[i]
            ? Math.sign(view.eye[i])
            : 0;
      if (sign && k && sign !== k) conflict = true;
      if (k) sign = k;
    }
    if (sign && !conflict) return sign;
  }
  return a.depth - b.depth;
}

/**
 * Orthographic viewer for layouts in true world units. focus() returns its
 * node synchronously. inspectCell()/focusCell() return the selected hit and
 * synchronously notify onCell. Camera changes refresh screen coordinates on
 * the next frame. onPick only fires for a direct click or tap on a tensor.
 */
export function mountTensorScene(
  canvas,
  { onCell, onPick, onZoom, onModelPick, onWorldLabels, onOverview } = {},
) {
  const ctx = canvas.getContext('2d');
  const events = new AbortController(),
    { signal } = events;
  let layout,
    sample,
    width = 1,
    height = 1,
    dpr = 1,
    frame = 0;
  let yaw = HOME_YAW,
    pitch = HOME_PITCH,
    target = [0, 0, 0],
    scale = 1,
    baseline = 1;
  let labels = true,
    overviewMode = false,
    interaction = 'pan',
    focal = null;
  let active = null,
    pinned = null,
    slice = null,
    lastSent = '',
    queuedPointer = null;
  let entries = [],
    placements = new Map(),
    lastMarkers = '',
    returnToAtlas = true,
    flight = null;
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const stores = new WeakMap(),
    tiles = new TileCache(),
    pointers = new Map();
  let pinchMid = null,
    pinchDistance = 0,
    pressedMode = 'pan';
  const makeView = () => cameraMap(basis(yaw, pitch), target, scale, width, height);
  const redraw = () => {
    if (!frame && !signal.aborted) frame = requestAnimationFrame(draw);
  };
  const zoomNotice = () => onZoom?.(scale / Math.max(baseline, 1e-12));
  function storeFor(model) {
    if (!stores.has(model))
      stores.set(model, {
        layers: new Map(),
        root: new Map(model.root.map((n) => [n.id, n])),
        stamp: null,
      });
    return stores.get(model);
  }
  function globalView() {
    const current = placements.get(layout?.id);
    if (!current) return makeView();
    const axes = basis(yaw, pitch);
    return cameraMap(axes, toWorld(current, target, axes), scale / current.unit, width, height);
  }
  function isWide() {
    const current = placements.get(layout?.id);
    return !!current && scale / current.unit <= 1.85;
  }
  function layerAt(i, model = layout) {
    const { layers } = storeFor(model);
    if (layers.has(i)) return layers.get(i);
    const data = model.layers[i];
    if (!data) return null;
    const nodes = typeof data.nodes === 'function' ? data.nodes() : data.nodes;
    const batch = { nodes, map: new Map(nodes.map((n) => [n.id, n])) };
    // Peek/render very large models without retaining every head of every layer.
    if (layers.size >= 8) layers.delete(layers.keys().next().value);
    layers.set(i, batch);
    return batch;
  }
  function lookup(id, layerIndex, model = layout) {
    const { root } = storeFor(model);
    if (root.has(id)) return root.get(id);
    if (layerIndex >= 0) {
      const local = layerAt(layerIndex, model)?.map.get(id);
      if (local) return local;
    }
    if (layerIndex > 0) return layerAt(layerIndex - 1, model)?.map.get(id);
    if (layerIndex === -1 && model.layers.length)
      return layerAt(model.layers.length - 1, model)?.map.get(id);
    return undefined;
  }
  function fit() {
    if (!layout) return 1;
    const v = cameraMap(basis(yaw, pitch), [0, 0, 0], 1, width, height);
    const r = screenBox(layout.bounds, v);
    return Math.max(
      1e-9,
      Math.min((width * 0.9) / Math.max(1, r[2] - r[0]), (height * 0.91) / Math.max(1, r[3] - r[1])),
    );
  }
  function inform(force = false) {
    if (!active) {
      if (lastSent || force) onCell?.(null);
      lastSent = '';
      return null;
    }
    const p = makeView().project(cellPoint(active.node, active.x, active.y, active.z));
    const hit = { ...active, screenX: p[0], screenY: p[1] };
    const key = [active.node.id, active.x, active.y, active.z, Math.round(p[0]), Math.round(p[1])].join(
      ':',
    );
    if (force || key !== lastSent) onCell?.(hit);
    lastSent = key;
    return hit;
  }
  function select(hit, pin = false, force = false) {
    const prev = active;
    active = hit ? { node: hit.node, x: hit.x, y: hit.y, z: hit.z } : null;
    if (pin) {
      pinned = active;
      if (!active) slice = null;
    }
    if (
      prev?.node.id !== active?.node.id ||
      prev?.x !== active?.x ||
      prev?.y !== active?.y ||
      prev?.z !== active?.z
    )
      redraw();
    return inform(force);
  }
  function pose() {
    return { target: [...target], scale, yaw, pitch };
  }
  function applyPose(p) {
    target = p.target;
    scale = p.scale;
    yaw = p.yaw;
    pitch = p.pitch;
  }
  function frameChange(start, options) {
    const finish = pose();
    queuedPointer = null;
    flight = null;
    const ratio = Math.max(start.scale, finish.scale) / Math.min(start.scale, finish.scale);
    const travel =
      (Math.hypot(...target.map((v, i) => v - start.target[i])) * Math.sqrt(start.scale * finish.scale)) /
      Math.max(width, height, 1);
    const yawDelta = Math.atan2(Math.sin(finish.yaw - start.yaw), Math.cos(finish.yaw - start.yaw));
    const angular = Math.abs(yawDelta) + Math.abs(finish.pitch - start.pitch);
    if (
      options?.animate &&
      !reducedMotion.matches &&
      ratio < 48 &&
      travel < 9 &&
      (ratio > 1.005 || travel > 0.005 || angular > 0.005)
    ) {
      flight = {
        start,
        finish,
        yawDelta,
        began: performance.now(),
        duration: clamp(325 + 44 * Math.log(ratio) + 34 * travel, 325, 510),
        pullback: Math.min(0.72, Math.log1p(Math.max(0, travel - 0.5) * 0.52)),
      };
      applyPose(start);
    }
    zoomNotice();
    redraw();
  }
  function advanceFlight(time) {
    if (!flight) return;
    const { start, finish, yawDelta, duration, began, pullback } = flight;
    const t = reducedMotion.matches ? 1 : clamp((time - began) / duration, 0, 1);
    if (t === 1) {
      applyPose(finish);
      flight = null;
    } else {
      const u = t * t * (3 - 2 * t);
      target = start.target.map((v, i) => v + (finish.target[i] - v) * u);
      scale = Math.exp(
        (1 - u) * Math.log(start.scale) +
          u * Math.log(finish.scale) -
          pullback * Math.sin(Math.PI * u) ** 2,
      );
      yaw = start.yaw + yawDelta * u;
      pitch = start.pitch + (finish.pitch - start.pitch) * u;
      redraw();
    }
    zoomNotice();
  }
  function resetOrientation() {
    yaw = HOME_YAW;
    pitch = HOME_PITCH;
  }
  function fitModel(options) {
    if (!layout) return;
    const start = pose();
    returnToAtlas = true;
    resetOrientation();
    const b = layout.bounds;
    target = [0, 1, 2].map((i) => (b[i] + b[i + 3]) / 2);
    const model = placements.get(layout.id);
    if (model) {
      const pad = worldInsets(width, height);
      const v = cameraMap(makeView(), target, 1, width, height);
      const r = screenBox(b, v);
      scale = Math.min(
        (width - 2 * pad.side) / Math.max(1, r[2] - r[0]),
        (height - pad.top - pad.bottom) / Math.max(1, r[3] - r[1]),
      );
      target = target.map((val, k) => val - (v.down[k] * (pad.top - pad.bottom)) / (2 * scale));
      baseline = model.unit;
    } else scale = baseline = fit();
    overviewMode = 'model';
    focal = null;
    select(null, true);
    frameChange(start, options);
  }
  function overview(options) {
    if (!layout) return;
    const model = placements.get(layout.id);
    if (!model) {
      fitModel(options);
      return;
    }
    const start = pose();
    resetOrientation();
    returnToAtlas = true;
    target = toLocal(model, [0, 0, 0], basis(yaw, pitch));
    scale = baseline = model.unit;
    overviewMode = 'world';
    focal = null;
    select(null, true);
    frameChange(start, options);
  }
  function setLayout(next) {
    if (!next) return;
    flight = null;
    const prior = layout && placements.has(layout.id) ? globalView() : null;
    if (layout !== next) {
      sample = null;
      tiles.clear();
    }
    if (placements.has(next.id) && placements.get(next.id).layout !== next) {
      entries = entries.map((entry) => (entry.id === next.id ? { ...entry, layout: next } : entry));
      placements = arrangeWorld(entries, width, height);
    }
    layout = next;
    focal = null;
    queuedPointer = null;
    select(null, true);
    const model = placements.get(next.id);
    if (!model) {
      fitModel();
      return;
    }
    baseline = model.unit;
    if (prior) {
      target = toLocal(model, prior.target, prior);
      scale = prior.scale * model.unit;
      zoomNotice();
      redraw();
    } else overview();
  }
  function setWorld(next) {
    entries = Array.from(next);
    placements = arrangeWorld(entries, width, height);
    if (!entries.length) {
      publishMarkers([]);
      if (layout) fitModel();
      return;
    }
    if (!placements.has(layout?.id)) setLayout(entries[0].layout);
    overview();
  }
  function focus(op, layer = 0, head = 0) {
    if (!layout) return null;
    flight = null;
    const i = clamp(Math.trunc(layer) || 0, 0, Math.max(0, layout.layers.length - 1));
    const local = layerAt(i)?.nodes ?? [];
    const nodes = [...local, ...layout.root];
    let n = nodes.find((n) => n.op === op && n.head === head) ?? nodes.find((n) => n.op === op);
    let packedOffset = 0;
    if (!n) {
      const packed = /^([qkv])(WeightBlock|Block)$/.exec(op);
      if (packed) {
        const packedOp = 'qkv' + packed[2];
        n = nodes.find((a) => a.op === packedOp && a.head === head);
        if (n) packedOffset = ('qkv'.indexOf(packed[1]) * n.cy) / 3;
      }
    }
    if (!n) {
      const preferred = {
        embed: ['residual0', 'tokEmbedObj'],
        norm: ['ln2.lnResid', 'ln1.lnResid'],
        attention: ['qBlock', 'attnMtx', 'qkvBlock'],
        ffn: ['mlpFc', 'mlpFcWeight'],
        output: ['logits', 'logitsSoftmax'],
      };
      for (const key of preferred[op] ?? []) {
        n = nodes.find((a) => a.op === key && (a.head === head || a.head < 0));
        if (n) break;
      }
      n ??= nodes.find((a) => a.stage === op && a.kind === 'i' && (a.head === head || a.head < 0));
      n ??= nodes.find((a) => a.stage === op);
    }
    if (!n) return null;
    resetOrientation();
    returnToAtlas = true;
    select(null, true);
    overviewMode = false;
    focal = n;
    const pixel = clamp(Math.min(height / 249, width / 290), width < 550 ? 2.58 : 2.8, 6.8);
    // A recurrent memory has just eight channels per axis in the teaching graph.
    // Frame its neighboring read/write tensors at a readable cell pitch.
    scale = (pixel * (op === 'deltaState' ? 1.55 : 1)) / layout.cell;
    target = center(n);
    // Place layer norm beside its learned scalars and the wide feed-forward
    // weight. For huge matrices, anchor near their first rows, at cell scale.
    if (n.cx > 300 || n.cy > 300) {
      target = cellPoint(
        n,
        Math.min(16, n.cx - 1),
        Math.min(packedOffset + 18, n.cy - 1),
        Math.floor((n.cz - 1) / 2),
      );
    }
    if (op === 'ln2.lnResid' && n.cx < 300) {
      const ax = makeView();
      const shift = ((width < 550 ? 0.16 : 0.075) * width) / scale;
      target = target.map((v, k) => v - shift * ax.right[k]);
    }
    zoomNotice();
    redraw();
    return n;
  }
  function activeNode(node) {
    if (!layout || !node) return null;
    const local = lookup(node.id, node.layer);
    return local && local.cx === node.cx && local.cy === node.cy && local.cz === node.cz ? local : null;
  }
  function inspectCell(node, x, y, z) {
    flight = null;
    node = activeNode(node);
    if (!node) return null;
    const indices = [x, y, z].map((v, i) => clamp(Math.floor(v) || 0, 0, node[dims[i]] - 1));
    if (slice?.id === node.id && slice.z !== indices[2]) {
      slice = { id: node.id, z: indices[2] };
      redraw();
    }
    return select({ node, x: indices[0], y: indices[1], z: indices[2] }, true, true);
  }
  function focusCell(node, x, y, z) {
    flight = null;
    node = activeNode(node);
    if (!node) return null;
    if (overviewMode || isWide()) resetOrientation();
    returnToAtlas = true;
    const idx = [x, y, z].map((v, i) => clamp(Math.floor(v) || 0, 0, node[dims[i]] - 1));
    target = cellPoint(node, ...idx);
    focal = node;
    overviewMode = false;
    slice = node.cz > 1 ? { id: node.id, z: idx[2] } : null;
    scale = Math.max(scale, 25 / layout.cell);
    zoomNotice();
    redraw();
    return inspectCell(node, ...idx);
  }
  function centerCell() {
    if (active) return focusCell(active.node, active.x, active.y, active.z);
    const n = focal ?? focus('norm');
    return n ? focusCell(n, Math.floor(n.cx / 2), Math.floor(n.cy / 2), Math.floor(n.cz / 2)) : null;
  }
  function pan(dx, dy) {
    flight = null;
    if (isWide() && (dx || dy)) returnToAtlas = false;
    const ax = makeView();
    target = target.map((value, i) => value - (dx * ax.right[i] + dy * ax.down[i]) / scale);
    overviewMode = false;
    queuedPointer = null;
    redraw();
  }
  function orbit(dx, dy) {
    flight = null;
    const old = basis(yaw, pitch),
      model = placements.get(layout?.id);
    const holdSlots = model && isWide();
    yaw = (yaw - dx * 0.006) % (Math.PI * 2);
    pitch = clamp(pitch + dy * 0.005, -1.34, 1.34);
    if (holdSlots) {
      // The observer may have arrived from any model or panned manually.
      // Preserve that exact projected pan while all volumes rotate in place.
      const next = basis(yaw, pitch);
      const offset = target.map((n, i) => n - model.localCenter[i]);
      const u = dot(offset, old.right),
        v = dot(offset, old.down),
        d = dot(offset, old.eye);
      target = model.localCenter.map((n, i) => n + u * next.right[i] + v * next.down[i] + d * next.eye[i]);
    }
    overviewMode = false;
    queuedPointer = null;
    redraw();
  }

  function guideZoomOut(previous, axes) {
    const model = placements.get(layout.id);
    if (!model || !returnToAtlas || scale >= previous) return;
    const ratio = scale / baseline,
      oldRatio = previous / baseline;
    if (ratio >= 1.8) return;
    const progress = (r) => {
      const t = clamp((1.8 - r) / 0.8, 0, 1);
      return t * t * (3 - 2 * t);
    };
    const p = progress(ratio),
      old = progress(oldRatio);
    const amount = old >= 1 ? 1 : clamp((p - old) / (1 - old), 0, 1);
    const point = toWorld(model, target, axes),
      u = dot(point, axes.right),
      v = dot(point, axes.down);
    const free = Math.max(0, 1 / ratio - 1),
      pad = worldInsets(width, height);
    const freeX = free * Math.max(0, width / 2 - pad.side - 65);
    const freeAbove = free * Math.max(0, height / 2 - pad.top - 40);
    const freeBelow = free * Math.max(0, height / 2 - pad.bottom - 8);
    const du = (clamp(u, -freeX, freeX) - u) * amount;
    const dv = (clamp(v, -freeBelow, freeAbove) - v) * amount;
    target = target.map((n, i) => n + (du * axes.right[i] + dv * axes.down[i]) / model.unit);
  }
  function zoomBy(factor, anchor = [width / 2, height / 2]) {
    if (!layout || !(factor > 0) || !Number.isFinite(factor)) return;
    flight = null;
    const previous = scale,
      wasWide = isWide(),
      axes = basis(yaw, pitch);
    const dx = anchor[0] - width / 2,
      dy = anchor[1] - height / 2;
    scale = clamp(scale * factor, baseline * 0.28, Math.max(80 / layout.cell, baseline * 2));
    const shift = 1 / previous - 1 / scale;
    target = target.map((n, i) => n + (dx * axes.right[i] + dy * axes.down[i]) * shift);
    if (scale > previous && scale >= baseline * 1.8) returnToAtlas = true;
    guideZoomOut(previous, axes);
    if (!wasWide && isWide()) select(null, true);
    overviewMode = false;
    zoomNotice();
    redraw();
  }

  function displayed(n) {
    if (slice?.id !== n.id) return n;
    const depth = n.dz / n.cz;
    return { ...n, z: n.z + slice.z * depth, dz: depth, cz: 1, sliceZ: slice.z, whole: n };
  }
  function prepared(raw, v, allowSlice = true) {
    const n = allowSlice ? displayed(raw) : raw,
      b = box(n);
    return { node: n, box: b, rect: screenBox(b, v), depth: dot(center(n), v.eye) };
  }
  function sceneStamp(model, v, layerRect, roleTint) {
    const cache = storeFor(model);
    const first = model.layers[0];
    const fw = layerRect[2] - layerRect[0],
      fh = layerRect[3] - layerRect[1];
    const density = Math.min(dpr, 620 / Math.max(1, fw, fh));
    const w = Math.max(4, Math.ceil(fw * density) + 2),
      h = Math.max(4, Math.ceil(fh * density) + 2);
    const signature = [yaw.toFixed(5), pitch.toFixed(5), w, h, v.scale.toPrecision(5)].join('/');
    if (cache.stamp?.signature === signature) return cache.stamp;
    const bitmap = document.createElement('canvas');
    bitmap.width = w;
    bitmap.height = h;
    const g = bitmap.getContext('2d');
    const b = first.bounds,
      mid = [0, 1, 2].map((i) => (b[i] + b[i + 3]) / 2);
    const small = cameraMap(
      v,
      mid,
      v.scale * Math.min((w - 2) / Math.max(1e-5, fw), (h - 2) / Math.max(1e-5, fh)),
      w,
      h,
    );
    const nodes = layerAt(0, model)
      .nodes.map((n) => prepared(n, small, false))
      .sort((a, b) => a.depth - b.depth);
    for (const item of nodes) paintNode(g, item.node, small, { flat: true, roleTint });
    cache.stamp = { signature, bitmap };
    return cache.stamp;
  }
  function inspectDependencies(hit) {
    if (!hit || !layout || (hit.node.causal && hit.x > hit.y)) return [];
    const n = hit.node,
      result = [],
      ids = n.deps ?? [];
    for (let i = 0; i < ids.length; i++) {
      const source = lookup(ids[i], n.layer);
      if (!source) continue;
      const mapping = n.depMaps?.[i];
      if (mapping?.span && mapping.span.axis !== 'i') {
        const { axis, start, end } = mapping.span;
        if (hit[axis] < start || hit[axis] >= end) continue;
      }
      result.push({ source, mapping });
    }
    return result;
  }
  function dependencySlice(hit, source, rule) {
    if (!rule?.map) return null;
    const map = rule.map,
      ix = [hit.x, hit.y, hit.z];
    const coord = { 0: 0, x: ix[0], y: ix[1], z: ix[2], b: ix[2], i: 0 };
    if (map.includes('t')) {
      const idx = layout.root.find((n) => n.op === 'idxObj');
      const token = idx && sample?.value(idx, ix[0], 0, ix[2]);
      if (!Number.isInteger(token)) return null;
      coord.t = token;
    }
    const span = rule.span;
    const axis = map.indexOf('i');
    let reductionStart = 0,
      reductionEnd = hit.node.dotLen ?? hit.node.reduction ?? Infinity;
    if (hit.node.limit === 'x+1') reductionEnd = Math.min(reductionEnd, hit.x + 1);
    if (hit.node.limit === 'y+1') reductionEnd = Math.min(reductionEnd, hit.y + 1);
    if (span) {
      if (span.axis === 'i') {
        reductionStart = Math.max(reductionStart, span.start);
        reductionEnd = Math.min(reductionEnd, span.end);
      } else if (coord[span.axis] < span.start || coord[span.axis] >= span.end) return null;
    }
    coord.i = reductionStart;
    const indices = Array.from({ length: 3 }, (_, k) => coord[map[k] ?? '0'] ?? 0);
    if (Array.isArray(rule.offset)) for (let k = 0; k < 3; k++) indices[k] += rule.offset[k] || 0;
    if (indices.some((v, i) => v < 0 || v >= source[dims[i]])) return null;
    const length =
      axis < 0 ? 1 : Math.min(source[dims[axis]] - indices[axis], reductionEnd - reductionStart);
    if (length <= 0) return null;
    return { indices, axis, length };
  }
  function layerIsStamp(model, layer, r, v) {
    return (
      model.repeatedLayerGeometry &&
      model.layers.length >= 8 &&
      model.cell * v.scale < 0.14 &&
      Math.max(r[2] - r[0], r[3] - r[1]) < 650 &&
      !!layer.bounds
    );
  }
  function views() {
    if (!placements.size) return [{ model: layout, v: makeView(), placement: null }];
    const g = globalView(),
      area = [0, 0, width, height],
      list = [];
    for (const m of placements.values()) {
      const v = localView(m, g),
        rect = screenBox(m.layout.bounds, v);
      if (!intersects(rect, area)) continue;
      list.push({ model: m.layout, v, placement: m, rect });
    }
    return list;
  }
  function publishMarkers(markers) {
    if (!onWorldLabels) return;
    const key = markers.map((m) => m.id + ':' + Math.round(m.x * 2) + ':' + Math.round(m.y * 2)).join('/');
    if (key === lastMarkers) return;
    lastMarkers = key;
    onWorldLabels(markers);
  }
  function candidates(area, v, model) {
    const found = [...model.root];
    for (const layer of model.layers) {
      if (intersects(screenBox(layer.bounds, v), area)) found.push(...layerAt(layer.index, model).nodes);
    }
    return found;
  }
  function pickScene(x, y, visibleModels, wide) {
    let hit = null,
      nearest = null;
    for (const { model, v, placement } of visibleModels) {
      const onlyModel = wide || model !== layout;
      const tol = onlyModel ? 4 : 0;
      const area = [x - tol, y - tol, x + tol, y + tol];
      if (!intersects(screenBox(model.bounds, v), area)) continue;
      for (const original of candidates(area, v, model)) {
        const n = model === layout ? displayed(original) : original,
          r = screenBox(box(n), v);
        if (!intersects(r, area)) continue;
        if (tol) {
          const dx = Math.max(0, r[0] - x, x - r[2]),
            dy = Math.max(0, r[1] - y, y - r[3]);
          const d = Math.hypot(dx, dy);
          if (d <= tol && (!nearest || d < nearest.delta)) nearest = { model: model.id, delta: d };
        }
        if (x < r[0] || y < r[1] || x > r[2] || y > r[3]) continue;
        for (const face of planes(n, v)) {
          const cell = pickFace(face, x, y, v);
          if (!cell) continue;
          const depth = placement
            ? (cell.distance - dot(placement.localCenter, v.eye)) * placement.unit
            : cell.distance;
          if (!hit || depth > hit.depth)
            hit = { model: onlyModel ? model.id : null, cell: onlyModel ? null : cell, depth };
        }
      }
    }
    return hit ?? nearest;
  }
  function drawModel(model, v, focused, wide) {
    const area = [0, 0, width, height],
      spacing = model.cell * v.scale;
    const roleTint = placements.size ? clamp((0.62 - spacing) / 0.56, 0, 1) : 0;
    const visible = model.root
      .map((n) => prepared(n, v, focused))
      .filter((item) => intersects(item.rect, area));
    const stamped = [];
    for (const layer of model.layers) {
      const r = screenBox(layer.bounds, v);
      if (!intersects(r, area)) continue;
      if (layerIsStamp(model, layer, r, v)) {
        stamped.push({ layer, r });
        continue;
      }
      const nodes = layerAt(layer.index, model).nodes;
      for (const n of nodes) {
        const item = prepared(n, v, focused);
        if (intersects(item.rect, area)) visible.push(item);
      }
    }
    if (stamped.length) {
      const atlas = sceneStamp(model, v, stamped[0].r, roleTint);
      stamped.sort(
        (a, b) => dot(a.layer.bounds.slice(0, 3), v.eye) - dot(b.layer.bounds.slice(0, 3), v.eye),
      );
      const img = atlas.bitmap;
      for (const { r } of stamped)
        ctx.drawImage(img, 1, 1, img.width - 2, img.height - 2, r[0], r[1], r[2] - r[0], r[3] - r[1]);
    }
    const cell = focused && !wide ? active : null;
    const deps = cell ? inspectDependencies(cell) : [],
      depIds = new Set(deps.map(({ source }) => source.id));
    if (!wide && spacing > 0.68 && visible.length < 280) {
      for (const { node } of visible)
        for (const id of node.deps ?? []) {
          const src = lookup(id, node.layer, model);
          if (src && src.id !== node.id) connect(ctx, src, node, v, model.cell);
        }
    }
    visible.sort((a, b) => compareDepth(a, b, v));
    for (const item of visible)
      paintNode(ctx, item.node, v, {
        sample: focused && !wide ? sample : null,
        roleTint,
        cache: tiles,
        shape: model.shape,
        flat: spacing < 0.075,
        selected: cell?.node.id === item.node.id,
        dependency: depIds.has(item.node.id),
      });
    if (labels && focused && !wide) {
      const used = [];
      if (spacing > 0.55 && visible.length < 340) {
        const ranked = [...visible].sort(
          (a, b) =>
            Number(b.node.id === cell?.node.id) - Number(a.node.id === cell?.node.id) ||
            (b.rect[2] - b.rect[0]) * (b.rect[3] - b.rect[1]) -
              (a.rect[2] - a.rect[0]) * (a.rect[3] - a.rect[1]),
        );
        for (const { node, rect: r } of ranked) {
          const w = r[2] - r[0],
            h = r[3] - r[1],
            isActive = node.id === cell?.node.id;
          if (!isActive && Math.max(w, h) < 10) continue;
          const p = v.project([node.x + node.dx / 2, node.y, node.z + node.dz]);
          const x = clamp(
            p[0],
            Math.max(0, r[0]) + Math.min(w / 2, 18),
            Math.max(Math.max(0, r[0]), Math.min(width, r[2]) - 12),
          );
          const y = Math.max(3, p[1]);
          const title =
            node.sliceZ != null
              ? node.name + ' · z=' + node.sliceZ
              : node.op?.endsWith('.lnSigma')
                ? 'γ'
                : node.op?.endsWith('.lnMu')
                  ? 'β'
                  : node.name;
          if (r[3] > 0 && r[1] < height)
            tag(ctx, title, x, y, clamp(11.1 + spacing * 0.68, 12, 16), area, used, isActive);
        }
      } else if (stamped.length || model.layers.length) {
        const interval = Math.max(1, Math.ceil(model.layers.length / 12));
        for (const layer of model.layers) {
          if (layer.index % interval !== 0) continue;
          const r = screenBox(layer.bounds, v);
          if (intersects(area, r) && r[3] - r[1] > 12)
            tag(ctx, 'L' + (layer.index + 1), r[0] + 10, r[1] + 12, 12, area, used);
        }
      }
    }
    if (cell) {
      const h = cell,
        end = v.project(cellPoint(h.node, h.x, h.y, h.z));
      const closeEnough = end[0] > -5 && end[0] < width + 5 && end[1] > -5 && end[1] < height + 5;
      if (closeEnough) {
        for (const { source, mapping } of deps) {
          const slice = dependencySlice(h, source, mapping);
          if (!slice && mapping?.span) continue;
          const p = slice
            ? paintSlice(ctx, source, slice.indices, v, { ...slice, color: colors[source.kind] })
            : v.project(center(source));
          const dy = end[1] - p[1],
            dx = end[0] - p[0];
          const direction = [dx, dy];
          placeArrow(ctx, p, end, [dx, 0.37 * dy], direction, palette[source.kind].flow, 1.1, true);
        }
        paintSlice(ctx, h.node, [h.x, h.y, h.z], v);
      }
    }
    if (!wide) return Infinity;
    // Use the geometry actually drawn. Empty corners of a model's global AABB
    // can sit well above its first tensor at oblique angles.
    let top = Infinity;
    for (const item of visible) top = Math.min(top, item.rect[1]);
    for (const item of stamped) top = Math.min(top, item.r[1]);
    return top;
  }
  function draw(time) {
    frame = 0;
    advanceFlight(time);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = '#f3f3f2';
    ctx.fillRect(0, 0, width, height);
    if (!layout) return;
    const models = views(),
      wide = isWide();
    if (flight) queuedPointer = null;
    if (queuedPointer && !pointers.size) {
      const hit = pickScene(...queuedPointer, models, wide);
      select(hit?.model || wide ? null : (hit?.cell ?? pinned));
      canvas.style.cursor = hit?.model ? 'pointer' : hit?.cell ? 'crosshair' : 'grab';
      queuedPointer = null;
    }
    const markers = [];
    for (const info of models) {
      const top = drawModel(info.model, info.v, info.model === layout, wide);
      if (!info.placement || !wide) continue;
      const x = info.v.project(info.placement.localCenter)[0],
        y = top - 4;
      if (x >= 0 && x <= width && y >= 16 && y <= height) markers.push({ id: info.model.id, x, y });
    }
    publishMarkers(markers);
    inform();
  }
  function resize() {
    const r = canvas.getBoundingClientRect();
    const oldW = width,
      oldH = height;
    width = Math.max(1, r.width);
    height = Math.max(1, r.height);
    dpr = Math.min(devicePixelRatio || 1, 2);
    if (canvas.width !== Math.round(width * dpr)) canvas.width = Math.round(width * dpr);
    if (canvas.height !== Math.round(height * dpr)) canvas.height = Math.round(height * dpr);
    if (layout && (width !== oldW || height !== oldH)) {
      if (entries.length) placements = arrangeWorld(entries, width, height);
      if (overviewMode === 'world') overview();
      else if (overviewMode === 'model') fitModel();
      else {
        baseline = placements.get(layout.id)?.unit ?? fit();
        zoomNotice();
      }
    }
    redraw();
  }
  const local = (e) => {
    const r = canvas.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  };
  const two = () => {
    const [a, b] = [...pointers.values()];
    return {
      dist: Math.hypot(a.pos[0] - b.pos[0], a.pos[1] - b.pos[1]),
      mid: [(a.pos[0] + b.pos[0]) / 2, (a.pos[1] + b.pos[1]) / 2],
    };
  };
  canvas.style.touchAction = 'none';
  canvas.style.cursor = 'grab';
  if (!canvas.hasAttribute('tabindex')) canvas.tabIndex = 0;
  const on = (type, fn, extra = {}) => canvas.addEventListener(type, fn, { signal, ...extra });
  on('pointerdown', (e) => {
    if (e.button !== 0 && e.button !== 2 && e.button !== 1) return;
    if (e.button === 1) e.preventDefault();
    if (flight) {
      flight = null;
      overviewMode = false;
    }
    const pos = local(e);
    pointers.set(e.pointerId, { pos, start: pos, moved: false });
    canvas.setPointerCapture(e.pointerId);
    queuedPointer = null;
    pressedMode = e.button === 1 || e.button === 2 || e.altKey || e.shiftKey ? 'orbit' : interaction;
    if (pointers.size > 1) {
      for (const p of pointers.values()) p.moved = true;
      ({ mid: pinchMid, dist: pinchDistance } = two());
    }
    canvas.style.cursor = 'grabbing';
  });
  on('pointermove', (e) => {
    const pos = local(e),
      p = pointers.get(e.pointerId);
    if (!p) {
      if (e.pointerType !== 'touch') {
        queuedPointer = pos;
        redraw();
      }
      return;
    }
    const dx = pos[0] - p.pos[0],
      dy = pos[1] - p.pos[1];
    p.pos = pos;
    p.moved ||= Math.hypot(pos[0] - p.start[0], pos[1] - p.start[1]) >= 4;
    if (!p.moved) return;
    if (pointers.size >= 2) {
      const { mid, dist } = two();
      if (pinchMid) pan(mid[0] - pinchMid[0], mid[1] - pinchMid[1]);
      if (pinchDistance > 1 && dist > 1) zoomBy(dist / pinchDistance, mid);
      pinchMid = mid;
      pinchDistance = dist;
    } else if (pressedMode === 'orbit') orbit(dx, dy);
    else pan(dx, dy);
  });
  function release(e) {
    const p = pointers.get(e.pointerId);
    if (!p) return;
    pointers.delete(e.pointerId);
    if (canvas.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId);
    if (pointers.size < 2) {
      pinchMid = null;
      pinchDistance = 0;
    }
    if (!p.moved && e.type === 'pointerup' && e.button === 0) {
      const target = layout && pickScene(...local(e), views(), isWide());
      if (target?.model) {
        select(null, true);
        if (onModelPick) onModelPick(target.model);
        else {
          setLayout(placements.get(target.model).layout);
          fitModel({ animate: true });
        }
      } else {
        const hit = select(target?.cell ?? null, true, true);
        if (hit) onPick?.(hit);
      }
    }
    if (!pointers.size) canvas.style.cursor = 'grab';
  }
  on('pointerup', release);
  on('pointercancel', release);
  on('lostpointercapture', release);
  on('pointerleave', () => {
    queuedPointer = null;
    if (!pointers.size) select(pinned);
  });
  on('contextmenu', (e) => e.preventDefault());
  on('auxclick', (e) => {
    if (e.button === 1) e.preventDefault();
  });
  on('mousedown', (e) => {
    if (e.button === 1) e.preventDefault();
  });
  on(
    'wheel',
    (e) => {
      e.preventDefault();
      const unit = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? height : 1;
      zoomBy(Math.exp(-clamp(e.deltaY * unit, -320, 320) * 0.0025), local(e));
    },
    { passive: false },
  );
  on('keydown', (e) => {
    const key = e.key,
      step = 42;
    const directions = {
      ArrowLeft: [step, 0],
      ArrowRight: [-step, 0],
      ArrowUp: [0, step],
      ArrowDown: [0, -step],
    };
    if (directions[key]) (e.shiftKey ? orbit : pan)(...directions[key]);
    else if (key === '+' || key === '=') zoomBy(1.32);
    else if (key === '-') zoomBy(1 / 1.32);
    else if (key === 'Home' || key === '0') {
      if (onOverview) onOverview();
      else overview({ animate: true });
    } else if (key.toLowerCase() === 'f') centerCell();
    else if (key === 'Escape') select(null, true);
    else return;
    e.preventDefault();
    e.stopPropagation();
  });
  const watcher = new ResizeObserver(resize);
  watcher.observe(canvas);
  window.addEventListener('resize', resize, { signal });
  resize();

  return {
    setWorld,
    setLayout,
    fitModel,
    overview,
    focus,
    focusCell,
    inspectCell,
    centerCell,
    zoomBy,
    orbit,
    pan,
    setMode(mode) {
      interaction = mode === 'orbit' ? 'orbit' : 'pan';
    },
    setValues(next) {
      sample = next;
      tiles.clear();
      redraw();
      inform(true);
    },
    refreshValues() {
      tiles.clear();
      redraw();
    },
    setLabels(on) {
      labels = !!on;
      redraw();
    },
    clearCell() {
      return select(null, true, true);
    },
    destroy() {
      events.abort();
      watcher.disconnect();
      cancelAnimationFrame(frame);
      entries = [];
      placements.clear();
      flight = null;
      publishMarkers([]);
      tiles.clear();
      pointers.clear();
    },
  };
}
