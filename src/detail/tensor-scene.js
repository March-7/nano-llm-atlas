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
} from './tensor-paint.js';

const HOME_YAW = 0.34,
  HOME_PITCH = 0.22;
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
export function mountTensorScene(canvas, { onCell, onPick, onZoom } = {}) {
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
    stamp = null,
    focal = null;
  let active = null,
    pinned = null,
    slice = null,
    lastSent = '',
    queuedPointer = null;
  const layers = new Map(),
    root = new Map(),
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
  function layerAt(i) {
    if (layers.has(i)) return layers.get(i);
    const data = layout?.layers[i];
    if (!data) return null;
    const nodes = typeof data.nodes === 'function' ? data.nodes() : data.nodes;
    const batch = { nodes, map: new Map(nodes.map((n) => [n.id, n])) };
    // Peek/render very large models without retaining every head of every layer.
    if (layers.size >= 8) layers.delete(layers.keys().next().value);
    layers.set(i, batch);
    return batch;
  }
  function lookup(id, layerIndex) {
    if (root.has(id)) return root.get(id);
    if (layerIndex >= 0) {
      const local = layerAt(layerIndex)?.map.get(id);
      if (local) return local;
    }
    if (layerIndex > 0) return layerAt(layerIndex - 1)?.map.get(id);
    if (layerIndex === -1 && layout.layers.length) return layerAt(layout.layers.length - 1)?.map.get(id);
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
  function resetOrientation() {
    yaw = HOME_YAW;
    pitch = HOME_PITCH;
  }
  function overview() {
    if (!layout) return;
    resetOrientation();
    const b = layout.bounds;
    target = [(b[0] + b[3]) / 2, (b[1] + b[4]) / 2, (b[2] + b[5]) / 2];
    scale = baseline = fit();
    overviewMode = true;
    focal = null;
    select(null, true);
    zoomNotice();
    redraw();
  }
  function setLayout(next) {
    layout = next;
    sample = null;
    tiles.clear();
    layers.clear();
    root.clear();
    stamp = null;
    for (const n of layout.root) root.set(n.id, n);
    overview();
  }
  function focus(op, layer = 0, head = 0) {
    if (!layout) return null;
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
      const ax = basis(yaw, pitch);
      const shift = ((width < 550 ? 0.16 : 0.075) * width) / scale;
      target = target.map((v, k) => v - shift * ax.right[k]);
    }
    zoomNotice();
    redraw();
    return n;
  }
  function inspectCell(node, x, y, z) {
    if (!node) return null;
    const indices = [x, y, z].map((v, i) => clamp(Math.floor(v) || 0, 0, node[dims[i]] - 1));
    if (slice?.id === node.id && slice.z !== indices[2]) {
      slice = { id: node.id, z: indices[2] };
      redraw();
    }
    return select({ node, x: indices[0], y: indices[1], z: indices[2] }, true, true);
  }
  function focusCell(node, x, y, z) {
    if (!layout || !node) return null;
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
    const ax = basis(yaw, pitch);
    target = target.map((value, i) => value - (dx * ax.right[i] + dy * ax.down[i]) / scale);
    overviewMode = false;
    queuedPointer = null;
    redraw();
  }
  function orbit(dx, dy) {
    yaw -= dx * 0.006;
    pitch = clamp(pitch + dy * 0.005, -1.34, 1.34);
    overviewMode = false;
    queuedPointer = null;
    redraw();
  }
  function zoomBy(factor, anchor = [width / 2, height / 2]) {
    if (!layout || !(factor > 0) || !Number.isFinite(factor)) return;
    const previous = scale;
    scale = clamp(scale * factor, baseline * 0.28, Math.max(80 / layout.cell, baseline * 2));
    const ax = basis(yaw, pitch),
      shift = 1 / previous - 1 / scale;
    target = target.map(
      (v, i) => v + shift * ((anchor[0] - width / 2) * ax.right[i] + (anchor[1] - height / 2) * ax.down[i]),
    );
    overviewMode = false;
    zoomNotice();
    redraw();
  }

  function displayed(n) {
    if (slice?.id !== n.id) return n;
    const depth = n.dz / n.cz;
    return { ...n, z: n.z + slice.z * depth, dz: depth, cz: 1, sliceZ: slice.z, whole: n };
  }
  function prepared(raw, v) {
    const n = displayed(raw),
      b = box(n);
    return { node: n, box: b, rect: screenBox(b, v), depth: dot(center(n), v.eye) };
  }
  function sceneStamp(v, layerRect) {
    const first = layout.layers[0];
    const fw = layerRect[2] - layerRect[0],
      fh = layerRect[3] - layerRect[1];
    const density = Math.min(dpr, 620 / Math.max(1, fw, fh));
    const w = Math.max(4, Math.ceil(fw * density) + 2),
      h = Math.max(4, Math.ceil(fh * density) + 2);
    const signature = [yaw.toFixed(5), pitch.toFixed(5), w, h, v.scale.toPrecision(5)].join('/');
    if (stamp?.signature === signature) return stamp;
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
    const nodes = layerAt(0)
      .nodes.map((n) => prepared(n, small))
      .sort((a, b) => a.depth - b.depth);
    for (const item of nodes) paintNode(g, item.node, small, { flat: true });
    stamp = { signature, bitmap };
    return stamp;
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
  function layerIsStamp(layer, r) {
    return (
      layout.repeatedLayerGeometry &&
      layout.layers.length > 12 &&
      layout.cell * scale < 0.14 &&
      Math.max(r[2] - r[0], r[3] - r[1]) < 650 &&
      !!layer.bounds
    );
  }
  function candidates(x, y, v) {
    const point = [x, y, x, y],
      found = [...layout.root];
    for (const layer of layout.layers) {
      if (intersects(screenBox(layer.bounds, v), point)) found.push(...layerAt(layer.index).nodes);
    }
    return found;
  }
  function pickAt(x, y, v = makeView()) {
    if (!layout) return null;
    let hit = null;
    const point = [x, y, x, y];
    for (const original of candidates(x, y, v)) {
      const n = displayed(original);
      if (!intersects(screenBox(box(n), v), point)) continue;
      for (const face of planes(n, v)) {
        const next = pickFace(face, x, y, v);
        if (next && (!hit || next.distance > hit.distance)) hit = next;
      }
    }
    return hit;
  }
  function draw() {
    frame = 0;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = '#f3f3f2';
    ctx.fillRect(0, 0, width, height);
    if (!layout) return;
    const v = makeView(),
      area = [0, 0, width, height];
    if (queuedPointer && !pointers.size) {
      const next = pickAt(...queuedPointer, v);
      select(next ?? pinned);
      canvas.style.cursor = next ? 'crosshair' : 'grab';
      queuedPointer = null;
    }
    let visible = layout.root.map((n) => prepared(n, v)).filter((item) => intersects(item.rect, area));
    const stamped = [];
    for (const layer of layout.layers) {
      const r = screenBox(layer.bounds, v);
      if (!intersects(r, area)) continue;
      if (layerIsStamp(layer, r)) {
        stamped.push({ layer, r });
        continue;
      }
      const nodes = layerAt(layer.index).nodes;
      for (const n of nodes) {
        const item = prepared(n, v);
        if (intersects(item.rect, area)) visible.push(item);
      }
    }
    if (stamped.length) {
      const atlas = sceneStamp(v, stamped[0].r);
      stamped.sort(
        (a, b) => dot(a.layer.bounds.slice(0, 3), v.eye) - dot(b.layer.bounds.slice(0, 3), v.eye),
      );
      const img = atlas.bitmap;
      for (const { r } of stamped)
        ctx.drawImage(img, 1, 1, img.width - 2, img.height - 2, r[0], r[1], r[2] - r[0], r[3] - r[1]);
    }
    const spacing = layout.cell * scale;
    const deps = inspectDependencies(active),
      depIds = new Set(deps.map(({ source }) => source.id));
    if (spacing > 0.68 && visible.length < 280) {
      for (const { node } of visible)
        for (const id of node.deps ?? []) {
          const src = lookup(id, node.layer);
          if (src && src.id !== node.id) connect(ctx, src, node, v, layout.cell);
        }
    }
    visible.sort((a, b) => compareDepth(a, b, v));
    for (const item of visible)
      paintNode(ctx, item.node, v, {
        sample,
        cache: tiles,
        shape: layout.shape,
        flat: spacing < 0.075,
        selected: active?.node.id === item.node.id,
        dependency: depIds.has(item.node.id),
      });
    if (labels) {
      const used = [];
      if (spacing > 0.55 && visible.length < 340) {
        const ranked = [...visible].sort(
          (a, b) =>
            Number(b.node.id === active?.node.id) - Number(a.node.id === active?.node.id) ||
            (b.rect[2] - b.rect[0]) * (b.rect[3] - b.rect[1]) -
              (a.rect[2] - a.rect[0]) * (a.rect[3] - a.rect[1]),
        );
        for (const { node, rect: r } of ranked) {
          const w = r[2] - r[0],
            h = r[3] - r[1],
            isActive = node.id === active?.node.id;
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
      } else if (stamped.length || layout.layers.length) {
        const interval = Math.max(1, Math.ceil(layout.layers.length / 12));
        for (const layer of layout.layers) {
          if (layer.index % interval !== 0) continue;
          const r = screenBox(layer.bounds, v);
          if (intersects(area, r) && r[3] - r[1] > 12)
            tag(ctx, 'L' + (layer.index + 1), r[0] + 10, r[1] + 12, 12, area, used);
        }
      }
    }
    if (active) {
      const h = active,
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
      baseline = fit();
      if (overviewMode) scale = baseline;
      zoomNotice();
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
    const pos = local(e);
    pointers.set(e.pointerId, { pos, start: pos, moved: false });
    canvas.setPointerCapture(e.pointerId);
    queuedPointer = null;
    pressedMode = e.button === 2 || e.altKey || e.shiftKey ? 'orbit' : interaction;
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
      const hit = select(pickAt(...local(e)), true, true);
      if (hit) onPick?.(hit);
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
    else if (key === 'Home' || key === '0') overview();
    else if (key.toLowerCase() === 'f') centerCell();
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
    setLayout,
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
      layers.clear();
      tiles.clear();
      pointers.clear();
      stamp = null;
    },
  };
}
