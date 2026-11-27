// Geometry adapted from Brendan Bycroft's LLM Visualization (MIT).
// https://github.com/bbycroft/llm-viz/blob/9da93742382f1bf36c020c38a1ace454e82c4490/src/llm/GptModelLayout.ts
// Copyright (c) 2023–2026 Brendan Bycroft. Full notice in assets/gpt-nano-sort-LICENSE.txt.
//
// A node is a tensor's bounding box in model space. X points right, Y down, Z forward.
// It describes the cells without allocating them. Model dimensions follow upstream Program.ts.
const CELL = 1.5;
const ROWS_PER_COLUMN = 12;
const previousResidual = Symbol('residual from preceding decoder block');
const layouts = new Map();

const shapes = {
  nanogpt: { T: 11, C: 48, nHeads: 3, nBlocks: 3, vocabSize: 3, context: 11 },
  'gpt2-small': { T: 1024, C: 768, nHeads: 12, nBlocks: 12, vocabSize: 50257, context: 1024 },
  'gpt2-xl': { T: 1024, C: 1600, nHeads: 25, nBlocks: 48, vocabSize: 50257, context: 1024 },
  // The reference drawing shows 1,024 time cells. The published GPT-3 context is 2,048.
  gpt3: { T: 1024, C: 12288, nHeads: 96, nBlocks: 96, vocabSize: 50257, context: 2048 },
};

function nodeId(layer, head, op) {
  return 'L' + layer + '.H' + head + '.' + op;
}

// A map names the source X/Y/Z cell indices in terms of the selected cell (x,y,z).
// i is a reduction index, t is the token ID at (x,z), and 0 is a singleton axis.
// offset is added to the mapped indices. span restricts the stated target or
// reduction axis to [start,end). For packed QKV two roles may share a source ID.
function from(source, map, role, slice) {
  return { source, mapping: { map, role, ...slice } };
}

function range(axis, start, end, offset) {
  return { span: { axis, start, end }, offset };
}

function drawing(shape) {
  const nodes = [];
  const sizes = {
    1: 1,
    B: shape.B,
    T: shape.T,
    C: shape.C,
    A: shape.A,
    V: shape.vocabSize,
    '4C': 4 * shape.C,
    '3A': 3 * shape.A,
  };

  // The source convention places most tensors by their right edge and their Z midpoint.
  function section(stage, head = -1) {
    return function add(op, name, kind, xyAxes, right, y, midZ, formula, inputs = [], flags = {}) {
      const axes = [...xyAxes.split(' '), kind === 'w' ? '1' : 'B'];
      const [cx, cy, cz] = axes.map((axis) => sizes[axis]);
      const dx = cx * CELL,
        dy = cy * CELL,
        dz = cz * CELL;
      const node = {
        op,
        head,
        name,
        kind,
        stage,
        x: right - dx,
        y,
        z: midZ - dz / 2,
        dx,
        dy,
        dz,
        cx,
        cy,
        cz,
        axes,
        formula,
        depMaps: inputs.map((input) => input.mapping),
        sources: inputs.map((input) => input.source),
        ...flags,
      };
      nodes.push(node);
      return node;
    };
  }
  return { nodes, section };
}

function layerNorm(newSection, prefix, src, y, left, shape, margin, stage = 'norm') {
  const add = newSection(stage);
  const affineRight = left - shape.T * CELL - margin;
  const mean = add(
    prefix + '.lnAgg1',
    'LN mean μ',
    'a',
    'T 1',
    left,
    y,
    0,
    'μ[t] = mean_c(x[t,c])',
    [from(src, 'xiz', 'mean')],
    { reduction: shape.C },
  );
  const invStd = add(
    prefix + '.lnAgg2',
    'LN inverse deviation',
    'a',
    'T 1',
    left,
    y + CELL,
    0,
    'r[t] = 1 / √(var_c(x[t,c]) + ε)',
    [from(src, 'xiz', 'variance')],
    { reduction: shape.C },
  );
  const outY = y + 2 * CELL + margin;
  const gain = add(prefix + '.lnSigma', 'LN gain γ', 'w', '1 C', affineRight, outY, 0, 'γ[c]');
  const bias = add(prefix + '.lnMu', 'LN bias β', 'w', '1 C', affineRight - CELL - margin, outY, 0, 'β[c]');
  const out = add(
    prefix + '.lnResid',
    'Layer norm',
    'i',
    'T C',
    left,
    outY,
    0,
    'u[t,c] = (x[t,c] − μ[t]) × r[t] × γ[c] + β[c]',
    [
      from(src, 'xyz', 'input'),
      from(mean, 'x0z', 'mean'),
      from(invStd, 'x0z', 'invStd'),
      from(gain, '0y0', 'scale'),
      from(bias, '0y0', 'bias'),
    ],
  );
  return out;
}

function attentionHead({
  section,
  shape,
  head,
  input,
  attentionRight,
  y,
  scoreY,
  valueY,
  margin,
  qkvGap,
  headWidth,
  packed,
}) {
  const { T, C, A, nHeads } = shape;
  const add = section('attention', head);
  const z = (head - (nHeads - 1) / 2) * headWidth;
  const weightRight = attentionRight - T * CELL - margin;
  const biasRight = weightRight - C * CELL - margin;
  const zStep = shape.B * CELL + qkvGap;
  const channels = [
    { key: 'q', name: 'Q', mid: z + zStep },
    { key: 'k', name: 'K', mid: z },
    { key: 'v', name: 'V', mid: z - zStep },
  ];
  const biases = channels.map(({ key, name, mid }) =>
    add(key + 'BiasBlock', name + ' bias', 'w', '1 A', biasRight, y, mid, 'b' + name + '[h,a]'),
  );
  let q, k, v;
  if (packed) {
    const weights = add(
      'qkvWeightBlock',
      'QKV weights',
      'w',
      'C 3A',
      weightRight,
      y,
      z,
      'stack(WQ[h], WK[h], WV[h])',
    );
    const inputs = [from(weights, 'iy0', 'weight'), from(input, 'xiz', 'input')];
    for (let part = 0; part < 3; part++) {
      inputs.push(
        from(biases[part], '0y0', 'bias', range('y', part * A, (part + 1) * A, [0, -part * A, 0])),
      );
    }
    q =
      k =
      v =
        add(
          'qkvBlock',
          'QKV vectors',
          'i',
          'T 3A',
          attentionRight,
          y,
          z,
          '[Q; K; V] = Wqkv · u + bqkv',
          inputs,
          { dotLen: C },
        );
  } else {
    [q, k, v] = channels.map(({ key, name, mid }, part) => {
      const weight = add(
        key + 'WeightBlock',
        name + ' weights',
        'w',
        'C A',
        weightRight,
        y,
        mid,
        'W' + name + '[h,a,c]',
      );
      return add(
        key + 'Block',
        name + ' vectors',
        'i',
        'T A',
        attentionRight,
        y,
        mid,
        name + '[t,a] = Σ_c W' + name + '[a,c] × u[t,c] + b' + name + '[a]',
        [from(weight, 'iy0', 'weight'), from(input, 'xiz', 'input'), from(biases[part], '0y0', 'bias')],
        { dotLen: C },
      );
    });
  }
  const statsRight = attentionRight - T * CELL - margin;
  const probRight = attentionRight - (T + 2) * CELL - 2 * margin;
  const scores = add(
    'attnMtx',
    'Causal attention scores',
    'i',
    'T T',
    attentionRight,
    scoreY,
    z,
    's[q,k] = Q[q] · K[k] / √A    (k ≤ q)',
    [from(q, 'yiz', 'query'), from(k, 'xiz', 'key', packed ? { offset: [0, A, 0] } : undefined)],
    { dotLen: A, causal: true },
  );
  const max = add(
    'attnMtxAgg2',
    'Attention row maximum',
    'a',
    '1 T',
    statsRight,
    scoreY,
    z,
    'm[q] = max_{k ≤ q} s[q,k]',
    [from(scores, 'iyz', 'max')],
    { reduction: T, limit: 'y+1' },
  );
  const inverse = add(
    'attnMtxAgg1',
    'Attention inverse exp-sum',
    'a',
    '1 T',
    statsRight - CELL,
    scoreY,
    z,
    'r[q] = 1 / Σ_{k ≤ q} exp(s[q,k] − m[q])',
    [from(scores, 'iyz', 'input'), from(max, '0yz', 'max')],
    { reduction: T, limit: 'y+1' },
  );
  const probs = add(
    'attnMtxSm',
    'Attention softmax',
    'i',
    'T T',
    probRight,
    scoreY,
    z,
    'p[q,k] = exp(s[q,k] − m[q]) × r[q]    (k ≤ q)',
    [from(scores, 'xyz', 'input'), from(inverse, '0yz', 'scale'), from(max, '0yz', 'max')],
    { causal: true },
  );
  return add(
    'vOutBlock',
    'Weighted values',
    'i',
    'T A',
    attentionRight,
    valueY,
    z,
    'o[t,a] = Σ_{k ≤ t} p[t,k] × V[k,a]',
    [
      from(v, 'iyz', 'value', packed ? { offset: [0, 2 * A, 0] } : undefined),
      from(probs, 'ixz', 'probability'),
    ],
    { dotLen: T, limit: 'x+1' },
  );
}

function decoderTemplate(shape, margin, left) {
  const { C, T, A, nHeads, nBlocks, B } = shape;
  const { nodes, section } = drawing(shape);
  const input = layerNorm(section, 'ln1', previousResidual, 0, left, shape, margin);
  const attentionRight = left - (T + 2) * CELL - 3 * margin;
  const weightRight = attentionRight - T * CELL - margin;
  const packed = nBlocks > ROWS_PER_COLUMN;
  const qkvGap = margin + (C * CELL) / 16;
  const interHeadGap = 3 * margin + (C * CELL) / 16;
  const headWidth = 3 * B * CELL + 2 * qkvGap + (packed ? 0 : interHeadGap);
  const scoreY = input.y + (packed ? 3 : 1) * A * CELL + margin;
  const valueY = scoreY + T * CELL + margin;
  const values = Array.from({ length: nHeads }, (_, head) =>
    attentionHead({
      section,
      shape,
      input,
      head,
      attentionRight,
      y: input.y,
      scoreY,
      valueY,
      margin,
      qkvGap,
      headWidth,
      packed,
    }),
  );

  const project = section('attention');
  const projY = Math.max(valueY + A * CELL + 2 * margin, input.y + C * CELL + margin);
  const projWeight = project(
    'projWeight',
    'Attention projection weights',
    'w',
    'C C',
    weightRight,
    projY,
    0,
    'WO[c,j]',
  );
  const projBias = project(
    'projBias',
    'Attention projection bias',
    'w',
    '1 C',
    weightRight - C * CELL - margin,
    projY,
    0,
    'bO[c]',
  );
  const joined = values.map((value, h) =>
    from(value, 'xiz', 'concat', range('i', h * A, (h + 1) * A, [0, -h * A, 0])),
  );
  const attentionOut = project(
    'attnOut',
    'Attention output',
    'i',
    'T C',
    attentionRight,
    projY,
    0,
    'a[t] = WO · concat(o₁[t], …, oH[t]) + bO',
    [from(projWeight, 'iy0', 'weight'), ...joined, from(projBias, '0y0', 'bias')],
    { dotLen: C },
  );
  const residualRight = (T * CELL) / 2;
  const attnResidual = project(
    'attnResidual',
    'Attention residual',
    'i',
    'T C',
    residualRight,
    projY,
    0,
    'r[t,c] = x[t,c] + a[t,c]',
    [from(previousResidual, 'xyz', 'skip'), from(attentionOut, 'xyz', 'add')],
  );

  const normalized = layerNorm(
    section,
    'ln2',
    attnResidual,
    projY + C * CELL + margin,
    left,
    shape,
    margin,
  );
  const ffn = section('ffn');
  const upY = normalized.y;
  const fcWeight = ffn(
    'mlpFcWeight',
    'MLP expansion weights',
    'w',
    '4C C',
    attentionRight,
    upY,
    0,
    'Wup[j,c]',
  );
  const fcBias = ffn(
    'mlpFcBias',
    'MLP expansion bias',
    'w',
    '4C 1',
    attentionRight,
    upY - CELL - margin,
    0,
    'bup[j]',
  );
  const fcY = upY + C * CELL + margin;
  const expanded = ffn(
    'mlpFc',
    'MLP expansion',
    'i',
    '4C T',
    attentionRight,
    fcY,
    0,
    'f[t,j] = Σ_c Wup[j,c] × u[t,c] + bup[j]',
    [from(fcWeight, 'xi0', 'weight'), from(normalized, 'yiz', 'input'), from(fcBias, 'x00', 'bias')],
    { dotLen: C },
  );
  const actY = fcY + T * CELL + margin;
  const gelu = ffn(
    'mlpAct',
    'GELU activation',
    'i',
    '4C T',
    attentionRight,
    actY,
    0,
    'g[t,j] = GELU(f[t,j])',
    [from(expanded, 'xyz', 'input')],
  );
  const downY = actY + T * CELL + margin;
  const downWeight = ffn(
    'mlpProjWeight',
    'MLP projection weights',
    'w',
    '4C C',
    attentionRight,
    downY,
    0,
    'Wdown[c,j]',
  );
  const downBias = ffn(
    'mlpProjBias',
    'MLP projection bias',
    'w',
    '1 C',
    attentionRight - 4 * C * CELL - margin,
    downY,
    0,
    'bdown[c]',
  );
  const down = ffn(
    'mlpResult',
    'MLP output',
    'i',
    'T C',
    attentionRight + margin + T * CELL,
    downY,
    0,
    'd[t,c] = Σ_j Wdown[c,j] × g[t,j] + bdown[c]',
    [from(downWeight, 'iy0', 'weight'), from(gelu, 'ixz', 'input'), from(downBias, '0y0', 'bias')],
    { dotLen: 4 * C },
  );
  ffn('mlpResidual', 'MLP residual', 'i', 'T C', residualRight, downY, 0, 'x′[t,c] = r[t,c] + d[t,c]', [
    from(attnResidual, 'xyz', 'skip'),
    from(down, 'xyz', 'add'),
  ]);
  return { nodes, pitch: downY + C * CELL + 3 * margin };
}

function endTensors(shape, margin, left, finalY, finalColumnX, id) {
  const { section, nodes } = drawing(shape);
  const { T, C, vocabSize, nBlocks } = shape;
  const start = section('embed');
  const residualRight = (T * CELL) / 2;
  const embedY = CELL + margin;
  const tokens = start('idxObj', 'Token IDs', 'i', 'T 1', residualRight, 0, 0, 'ids[t]');
  const tokensTable = start('tokEmbedObj', 'Token embedding', 'w', 'V C', left, embedY, 0, 'E[v,c]');
  const positions = start(
    'posEmbedObj',
    'Position embedding',
    'w',
    'T C',
    residualRight + margin + T * CELL,
    embedY,
    0,
    'P[t,c]',
  );
  start(
    'residual0',
    'Input embedding',
    'i',
    'T C',
    residualRight,
    embedY,
    0,
    'x[t,c] = E[ids[t],c] + P[t,c]',
    [from(tokensTable, 'ty0', 'gather'), from(positions, 'xy0', 'position'), from(tokens, 'x0z', 'index')],
  );

  const finalLeft = left + finalColumnX;
  const previous = { layer: nBlocks - 1, head: -1, op: 'mlpResidual' };
  const final = layerNorm(section, 'ln_f', previous, finalY, finalLeft, shape, margin, 'output');
  const read = section('output');
  const logitsY = final.y + C * CELL + margin;
  const tied = id.startsWith('gpt2-');
  // Token and output tables are two views of one matrix in GPT-2's implementation.
  const weight = read(
    'lmHeadWeight',
    'LM head weights',
    'w',
    'C V',
    finalLeft - T * CELL - margin,
    logitsY,
    0,
    tied ? 'Wlm[v,c] = E[v,c]' : 'Wlm[v,c]',
    tied ? [from(tokensTable, 'yx0', 'tied')] : [],
  );
  const logits = read(
    'logits',
    'Vocabulary logits',
    'i',
    'T V',
    finalLeft,
    logitsY,
    0,
    'l[t,v] = Σ_c Wlm[v,c] × u[t,c]',
    [from(weight, 'iy0', 'weight'), from(final, 'xiz', 'input')],
    { dotLen: C },
  );
  const aggY = logitsY + vocabSize * CELL + margin;
  const max = read(
    'logitsAgg2',
    'Vocabulary maximum',
    'a',
    'T 1',
    finalLeft,
    aggY,
    0,
    'm[t] = max_v l[t,v]',
    [from(logits, 'xiz', 'max')],
    { reduction: vocabSize },
  );
  const inverse = read(
    'logitsAgg1',
    'Vocabulary inverse exp-sum',
    'a',
    'T 1',
    finalLeft,
    aggY + CELL,
    0,
    'r[t] = 1 / Σ_v exp(l[t,v] − m[t])',
    [from(logits, 'xiz', 'input'), from(max, 'x0z', 'max')],
    { reduction: vocabSize },
  );
  read(
    'logitsSoftmax',
    'Next-token probabilities',
    'i',
    'T V',
    finalLeft,
    aggY + 2 * CELL + margin,
    0,
    'p[t,v] = exp(l[t,v] − m[t]) × r[t]',
    [from(logits, 'xyz', 'input'), from(inverse, 'x0z', 'scale'), from(max, 'x0z', 'max')],
  );
  return nodes;
}

function measure(nodes) {
  const bounds = [Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity];
  for (const { x, y, z, dx, dy, dz } of nodes) {
    bounds[0] = Math.min(bounds[0], x);
    bounds[1] = Math.min(bounds[1], y);
    bounds[2] = Math.min(bounds[2], z);
    bounds[3] = Math.max(bounds[3], x + dx);
    bounds[4] = Math.max(bounds[4], y + dy);
    bounds[5] = Math.max(bounds[5], z + dz);
  }
  return bounds;
}

function instantiate(node, layer, dx, dy) {
  const { sources, ...geometry } = node;
  const deps = sources.map((src) => {
    if (src !== previousResidual) return nodeId(src.layer ?? layer, src.head, src.op);
    return layer === 0 ? nodeId(-1, -1, 'residual0') : nodeId(layer - 1, -1, 'mlpResidual');
  });
  return {
    ...geometry,
    id: nodeId(layer, node.head, node.op),
    layer,
    x: node.x + dx,
    y: node.y + dy,
    deps,
  };
}

/** Complete classic geometry. Layers share one tensor template and instantiate on demand. */
export function classicLayout(id) {
  if (layouts.has(id)) return layouts.get(id);
  if (!Object.hasOwn(shapes, id)) throw new RangeError('Unknown classic model: ' + id);
  const config = shapes[id];
  const shape = { ...config, A: config.C / config.nHeads, B: 1, tokens: config.T };
  const { T, C, nBlocks } = shape;
  const margin = Math.max(12, C / 10);
  const left = (-T * CELL) / 2 - margin;
  const { nodes: template, pitch } = decoderTemplate(shape, margin, left);
  const firstY = CELL + C * CELL + 6 * margin;
  const columnWidth = 14 * C * CELL + 2 * margin;
  const localBounds = measure(template);
  const cache = new Map();
  const layers = Array.from({ length: nBlocks }, (_, index) => {
    const x = Math.floor(index / ROWS_PER_COLUMN) * columnWidth;
    const rowY = (index % ROWS_PER_COLUMN) * pitch;
    const y = rowY + firstY;
    const bounds = localBounds.map((value, i) => value + [x, y, 0][i % 3]);
    return {
      index,
      bounds,
      offset: [x, rowY, 0],
      nodeCount: template.length,
      nodes() {
        if (!cache.has(index)) {
          if (cache.size >= 3) cache.delete(cache.keys().next().value);
          cache.set(
            index,
            template.map((node) => instantiate(node, index, x, y)),
          );
        }
        return cache.get(index);
      },
    };
  });
  const [lastX, lastY] = layers[nBlocks - 1].offset;
  const root = endTensors(shape, margin, left, firstY + lastY + pitch, lastX, id).map((node) =>
    instantiate(node, -1, 0, 0),
  );
  const bounds = measure(root);
  for (const layer of layers) {
    for (let i = 0; i < 3; i++) {
      bounds[i] = Math.min(bounds[i], layer.bounds[i]);
      bounds[i + 3] = Math.max(bounds[i + 3], layer.bounds[i + 3]);
    }
  }
  const result = {
    id,
    shape,
    cell: CELL,
    bounds,
    root,
    layers,
    repeatedLayerGeometry: true,
    nodeCount: root.length + nBlocks * template.length,
  };
  layouts.set(id, result);
  return result;
}
