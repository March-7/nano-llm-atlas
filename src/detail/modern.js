// Small, deterministic teaching networks. Weights are generated independently of model checkpoints.
// Qwen recurrent equations use https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py

function makeId(layer, head, op) {
  return `L${layer}.H${head}.${op}`;
}

function label(op, name, row, lane = 1, extra = {}) {
  return { op, name, row, lane, ...extra };
}

function link(tensor, map, role, options = {}) {
  return { id: typeof tensor === 'string' ? tensor : tensor.id, mapping: { map, role, ...options } };
}

function noise(key) {
  let seed = 2166136261;
  for (const c of key) seed = Math.imul(seed ^ c.charCodeAt(0), 16777619);
  return () => {
    seed = (Math.imul(1664525, seed) + 1013904223) >>> 0;
    return (seed + 0.5) / 4294967296;
  };
}

function number(v) {
  return v === -Infinity ? '−∞' : Number(v.toPrecision(6)).toString();
}

function sum(n, term) {
  let result = 0;
  for (let i = 0; i < n; i++) result += term(i);
  return result;
}

function dot(n, a, b) {
  return sum(n, (i) => a(i) * b(i));
}

function dotText(n, a, b, result, divisor = 1) {
  const pair = (i) => `${number(a(i))} × ${number(b(i))}`;
  const body = n === 1 ? pair(0) : `${pair(0)}${n === 2 ? ' + ' : ' + … + '}${pair(n - 1)}`;
  return `n = ${n}   (${body})${divisor === 1 ? '' : ' / ' + number(divisor)} = ${number(result)}`;
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function silu(x) {
  return x * sigmoid(x);
}

function bounds(nodes) {
  const result = [Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity];
  for (const n of nodes) {
    const xyz = [n.x, n.y, n.z];
    const size = [n.dx, n.dy, n.dz];
    for (let axis = 0; axis < 3; axis++) {
      result[axis] = Math.min(result[axis], xyz[axis]);
      result[axis + 3] = Math.max(result[axis + 3], xyz[axis] + size[axis]);
    }
  }
  return result;
}

class TeachingGraph {
  constructor(id) {
    this.id = id;
    const qwen = id === 'qwen';
    this.shape = {
      T: 8,
      B: 1,
      A: 8,
      C: qwen ? 40 : 32,
      nHeads: qwen ? 6 : 4,
      numKVHeads: 1,
      nBlocks: qwen ? 4 : 1,
      intermediate: qwen ? 136 : 112,
      vocabSize: 16,
      tokens: 8,
      context: 8,
    };
    this.tensors = new Map();
    this.layer = -1;
    this.stage = 'embed';
  }

  tensor(meta, nx, ny, calc, formula, inputs = [], substitute) {
    const {
      op,
      name,
      row,
      lane,
      head = -1,
      plane = 0,
      depth = 1,
      kind = 'i',
      axes = ['T', 'C', 'B'],
      ...flags
    } = meta;
    const { layer, stage } = this;
    const id = makeId(layer, head, op);
    if (this.tensors.has(id)) throw new Error(`Duplicate tensor ${id}`);
    const cell = 1.5;
    const dx = nx * cell,
      dy = ny * cell,
      dz = depth * cell;
    const right = [12, -20, -62, -145, -230][lane];
    const node = {
      id,
      op,
      name,
      stage,
      kind,
      head,
      layer,
      cx: nx,
      cy: ny,
      cz: depth,
      axes,
      dx,
      dy,
      dz,
      x: right - dx,
      y: 0,
      z: plane - dz / 2,
      formula,
      deps: inputs.map((d) => d.id),
      depMaps: inputs.map((d) => d.mapping),
      ...flags,
    };
    const data = new Float64Array(nx * ny * depth);
    for (let z = 0; z < depth; z++)
      for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) data[(z * ny + y) * nx + x] = calc(x, y, z);
    const tensor = {
      id,
      node,
      row,
      nx,
      ny,
      nz: depth,
      v: (x, y = 0, z = 0) => data[(z * ny + y) * nx + x],
      substitute,
    };
    this.tensors.set(id, tensor);
    return tensor;
  }

  weight(meta, nx, ny, axes, scale = Math.sqrt(3 / nx), offset = 0) {
    const random = noise(`${this.id}/${this.layer}/${meta.head ?? -1}/${meta.op}`);
    return this.tensor(
      { ...meta, kind: 'w', axes: [...axes, '1'] },
      nx,
      ny,
      () => offset + (2 * random() - 1) * scale,
      'θ = fixed teaching weight',
      [],
      (x, y, _z, value) => `${meta.op}[${y},${x}] = ${number(value)}`,
    );
  }

  project(meta, source, size, outAxis, weightOp = meta.op + 'Weight') {
    const width = source.ny;
    const W = this.weight({ ...meta, name: meta.name + ' weights', op: weightOp, lane: 2 }, width, size, [
      source.node.axes[1],
      outAxis,
    ]);
    return this.tensor(
      { ...meta, axes: ['T', outAxis, 'B'], dotLen: width },
      this.shape.T,
      size,
      (t, d) =>
        dot(
          width,
          (i) => W.v(i, d),
          (i) => source.v(t, i),
        ),
      'y[t,d] = Σᵢ W[d,i] × x[t,i]',
      [link(W, 'iy0', 'weight'), link(source, 'xiz', 'input')],
      (t, d, _z, value) =>
        dotText(
          width,
          (i) => W.v(i, d),
          (i) => source.v(t, i),
          value,
        ),
    );
  }

  norm(meta, input, eps, deltaGain = this.id === 'qwen', sharedGain) {
    const D = input.ny;
    const channel = input.node.axes[1];
    const gain =
      sharedGain ??
      this.weight(
        { ...meta, op: meta.op + '.gain', name: 'RMS gain', lane: 2 },
        1,
        D,
        ['1', channel],
        0.1,
        deltaGain ? 0 : 1,
      );
    const inv = this.tensor(
      {
        ...meta,
        op: meta.op + '.invRms',
        name: 'Inverse RMS',
        lane: 3,
        kind: 'a',
        axes: ['T', '1', 'B'],
        reduction: D,
      },
      this.shape.T,
      1,
      (t) => 1 / Math.sqrt(sum(D, (i) => input.v(t, i) ** 2) / D + eps),
      'r[t] = 1 / √(meanᵢ x[t,i]² + ε)',
      [link(input, 'xiz', 'squares')],
      (t, _y, _z, value) =>
        `n = ${D}   1 / √(${number(sum(D, (i) => input.v(t, i) ** 2) / D)} + ${eps}) = ${number(value)}`,
    );
    return this.tensor(
      { ...meta, axes: ['T', channel, 'B'] },
      this.shape.T,
      D,
      (t, d) => input.v(t, d) * inv.v(t) * (gain.v(0, d) + (deltaGain ? 1 : 0)),
      deltaGain ? 'u[t,d] = x[t,d] × r[t] × (1 + w[d])' : 'u[t,d] = x[t,d] × r[t] × γ[d]',
      [link(input, 'xyz', 'input'), link(inv, 'x0z', 'invRms'), link(gain, '0y0', 'gain')],
      (t, d, _z, value) =>
        `${number(input.v(t, d))} × ${number(inv.v(t))} × ${number(gain.v(0, d) + (deltaGain ? 1 : 0))} = ${number(value)}`,
    );
  }

  residual(meta, input, update) {
    return this.tensor(
      meta,
      this.shape.T,
      this.shape.C,
      (t, d) => input.v(t, d) + update.v(t, d),
      'r[t,d] = x[t,d] + Δ[t,d]',
      [link(input, 'xyz', 'skip'), link(update, 'xyz', 'add')],
      (t, d, _z, value) => `${number(input.v(t, d))} + ${number(update.v(t, d))} = ${number(value)}`,
    );
  }

  finish(scope) {
    const all = [...this.tensors.values()];
    const bands = new Map();
    for (const { row, node } of all) bands.set(row, Math.max(bands.get(row) ?? 0, node.dy));
    let y = 0;
    for (const row of [...bands.keys()].sort((a, b) => a - b)) {
      const height = bands.get(row);
      bands.set(row, y);
      y += height + 14;
    }
    const nodes = all.map((r) => {
      r.node.y = bands.get(r.row);
      return r.node;
    });
    const root = nodes.filter((n) => n.layer === -1);
    const layers = Array.from({ length: this.shape.nBlocks }, (_, index) => {
      const group = nodes.filter((n) => n.layer === index);
      return {
        index,
        nodeCount: group.length,
        nodes: () => group,
        bounds: bounds(group),
        offset: [0, 0, 0],
      };
    });
    const layout = {
      id: this.id,
      shape: this.shape,
      cell: 1.5,
      bounds: bounds(nodes),
      root,
      layers,
      nodeCount: nodes.length,
      scope,
    };
    const table = this.tensors;
    const locate = (n, x, y, z) => {
      if (
        !n ||
        !Number.isInteger(x) ||
        !Number.isInteger(y) ||
        !Number.isInteger(z) ||
        x < 0 ||
        y < 0 ||
        z < 0
      )
        return null;
      const t = table.get(n.id ?? makeId(n.layer ?? -1, n.head ?? -1, n.op));
      return t && x < t.nx && y < t.ny && z < t.nz ? t : null;
    };
    const sample = {
      value(n, x, y, z = 0) {
        return locate(n, x, y, z)?.v(x, y, z) ?? null;
      },
      explain(n, x, y, z = 0) {
        const t = locate(n, x, y, z);
        if (!t) return { formula: '∅', substitution: `(${x}, ${y}, ${z})` };
        const v = t.v(x, y, z);
        return {
          formula: t.node.formula,
          substitution: t.substitute(x, y, z, v),
          source: 'Fixed teaching example · ' + t.id,
        };
      },
    };
    return { layout, sample };
  }
}

function feedForward(b, input, row) {
  const { T, C, intermediate: F } = b.shape;
  b.stage = 'ffn';
  const u = b.norm(label('ln2.lnResid', 'Pre-FFN RMSNorm', row), input, b.id === 'qwen' ? 1e-6 : 1e-5);
  const gate = b.project(label('mlpGate', 'Gate projection', row + 1, 1, { plane: 15 }), u, F, 'F');
  const up = b.project(label('mlpUp', 'Up projection', row + 1, 1, { plane: -15 }), u, F, 'F');
  const act = b.tensor(
    label('mlpAct', 'SwiGLU', row + 2, 1, { axes: ['T', 'F', 'B'] }),
    T,
    F,
    (t, j) => silu(gate.v(t, j)) * up.v(t, j),
    'a[t,j] = SiLU(gate[t,j]) × up[t,j]',
    [link(gate, 'xyz', 'gate'), link(up, 'xyz', 'up')],
    (t, j, _z, value) => `SiLU(${number(gate.v(t, j))}) × ${number(up.v(t, j))} = ${number(value)}`,
  );
  const down = b.project(label('mlpResult', 'Down projection', row + 3), act, C, 'C');
  return b.residual(label('mlpResidual', 'FFN residual', row + 3, 0), input, down);
}

function concatHeads(b, op, heads, row) {
  const D = heads[0].ny;
  const span = (h) => ({ span: { axis: 'y', start: h * D, end: (h + 1) * D }, offset: [0, -h * D, 0] });
  return b.tensor(
    label(op, 'Concatenate heads', row, 1, { axes: ['T', 'H·A', 'B'] }),
    b.shape.T,
    D * heads.length,
    (t, j) => heads[Math.floor(j / D)].v(t, j % D),
    'v[t,h·A + a] = head[h][t,a]',
    heads.map((r, h) => link(r, 'xyz', 'concat', span(h))),
    (t, j, _z, v) => `h = ${Math.floor(j / D)}   a = ${j % D}   v[${t},${j}] = ${number(v)}`,
  );
}

function rotary(b, meta, source) {
  const qwen = b.id === 'qwen';
  const D = source.ny,
    R = qwen ? 2 : D;
  const frequencies = Array.from({ length: R / 2 }, (_, i) => {
    let f = (qwen ? 1e7 : 5e5) ** ((-2 * i) / R);
    if (!qwen) {
      const wavelength = (2 * Math.PI) / f;
      const blend = Math.max(0, Math.min(1, (8192 / wavelength - 1) / 3));
      f = ((1 - blend) * f) / 8 + blend * f;
    }
    return f;
  });
  const angle = (t, d) => t * frequencies[d % (R / 2)];
  const other = (d) => (d + R / 2) % R;
  const sign = (d) => (d < R / 2 ? -1 : 1);
  const deps = [link(source, 'xyz', 'input')];
  for (let half = 0; half < 2; half++)
    deps.push(
      link(source, 'xyz', 'rotate', {
        span: { axis: 'y', start: (half * R) / 2, end: ((half + 1) * R) / 2 },
        offset: [0, half === 0 ? R / 2 : -R / 2, 0],
      }),
    );
  return b.tensor(
    { ...meta, axes: ['T', 'A', 'B'] },
    b.shape.T,
    D,
    (t, d) =>
      d >= R
        ? source.v(t, d)
        : source.v(t, d) * Math.cos(angle(t, d)) + sign(d) * source.v(t, other(d)) * Math.sin(angle(t, d)),
    `RoPE(x,t) on ${R}/${D} channels`,
    deps,
    (t, d, _z, v) =>
      d >= R
        ? `d = ${d} ≥ ${R}   y = x = ${number(v)}`
        : `${number(source.v(t, d))} × cos(${number(angle(t, d))}) ${sign(d) < 0 ? '−' : '+'} ${number(source.v(t, other(d)))} × sin(${number(angle(t, d))}) = ${number(v)}`,
  );
}

function groupedAttention(b, source, row) {
  b.stage = 'attention';
  const { T, A, nHeads, C } = b.shape;
  const qwen = b.id === 'qwen';
  const outer = (nHeads / 2 + 1) * 23;
  const sharedK = b.project(label('kBlock', 'Shared key', row, 1, { plane: -outer }), source, A, 'A');
  const v = b.project(label('vBlock', 'Shared value', row, 1, { plane: outer }), source, A, 'A');
  const keyNorm = qwen
    ? b.norm(label('kNorm', 'Key RMSNorm', row + 1, 1, { plane: -outer }), sharedK, 1e-6)
    : sharedK;
  const k = rotary(b, label('kRotary', 'Key RoPE', row + 2, 1, { plane: -outer }), keyNorm);
  const queryGain = qwen
    ? b.weight(label('qNormGain', 'Shared query RMS gain offset', row + 1, 2), 1, A, ['1', 'A'], 0.1)
    : null;
  const outputs = Array.from({ length: nHeads }, (_, h) => {
    const plane = (h - (nHeads - 1) / 2) * 23;
    const meta = (op, name, r, lane = 1, more = {}) =>
      label(op, name, r, lane, { head: h, plane, ...more });
    const rawQ = b.project(meta('qBlock', 'Query projection', row), source, A, 'A');
    const nq = qwen ? b.norm(meta('qNorm', 'Query RMSNorm', row + 1), rawQ, 1e-6, true, queryGain) : rawQ;
    const q = rotary(b, meta('qRotary', 'Query RoPE', row + 2), nq);
    const scores = b.tensor(
      meta('attnMtx', 'Causal GQA scores', row + 3, 1, { axes: ['T', 'T', 'B'], causal: true, dotLen: A }),
      T,
      T,
      (s, t) =>
        s > t
          ? -Infinity
          : dot(
              A,
              (i) => q.v(t, i),
              (i) => k.v(s, i),
            ) / Math.sqrt(A),
      's[t,k] = qₕ[t] · kshared[k] / √A    (k ≤ t)',
      [link(q, 'yi0', 'query'), link(k, 'xi0', 'sharedKey')],
      (s, t, _z, v) =>
        s > t
          ? `${s} > ${t} ⇒ −∞`
          : dotText(
              A,
              (i) => q.v(t, i),
              (i) => k.v(s, i),
              v,
              Math.sqrt(A),
            ),
    );
    const maxima = Array.from({ length: T }, (_, t) =>
      Math.max(...Array.from({ length: t + 1 }, (_, s) => scores.v(s, t))),
    );
    const inv = maxima.map((m, t) => 1 / sum(t + 1, (s) => Math.exp(scores.v(s, t) - m)));
    const probs = b.tensor(
      meta('attnMtxSm', 'Causal GQA probabilities', row + 4, 1, {
        axes: ['T', 'T', 'B'],
        causal: true,
        reduction: T,
        limit: 'y+1',
      }),
      T,
      T,
      (s, t) => (s > t ? 0 : Math.exp(scores.v(s, t) - maxima[t]) * inv[t]),
      'p[t,k] = exp(s[t,k] − m[t]) / Σⱼ≤ₜ exp(s[t,j] − m[t])',
      [link(scores, 'xyz', 'score'), link(scores, 'iyz', 'normalizer')],
      (s, t, _z, v) =>
        s > t
          ? `${s} > ${t} ⇒ 0`
          : `n = ${t + 1}   exp(${number(scores.v(s, t))} − ${number(maxima[t])}) × ${number(inv[t])} = ${number(v)}`,
    );
    const mixed = b.tensor(
      meta('vOutBlock', 'Weighted shared values', row + 5, 1, {
        axes: ['T', 'A', 'B'],
        dotLen: T,
        limit: 'x+1',
      }),
      T,
      A,
      (t, a) =>
        dot(
          t + 1,
          (i) => probs.v(i, t),
          (i) => v.v(i, a),
        ),
      'o[t,a] = Σₖ≤ₜ p[t,k] × vshared[k,a]',
      [link(probs, 'ix0', 'probability'), link(v, 'iy0', 'sharedValue')],
      (t, a, _z, out) =>
        dotText(
          t + 1,
          (i) => probs.v(i, t),
          (i) => v.v(i, a),
          out,
        ),
    );
    if (!qwen) return mixed;
    const gate = b.project(
      meta('attnGate', 'Output gate projection', row + 5, 1, { plane: plane + 8 }),
      source,
      A,
      'A',
    );
    return b.tensor(
      meta('attnGated', 'Sigmoid output gate', row + 6, 1, { axes: ['T', 'A', 'B'] }),
      T,
      A,
      (t, a) => mixed.v(t, a) * sigmoid(gate.v(t, a)),
      'g[t,a] = o[t,a] × sigmoid(z[t,a])',
      [link(mixed, 'xyz', 'output'), link(gate, 'xyz', 'gate')],
      (t, a, _z, value) => `${number(mixed.v(t, a))} × sigmoid(${number(gate.v(t, a))}) = ${number(value)}`,
    );
  });
  const joined = concatHeads(b, 'attnConcat', outputs, row + 7);
  return b.project(label('attnOut', 'Attention output projection', row + 8), joined, C, 'C', 'projWeight');
}

function deltaRecurrence(T, D, key, value, decay, beta, head) {
  const states = new Float64Array(T * D * D);
  const decayed = new Float64Array(states.length);
  const memory = new Float64Array(T * D),
    write = new Float64Array(T * D);
  const index = (a, d, t) => (t * D + d) * D + a;
  for (let t = 0; t < T; t++) {
    for (let d = 0; d < D; d++) {
      for (let a = 0; a < D; a++) {
        decayed[index(a, d, t)] = t ? decay.v(t, head) * states[index(a, d, t - 1)] : 0;
      }
      memory[t * D + d] = dot(
        D,
        (a) => key.v(t, a),
        (a) => decayed[index(a, d, t)],
      );
      write[t * D + d] = beta.v(t, head) * (value.v(t, d) - memory[t * D + d]);
      for (let a = 0; a < D; a++)
        states[index(a, d, t)] = decayed[index(a, d, t)] + key.v(t, a) * write[t * D + d];
    }
  }
  return {
    state: (a, d, t) => states[index(a, d, t)],
    decayed: (a, d, t) => decayed[index(a, d, t)],
    memory: (t, d) => memory[t * D + d],
    write: (t, d) => write[t * D + d],
  };
}

function deltaAttention(b, source, row) {
  b.stage = 'delta';
  const { T, A: D, C } = b.shape;
  const H = 3,
    K = 4,
    packedWidth = (2 + H) * D;
  const qkv = b.project(label('deltaProjected', 'Packed Q / K / V', row), source, packedWidth, 'QKV');
  const wConv = b.weight(label('deltaKernel', 'Causal depthwise kernel', row + 1, 2), K, packedWidth, [
    'K',
    'QKV',
  ]);
  const convolve = (t, j) => sum(Math.min(K, t + 1), (s) => wConv.v(K - 1 - s, j) * qkv.v(t - s, j));
  const taps = Array.from({ length: K }, (_, s) =>
    link(qkv, 'xyz', 'tap', {
      span: { axis: 'x', start: s, end: T },
      offset: [-s, 0, 0],
    }),
  );
  const conv = b.tensor(
    label('deltaConv', 'Depthwise causal convolution + SiLU', row + 1, 1, {
      axes: ['T', 'QKV', 'B'],
      reduction: K,
    }),
    T,
    packedWidth,
    (t, j) => silu(convolve(t, j)),
    'u[t,j] = SiLU(Σᵣ w[j,K−1−r] × p[t−r,j])',
    [link(wConv, 'iy0', 'kernel'), ...taps],
    (t, j, _z, value) =>
      `K = ${K}   n = ${Math.min(K, t + 1)}   SiLU(${number(convolve(t, j))}) = ${number(value)}`,
  );
  function direction(op, name, offset, divisor, plane) {
    const norms = Array.from({ length: T }, (_, t) =>
      Math.sqrt(sum(D, (d) => conv.v(t, d + offset) ** 2) + 1e-6),
    );
    return b.tensor(
      label(op, name, row + 2, 1, { plane, axes: ['T', 'A', 'B'], reduction: D }),
      T,
      D,
      (t, a) => conv.v(t, a + offset) / (norms[t] * divisor),
      divisor === 1 ? 'k = u / √(Σ u² + ε)' : 'q = u / (√(Σ u² + ε) × √A)',
      [
        link(conv, 'xyz', 'input', { offset: [0, offset, 0] }),
        link(conv, 'xi0', 'norm', { offset: [0, offset, 0] }),
      ],
      (t, a, _z, v) =>
        `${number(conv.v(t, a + offset))} / (${number(norms[t])} × ${number(divisor)}) = ${number(v)}`,
    );
  }
  const q = direction('deltaQuery', 'Shared normalized query', 0, Math.sqrt(D), -54);
  const k = direction('deltaKey', 'Shared normalized key', D, 1, 54);
  const rawA = b.project(
    label('deltaA', 'Time-step projection', row + 2, 1, { plane: -25 }),
    source,
    H,
    'Hv',
  );
  const rawB = b.project(
    label('deltaB', 'Write-gate projection', row + 2, 1, { plane: 25 }),
    source,
    H,
    'Hv',
  );
  const logA = b.weight(
    label('deltaLogA', 'Log decay rate', row + 3, 2, { plane: -16 }),
    1,
    H,
    ['1', 'Hv'],
    0.4,
    -0.2,
  );
  const dt = b.weight(
    label('deltaTimeBias', 'Time-step bias', row + 3, 2, { plane: 16 }),
    1,
    H,
    ['1', 'Hv'],
    0.3,
    -0.4,
  );
  const rate = (t, h) => -Math.exp(logA.v(0, h)) * Math.log1p(Math.exp(rawA.v(t, h) + dt.v(0, h)));
  const decay = b.tensor(
    label('deltaDecay', 'State retention', row + 3, 1, { axes: ['T', 'Hv', 'B'], plane: -17 }),
    T,
    H,
    (t, h) => Math.exp(rate(t, h)),
    'ρ = exp(−exp(Alog) × softplus(a + Δt))',
    [link(rawA, 'xyz', 'time'), link(logA, '0y0', 'rate'), link(dt, '0y0', 'bias')],
    (t, h, _z, value) =>
      `exp(−exp(${number(logA.v(0, h))}) × softplus(${number(rawA.v(t, h))} + ${number(dt.v(0, h))})) = ${number(value)}`,
  );
  const beta = b.tensor(
    label('deltaBeta', 'Write strength', row + 3, 1, { axes: ['T', 'Hv', 'B'], plane: 17 }),
    T,
    H,
    (t, h) => sigmoid(rawB.v(t, h)),
    'β = sigmoid(b)',
    [link(rawB, 'xyz', 'input')],
    (t, h, _z, value) => `sigmoid(${number(rawB.v(t, h))}) = ${number(value)}`,
  );
  const zGate = b.project(
    label('deltaZ', 'Read gate projection', row + 3, 1, { plane: 54 }),
    source,
    H * D,
    'Hv·A',
  );
  const readGain = b.weight(
    label('deltaReadGain', 'Shared gated RMS gain', row + 10, 2),
    1,
    D,
    ['1', 'A'],
    0.1,
    1,
  );
  const heads = Array.from({ length: H }, (_, h) => {
    const plane = (h - 1) * 29;
    const meta = (op, name, r, more = {}) =>
      label(op, name, r, 1, { head: h, plane, axes: ['T', 'A', 'B'], ...more });
    const values = b.tensor(
      meta('deltaValue', 'Value channels', row + 4),
      T,
      D,
      (t, d) => conv.v(t, (2 + h) * D + d),
      'vₕ[t,d] = u[t,(2+h)·A+d]',
      [link(conv, 'xyz', 'value', { offset: [0, (2 + h) * D, 0] })],
      (t, d, _z, value) => `u[${t},${(2 + h) * D + d}] = ${number(value)}`,
    );
    const history = deltaRecurrence(T, D, k, values, decay, beta, h);
    const volume = { depth: T, axes: ['Ak', 'Av', 'T'] };
    const previous = b.tensor(
      meta('deltaPrevious', 'Incoming state', row + 4, { ...volume, lane: 3 }),
      D,
      D,
      (a, d, t) => (t ? history.state(a, d, t - 1) : 0),
      'Sprev[t] = S[t−1]    Sprev[0] = 0',
      [
        link(makeId(b.layer, h, 'deltaState'), 'xyz', 'previous', {
          span: { axis: 'z', start: 1, end: T },
          offset: [0, 0, -1],
        }),
      ],
      (a, d, t, v) => (t ? `S[${t - 1},${a},${d}] = ${number(v)}` : `Sinit[${a},${d}] = 0`),
    );
    const shrunk = b.tensor(
      meta('deltaRetained', 'Retained state', row + 5, volume),
      D,
      D,
      history.decayed,
      'S̄[t,a,d] = ρ[t,h] × Sprev[t,a,d]',
      [link(decay, 'z00', 'retention', { offset: [0, h, 0] }), link(previous, 'xyz', 'state')],
      (a, d, t, v) => `${number(decay.v(t, h))} × ${number(previous.v(a, d, t))} = ${number(v)}`,
    );
    const memory = b.tensor(
      meta('deltaMemory', 'Value recalled by key', row + 6, { dotLen: D }),
      T,
      D,
      history.memory,
      'm[t,d] = Σₐ k[t,a] × S̄[t,a,d]',
      [link(k, 'xi0', 'key'), link(shrunk, 'iyx', 'state')],
      (t, d, _z, v) =>
        dotText(
          D,
          (a) => k.v(t, a),
          (a) => shrunk.v(a, d, t),
          v,
        ),
    );
    const write = b.tensor(
      meta('deltaWrite', 'Value correction', row + 7),
      T,
      D,
      history.write,
      'Δ[t,d] = β[t,h] × (v[t,d] − m[t,d])',
      [
        link(beta, 'x00', 'gate', { offset: [0, h, 0] }),
        link(values, 'xyz', 'value'),
        link(memory, 'xyz', 'retrieved'),
      ],
      (t, d, _z, v) =>
        `${number(beta.v(t, h))} × (${number(values.v(t, d))} − ${number(memory.v(t, d))}) = ${number(v)}`,
    );
    const state = b.tensor(
      meta('deltaState', 'Gated DeltaNet state', row + 8, volume),
      D,
      D,
      history.state,
      'S[t,a,d] = S̄[t,a,d] + k[t,a] × Δ[t,d]',
      [link(shrunk, 'xyz', 'state'), link(k, 'zx0', 'key'), link(write, 'zy0', 'correction')],
      (a, d, t, v) =>
        `${number(shrunk.v(a, d, t))} + ${number(k.v(t, a))} × ${number(write.v(t, d))} = ${number(v)}`,
    );
    const read = b.tensor(
      meta('deltaRead', 'Recurrent readout', row + 9, { dotLen: D }),
      T,
      D,
      (t, d) =>
        dot(
          D,
          (a) => q.v(t, a),
          (a) => state.v(a, d, t),
        ),
      'o[t,d] = Σₐ q[t,a] × S[t,a,d]',
      [link(q, 'xi0', 'query'), link(state, 'iyx', 'state')],
      (t, d, _z, v) =>
        dotText(
          D,
          (a) => q.v(t, a),
          (a) => state.v(a, d, t),
          v,
        ),
    );
    const normalized = b.norm(
      meta('deltaReadNorm', 'Readout RMSNorm', row + 10),
      read,
      1e-6,
      false,
      readGain,
    );
    return b.tensor(
      meta('deltaGated', 'SiLU-gated readout', row + 11),
      T,
      D,
      (t, d) => normalized.v(t, d) * silu(zGate.v(t, h * D + d)),
      'o′[t,d] = RMSNorm(o[t])[d] × SiLU(z[t,h,d])',
      [link(normalized, 'xyz', 'value'), link(zGate, 'xyz', 'gate', { offset: [0, h * D, 0] })],
      (t, d, _z, v) =>
        `${number(normalized.v(t, d))} × SiLU(${number(zGate.v(t, h * D + d))}) = ${number(v)}`,
    );
  });
  const joined = concatHeads(b, 'deltaConcat', heads, row + 12);
  return b.project(label('deltaOut', 'Recurrent output projection', row + 13), joined, C, 'C');
}

const examples = new Map();

export function modernExample(id) {
  if (id !== 'llama' && id !== 'qwen') throw new RangeError('Unknown teaching architecture: ' + id);
  if (examples.has(id)) return examples.get(id);
  const b = new TeachingGraph(id);
  const { T, C, nBlocks, vocabSize: V } = b.shape;
  const tokens = [3, 1, 4, 1, 5, 9, 2, 6];
  const ids = b.tensor(
    label('idxObj', 'Teaching token IDs', 0, 0, { axes: ['T', '1', 'B'] }),
    T,
    1,
    (t) => tokens[t],
    'ids[t]',
    [],
    (t, _y, _z, value) => `ids[${t}] = ${value}`,
  );
  const embedding = b.weight(label('tokEmbedObj', 'Token embedding', 1, 2), V, C, ['V', 'C'], 0.75);
  let residual = b.tensor(
    label('embed', 'Input embedding', 1, 0),
    T,
    C,
    (t, d) => embedding.v(ids.v(t), d),
    'x[t,d] = E[ids[t],d]',
    [link(embedding, 'ty0', 'gather'), link(ids, 'x0z', 'index')],
    (t, d, _z, v) => `E[${tokens[t]},${d}] = ${number(v)}`,
  );
  let row = 2;
  for (let l = 0; l < nBlocks; l++) {
    b.layer = l;
    const delta = id === 'qwen' && l < 3;
    b.stage = delta ? 'delta' : id === 'llama' ? 'norm' : 'attention';
    const pre = b.norm(
      label('ln1.lnResid', 'Pre-mixer RMSNorm', row),
      residual,
      id === 'llama' ? 1e-5 : 1e-6,
    );
    const mixed = delta ? deltaAttention(b, pre, row + 1) : groupedAttention(b, pre, row + 1);
    const mixerEnd = row + (delta ? 14 : 9);
    const attended = b.residual(label('attnResidual', 'Mixer residual', mixerEnd, 0), residual, mixed);
    residual = feedForward(b, attended, mixerEnd + 1);
    row = mixerEnd + 5;
  }
  b.layer = -1;
  b.stage = 'output';
  const normalized = b.norm(
    label('ln_f.lnResid', 'Final RMSNorm', row),
    residual,
    id === 'llama' ? 1e-5 : 1e-6,
  );
  const logits = b.project(
    label('logits', 'Vocabulary logits', row + 1),
    normalized,
    V,
    'V',
    'lmHeadWeight',
  );
  const max = Array.from({ length: T }, (_, t) =>
    Math.max(...Array.from({ length: V }, (_, v) => logits.v(t, v))),
  );
  const inv = max.map((m, t) => 1 / sum(V, (v) => Math.exp(logits.v(t, v) - m)));
  b.tensor(
    label('logitsSoftmax', 'Teaching vocabulary probabilities', row + 2, 1, {
      axes: ['T', 'V', 'B'],
      reduction: V,
    }),
    T,
    V,
    (t, v) => Math.exp(logits.v(t, v) - max[t]) * inv[t],
    'P(v | t) = exp(l[t,v] − m[t]) / Σᵢ exp(l[t,i] − m[t])',
    [link(logits, 'xyz', 'logit'), link(logits, 'xiz', 'normalizer')],
    (t, v, _z, result) =>
      `n = ${V}   exp(${number(logits.v(t, v))} − ${number(max[t])}) × ${number(inv[t])} = ${number(result)}`,
  );
  const scope = {
    zh:
      id === 'llama'
        ? '固定教学权重。8 个 token、16 个词元、32 维、4 个查询头共用 1 组 KV、112 维 SwiGLU。展开 1 个代表层。'
        : '固定教学权重。8 个纯文本 token、16 个词元、40 维、136 维 SwiGLU。展开 3 层 DeltaNet 和 1 层门控 GQA。省略视觉塔、MTP 与后续重复周期。',
    en:
      id === 'llama'
        ? 'Fixed teaching weights. Eight positions, 16 token IDs, width 32, four query heads per KV head, and a 112-channel SwiGLU. One representative decoder layer.'
        : 'Fixed teaching weights. Eight text positions, 16 token IDs, width 40, and a 136-channel SwiGLU. Three DeltaNet layers and one gated GQA layer are evaluated. The vision tower, MTP and additional cycles are omitted.',
  };
  const result = b.finish(scope);
  examples.set(id, result);
  return result;
}
