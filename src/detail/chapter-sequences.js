// A read frame can land on the row it inspects. Temporary scalars (for example
// mean(x²)) belong in the equation, never in a fabricated intermediate tensor.
function span(node, x, y, z = 0, axis = -1, length = 1) {
  if (!node) return null;
  const indices = [x, y, z];
  const size = [node.cx, node.cy, node.cz];
  if (indices.some((v, i) => !Number.isInteger(v) || v < 0 || v >= size[i])) return null;
  if (!Number.isInteger(length) || length < 1 || (axis >= 0 && indices[axis] + length > size[axis]))
    return null;
  return { node, indices, axis, length };
}

const row = (n, t, z = 0) => n && span(n, t, 0, z, 1, n.cy);
const known = (v) => typeof v === 'number' && !Number.isNaN(v);
const fmt = (v) => (v === -Infinity ? '−∞' : v === Infinity ? '∞' : Number(v.toPrecision(6)).toString());
const result = (expr, value) => (known(value) ? `${expr} = ${fmt(value)}` : expr);

function equation(expr, values, substitute, value) {
  const withTerms = values.every(known) ? `${expr} = ${substitute(...values.map(fmt))}` : expr;
  return result(withTerms, value);
}
function frame(zh, en, formula, sources, output) {
  return output && { title: { zh, en }, formula, sources: sources.filter(Boolean), output };
}

// The transport passes the number of visited cells, not a zero-based index.
// The callback only formats precomputed prefix statistics while the camera moves.
function progressive(phase, source, describe) {
  if (!phase || !source) return phase;
  return {
    ...phase,
    readoutLength: source.length,
    readout(count) {
      const k = Math.max(1, Math.min(source.length, Math.trunc(count) || 1));
      return `k = ${k}/${source.length}    ${describe(k)}`;
    },
  };
}

function along(c, s, visit) {
  if (!s || !c.sample?.value) return false;
  const coord = s.indices.slice();
  for (let i = 0; i < s.length; i++) {
    if (s.axis >= 0) coord[s.axis] = s.indices[s.axis] + i;
    const v = c.sample.value(s.node, ...coord);
    if (!known(v)) return false;
    visit(v);
  }
  return true;
}
function mean(c, s, transform) {
  let total = 0;
  return along(c, s, (v) => {
    total += transform(v);
  })
    ? total / s.length
    : undefined;
}
function softmaxStats(c, s) {
  const maxima = [],
    sums = [];
  let max = -Infinity,
    sum = 0;
  const complete = along(c, s, (v) => {
    if (v > max) {
      sum = sum * Math.exp(max - v) + 1;
      max = v;
    } else if (v > -Infinity) {
      sum += Math.exp(v - max);
    }
    maxima.push(max);
    sums.push(sum);
  });
  return {
    max: complete && sum > 0 ? max : undefined,
    inverse: complete && sum > 0 ? 1 / sum : undefined,
    showMax(k) {
      return result('m(k) = maxᵢ<ₖ sᵢ', maxima[k - 1]);
    },
    showSum(k, fullMax) {
      // All terms in this phase use the same maximum of the ENTIRE visible row.
      // Convert the numerically stable running sum to that fixed reference scale.
      const shifted =
        known(sums[k - 1]) && known(fullMax) ? sums[k - 1] * Math.exp(maxima[k - 1] - fullMax) : undefined;
      return `${result('m', fullMax)}    ${result('Z(k) = Σᵢ<ₖ exp(sᵢ − m)', shifted)}    ${result('r(k) = 1 / Z(k)', known(shifted) ? 1 / shifted : undefined)}`;
    },
  };
}

function embed(c) {
  const { node, x: t, y: d, z, at, read, dep } = c;
  const id = span(dep(node, 'index'), t, 0, z);
  const table = dep(node, 'gather');
  const j = read(id);
  const valid = table && Number.isInteger(j) && j >= 0 && j < table.cx;
  const cell = valid ? span(table, j, d) : null;
  // With an unknown token ID, the possible values at channel d lie along V.
  const candidates = table && span(table, 0, d, 0, 0, table.cx);
  const lookup = valid ? `E[${j},${d}]` : `E[ids[${t}],${d}]`;
  const pos = span(dep(node, 'position'), t, d);
  const reads = [
    frame(
      '读取这个位置的词元 ID',
      'Read the token ID at this position',
      result(`j = ids[${t}]`, j),
      [id],
      id,
    ),
    frame(
      valid ? '用词元 ID 查嵌入表' : '标出未知词元的候选表行',
      valid ? 'Index the embedding table' : 'Locate the candidate rows for the unknown token',
      valid ? result(lookup, read(cell)) : `0 ≤ ids[${t}] < ${table?.cx ?? 'V'}    ${lookup}`,
      [id, cell ?? candidates],
      cell ?? candidates,
    ),
  ];
  if (pos) {
    reads.push(
      frame(
        '读取同一位置的位置向量',
        'Read the learned positional feature',
        result(`P[${t},${d}]`, read(pos)),
        [pos],
        pos,
      ),
    );
  }
  reads.push(
    frame(
      '写入残差流',
      'Write the first residual feature',
      pos
        ? equation(
            `x[${t},${d}] = ${lookup} + P[${t},${d}]`,
            [read(cell), read(pos)],
            (a, b) => `${a} + ${b}`,
            read(at),
          )
        : result(`x[${t},${d}] = ${lookup}`, read(at)),
      [id, cell ?? candidates, pos],
      at,
    ),
  );
  return reads;
}

function normalize(c) {
  const { node, x: t, y: d, z, at, read, dep, modelId } = c;
  const input = dep(node, 'input');
  const vector = row(input, t, z);
  const selected = span(input, t, d, z);
  const rms = Boolean(dep(node, 'invRms'));
  const meanNode = span(dep(node, 'mean'), t, 0, z);
  const inverse = span(dep(node, rms ? 'invRms' : 'invStd'), t, 0, z);
  const gain = span(dep(node, rms ? 'gain' : 'scale'), 0, d);
  const bias = span(dep(node, 'bias'), 0, d);
  const sums = [];
  let total = 0;
  const complete = along(c, vector, (v) => {
    total += rms ? v * v : v;
    sums.push(total);
  });
  const width = input?.cy ?? node.cy;
  const fullAverage = complete ? total / width : undefined;
  const average = read(meanNode) ?? (rms ? undefined : fullAverage);
  const eps = modelId === 'qwen' ? '10⁻⁶' : '10⁻⁵';
  const moment = rms
    ? fullAverage
    : known(average)
      ? mean(c, vector, (v) => (v - average) ** 2)
      : undefined;
  const gainOffset = modelId === 'qwen' && node.formula.includes('1 + w');
  const u = read(selected),
    r = read(inverse),
    g = read(gain),
    b = read(bias);
  const effective = known(g) ? g + Number(gainOffset) : undefined;
  return [
    progressive(
      rms
        ? frame(
            '读取同一词元的所有通道并求均方',
            'Read one token and average its squared channels',
            result(`m₂ = Σᵢ x[${t},i]² / ${width}`, moment),
            [vector],
            vector,
          )
        : frame(
            '对同一词元的通道求均值',
            'Average the channels of one token',
            result(`μ[${t}] = Σᵢ x[${t},i] / ${width}`, average),
            [vector],
            meanNode,
          ),
      vector,
      (k) => {
        const partial = sums[k - 1];
        const sum = rms ? 'Σᵢ<ₖ xᵢ²' : 'Σᵢ<ₖ xᵢ';
        const statistic = rms ? 'm₂(k)' : 'μ(k)';
        return `${result(sum, partial)}    ${result(`${statistic} = ${known(partial) ? fmt(partial) : sum} / ${k}`, known(partial) ? partial / k : undefined)}`;
      },
    ),
    frame(
      rms ? '写入均方根的倒数' : '中心化并求标准差倒数',
      rms ? 'Store the inverse RMS' : 'Center and compute the inverse deviation',
      result(`r[${t}] = 1 / √(${known(moment) ? fmt(moment) : rms ? 'm₂' : 'meanᵢ((xᵢ−μ)²)'} + ${eps})`, r),
      [vector, meanNode],
      inverse,
    ),
    frame(
      rms ? '读取当前通道的增益' : '读取尺度和偏置',
      rms ? 'Read this channel’s gain' : 'Read the affine scale and bias',
      rms
        ? result(gainOffset ? `γ[${d}] = 1 + w[${d}]` : `γ[${d}]`, effective)
        : `${result(`γ[${d}]`, g)}    ${result(`β[${d}]`, b)}`,
      [gain, bias],
      gain,
    ),
    frame(
      '缩放并写回选中单元',
      'Scale and write the selected cell',
      rms
        ? equation(
            `u[${t},${d}] = x × r × γ`,
            [u, r, effective],
            (a, i, gain) => `${a} × ${i} × ${gain}`,
            read(at),
          )
        : equation(
            `u[${t},${d}] = (x − μ) × r × γ + β`,
            [u, average, r, g, b],
            (a, m, i, gain, bias) => `(${a} − ${m}) × ${i} × ${gain} + ${bias}`,
            read(at),
          ),
      [selected, meanNode, inverse, gain, bias],
      at,
    ),
  ];
}

function queryWeight(c) {
  const { node, x: channel, y: packed, at, read, find } = c;
  const merged = node.op === 'qkvWeightBlock',
    height = merged ? node.cy / 3 : node.cy;
  const letter = merged ? 'qkv'[Math.floor(packed / height)] : node.op[0];
  const feature = packed % height,
    name = letter.toUpperCase();
  const projected = find(merged ? 'qkvBlock' : letter + 'Block', node.layer, node.head);
  const u = find('ln1.lnResid', node.layer, -1);
  const b = span(find(letter + 'BiasBlock', node.layer, node.head), 0, feature);
  const selectedInput = span(u, 0, channel),
    out = span(projected, 0, packed);
  const w = read(at),
    v = read(selectedInput);
  return [
    frame(
      '读取所选头的投影权重',
      'Read the projection weight for this head',
      result(`W${name}[h=${node.head},a=${feature},c=${channel}]`, w),
      [at],
      at,
    ),
    frame(
      '在位置 0 匹配输入通道',
      'Pair it with the input channel at position 0',
      equation(
        `p = W${name}[a,${channel}] × u[0,${channel}]`,
        [w, v],
        (a, b) => `${a} × ${b}`,
        known(w) && known(v) ? w * v : undefined,
      ),
      [at, selectedInput],
      selectedInput,
    ),
    frame(
      '归约全部输入通道并写入投影',
      'Reduce the channels and write the projected feature',
      result(
        `${name}[0,${feature}] = Σᵢ W${name}[${feature},i] × u[0,i] + b${name}[${feature}]`,
        read(out),
      ),
      [span(node, 0, packed, 0, 0, node.cx), row(u, 0), b],
      out,
    ),
  ];
}

function projectHeads(c) {
  const { node, x: channel, y: d, at, read, find, resolve } = c;
  const outNode = find('attnOut', node.layer, -1);
  if (!outNode) return [];
  const feeds = outNode.depMaps.flatMap((mapping, i) =>
    mapping.role === 'concat'
      ? [{ node: resolve(outNode.deps[i], node.layer), start: mapping.span.start, end: mapping.span.end }]
      : [],
  );
  const feed = feeds.find((part) => channel >= part.start && channel < part.end);
  if (!feed?.node) return [];
  const a = channel - feed.start;
  const src = span(feed.node, 0, a),
    output = span(outNode, 0, d);
  const bias = span(find('projBias', node.layer, -1), 0, d);
  return [
    frame(
      '读取输出投影的一个系数',
      'Read a coefficient of the output projection',
      result(`WO[${d},${channel}]`, read(at)),
      [at],
      at,
    ),
    frame(
      '定位拼接通道来自哪个头',
      'Locate the originating head and channel',
      result(`concat[0,${channel}] = head[${feed.node.head}][0,${a}]`, read(src)),
      [src],
      src,
    ),
    frame(
      '合并各头并写入位置 0',
      'Mix all heads and write at position 0',
      result(`a[0,${d}] = Σⱼ WO[${d},j] × concat[0,j] + bO[${d}]`, read(output)),
      [span(node, 0, d, 0, 0, node.cx), ...feeds.map((part) => row(part.node, 0)), bias],
      output,
    ),
  ];
}

function attention(c) {
  const { node, x: k, y: t, z, at, read, dep, find, modelId } = c;
  const scores = dep(node, 'score') ?? dep(node, 'input');
  const score = span(scores, k, t, z),
    visible = span(scores, 0, t, z, 0, t + 1);
  if (k > t)
    return [
      frame(
        '只保留当前及以前的位置',
        'Keep the causal prefix',
        `0 ≤ j ≤ ${t}, k = ${k}`,
        [visible],
        visible,
      ),
      frame('屏蔽未来的键', 'Mask the future key', `k = ${k} > t = ${t} ⇒ s[t,k] = −∞`, [score], score),
      frame(
        '被屏蔽的位置概率为零',
        'The masked position contributes zero probability',
        `p[${t},${k}] = 0`,
        [score],
        at,
      ),
    ];
  const Q = dep(scores, 'query'),
    K = dep(scores, 'sharedKey') ?? dep(scores, 'key');
  const A = scores?.dotLen ?? 0;
  const packed = Q && K && Q.id === K.id;
  const qs = span(Q, t, 0, z, 1, A),
    ks = span(K, k, packed ? A : 0, z, 1, A);
  const stats = softmaxStats(c, visible);
  const maxCell = span(find('attnMtxAgg2', node.layer, node.head), 0, t, z);
  const invCell = span(find('attnMtxAgg1', node.layer, node.head), 0, t, z);
  const max = read(maxCell) ?? stats.max,
    inv = read(invCell) ?? stats.inverse;
  const grouped = modelId === 'llama' || modelId === 'qwen';
  const steps = [
    frame(
      grouped ? '旋转查询与共享键做点积' : '查询与键做缩放点积',
      grouped ? 'Dot the rotated query with its shared key' : 'Take the scaled query–key dot product',
      result(`s[${t},${k}] = Qₕ[${t},:] · ${grouped ? 'Kshared' : 'Kₕ'}[${k},:] / √${A}`, read(score)),
      [qs, ks],
      score,
    ),
    progressive(
      frame(
        '在可见前缀上归一化',
        'Normalize across the causal prefix',
        `${result(`m = max(s[${t},0…${t}])`, max)}    ${result('r = 1 / Σⱼ≤ₜ exp(s[t,j] − m)', inv)}`,
        [visible, maxCell],
        invCell ?? visible,
      ),
      visible,
      (k) => stats.showSum(k, max),
    ),
    frame(
      '写入这个键的注意力概率',
      'Write the probability of this key',
      equation(
        `p[${t},${k}] = exp(s − m) × r`,
        [read(score), max, inv],
        (s, m, r) => `exp(${s} − ${m}) × ${r}`,
        read(at),
      ),
      [score, visible, maxCell, invCell],
      at,
    ),
  ];
  const mixed = find('vOutBlock', node.layer, node.head);
  const mixedCell = span(mixed, t, 0, z);
  const values = dep(mixed, 'sharedValue') ?? dep(mixed, 'value');
  const valueColumn = span(values, 0, values?.op === 'qkvBlock' ? 2 * A : 0, z, 0, t + 1);
  const pRow = span(node, 0, t, z, 0, t + 1);
  if (modelId === 'qwen') {
    const gate = span(find('attnGate', node.layer, node.head), t, 0, z);
    const gated = span(find('attnGated', node.layer, node.head), t, 0, z);
    steps.push(
      frame(
        '汇聚值，再施加输出门',
        'Mix values, then apply the output gate',
        `o = Σⱼ≤ₜ p[t,j] × V[j,0]    ${equation(`g[${t},0] = o × sigmoid(z)`, [read(mixedCell), read(gate)], (v, g) => `${v} × sigmoid(${g})`, read(gated))}`,
        [pRow, valueColumn, mixedCell, gate],
        gated,
      ),
    );
  } else {
    steps.push(
      frame(
        grouped ? '读取共享值的第 0 个通道' : '汇聚值的第 0 个通道',
        grouped ? 'Read channel 0 of the shared values' : 'Mix value channel 0',
        result(`oₕ[${t},0] = Σⱼ≤ₜ p[${t},j] × V[j,0]`, read(mixedCell)),
        [pRow, valueColumn],
        mixedCell,
      ),
    );
  }
  return steps;
}

// The classic MLP expansion is drawn transposed. Use the draw-time cell maps
// when addressing an affine operation so the weights and hidden features agree.
function affine(c, target, label) {
  if (!target) return { formula: label, sources: [] };
  const n = target.node;
  const [x, y, z] = target.indices;
  const coordinates = { x, y, z, i: 0, 0: 0 };
  const sources = n.deps.map((id, i) => {
    const source = c.resolve(id, n.layer),
      mapping = n.depMaps[i];
    const start = [...mapping.map].map((key, j) => coordinates[key] + (mapping.offset?.[j] ?? 0));
    const axis = mapping.map.indexOf('i');
    return span(source, ...start, axis, axis < 0 ? 1 : n.dotLen);
  });
  const weight = sources[n.depMaps.findIndex((mapping) => mapping.role === 'weight')];
  const input = sources[n.depMaps.findIndex((mapping) => mapping.role === 'input')];
  const bias = sources[n.depMaps.findIndex((mapping) => mapping.role === 'bias')];
  function endpoint(s, last) {
    if (!s) return undefined;
    const coord = s.indices.slice();
    if (last && s.axis >= 0) coord[s.axis] += s.length - 1;
    return c.read(span(s.node, ...coord));
  }
  const values = [
    endpoint(weight, false),
    endpoint(input, false),
    endpoint(weight, true),
    endpoint(input, true),
  ];
  if (bias) values.push(c.read(bias));
  const formula = equation(
    label,
    values,
    (w0, u0, w1, u1, b) => {
      const final = n.dotLen === 1 ? '' : (n.dotLen === 2 ? ' + ' : ' + … + ') + w1 + ' × ' + u1;
      return w0 + ' × ' + u0 + final + (b === undefined ? '' : ' + ' + b);
    },
    c.read(target),
  );
  return { formula, sources };
}

function activation(c) {
  const { node, x, y, z, at, read, dep } = c;
  const gateNode = dep(node, 'gate');
  if (gateNode) {
    const gate = span(gateNode, x, y, z),
      up = span(dep(node, 'up'), x, y, z);
    const g = read(gate),
      u = read(up);
    const gateDot = affine(c, gate, `g[${x},${y}] = Wgate[${y},:] · u[${x},:]`);
    const upDot = affine(c, up, `v[${x},${y}] = Wup[${y},:] · u[${x},:]`);
    const activated = known(g) ? g / (1 + Math.exp(-g)) : undefined;
    return [
      frame('计算门支路的投影', 'Project the gate branch', gateDot.formula, gateDot.sources, gate),
      frame('计算上投影支路', 'Project the value branch', upDot.formula, upDot.sources, up),
      frame(
        '在门支路上应用 SiLU',
        'Apply SiLU to the gate',
        equation('SiLU(g) = g / (1 + exp(−g))', [g], (gv) => `${gv} / (1 + exp(−(${gv})))`, activated),
        [gate],
        gate,
      ),
      frame(
        '逐通道相乘并写入 SwiGLU',
        'Multiply the branches and write SwiGLU',
        equation(`a[${x},${y}] = SiLU(g) × v`, [activated, u], (a, b) => `${a} × ${b}`, read(at)),
        [gate, up],
        at,
      ),
    ];
  }
  const pre = span(dep(node, 'input'), x, y, z);
  const dot = affine(c, pre, `f[${y},${x}] = Wup[${x},:] · u[${y},:] + bup[${x}]`);
  const f = read(pre);
  const tanh = known(f) ? Math.tanh(Math.sqrt(2 / Math.PI) * (f + 0.044715 * f ** 3)) : undefined;
  return [
    frame('展开当前词元的通道', 'Compute the expansion at this token', dot.formula, dot.sources, pre),
    frame(
      '计算 GELU 的平滑门',
      'Evaluate the smooth GELU gate',
      result('b = tanh(√(2/π) × (f + 0.044715 f³))', tanh),
      [pre],
      pre,
    ),
    frame(
      '写入激活通道',
      'Write the activated channel',
      equation(`a[${y},${x}] = ½f × (1 + b)`, [f, tanh], (v, g) => `0.5 × ${v} × (1 + ${g})`, read(at)),
      [pre],
      at,
    ),
  ];
}

function residual(c) {
  const { node, at, x, y, z, read, dep } = c;
  const skip = span(dep(node, 'skip'), x, y, z);
  const update = span(dep(node, 'add'), x, y, z);
  const hasBias = Boolean(dep(update?.node, 'bias'));
  const down = affine(c, update, `Δ[${x},${y}] = Wdown[${y},:] · a[${x},:]${hasBias ? ' + bdown' : ''}`);
  return [
    frame(
      '保留旁路中的原有特征',
      'Keep the original skip feature',
      result(`r[${x},${y}]`, read(skip)),
      [skip],
      skip,
    ),
    frame('把前馈更新投回隐藏空间', 'Project the FFN update back', down.formula, down.sources, update),
    frame(
      '在相同位置与通道相加',
      'Add at the same position and channel',
      equation(`x′[${x},${y}] = r + Δ`, [read(skip), read(update)], (r, d) => `${r} + ${d}`, read(at)),
      [skip, update],
      at,
    ),
  ];
}

function readout(c) {
  const { node, x: t, y: v, z, at, read, find, dep } = c;
  const logits = dep(node, 'logit') ?? dep(node, 'input');
  const logit = span(logits, t, v, z),
    vocabulary = row(logits, t, z);
  const score = affine(c, logit, `ℓ[${t},${v}] = Wlm[${v},:] · u[${t},:]`);
  const maximum = span(find('logitsAgg2', -1, -1), t, 0, z);
  const inverse = span(find('logitsAgg1', -1, -1), t, 0, z);
  const stats = softmaxStats(c, vocabulary);
  const m = read(maximum) ?? stats.max,
    r = read(inverse) ?? stats.inverse;
  return [
    frame(
      '把最终特征投影到词表',
      'Project the final features into the vocabulary',
      score.formula,
      score.sources,
      logit,
    ),
    progressive(
      frame(
        '找出该位置的词表最大分数',
        'Find the vocabulary maximum at this position',
        result(`m = maxⱼ ℓ[${t},j]`, m),
        [vocabulary],
        maximum ?? vocabulary,
      ),
      vocabulary,
      (k) => stats.showMax(k),
    ),
    progressive(
      frame(
        '归约整个词表的指数和',
        'Reduce the shifted exponentials over the vocabulary',
        result(`r = 1 / Σⱼ exp(ℓ[${t},j] − ${known(m) ? fmt(m) : 'm'})`, r),
        [vocabulary, maximum],
        inverse ?? vocabulary,
      ),
      vocabulary,
      (k) => stats.showSum(k, m),
    ),
    frame(
      '写入选中词元的概率',
      'Write this vocabulary entry’s probability',
      equation(
        `P[${t},${v}] = exp(ℓ − m) × r`,
        [read(logit), m, r],
        (lv, m, r) => `exp(${lv} − ${m}) × ${r}`,
        read(at),
      ),
      [logit, maximum, inverse, ...(inverse ? [] : [vocabulary])],
      at,
    ),
  ];
}

function delta(c) {
  const { node, x: a, y: d, z: t, at, read, dep } = c;
  const h = node.head;
  const retained = dep(node, 'state'),
    key = dep(node, 'key'),
    write = dep(node, 'correction');
  const incoming = span(dep(retained, 'state'), a, d, t);
  const previous = t ? span(node, a, d, t - 1) : null;
  const rho = span(dep(retained, 'retention'), t, h);
  const kept = span(retained, a, d, t);
  const memory = span(dep(write, 'retrieved'), t, d);
  const beta = span(dep(write, 'gate'), t, h),
    v = span(dep(write, 'value'), t, d);
  const difference = span(write, t, d),
    k = span(key, t, a);
  return [
    frame(
      t ? '读取上一个时间切片并衰减' : '从零状态开始',
      t ? 'Retain the previous time slice' : 'Start from the zero state',
      t
        ? equation(
            `S̄[${t},${a},${d}] = ρ[${t},${h}] × S[${t - 1},${a},${d}]`,
            [read(rho), read(previous)],
            (r, s) => `${r} × ${s}`,
            read(kept),
          )
        : `Sinit[${a},${d}] = 0    ${result(`S̄[0,${a},${d}] = ρ[0,${h}] × 0`, read(kept))}`,
      [previous, incoming, rho],
      kept,
    ),
    frame(
      '用当前键读取保留下来的状态',
      'Recall from the retained state with the current key',
      result(`m[${t},${d}] = Σᵢ k[${t},i] × S̄[${t},i,${d}]`, read(memory)),
      [row(key, t), retained && span(retained, 0, d, t, 0, retained.cx)],
      memory,
    ),
    frame(
      '门控当前值与记忆读数之差',
      'Gate the error between value and recall',
      equation(
        `Δ[${t},${d}] = β × (v − m)`,
        [read(beta), read(v), read(memory)],
        (b, v, m) => `${b} × (${v} − ${m})`,
        read(difference),
      ),
      [beta, v, memory],
      difference,
    ),
    frame(
      '把外积修正写入当前状态',
      'Write the outer-product correction to this state cell',
      equation(
        `S[${t},${a},${d}] = S̄ + k × Δ`,
        [read(kept), read(k), read(difference)],
        (s, k, d) => `${s} + ${k} × ${d}`,
        read(at),
      ),
      [kept, k, difference],
      at,
    ),
  ];
}

export function chapterSequence({ modelId, node, x, y, z = 0, sample, find, resolve }) {
  const at = span(node, x, y, z);
  if (!at) return [];
  const c = {
    node,
    x,
    y,
    z,
    modelId,
    at,
    sample,
    find,
    resolve,
    read: (s) => (s ? sample?.value?.(s.node, ...s.indices) : undefined),
    dep(n, role) {
      const index = n?.depMaps?.findIndex((map) => map.role === role);
      return index >= 0 ? resolve(n.deps[index], n.layer) : undefined;
    },
  };
  let frames;
  switch (node.op) {
    case 'embed':
    case 'residual0':
      frames = embed(c);
      break;
    case 'ln1.lnResid':
    case 'ln2.lnResid':
    case 'ln_f.lnResid':
      frames = normalize(c);
      break;
    case 'qWeightBlock':
    case 'kWeightBlock':
    case 'vWeightBlock':
    case 'qkvWeightBlock':
      frames = queryWeight(c);
      break;
    case 'projWeight':
      frames = projectHeads(c);
      break;
    case 'attnMtxSm':
      frames = attention(c);
      break;
    case 'mlpAct':
      frames = activation(c);
      break;
    case 'mlpResidual':
      frames = residual(c);
      break;
    case 'logitsSoftmax':
      frames = readout(c);
      break;
    case 'deltaState':
      frames = delta(c);
      break;
    default:
      return [];
  }
  return frames.filter(Boolean);
}
