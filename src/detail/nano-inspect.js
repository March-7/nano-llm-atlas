function show(n) {
  if (n === -Infinity) return '−∞';
  if (n === Infinity) return '∞';
  return Number(n.toPrecision(6)).toString();
}

function grid(nx, ny, read, formula, substitute, source) {
  return { nx, ny, read, formula, substitute, source };
}

// The drawing's x/y strides address a checkpoint tensor in its native row-major order.
function parameterGrid(tensors, key, nx, ny, strideX, strideY, start = 0) {
  const { shape, values } = tensors[key];
  const index = (x, y) => start + x * strideX + y * strideY;
  const source = (x, y) => {
    const flat = index(x, y);
    const coordinates =
      shape.length === 1 ? `${flat}` : `${Math.floor(flat / shape[1])}, ${flat % shape[1]}`;
    return `${key}[${coordinates}]`;
  };
  return grid(
    nx,
    ny,
    (x, y) => values[index(x, y)],
    'θ = checkpoint[index]',
    (x, y, result) => `${source(x, y)} = ${show(result)}`,
    source,
  );
}

// Show both endpoint products and the exact number of terms used in the result.
function dotExample(n, a, b, answer, bias, divisor = 1) {
  const term = (i) => `${show(a(i))} × ${show(b(i))}`;
  let sum = term(0);
  if (n > 1) sum += (n === 2 ? ' + ' : ' + … + ') + term(n - 1);
  if (divisor !== 1) sum = `(${sum}) / ${show(divisor)}`;
  if (bias !== undefined) sum = `${show(bias)} + ${sum}`;
  return `n = ${n}   ${sum} = ${show(answer)}`;
}

function linearGrid(
  tensors,
  rows,
  { width, input, output, weight, bias, offset = 0, transposed = false, formula },
) {
  const matrix = tensors[weight];
  const count = matrix.shape[1];
  const b = tensors[bias]?.values;
  const read = (x, y) => output(rows[transposed ? y : x])[offset + (transposed ? x : y)];
  const describe = (x, y, result) => {
    const vector = input(rows[transposed ? y : x]);
    const j = offset + (transposed ? x : y);
    return dotExample(
      count,
      (i) => matrix.values[j * count + i],
      (i) => vector[i],
      result,
      b?.[j],
    );
  };
  return grid(
    transposed ? width : rows.length,
    transposed ? rows.length : width,
    read,
    formula,
    describe,
    weight,
  );
}

function registerNorm(target, name, rows, pick, tensors, key, width) {
  const get = (t) => pick(rows[t]);
  target[`${name}.lnSigma`] = parameterGrid(tensors, key + '.weight', 1, width, 0, 1);
  target[`${name}.lnMu`] = parameterGrid(tensors, key + '.bias', 1, width, 0, 1);
  target[`${name}.lnAgg1`] = grid(
    rows.length,
    1,
    (t) => get(t).mean,
    'μ[t] = Σᵢ x[t,i] / C',
    (t, _y, value) => {
      const v = get(t).input;
      return `C = ${width}   (${show(v[0])} + … + ${show(v[width - 1])}) / ${width} = ${show(value)}`;
    },
  );
  target[`${name}.lnAgg2`] = grid(
    rows.length,
    1,
    (t) => get(t).scale,
    'r[t] = 1 / √(Var(x[t]) + ε)',
    (t, _y, value) =>
      `μ = ${show(get(t).mean)}   1 / √(${show(get(t).variance)} + 0.00001) = ${show(value)}`,
  );
  target[`${name}.lnResid`] = grid(
    rows.length,
    width,
    (t, d) => get(t).output[d],
    'LN[t,d] = (x[t,d] − μ[t]) × r[t] × γ[d] + β[d]',
    (t, d, value) => {
      const s = get(t);
      const gamma = tensors[key + '.weight'].values[d];
      const beta = tensors[key + '.bias'].values[d];
      const center = `${show(s.input[d])} ${s.mean < 0 ? '+' : '−'} ${show(Math.abs(s.mean))}`;
      const shift = `${beta < 0 ? '−' : '+'} ${show(Math.abs(beta))}`;
      return `Var = ${show(s.variance)}   (${center}) × ${show(s.scale)} × ${show(gamma)} ${shift} = ${show(value)}`;
    },
    key,
  );
}

// The first aggregate channel is the reciprocal shifted exponential sum. The second is the row maximum.
function softmaxAggregates(rows, pick, vertical) {
  const get = (x, y) => pick(rows[vertical ? y : x]);
  const nx = vertical ? 1 : rows.length;
  const ny = vertical ? rows.length : 1;
  const inverse = grid(
    nx,
    ny,
    (x, y) => get(x, y).inverseSum,
    'z⁻¹ = 1 / Σᵢ exp(sᵢ − m)',
    (x, y, result) => {
      const { logits, peak } = get(x, y);
      const term = (v) => `exp(${show(v)} − ${show(peak)})`;
      let sum = term(logits[0]);
      if (logits.length > 1) sum += (logits.length === 2 ? ' + ' : ' + … + ') + term(logits.at(-1));
      return `n = ${logits.length}   1 / (${sum}) = ${show(result)}`;
    },
  );
  const max = grid(
    nx,
    ny,
    (x, y) => get(x, y).peak,
    'm = maxᵢ sᵢ',
    (x, y, result) => `max(${Array.from(get(x, y).logits, show).join(', ')}) = ${show(result)}`,
  );
  return [inverse, max];
}

function probabilityExample(sm, i, value) {
  return `exp(${show(sm.logits[i])} − ${show(sm.peak)}) × ${show(sm.inverseSum)} = ${show(value)}`;
}

function headGrids(tensors, rows, base, h, width, headSize) {
  const target = Object.create(null);
  for (const [index, letter] of ['q', 'k', 'v'].entries()) {
    const offset = index * width + h * headSize;
    const key = base + 'attn.c_attn';
    target[letter + 'WeightBlock'] = parameterGrid(
      tensors,
      key + '.weight',
      width,
      headSize,
      1,
      width,
      offset * width,
    );
    target[letter + 'BiasBlock'] = parameterGrid(tensors, key + '.bias', 1, headSize, 0, 1, offset);
    target[letter + 'Block'] = linearGrid(tensors, rows, {
      width: headSize,
      input: (r) => r.ln1.output,
      output: (r) => r.qkv,
      weight: key + '.weight',
      bias: key + '.bias',
      offset,
      formula: `${letter.toUpperCase()}[t,h,a] = b[h,a] + Σᵢ W[h,a,i] × LN₁[t,i]`,
    });
  }
  const head = (row) => row.heads[h];
  const T = rows.length;
  const headOffset = h * headSize;
  target.attnMtx = grid(
    T,
    T,
    (k, q) => (k > q ? -Infinity : head(rows[q]).logits[k]),
    (k, q) => (k > q ? 'k > q ⇒ score = −∞' : 's[q,k] = Q[q] · K[k] / √dₕ'),
    (k, q, result) =>
      k > q
        ? `${k} > ${q} ⇒ −∞`
        : dotExample(
            headSize,
            (i) => rows[q].qkv[headOffset + i],
            (i) => rows[k].qkv[width + headOffset + i],
            result,
            undefined,
            Math.sqrt(headSize),
          ),
  );
  [target.attnMtxAgg1, target.attnMtxAgg2] = softmaxAggregates(rows, head, true);
  target.attnMtxSm = grid(
    T,
    T,
    (k, q) => (k > q ? 0 : head(rows[q]).probabilities[k]),
    (k, q) => (k > q ? 'k > q ⇒ α[q,k] = 0' : 'α[q,k] = exp(s[q,k] − m[q]) × z[q]⁻¹'),
    (k, q, result) => (k > q ? `${k} > ${q} ⇒ exp(−∞) = 0` : probabilityExample(head(rows[q]), k, result)),
  );
  target.vOutBlock = grid(
    T,
    headSize,
    (t, d) => rows[t].mixed[headOffset + d],
    'o[t,h,a] = Σₖ≤ₜ α[t,k] × V[k,h,a]',
    (t, d, result) =>
      dotExample(
        t + 1,
        (k) => head(rows[t]).probabilities[k],
        (k) => rows[k].qkv[2 * width + headOffset + d],
        result,
      ),
  );
  return target;
}

export function makeNanoInspector({ prefix, config, tensors, embed, layers, final }) {
  const { n_embd: width, n_head: heads, vocab_size: vocab, block_size: context } = config;
  const T = prefix.length;
  const tokenIds = Array.from(prefix, (letter) => 'ABC'.indexOf(letter));
  const root = Object.create(null);
  root.idxObj = grid(
    T,
    1,
    (t) => tokenIds[t],
    'token[t] = id(letter[t])',
    (t, _y, value) => `t = ${t}   ${prefix[t]} → ${value}`,
  );
  root.tokEmbedObj = parameterGrid(tensors, 'transformer.wte.weight', vocab, width, width, 1);
  root.posEmbedObj = parameterGrid(tensors, 'transformer.wpe.weight', context, width, width, 1);
  root.residual0 = grid(
    T,
    width,
    (t, d) => embed[t][d],
    'x₀[t,d] = E[token[t],d] + P[t,d]',
    (t, d, result) => {
      const v = root.tokEmbedObj.read(tokenIds[t], d);
      const p = root.posEmbedObj.read(t, d);
      return `E[${tokenIds[t]},${d}] + P[${t},${d}] = ${show(v)} + ${show(p)} = ${show(result)}`;
    },
  );
  registerNorm(root, 'ln_f', final, (row) => row.ln, tensors, 'transformer.ln_f', width);
  root.lmHeadWeight = parameterGrid(tensors, 'lm_head.weight', width, vocab, 1, width);
  root.logits = linearGrid(tensors, final, {
    width: vocab,
    input: (r) => r.ln.output,
    output: (r) => r.sm.logits,
    weight: 'lm_head.weight',
    formula: 'logit[t,v] = Σᵢ Wᵤ[v,i] × LNƒ[t,i]',
  });
  [root.logitsAgg1, root.logitsAgg2] = softmaxAggregates(final, (row) => row.sm, false);
  root.logitsSoftmax = grid(
    T,
    vocab,
    (t, v) => final[t].sm.probabilities[v],
    'P(v | prefix) = exp(logit[t,v] − m[t]) × z[t]⁻¹',
    (t, v, result) => probabilityExample(final[t].sm, v, result),
  );

  const byLayer = layers.map((rows, l) => {
    const base = `transformer.h.${l}.`;
    const target = Object.create(null);
    registerNorm(target, 'ln1', rows, (row) => row.ln1, tensors, base + 'ln_1', width);
    registerNorm(target, 'ln2', rows, (row) => row.ln2, tensors, base + 'ln_2', width);
    const perHead = Array.from({ length: heads }, (_, h) =>
      headGrids(tensors, rows, base, h, width, width / heads),
    );
    const ffWidth = tensors[base + 'mlp.c_fc.weight'].shape[0];
    const wo = base + 'attn.c_proj',
      up = base + 'mlp.c_fc',
      down = base + 'mlp.c_proj';
    target.projWeight = parameterGrid(tensors, wo + '.weight', width, width, 1, width);
    target.projBias = parameterGrid(tensors, wo + '.bias', 1, width, 0, 1);
    target.attnOut = linearGrid(tensors, rows, {
      width,
      input: (r) => r.mixed,
      output: (r) => r.projected,
      weight: wo + '.weight',
      bias: wo + '.bias',
      formula: 'a[t,d] = bₒ[d] + Σᵢ Wₒ[d,i] × concat(heads)[t,i]',
    });
    target.attnResidual = grid(
      T,
      width,
      (t, d) => rows[t].attended[d],
      'u[t,d] = x[t,d] + a[t,d]',
      (t, d, result) => `${show(rows[t].input[d])} + ${show(rows[t].projected[d])} = ${show(result)}`,
    );
    // The expansion matrices display the intermediate FFN channel along x.
    target.mlpFcWeight = parameterGrid(tensors, up + '.weight', ffWidth, width, width, 1);
    target.mlpFcBias = parameterGrid(tensors, up + '.bias', ffWidth, 1, 1, 0);
    target.mlpFc = linearGrid(tensors, rows, {
      width: ffWidth,
      transposed: true,
      input: (r) => r.ln2.output,
      output: (r) => r.fc,
      weight: up + '.weight',
      bias: up + '.bias',
      formula: 'f[t,j] = b₁[j] + Σᵢ W₁[j,i] × LN₂[t,i]',
    });
    target.mlpAct = grid(
      ffWidth,
      T,
      (j, t) => rows[t].expanded[j],
      'GELU(f) = ½f × (1 + tanh(√(2/π) × (f + 0.044715f³)))',
      (j, t, result) => {
        const f = show(rows[t].fc[j]);
        return `½ × (${f}) × (1 + tanh(0.797885 × (${f} + 0.044715 × (${f})³))) = ${show(result)}`;
      },
    );
    target.mlpProjWeight = parameterGrid(tensors, down + '.weight', ffWidth, width, 1, ffWidth);
    target.mlpProjBias = parameterGrid(tensors, down + '.bias', 1, width, 0, 1);
    target.mlpResult = linearGrid(tensors, rows, {
      width,
      input: (r) => r.expanded,
      output: (r) => r.update,
      weight: down + '.weight',
      bias: down + '.bias',
      formula: 'g[t,d] = b₂[d] + Σⱼ W₂[d,j] × GELU(f[t,j])',
    });
    target.mlpResidual = grid(
      T,
      width,
      (t, d) => rows[t].output[d],
      'x′[t,d] = u[t,d] + g[t,d]',
      (t, d, result) => `${show(rows[t].attended[d])} + ${show(rows[t].update[d])} = ${show(result)}`,
    );
    return { target, perHead };
  });

  function at(node, x, y, z) {
    if (!node || z !== 0 || !Number.isInteger(x) || !Number.isInteger(y) || x < 0 || y < 0) return null;
    const { layer = -1, head = -1, op } = node;
    const table =
      layer === -1 ? root : head === -1 ? byLayer[layer]?.target : byLayer[layer]?.perHead[head];
    const cell = table?.[op];
    return cell && x < cell.nx && y < cell.ny ? cell : null;
  }
  return {
    value(node, x, y, z = 0) {
      const cell = at(node, x, y, z);
      return cell ? cell.read(x, y) : null;
    },
    explain(node, x, y, z = 0) {
      const cell = at(node, x, y, z);
      if (!cell) return { formula: '∅', substitution: `(${x}, ${y}, ${z})   T = ${T}` };
      const formula = typeof cell.formula === 'function' ? cell.formula(x, y) : cell.formula;
      const substitution = cell.substitute(x, y, cell.read(x, y));
      if (!cell.source) return { formula, substitution };
      const source = typeof cell.source === 'function' ? cell.source(x, y) : cell.source;
      return { formula, substitution, source };
    },
  };
}
