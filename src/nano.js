import { makeNanoInspector } from './detail/nano-inspect.js';

// Trained sorting checkpoint from bbycroft/llm-viz, revision 9da93742382f.
// The original MIT notice accompanies the weights in assets/gpt-nano-sort-LICENSE.txt.
const tokens = 'ABC';
const checkpoint = new URL('../assets/gpt-nano-sort-model.json', import.meta.url);
let pendingWeights;

async function loadWeights() {
  const response = await fetch(checkpoint);
  if (!response.ok) throw new Error(`NanoGPT checkpoint: HTTP ${response.status}`);
  const { config, ...tensors } = await response.json();

  for (const [name, tensor] of Object.entries(tensors)) {
    const raw = atob(tensor.data);
    const bytes = Uint8Array.from(raw, (char) => char.charCodeAt(0));
    const view = new DataView(bytes.buffer);
    tensor.values = Float32Array.from({ length: bytes.length / 4 }, (_, i) => view.getFloat32(i * 4, true));
    delete tensor.data;
    if (tensor.values.length !== tensor.shape.reduce((size, dim) => size * dim, 1)) {
      throw new Error(`Incomplete tensor: ${name}`);
    }
  }
  return { config, tensors };
}

function linear(input, matrix, bias) {
  const [rows, columns] = matrix.shape;
  const result = new Float32Array(rows);
  for (let row = 0; row < rows; row++) {
    let sum = bias?.values[row] ?? 0;
    const offset = row * columns;
    for (let col = 0; col < columns; col++) sum += matrix.values[offset + col] * input[col];
    result[row] = sum;
  }
  return result;
}

function layerNorm(input, gain, bias, record) {
  const mean = input.reduce((sum, x) => sum + x, 0) / input.length;
  const variance = input.reduce((sum, x) => sum + (x - mean) ** 2, 0) / input.length;
  const scale = 1 / Math.sqrt(variance + 1e-5);
  const output = input.map((x, i) => (x - mean) * scale * gain.values[i] + bias.values[i]);
  if (record) Object.assign(record, { input, mean, variance, scale, output });
  return output;
}

function softmax(logits, record) {
  const peak = Math.max(...logits);
  const exp = Array.from(logits, (x) => Math.exp(x - peak));
  const sum = exp.reduce((total, x) => total + x, 0);
  const probabilities = exp.map((x) => x / sum);
  if (record) Object.assign(record, { logits, peak, inverseSum: 1 / sum, probabilities });
  return probabilities;
}

function residual(input, update) {
  return input.map((x, i) => x + update[i]);
}

function gelu(x) {
  return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)));
}

function causalAttention(qkv, position, heads, width, headWeights, inspectedHeads) {
  const headSize = width / heads;
  const mixed = new Float32Array(width);
  for (let head = 0; head < heads; head++) {
    const start = head * headSize;
    const logits = [];
    for (let source = 0; source <= position; source++) {
      let score = 0;
      for (let d = start; d < start + headSize; d++) {
        score += qkv[position][d] * qkv[source][width + d];
      }
      logits.push(score / Math.sqrt(headSize));
    }
    const record = inspectedHeads ? {} : null;
    const weights = softmax(logits, record);
    inspectedHeads?.push(record);
    headWeights?.push(weights);
    for (let d = start; d < start + headSize; d++) {
      let sum = 0;
      for (let s = 0; s <= position; s++) sum += weights[s] * qkv[s][2 * width + d];
      mixed[d] = sum;
    }
  }
  return mixed;
}

/**
 * Predict the next A/B/C token after up to eleven letters. Traces capture the last
 * position, using the first decoder block for normalization, attention, and GELU.
 */
export async function predictNano(prefix, { trace = false } = {}) {
  return forward(prefix, trace, false);
}

/** Inspect all checkpoint parameters and computed activations for one sorting prefix. */
export async function inspectNano(prefix) {
  return forward(prefix, false, true);
}

async function forward(prefix, trace, inspect) {
  if (typeof prefix !== 'string' || prefix.length === 0 || prefix.length > 11 || /[^ABC]/.test(prefix)) {
    throw new RangeError('Expected a prefix of 1–11 uppercase A/B/C letters.');
  }
  pendingWeights ??= loadWeights().catch((error) => {
    pendingWeights = undefined;
    throw error;
  });
  const { config, tensors } = await pendingWeights;
  const width = config.n_embd;
  const tokenEmb = tensors['transformer.wte.weight'].values;
  const posEmb = tensors['transformer.wpe.weight'].values;
  let states = Array.from(prefix, (letter, t) => {
    const offset = tokens.indexOf(letter) * width;
    return Float32Array.from({ length: width }, (_, d) => tokenEmb[offset + d] + posEmb[t * width + d]);
  });
  const last = prefix.length - 1;
  const traces = trace ? { embed: [Array.from(states[last])], attention: [] } : null;
  const snapshot = inspect ? { prefix, config, tensors, embed: states, layers: [], final: [] } : null;

  for (let i = 0; i < config.n_layer; i++) {
    const block = 'transformer.h.' + i + '.';
    const tensor = (key) => tensors[block + key];
    const firstBlockTrace = i === 0 ? traces : null;
    const rows = inspect ? states.map((input) => ({ input, ln1: {}, ln2: {}, heads: [] })) : null;
    const qkv = states.map((state, t) => {
      const norm = layerNorm(state, tensor('ln_1.weight'), tensor('ln_1.bias'), rows?.[t].ln1);
      if (firstBlockTrace && t === last) firstBlockTrace.norm = [Array.from(norm)];
      return linear(norm, tensor('attn.c_attn.weight'), tensor('attn.c_attn.bias'));
    });
    states = states.map((state, t) => {
      const capture = t === last ? firstBlockTrace : null;
      const row = rows?.[t];
      const mixed = causalAttention(qkv, t, config.n_head, width, capture?.attention, row?.heads);
      const projected = linear(mixed, tensor('attn.c_proj.weight'), tensor('attn.c_proj.bias'));
      const attended = residual(state, projected);
      const norm = layerNorm(attended, tensor('ln_2.weight'), tensor('ln_2.bias'), row?.ln2);
      const fc = linear(norm, tensor('mlp.c_fc.weight'), tensor('mlp.c_fc.bias'));
      const expanded = fc.map(gelu);
      if (capture) capture.ffn = [Array.from(expanded)];
      const update = linear(expanded, tensor('mlp.c_proj.weight'), tensor('mlp.c_proj.bias'));
      const output = residual(attended, update);
      if (row)
        Object.assign(row, { qkv: qkv[t], mixed, projected, attended, fc, expanded, update, output });
      return output;
    });
    if (rows) snapshot.layers.push(rows);
  }
  function readout(state, row) {
    const final = layerNorm(
      state,
      tensors['transformer.ln_f.weight'],
      tensors['transformer.ln_f.bias'],
      row?.ln,
    );
    return softmax(linear(final, tensors['lm_head.weight']), row?.sm);
  }
  if (snapshot) {
    for (const state of states) {
      const row = { ln: {}, sm: {} };
      readout(state, row);
      snapshot.final.push(row);
    }
    return makeNanoInspector(snapshot);
  }
  const probabilities = readout(states[last]);
  const prediction = {
    probabilities,
    nextToken: tokens[probabilities.indexOf(Math.max(...probabilities))],
  };
  if (traces) {
    traces.output = [probabilities];
    prediction.traces = traces;
  }
  return prediction;
}
