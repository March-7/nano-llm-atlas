import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { chapters } from '../src/detail/catalog.js';
import { chapterSequence } from '../src/detail/chapter-sequences.js';
import { classicLayout } from '../src/detail/classic-layout.js';
import { modernExample } from '../src/detail/modern.js';
import { inspectNano } from '../src/nano.js';

const display = (value) => Number(value.toPrecision(6)).toString();

function graph(modelId, actual) {
  const modern = ['llama', 'qwen'].includes(modelId);
  const { layout, sample } = modern
    ? modernExample(modelId)
    : { layout: classicLayout(modelId), sample: actual };
  const byLayer = new Map();
  function group(layer) {
    if (!byLayer.has(layer)) {
      const nodes = layer < 0 ? layout.root : layout.layers[layer].nodes();
      byLayer.set(layer, new Map(nodes.map((n) => [n.id, n])));
    }
    return byLayer.get(layer);
  }
  const find = (op, layer, head) => group(layer).get(`L${layer}.H${head}.${op}`);
  const resolve = (id) => group(Number(id.slice(1, id.indexOf('.')))).get(id);
  function frames(node, x, y, z = 0) {
    return chapterSequence({ modelId, node, x, y, z, sample, find, resolve });
  }
  return { layout, sample, group, find, resolve, frames };
}

test('all 42 chapter routes provide three or four frames using real, in-bounds source and destination slices', () => {
  let routes = 0;
  for (const id of ['nanogpt', 'gpt2-small', 'gpt2-xl', 'gpt3', 'llama', 'qwen']) {
    const g = graph(id);
    for (const chapter of chapters(id)) {
      const last = id === 'qwen' && chapter.op === 'deltaState' ? 2 : g.layout.layers.length - 1;
      const node = [...g.group(-1).values(), ...g.group(last).values()].findLast(
        (n) => n.op === chapter.op,
      );
      assert.ok(node);
      for (const [x, y, z] of [
        [0, 0, 0],
        [node.cx - 1, node.cy - 1, node.cz - 1],
      ]) {
        const frames = g.frames(node, x, y, z);
        assert.ok(frames.length >= 3 && frames.length <= 4, node.id);
        for (const f of frames) {
          assert.ok(f.title.zh && f.title.en && f.formula);
          assert.doesNotMatch(f.formula, /undefined|NaN|null/);
          assert.ok(f.sources.length > 0);
          for (const s of [f.output, ...f.sources]) {
            assert.equal(s.node, g.resolve(s.node.id));
            assert.equal(s.indices.length, 3);
            assert.ok(Number.isInteger(s.axis) && s.axis >= -1 && s.axis <= 2);
            assert.ok(Number.isInteger(s.length) && s.length >= 1);
            if (s.axis === -1) assert.equal(s.length, 1);
            for (const [axis, dim] of [s.node.cx, s.node.cy, s.node.cz].entries()) {
              const end = s.indices[axis] + (s.axis === axis ? s.length : 1);
              assert.ok(Number.isInteger(s.indices[axis]) && s.indices[axis] >= 0 && end <= dim);
            }
          }
        }
      }
      routes++;
    }
  }
  assert.equal(routes, 42);
});

test('Llama RMSNorm reduces the selected token and uses its actual inverse RMS and channel gain', () => {
  const g = graph('llama'),
    t = 5,
    d = 12;
  const n = g.find('ln1.lnResid', 0, -1),
    input = g.find('embed', -1, -1);
  const steps = g.frames(n, t, d);
  const row = steps[0].sources[0];
  assert.equal(row.node, input);
  assert.deepEqual(row.indices, [t, 0, 0]);
  assert.equal(row.axis, 1);
  assert.equal(row.length, 32);
  const value = (node, x, y) => g.sample.value(node, x, y);
  const moment =
    Array.from({ length: 32 }, (_, i) => value(input, t, i) ** 2).reduce((a, b) => a + b, 0) / 32;
  const inverse = 1 / Math.sqrt(moment + 1e-5);
  const gain = value(g.find('ln1.lnResid.gain', 0, -1), 0, d);
  assert.ok(steps[0].formula.endsWith(display(moment)));
  assert.ok(steps[1].formula.endsWith(display(inverse)));
  assert.ok(steps[2].formula.endsWith(display(gain)));
  assert.ok(steps[3].formula.endsWith(display(value(input, t, d) * inverse * gain)));
  assert.deepEqual(steps[3].output.indices, [t, d, 0]);
  assert.equal(steps[3].output.node, n);
  assert.ok(steps.every((s) => !s.formula.includes('β')));
});

test('RMSNorm reports cached partial squares and mean squares at exactly the visited channel count', () => {
  const g = graph('llama'),
    t = 3,
    d = 4;
  const n = g.find('ln1.lnResid', 0, -1),
    input = g.find('embed', -1, -1);
  let reads = 0;
  const sample = {
    value(...args) {
      reads++;
      return g.sample.value(...args);
    },
  };
  const [phase] = chapterSequence({
    modelId: 'llama',
    node: n,
    x: t,
    y: d,
    z: 0,
    find: g.find,
    resolve: g.resolve,
    sample,
  });
  const atBuild = reads;
  assert.equal(phase.readoutLength, 32);
  for (const k of [1, 11, 32]) {
    const sum = Array.from({ length: k }, (_, i) => g.sample.value(input, t, i) ** 2).reduce(
      (a, b) => a + b,
      0,
    );
    assert.ok(phase.readout(k).startsWith(`k = ${k}/32`));
    assert.ok(phase.readout(k).includes(`Σᵢ<ₖ xᵢ² = ${display(sum)}`));
    assert.ok(phase.readout(k).endsWith(display(sum / k)));
  }
  assert.equal(phase.readout(0), phase.readout(1));
  assert.equal(phase.readout(900), phase.readout(32));
  assert.equal(reads, atBuild);
  const classic = graph('gpt2-xl');
  const [symbolic] = classic.frames(classic.find('ln2.lnResid', 16, -1), 22, 18);
  assert.equal(symbolic.readoutLength, 1600);
  assert.ok(symbolic.readout(13).includes('μ(k) = Σᵢ<ₖ xᵢ / 13'));
});

test('Qwen retains the previous time slice, recalls over key channels and writes an outer-product cell', () => {
  const g = graph('qwen'),
    level = 2,
    head = 2,
    t = 4,
    a = 6,
    d = 3;
  const n = g.find('deltaState', level, head),
    steps = g.frames(n, a, d, t);
  assert.deepEqual(
    steps.map((s) => s.output.node.op),
    ['deltaRetained', 'deltaMemory', 'deltaWrite', 'deltaState'],
  );
  const prior = steps[0].sources.find((s) => s.node === n);
  assert.deepEqual(prior.indices, [a, d, t - 1]);
  const decay = steps[0].sources.find((s) => s.node.op === 'deltaDecay');
  assert.deepEqual(decay.indices, [t, head, 0]);
  const retained = steps[1].sources.find((s) => s.node.op === 'deltaRetained');
  assert.deepEqual(retained.indices, [0, d, t]);
  assert.equal(retained.axis, 0);
  assert.equal(retained.length, n.cx);
  const key = g.find('deltaKey', level, -1);
  const v = g.sample.value;
  const kept = v(n, a, d, t - 1) * v(decay.node, t, head);
  const memory = Array.from({ length: n.cx }, (_, i) => v(key, t, i) * v(retained.node, i, d, t)).reduce(
    (a, b) => a + b,
    0,
  );
  const update =
    v(g.find('deltaBeta', level, -1), t, head) * (v(g.find('deltaValue', level, head), t, d) - memory);
  [kept, memory, update, kept + v(key, t, a) * update].forEach((value, i) => {
    assert.ok(steps[i].formula.endsWith(display(value)), steps[i].formula);
  });
  const first = g.frames(n, a, d, 0);
  assert.ok(!first[0].sources.some((s) => s.node === n));
  assert.match(first[0].formula, /Sinit/);
});

test('large classic embeddings keep unknown token IDs symbolic and packed weights are read as parameters', () => {
  const g = graph('gpt3'),
    t = 17,
    d = 389;
  const embed = g.frames(g.find('residual0', -1, -1), t, d);
  assert.ok(embed[1].formula.includes('ids[17]'));
  assert.equal(embed[1].output.node.op, 'tokEmbedObj');
  assert.deepEqual(embed[1].output.indices, [0, d, 0]);
  assert.equal(embed[1].output.axis, 0);
  assert.equal(embed[1].output.length, 50257);
  const n = g.find('qkvWeightBlock', 95, 95);
  const steps = g.frames(n, 721, 260);
  assert.equal(steps[0].output.node, n);
  assert.match(steps[0].formula, /WV\[h=95,a=4,c=721\]/);
  assert.equal(steps.at(-1).output.node.op, 'qkvBlock');
  assert.deepEqual(steps.at(-1).output.indices, [0, 260, 0]);
  assert.ok(steps.at(-1).sources.some((s) => s.node.op === 'vBiasBlock' && s.indices[1] === 4));
  const proj = g.frames(g.find('projWeight', 95, -1), 12287, 222);
  assert.equal(proj[1].output.node.head, 95);
  assert.equal(proj[1].output.indices[1], 127);
  assert.equal(proj.at(-1).sources.filter((s) => s.node.op === 'vOutBlock').length, 96);
});

test('attention resolves packed K offsets, shared RoPE keys, causal masking and the downstream Qwen gate', () => {
  const classic = graph('gpt2-xl');
  const head = classic.find('attnMtxSm', 47, 24);
  const packed = classic.frames(head, 2, 4);
  assert.deepEqual(
    packed[0].sources.map((s) => s.indices),
    [
      [4, 0, 0],
      [2, 64, 0],
    ],
  );
  assert.ok(packed[0].sources.every((s) => s.node.op === 'qkvBlock' && s.axis === 1 && s.length === 64));
  assert.ok(packed[1].sources.some((s) => s.node.op === 'attnMtx' && s.length === 5));
  const masked = classic.frames(head, 7, 3);
  assert.equal(masked.length, 3);
  assert.ok(masked.at(-1).formula.endsWith('= 0'));
  assert.ok(masked.every((f) => f.sources.every((s) => s.node.op !== 'qkvBlock')));
  const qwen = graph('qwen');
  const n = qwen.find('attnMtxSm', 3, 5),
    steps = qwen.frames(n, 2, 4);
  assert.deepEqual(
    steps[0].sources.map((s) => [s.node.op, s.node.head]),
    [
      ['qRotary', 5],
      ['kRotary', -1],
    ],
  );
  assert.equal(steps[2].output.node, n);
  const final = steps[3].output;
  assert.equal(final.node.op, 'attnGated');
  assert.deepEqual(final.indices, [4, 0, 0]);
  assert.ok(steps[3].formula.endsWith(display(qwen.sample.value(final.node, ...final.indices))));
});

test('SwiGLU follows two real projections and the vocabulary readout reduces every vocabulary entry', () => {
  const g = graph('llama');
  const act = g.find('mlpAct', 0, -1);
  const stages = g.frames(act, 7, 105);
  assert.deepEqual(
    stages.map((s) => s.output.node.op),
    ['mlpGate', 'mlpUp', 'mlpGate', 'mlpAct'],
  );
  assert.ok(
    stages[0].sources.some((s) => s.node.op === 'mlpGateWeight' && s.axis === 0 && s.length === 32),
  );
  assert.ok(stages[3].formula.endsWith(display(g.sample.value(act, 7, 105))));
  const probs = g.find('logitsSoftmax', -1, -1);
  const softmax = g.frames(probs, 6, 3);
  assert.equal(softmax[1].sources[0].axis, 1);
  assert.equal(softmax[1].sources[0].length, 16);
  assert.ok(softmax.at(-1).formula.endsWith(display(g.sample.value(probs, 6, 3))));
  const logits = g.find('logits', -1, -1);
  const values = Array.from({ length: 16 }, (_, j) => g.sample.value(logits, 6, j));
  const first = values.slice(0, 5),
    fullMax = Math.max(...values);
  assert.equal(softmax[1].readoutLength, 16);
  assert.ok(softmax[1].readout(5).endsWith(display(Math.max(...first))));
  const normalizer = first.reduce((total, v) => total + Math.exp(v - fullMax), 0);
  assert.ok(softmax[2].readout(5).endsWith(display(1 / normalizer)));
});

test('NanoGPT reads checkpoint LayerNorm and GELU values, and leaves an uncomputed prefix position symbolic', async (t) => {
  const weights = new URL('../assets/gpt-nano-sort-model.json', import.meta.url);
  t.mock.method(globalThis, 'fetch', async () => new Response(await readFile(weights)));
  const sample = await inspectNano('ACB');
  const g = graph('nanogpt', sample);
  const ln = g.find('ln2.lnResid', 2, -1),
    norm = g.frames(ln, 2, 9);
  const raw = g.find('attnResidual', 2, -1);
  const sum = Array.from({ length: 7 }, (_, i) => sample.value(raw, 2, i)).reduce((a, b) => a + b, 0);
  assert.equal(norm[0].readoutLength, 48);
  assert.ok(norm[0].readout(7).includes(`Σᵢ<ₖ xᵢ = ${display(sum)}`));
  assert.ok(norm[0].readout(7).endsWith(display(sum / 7)));
  assert.equal(norm[0].output.node.op, 'ln2.lnAgg1');
  assert.ok(norm[0].formula.endsWith(display(sample.value(g.find('ln2.lnAgg1', 2, -1), 2, 0))));
  assert.ok(norm[2].formula.includes('β[9]'));
  assert.ok(norm[3].formula.endsWith(display(sample.value(ln, 2, 9))));
  const activation = g.find('mlpAct', 2, -1),
    gelu = g.frames(activation, 103, 2);
  assert.ok(
    gelu[0].sources.some((s) => s.node.op === 'mlpFcWeight' && s.axis === 1 && s.indices[0] === 103),
  );
  assert.ok(gelu.at(-1).formula.endsWith(display(sample.value(activation, 103, 2))));
  const unknown = g.frames(g.find('residual0', -1, -1), 8, 9);
  assert.equal(unknown[1].output.axis, 0);
  assert.doesNotMatch(unknown.at(-1).formula, / = -?\d+(\.\d+)?$/);
});
