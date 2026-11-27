import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { test } from 'node:test';
import { inspectNano, predictNano } from '../src/nano.js';

const checkpoint = new URL('../assets/gpt-nano-sort-model.json', import.meta.url);

function mockCheckpoint(t) {
  t.mock.method(globalThis, 'fetch', async (url) => {
    assert.equal(String(url), checkpoint.href);
    return new Response(await readFile(checkpoint));
  });
}

function close(actual, expected, label, tolerance = 3e-6) {
  assert.equal(typeof actual, 'number', label);
  assert.ok(Math.abs(actual - expected) < tolerance, `${label}   ${actual} ≠ ${expected}`);
}

test('all 85,728 learned scalars have their original checkpoint indices in the drawing', async function (t) {
  mockCheckpoint(t);
  const i = await inspectNano('A');
  const source = JSON.parse(await readFile(checkpoint, 'utf8'));
  let count = 0;
  function verify(key, address) {
    const tensor = source[key];
    const bytes = Buffer.from(tensor.data, 'base64');
    for (let j = 0; j < bytes.length / 4; j++) {
      const row = tensor.shape.length === 1 ? j : Math.floor(j / tensor.shape[1]);
      const col = tensor.shape.length === 1 ? 0 : j % tensor.shape[1];
      const [op, layer, head, x, y] = address(row, col);
      const node = { op, layer, head };
      assert.equal(i.value(node, x, y), bytes.readFloatLE(4 * j), `${key}[${row},${col}]`);
      if (j === 0 || j === Math.floor(bytes.length / 8) || j === bytes.length / 4 - 1) {
        const indices = tensor.shape.length === 1 ? `${row}` : `${row}, ${col}`;
        assert.equal(i.explain(node, x, y).source, `${key}[${indices}]`);
      }
      count++;
    }
    delete source[key];
  }
  verify('transformer.wte.weight', (r, c) => ['tokEmbedObj', -1, -1, r, c]);
  verify('transformer.wpe.weight', (r, c) => ['posEmbedObj', -1, -1, r, c]);
  verify('lm_head.weight', (r, c) => ['lmHeadWeight', -1, -1, c, r]);
  for (const kind of ['weight', 'bias']) {
    const slot = kind === 'weight' ? 'lnSigma' : 'lnMu';
    verify(`transformer.ln_f.${kind}`, (r) => [`ln_f.${slot}`, -1, -1, 0, r]);
    for (let l = 0; l < 3; l++) {
      const prefix = `transformer.h.${l}.`;
      for (const n of [1, 2]) verify(`${prefix}ln_${n}.${kind}`, (r) => [`ln${n}.${slot}`, l, -1, 0, r]);
      const weight = kind === 'weight';
      verify(`${prefix}attn.c_attn.${kind}`, (r, c) => [
        'qkv'[Math.floor(r / 48)] + (weight ? 'WeightBlock' : 'BiasBlock'),
        l,
        Math.floor(r / 16) % 3,
        c,
        r % 16,
      ]);
      verify(`${prefix}attn.c_proj.${kind}`, (r, c) => [weight ? 'projWeight' : 'projBias', l, -1, c, r]);
      verify(`${prefix}mlp.c_fc.${kind}`, (r, c) => [weight ? 'mlpFcWeight' : 'mlpFcBias', l, -1, r, c]);
      verify(`${prefix}mlp.c_proj.${kind}`, (r, c) => [
        weight ? 'mlpProjWeight' : 'mlpProjBias',
        l,
        -1,
        c,
        r,
      ]);
    }
  }
  assert.equal(count, 85728);
  assert.deepEqual(Object.keys(source).sort(), [
    'config',
    ...[0, 1, 2].map((l) => `transformer.h.${l}.attn.bias`),
  ]);
});

test('computed cells reproduce PyTorch intermediates across three layers and track the current prefix', async function (t) {
  mockCheckpoint(t);
  const i = await inspectNano('AACBABAAAAA');
  // Selected scalar coordinates from batch 0 of the upstream PyTorch fixture.
  // https://github.com/bbycroft/llm-viz/blob/9da93742382f1bf36c020c38a1ace454e82c4490/public/gpt-nano-sort-t0-partials.json
  const cases = [
    ['residual0', -1, -1, 9, 41, -0.01537777483],
    ['ln1.lnResid', 0, -1, 5, 39, -0.696860671],
    ['qBlock', 0, 0, 6, 8, -0.1457741857],
    ['kBlock', 0, 1, 4, 15, 0.339645952],
    ['vBlock', 0, 2, 10, 2, -0.06708676368],
    ['attnMtx', 0, 2, 3, 7, 0.3097643852],
    ['attnMtxSm', 0, 2, 3, 7, 0.1629435718],
    ['vOutBlock', 0, 1, 6, 3, -0.001482534804],
    ['attnOut', 0, -1, 5, 39, 0.02446414903],
    ['attnResidual', 0, -1, 5, 39, -0.007425945252],
    ['ln2.lnResid', 0, -1, 5, 31, 0.3879418373],
    ['mlpFc', 0, -1, 137, 8, 0.05657975748],
    ['mlpAct', 0, -1, 181, 9, 0.006331671029],
    ['mlpResult', 0, -1, 7, 21, 0.00415666448],
    ['mlpResidual', 0, -1, 7, 21, 0.009294029325],
    ['mlpResidual', 1, -1, 8, 26, -0.3381918073],
    ['mlpResidual', 2, -1, 10, 47, -1.051177382],
    ['ln_f.lnResid', -1, -1, 7, 31, -1.371796012],
    ['logits', -1, -1, 10, 0, -3.559364796],
    ['logitsSoftmax', -1, -1, 10, 2, 0.9999638796],
  ];
  for (const [op, layer, head, x, y, expected] of cases) {
    const node = { op, layer, head };
    close(i.value(node, x, y), expected, `${op} L${layer} H${head}`);
    const info = i.explain(node, x, y);
    assert.ok(info.formula && info.substitution.includes('='));
  }
  const short = await inspectNano('AB');
  assert.equal(short.value({ op: 'residual0' }, 2, 1), null);
  assert.equal(short.value({ op: 'mlpAct', layer: 1 }, 19, 2), null);
  assert.equal(typeof short.value({ op: 'posEmbedObj' }, 10, 2), 'number');
  assert.equal(short.value({ op: 'idxObj' }, 1, 0), 1);
  assert.equal(short.value({ op: 'idxObj' }, 1, 0, 1), null);
  assert.equal(short.value({ op: 'idxObj' }, -1, 0), null);
  assert.equal(short.value({ op: 'unrecognized' }, 0, 0), null);
  await assert.rejects(inspectNano('A'.repeat(12)), RangeError);
  for (const prefix of ['A', 'CBABAC', 'AAABBCABABC']) {
    const detail = await inspectNano(prefix);
    const compact = await predictNano(prefix, { trace: true });
    for (let v = 0; v < 3; v++)
      assert.equal(detail.value({ op: 'logitsSoftmax' }, prefix.length - 1, v), compact.probabilities[v]);
  }
});

test('all three layers expose consistent causal masks, attention aggregates, norms and FFN reductions', async function (t) {
  mockCheckpoint(t);
  const prefix = 'CABABC';
  const i = await inspectNano(prefix);
  for (let layer = 0; layer < 3; layer++) {
    for (let pos = 0; pos < prefix.length; pos++) {
      const v = (op, x, y, head = -1) => i.value({ op, layer, head }, x, y);
      for (let head = 0; head < 3; head++) {
        const scores = [];
        for (let k = 0; k < prefix.length; k++) {
          const score = v('attnMtx', k, pos, head);
          if (k > pos) {
            assert.equal(score, -Infinity);
            assert.equal(v('attnMtxSm', k, pos, head), 0);
          } else {
            let dot = 0;
            for (let d = 0; d < 16; d++) dot += v('qBlock', pos, d, head) * v('kBlock', k, d, head);
            close(score, dot / 4, 'scaled dot product');
            scores.push(score);
          }
        }
        const m = Math.max(...scores);
        const z = scores.reduce((total, a) => total + Math.exp(a - m), 0);
        close(v('attnMtxAgg1', 0, pos, head), 1 / z, 'inverse exponential sum');
        close(v('attnMtxAgg2', 0, pos, head), m, 'row maximum');
        let mixed = 0;
        scores.forEach((s, k) => {
          const p = Math.exp(s - m) / z;
          close(v('attnMtxSm', k, pos, head), p, 'attention probability');
          mixed += p * v('vBlock', k, 7, head);
        });
        close(v('vOutBlock', pos, 7, head), mixed, 'weighted values');
      }
      for (const ln of ['ln1', 'ln2']) {
        const inOp =
          ln === 'ln2'
            ? { op: 'attnResidual', layer }
            : layer
              ? { op: 'mlpResidual', layer: layer - 1 }
              : { op: 'residual0' };
        const input = Array.from({ length: 48 }, (_, d) => i.value(inOp, pos, d));
        const mean = input.reduce((a, b) => a + b, 0) / 48;
        const variance = input.reduce((a, b) => a + (b - mean) ** 2, 0) / 48;
        const inv = 1 / Math.sqrt(variance + 1e-5);
        close(v(ln + '.lnAgg1', pos, 0), mean, 'mean');
        close(v(ln + '.lnAgg2', pos, 0), inv, 'inverse standard deviation');
        const channel = 29;
        const expected =
          (input[channel] - mean) * inv * v(ln + '.lnSigma', 0, channel) + v(ln + '.lnMu', 0, channel);
        close(v(ln + '.lnResid', pos, channel), expected, 'normalized activation');
      }
      let proj = v('mlpProjBias', 0, 17);
      for (let j = 0; j < 192; j++) proj += v('mlpAct', j, pos) * v('mlpProjWeight', j, 17);
      close(v('mlpResult', pos, 17), proj, 'FFN contraction');
    }
  }
});
