import assert from 'node:assert/strict';
import { test } from 'node:test';
import { modernExample } from '../src/detail/modern.js';

function close(actual, expected, precision = 1e-11) {
  assert.ok(Math.abs(actual - expected) < precision, `${actual} ≠ ${expected}`);
}

function read(sample, op, layer = -1, head = -1) {
  const node = { op, layer, head };
  return (x, y, z = 0) => sample.value(node, x, y, z);
}

test('every modern teaching tensor has computed scalars, formulas, geometry and resolvable inputs', function () {
  for (const [model, expectedCells] of [
    ['llama', 21184],
    ['qwen', 131714],
  ]) {
    const { layout, sample } = modernExample(model);
    assert.match(layout.scope.en, /Fixed teaching weights/);
    const nodes = [...layout.root, ...layout.layers.flatMap((l) => l.nodes())];
    assert.equal(nodes.length, layout.nodeCount);
    const ids = new Set(nodes.map((n) => n.id));
    assert.equal(ids.size, nodes.length);
    let count = 0;
    for (const n of nodes) {
      assert.ok(
        n.deps.every((d) => ids.has(d)),
        `dangling input of ${n.id}`,
      );
      assert.equal(n.deps.length, n.depMaps.length);
      assert.equal(n.axes.length, 3);
      for (let z = 0; z < n.cz; z++)
        for (let y = 0; y < n.cy; y++)
          for (let x = 0; x < n.cx; x++) {
            const value = sample.value(n, x, y, z);
            if (n.op === 'attnMtx' && x > y) assert.equal(value, -Infinity);
            else assert.ok(Number.isFinite(value), n.id);
            count++;
          }
      const info = sample.explain(n, n.cx - 1, n.cy - 1, n.cz - 1);
      assert.ok(info.formula && info.substitution && info.source.includes('teaching'));
      assert.doesNotMatch(info.substitution, /NaN|undefined/);
    }
    assert.equal(count, expectedCells);
    assert.equal(sample.value(nodes[0], -1, 0), null);
  }
  assert.throws(() => modernExample('unknown'), RangeError);
});

test('fixed-weight forwards match independent numerical references and expose actual shared KV and partial RoPE', function () {
  // Gold scalars from an independent NumPy implementation, using the displayed
  // teaching matrices and a batched [heads, keys, values] recurrence.
  const expected = {
    llama: [
      0.04178106738292827, 0.02959241293103159, 0.0482337380857981, 0.01963836033115231,
      0.06903819012046869, 0.13274150746440164,
    ],
    qwen: [
      0.24465381800431016, 0.037387936694082725, 0.04376117023961839, 0.06422684991696814,
      0.015629772938975196, 0.020629298985565585,
    ],
  };
  for (const model of ['llama', 'qwen']) {
    const { sample, layout } = modernExample(model);
    const p = read(sample, 'logitsSoftmax');
    expected[model].forEach((e, v) => close(p(7, v), e));
    for (let t = 0; t < 8; t++)
      close(
        Array.from({ length: 16 }, (_, v) => p(t, v)).reduce((a, b) => a + b),
        1,
      );
    const finalLayer = layout.layers.at(-1);
    const l = finalLayer.index;
    const nodes = finalLayer.nodes();
    assert.equal(nodes.filter((n) => n.op === 'kBlock').length, 1);
    assert.equal(nodes.filter((n) => n.op === 'vBlock').length, 1);
    assert.equal(nodes.filter((n) => n.op === 'qBlock').length, layout.shape.nHeads);
    for (let h = 0; h < layout.shape.nHeads; h++) {
      const attn = read(sample, 'attnMtxSm', l, h);
      const out = read(sample, 'vOutBlock', l, h);
      const value = read(sample, 'vBlock', l);
      for (let t = 0; t < 8; t++) {
        let sum = 0,
          weighted = 0;
        for (let s = 0; s < 8; s++) {
          if (s > t) assert.equal(attn(s, t), 0);
          sum += attn(s, t);
          weighted += attn(s, t) * value(s, 4);
        }
        close(sum, 1);
        close(out(t, 4), weighted);
        if (model === 'qwen') {
          const gate = read(sample, 'attnGate', l, h)(t, 4);
          close(read(sample, 'attnGated', l, h)(t, 4), weighted / (1 + Math.exp(-gate)));
          const pre = read(sample, 'qNorm', l, h),
            post = read(sample, 'qRotary', l, h);
          for (let d = 2; d < 8; d++) assert.equal(post(t, d), pre(t, d));
          close(post(t, 0), pre(t, 0) * Math.cos(t) - pre(t, 1) * Math.sin(t));
        }
      }
    }
  }
});

test('each DeltaNet head advances its state from the preceding token before reading it out', function () {
  const { sample } = modernExample('qwen');
  for (let l = 0; l < 3; l++) {
    const key = read(sample, 'deltaKey', l);
    const query = read(sample, 'deltaQuery', l);
    for (let h = 0; h < 3; h++) {
      const state = read(sample, 'deltaState', l, h);
      const prev = read(sample, 'deltaPrevious', l, h);
      const out = read(sample, 'deltaRead', l, h);
      for (let t = 0; t < 8; t++) {
        const retention = read(sample, 'deltaDecay', l)(t, h);
        const beta = read(sample, 'deltaBeta', l)(t, h);
        assert.ok(retention > 0 && retention < 1 && beta > 0 && beta < 1);
        for (let d = 0; d < 8; d++) {
          let recall = 0,
            result = 0;
          for (let k = 0; k < 8; k++) {
            close(prev(k, d, t), t ? state(k, d, t - 1) : 0);
            recall += key(t, k) * retention * prev(k, d, t);
            result += query(t, k) * state(k, d, t);
          }
          const update = beta * (read(sample, 'deltaValue', l, h)(t, d) - recall);
          for (let k = 0; k < 8; k++) close(state(k, d, t), retention * prev(k, d, t) + key(t, k) * update);
          close(out(t, d), result);
        }
      }
    }
  }
  close(read(sample, 'deltaState', 2, 2)(4, 3, 6), -0.046410120189356446);
  close(read(sample, 'attnMtxSm', 3, 4)(3, 7), 0.3472868724371673);
});
