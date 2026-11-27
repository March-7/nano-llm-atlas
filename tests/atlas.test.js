import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { test } from 'node:test';
import { models } from '../src/catalog.js';
import { buildArchitecture } from '../src/geometry.js';
import { predictNano } from '../src/nano.js';

test('each inspector stage maps to geometry in both compact and exploded views', () => {
  for (const model of models) {
    for (const spread of [0, 1]) {
      const { stations, parts } = buildArchitecture(model.id, spread);
      assert.deepEqual(
        stations.map((s) => s.stage),
        model.stages.map((s) => s.id),
      );
      for (const { id } of model.stages) assert.ok(parts.some((part) => part.stage === id));
      const count = (prefix) => parts.filter((part) => part.id.startsWith(prefix)).length;
      if (model.id === 'nanogpt') {
        assert.equal(count('masked-scores-'), 3);
        assert.equal(count('position-embedding'), 1);
      } else {
        assert.equal(count('shared-key'), 1);
        assert.equal(count('shared-value'), 1);
        assert.equal(count('query-head-'), model.id === 'llama' ? 4 : 6);
      }
      if (model.id === 'qwen') {
        assert.equal(count('delta-state-'), 3);
        assert.equal(count('delta-ffn-'), 3);
        assert.equal(count('ffn-gate'), 1);
      }
    }
  }
});

test('checkpoint inference reproduces the PyTorch activations and sorts the complete task', async (t) => {
  const asset = new URL('../assets/gpt-nano-sort-model.json', import.meta.url);
  t.mock.method(globalThis, 'fetch', async (url) => {
    assert.equal(String(url), asset.href);
    return new Response(await readFile(asset));
  });

  // Last input position of the first sample in Bycroft's PyTorch fixture.
  // https://github.com/bbycroft/llm-viz/blob/9da93742382f1bf36c020c38a1ace454e82c4490/public/gpt-nano-sort-t0-partials.json
  const ref = {
    embed: [-0.03507731, 0.08334499, -0.0539934],
    norm: [-0.7673949, 1.80938005, -1.14414966],
    attention: [0.13510682, 0.17991142, 0.02990245],
    ffn: [-0.07287187, 1.38484704, 1.20068181],
    output: [0.99998367, 0.0000044, 0.00001191],
  };
  const result = await predictNano('AACBAB', { trace: true });
  for (const [stage, values] of Object.entries(ref)) {
    values.forEach((expected, i) => {
      assert.ok(Math.abs(result.traces[stage][0][i] - expected) < 1e-6, stage + ' channel ' + i);
    });
  }
  assert.deepEqual(
    result.traces.attention.map((row) => row.length),
    [6, 6, 6],
  );
  for (const row of result.traces.attention) {
    assert.ok(Math.abs(row.reduce((sum, v) => sum + v, 0) - 1) < 1e-6);
  }
  const plain = await predictNano('AACBAB');
  assert.deepEqual(plain.probabilities, result.probabilities);
  assert.equal(plain.traces, undefined);
  for (const invalid of ['', 'AXB', 'A'.repeat(12)]) {
    await assert.rejects(predictNano(invalid), RangeError);
  }

  for (let n = 0; n < 3 ** 6; n++) {
    const input = Array.from(n.toString(3).padStart(6, '0'), (digit) => 'ABC'[Number(digit)]).join('');
    let output = '';
    while (output.length < 6) output += (await predictNano(input + output)).nextToken;
    assert.equal(output, [...input].sort().join(''), input);
  }
});
