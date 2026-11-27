import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { test } from 'node:test';
import { models, chapters, isModern } from '../src/detail/catalog.js';
import { classicLayout } from '../src/detail/classic-layout.js';
import { modernExample } from '../src/detail/modern.js';
import { predictNano } from '../src/nano.js';

test('each walkthrough entry resolves to a visible tensor in the six published graphs', () => {
  assert.deepEqual(
    models.map((model) => model.id),
    ['gpt2-small', 'nanogpt', 'gpt2-xl', 'gpt3', 'llama', 'qwen'],
  );
  for (const { id } of models) {
    const graph = isModern(id) ? modernExample(id).layout : classicLayout(id);
    const first = graph.layers[0].nodes();
    const last = graph.layers.at(-1).nodes();
    for (const item of chapters(id)) {
      assert.ok(
        [...graph.root, ...first, ...last].some((node) => node.op === item.op || node.stage === item.op),
        id + ' / ' + item.op,
      );
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
