import assert from 'node:assert/strict';
import test from 'node:test';
import { classicLayout } from '../src/detail/classic-layout.js';

// Counts and coordinates checked against genGptModelLayout at bbycroft/llm-viz@9da9374.
const models = [
  ['nanogpt', 3, 3, 206],
  ['gpt2-small', 12, 12, 2294],
  ['gpt2-xl', 48, 25, 13070],
  ['gpt3', 96, 96, 94286],
];

for (const [id, depth, heads, count] of models) {
  test(id + ' contains every tensor and all references resolve across folds', () => {
    const graph = classicLayout(id);
    assert.equal(graph.nodeCount, count);
    assert.equal(graph.root.length, 14);
    assert.equal(graph.layers.length, depth);
    const seen = new Set(graph.root.map((node) => node.id));
    const used = new Set(graph.root.flatMap((node) => node.deps));
    for (const layer of graph.layers) {
      const nodes = layer.nodes();
      const residual = nodes.find((node) => node.op === 'attnResidual');
      const prior = layer.index ? 'L' + (layer.index - 1) + '.H-1.mlpResidual' : 'L-1.H-1.residual0';
      assert.ok(residual.deps.includes(prior));
      assert.equal(new Set(nodes.map((node) => node.head).filter((h) => h >= 0)).size, heads);
      const output = nodes.find((node) => node.op === 'attnOut');
      assert.equal(output.depMaps.filter((map) => map.role === 'concat').length, heads);
      assert.equal(nodes.find((node) => node.op === 'vOutBlock').dotLen, graph.shape.T);
      assert.equal(nodes.find((node) => node.op === 'mlpResult').dotLen, 4 * graph.shape.C);
      for (const node of nodes) {
        assert.ok(!seen.has(node.id));
        seen.add(node.id);
        assert.equal(node.deps.length, node.depMaps.length);
        for (const input of node.deps) used.add(input);
      }
    }
    assert.equal(seen.size, count);
    for (const input of used) assert.ok(seen.has(input), input);
    assert.deepEqual(graph.root.find((node) => node.op === 'ln_f.lnAgg1').deps, [
      'L' + (depth - 1) + '.H-1.mlpResidual',
    ]);
  });
}

test('NanoGPT uses the original cell scale and projection coordinates', () => {
  const g = classicLayout('nanogpt');
  const first = [...g.root, ...g.layers[0].nodes()];
  const coordinates = (op) => {
    const { x, y, z, dx, dy, dz } = first.find((node) => node.op === op);
    return [x, y, z, dx, dy, dz];
  };
  assert.equal(g.cell, 1.5);
  assert.deepEqual(coordinates('residual0'), [-8.25, 13.5, -0.75, 16.5, 72, 1.5]);
  assert.deepEqual(coordinates('qWeightBlock'), [-176.25, 160.5, -60.75, 72, 24, 1.5]);
  assert.deepEqual(coordinates('attnMtx'), [-92.25, 196.5, -78.75, 16.5, 16.5, 1.5]);
  assert.deepEqual(coordinates('logitsSoftmax'), [-36.75, 1702.5, -0.75, 16.5, 4.5, 1.5]);
});

test('GPT-2 XL repeats its template in columns and retains packed head slices', () => {
  const g = classicLayout('gpt2-xl');
  const [first, folded] = [g.layers[0], g.layers[12]];
  const firstNodes = first.nodes();
  assert.deepEqual(folded.offset, [33920, 0, 0]);
  folded.nodes().forEach((n, i) => {
    assert.equal(n.x, firstNodes[i].x + folded.offset[0]);
    assert.equal(n.y, firstNodes[i].y);
    assert.equal(n.z, firstNodes[i].z);
  });
  const qkv = firstNodes.find((n) => n.op === 'qkvBlock');
  assert.equal(qkv.cy, 3 * g.shape.A);
  assert.deepEqual(
    qkv.depMaps.filter((d) => d.role === 'bias').map((d) => d.span.start),
    [0, 64, 128],
  );
  const score = firstNodes.find((n) => n.op === 'attnMtx');
  assert.deepEqual(score.deps, [qkv.id, qkv.id]);
  assert.deepEqual(score.depMaps[1].offset, [0, 64, 0]);
  assert.equal(g.shape.T, 1024);
  assert.equal(classicLayout('gpt3').shape.context, 2048);
});
