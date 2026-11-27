import { test } from 'node:test';
import assert from 'node:assert/strict';
import { basis, cameraMap } from '../src/detail/tensor-paint.js';
import { paintComputation, phaseReadCount, readCount, spanBounds } from '../src/detail/flow-paint.js';

const tensor = (id, kind, x, channels) => ({
  id,
  kind,
  x,
  y: 0,
  z: 0,
  dx: 4,
  dy: channels,
  dz: 1,
  cx: 4,
  cy: channels,
  cz: 1,
});
const source = tensor('input', 'i', 0, 32),
  result = tensor('inverse', 'a', 16, 1);
const row = { node: source, indices: [2, 0, 0], axis: 1, length: 32 };
const inverse = { node: result, indices: [2, 0, 0], axis: -1, length: 1 };
const read = { sources: [row], output: row, readoutLength: 32 };
const transfer = { ...read, output: inverse };

test('the numerical read head follows the addressed slice in read and transfer phases', () => {
  assert.deepEqual(spanBounds(row), [2, 0, 0, 3, 32, 1]);
  assert.deepEqual(spanBounds(inverse), [18, 0, 0, 19, 1, 1]);
  assert.equal(spanBounds({ ...row, indices: [4, 0, 0] }), null);
  assert.equal(phaseReadCount(read, 0), 1);
  assert.equal(phaseReadCount(transfer, 1), 32);
  for (let i = 0; i <= 100; i++) {
    const t = i / 100;
    assert.equal(phaseReadCount(read, t), readCount(t, 32));
    if (i) assert.ok(phaseReadCount(transfer, t) >= phaseReadCount(transfer, (i - 1) / 100));
  }
  assert.ok(phaseReadCount(transfer, 0.4) > phaseReadCount(read, 0.4));
});

// Record path geometry on a fixed camera, deliberately excluding transport DOM
// and labels. A progress bar or camera tween alone cannot satisfy this check.
function geometryAt(phase, t) {
  const commands = [];
  const ctx = { globalAlpha: 1 };
  for (const op of [
    'save',
    'restore',
    'beginPath',
    'clip',
    'rect',
    'strokeRect',
    'closePath',
    'arc',
    'moveTo',
    'lineTo',
    'bezierCurveTo',
    'fill',
    'stroke',
  ]) {
    ctx[op] = (...args) => commands.push([op, ...args]);
  }
  const view = cameraMap(basis(0.35, 0.18), [8, 14, 0], 9, 600, 480);
  paintComputation(ctx, view, phase, t);
  return commands;
}
test('on a stationary camera both the read strip and the travelling cells move, and scrubbing is reversible', () => {
  for (const phase of [read, transfer]) {
    const initial = geometryAt(phase, 0.22),
      later = geometryAt(phase, 0.54);
    assert.ok(initial.some(([op]) => op === 'fill') && later.some(([op]) => op === 'fill'));
    assert.notDeepEqual(initial, later);
    assert.deepEqual(geometryAt(phase, 0.22), initial);
  }
});
