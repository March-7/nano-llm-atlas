import assert from 'node:assert/strict';
import { test } from 'node:test';
import { basis, cameraMap, planes, pickFace } from '../src/detail/tensor-paint.js';

// Project known model-space cells, then recover them from CSS-pixel coordinates.
test('picking a rotated tensor recovers the original cell, even near the end of a large matrix', () => {
  const size = [12288, 384, 3],
    origin = [809042, 494287, -189];
  const n = {
    x: origin[0],
    y: origin[1],
    z: origin[2],
    dx: size[0] * 1.5,
    dy: size[1] * 1.5,
    dz: size[2] * 1.5,
    cx: size[0],
    cy: size[1],
    cz: size[2],
  };
  for (const [yaw, pitch] of [
    [0.34, 0.22],
    [-1.2, 0.48],
    [2.6, -0.32],
  ]) {
    const view = cameraMap(basis(yaw, pitch), origin, 19, 1280, 720);
    for (const face of planes(n, view)) {
      for (const cell of [
        [0, 0, 0],
        [7391, 127, 1],
        [12287, 383, 2],
      ]) {
        const expected = [...cell];
        expected[face.normal] = face.high ? size[face.normal] - 1 : 0;
        const world = expected.map((i, k) => origin[k] + 1.5 * (i + 0.5));
        world[face.normal] = origin[face.normal] + 1.5 * (face.high ? size[face.normal] : 0);
        const [x, y] = view.project(world);
        const hit = pickFace(face, x, y, view);
        assert.deepEqual([hit.x, hit.y, hit.z], expected);
      }
    }
  }
});

test('an exposed internal time slice returns the original recurrent-state coordinate', () => {
  const whole = { id: 'delta', x: 10, y: 50, z: -8, dx: 12, dy: 12, dz: 12, cx: 8, cy: 8, cz: 8 };
  const slice = { ...whole, whole, z: whole.z + 4 * 1.5, dz: 1.5, cz: 1, sliceZ: 4 };
  const view = cameraMap(basis(0.34, 0.22), [0, 0, 0], 6, 600, 600);
  const face = planes(slice, view).find((f) => f.normal === 2);
  const [px, py] = view.project([slice.x + 1.5 * 2.5, slice.y + 1.5 * 3.5, slice.z + slice.dz]);
  const hit = pickFace(face, px, py, view);
  assert.equal(hit.node, whole);
  assert.deepEqual([hit.x, hit.y, hit.z], [2, 3, 4]);
});
