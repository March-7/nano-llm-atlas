import assert from 'node:assert/strict';
import { test } from 'node:test';
import { chapterPlayback, phaseMillis } from '../src/detail/chapter-playback.js';

function clock(onComplete = () => {}) {
  let now = 0,
    id = 0;
  const pending = new Map(),
    renders = [];
  const player = chapterPlayback({
    onFrame: (state) => renders.push(state),
    onComplete,
    now: () => now,
    schedule(fn) {
      pending.set(++id, fn);
      return id;
    },
    cancel: (n) => pending.delete(n),
  });
  function tick(time) {
    now += time * phaseMillis;
    const current = [...pending.values()];
    pending.clear();
    for (const fn of current) fn();
  }
  return { player, pending, renders, tick };
}
test('pause, seek and resume repaint precisely the chosen computation instant', () => {
  let completed = 0;
  const { player, pending, tick } = clock(() => completed++);
  player.start(['input', 'inverse', 'output']);
  tick(1.25);
  assert.equal(player.state.phase, 'inverse');
  assert.equal(player.state.progress, 0.25);
  player.pause();
  tick(52);
  assert.equal(player.state.position, 1.25);
  assert.equal(pending.size, 0);
  player.seek(2.5);
  assert.equal(player.state.phase, 'output');
  player.play();
  tick(0.5);
  assert.equal(player.state.status, 'ended');
  assert.equal(player.state.progress, 1);
  assert.equal(completed, 1);
  player.play();
  assert.equal(player.state.index, 0);
  tick(0.5);
  player.refresh(['input with values', 'inverse', 'output']);
  assert.equal(player.state.phase, 'input with values');
  assert.equal(player.state.progress, 0.5);
});
test('replacing the chapter or closing the graph invalidates queued animation frames', () => {
  let completed = 0;
  const { player, pending, renders, tick } = clock(() => completed++);
  player.start(['old']);
  const stale = [...pending.values()][0];
  player.start(['new', 'result'], false);
  stale();
  tick(8);
  assert.equal(player.state.phase, 'new');
  assert.equal(player.state.status, 'paused');
  assert.equal(pending.size, 0);
  player.play();
  const obsolete = [...pending.values()][0];
  player.start([], false);
  obsolete();
  assert.equal(completed, 0);
  assert.equal(renders.at(-1), null);
});
test('continuous walkthrough starts its next chapter only after the last computation frame', () => {
  const { player, pending, renders, tick } = clock(() => player.start(['next input', 'next output']));
  player.start(['read', 'write']);
  tick(1.9);
  assert.equal(player.state.phase, 'write');
  tick(0.11);
  assert.deepEqual(
    renders.slice(-2).map((v) => [v.status, v.phase]),
    [
      ['ended', 'write'],
      ['playing', 'next input'],
    ],
  );
  assert.equal(pending.size, 1);
  player.seek(2);
  assert.equal(pending.size, 0);
  assert.equal(player.state.status, 'ended');
});
