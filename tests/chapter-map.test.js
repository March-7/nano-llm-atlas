import assert from 'node:assert/strict';
import test from 'node:test';
import { chapters, models } from '../src/detail/catalog.js';
import { chapterMap } from '../src/detail/chapter-map.js';

test('every model chapter has one localized keyboard target in the map', () => {
  for (const { id } of models) {
    const items = chapters(id);
    for (const lang of ['zh', 'en']) {
      const active = items.length - 1;
      const svg = chapterMap(id, active, lang);
      const targets = [
        ...svg.matchAll(
          /<g class="cf-node" role="button" tabindex="0" data-chapter="(\d+)" aria-pressed="(true|false)" aria-label="([^"]+)">/g,
        ),
      ];
      assert.deepEqual(
        targets.map((t) => Number(t[1])),
        items.map((_, i) => i),
      );
      assert.deepEqual(
        targets.map((t) => t[3]),
        items.map((item) => item.label[lang]),
      );
      assert.deepEqual(
        targets.filter((t) => t[2] === 'true').map((t) => Number(t[1])),
        [active],
      );
      assert.ok(svg.includes('aria-labelledby=') && svg.includes('aria-describedby='));
    }
  }
});

test('the overview has no selected button and embedded maps have independent marker IDs', () => {
  const first = chapterMap('qwen', null, 'en');
  const second = chapterMap('qwen', -1, 'en');
  const target = /<g [^>]*data-chapter="\d+"[^>]*aria-pressed="true"/;
  assert.doesNotMatch(first, target);
  assert.doesNotMatch(second, target);
  const ids = [...(first + second).matchAll(/\bid="([^"]+)"/g)].map((match) => match[1]);
  assert.equal(new Set(ids).size, ids.length);
  for (const svg of [first, second]) {
    const marker = /<marker id="([^"]+)"/.exec(svg)[1];
    assert.ok(svg.includes('url(#' + marker + ')'));
  }
});

test('chapter diagrams keep the layer mechanisms and repetition counts distinct', () => {
  assert.match(chapterMap('gpt2-xl', -1, 'en'), /Decoder ×48/);
  const llama = chapterMap('llama', -1, 'en');
  assert.match(llama, /Decoder ×32/);
  assert.match(llama, /RoPE · Q, K/);
  assert.match(llama, /GQA · 4Q \/ KV/);
  const qwen = chapterMap('qwen', -1, 'en');
  assert.match(qwen, /4 layers ×16/);
  assert.match(qwen, /RMS → DeltaNet/);
  assert.match(qwen, /RMS → Gated GQA/);
  assert.equal([...qwen.matchAll(/RMS → SwiGLU/g)].length, 2);
});
