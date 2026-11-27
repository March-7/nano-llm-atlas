import assert from 'node:assert/strict';
import test from 'node:test';
import { classicLayout } from '../src/detail/classic-layout.js';
import { gpt2Values, weightAddress } from '../src/detail/gpt2-values.js';

// Four original little-endian bytes per sample, read independently with Python
// struct.unpack('<f', ...) and exact HTTP ranges from the two pinned public GPT-2
// checkpoints (OpenAI, MIT). Raw offsets are absolute file offsets, including the header.
// Each tensor entry stores its real shape and its start relative to the data section.
const fixtures = {
  'gpt2-small': {
    repo: 'gpt2',
    revision: '607a30d783dfa663caf39e06633721c8d4cfcd7e',
    size: 548105171,
    header: 14283,
    tensors: {
      'h.0.attn.c_attn.weight': [[768, 2304], 270682112],
      'h.0.attn.c_attn.bias': [[2304], 85365760],
      'h.11.attn.c_attn.weight': [[768, 2304], 108468224],
      'h.11.mlp.c_fc.weight': [[768, 3072], 40227840],
      'h.11.mlp.c_proj.weight': [[3072, 768], 160424960],
      'h.11.attn.c_proj.weight': [[768, 768], 139166720],
      'h.11.ln_2.bias': [[768], 270679040],
      'wte.weight': [[50257, 768], 287209472],
      'h.2.attn.c_attn.weight': [[768, 2304], 541012992], // actual final record
    },
    // layer, head, visual operation, x, y, canonical tensor coordinates, file offset, bytes
    samples: [
      [0, 2, 'qWeightBlock', 7, 5, 'h.0.attn.c_attn.weight', [7, 133], 270761447, '6207bc3d'],
      [0, 0, 'kBiasBlock', 0, 13, 'h.0.attn.c_attn.bias', [781], 85383175, '5515dabc'],
      [11, 7, 'kWeightBlock', 13, 17, 'h.11.attn.c_attn.weight', [13, 1233], 108607255, '264ed83d'],
      [11, -1, 'mlpFcWeight', 567, 31, 'h.11.mlp.c_fc.weight', [31, 567], 40625327, 'b9fcbbbd'],
      [11, -1, 'mlpProjWeight', 901, 41, 'h.11.mlp.c_proj.weight', [901, 41], 163207287, 'e992cabe'],
      [11, -1, 'projWeight', 7, 23, 'h.11.attn.c_proj.weight', [7, 23], 139202607, '94ed333e'],
      [11, -1, 'ln2.lnMu', 0, 17, 'h.11.ln_2.bias', [17], 270693399, '93f8bc3c'],
      [-1, -1, 'tokEmbedObj', 42, 10, 'wte.weight', [42, 10], 287352827, '2ec6d13d'],
      [-1, -1, 'lmHeadWeight', 767, 50256, 'wte.weight', [50256, 767], 441613263, '2826f73d'],
    ],
  },
  'gpt2-xl': {
    repo: 'gpt2-xl',
    revision: '15ea56dee5df4983c59b2538573817e1667135e2',
    size: 6431829964,
    header: 58564,
    tensors: {
      'h.0.attn.c_attn.weight': [[1600, 4800], 4213504],
      'h.0.attn.c_attn.bias': [[4800], 4194304],
      'h.47.attn.c_attn.weight': [[1600, 4800], 5344828672],
      'h.47.mlp.c_fc.weight': [[1600, 6400], 5385846272],
      'h.47.mlp.c_proj.weight': [[6400, 1600], 5426812672],
      'h.47.attn.c_proj.weight': [[1600, 1600], 5375555072],
      'h.47.ln_2.bias': [[1600], 5385807872],
      'wte.weight': [[50257, 1600], 6110126592],
    },
    samples: [
      [0, 1, 'qkvWeightBlock', 7, 69, 'h.0.attn.c_attn.weight', [7, 1669], 4413152, '493f3b3b'],
      [0, 0, 'kBiasBlock', 0, 13, 'h.0.attn.c_attn.bias', [1613], 4259328, '7a75353d'],
      [47, 24, 'qkvWeightBlock', 9, 159, 'h.47.attn.c_attn.weight', [9, 4767], 5345079112, '8d067d3d'],
      [47, -1, 'mlpFcWeight', 4133, 23, 'h.47.mlp.c_fc.weight', [23, 4133], 5386510176, '8ee3863c'],
      [47, -1, 'mlpProjWeight', 5243, 317, 'h.47.mlp.c_proj.weight', [5243, 317], 5460427712, '22d4a23b'],
      [47, -1, 'projWeight', 7, 23, 'h.47.attn.c_proj.weight', [7, 23], 5375658536, '74d5e5bc'],
      [47, -1, 'ln2.lnMu', 0, 17, 'h.47.ln_2.bias', [17], 5385866512, '55f5963d'],
      [-1, -1, 'tokEmbedObj', 42, 10, 'wte.weight', [42, 10], 6110454004, 'c83d4d3d'],
      [-1, -1, 'lmHeadWeight', 1599, 50256, 'wte.weight', [50256, 1599], 6431829960, '90a9f73d'],
    ],
  },
};

// Emulate only the real header's relevant records and the sampled bytes. Even the
// final page of XL lives in a small buffer; other offsets carry no fixture value.
function serve(fixture) {
  const records = { __metadata__: { format: 'pt' } };
  for (const [key, [shape, start]] of Object.entries(fixture.tensors)) {
    const bytes = shape.reduce((n, axis) => n * axis, 4);
    records[key] = { dtype: 'F32', shape, data_offsets: [start, start + bytes] };
  }
  const header = Buffer.from(JSON.stringify(records).padEnd(fixture.header));
  return (url, { headers }) => {
    const match = /^bytes=(\d+)-(\d+)$/.exec(headers.Range);
    assert.ok(match);
    const [start, end] = match.slice(1).map(Number);
    const endpoint = new URL(url);
    assert.equal(endpoint.origin, 'https://huggingface.co');
    assert.equal(
      endpoint.pathname,
      '/openai-community/' + fixture.repo + '/resolve/' + fixture.revision + '/model.safetensors',
    );
    assert.equal(endpoint.searchParams.get('tensor_atlas_range'), start + '-' + end);
    let bytes;
    if (start === 0) {
      assert.equal(end, 7);
      bytes = Buffer.alloc(8);
      bytes.writeBigUInt64LE(BigInt(header.length));
    } else if (start === 8) {
      assert.equal(end, header.length + 7);
      bytes = header;
    } else {
      assert.ok(end < fixture.size && end - start < 4096);
      bytes = Buffer.alloc(end - start + 1);
      for (const sample of fixture.samples) {
        const [address, hex] = sample.slice(-2);
        if (address >= start && address + 3 <= end) Buffer.from(hex, 'hex').copy(bytes, address - start);
      }
    }
    return new Response(bytes, {
      status: 206,
      headers: { 'Content-Range': 'bytes ' + start + '-' + end + '/' + fixture.size },
    });
  };
}

for (const [id, fixture] of Object.entries(fixtures)) {
  test(id + ' decodes exact first-layer, last-layer and tied weight samples', async (t) => {
    const remote = t.mock.method(globalThis, 'fetch', serve(fixture));
    const model = classicLayout(id),
      reader = gpt2Values(id);
    for (const [layer, head, op, x, y, key, coords, , hex] of fixture.samples) {
      const nodes = layer === -1 ? model.root : model.layers[layer].nodes();
      const node = nodes.find((n) => n.op === op && n.head === head);
      assert.deepEqual(weightAddress(id, node, x, y), [key, coords]);
      const expected = Buffer.from(hex, 'hex').readFloatLE();
      assert.equal(await reader.loadCell(node, x, y), expected, node.id);
      assert.equal(reader.value(node, x, y), expected);
    }
    const before = remote.mock.callCount();
    const tok = model.root.find((n) => n.op === 'tokEmbedObj');
    const lm = model.root.find((n) => n.op === 'lmHeadWeight');
    assert.equal(await reader.loadCell(lm, 10, 42), reader.value(tok, 42, 10));
    assert.equal(remote.mock.callCount(), before, 'tied input/output weights share a cache page');
  });
}

test('a missing Content-Range or a mismatched byte range is rejected and can be retried', async (t) => {
  const fixture = fixtures['gpt2-small'];
  const good = serve(fixture);
  let calls = 0;
  t.mock.method(globalThis, 'fetch', (url, options) => {
    calls++;
    if (calls === 1) return new Response(new Uint8Array(8), { status: 206 });
    if (calls === 2)
      return new Response(new Uint8Array(8), {
        status: 206,
        headers: { 'Content-Range': 'bytes 8-15/' + fixture.size },
      });
    return good(url, options);
  });
  const reader = gpt2Values('gpt2-small');
  const token = classicLayout('gpt2-small').root.find((n) => n.op === 'tokEmbedObj');
  await assert.rejects(reader.loadCell(token, 42, 10), /byte range/);
  await assert.rejects(reader.loadCell(token, 42, 10), /byte range/);
  assert.equal(await reader.loadCell(token, 42, 10), Buffer.from('2ec6d13d', 'hex').readFloatLE());
});

test('out-of-bounds coordinates and aborted selections cause no requests', async (t) => {
  const remote = t.mock.method(globalThis, 'fetch', () => {
    throw new Error('Unexpected network load');
  });
  const node = classicLayout('gpt2-small')
    .layers[0].nodes()
    .find((n) => n.op === 'qWeightBlock');
  const reader = gpt2Values('gpt2-small');
  assert.equal(weightAddress('gpt2-small', node, 0, node.cy), null);
  assert.equal(await reader.loadCell(node, node.cx, 0), null);
  assert.equal(await reader.loadCell(node, 0.5, 0), null);
  await assert.rejects(reader.loadCell(node, 0, 0, AbortSignal.abort()), { name: 'AbortError' });
  assert.equal(remote.mock.callCount(), 0);
});
