// Pinned public Hugging Face conversions of OpenAI's Conv1D-form GPT-2 weights.
// Safetensors offsets and HTTP byte ranges are 64-bit numbers, not bitwise offsets.
// OpenAI stores Conv1D as [input, output], splits c_attn along the output axis
// into Q, K, V (then heads within each), and reuses wte for the vocabulary head.
// https://github.com/openai/gpt-2/blob/master/src/model.py
const releases = {
  'gpt2-small': ['gpt2', '607a30d783dfa663caf39e06633721c8d4cfcd7e', 768, 12],
  'gpt2-xl': ['gpt2-xl', '15ea56dee5df4983c59b2538573817e1667135e2', 1600, 25],
};

export function weightAddress(id, node, x, y) {
  if (node?.kind !== 'w' || !Object.hasOwn(releases, id)) return null;
  if (
    !Number.isSafeInteger(x) ||
    !Number.isSafeInteger(y) ||
    x < 0 ||
    y < 0 ||
    x >= (node.cx ?? Infinity) ||
    y >= (node.cy ?? Infinity)
  )
    return null;
  const [, , width, heads] = releases[id];
  const headWidth = width / heads;
  const prefix = `h.${node.layer}.`;
  switch (node.op) {
    case 'tokEmbedObj':
      return ['wte.weight', [x, y]];
    case 'posEmbedObj':
      return ['wpe.weight', [x, y]];
    case 'lmHeadWeight':
      return ['wte.weight', [y, x]]; // tied to input embedding
    case 'projWeight':
      return [prefix + 'attn.c_proj.weight', [x, y]];
    case 'projBias':
      return [prefix + 'attn.c_proj.bias', [y]];
    case 'mlpFcWeight':
      return [prefix + 'mlp.c_fc.weight', [y, x]];
    case 'mlpFcBias':
      return [prefix + 'mlp.c_fc.bias', [x]];
    case 'mlpProjWeight':
      return [prefix + 'mlp.c_proj.weight', [x, y]];
    case 'mlpProjBias':
      return [prefix + 'mlp.c_proj.bias', [y]];
  }
  const norm = /^(ln1|ln2|ln_f)\.(lnSigma|lnMu)$/.exec(node.op);
  if (norm) {
    const path = norm[1] === 'ln_f' ? 'ln_f' : prefix + (norm[1] === 'ln1' ? 'ln_1' : 'ln_2');
    return [path + (norm[2] === 'lnSigma' ? '.weight' : '.bias'), [y]];
  }
  const projection = /^(q|k|v|qkv)(Weight|Bias)Block$/.exec(node.op);
  if (projection) {
    if (!Number.isInteger(node.head) || node.head < 0 || node.head >= heads) return null;
    if (y >= (projection[1] === 'qkv' ? 3 : 1) * headWidth) return null;
    const component = projection[1] === 'qkv' ? Math.floor(y / headWidth) : 'qkv'.indexOf(projection[1]);
    const out = component * width + node.head * headWidth + (y % headWidth);
    return projection[2] === 'Weight'
      ? [prefix + 'attn.c_attn.weight', [x, out]]
      : [prefix + 'attn.c_attn.bias', [out]];
  }
  return null;
}

async function readRange(base, start, end, signal) {
  // The query also distinguishes ranges in caches that ignore the Range header.
  const response = await fetch(base + '?tensor_atlas_range=' + start + '-' + end, {
    headers: { Range: `bytes=${start}-${end}` },
    signal,
  });
  // Both pinned endpoints expose Content-Range through CORS. A 206 of the
  // correct length could still be the wrong cached page, so verify its address.
  const range = /^bytes (\d+)-(\d+)\/(\d+)$/i.exec(response.headers.get('Content-Range') ?? '');
  if (
    response.status !== 206 ||
    !range ||
    Number(range[1]) !== start ||
    Number(range[2]) !== end ||
    Number(range[3]) <= end
  ) {
    await response.body?.cancel();
    throw new Error('The checkpoint server did not return the requested byte range.');
  }
  const data = await response.arrayBuffer();
  if (data.byteLength !== end - start + 1) throw new Error('Truncated checkpoint range.');
  return data;
}

export function gpt2Values(id) {
  if (!Object.hasOwn(releases, id)) throw new RangeError('Unknown GPT-2 release: ' + id);
  const [repo, revision] = releases[id];
  const base = `https://huggingface.co/openai-community/${repo}/resolve/${revision}/model.safetensors`;
  const pages = new Map();
  const pageSize = 4096;
  let index, loading;
  async function openIndex() {
    if (!loading)
      loading = (async () => {
        const bytes = await readRange(base, 0, 7, AbortSignal.timeout(16000));
        const length = Number(new DataView(bytes).getBigUint64(0, true));
        if (!Number.isSafeInteger(length) || length < 16 || length > 2 ** 20)
          throw new Error('Invalid safetensors header.');
        const header = JSON.parse(
          new TextDecoder().decode(await readRange(base, 8, 7 + length, AbortSignal.timeout(16000))),
        );
        const records = Object.values(header).filter((record) => record.data_offsets);
        index = {
          header,
          origin: 8 + length,
          end: 8 + length + Math.max(...records.map((r) => r.data_offsets[1])),
        };
      })().catch((error) => {
        loading = undefined;
        throw error;
      });
    return loading;
  }
  function addressOf(node, x, y) {
    const pair = weightAddress(id, node, x, y);
    if (!index || !pair) return null;
    const [key, coords] = pair;
    const tensor = index.header[key] || index.header['transformer.' + key];
    if (!tensor || tensor.dtype !== 'F32' || tensor.shape.length !== coords.length) return null;
    let scalar = 0;
    for (let axis = 0; axis < coords.length; axis++) {
      if (!Number.isInteger(coords[axis]) || coords[axis] < 0 || coords[axis] >= tensor.shape[axis])
        return null;
      scalar = scalar * tensor.shape[axis] + coords[axis];
    }
    return index.origin + tensor.data_offsets[0] + scalar * 4;
  }
  // Align pages to the DATA section: safetensors headers need not be 4/4096 aligned.
  function pageOf(offset) {
    return index.origin + Math.floor((offset - index.origin) / pageSize) * pageSize;
  }
  function value(node, x, y) {
    const offset = addressOf(node, x, y);
    if (offset === null) return null;
    const page = pageOf(offset);
    return pages.get(page)?.getFloat32(offset - page, true) ?? null;
  }
  return {
    value,
    explain(node, x, y) {
      const pair = weightAddress(id, node, x, y);
      return {
        formula: node.formula || '',
        substitution: pair ? `${pair[0]}[${pair[1].join(', ')}]` : '',
        source: pair ? repo + ' · ' + revision.slice(0, 7) : '',
      };
    },
    async loadCell(node, x, y, signal) {
      if (!weightAddress(id, node, x, y)) return null;
      signal?.throwIfAborted();
      await openIndex();
      signal?.throwIfAborted();
      const offset = addressOf(node, x, y);
      if (offset === null) throw new Error('Missing tensor in the pinned checkpoint.');
      const page = pageOf(offset);
      if (!pages.has(page)) {
        const bytes = await readRange(base, page, Math.min(index.end - 1, page + pageSize - 1), signal);
        if (pages.size >= 80) pages.delete(pages.keys().next().value);
        pages.set(page, new DataView(bytes));
      }
      return value(node, x, y);
    },
  };
}
