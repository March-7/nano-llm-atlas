/*
 * World coordinates: x follows the residual stream, y is height, z is depth.
 * The grids sample each tensor. Cell colors describe its role, not model weights.
 */
export const inks = {
  embed: [113, 225, 188],
  norm: [120, 198, 243],
  delta: [117, 193, 245],
  attention: [182, 153, 247],
  ffn: [246, 164, 123],
  output: [132, 179, 249],
  bus: [155, 166, 193],
};

export function buildArchitecture(model, spread) {
  const parts = [];
  const traces = [];
  const stations = [];
  const span = 0.84 + 0.16 * spread;
  const lift = 0.7 + 0.3 * spread;
  const x = (v) => v * span;
  const y = (v) => v * lift;
  const z = (v) => v * (0.64 + 0.36 * spread);

  function box(id, stage, cx, bottom, cz, width, height, depth, grid, options = {}) {
    const center = [x(cx), y(bottom) + height / 2, z(cz)];
    const size = [width, height, depth];
    const min = center.map((v, i) => v - size[i] / 2);
    const max = center.map((v, i) => v + size[i] / 2);
    parts.push({
      id,
      stage,
      center,
      size,
      min,
      max,
      grid,
      finish: 'tensor',
      seed: parts.length + 1,
      ...options,
    });
  }

  function wire(stage, points, style = {}) {
    traces.push({ stage, points: points.map(([a, b, c]) => [x(a), y(b), z(c)]), ...style });
  }

  function station(stage, title, short, anchor, side, options = {}) {
    stations.push({
      stage,
      title,
      short,
      anchor: [x(anchor[0]), y(anchor[1]), z(anchor[2])],
      side,
      ...options,
    });
  }

  function pad(id, stage, left, right, front = 1.15, back = -1.15) {
    box(
      id,
      stage,
      (left + right) / 2,
      -0.18,
      (front + back) / 2,
      (right - left) * span,
      0.13,
      z(front - back),
      [1, 1, 1],
      { finish: 'board' },
    );
  }

  function residual(start, end, atZ, annotate = false) {
    wire(
      'bus',
      [
        [start, 0.19, 0],
        [start, 0.19, atZ * 0.82],
        [start + 0.18, 0.19, atZ],
        [end - 0.18, 0.19, atZ],
        [end, 0.19, atZ * 0.82],
        [end, 0.19, 0],
      ],
      { skip: true, annotate },
    );
    box('add-' + end, 'bus', end, 0.07, 0, 0.18, 0.18, 0.18, [1, 1, 1], { finish: 'joint' });
  }

  function embedding(cx, positions) {
    pad('embedding-board', 'embed', cx - 1.0, cx + 0.9, 1.04, -0.95);
    box('token-embedding', 'embed', cx, 0.08, -0.34, 1.47, 1.74, 0.53, [12, positions ? 3 : 12, 3], {
      print: positions ? 'E' : 'Wₑ',
    });
    if (positions) {
      box('position-embedding', 'embed', cx + 0.05, 0.06, 0.66, 1.33, 0.68, 0.27, [12, 5, 2], {
        print: 'P',
      });
      wire('embed', [
        [cx, 0.24, -0.34],
        [cx + 0.65, 0.24, -0.34],
        [cx + 0.84, 0.24, 0],
        [cx + 1.1, 0.24, 0],
      ]);
      wire('embed', [
        [cx, 0.24, 0.66],
        [cx + 0.66, 0.24, 0.66],
        [cx + 0.84, 0.24, 0],
      ]);
    } else {
      box('lookup-row', 'embed', cx + 0.02, 0.06, 0.57, 1.46, 0.2, 0.34, [12, 1, 2]);
    }
    station('embed', positions ? 'Token + position' : 'Embedding', 'Embed', [cx, 1.82, -0.36], -1);
  }

  function normalization(cx, legacy) {
    pad('norm-board', 'norm', cx - 0.79, cx + 0.79, 0.74, -0.76);
    const layers = [0.05, 0.43, 0.81];
    layers.forEach((height, i) => {
      box('norm-row-' + i, 'norm', cx, height, 0.05 - i * 0.06, 1.16, 0.15, 0.72, [9, 1, 4], {
        finish: i === 2 ? 'tensor' : 'ribbon',
      });
    });
    station('norm', legacy ? 'LayerNorm' : 'RMSNorm', legacy ? 'LN' : 'RMS', [cx, 0.06, 0.64], 1);
  }

  // Only the Q lane is replicated in a grouped-query bank. Its narrow K and V
  // banks remain one shared pair for this displayed group.
  function attention(start, width, queries, gate) {
    const grouped = queries > 1;
    const qx = start + 0.63;
    const sx = start + width - 1.32;
    const end = start + width;
    pad('attention-board', 'attention', start - 0.12, end + 0.04, 1.43, -1.38);

    if (!grouped) {
      const lanes = [
        { name: 'Q', depth: -0.91 },
        { name: 'K', depth: 0 },
        { name: 'V', depth: 0.91 },
      ];
      lanes.forEach(({ name, depth }) => {
        box('project-' + name, 'attention', qx, 0.24, depth, 0.88, 1.46, 0.28, [6, 8, 2], {
          print: name,
          printY: 0.15,
        });
        wire('attention', [
          [start - 0.27, 0.25, 0],
          [start - 0.04, 0.25, depth],
          [qx, 0.25, depth],
        ]);
      });
    } else {
      const rear = -0.82;
      const gap = queries === 6 ? 0.21 : 0.29;
      for (let i = 0; i < queries; i++) {
        const depth = rear + i * gap;
        box(
          'query-head-' + i,
          'attention',
          qx - 0.06,
          0.41 + 0.1 * (queries - 1 - i),
          depth,
          0.83,
          1.02,
          0.115,
          [6, 6, 1],
          { print: i === queries - 1 ? 'Q' : '' },
        );
      }
      box('shared-key', 'attention', qx - 0.33, 0.06, 1.14, 0.46, 0.52, 0.39, [3, 4, 2], { print: 'K' });
      box('shared-value', 'attention', qx + 0.28, 0.06, 1.14, 0.46, 0.52, 0.39, [3, 4, 2], { print: 'V' });
      wire('attention', [
        [qx - 0.33, 0.34, 1.14],
        [qx - 0.33, 0.34, 0.59],
        [qx - 0.62, 0.34, 0.39],
        [qx - 0.62, 0.34, rear],
        [qx, 0.64, rear],
      ]);
    }

    // A lower-triangular score table is visible on the front of each slab.
    const count = grouped ? 1 : 3;
    for (let i = 0; i < count; i++) {
      const front = grouped ? 0.04 : -0.67 + i * 0.4;
      const height = 0.39 + (count - 1 - i) * 0.26;
      box('masked-scores-' + i, 'attention', sx, height, front, 1.74, 1.78, 0.18, [10, 10, 1], {
        finish: 'causal',
      });
    }
    wire('attention', [
      [qx + 0.36, 0.46, -0.82],
      [sx - 1.05, 0.46, -0.82],
      [sx - 0.84, 0.46, -0.46],
      [sx, 0.46, -0.46],
    ]);
    wire('attention', [
      [qx + 0.33, 0.33, 1.04],
      [end - 0.41, 0.33, 1.04],
      [end - 0.19, 0.33, 0.7],
      [end - 0.19, 0.33, 0],
    ]);
    box(
      'attention-output',
      'attention',
      end - 0.2,
      0.1,
      -0.02,
      0.33,
      gate ? 0.95 : 0.68,
      0.65,
      [2, gate ? 6 : 4, 4],
      { finish: 'ribbon', print: gate ? 'g' : '' },
    );
    station(
      'attention',
      grouped ? (gate ? 'Gated GQA · 6:1' : 'GQA · 4:1') : 'Causal attention',
      grouped ? (gate ? 'GQA 6:1' : 'GQA 4:1') : 'MHA',
      [sx, grouped ? 2.17 : 2.68, -0.3],
      -1,
    );
  }

  function feedForward(left, width, gated) {
    const right = left + width;
    const expand = left + 0.58;
    const reduce = right - 0.38;
    pad('ffn-board', 'ffn', left - 0.18, right, 1.09, -1.14);
    // The second pre-normalization sits immediately before the dense FFN.
    box(
      'ffn-pre-norm',
      model === 'qwen' ? 'ffn' : 'norm',
      left - 0.1,
      0.06,
      0.04,
      0.18,
      0.2,
      0.58,
      [1, 1, 4],
      { finish: 'ribbon', swatch: 'norm' },
    );
    if (gated) {
      box('ffn-gate', 'ffn', expand, 0.14, -0.55, 0.88, 2.42, 0.32, [6, 15, 2], {
        print: 'G',
        printY: 0.17,
      });
      box('ffn-up', 'ffn', expand, 0.1, 0.4, 0.88, 1.93, 0.34, [6, 12, 2], { print: 'U' });
      wire('ffn', [
        [expand, 0.4, -0.55],
        [reduce - 0.63, 0.4, -0.55],
        [reduce - 0.4, 0.4, 0],
      ]);
      wire('ffn', [
        [expand, 0.4, 0.4],
        [reduce - 0.6, 0.4, 0.4],
        [reduce - 0.4, 0.4, 0],
      ]);
    } else {
      box('ffn-expand', 'ffn', expand, 0.1, -0.11, 0.99, 2.48, 0.72, [7, 15, 5], { print: 'W₁' });
    }
    box('ffn-down', 'ffn', reduce, 0.1, 0.08, 0.51, 1.19, 0.58, [4, 8, 4], { print: gated ? 'D' : 'W₂' });
    station(
      'ffn',
      gated ? 'SwiGLU' : 'GELU MLP',
      gated ? 'SwiGLU' : 'MLP',
      [expand, gated ? 2.56 : 2.58, -0.27],
      -1,
    );
  }

  function head(cx, legacy) {
    pad('readout-board', 'output', cx - 0.8, cx + 0.8, 0.87, -0.78);
    box('final-norm', 'output', cx - 0.63, 0.07, 0.1, 0.18, 0.27, 0.61, [1, 2, 4], { finish: 'ribbon' });
    box('lm-projection', 'output', cx + 0.02, 0.09, -0.17, 0.85, 1.63, 0.57, [legacy ? 3 : 6, 10, 3], {
      print: 'Wₒ',
    });
    box('logits-strip', 'output', cx + 0.24, 0.07, 0.58, 0.74, 0.21, 0.24, [legacy ? 3 : 6, 1, 1], {
      finish: 'tensor',
    });
    station('output', 'LM head', 'Head', [cx + 0.23, 0.08, 0.78], 1);
  }

  function deltaCell(cx, index) {
    pad('delta-board-' + index, 'delta', cx - 0.7, cx + 0.68, 1.04, -1.0);
    box('delta-pre-norm-' + index, 'delta', cx - 0.54, 0.09, 0.04, 0.18, 0.22, 0.56, [1, 2, 4], {
      finish: 'ribbon',
    });
    box('delta-state-' + index, 'delta', cx, 0.16, -0.33, 0.94, 1.57, 0.47, [7, 10, 3], {
      finish: 'state',
      print: 'Δ' + index,
    });
    // A recurrent state has a separate feedback path from the residual stream.
    const back = -0.79;
    wire(
      'delta',
      [
        [cx + 0.24, 1.4, -0.32],
        [cx + 0.42, 1.4, back],
        [cx + 0.42, 1.93, back],
        [cx + 0.29, 2.06, back],
        [cx - 0.28, 2.06, back],
        [cx - 0.42, 1.93, back],
        [cx - 0.42, 1.28, back],
        [cx - 0.23, 1.28, -0.32],
      ],
      { feedback: true },
    );
    box('delta-ffn-' + index, 'ffn', cx + 0.14, 0.08, 0.54, 0.84, 0.43, 0.39, [6, 3, 3], {
      print: 'F',
      finish: 'ribbon',
    });
    wire('ffn', [
      [cx, 0.3, -0.3],
      [cx + 0.36, 0.3, 0.04],
      [cx + 0.14, 0.3, 0.54],
    ]);
  }

  if (model === 'qwen') {
    embedding(-8.02, false);
    [-5.91, -4.27, -2.63].forEach((cx, i) => deltaCell(cx, i + 1));
    station('delta', 'Gated DeltaNet ×3', 'DeltaNet ×3', [-4.27, 2.07, -0.78], -1);
    attention(-1.29, 3.87, 6, true);
    feedForward(3.2, 2.28, true);
    head(7.03, false);
    residual(-1.52, 2.75, 1.8, true);
    residual(2.86, 5.72, 1.5);
    wire('bus', [
      [-8.02, 0.17, 0],
      [7.03, 0.17, 0],
    ]);
  } else {
    const legacy = model === 'nanogpt';
    embedding(-6.81, legacy);
    normalization(-4.59, legacy);
    attention(-3.04, 4.44, legacy ? 1 : 4, false);
    feedForward(2.22, 2.5, !legacy);
    head(6.22, legacy);
    residual(-5.6, 1.6, 1.88, true);
    residual(1.79, 4.94, 1.5);
    wire('bus', [
      [-6.81, 0.17, 0],
      [6.22, 0.17, 0],
    ]);
  }
  return { parts, traces, stations };
}
