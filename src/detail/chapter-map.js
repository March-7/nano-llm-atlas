import { chapters, models } from './catalog.js';

const center = 76;
let sequence = 0;

function xml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;');
}

const styles = `
  .chapter-flow text { font-family: var(--sans,system-ui), "PingFang SC", sans-serif; }
  .chapter-flow .cf-caption { font-size:9.5px; font-weight:650; fill:#485863; }
  .chapter-flow .cf-node { cursor:pointer; outline:none; }
  .chapter-flow .cf-shape { stroke:#85919a; stroke-width:.75; }
  .chapter-flow .cf-embed { fill:#ecdef3; }
  .chapter-flow .cf-norm { fill:#f3ebbf; }
  .chapter-flow .cf-attn { fill:#dbe7f4; }
  .chapter-flow .cf-ffn { fill:#d4eade; }
  .chapter-flow .cf-output { fill:#d9e8cf; }
  .chapter-flow .cf-sum { fill:#fbfaf5; }
  .chapter-flow .cf-label { font-size:11px; font-weight:550; fill:#27333a; text-anchor:middle; dominant-baseline:central; }
  .chapter-flow .cf-small { font-size:9.7px; }
  .chapter-flow .cf-plus { fill:none; stroke:#42535b; stroke-width:1.15; stroke-linecap:round; }
  .chapter-flow .cf-node:hover .cf-shape, .chapter-flow .cf-node:focus-visible .cf-shape {
    stroke:#24699b; stroke-width:1.8;
  }
  .chapter-flow .cf-node[aria-pressed="true"] .cf-shape {
    stroke:#235a84; stroke-width:2.1;
  }
  .chapter-flow .cf-node[aria-pressed="true"] .cf-label { fill:#153f60; font-weight:750; }
  .chapter-flow .cf-node:focus-visible { outline:1px dashed #265789; outline-offset:1px; }
`;

function makeDrawing(items, lang, marker, active) {
  const backgrounds = [],
    edges = [],
    decorations = [];
  const pieces = items.map(() => []);
  const tr = (zh, en) => (lang === 'zh' ? zh : en);

  function frame(x, y, w, h, label, nested = false) {
    backgrounds.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${nested ? 3 : 5}"
      fill="${nested ? '#fbfaf4' : '#f5f3e9'}" stroke="${nested ? '#d6d4c6' : '#c5c7bc'}" stroke-width=".8"
      ${nested ? '' : 'stroke-dasharray="2.6 2.2"'}/>
      <text class="cf-caption" x="${x + 5}" y="${y + 11}">${xml(label)}</text>`);
  }
  function line(path, arrow = false, faint = false) {
    edges.push(`<path d="${path}" fill="none" stroke="${faint ? '#899895' : '#5b6b73'}" stroke-width="1.1"
      stroke-linejoin="round" stroke-linecap="round" ${arrow ? `marker-end="url(#${marker})"` : ''}/>`);
  }
  function down(from, to, x = center) {
    line(`M${x} ${from}V${to - 1.1}`, true);
  }
  function box(i, y, label, tone, { x = 27, w = 98, h = 17, small = false } = {}) {
    pieces[i].push(`<rect class="cf-shape cf-${tone}" x="${x}" y="${y}" width="${w}" height="${h}" rx="3"/>
      <text class="cf-label${small ? ' cf-small' : ''}" x="${x + w / 2}" y="${y + h / 2}">${xml(label)}</text>`);
  }
  function sum(y, i) {
    const drawing = `<circle class="cf-shape cf-sum" cx="${center}" cy="${y}" r="4.7"/>
      <path class="cf-plus" d="M${center - 2.25} ${y}h4.5m-2.25 -2.25v4.5"/>`;
    if (i === undefined) decorations.push(drawing);
    else pieces[i].push(drawing);
  }
  function skip(start, join, chapter, rail = 138) {
    edges.push(`<circle cx="${center}" cy="${start}" r="1.6" fill="#64747b"/>`);
    line(
      `M${center} ${start}H${rail - 2}Q${rail} ${start} ${rail} ${start + 2}V${join - 2}Q${rail} ${join} ${rail - 2} ${join}H${center + 6}`,
      true,
      true,
    );
    sum(join, chapter);
  }
  function readout(chapter, normLabel) {
    down(242, 255);
    box(chapter, 255, normLabel, 'norm', { w: 86, x: 33 });
    down(272, 282);
    box(chapter, 282, tr('词表 → softmax', 'Logits → softmax'), 'output', { x: 24, w: 104, h: 19 });
  }
  function render() {
    return (
      backgrounds.join('') +
      edges.join('') +
      decorations.join('') +
      pieces
        .map(
          (parts, i) =>
            `<g class="cf-node" role="button" tabindex="0" data-chapter="${i}" aria-pressed="${i === active}" aria-label="${xml(items[i].label[lang])}">
        <title>${xml(items[i].label[lang])}</title>${parts.join('')}</g>`,
        )
        .join('')
    );
  }
  return { tr, frame, line, down, box, sum, skip, readout, render };
}

function classic(g, layers) {
  const { tr, frame, box, down, line, sum, skip, readout } = g;
  box(0, 5, tr('词元嵌入', 'Token emb.'), 'embed', { x: 5, w: 66, h: 18, small: true });
  box(0, 5, tr('位置嵌入', 'Pos. emb.'), 'embed', { x: 81, w: 66, h: 18, small: true });
  line('M38 23V30Q38 32 40 32H67L71 35', true);
  line('M114 23V30Q114 32 112 32H85L81 35', true);
  sum(36, 0);
  frame(3, 49, 146, 197, tr('解码块', 'Decoder') + ' ×' + layers);
  down(41, 70);
  skip(64, 171, 6);
  box(1, 70, 'LayerNorm', 'norm');
  line('M76 87V92M46 96V94Q46 92 48 92H104Q106 92 106 94V96');
  for (const [j, label] of ['Q', 'K', 'V'].entries()) {
    const x = 33 + 30 * j;
    box(2, 97, label, 'attn', { x, w: 26, h: 15 });
    line(`M${x + 13} 112V116`);
  }
  line('M76 92V97M46 116H106');
  down(116, 121);
  box(3, 121, tr('因果多头注意力', 'Causal MHA'), 'attn');
  down(138, 145);
  box(4, 145, tr('输出投影', 'Output projection'), 'attn', { small: true });
  down(162, 166);
  down(176, 183);
  skip(179, 237, 6);
  box(1, 183, 'LayerNorm', 'norm');
  down(200, 207);
  box(5, 207, 'MLP · GELU', 'ffn');
  down(224, 232);
  readout(7, 'LayerNorm');
}

function llama(g) {
  const { tr, frame, box, down, skip, readout } = g;
  box(0, 6, tr('词元嵌入', 'Token embedding'), 'embed', { x: 20, w: 112, h: 20 });
  frame(3, 42, 146, 204, tr('解码块 ×32', 'Decoder ×32'));
  down(26, 66);
  skip(58, 168);
  box(1, 66, 'RMSNorm', 'norm');
  down(83, 97);
  box(2, 97, 'RoPE · Q, K', 'attn');
  down(114, 130);
  box(2, 130, 'GQA · 4Q / KV', 'attn');
  down(147, 163);
  down(173, 183);
  skip(178, 237);
  box(1, 183, 'RMSNorm', 'norm');
  down(200, 209);
  box(3, 209, 'SwiGLU', 'ffn');
  down(226, 232);
  readout(4, 'RMSNorm');
}

function qwen(g) {
  const { tr, frame, box, down, skip, readout } = g;
  box(0, 6, tr('文本嵌入', 'Text embedding'), 'embed', { x: 20, w: 112, h: 20 });
  frame(2, 39, 148, 210, tr('四层 ×16', '4 layers ×16'));
  frame(7, 54, 138, 91, tr('前 3 层', 'Layers 1–3'), true);
  frame(7, 149, 138, 95, tr('第 4 层', 'Layer 4'), true);
  down(26, 72);
  skip(68, 98);
  box(1, 72, 'RMS → DeltaNet', 'attn', { x: 25, w: 102 });
  down(89, 93);
  down(103, 111);
  skip(106, 138);
  box(3, 111, 'RMS → SwiGLU', 'ffn', { x: 25, w: 102 });
  down(128, 133);
  down(143, 168);
  skip(163, 197);
  box(2, 168, tr('RMS → 门控 GQA', 'RMS → Gated GQA'), 'attn', { x: 22, w: 108, small: true });
  down(185, 192);
  down(202, 210);
  skip(206, 237);
  box(3, 210, 'RMS → SwiGLU', 'ffn', { x: 25, w: 102 });
  down(227, 232);
  readout(4, 'RMSNorm');
}

/** Inline SVG with one keyboard target per chapter. The surrounding navigator handles activation. */
export function chapterMap(modelId, activeChapter = -1, language = 'zh') {
  const lang = language === 'en' ? 'en' : 'zh';
  const model = models.find((item) => item.id === modelId);
  if (!model) return '';
  const id = 'chapter-flow-' + ++sequence;
  const arrow = id + '-arrow';
  const g = makeDrawing(chapters(model.id), lang, arrow, activeChapter);
  if (model.id === 'llama') llama(g);
  else if (model.id === 'qwen') qwen(g);
  else classic(g, model.layers);
  const title = model.name + g.tr(' 架构章节图', ' architecture chapters');
  const desc =
    model.id === 'qwen'
      ? g.tr(
          '每组前三层各含 DeltaNet 和稠密 SwiGLU，第四层使用门控 GQA 和 SwiGLU。两条子层均有 RMSNorm 与残差。',
          'Three DeltaNet layers precede one gated GQA layer in each cycle. Every layer has a dense SwiGLU. Each sublayer uses RMSNorm and a residual connection.',
        )
      : g.tr(
          '沿箭头从嵌入走向输出。右侧的旁路在加号处合并残差。',
          'Follow the arrows from embedding to output. Skip paths rejoin at the plus signs.',
        );
  return `<svg xmlns="http://www.w3.org/2000/svg" class="chapter-flow" role="group" viewBox="0 0 152 306"
      width="152" height="306" style="display:block;width:100%;height:auto;max-height:335px"
      aria-labelledby="${id}-title" aria-describedby="${id}-description">
    <title id="${id}-title">${xml(title)}</title><desc id="${id}-description">${xml(desc)}</desc>
    <defs><marker id="${arrow}" viewBox="0 0 4 4" refX="3.55" refY="2" markerWidth="3.3" markerHeight="3.3" orient="auto">
      <path d="M0 .3 4 2 0 3.7z" fill="#65747a"/></marker></defs>
    <style>${styles}</style>${g.render()}</svg>`;
}
