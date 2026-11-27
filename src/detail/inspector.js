import { escapeHTML as e } from '../ui.js';
import { isModern, isGPT2 } from './catalog.js';

export const cellKey = (hit) => (hit ? [hit.node.id, hit.x, hit.y, hit.z].join('/') : '');
function digits(n) {
  if (n === -Infinity) return '−∞';
  if (n === Infinity) return '∞';
  return Number(n.toPrecision(7)).toString().replace(/^-/, '−');
}
export function axisNames(node) {
  const axes = node.axes.map(
    (a) => ({ 1: '', B: 'b', T: 't', C: 'c', A: 'd', '4C': 'f', V: 'v', '3A': '3d' })[a] ?? a,
  );
  if (node.op.startsWith('attnMtx') && node.cx > 1 && node.cy > 1) return ['s', 't', axes[2]];
  if (axes[0] && axes[0] === axes[1]) return [axes[0] + 'ᵢ', axes[1] + 'ₒ', axes[2]];
  return axes;
}
function equation(node, model, basic) {
  if (!isModern(model.id) && node.op.endsWith('.lnResid'))
    return `<math displaystyle="true" aria-label="LayerNorm"><mrow><mfrac><mrow><msub><mi>x</mi><mrow><mi>t</mi><mo>,</mo><mi>c</mi></mrow></msub><mo>−</mo><msub><mi>μ</mi><mi>t</mi></msub></mrow><msqrt><mrow><msub><mi mathvariant="normal">Var</mi><mi>t</mi></msub><mo>+</mo><mi>ε</mi></mrow></msqrt></mfrac><mo>·</mo><msub><mi>γ</mi><mi>c</mi></msub><mo>+</mo><msub><mi>β</mi><mi>c</mi></msub></mrow></math>`;
  return '<span>' + e(basic || node.formula || node.name) + '</span>';
}
export function readCell(hit, model, sample, lang, prefix, status = '') {
  const t = (zh, en) => (lang === 'zh' ? zh : en);
  const { node, x, y, z } = hit;
  const value = sample?.value(node, x, y, z);
  const detail = sample?.explain(node, x, y, z);
  const names = axisNames(node),
    counts = [node.cx, node.cy, node.cz],
    indices = [x, y, z];
  const coords = names
    .map((name, i) =>
      counts[i] > 1 || (i < 2 && name && name !== '1')
        ? `<span class="axis-${['x', 'y', 'z'][i]}">${e(name || ['x', 'y', 'z'][i])}: ${indices[i]}</span>`
        : '',
    )
    .join('');
  const path = [node.layer >= 0 ? 'L' + (node.layer + 1) : '', node.head >= 0 ? 'H' + (node.head + 1) : '']
    .filter(Boolean)
    .join(' / ');
  const shape = counts
    .slice(0, node.cz === 1 ? 2 : 3)
    .map(String)
    .join(' × ');
  let provenance;
  if (model.id === 'nanogpt') {
    provenance =
      node.kind === 'w'
        ? t('训练权重', 'Trained checkpoint')
        : t('前向计算', 'Forward pass') + ' · ' + prefix;
    if (value == null && sample)
      provenance = t('此位置尚未进入当前前缀', 'Position outside the current prefix') + ' · ' + prefix;
  } else if (isModern(model.id)) {
    provenance = t('缩小模型 · 固定教学参数', 'Scaled model · teaching parameters');
  } else if (isGPT2(model.id) && node.kind === 'w') {
    provenance =
      status ||
      (value != null
        ? t('公开检查点 · 按需读取', 'Public checkpoint · range read')
        : t('正在读取公开权重…', 'Reading public weight…'));
  } else if (model.id === 'gpt3') {
    provenance = t('结构示意 · GPT-3 175B 权重未公开', 'Architecture · GPT-3 175B weights are unpublished');
  } else {
    provenance = t('结构示意 · 激活取决于输入', 'Architecture · activations depend on input');
  }
  const result = value == null ? '' : ' <b class="tip-result">= ' + e(digits(value)) + '</b>';
  const formula = detail?.formula === '∅' ? node.formula : detail?.formula;
  const symbol = equation(node, model, formula);
  const note = detail?.substitution || '';
  const pathNote = detail?.source ? ` · ${e(detail.source)}` : '';
  const body = `<div class="tip-heading"><strong>${e(node.name || node.op)}</strong><small>${path} · ${shape}</small></div>
    <div class="tip-coords">${coords}</div><div class="tip-formula">${symbol}${result}</div>
    <code class="tip-substitution">${e(note)}</code><span class="tip-source">${e(provenance)}${pathNote}</span>`;
  const label = `${node.name} [${x}, ${y}]${value == null ? '' : ' = ' + digits(value)}`;
  const pinned = `<b>${e(label)}</b> · ${path ? e(path) + ' · ' : ''}${shape}<br><code>${e(formula || node.formula || '')}${note ? '　' + e(note) : ''}</code><br>${e(provenance)}${pathNote}`;
  return { body, pinned, label, value };
}
export function placeTooltip(tip, leader, hit, width, height) {
  if (!hit || hit.screenX < 0 || hit.screenY < 0 || hit.screenX >= width || hit.screenY >= height) {
    tip.hidden = true;
    leader.hidden = true;
    return;
  }
  tip.hidden = false;
  leader.hidden = false;
  const w = tip.offsetWidth,
    h = tip.offsetHeight;
  const { screenX: x, screenY: y } = hit,
    margin = 8,
    gap = 18;
  const clamp = (v, min, max) => Math.max(min, Math.min(v, max));
  const left = clamp(x + gap + w <= width - margin ? x + gap : x - w - gap, margin, width - w - margin);
  const top = clamp(y - h - gap > margin ? y - h - gap : y + gap, margin, height - h - margin);
  tip.style.left = Math.round(left) + 'px';
  tip.style.top = Math.round(top) + 'px';
  const endX = clamp(x, left + 4, left + w - 4),
    endY = top > y ? top : top + h;
  const color = hit.node.kind === 'w' ? '#6876a2' : '#608762';
  const path = leader.querySelector('path'),
    dot = leader.querySelector('circle');
  path.setAttribute('d', `M ${x} ${y} L ${endX} ${endY}`);
  path.setAttribute('stroke', color);
  dot.setAttribute('cx', x);
  dot.setAttribute('cy', y);
  dot.setAttribute('fill', color);
}
