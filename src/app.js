import { escapeHTML as escape, icon } from './ui.js';
import { mountLab } from './lab.js?v=2';
import { inspectNano } from './nano.js?v=2';
import { models, chapters, tabLabels, isGPT2, isModern } from './detail/catalog.js';
import { classicLayout } from './detail/classic-layout.js';
import { gpt2Values } from './detail/gpt2-values.js';
import { mountTensorScene } from './detail/tensor-scene.js';
import { createLocator } from './detail/locator.js';
import { cellKey, readCell, placeTooltip } from './detail/inspector.js';
import { messages, guideRows } from './detail/words.js';

const $ = (id) => document.getElementById(id);
const params = () => new URL(location.href).searchParams;
const languageKey = 'tensor-atlas-language';
function savedLang() {
  try {
    return localStorage.getItem(languageKey);
  } catch {
    return null;
  }
}
let lang = (params().get('lang') || savedLang()) === 'en' ? 'en' : 'zh';
const tr = (key) => messages[lang][key];
const t = (zh, en) => (lang === 'zh' ? zh : en);
const number = (n) => n.toLocaleString('en-US');
const integer = (v, fallback = 0) =>
  v !== null && v !== '' && Number.isInteger(Number(v)) ? Number(v) : fallback;
const external = (href, label, cls) =>
  `<a href="${escape(href)}" class="${cls}" target="_blank" rel="noopener noreferrer">${escape(label)}${icon('external')}</a>`;
const canvas = $('scene'),
  tip = $('cell-tip'),
  leader = $('callout-line');
let model,
  layout,
  sample,
  prefix = 'AACBAB',
  samplePrefix = '',
  pendingPrefix = '',
  nanoRequest = 0,
  modelRequest = 0;
let stage = 'norm',
  operation = 'ln2.lnResid',
  layer = 0,
  head = 0,
  mode = 'detail',
  pinned = null,
  hover = null;
let labelState = true,
  orbitMode = false,
  playTimer = 0,
  loadTimer = 0,
  readController = null,
  weightStatus = '',
  readKey = '';
const releaseSamples = new Map();
let appliedSearch = location.search;

document.querySelectorAll('i[data-icon]').forEach((slot) => {
  slot.innerHTML = icon(slot.dataset.icon);
});
const scene = mountTensorScene(canvas, {
  onCell: updateHover,
  onPick(hit) {
    if (hit) {
      stopTour();
      if (mode === 'overview') mode = 'detail';
      pin(hit);
    }
  },
  onZoom(ratio) {
    $('zoom').value =
      (ratio >= 1000 ? Number((ratio / 1000).toPrecision(3)) + 'k' : Number(ratio.toPrecision(3))) + '×';
  },
});
const locator = createLocator((node, x, y, z) => {
  stopTour();
  mode = 'cell';
  const hit = scene.focusCell(node, x, y, z);
  if (hit) pin(hit);
});
const lab = mountLab(
  $('lab'),
  (next, which = {}) => {
    if (!layout) return;
    stopTour();
    const item = chapters(model.id).find((ch) => ch.stage === next);
    if (which.layer != null) layer = which.layer;
    if (which.queryHead != null) head = which.queryHead % 4;
    if (item) goChapter(item);
  },
  updatePrefix,
);

function cancelRead() {
  clearTimeout(loadTimer);
  loadTimer = 0;
  readController?.abort();
  readController = null;
  weightStatus = '';
  readKey = '';
}
function refreshTip() {
  if (!hover || !model) {
    tip.hidden = true;
    leader.hidden = true;
    return;
  }
  const view = readCell(hover, model, sample, lang, prefix, weightStatus);
  if (tip.dataset.cell !== cellKey(hover) || tip.dataset.content !== view.body) {
    tip.dataset.cell = cellKey(hover);
    tip.dataset.content = view.body;
    tip.innerHTML = view.body;
  }
  placeTooltip(tip, leader, hover, canvas.clientWidth, canvas.clientHeight);
}
function updateHover(hit) {
  if (cellKey(hover) !== cellKey(hit)) cancelRead();
  hover = hit;
  refreshTip();
  if (
    !hit ||
    !sample?.loadCell ||
    hit.node.kind !== 'w' ||
    sample.value(hit.node, hit.x, hit.y, hit.z) != null
  )
    return;
  const key = cellKey(hit);
  if (readKey === key) return;
  readKey = key;
  const source = sample;
  const controller = new AbortController();
  readController = controller;
  loadTimer = setTimeout(async () => {
    try {
      const value = await source.loadCell(
        hit.node,
        hit.x,
        hit.y,
        AbortSignal.any([controller.signal, AbortSignal.timeout(18000)]),
      );
      if (controller.signal.aborted || source !== sample || key !== cellKey(hover)) return;
      weightStatus = value == null ? t('此权重未映射', 'Weight unavailable') : '';
      scene.refreshValues();
    } catch {
      if (controller.signal.aborted || source !== sample || key !== cellKey(hover)) return;
      weightStatus = t('未能取得权重 · 移开后再悬停可重试', 'Weight unavailable · hover again to retry');
    }
    refreshTip();
    drawPinned();
  }, 220);
}
function pin(hit, save = true) {
  pinned = hit;
  operation = hit.node.op;
  stage = hit.node.stage;
  if (hit.node.layer >= 0) layer = hit.node.layer;
  if (hit.node.head >= 0) head = hit.node.head;
  drawChapters();
  drawPinned();
  lab.selectStage(stage);
  if (save) writeURL();
}
function drawPinned() {
  $('picked-panel').hidden = !pinned;
  if (pinned)
    $('picked').innerHTML = readCell(
      pinned,
      model,
      sample,
      lang,
      prefix,
      cellKey(pinned) === cellKey(hover) ? weightStatus : '',
    ).pinned;
}
async function updatePrefix(next) {
  if (!/^[ABC]{1,11}$/.test(next)) return;
  prefix = next;
  if (model?.id !== 'nanogpt' || !layout || layout.id !== 'nanogpt') return;
  if ((sample && samplePrefix === next) || pendingPrefix === next) return;
  const request = ++nanoRequest;
  pendingPrefix = next;
  sample = null;
  samplePrefix = '';
  scene.setValues(null);
  drawCaption();
  try {
    const data = await inspectNano(next);
    if (request !== nanoRequest || model.id !== 'nanogpt') return;
    sample = data;
    samplePrefix = next;
    scene.setValues(sample);
    drawCaption();
    drawPinned();
    // Preserve the prefix of a shared or pinned measurement when the experiment advances.
    if (params().has('prefix') || next !== 'AACBAB') writeURL();
  } catch {
    if (request === nanoRequest && model.id === 'nanogpt')
      $('value-source').textContent = t(
        '权重载入失败，请重新载入页面',
        'Checkpoint unavailable. Reload to retry.',
      );
  } finally {
    if (request === nanoRequest) pendingPrefix = '';
  }
}

function applyCopy() {
  document.documentElement.lang = lang === 'zh' ? 'zh-CN' : 'en';
  document.title = 'Tensor Atlas · ' + tr('title');
  document.querySelectorAll('[data-t]').forEach((el) => {
    el.textContent = tr(el.dataset.t);
  });
  document.querySelectorAll('[data-tip]').forEach((el) => {
    el.title = tr(el.dataset.tip);
    el.setAttribute('aria-label', el.title);
  });
  $('language').querySelector('span').textContent = lang === 'zh' ? 'EN' : '中文';
  $('language').setAttribute('aria-label', lang === 'zh' ? 'Switch to English' : '切换到中文');
  canvas.setAttribute('aria-label', tr('cellCanvas'));
  $('model-list').setAttribute('aria-label', tr('select'));
  $('guide-body').innerHTML = guideRows(lang)
    .map(([a, b]) => `<div class="guide-row"><span>${escape(a)}</span><kbd>${escape(b)}</kbd></div>`)
    .join('');
  updateFullscreen();
}
function drawTabs() {
  const list = $('model-list');
  if (!list.children.length)
    list.innerHTML = models
      .map((m) => `<button type="button" data-model="${m.id}">${tabLabels[m.id]}</button>`)
      .join('');
  for (const b of list.children) b.setAttribute('aria-pressed', String(b.dataset.model === model.id));
}
function drawCaption() {
  if (!model) return;
  $('view-model').textContent = model.name;
  let size = `${model.displayParams} · ${model.layers} ${t('层', 'layers')}`;
  if (isModern(model.id))
    size += ` · ${t('图示', 'showing')} ${model.id === 'llama' ? 1 : 4} ${t('层', 'layers')}`;
  else size += ` · ${model.heads} ${t('头 / 层', 'heads / layer')}`;
  $('view-stats').textContent = size;
  $('value-source').textContent =
    model.id === 'nanogpt'
      ? (sample
          ? t('训练权重与实测激活', 'Trained weights & live activations')
          : t('载入训练权重…', 'Loading checkpoint…')) +
        ' · ' +
        prefix
      : isModern(model.id)
        ? t('数值来自缩小模型的固定教学参数', 'Values from the scaled model’s fixed teaching parameters')
        : isGPT2(model.id)
          ? t('公开权重按需读取 · 激活展示公式', 'Weights read on demand · equations for activations')
          : t('结构示意 · 未公开检查点', 'Architecture · unpublished checkpoint');
  const link = $('gallery-short');
  link.href = model.gallery.url;
  link.textContent =
    (model.id === 'gpt3'
      ? t('原论文配置', 'Original specifications')
      : t('二维图谱', '2D diagram') +
        (model.id === 'nanogpt' || model.id === 'llama' || model.id === 'gpt2-small'
          ? t(' · 同族', ' · related')
          : '')) + ' ↗';
  link.title = model.gallery.label[lang];
}
function drawSources() {
  $('facts').innerHTML = model.stats
    .map((v) => `<div><dt>${escape(v.label[lang])}</dt><dd>${escape(v.value)}</dd></div>`)
    .join('');
  const teaching = isModern(model.id) && layout?.scope ? ' ' + layout.scope[lang] : '';
  $('scope').textContent = model.scope[lang] + teaching;
  $('reading').innerHTML = `${external(model.gallery.url, model.gallery.label[lang], 'gallery-link')}
    <p>${escape(model.gallery.note[lang])}</p><div class="source-list">${model.sources.map((s) => external(s.url, s.label[lang], 'source-link')).join('')}</div>`;
}
function drawChapters() {
  if (!model) return;
  const items = chapters(model.id);
  let current = items.findIndex((ch) => ch.op === operation);
  if (current < 0) current = items.findIndex((ch) => ch.stage === stage);
  $('stage-list').innerHTML = items
    .map(
      (ch, i) =>
        `<button type="button" data-chapter="${i}" aria-pressed="${mode !== 'overview' && i === current}">${escape(ch.label[lang])}</button>`,
    )
    .join('');
  const description = model.stages?.find((s) => s.id === stage)?.summary[lang];
  $('stage-description').textContent =
    description ??
    t(
      '点击步骤定位到对应张量，或通过“定位张量”跳转到指定层、头与标量坐标。',
      'Select a step or use Find tensor to jump to a particular layer, head and scalar index.',
    );
  $('play').innerHTML =
    icon(playTimer ? 'pause' : 'play') + '<span>' + tr(playTimer ? 'pause' : 'play') + '</span>';
  $('play').setAttribute('aria-pressed', String(!!playTimer));
}
function stopTour() {
  if (!playTimer) return;
  clearInterval(playTimer);
  playTimer = 0;
  drawChapters();
}
function preferred(op) {
  if (model.id === 'qwen') {
    if (op === 'deltaState') return layer % 4 === 3 ? 0 : layer;
    if (op === 'attnMtxSm') return 3;
  }
  return layer;
}
function goChapter(ch, save = true) {
  mode = 'detail';
  stage = ch.stage;
  operation = ch.op;
  const n = scene.focus(operation, preferred(operation), head);
  if (!n) return;
  layer = n.layer < 0 ? layer : n.layer;
  const x = n.cx > 1 ? Math.min(4, n.cx - 1) : 0,
    y = n.cy > 1 ? Math.min(n.op === 'attnMtxSm' ? 5 : 11, n.cy - 1) : 0;
  const hit = scene.inspectCell(n, x, y, 0);
  if (hit) pin(hit, save);
}
function fitAll(save = true) {
  pinned = null;
  mode = 'overview';
  scene.overview();
  drawChapters();
  drawPinned();
  if (save) writeURL();
}
function writeURL(method = 'replace') {
  if (!model || !layout) return;
  const url = new URL(location.href);
  url.hash = '';
  for (const key of ['stage', 'tensor', 'head', 'layer', 'x', 'y', 'z', 'prefix', 'view'])
    url.searchParams.delete(key);
  const q = url.searchParams;
  q.set('model', model.id);
  q.set('lang', lang);
  if (mode === 'overview') q.set('view', 'network');
  else {
    q.set('stage', stage);
    q.set('tensor', operation);
    if ((pinned?.node.layer ?? layer) >= 0) q.set('layer', String(layer + 1));
    if (pinned?.node.head >= 0) q.set('head', String(pinned.node.head + 1));
    if (pinned) {
      q.set('x', String(pinned.x));
      q.set('y', String(pinned.y));
      if (pinned.z) q.set('z', String(pinned.z));
    }
    if (
      mode === 'cell' ||
      (pinned &&
        (pinned.z > 0 ||
          ((pinned.node.cx > 300 || pinned.node.cy > 300) && (pinned.x > 32 || pinned.y > 32))))
    )
      q.set('view', 'cell');
  }
  if (model.id === 'nanogpt' && prefix !== 'AACBAB') q.set('prefix', prefix);
  history[method === 'push' ? 'pushState' : 'replaceState'](null, '', url);
  appliedSearch = url.search;
}

async function chooseModel(id, route = null, save = false) {
  const next = models.find((m) => m.id === id) || models.find((m) => m.id === 'nanogpt');
  const request = ++modelRequest;
  ++nanoRequest;
  stopTour();
  cancelRead();
  locator.close();
  model = next;
  layout = null;
  sample = null;
  samplePrefix = '';
  pendingPrefix = '';
  hover = null;
  pinned = null;
  tip.hidden = true;
  leader.hidden = true;
  drawTabs();
  drawCaption();
  drawSources();
  const modern = isModern(id);
  let nextLayout, data;
  if (modern) {
    const { modernExample } = await import('./detail/modern.js');
    if (request !== modelRequest) return;
    const result = modernExample(id);
    nextLayout = result.layout;
    data = result.sample;
  } else {
    nextLayout = classicLayout(next.id);
    if (isGPT2(next.id)) {
      if (!releaseSamples.has(next.id)) releaseSamples.set(next.id, gpt2Values(next.id));
      data = releaseSamples.get(next.id);
    }
  }
  layout = nextLayout;
  sample = data ?? null;
  layer = Math.max(0, Math.min(layout.layers.length - 1, integer(route?.get('layer'), 1) - 1));
  head = Math.max(0, Math.min(model.heads - 1, integer(route?.get('head'), 1) - 1));
  scene.setLayout(layout);
  scene.setValues(sample);
  scene.setLabels(labelState);
  locator.set(layout, lang);
  const tour = chapters(model.id);
  const defaultStage = id === 'qwen' ? 'delta' : id === 'llama' ? 'attention' : 'norm';
  const chapter =
    tour.find((ch) => ch.stage === route?.get('stage')) ||
    tour.find((ch) => ch.stage === defaultStage) ||
    tour[0];
  const defaultOverview = (isGPT2(id) || id === 'gpt3') && !route?.get('tensor') && !route?.get('stage');
  if (route?.get('view') === 'network' || defaultOverview) {
    stage = 'embed';
    operation = 'residual0';
    fitAll(false);
  } else {
    const op = route?.get('tensor') || chapter.op;
    const node = scene.focus(op, preferred(op), head);
    if (node) {
      operation = node.op;
      stage = node.stage;
      mode = route?.get('view') === 'cell' ? 'cell' : 'detail';
      layer = node.layer < 0 ? layer : node.layer;
      const x = integer(route?.get('x'), Math.min(4, node.cx - 1)),
        y = integer(route?.get('y'), Math.min(op === 'attnMtxSm' ? 5 : 11, node.cy - 1)),
        z = integer(route?.get('z'), op === 'deltaState' ? Math.min(node.cz - 1, 4) : 0);
      const hit = mode === 'cell' ? scene.focusCell(node, x, y, z) : scene.inspectCell(node, x, y, z);
      if (hit) pin(hit, false);
    } else goChapter(chapter, false);
  }
  if (model.id === 'nanogpt' && /^[ABC]{6,11}$/.test(route?.get('prefix') || ''))
    lab.restorePrefix(route.get('prefix'));
  lab.show(model.id, lang, stage);
  $('lab-link').hidden = $('lab').hidden;
  if (model.id === 'nanogpt') updatePrefix(lab.currentPrefix() || prefix);
  drawCaption();
  drawSources();
  drawChapters();
  drawPinned();
  if (save) writeURL('push');
}

function updateFullscreen() {
  const active = document.fullscreenElement === $('viewer') || $('viewer').classList.contains('theater');
  $('expand').innerHTML = icon(active ? 'collapse' : 'expand');
  $('expand').setAttribute('aria-pressed', String(active));
  $('expand').setAttribute('aria-label', tr(active ? 'collapse' : 'expand'));
  $('expand').title = tr(active ? 'collapse' : 'expand');
}
function theater(flag) {
  $('viewer').classList.toggle('theater', flag);
  document.body.classList.toggle('in-theater', flag);
  updateFullscreen();
  $('expand').focus({ preventScroll: true });
}
$('expand').addEventListener('click', async () => {
  if (document.fullscreenElement) {
    await document.exitFullscreen();
    return;
  }
  if ($('viewer').classList.contains('theater')) {
    theater(false);
    return;
  }
  try {
    await $('viewer').requestFullscreen();
  } catch {
    theater(true);
  }
});
document.addEventListener('fullscreenchange', updateFullscreen);
$('model-list').addEventListener('click', (event) => {
  const b = event.target.closest('[data-model]');
  if (b && b.dataset.model !== model?.id) chooseModel(b.dataset.model, null, true);
});
$('stage-list').addEventListener('click', (event) => {
  const b = event.target.closest('[data-chapter]');
  if (!b || !layout) return;
  stopTour();
  goChapter(chapters(model.id)[Number(b.dataset.chapter)]);
  $('explorer').scrollIntoView({ block: 'start' });
  canvas.focus({ preventScroll: true });
});
$('play').addEventListener('click', () => {
  if (!layout) return;
  if (playTimer) {
    stopTour();
    return;
  }
  const list = chapters(model.id);
  let current = list.findIndex((ch) => ch.op === operation);
  if (current < 0 || current === list.length - 1) current = 0;
  goChapter(list[current]);
  $('explorer').scrollIntoView({ block: 'start' });
  playTimer = setInterval(() => {
    goChapter(list[++current]);
    if (current === list.length - 1) stopTour();
  }, 5000);
  drawChapters();
});
$('fit').addEventListener('click', () => {
  stopTour();
  if (layout) fitAll();
});
$('detail').addEventListener('click', () => {
  stopTour();
  if (!layout) return;
  const ch = chapters(model.id).find((ch) => ch.stage === (model.id === 'qwen' ? 'delta' : 'norm'));
  if (pinned) {
    const hit = pinned;
    mode = 'detail';
    scene.focus(hit.node.op, Math.max(0, hit.node.layer), Math.max(0, hit.node.head));
    pin(scene.inspectCell(hit.node, hit.x, hit.y, hit.z));
  } else goChapter(ch);
});
$('cell-zoom').addEventListener('click', () => {
  stopTour();
  if (!layout) return;
  if (!pinned && !hover)
    goChapter(
      chapters(model.id).find((ch) => ch.stage === (model.id === 'qwen' ? 'delta' : 'norm')),
      false,
    );
  mode = 'cell';
  const hit = scene.centerCell();
  if (hit) pin(hit);
});
$('locate').addEventListener('click', () => {
  stopTour();
  if (layout) locator.toggle(hover || pinned);
});
$('zoom-in').addEventListener('click', () => scene.zoomBy(1.5));
$('zoom-out').addEventListener('click', () => scene.zoomBy(1 / 1.5));
$('labels').addEventListener('click', () => {
  labelState = !labelState;
  scene.setLabels(labelState);
  $('labels').setAttribute('aria-pressed', String(labelState));
});
$('rotate').addEventListener('click', () => {
  orbitMode = !orbitMode;
  scene.setMode(orbitMode ? 'orbit' : 'pan');
  $('rotate').setAttribute('aria-pressed', String(orbitMode));
});
$('return-cell').addEventListener('click', () => {
  if (!pinned) return;
  const h = pinned;
  mode = 'cell';
  $('explorer').scrollIntoView();
  pin(scene.focusCell(h.node, h.x, h.y, h.z));
});
$('language').addEventListener('click', () => {
  lang = lang === 'zh' ? 'en' : 'zh';
  try {
    localStorage.setItem(languageKey, lang);
  } catch {}
  applyCopy();
  if (layout) {
    drawTabs();
    drawChapters();
    drawCaption();
    drawSources();
    locator.set(layout, lang);
    lab.show(model.id, lang, stage);
    drawPinned();
    refreshTip();
    writeURL();
  }
});
$('show-guide').addEventListener('click', () => {
  stopTour();
  $('guide-dialog').showModal();
});
$('close-guide').addEventListener('click', () => $('guide-dialog').close());
document.addEventListener('keydown', (event) => {
  if (event.key !== 'Escape') return;
  if (!$('locator').hidden) locator.close(true);
  else if ($('viewer').classList.contains('theater')) theater(false);
});
document.addEventListener('visibilitychange', () => {
  if (document.hidden) stopTour();
});
document.addEventListener(
  'keydown',
  (event) => {
    if (!$('viewer').classList.contains('theater')) return;
    if (event.key === 'Escape' && $('locator').hidden) {
      event.preventDefault();
      event.stopPropagation();
      theater(false);
    } else if (event.key === 'Tab') {
      const controls = [...$('viewer').querySelectorAll('canvas,button,input,select')].filter(
        (el) => !el.disabled && el.offsetParent !== null,
      );
      const first = controls[0],
        last = controls.at(-1);
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      }
      if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    }
  },
  true,
);
window.addEventListener('popstate', () => {
  if (location.search === appliedSearch) return;
  appliedSearch = location.search;
  const q = params();
  lang = (q.get('lang') || savedLang()) === 'en' ? 'en' : 'zh';
  applyCopy();
  chooseModel(q.get('model') || 'nanogpt', q);
});
applyCopy();
chooseModel(params().get('model') || 'nanogpt', params());
