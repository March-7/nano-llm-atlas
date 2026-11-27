import { escapeHTML as escape, icon } from './ui.js';
import { mountLab } from './lab.js?v=2';
import { inspectNano } from './nano.js?v=2';
import { models, chapters, tabLabels, isGPT2, isModern } from './detail/catalog.js';
import { classicLayout } from './detail/classic-layout.js';
import { gpt2Values } from './detail/gpt2-values.js';
import { mountTensorScene } from './detail/tensor-scene.js?v=4';
import { chapterPlayback } from './detail/chapter-playback.js';
import { phaseReadCount } from './detail/flow-paint.js';
import { modernExample } from './detail/modern.js';
import { mountReader } from './detail/reader.js?v=4';
import { createLocator } from './detail/locator.js';
import { cellKey, readCell, placeTooltip } from './detail/inspector.js';
import { messages, guideRows } from './detail/words.js?v=4';

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
  nanoRequest = 0;
let stage = 'norm',
  operation = 'ln2.lnResid',
  layer = 0,
  head = 0,
  mode = 'all',
  pinned = null,
  hover = null;
let labelState = true,
  orbitMode = false,
  tourRunning = false,
  loadTimer = 0,
  readController = null,
  weightStatus = '',
  readKey = '';
const releaseSamples = new Map();
const motionPreference = window.matchMedia('(prefers-reduced-motion: reduce)');
const player = $('motion-player');
let motion = null;
const playback = chapterPlayback({
  onFrame: displayPlayback,
  onComplete() {
    if (!tourRunning) return;
    const list = chapters(model.id);
    const index = list.findIndex((ch) => ch.op === operation);
    if (index >= 0 && index + 1 < list.length) goChapter(list[index + 1], true, { autoplay: true });
    else {
      tourRunning = false;
      drawChapters();
    }
  },
});
let multipleModelsInView = false;
const worldEntries = models.map(({ id, name }) => ({
  id,
  name,
  layout: isModern(id) ? modernExample(id).layout : classicLayout(id),
}));
const layouts = new Map(worldEntries.map((entry) => [entry.id, entry.layout]));
const worldPins = new Map(
  models.map((m) => {
    const pin = document.createElement('button');
    pin.type = 'button';
    pin.className = 'world-pin';
    pin.dataset.worldModel = m.id;
    pin.hidden = true;
    const name = document.createElement('strong');
    name.textContent = isModern(m.id) ? m.short : m.name;
    const size = document.createElement('small');
    size.textContent = m.displayParams + (isModern(m.id) ? ' · nano' : '');
    pin.append(name, size);
    $('world-labels').append(pin);
    return [m.id, pin];
  }),
);
const reader = mountReader({
  onChapter(index) {
    if (!layout) return;
    stopTour();
    goChapter(chapters(model.id)[index]);
  },
  onModelOverview() {
    stopTour();
    showOverview('overview');
  },
  onAll() {
    stopTour();
    showOverview('all');
  },
});
let appliedSearch = location.search;

document.querySelectorAll('i[data-icon]').forEach((slot) => {
  slot.innerHTML = icon(slot.dataset.icon);
});
const scene = mountTensorScene(canvas, {
  onCell: updateHover,
  onWorldLabels: placeWorldLabels,
  onModelPick: visitModel,
  onExitComputation: stopTour,
  playbackInsets() {
    const bounds = canvas.getBoundingClientRect();
    const tools = $('viewer').querySelector('.diagram-toolbar').getBoundingClientRect();
    return {
      top: tools.bottom - bounds.top + 19,
      bottom: bounds.bottom - player.getBoundingClientRect().top + 18,
    };
  },
  onOverview() {
    stopTour();
    showOverview('all');
  },
  onPick(hit) {
    if (hit) {
      stopTour();
      if (mode === 'overview' || mode === 'all') mode = 'detail';
      pin(hit);
    }
  },
  onZoom(ratio) {
    $('zoom').value =
      (ratio >= 1000 ? Number((ratio / 1000).toPrecision(3)) + 'k' : Number(ratio.toPrecision(3))) + '×';
  },
});
scene.setWorld(worldEntries);
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
  if (!hover || !model || (motion && motion.status !== 'ended' && cellKey(hover) === cellKey(pinned))) {
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
      refreshPlayback();
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
  drawCaption();
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
  refreshPlayback();
  drawCaption();
  try {
    const data = await inspectNano(next);
    if (request !== nanoRequest || model.id !== 'nanogpt') return;
    sample = data;
    samplePrefix = next;
    scene.setValues(sample);
    refreshPlayback();
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
  $('world-labels').setAttribute('aria-label', t('共享场景中的模型', 'Models in the shared scene'));
  for (const [id, button] of worldPins) {
    const name = models.find((m) => m.id === id).name;
    button.setAttribute('aria-label', t('聚焦 ' + name, 'Focus ' + name));
  }
  $('guide-body').innerHTML = guideRows(lang)
    .map(([a, b]) => `<div class="guide-row"><span>${escape(a)}</span><kbd>${escape(b)}</kbd></div>`)
    .join('');
  updateFullscreen();
}
function placeWorldLabels(markers) {
  const visible = new Set();
  for (const { id, x, y } of markers) {
    const button = worldPins.get(id);
    if (!button) continue;
    button.hidden = false;
    button.style.left = Math.round(x) + 'px';
    button.style.top = Math.round(y) + 'px';
    visible.add(id);
  }
  for (const [id, button] of worldPins) if (!visible.has(id)) button.hidden = true;
  const multiple = visible.size > 1;
  if (multiple !== multipleModelsInView) {
    multipleModelsInView = multiple;
    drawCaption();
  }
}
function drawTabs() {
  const list = $('model-list');
  if (!list.children.length)
    list.innerHTML = models
      .map((m) => `<button type="button" data-model="${m.id}">${tabLabels[m.id]}</button>`)
      .join('');
  for (const b of list.children) b.setAttribute('aria-pressed', String(b.dataset.model === model.id));
  for (const [id, pin] of worldPins) pin.setAttribute('aria-current', String(id === model.id));
}
function drawCaption() {
  if (!model) return;
  const panorama = mode === 'all' || multipleModelsInView;
  $('gallery-short').hidden = panorama;
  if (panorama) {
    $('view-model').textContent = t('模型全览', 'Model atlas');
    $('view-stats').textContent = t('6 个模型', '6 models');
    $('value-source').textContent = t(
      '各模型分别缩放 · 点击图中模型即可聚焦',
      'Models scaled individually · select a network to focus',
    );
    return;
  }
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
  reader.render(model, { stage, operation, mode, lang, playing: tourRunning });
  $('fit').setAttribute('aria-pressed', String(mode === 'all'));
}
function stopTour() {
  if (!motion && !tourRunning) return;
  tourRunning = false;
  playback.start([], false);
  drawChapters();
}
function pausePlayback() {
  tourRunning = false;
  playback.pause();
  drawChapters();
}
function displayPlayback(next) {
  const previous = motion;
  motion = next;
  player.hidden = !next;
  if (!next) {
    scene.displayComputation(null);
    refreshTip();
    return;
  }
  const { phase, index, count, progress, position, status } = next;
  const visibleProgress = motionPreference.matches ? 1 : progress;
  if (previous?.phase !== phase || player.lang !== lang) {
    player.lang = lang;
    $('motion-count').textContent = index + 1 + ' / ' + count;
    $('motion-title').textContent = phase.title[lang];
    $('motion-seek').setAttribute('aria-valuetext', index + 1 + ' / ' + count + ' · ' + phase.title[lang]);
    $('motion-prev').disabled = index === 0;
    $('motion-next').disabled = index === count - 1;
  }
  const visited = phase.readout && phaseReadCount(phase, visibleProgress);
  const formula = visited > 0 && visited < phase.readoutLength ? phase.readout(visited) : phase.formula;
  if ($('motion-formula').textContent !== formula) $('motion-formula').textContent = formula;
  const label =
    status === 'playing'
      ? t('暂停本节动画', 'Pause animation')
      : status === 'ended'
        ? t('重播本节动画', 'Replay animation')
        : t('播放本节动画', 'Play animation');
  const toggle = $('motion-toggle');
  if (toggle.title !== label) {
    toggle.title = label;
    toggle.setAttribute('aria-label', label);
    toggle.innerHTML = icon(status === 'playing' ? 'pause' : status === 'ended' ? 'reset' : 'play');
  }
  if (player.dataset.state !== status) player.dataset.state = status;
  $('motion-seek').value = Math.round((1000 * position) / count);
  scene.displayComputation(phase, visibleProgress, status === 'playing');
  if (!previous || previous.status !== status) refreshTip();
}
function refreshPlayback() {
  if (!motion || !pinned) return;
  playback.refresh(scene.prepareComputation(pinned, { animate: false, refresh: true }));
}
function preferred(op) {
  if (model.id === 'qwen') {
    if (op === 'deltaState') return layer % 4 === 3 ? 0 : layer;
    if (op === 'attnMtxSm') return 3;
  }
  return layer;
}
function goChapter(
  ch,
  save = true,
  { autoplay = !motionPreference.matches, cell = null, animate = true } = {},
) {
  if (!ch) return;
  mode = 'detail';
  stage = ch.stage;
  operation = ch.op;
  const n = cell?.node.op === ch.op ? cell.node : scene.findNode(operation, preferred(operation), head);
  if (!n) return;
  layer = n.layer < 0 ? layer : n.layer;
  const tMax = model.id === 'nanogpt' ? prefix.length - 1 : Infinity;
  const x = cell?.x ?? Math.min(4, n.cx - 1, n.axes[0] === 'T' ? tMax : Infinity);
  const y = cell?.y ?? Math.min(n.op === 'attnMtxSm' ? Math.min(5, tMax) : 11, n.cy - 1);
  const z = cell?.z ?? (n.op === 'deltaState' ? Math.min(2, n.cz - 1) : 0);
  const hit = scene.inspectCell(n, x, y, z);
  if (!hit) return;
  pin(hit, save);
  player.hidden = false;
  const phases = scene.prepareComputation(hit, { animate });
  playback.start(phases, autoplay);
  if (!phases.length) {
    scene.focus(n.op, n.layer, n.head);
    scene.inspectCell(n, x, y, z);
  }
}
function showOverview(view, save = true, animate = save) {
  if (!layout) return;
  pinned = null;
  mode = view;
  stage = 'embed';
  operation = isModern(model.id) ? 'embed' : 'residual0';
  if (view === 'all') scene.overview({ animate });
  else scene.fitModel({ animate });
  drawChapters();
  drawCaption();
  drawPinned();
  lab.selectStage(stage);
  if (save) writeURL();
}
function visitModel(id) {
  if (!layouts.has(id)) return;
  stopTour();
  if (model?.id === id && layout) {
    showOverview('overview');
    return;
  }
  chooseModel(id, new URLSearchParams({ view: 'network' }), true);
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
  if (mode === 'all') q.set('view', 'all');
  else if (mode === 'overview') q.set('view', 'network');
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

function chooseModel(id, route = null, save = false) {
  const next = models.find((m) => m.id === id) || models.find((m) => m.id === 'nanogpt');
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
  layout = layouts.get(next.id);
  if (isModern(next.id)) sample = modernExample(next.id).sample;
  else if (isGPT2(next.id)) {
    if (!releaseSamples.has(next.id)) releaseSamples.set(next.id, gpt2Values(next.id));
    sample = releaseSamples.get(next.id);
  }
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
  const addressed = route?.has('tensor') || route?.has('stage');
  const openWorld =
    route?.get('view') === 'all' || (!route?.has('model') && !addressed && !route?.has('view'));
  if (openWorld) showOverview('all', false, save);
  else if (route?.get('view') === 'network' || (!addressed && (isGPT2(next.id) || next.id === 'gpt3'))) {
    showOverview('overview', false, save);
  } else if (route?.has('stage') && !route.has('tensor') && route.get('view') !== 'cell') {
    goChapter(chapter, false, { animate: false });
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
  else reader.followHash();
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
  const button = event.target.closest('[data-model]');
  if (button) visitModel(button.dataset.model);
});
$('world-labels').addEventListener('click', (event) => {
  const button = event.target.closest('[data-world-model]');
  if (button) visitModel(button.dataset.worldModel);
});
$('play').addEventListener('click', () => {
  if (!layout) return;
  if (tourRunning) {
    pausePlayback();
    return;
  }
  const list = chapters(model.id);
  tourRunning = true;
  if (motion && motion.status !== 'ended') playback.play();
  else {
    let current = mode === 'all' || mode === 'overview' ? 0 : list.findIndex((ch) => ch.op === operation);
    if (current < 0 || (current === list.length - 1 && motion?.status === 'ended')) current = 0;
    goChapter(list[current], true, {
      autoplay: true,
      cell: pinned?.node.op === list[current].op ? pinned : null,
    });
  }
  drawChapters();
});
$('lesson-demo').addEventListener('click', () => {
  const ch =
    chapters(model.id).find((c) => c.op === operation) || chapters(model.id).find((c) => c.stage === stage);
  const cell = pinned?.node.op === ch?.op ? pinned : null;
  stopTour();
  goChapter(ch, true, { autoplay: true, cell });
});
$('motion-toggle').addEventListener('click', () => {
  if (motion?.status === 'playing') pausePlayback();
  else playback.play();
});
function seekMotion(position) {
  tourRunning = false;
  playback.seek(position);
  drawChapters();
}
$('motion-seek').addEventListener('input', (event) => {
  if (motion) seekMotion((Number(event.target.value) * motion.count) / 1000);
});
function stepMotion(direction) {
  if (motion) seekMotion(Math.max(0, Math.min(motion.count - 1, motion.index + direction)) + 0.88);
}
$('motion-prev').addEventListener('click', () => stepMotion(-1));
$('motion-next').addEventListener('click', () => stepMotion(1));
$('motion-close').addEventListener('click', () => {
  stopTour();
  if (pinned) {
    scene.focus(pinned.node.op, Math.max(0, pinned.node.layer), Math.max(0, pinned.node.head));
    scene.inspectCell(pinned.node, pinned.x, pinned.y, pinned.z);
  }
  canvas.focus({ preventScroll: true });
});
$('fit').addEventListener('click', () => {
  stopTour();
  showOverview('all');
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
  if (!pinned && !hover) {
    goChapter(
      chapters(model.id).find((ch) => ch.stage === (model.id === 'qwen' ? 'delta' : 'norm')),
      false,
    );
    stopTour();
  }
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
  stopTour();
  const h = pinned;
  mode = 'cell';
  reader.revealScene();
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
    if (motion) displayPlayback(playback.state);
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
  if (
    event.key === ' ' &&
    motion &&
    !event.target.closest?.(
      'button,input,select,a,textarea,dialog,[role="button"],[contenteditable="true"]',
    )
  ) {
    event.preventDefault();
    if (!event.repeat) {
      if (motion.status === 'playing') pausePlayback();
      else playback.play();
    }
    return;
  }
  if (event.key !== 'Escape') return;
  if (!$('locator').hidden) locator.close(true);
  else if ($('viewer').classList.contains('theater')) theater(false);
});
document.addEventListener('visibilitychange', () => {
  if (document.hidden && motion) pausePlayback();
});
motionPreference.addEventListener('change', () => {
  if (!motion) return;
  if (motionPreference.matches) pausePlayback();
  displayPlayback(playback.state);
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
