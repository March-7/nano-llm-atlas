import { models } from './catalog.js';
import { mountScene } from './scene.js';
import { mountLab } from './lab.js';
import { escapeHTML as e, icon, specimen } from './ui.js';

const words = {
  zh: {
    skip: '跳到模型探索器',
    nav: '导航',
    explore: '探索器',
    references: '参考资料',
    title: '语言模型解剖台',
    intro: '旋转三维张量，沿着一次前向传播，探索语言模型的内部结构。',
    guide: '操作指南',
    chooseModel: '选择模型',
    labels: '标注',
    explode: '分解',
    gesture: '拖动旋转 · 滚轮或双指缩放',
    zoom: '缩放比例',
    zoomOut: '缩小',
    zoomIn: '放大',
    reset: '重置相机',
    expand: '全屏查看',
    collapse: '退出全屏',
    canvas: '可旋转的三维张量模型。点击张量或使用下方构件导览查看说明。',
    inspector: '构件检查器',
    previous: '上一个构件',
    next: '下一个构件',
    view2d: '对照二维图谱',
    path: '构件导览',
    play: '自动播放',
    pause: '暂停导览',
    specs: '模型规格',
    reading: '二维图谱与来源',
    inspired: '交互灵感来自',
    close: '关闭',
    gallery: 'Architecture Gallery',
    nanoDesc: '85.7K 参数 · 字母排序',
    llamaDesc: 'Instruct · 分组查询注意力',
    qwenDesc: '稠密模型 · 混合注意力',
  },
  en: {
    skip: 'Skip to the model explorer',
    nav: 'Navigation',
    explore: 'Explorer',
    references: 'References',
    title: 'Inside a language model',
    intro: 'Rotate the tensors and follow a forward pass through three language model architectures.',
    guide: 'Controls',
    chooseModel: 'Choose a model',
    labels: 'Labels',
    explode: 'Explode',
    gesture: 'Drag to orbit · Scroll or pinch to zoom',
    zoom: 'Zoom level',
    zoomOut: 'Zoom out',
    zoomIn: 'Zoom in',
    reset: 'Reset camera',
    expand: 'Enter full screen',
    collapse: 'Exit full screen',
    canvas: 'Interactive 3D tensor model. Select a tensor or use the component walkthrough below.',
    inspector: 'Component inspector',
    previous: 'Previous component',
    next: 'Next component',
    view2d: 'Open 2D reference',
    path: 'Component walkthrough',
    play: 'Play tour',
    pause: 'Pause tour',
    specs: 'Model specifications',
    reading: '2D diagrams & sources',
    inspired: 'Interaction inspired by',
    close: 'Close',
    gallery: 'Architecture Gallery',
    nanoDesc: '85.7K parameters · Letter sorting',
    llamaDesc: 'Instruct · Grouped-query attention',
    qwenDesc: 'Dense model · Hybrid attention',
  },
};
const palettes = {
  embed: ['#64bca1', '#e6f5ef', '#28654f'],
  norm: ['#74add8', '#e8f2fa', '#3b6384'],
  delta: ['#74add8', '#e8f2fa', '#3b6384'],
  attention: ['#a288d9', '#efeaf9', '#674c95'],
  ffn: ['#d99a72', '#faeee5', '#8a5436'],
  output: ['#799fd9', '#e9eff9', '#486593'],
};
const titles = { nanogpt: 'NanoGPT', llama: 'Llama 3.1 8B', qwen: 'Qwen3.8 27B' };
const descriptions = { nanogpt: 'nanoDesc', llama: 'llamaDesc', qwen: 'qwenDesc' };
const stepNames = {
  embed: ['嵌入', 'Embedding'],
  norm: ['归一化', 'Normalization'],
  delta: ['DeltaNet ×3', 'DeltaNet ×3'],
  attention: ['注意力', 'Attention'],
  ffn: ['前馈网络', 'Feed-forward'],
  output: ['输出', 'Readout'],
};
const $ = (id) => document.getElementById(id);
const fromURL = () => new URL(location.href).searchParams;
function savedLanguage() {
  try {
    return localStorage.getItem('tensor-atlas-language');
  } catch {
    return null;
  }
}
function saveLanguage(value) {
  try {
    localStorage.setItem('tensor-atlas-language', value);
  } catch {
    /* Storage is optional. */
  }
}
const initialParams = fromURL();
let lang = (initialParams.get('lang') || savedLanguage()) === 'en' ? 'en' : 'zh';
let model = models.find((item) => item.id === initialParams.get('model')) || models[0];
let index = Math.max(
  0,
  model.stages.findIndex((s) => s.id === initialParams.get('stage')),
);
let labels = true,
  exploded = true,
  timer = 0;
const tr = (key) => words[lang][key];
const local = (text) => text[lang];
const two = (n) => String(n).padStart(2, '0');
const paletteStyle = (id) =>
  ['--spot', '--tint', '--deep'].map((prop, i) => prop + ':' + palettes[id][i]).join(';');
const external = (url, title, cls) =>
  `<a class="${cls}" href="${e(url)}" target="_blank" rel="noopener noreferrer">${e(title)}${icon('external')}</a>`;
const stageNow = () => model.stages[index];

document.querySelectorAll('i[data-icon]').forEach((slot) => {
  slot.innerHTML = icon(slot.dataset.icon);
});

const scene = mountScene($('scene'), {
  onSelect: (id) => chooseStage(id),
  onZoom: (value) => {
    $('zoom').textContent = Math.round(value * 100) + '%';
  },
  onHover: (id) => {
    document
      .querySelectorAll('[data-stage]')
      .forEach((b) => b.classList.toggle('hovered', b.dataset.stage === id));
  },
});
const lab = mountLab($('lab'), (id) => chooseStage(id));

function syncURL(mode = 'replace') {
  const url = new URL(location.href);
  url.searchParams.set('model', model.id);
  url.searchParams.set('stage', stageNow().id);
  url.searchParams.set('lang', lang);
  history[mode === 'push' ? 'pushState' : 'replaceState'](null, '', url);
}
function buttonTip(button, text) {
  button.title = text;
  button.setAttribute('aria-label', text);
}
function setCopy() {
  document.documentElement.lang = lang === 'zh' ? 'zh-CN' : 'en';
  document.title = 'Tensor Atlas · ' + tr('title');
  document.querySelectorAll('[data-t]').forEach((el) => {
    el.textContent = tr(el.dataset.t);
  });
  document.querySelectorAll('[data-aria]').forEach((el) => {
    el.setAttribute('aria-label', tr(el.dataset.aria));
  });
  document.querySelectorAll('[data-tip]').forEach((el) => buttonTip(el, tr(el.dataset.tip)));
  $('language').querySelector('span').textContent = lang === 'zh' ? 'EN' : '中文';
  buttonTip($('language'), lang === 'zh' ? 'Switch to English' : '切换到中文');
  $('guide-body').innerHTML = guideContent();
  setFullscreenUI();
}
function guideContent() {
  const rows =
    lang === 'zh'
      ? [
          ['旋转模型', '拖动 / 方向键'],
          ['缩放视图', '滚轮 / 双指 / + −'],
          ['平移视图', 'Shift + 拖动 / 双指拖动'],
          ['查看构件', '点击张量或构件按钮'],
          ['恢复默认相机', 'Home / 0'],
          ['退出全屏或指南', 'Esc'],
        ]
      : [
          ['Orbit the model', 'Drag / Arrow keys'],
          ['Zoom', 'Scroll / Pinch / + −'],
          ['Pan', 'Shift + drag / Two-finger drag'],
          ['Inspect a component', 'Select a tensor or a step'],
          ['Reset the camera', 'Home / 0'],
          ['Close full screen or guide', 'Esc'],
        ];
  const note =
    lang === 'zh'
      ? '先点击三维视窗或用 Tab 将焦点移入，再使用相机快捷键。网格颜色区分构件。在 NanoGPT 实验区可以读取实际张量数值。'
      : 'Click or tab into the viewport to use camera keys. Grid colors identify components. The NanoGPT experiment shows measured tensor values.';
  return (
    rows
      .map(([a, b]) => '<div class="guide-row"><span>' + a + '</span><kbd>' + b + '</kbd></div>')
      .join('') +
    '<p class="guide-note">' +
    note +
    '</p>'
  );
}
function drawModelChoices() {
  $('model-list').innerHTML = models
    .map(
      (item) => `
    <button type="button" class="model-choice" data-model="${item.id}" aria-pressed="${item.id === model.id}"
      aria-label="${e(item.name)}" title="${e(local(item.subtitle))}">
      <span class="thumb">${specimen(item.id)}</span>
      <span class="choice-copy"><b>${titles[item.id]}</b><small>${tr(descriptions[item.id])}</small></span>
      <span class="radio" aria-hidden="true"></span>
    </button>`,
    )
    .join('');
}
function drawPath() {
  $('stage-list').innerHTML = model.stages
    .map((s, i) => {
      let label = stepNames[s.id][lang === 'zh' ? 0 : 1];
      if (s.id === 'attention' && model.id !== 'nanogpt')
        label = model.id === 'llama' ? 'GQA' : lang === 'zh' ? '门控 GQA' : 'Gated GQA';
      const connector =
        i < model.stages.length - 1 ? '<span class="stage-split" aria-hidden="true"></span>' : '';
      return `<div class="stage-wrap">
      <button type="button" class="stage-button" data-stage="${s.id}" style="${paletteStyle(s.id)}"
        aria-label="${two(i + 1)} · ${e(local(s.title))}" aria-pressed="${i === index}">
        <span class="stage-number">${two(i + 1)}</span><span class="stage-label">${e(label)}</span>
      </button>${connector}
    </div>`;
    })
    .join('');
}
function drawComponent() {
  const selected = stageNow(),
    card = $('component'),
    name = local(selected.title);
  card.setAttribute('style', paletteStyle(selected.id));
  const dims = selected.details
    .map((metric) => '<div><dt>' + e(local(metric.label)) + '</dt><dd>' + e(metric.value) + '</dd></div>')
    .join('');
  const jumpLabels = {
    nanogpt: ['查看该构件的实测张量', 'Inspect its live tensor'],
    llama: ['查看 32 个查询头的分组', 'Explore the 32 query heads'],
    qwen: ['查看 64 层的排布', 'Explore all 64 layers'],
  };
  card.innerHTML = `
    <span class="tag">${e(selected.tag)}</span>
    <h3>${e(name)}</h3><p>${e(local(selected.summary))}</p>
    <code class="formula">${e(selected.formula)}</code>
    <dl class="dimensions">${dims}</dl>
    <a class="lab-jump" href="#lab">${icon('beaker')}${jumpLabels[model.id][lang === 'zh' ? 0 : 1]} ↓</a>`;
  $('fullscreen-component').innerHTML = `<b>${e(name)}</b><span>${e(selected.formula)}</span>`;
  $('step-index').textContent = two(index + 1) + ' / ' + two(model.stages.length);
  $('previous').disabled = index === 0;
  $('next').disabled = index === model.stages.length - 1;
  document
    .querySelectorAll('[data-stage]')
    .forEach((button) => button.setAttribute('aria-pressed', String(button.dataset.stage === selected.id)));
  $('scene').setAttribute('aria-label', tr('canvas') + ' ' + model.name + ' · ' + name);
  scene.select(selected.id);
}
function drawSources() {
  $('facts').innerHTML = model.stats
    .map((m) => '<div><dt>' + e(local(m.label)) + '</dt><dd>' + e(m.value) + '</dd></div>')
    .join('');
  $('scope').textContent = local(model.scope);
  const guide = model.gallery,
    label = local(guide.label);
  $('reading').innerHTML = `${external(guide.url, label, 'gallery-link')}
    <p>${e(local(guide.note))}</p>
    <div class="source-list">${model.sources.map((s) => external(s.url, local(s.label), 'source-link')).join('')}</div>`;
  const link = $('gallery-short');
  link.href = guide.url;
  link.title = label + ' · ' + tr('gallery');
  link.querySelector('span').textContent =
    model.id === 'qwen'
      ? tr('view2d')
      : lang === 'zh'
        ? '二维图谱 · 同族参考'
        : '2D diagram · related model';
  link.setAttribute('aria-label', tr('view2d') + ' · ' + label);
}
function showModel() {
  $('view-model').textContent = model.name;
  const depth = model.stats.find((stat) => stat.shown);
  $('display-scale').textContent =
    lang === 'zh'
      ? `图示 ${depth.shown} / ${depth.value} 层`
      : `${depth.shown} of ${depth.value} layers shown`;
  drawModelChoices();
  drawPath();
  drawSources();
  scene.setModel(model.id);
  scene.setLabels(labels);
  scene.setExploded(exploded);
  drawComponent();
  drawPlay();
  lab.show(model.id, lang, stageNow().id);
}
function setIndex(next, sync = true) {
  if (next < 0 || next >= model.stages.length || next === index) return;
  index = next;
  drawComponent();
  lab.selectStage(stageNow().id);
  if (sync) syncURL();
}
function chooseStage(id) {
  stopTour();
  setIndex(model.stages.findIndex((s) => s.id === id));
}
function renderLang() {
  setCopy();
  showModel();
}
function drawPlay() {
  $('play').innerHTML =
    icon(timer ? 'pause' : 'play') + '<span>' + tr(timer ? 'pause' : 'play') + '</span>';
  $('play').setAttribute('aria-pressed', String(Boolean(timer)));
}
function stopTour() {
  if (!timer) return;
  clearInterval(timer);
  timer = 0;
  drawPlay();
}
function toggleTour() {
  if (timer) {
    stopTour();
    return;
  }
  if (index === model.stages.length - 1) setIndex(0);
  timer = setInterval(() => {
    setIndex(index + 1);
    if (index === model.stages.length - 1) stopTour();
  }, 5500);
  drawPlay();
}
function setFullscreenUI() {
  const active = document.fullscreenElement === $('viewer') || $('viewer').classList.contains('theater');
  const button = $('expand');
  button.innerHTML = icon(active ? 'collapse' : 'expand');
  button.setAttribute('aria-pressed', String(active));
  buttonTip(button, tr(active ? 'collapse' : 'expand'));
}
function fallbackTheater(enabled) {
  $('viewer').classList.toggle('theater', enabled);
  document.body.classList.toggle('in-theater', enabled);
  setFullscreenUI();
  $('expand').focus({ preventScroll: true });
}
async function toggleFullscreen() {
  if (document.fullscreenElement) {
    await document.exitFullscreen();
    return;
  }
  if ($('viewer').classList.contains('theater')) {
    fallbackTheater(false);
    return;
  }
  if ($('viewer').requestFullscreen) {
    try {
      await $('viewer').requestFullscreen();
      return;
    } catch {
      /* Some embedded browsers use the CSS viewport. */
    }
  }
  fallbackTheater(true);
}
$('model-list').addEventListener('click', (event) => {
  const choice = event.target.closest('[data-model]');
  if (!choice || choice.dataset.model === model.id) return;
  stopTour();
  model = models.find((item) => item.id === choice.dataset.model);
  index = 0;
  showModel();
  syncURL('push');
  $('model-list')
    .querySelector('[data-model="' + model.id + '"]')
    .focus({ preventScroll: true });
});
$('stage-list').addEventListener('click', (event) => {
  const step = event.target.closest('[data-stage]');
  if (step) chooseStage(step.dataset.stage);
});
$('previous').addEventListener('click', () => {
  stopTour();
  setIndex(index - 1);
});
$('next').addEventListener('click', () => {
  stopTour();
  setIndex(index + 1);
});
$('play').addEventListener('click', toggleTour);
$('language').addEventListener('click', () => {
  lang = lang === 'zh' ? 'en' : 'zh';
  saveLanguage(lang);
  renderLang();
  syncURL();
});
$('labels').addEventListener('click', () => {
  labels = !labels;
  scene.setLabels(labels);
  $('labels').setAttribute('aria-pressed', String(labels));
});
$('explode').addEventListener('click', () => {
  exploded = !exploded;
  scene.setExploded(exploded);
  $('explode').setAttribute('aria-pressed', String(exploded));
});
$('zoom-in').addEventListener('click', () => scene.zoomBy(1.17));
$('zoom-out').addEventListener('click', () => scene.zoomBy(1 / 1.17));
$('reset').addEventListener('click', () => scene.reset());
$('expand').addEventListener('click', toggleFullscreen);
document.addEventListener('fullscreenchange', setFullscreenUI);
$('show-guide').addEventListener('click', () => {
  stopTour();
  $('guide-dialog').showModal();
});
$('close-guide').addEventListener('click', () => $('guide-dialog').close());
$('guide-dialog').addEventListener('click', (event) => {
  if (event.target !== event.currentTarget) return;
  const b = event.currentTarget.getBoundingClientRect();
  if (
    event.clientX < b.left ||
    event.clientX > b.right ||
    event.clientY < b.top ||
    event.clientY > b.bottom
  )
    event.currentTarget.close();
});
document.addEventListener('keydown', (event) => {
  if (!$('viewer').classList.contains('theater')) return;
  if (event.key === 'Escape') {
    fallbackTheater(false);
    return;
  }
  if (event.key === 'Tab') {
    const focusable = $('viewer').querySelectorAll('button,canvas');
    const first = focusable[0],
      last = focusable[focusable.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    }
    if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  }
});
document.addEventListener('visibilitychange', () => {
  if (document.hidden) stopTour();
});
new IntersectionObserver(
  (entries) => {
    const section = entries[0].isIntersecting ? 'references' : 'explorer';
    document.querySelectorAll('[data-nav]').forEach((a) => {
      if (a.dataset.nav === section) a.setAttribute('aria-current', 'location');
      else a.removeAttribute('aria-current');
    });
  },
  { threshold: 0.35 },
).observe($('references'));
window.addEventListener('popstate', () => {
  stopTour();
  const params = fromURL();
  lang = (params.get('lang') || savedLanguage()) === 'en' ? 'en' : 'zh';
  model = models.find((item) => item.id === params.get('model')) || models[0];
  index = Math.max(
    0,
    model.stages.findIndex((s) => s.id === params.get('stage')),
  );
  renderLang();
});
renderLang();
