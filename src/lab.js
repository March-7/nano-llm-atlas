import { escapeHTML, icon } from './ui.js';

const letters = 'ABC';

function asPercent(probability) {
  const percent = probability * 100;
  if (percent > 0 && percent < 0.0001) return percent.toExponential(1) + '%';
  if (percent > 99.9999 && percent < 100) return '>99.9999%';
  const digits = percent < 0.01 || percent > 99.99 ? 4 : 2;
  return percent.toFixed(digits) + '%';
}

const traceInfo = {
  embed: ['词元与位置嵌入', 'Token + position vector'],
  norm: ['第 1 层 · 注意力前的 LayerNorm', 'Block 1 · pre-attention LayerNorm'],
  attention: ['第 1 层 · 三个头的注意力概率', 'Block 1 · attention probabilities by head'],
  ffn: ['第 1 层 · GELU 激活', 'Block 1 · GELU activations'],
  output: ['输出层 · Softmax 概率', 'Readout · softmax probabilities'],
};

export function mountLab(host, onInspect) {
  let modelId, lang, stage;
  let input = 'AACBAB',
    generated = '',
    prediction = null,
    pending = false,
    problem = false,
    version = 0;
  let queryHead = 0,
    layer = 4;
  const t = (zh, en) => (lang === 'zh' ? zh : en);
  const el = (selector) => host.querySelector(selector);
  const valid = () => /^[ABC]{6}$/.test(input);
  const heading = (title, intro) => `
    <div class="lab-heading"><h2 id="lab-heading">${icon('beaker')}${title}</h2></div>
    <p class="lab-intro">${intro}</p>`;

  function showNano() {
    const intro = heading(
      t('用 NanoGPT 排序', 'Sort with NanoGPT'),
      t(
        '输入由 A、B、C 组成的六个字母。浏览器会运行原演示的训练权重，每一步选择概率最高的下一个字母。',
        'Enter six letters using A, B and C. The browser runs the original trained checkpoint and picks the most probable next letter at each step.',
      ),
    );
    host.innerHTML = `${intro}
      <div class="lab-content">
        <div class="experiment">
          <label class="field-label" for="sequence">${t('输入序列', 'Input sequence')}</label>
          <div class="lab-controls">
            <input id="sequence" class="sequence-input" type="text" spellcheck="false" autocomplete="off"
              autocapitalize="characters" aria-describedby="lab-message">
            <div class="lab-actions">
              <button type="button" id="generate" class="action-button">${icon('play')}${t('预测一个字母', 'Predict one letter')}</button>
              <button type="button" id="restart" class="minor-button">${icon('reset')}${t('重置输出', 'Clear output')}</button>
            </div>
          </div>
          <div class="token-strip" id="tokens"></div><p id="lab-message" class="lab-info" aria-live="polite"></p>
        </div>
        <div class="lab-output"><h3 id="probability-heading"></h3><div id="probabilities"></div></div>
      </div>
      <div class="trace-panel" id="trace"></div>`;
    el('#sequence').value = input;
    drawNano();
  }

  function drawNano() {
    if (modelId !== 'nanogpt') return;
    const isValid = valid();
    const step = Math.min(6, generated.length + 1);
    el('#sequence').setAttribute('aria-invalid', String(!isValid));
    const generate = el('#generate');
    generate.disabled = !isValid || pending || generated.length === 6;
    generate.innerHTML =
      icon(problem ? 'reset' : 'play') +
      (problem ? t('重试', 'Retry') : t('预测一个字母', 'Predict one letter'));
    el('#restart').disabled = generated.length === 0;
    el('#tokens').innerHTML =
      '<small>' +
      t('已生成', 'Generated') +
      '</small>' +
      Array.from(
        { length: 6 },
        (_, i) =>
          '<span class="' + (generated[i] ? 'filled' : '') + '">' + (generated[i] || '·') + '</span>',
      ).join('');
    const message = el('#lab-message');
    message.classList.toggle('error', !isValid || problem);
    message.textContent = !isValid
      ? t('请输入恰好六个字母，仅使用 A、B、C。', 'Enter exactly six letters using A, B and C.')
      : problem
        ? t(
            '权重加载失败，点击重试可重新载入。',
            'The checkpoint could not load. Select Retry to load it again.',
          )
        : pending
          ? t('正在计算…', 'Computing…')
          : generated.length === 6
            ? t(`生成完成，模型输出为 ${generated}。`, `Complete. The model produced ${generated}.`)
            : '';
    el('#probability-heading').textContent = t(
      '第 ' + step + ' 个输出字母的概率',
      'Probabilities for output letter ' + step,
    );
    el('#probabilities').innerHTML = Array.from(letters, (letter, i) => {
      const p = prediction?.probabilities[i];
      const width = p === undefined ? 0 : p * 100;
      const value = p === undefined ? '—' : asPercent(p);
      return (
        '<div class="probability"><span>' +
        letter +
        '</span><b><i style="width:' +
        width +
        '%"></i></b><span>' +
        value +
        '</span></div>'
      );
    }).join('');
    drawTrace();
  }

  function drawTrace() {
    if (modelId !== 'nanogpt') return;
    const panel = el('#trace'),
      rows = prediction?.traces?.[stage];
    panel.hidden = !rows;
    if (!rows) return;
    const context = input + generated.slice(0, 5);
    const isAttention = stage === 'attention',
      isProbability = isAttention || stage === 'output';
    const size = rows[0].length,
      shown = Math.min(size, 12);
    const max = Math.max(1e-8, ...rows.flatMap((row) => row.slice(0, shown).map(Math.abs)));
    const title = traceInfo[stage][lang === 'zh' ? 0 : 1];
    const position = context.length - 1;
    const localTitle = t('实测张量', 'Live tensor') + ' · ' + title;
    const format = (v) => (isProbability ? asPercent(v) : v.toFixed(3));
    const cells = rows
      .map((row, r) => {
        const header = isAttention ? 'H' + r : stage === 'output' ? 'p' : 'x';
        const data = row
          .slice(0, shown)
          .map((v, i) => {
            const key = isAttention ? context[i] + i : stage === 'output' ? letters[i] : 'd' + i;
            const alpha = (0.07 + (0.26 * Math.abs(v)) / max).toFixed(3);
            const rgb = v < 0 ? '218,137,91' : '125,98,183';
            return (
              '<div class="trace-value" style="background:rgba(' +
              rgb +
              ',' +
              alpha +
              ')"><small>' +
              key +
              '</small><span>' +
              format(v) +
              '</span></div>'
            );
          })
          .join('');
        return '<div class="trace-row"><b>' + header + '</b>' + data + '</div>';
      })
      .join('');
    const scope = !isProbability
      ? t('前 ' + shown + ' / ' + size + ' 维', 'first ' + shown + ' of ' + size + ' channels')
      : isAttention
        ? t('已见 ' + size + ' 个位置', size + ' visible positions')
        : t('三个候选字母', 'three candidates');
    panel.innerHTML =
      '<div class="trace-head"><b>' +
      escapeHTML(localTitle) +
      '</b><span>' +
      context[position] +
      ' · t=' +
      position +
      ' · ' +
      scope +
      '</span></div><div class="trace-scroll" role="group" aria-label="' +
      escapeHTML(localTitle) +
      '" style="--count:' +
      shown +
      '">' +
      cells +
      '</div>';
  }

  async function compute() {
    const request = ++version;
    prediction = null;
    problem = false;
    pending = valid();
    drawNano();
    if (!pending) return;
    try {
      const { predictNano } = await import('./nano.js');
      const result = await predictNano(input + generated.slice(0, 5), { trace: true });
      if (request === version) prediction = result;
    } catch {
      if (request === version) problem = true;
    } finally {
      if (request === version) {
        pending = false;
        drawNano();
      }
    }
  }

  function showLlama() {
    const groups = Array.from({ length: 8 }, (_, kv) => {
      const heads = Array.from({ length: 4 }, (_, i) => {
        const q = 4 * kv + i;
        return (
          '<button type="button" data-head="' +
          q +
          '" aria-label="' +
          t('查询头 ', 'Query head ') +
          q +
          '">' +
          q +
          '</button>'
        );
      }).join('');
      return '<div class="query-group">' + heads + '</div>';
    }).join('');
    const intro = heading(
      t('追踪一组共享键值头', 'Trace a shared KV pair'),
      t(
        '点选一个查询头，查看它在 8 组键值头中的归属。同组的四个查询头各自产生注意力分数。',
        'Pick any of the 32 query heads. Four heads share each KV pair while computing their own attention scores.',
      ),
    );
    host.innerHTML = `${intro}
      <div class="lab-content">
        <div class="experiment">
          <span class="lab-label">${t('查询头编号（从 0 开始）', 'Query head index (zero-based)')}</span>
          <div class="head-map">${groups}</div>
        </div>
        <div class="lab-output">
          <h3>${t('共享映射', 'Shared projections')}</h3><p id="lab-readout" class="lab-readout"></p>
          <p id="lab-message" class="lab-info" aria-live="polite"></p>
        </div>
      </div>`;
    drawMap();
  }

  function showQwen() {
    const layers = Array.from(
      { length: 64 },
      (_, i) =>
        '<button type="button" data-layer="' +
        (i + 1) +
        '" class="' +
        ((i + 1) % 4 ? '' : 'full') +
        '" aria-label="' +
        t('第 ' + (i + 1) + ' 层', 'Layer ' + (i + 1)) +
        '">' +
        (i + 1) +
        '</button>',
    ).join('');
    const intro = heading(
      t('定位 Qwen 的混合层', 'Locate a hybrid decoder layer'),
      t(
        '点选原模型的任一层。每四层中的前三层更新递归状态，第四层读取全部历史键值。',
        'Select any original layer. The first three in each cycle update recurrent state. The fourth attends over the full key/value history.',
      ),
    );
    host.innerHTML = `${intro}
      <div class="lab-content">
        <div class="experiment">
          <span class="lab-label">${t('语言解码器 · 第 1–64 层', 'Text decoder · layers 1–64')}</span>
          <div class="layer-map">${layers}</div>
        </div>
        <div class="lab-output">
          <h3 id="cycle-label"></h3><p id="lab-readout" class="lab-readout"></p>
          <p id="lab-message" class="lab-info" aria-live="polite"></p>
        </div>
      </div>`;
    drawMap();
  }

  function drawMap() {
    if (modelId === 'llama') {
      const kv = Math.floor(queryHead / 4),
        first = 4 * kv;
      host.querySelectorAll('[data-head]').forEach((button) => {
        const q = Number(button.dataset.head);
        button.tabIndex = q === queryHead ? 0 : -1;
        button.classList.toggle('shared', Math.floor(q / 4) === kv);
        button.setAttribute('aria-pressed', String(q === queryHead));
      });
      el('#lab-readout').textContent = 'Q' + queryHead + ' → K' + kv + ' / V' + kv;
      el('#lab-message').textContent = t(
        'Q' + first + ' 至 Q' + (first + 3) + ' 共用这组 K、V，单头宽度为 128。',
        'Q' +
          first +
          ' through Q' +
          (first + 3) +
          ' share these K and V heads. Each head has 128 channels.',
      );
    } else {
      host.querySelectorAll('[data-layer]').forEach((button) => {
        const selected = Number(button.dataset.layer) === layer;
        button.tabIndex = selected ? 0 : -1;
        button.setAttribute('aria-pressed', String(selected));
      });
      const cycle = Math.ceil(layer / 4),
        slot = ((layer - 1) % 4) + 1;
      el('#cycle-label').textContent = t(
        '第 ' + cycle + ' / 16 组 · 组内第 ' + slot + ' 层',
        'Cycle ' + cycle + ' / 16 · layer ' + slot + ' of 4',
      );
      el('#lab-readout').textContent = slot === 4 ? 'Gated GQA → SwiGLU' : 'Gated DeltaNet → SwiGLU';
      el('#lab-message').textContent =
        slot === 4
          ? t(
              '这一层缓存每个位置的 4 组键值，供 24 个查询头读取。',
              'This layer caches four KV pairs per position for 24 query heads.',
            )
          : t(
              '这一层将历史压入定长状态。它拥有独立的状态、归一化和前馈权重。',
              'This layer summarizes history in a fixed-size state. Its state, normalization and feed-forward weights belong to this layer.',
            );
    }
  }

  host.addEventListener('input', (event) => {
    if (event.target.id !== 'sequence') return;
    input = event.target.value.toUpperCase();
    generated = '';
    compute();
  });
  host.addEventListener('click', (event) => {
    const button = event.target.closest('button');
    if (!button) return;
    if (button.id === 'generate') {
      if (problem) {
        compute();
        return;
      }
      if (!prediction || pending || generated.length >= 6) return;
      generated += prediction.nextToken;
      if (generated.length === 6) drawNano();
      else compute();
    } else if (button.id === 'restart') {
      generated = '';
      compute();
    } else if (button.hasAttribute('data-head')) {
      queryHead = Number(button.dataset.head);
      drawMap();
      onInspect('attention');
    } else if (button.hasAttribute('data-layer')) {
      layer = Number(button.dataset.layer);
      drawMap();
      onInspect(layer % 4 ? 'delta' : 'attention');
    }
  });
  host.addEventListener('keydown', (event) => {
    const button = event.target.closest('[data-head],[data-layer]');
    if (!button) return;
    const name = button.hasAttribute('data-head') ? 'head' : 'layer';
    const min = name === 'head' ? 0 : 1,
      max = name === 'head' ? 31 : 64;
    const row = name === 'head' ? 2 : window.innerWidth <= 540 ? 8 : 16;
    const n = Number(button.dataset[name]);
    const keys = {
      ArrowRight: n + 1,
      ArrowLeft: n - 1,
      ArrowDown: n + row,
      ArrowUp: n - row,
      Home: min,
      End: max,
    };
    if (!(event.key in keys)) return;
    event.preventDefault();
    const target = el('[data-' + name + '="' + Math.max(min, Math.min(max, keys[event.key])) + '"]');
    target.focus();
    target.click();
  });
  return {
    show(id, locale, selection) {
      const previous = modelId;
      modelId = id;
      lang = locale;
      stage = selection;
      if (id === 'nanogpt') {
        showNano();
        if ((previous !== id || !prediction) && !pending) compute();
      } else if (id === 'llama') {
        showLlama();
      } else {
        if (stage === 'delta' && layer % 4 === 0) layer--;
        if (stage === 'attention' && layer % 4) layer = Math.ceil(layer / 4) * 4;
        showQwen();
      }
    },
    selectStage(selection) {
      stage = selection;
      if (modelId === 'nanogpt') drawTrace();
      if (modelId === 'qwen') {
        if (selection === 'attention' && layer % 4) layer = Math.ceil(layer / 4) * 4;
        if (selection === 'delta' && layer % 4 === 0) layer--;
        drawMap();
      }
    },
  };
}
