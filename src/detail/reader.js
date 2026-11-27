import { escapeHTML as escape, icon } from '../ui.js';
import { chapters } from './catalog.js';
import { chapterMap } from './chapter-map.js';
import { messages } from './words.js?v=4';

const $ = (id) => document.getElementById(id);

function introduction(model, lang) {
  if (model.subtitle) return model.subtitle[lang];
  return lang === 'zh'
    ? `沿着 ${model.layers} 个解码块，追踪一个词元怎样得到下一个词元的分数。`
    : `Follow a token through ${model.layers} decoder blocks to its next-token scores.`;
}

function lesson(model, section, mode, lang) {
  const t = (zh, en) => (lang === 'zh' ? zh : en);
  if (!section) {
    if (mode === 'all')
      return {
        title: t('从全览开始', 'Start with the atlas'),
        text: t(
          '六个模型共用一个三维场景。点图中的模型名可拉近镜头。也可以从章节目录选一个构件，直接查看它的权重、激活和计算公式。',
          'Six networks share one 3D scene. Select a model label to frame its network, or choose a component from the contents to inspect its tensors and equations.',
        ),
      };
    return { title: t('一次前向传播', 'A forward pass'), text: model.scope[lang] };
  }
  const { stage, op } = section;
  const specific = model.stages?.find((item) => item.id === stage);
  // The three attention chapters visit different tensors, so explain the selected operation.
  if (op.includes('WeightBlock') && stage === 'attention')
    return {
      title: t('从隐藏状态得到 Q、K、V', 'Project the hidden state into Q, K and V'),
      text: t(
        '每个注意力头用三组权重从同一输入得到查询、键和值。选择其中一个权重单元，可以读取它连接的输入通道和输出通道。',
        'Each attention head projects the same input into a query, key and value. Inspect a weight to find the input and output channels it connects.',
      ),
      formula: 'Q = XWqᵀ + bq   ·   K = XWkᵀ + bk   ·   V = XWvᵀ + bv',
    };
  if (op === 'projWeight')
    return {
      title: t('合并各头的读出', 'Combine the heads'),
      text: t(
        '将各个头的读出拼接起来，再用输出投影混合通道，使更新可以加回残差流。',
        'Concatenate the head outputs. The output projection mixes their channels before the update rejoins the residual stream.',
      ),
      formula: 'Y = concat(H₁, …, Hₙ) Wₒᵀ + bₒ',
    };
  if (op === 'mlpResidual')
    return {
      title: t('把前馈更新加回残差流', 'Add the feed-forward update'),
      text: t(
        '选中一个位置、一个通道，查看来自注意力残差与前馈输出的两个加数。这个和成为下一层的输入。',
        'Inspect one position and channel to trace both addends: the attention residual and the feed-forward output. Their sum becomes the next block’s input.',
      ),
      formula: 'xₒᵤₜ = xₐₜₜₙ + FFN(LN(xₐₜₜₙ))',
    };
  if (specific)
    return { title: specific.title[lang], text: specific.summary[lang], formula: specific.formula };
  const generic = {
    embed: [
      t('词元与位置', 'Token and position'),
      t(
        '查找词元和位置各自对应的向量，再逐通道相加。',
        'Look up the token and position vectors, then add them channel by channel.',
      ),
      'xₜ = E[tokenₜ] + P[t]',
    ],
    norm: [
      t('逐位置归一化', 'Normalize each position'),
      t(
        '在同一位置沿隐藏通道计算均值与方差，归一化后应用学到的缩放和偏置。',
        'Compute mean and variance over the hidden channels of each position, then apply the learned scale and bias.',
      ),
      'LN(x) = γ ⊙ (x − μ) / √(Var(x) + ε) + β',
    ],
    attention: [
      t('读取此前的位置', 'Read the causal history'),
      t(
        '查询与键的点积给出匹配分数。因果掩码限定可见位置，Softmax 把这些位置的分数归一为权重。',
        'Dot products between queries and keys score the matches. The causal mask determines visible positions, and softmax normalizes their scores.',
      ),
      'A = softmax(QKᵀ / √d + mask)',
    ],
    ffn: [
      t('展开和压回通道', 'Expand and project the channels'),
      t(
        '每个位置单独经过前馈网络。第一层扩大通道，GELU 激活保留一部分信号，再将其投回隐藏宽度。',
        'The feed-forward network acts on each position independently. A projection expands the channels, GELU transforms them, and a second projection restores the hidden width.',
      ),
      'FFN(x) = down(GELU(up(x)))',
    ],
    output: [
      t('读出词表分数', 'Read out vocabulary scores'),
      t(
        '最终归一化后的隐藏状态投影到词表。最后一个位置的分数决定下一个词元的分布。',
        'The final normalized state is projected onto the vocabulary. The last position’s scores determine the next-token distribution.',
      ),
      'p(next) = softmax(Wₒ LN(xₜ))',
    ],
  };
  const [title, text, formula] = generic[stage] || generic.output;
  return { title, text, formula };
}

export function mountReader({ onChapter, onModelOverview, onAll }) {
  const panel = $('reader'),
    scroll = $('reader-scroll'),
    narrow = window.matchMedia('(max-width: 760px)');
  let model,
    lang = 'zh',
    items = [],
    current = -1,
    contentKey = '';
  const t = (zh, en) => (lang === 'zh' ? zh : en);
  function showMobile(open, focusButton = false) {
    const visible = open && narrow.matches;
    $('workspace').classList.toggle('reader-open', visible);
    $('reader-toggle').setAttribute('aria-expanded', String(visible));
    if (focusButton && narrow.matches) $('reader-toggle').focus({ preventScroll: true });
  }
  function scrollTo(id) {
    const target = $(id);
    if (!target || target.hidden) return;
    const offset = target.getBoundingClientRect().top - scroll.getBoundingClientRect().top;
    scroll.scrollTop += offset;
  }
  function openSection(id) {
    if (!['chapters', 'lab', 'references'].includes(id) || $(id).hidden) return;
    showMobile(true);
    requestAnimationFrame(() => scrollTo(id));
  }
  function forward() {
    if (!model) return;
    if (current + 1 >= items.length) {
      onAll();
      return;
    }
    onChapter(current + 1);
  }
  function stepFrom(event) {
    const button = event.target.closest('[data-chapter]');
    if (!button || !model) return;
    if (event.type === 'keydown') {
      if (event.key !== 'Enter' && event.key !== ' ') return;
      if (button.tagName.toLowerCase() === 'button') return;
      event.preventDefault();
    }
    onChapter(Number(button.dataset.chapter));
  }
  $('chapters').addEventListener('click', stepFrom);
  $('chapter-map').addEventListener('keydown', stepFrom);
  $('model-intro').addEventListener('click', onModelOverview);
  $('chapter-prev').addEventListener('click', () => {
    if (current > 0) onChapter(current - 1);
    else if (current === 0) onModelOverview();
  });
  $('chapter-next').addEventListener('click', forward);
  $('continue').addEventListener('click', () => {
    forward();
    scrollTo('chapters');
  });
  $('reader-toggle').addEventListener('click', () =>
    showMobile(!$('workspace').classList.contains('reader-open')),
  );
  $('reader-close').addEventListener('click', () => showMobile(false, true));
  narrow.addEventListener('change', () => showMobile(false));
  document.querySelector('.studio-nav').addEventListener('click', (event) => {
    const link = event.target.closest('a[href^="#"]');
    if (!link || event.metaKey || event.ctrlKey || event.altKey || event.shiftKey) return;
    event.preventDefault();
    if (location.hash !== link.hash) history.pushState(null, '', link.hash);
    openSection(link.hash.slice(1));
  });
  function followHash() {
    if (location.hash) openSection(location.hash.slice(1));
  }
  window.addEventListener('hashchange', followHash);
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && panel.contains(document.activeElement) && narrow.matches)
      showMobile(false, true);
  });
  return {
    followHash,
    revealScene() {
      showMobile(false);
    },
    render(next, state) {
      const changed = next.id !== model?.id;
      model = next;
      lang = state.lang;
      const { mode, playing } = state;
      items = chapters(model.id);
      const atOverview = mode === 'all' || mode === 'overview';
      current = atOverview ? -1 : items.findIndex((ch) => ch.op === state.operation);
      if (!atOverview && current < 0) current = items.findIndex((ch) => ch.stage === state.stage);
      const section = items[current],
        overviewLabel = messages[lang].modelOverview;
      const key = [model.id, lang, current, atOverview ? mode : 'chapter'].join('/');
      if (changed) scroll.scrollTop = 0;
      if (key !== contentKey) {
        const focused = document.activeElement;
        const focusGroup = $('chapter-map').contains(focused)
          ? $('chapter-map')
          : $('stage-list').contains(focused)
            ? $('stage-list')
            : null;
        const focusChapter = focused?.closest('[data-chapter]')?.dataset.chapter;
        contentKey = key;
        $('reader-model').textContent = model.name;
        $('reader-lead').textContent = introduction(model, lang);
        $('chapter-location').textContent = section ? section.label[lang] : t('总览', 'Overview');
        $('chapter-prev').disabled = current < 0;
        $('chapter-next').disabled = current === items.length - 1;
        $('model-intro').setAttribute('aria-pressed', String(atOverview));
        $('stage-list').innerHTML = items
          .map(
            (ch, i) => `<button type="button" data-chapter="${i}" aria-pressed="${i === current}">
          <span class="toc-number">${String(i + 1).padStart(2, '0')}</span><span>${escape(ch.label[lang])}</span></button>`,
          )
          .join('');
        $('chapter-map').innerHTML = chapterMap(model.id, current, lang);
        if (focusGroup && focusChapter !== undefined)
          focusGroup.querySelector(`[data-chapter="${focusChapter}"]`)?.focus({ preventScroll: true });
        const info = lesson(model, section, mode, lang);
        $('lesson-heading').textContent = info.title;
        $('stage-description').textContent = info.text;
        $('lesson-formula').hidden = !info.formula;
        $('lesson-formula').textContent = info.formula || '';
        $('lesson-demo').hidden = !section;
        const progress = $('chapter-progress');
        progress.max = items.length;
        progress.value = current + 1;
        progress.setAttribute('aria-label', t('章节进度', 'Chapter progress'));
        $('continue-text').textContent =
          current + 1 < items.length
            ? t('下一节 · ', 'Next · ') + items[current + 1].label[lang]
            : t('回到六模型全览', 'Back to all six models');
        $('chapter-location').title = section ? `${current + 1} / ${items.length}` : overviewLabel;
      }
      $('play').innerHTML =
        icon(playing ? 'pause' : 'play') + `<span>${messages[lang][playing ? 'pause' : 'play']}</span>`;
      $('play').setAttribute('aria-pressed', String(playing));
    },
  };
}
