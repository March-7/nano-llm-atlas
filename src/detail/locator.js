import { axisNames } from './inspector.js';

export function createLocator(onGo) {
  const el = (id) => document.getElementById(id);
  const form = el('locator'),
    button = el('locate');
  const layer = el('layer'),
    head = el('head'),
    tensor = el('tensor');
  const fields = ['x', 'y', 'z'].map((axis) => el('coord-' + axis));
  let layout,
    options = [],
    lang = 'zh';
  const t = (zh, en) => (lang === 'zh' ? zh : en);
  const option = (label, value) => new Option(label, String(value));
  const nodes = () => (Number(layer.value) < 0 ? layout.root : layout.layers[Number(layer.value)].nodes());
  function updateFields(coords) {
    const n = options[tensor.selectedIndex];
    if (!n) return;
    const names = axisNames(n);
    fields.forEach((f, i) => {
      const count = [n.cx, n.cy, n.cz][i];
      const key = ['x', 'y', 'z'][i];
      f.max = count - 1;
      f.value = Math.max(0, Math.min(count - 1, Math.floor(coords?.[i] ?? Number(f.value)) || 0));
      el('axis-' + key).textContent = (names[i] || key) + ' · 0–' + (count - 1).toLocaleString('en-US');
    });
    el('coord-z-field').hidden = n.cz <= 1;
  }
  function listTensors(next) {
    const previous = next?.id ?? tensor.value;
    options = nodes().filter((n) => n.head < 0 || n.head === Number(head.value));
    tensor.replaceChildren(...options.map((n) => option(n.name + ' · ' + n.cx + '×' + n.cy, n.id)));
    if (options.some((n) => n.id === previous)) tensor.value = previous;
    else if (next?.op && options.some((n) => n.op === next.op))
      tensor.selectedIndex = options.findIndex((n) => n.op === next.op);
    updateFields();
  }
  function listHeads(next) {
    const heads = [
      ...new Set(
        nodes()
          .map((n) => n.head)
          .filter((h) => h >= 0),
      ),
    ].sort((a, b) => a - b);
    const old = Number(head.value);
    head.replaceChildren(
      ...(heads.length ? heads.map((h) => option('H' + (h + 1), h)) : [option(t('共用', 'Shared'), -1)]),
    );
    head.disabled = !heads.length;
    const desired = next?.head >= 0 ? next.head : old;
    if (heads.includes(desired)) head.value = desired;
    listTensors(next);
  }
  function close(focus = false) {
    form.hidden = true;
    button.setAttribute('aria-expanded', 'false');
    if (focus) button.focus({ preventScroll: true });
  }
  function open(hit) {
    form.hidden = false;
    button.setAttribute('aria-expanded', 'true');
    const n = hit?.node;
    if (n) {
      layer.value = n.layer;
      listHeads(n);
      updateFields([hit.x, hit.y, hit.z]);
    }
    layer.focus({ preventScroll: true });
  }
  layer.addEventListener('change', () => listHeads(options[tensor.selectedIndex]));
  head.addEventListener('change', () => listTensors(options[tensor.selectedIndex]));
  tensor.addEventListener('change', () => updateFields());
  el('locator-close').addEventListener('click', () => close(true));
  form.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') {
      event.stopPropagation();
      close(true);
    }
  });
  form.addEventListener('submit', (event) => {
    event.preventDefault();
    const n = options[tensor.selectedIndex];
    if (!n) return;
    close();
    onGo(n, ...fields.map((f) => Number(f.value)));
    el('scene').focus({ preventScroll: true });
  });
  return {
    close,
    toggle(hit) {
      if (form.hidden) open(hit);
      else close(true);
    },
    set(next, language) {
      layout = next;
      lang = language;
      const current = layer.value;
      layer.replaceChildren(
        option(t('嵌入与输出', 'Embedding & readout'), -1),
        ...layout.layers.map((group, i) => option(t('第 ' + (i + 1) + ' 层', 'Layer ' + (i + 1)), i)),
      );
      layer.value =
        current !== '' && Number(current) >= -1 && Number(current) < layout.layers.length ? current : '0';
      listHeads();
    },
  };
}
