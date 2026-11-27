const paths = {
  github:
    '<path fill="currentColor" stroke="none" d="M12 .9a11.1 11.1 0 0 0-3.51 21.63c.55.1.75-.24.75-.53v-2.07c-3.07.67-3.72-1.3-3.72-1.3-.5-1.28-1.23-1.62-1.23-1.62-1-.69.08-.68.08-.68 1.11.08 1.69 1.14 1.69 1.14.99 1.69 2.6 1.2 3.24.92.1-.72.39-1.2.7-1.48-2.45-.28-5.03-1.23-5.03-5.47 0-1.21.43-2.2 1.14-2.98-.12-.28-.5-1.41.11-2.94 0 0 .93-.3 3.05 1.14a10.56 10.56 0 0 1 5.55 0c2.12-1.44 3.05-1.14 3.05-1.14.61 1.53.23 2.66.11 2.94.71.78 1.14 1.77 1.14 2.98 0 4.25-2.59 5.18-5.05 5.46.4.35.75 1.02.75 2.06V22c0 .3.2.64.76.53A11.1 11.1 0 0 0 12 .9Z"/>',
  globe: '<circle cx="12" cy="12" r="9"/><ellipse cx="12" cy="12" rx="4" ry="9"/><path d="M3 12h18"/>',
  compass: '<circle cx="12" cy="12" r="9"/><path d="m16.3 7.7-2.5 6.1-6.1 2.5 2.5-6.1z"/>',
  box: '<path d="m12 2.8 8.5 4.8v8.9l-8.5 4.8-8.5-4.8V7.6zM3.7 7.8l8.3 4.7 8.3-4.7M12 12.5v8.6"/>',
  tag: '<path d="M3.5 4h8.2L21 13.3 13.3 21 3.5 11.2z"/><circle cx="8" cy="8.4" r="1.15" fill="currentColor" stroke="none"/>',
  layers: '<path d="m3 9 9-5 9 5-9 5zM3.5 13l8.5 4.7 8.5-4.7M3.5 17l8.5 4.7 8.5-4.7"/>',
  scan: '<path d="M8 3.5H4v4M16 3.5h4v4M8 20.5H4v-4M16 20.5h4v-4"/><path d="m12 7 4.5 2.7v5L12 17.3l-4.5-2.6v-5z"/>',
  orbit:
    '<ellipse cx="12" cy="12" rx="10" ry="4.6" transform="rotate(-27 12 12)"/><circle cx="12" cy="12" r="4.9"/><circle cx="20" cy="6.7" r="1.6" fill="currentColor" stroke="none"/>',
  minus: '<path d="M5 12h14"/>',
  plus: '<path d="M5 12h14m-7-7v14"/>',
  reset: '<path d="M4 10a8 8 0 1 1 .8 6.1M4 4v6h6"/>',
  expand: '<path d="M8 3.5H3.5V8m12-4.5H20V8M3.5 16v4.5H8m12-4.5v4.5h-4.5"/>',
  collapse: '<path d="M3.5 8H8V3.5M20.5 8H16V3.5M8 20.5V16H3.5m12.5 4.5V16h4.5"/>',
  left: '<path d="m14.5 5-7 7 7 7"/>',
  right: '<path d="m9.5 5 7 7-7 7"/>',
  arrow: '<path d="M4 12h15m-6-6 6 6-6 6"/>',
  external: '<path d="M6 18 18 6M7 6h11v11"/>',
  route:
    '<circle cx="5" cy="17.5" r="2"/><circle cx="19" cy="6.5" r="2"/><path d="M7 17.5h8a3 3 0 0 0 0-6H9a2.5 2.5 0 0 1 0-5h8"/>',
  play: '<path d="m7 4.5 12 7.5L7 19.5z" fill="currentColor" stroke-width="1.2"/>',
  pause: '<path d="M8 5v14m8-14v14" stroke-width="3.4"/>',
  sliders:
    '<path d="M4 6h7m6 0h3M4 12h1m6 0h9M4 18h10m5 0h1"/><circle cx="14" cy="6" r="2.5"/><circle cx="8" cy="12" r="2.5"/><circle cx="16.5" cy="18" r="2.5"/>',
  book: '<path d="M12 6c-2-1.8-5.5-2.5-9-1.7v13c3.5-.8 7-.1 9 1.7 2-1.8 5.5-2.5 9-1.7v-13c-3.5-.8-7-.1-9 1.7Zm0 0v13"/>',
  beaker: '<path d="M9 3h6m-5 0v6l-5.8 9a2 2 0 0 0 1.7 3h12.2a2 2 0 0 0 1.7-3L14 9V3M7 15h10"/>',
  grid: '<rect x="3.5" y="3.5" width="17" height="17" rx="2"/><path d="M3.5 10h17M10 3.5v17"/>',
  link: '<path d="m10 13.5 4-4m-3.4-3 1.6-1.6a4.3 4.3 0 0 1 6.1 6.1l-1.6 1.6m-3.3 4.9-1.6 1.6a4.3 4.3 0 0 1-6.1-6.1l1.6-1.6"/>',
  close: '<path d="m5.5 5.5 13 13m0-13-13 13"/>',
  check: '<path d="m4 12 5 5L20 6"/>',
};

export function icon(name) {
  return (
    '<svg class="icon" viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">' +
    paths[name] +
    '</svg>'
  );
}

export function escapeHTML(value) {
  const entities = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
  return String(value).replace(/[&<>"']/g, (character) => entities[character]);
}
