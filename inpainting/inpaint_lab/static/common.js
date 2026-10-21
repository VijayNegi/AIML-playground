// Shared helpers for the local review and batch pages (plain ES module, no build step).
export const token = document.querySelector('meta[name="csrf-token"]').content;

export async function api(path, data) {
  const options = data === undefined ? {} : {
    method: 'POST',
    headers: {'Content-Type': 'application/json', 'X-CSRF-Token': token},
    body: JSON.stringify(data),
  };
  const response = await fetch(path, options);
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || 'Request failed');
  return result;
}

export function message(text, error = false) {
  const node = document.getElementById('message');
  node.textContent = text;
  node.className = error ? 'error' : '';
}

// Build DOM safely: strings become text nodes, never HTML.
export function element(tag, attributes = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attributes)) {
    if (value === undefined || value === null || value === false) continue;
    if (key === 'text') node.textContent = value;
    else node.setAttribute(key, value === true ? '' : value);
  }
  node.append(...children.filter(child => child !== null && child !== undefined && child !== ''));
  return node;
}

export function duration(seconds) {
  if (seconds === null || seconds === undefined) return 'unknown';
  const value = Math.ceil(seconds);
  return `${Math.floor(value / 3600)}h ${Math.floor(value % 3600 / 60)}m ${value % 60}s`;
}

export function estimateText(estimate) {
  if (!estimate) return 'Estimate unavailable.';
  const finish = estimate.seconds === null ? '' : `\nApproximate local finish: ${new Date(Date.now() + estimate.seconds * 1000).toLocaleString()}`;
  return `Estimated duration: ${duration(estimate.seconds)}${finish}\nKnown components: ${duration(estimate.known_seconds)}; scaled observed range: ${duration(estimate.observed_low_seconds)}–${duration(estimate.observed_high_seconds)}\n${estimate.confidence}; minimum matched samples per component: ${estimate.minimum_samples}\n${estimate.warnings.join('\n')}`;
}

export function badge(status) {
  return element('span', {class: `badge badge-${status.replaceAll(' ', '-')}`, text: status});
}

// Mark the current section in the shared header.
for (const link of document.querySelectorAll('.app-header nav a')) {
  if (new URL(link.href).pathname === location.pathname) link.setAttribute('aria-current', 'page');
}
