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
