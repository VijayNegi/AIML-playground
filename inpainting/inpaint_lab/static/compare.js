// Compare 2–3 runs side by side; mark best/worst for metrics whose direction is known.
import {api, message, element, badge} from '/static/common.js';

const $ = id => document.getElementById(id);
const LETTERS = 'ABC';
// Always differ between runs, so they would bury real setting changes.
const BOOKKEEPING = ['steps_dir', 'batch_id', 'configuration_id', 'sweep'];
const names = [...new Set(new URLSearchParams(location.search).getAll('run'))].slice(0, 3);
const finite = value => typeof value === 'number' && Number.isFinite(value);
const fileUrl = (run, name) => `/api/runs/file?run=${encodeURIComponent(run)}&name=${encodeURIComponent(name)}`;
const show = value => value === null || value === undefined ? '—' : finite(value) ? String(Number(value.toPrecision(4)))
  : typeof value === 'string' ? value : JSON.stringify(value);
const label = key => key.replace(/^quality_/, '').replaceAll('_', ' ').replace('delta e76', 'color mismatch (ΔE)')
  .replace('gradient excess', 'extra edge strength');

// Which way is better; null means the value has no quality direction (raw gradients, sizes, settings).
export function direction(key) {
  if (key === 'quality_maniqa' || key === 'quality_maniqa_min') return 'higher';
  if (/(_delta_e76|_gradient_excess|_time)$|^quality_preservation_(mae|changed_fraction)$/.test(key)) return 'lower';
  return null;
}

const SECTIONS = [
  ['Overall quality', key => /^quality_(maniqa|maniqa_min|preservation_mae|preservation_changed_fraction)$/.test(key)],
  ['Gap seams', key => key.startsWith('quality_gap_') && direction(key)],
  ['Other seams (editable area, model mask, pasted crop)', key => /^quality_(editable|mask|patch)_/.test(key) && direction(key)],
  ['Timing', key => key.endsWith('_time')],
];

// One class per run: 'best' / 'worst' when values differ and the direction is known.
export function marks(values, better, key = '') {
  // Negative extra edge strength = smoother than surroundings, not "better"; only a visible edge (> 0) is worse.
  const score = key.endsWith('_gradient_excess') ? values.map(value => finite(value) ? Math.max(0, value) : value) : values;
  const present = score.filter(finite);
  if (!better || present.length < 2 || new Set(present).size < 2) return values.map(() => '');
  const best = better === 'higher' ? Math.max(...present) : Math.min(...present);
  const worst = better === 'higher' ? Math.min(...present) : Math.max(...present);
  return score.map(value => value === best ? 'best' : value === worst ? 'worst' : '');
}

const differs = values => new Set(values.map(value => JSON.stringify(value ?? null))).size > 1;

function row(key, values, classes, arrow) {
  return element('tr', {class: !classes.some(Boolean) && differs(values) ? 'differs' : null},
    element('th', {scope: 'row', title: key}, label(key), arrow ? element('span', {class: 'arrow', text: arrow === 'higher' ? ' ↑' : ' ↓'}) : null),
    ...values.map((value, index) => element('td', {class: classes[index] || null}, element('code', {text: show(value)}))));
}

function table(runs, caption, rows) {
  return element('table', {class: 'compare-table'},
    element('caption', {text: caption}),
    element('thead', {}, element('tr', {}, element('th', {text: ''}), ...runs.map((run, index) => element('th', {scope: 'col', text: `Run ${LETTERS[index]}`})))),
    element('tbody', {}, ...rows));
}

function render(runs) {
  const metrics = runs.map(run => run.metrics || {});
  const numeric = [...new Set(metrics.flatMap(Object.keys))].filter(key => metrics.some(m => finite(m[key])));
  const wins = runs.map(() => 0);
  const used = new Set();
  const sections = SECTIONS.map(([caption, test]) => {
    const keys = numeric.filter(key => test(key) && !used.has(key));
    keys.forEach(key => used.add(key));
    const rows = keys.map(key => {
      const values = metrics.map(m => m[key]);
      const classes = marks(values, direction(key), key);
      if (caption !== 'Timing') classes.forEach((mark, index) => { if (mark === 'best') wins[index] += 1; });
      return row(key, values, classes, direction(key));
    });
    return rows.length ? table(runs, caption, rows) : null;
  });
  const qualityChecks = numeric.filter(key => used.has(key) && direction(key) && !key.endsWith('_time')).length;
  const otherRows = numeric.filter(key => !used.has(key)).map(key => row(key, metrics.map(m => m[key]), [], null));

  const params = runs.map(run => run.params);
  const keys = [...new Set(params.flatMap(Object.keys))].filter(key => key !== 'output');
  const changed = keys.filter(key => !BOOKKEEPING.includes(key) && differs(params.map(p => p[key])));
  const same = keys.filter(key => !changed.includes(key));

  $('compare').replaceChildren(
    element('div', {class: 'compare-grid', style: `--columns:${runs.length}`}, ...runs.map((run, index) => element('section', {class: 'compare-column'},
      element('div', {class: 'compare-head'},
        element('span', {class: 'letter', text: LETTERS[index]}),
        element('div', {},
          element('a', {href: `/?run=${encodeURIComponent(run.name)}`, text: new Date(run.created).toLocaleString()}), ' ', badge(run.status),
          element('p', {class: 'muted', text: `Best on ${wins[index]} of ${qualityChecks} quality checks`}))),
      element('figure', {class: 'hero'}, element('a', {target: '_blank', rel: 'noopener'}, element('img', {alt: `Run ${LETTERS[index]}`})),
        element('figcaption', {}))))),
    element('div', {class: 'compare-tables'},
      changed.length ? table(runs, 'Settings that differ', changed.map(key => row(key, params.map(p => p[key]), [], null)))
        : element('p', {class: 'muted', text: 'All settings are identical.'}),
      ...sections,
      same.length ? element('details', {}, element('summary', {text: `${same.length} identical settings and run identifiers`}),
        table(runs, 'Identical settings and run identifiers', same.map(key => row(key, params.map(p => p[key]), [], null)))) : null,
      otherRows.length ? element('details', {}, element('summary', {text: `${otherRows.length} other metrics (no better/worse)`}),
        table(runs, 'Other metrics', otherRows)) : null));
  showImages(runs, 'output');
  for (const button of document.querySelectorAll('#view button')) button.onclick = () => {
    for (const other of document.querySelectorAll('#view button')) other.setAttribute('aria-pressed', String(other === button));
    showImages(runs, button.dataset.image);
  };
}

function showImages(runs, choice) {
  document.querySelectorAll('.compare-column').forEach((column, index) => {
    const run = runs[index];
    const name = choice === 'output' ? run.output : choice;
    const available = name && run.images.includes(name);
    column.querySelector('figure a').href = available ? fileUrl(run.name, name) : '#';
    column.querySelector('img').hidden = !available;
    if (available) column.querySelector('img').src = fileUrl(run.name, name);
    column.querySelector('figcaption').textContent = available ? name : `${name || 'Output'} not saved for this run`;
  });
}

if (names.length < 2) {
  $('compare').replaceChildren(element('p', {class: 'muted'}, 'Tick 2–3 runs on the ', element('a', {href: '/', text: 'Runs page'}), ', then press Compare.'));
} else {
  Promise.all(names.map(name => api(`/api/runs/detail?run=${encodeURIComponent(name)}`)))
    .then(render).catch(error => message(error.message, true));
}
