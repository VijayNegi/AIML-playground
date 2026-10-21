// Run history: list every run, show images, settings, metrics, and offer rerun.
import {api, message, element, badge} from '/static/common.js';

const $ = id => document.getElementById(id);
const IMAGE_LABELS = {
  'output.png': 'Output', 'output_comparison.png': 'Before / after', 'collage_canvas.png': 'Input canvas',
  'collage_mask.png': 'Blend mask', 'debug_crop_image.png': 'Crop sent to model', 'debug_crop_mask.png': 'Crop mask',
  'debug_crop_result.png': 'Model output (crop)', 'quality_context.png': 'MANIQA context (legacy)',
};
const SETTING_LABELS = {
  images: 'Input images', prompt: 'Prompt', negative_prompt: 'Negative prompt', model: 'Model', model_id: 'Model ID',
  gap: 'Gap (px)', feather: 'Feather (px)', steps: 'Steps', guidance_scale: 'Guidance scale', seed: 'Seed',
  strength: 'Strength', height: 'Height', max_crop_size: 'Max crop size', save_intermediate: 'Save canvas & mask',
};
let runs = [];
let selected = new URLSearchParams(location.search).get('run');

const finite = value => typeof value === 'number' && Number.isFinite(value);
const fixed = (value, digits = 3) => finite(value) ? value.toFixed(digits) : '—';
const seconds = value => finite(value) ? `${value.toFixed(1)} s` : '—';
const fileUrl = (run, name) => `/api/runs/file?run=${encodeURIComponent(run)}&name=${encodeURIComponent(name)}`;
const show = value => value === null || value === undefined ? '—'
  : typeof value === 'string' ? value : JSON.stringify(value, null, Array.isArray(value) || typeof value !== 'object' ? 0 : 2);

function renderList() {
  const filter = $('filter').value;
  const visible = runs.filter(run => !filter || run.status === filter);
  $('run-count').textContent = `${visible.length} of ${runs.length} runs`;
  $('run-list').replaceChildren(...visible.map(run => {
    const p = run.params;
    const button = element('button', {type: 'button', class: 'run-item', 'aria-current': run.name === selected ? 'true' : null},
      element('span', {class: 'run-item-head'}, element('strong', {text: new Date(run.created).toLocaleString()}), badge(run.status)),
      element('span', {class: 'muted', text: `gap ${show(p.gap)} · feather ${show(p.feather)} · steps ${show(p.steps)} · guidance ${show(p.guidance_scale)} · seed ${show(p.seed)}`}),
      element('span', {class: 'muted', text: `MANIQA ${fixed(run.maniqa)}${run.batch ? ' · ' + (run.batch.batch_kind === 'single' ? 'UI run' : run.batch.batch_name) : ''}`}));
    button.addEventListener('click', () => select(run.name));
    return element('li', {}, button);
  }));
  if (!visible.length) $('run-list').replaceChildren(element('li', {class: 'muted', text: 'No runs match this filter.'}));
}

function keyValueTable(entries, labels = {}) {
  return element('table', {class: 'kv'}, element('tbody', {}, ...entries.map(([key, value]) =>
    element('tr', {}, element('th', {scope: 'row', title: key, text: labels[key] || key}), element('td', {}, element('code', {text: show(value)}))))));
}

function stat(label, value, hint) {
  return element('div', {class: 'stat'}, element('span', {class: 'stat-value', text: value}),
    element('span', {class: 'stat-label', text: label}), hint ? element('small', {class: 'muted', text: hint}) : null);
}

function renderDetail(run) {
  const metrics = run.metrics || {};
  const scalars = Object.entries(metrics).filter(([, value]) => value === null || typeof value !== 'object');
  const structured = Object.entries(metrics).filter(([, value]) => value !== null && typeof value === 'object');
  const batch = run.batch;
  const params = Object.entries(run.params);
  params.sort(([a], [b]) => (a in SETTING_LABELS ? 0 : 1) - (b in SETTING_LABELS ? 0 : 1));
  $('run-detail').replaceChildren(
    element('div', {class: 'detail-head'},
      element('div', {},
        element('h2', {}, element('span', {text: new Date(run.created).toLocaleString()}), ' ', badge(run.status)),
        element('p', {class: 'muted'}, element('code', {text: run.name}),
          batch ? element('span', {}, ' · ', batch.batch_kind === 'single' ? 'Started from New run'
            : element('a', {href: `/batches?batch=${encodeURIComponent(batch.batch_id)}`, text: `Batch: ${batch.batch_name}`})) : null)),
      element('div', {class: 'actions'},
        element('a', {class: 'button primary', href: `/run?from=${encodeURIComponent(run.name)}`, text: 'Rerun with these settings'}),
        run.status === 'pending review' ? element('a', {class: 'button', href: `/review?run=${encodeURIComponent(run.name)}`, text: 'Review'}) : null)),
    batch?.error ? element('p', {class: 'error-box', text: `Error: ${batch.error}`}) : null,
    element('div', {class: 'stats'},
      stat('MANIQA (median tile)', fixed(metrics.quality_maniqa), 'higher is better'),
      stat('Lowest tile', fixed(metrics.quality_maniqa_min)),
      stat('Original pixels changed', finite(metrics.quality_preservation_changed_fraction)
        ? `${(metrics.quality_preservation_changed_fraction * 100).toFixed(2)}%` : '—', 'lower is better'),
      stat('Inference', seconds(metrics.inference_time)),
      stat('Total run time', seconds(metrics.wall_time))),
    run.review ? element('section', {class: 'card'}, element('h3', {text: 'Human review'}),
      element('p', {text: `Seam ${run.review.seam_quality}/5 · Coherence ${run.review.scene_coherence}/5 · Preservation ${run.review.preservation}/5`}),
      run.review.notes ? element('p', {text: run.review.notes}) : null) : null,
    element('h3', {text: 'Images'}),
    run.images.length ? element('div', {class: 'gallery'}, ...run.images.map(name =>
      element('figure', {}, element('a', {href: fileUrl(run.name, name), target: '_blank', rel: 'noopener'},
        element('img', {src: fileUrl(run.name, name), alt: IMAGE_LABELS[name] || name, loading: 'lazy'})),
        element('figcaption', {}, element('strong', {text: IMAGE_LABELS[name] || name}), ' ', element('small', {class: 'muted', text: name})))))
      : element('p', {class: 'muted', text: 'No images saved for this run.'}),
    run.step_images ? element('p', {class: 'muted', text: `${run.step_images} denoising-step images saved in intermediate_steps/.`}) : null,
    !run.images.includes('collage_canvas.png') ? element('p', {class: 'muted', text: 'Tip: enable “Save canvas & mask” under Advanced when running to keep the input canvas and blend mask.'}) : null,
    element('div', {class: 'two-col'},
      element('section', {}, element('h3', {text: 'Settings'}), keyValueTable(params, SETTING_LABELS)),
      element('section', {}, element('h3', {text: 'Metrics'}),
        run.metrics ? keyValueTable(scalars) : element('p', {class: 'muted', text: 'No metrics recorded (run failed or still running).'}),
        ...structured.map(([key, value]) => element('details', {}, element('summary', {text: key}), element('pre', {text: JSON.stringify(value, null, 2)}))))));
}

async function select(name) {
  selected = name;
  history.replaceState(null, '', `/?run=${encodeURIComponent(name)}`);
  renderList();
  try { renderDetail(await api(`/api/runs/detail?run=${encodeURIComponent(name)}`)); }
  catch (error) { message(error.message, true); }
}

async function refresh() {
  const data = await api('/api/runs');
  runs = data.runs;
  renderList();
}

$('filter').addEventListener('change', renderList);
setInterval(async () => {
  // Keep running jobs current without disturbing the open detail view otherwise.
  if (!runs.some(run => run.status === 'running')) return;
  const wasRunning = runs.find(run => run.name === selected)?.status === 'running';
  await refresh().catch(() => {});
  if (wasRunning && runs.find(run => run.name === selected)?.status !== 'running') select(selected);
}, 5000);

refresh().then(() => {
  if (selected && runs.some(run => run.name === selected)) select(selected);
  else if (runs.length) select(runs[0].name);
  else $('run-detail').replaceChildren(element('p', {class: 'muted'}, 'No runs yet. ', element('a', {href: '/run', text: 'Start a new run'}), '.'));
}).catch(error => message(error.message, true));
