// New run: a one-job batch through the shared worker, optionally prefilled from an earlier run.
import {api, message, element, badge, duration, estimateText} from '/static/common.js';

const $ = id => document.getElementById(id);
const form = $('run-form');
const NUMBERS = ['gap', 'feather', 'steps', 'guidance_scale', 'seed'];
const DONE = ['completed', 'failed', 'interrupted', 'paused'];
let catalog = null;
let activeBatch = null;
let timer = null;

function preview(side) {
  const select = form.elements[side];
  const image = $(`${side}-preview`);
  image.hidden = !select.value;
  if (select.value) image.src = '/api/batches/sample?path=' + encodeURIComponent(select.value);
}

function apply(settings) {
  for (const key of ['model', 'prompt', 'negative_prompt', ...NUMBERS]) {
    if (settings[key] !== undefined && settings[key] !== null) form.elements[key].value = settings[key];
  }
  if ('height' in settings) form.elements.height.value = settings.height ?? '';
  if ('save_intermediate' in settings) form.elements.save_intermediate.checked = Boolean(settings.save_intermediate);
  if (Array.isArray(settings.images)) {
    ['left', 'right'].forEach((side, index) => {
      form.elements[side].value = settings.images[index] && catalog.images.includes(settings.images[index]) ? settings.images[index] : '';
      preview(side);
    });
  }
}

function definition() {
  const baseline = {model: form.elements.model.value, prompt: form.elements.prompt.value,
    negative_prompt: form.elements.negative_prompt.value, save_intermediate: form.elements.save_intermediate.checked,
    height: form.elements.height.value ? Number(form.elements.height.value) : null};
  for (const key of NUMBERS) baseline[key] = Number(form.elements[key].value);
  return {name: `Single run ${new Date().toLocaleString()}`, kind: 'single', mode: 'one_at_a_time', sweeps: {},
    images: [form.elements.left.value, form.elements.right.value], baseline};
}

function renderStatus(batch, estimate) {
  const job = batch.jobs[0];
  const attempt = job.attempts.at(-1);
  const folder = attempt?.run_dir.split(/[\\/]/).pop();
  const running = !DONE.includes(batch.status);
  $('run-button').disabled = running;
  $('status').replaceChildren(
    element('p', {}, badge(job.status === 'pending' && batch.status === 'running' ? 'queued' : job.status)),
    element('p', {text: `Elapsed: ${duration(batch.elapsed_seconds + (batch.running_elapsed_seconds || 0))}`}),
    running ? element('p', {class: 'summary', text: estimateText(estimate)}) : null,
    batch.status === 'paused' && job.status === 'pending' ? element('p', {},
      'Saved but not started (another batch may be running). ',
      element('a', {href: `/batches?batch=${encodeURIComponent(batch.id)}`, text: 'Start it from Batches'}), '.') : null,
    attempt?.error ? element('p', {class: 'error-box', text: `Error: ${attempt.error}`}) : null,
    attempt?.warnings && Object.keys(attempt.warnings).length ? element('p', {class: 'muted', text: `Warnings: ${JSON.stringify(attempt.warnings)}`}) : null,
    folder && job.status !== 'pending' ? element('p', {class: 'actions'},
      element('a', {class: 'button primary', href: `/?run=${encodeURIComponent(folder)}`, text: 'Open run details'}),
      job.status === 'completed' ? element('a', {class: 'button', href: `/review?run=${encodeURIComponent(folder)}`, text: 'Review'}) : null) : null);
  return running;
}

async function poll() {
  clearTimeout(timer);
  try {
    const data = await api(`/api/batches?batch=${encodeURIComponent(activeBatch)}`);
    const batch = data.batches.find(item => item.id === activeBatch);
    if (batch && renderStatus(batch, data.estimate)) timer = setTimeout(poll, 2000);
  } catch (error) {
    message(error.message, true);
    timer = setTimeout(poll, 5000);
  }
}

form.addEventListener('submit', async event => {
  event.preventDefault();
  $('run-button').disabled = true;
  try {
    const settings = definition();
    message('Validating…');
    const checked = await api('/api/batches/preview', settings);
    const batch = await api('/api/batches/create', {definition: settings, expected: checked.expected});
    activeBatch = batch.id;
    history.replaceState(null, '', `/run?batch=${encodeURIComponent(batch.id)}`);
    try {
      await api('/api/batches/action', {id: batch.id, action: 'resume'});
      message('Run started.');
    } catch (error) {
      message(`Run saved but not started: ${error.message}`, true);
    }
    await poll();
  } catch (error) {
    message(error.message, true);
    $('run-button').disabled = false;
  }
});

for (const side of ['left', 'right']) form.elements[side].addEventListener('change', () => preview(side));
$('random-seed').addEventListener('click', () => { form.elements.seed.value = Math.floor(Math.random() * 2 ** 32); });
$('reset').addEventListener('click', () => { apply(catalog.defaults); form.elements.height.value = ''; form.elements.save_intermediate.checked = false; });

async function initialize() {
  catalog = await api('/api/batches/catalog');
  for (const side of ['left', 'right']) {
    const select = form.elements[side];
    select.append(element('option', {value: '', text: 'Choose an image…'}),
      ...catalog.images.map(name => element('option', {value: name, text: name})));
  }
  for (const name of catalog.models) form.elements.model.append(element('option', {value: name, text: name}));
  apply({...catalog.defaults, images: catalog.images.slice(0, 2)});

  const query = new URLSearchParams(location.search);
  if (query.get('from')) {
    const run = await api(`/api/runs/detail?run=${encodeURIComponent(query.get('from'))}`);
    apply(run.rerun);
    $('from-note').replaceChildren(
      element('p', {}, 'Settings loaded from ', element('a', {href: `/?run=${encodeURIComponent(run.name)}`, text: run.name}),
        '. Edit anything, then press Run.'),
      ...run.rerun_warnings.map(text => element('p', {class: 'warning', text})));
    $('from-note').hidden = false;
  }
  if (query.get('batch')) {
    activeBatch = query.get('batch');
    await poll();
  }
}

initialize().catch(error => message(error.message, true));
