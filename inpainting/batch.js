const form = document.getElementById('builder');
const token = document.querySelector('meta[name="csrf-token"]').content;
const specifications = [
  ['gap', 'Gap (px)', 1, 224, 1], ['feather', 'Feather (px)', 0, 1024, 1],
  ['steps', 'Steps', 2, 1000, 1], ['guidance_scale', 'Guidance', 0, 50, 0.1], ['seed', 'Seed', 0, 4294967295, 1]
];
let preview = null;
let selectedId = '';
let history = [];
let polling = false;

function message(text, error = false) {
  const node = document.getElementById('message');
  node.textContent = text;
  node.className = error ? 'error' : '';
}
async function api(path, data) {
  const response = await fetch(path, data === undefined ? {} : {method: 'POST', headers: {'Content-Type': 'application/json', 'X-CSRF-Token': token}, body: JSON.stringify(data)});
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || 'Request failed');
  return result;
}
function duration(seconds) {
  if (seconds === null || seconds === undefined) return 'unknown';
  const value = Math.ceil(seconds);
  return `${Math.floor(value / 3600)}h ${Math.floor(value % 3600 / 60)}m ${value % 60}s`;
}
function estimateText(estimate) {
  if (!estimate) return 'Estimate unavailable.';
  const finish = estimate.seconds === null ? '' : `\nApproximate local finish: ${new Date(Date.now() + estimate.seconds * 1000).toLocaleString()}`;
  return `Estimated duration: ${duration(estimate.seconds)}${finish}\nKnown components: ${duration(estimate.known_seconds)}; scaled observed range: ${duration(estimate.observed_low_seconds)}–${duration(estimate.observed_high_seconds)}\n${estimate.confidence}; minimum matched samples per component: ${estimate.minimum_samples}\n${estimate.warnings.join('\n')}`;
}
function invalidate() {
  preview = null;
  document.getElementById('save').disabled = true;
  document.getElementById('start').disabled = true;
  document.getElementById('preview').hidden = true;
}
function definition() {
  const baseline = {model: form.elements.model.value, prompt: form.elements.prompt.value,
    negative_prompt: form.elements.negative_prompt.value, height: form.elements.height.value ? Number(form.elements.height.value) : null};
  const sweeps = {};
  for (const [name] of specifications) {
    baseline[name] = Number(form.elements[name].value);
    if (form.elements[`${name}_mode`].value === 'sweep') {
      const text = form.elements[`${name}_values`].value;
      if (!text.trim() || text.split(',').some(part => !part.trim())) throw new Error(`${name}: enter comma-separated numbers without empty values`);
      sweeps[name] = text.split(',').map(value => Number(value.trim()));
      if (sweeps[name].some(value => !Number.isFinite(value))) throw new Error(`${name}: invalid number`);
    }
  }
  return {name: form.elements.name.value, images: [form.elements.left.value, form.elements.right.value], baseline, sweeps,
    mode: form.elements.mode.value, allow_legacy_timings: form.elements.legacy.checked};
}
function row(values) {
  const node = document.createElement('tr');
  for (const value of values) {
    const cell = document.createElement('td');
    cell.textContent = value;
    node.append(cell);
  }
  return node;
}
form.addEventListener('input', invalidate);
form.addEventListener('change', event => {
  invalidate();
  if (event.target.name?.endsWith('_mode')) {
    const name = event.target.name.replace('_mode', '');
    form.elements[`${name}_values`].disabled = event.target.value !== 'sweep';
  }
  for (const side of ['left', 'right']) if (event.target.name === side) {
    const image = document.getElementById(`${side}-preview`);
    image.hidden = !event.target.value;
    image.src = '/api/batches/sample?path=' + encodeURIComponent(event.target.value);
  }
});
form.addEventListener('submit', async event => {
  event.preventDefault();
  try {
    message('Validating jobs and reading timing history…');
    const settings = definition();
    const result = await api('/api/batches/preview', settings);
    if (JSON.stringify(settings) !== JSON.stringify(definition())) throw new Error('Settings changed during preview. Preview again.');
    preview = result;
    document.getElementById('preview-estimate').textContent = `${result.jobs.length} unique jobs\n` + estimateText(result.estimate);
    document.getElementById('preview-jobs').replaceChildren(...result.jobs.map((job, index) => row([index + 1, job.sweep,
      job.params.gap, job.params.feather, job.params.steps, job.params.guidance_scale, job.params.seed,
      `${job.geometry.adjusted_gap}px / ${job.geometry.tiles}`])));
    document.getElementById('preview').hidden = false;
    document.getElementById('save').disabled = false;
    document.getElementById('start').disabled = false;
    message('Preview ready. No inference has started.');
  } catch (error) { invalidate(); message(error.message, true); }
});
async function create(start) {
  if (!preview) return;
  document.getElementById('save').disabled = true;
  document.getElementById('start').disabled = true;
  try {
    const batch = await api('/api/batches/create', {definition: definition(), expected: preview.expected});
    selectedId = batch.id;
    invalidate();
    if (start) await api('/api/batches/action', {id: selectedId, action: 'resume'});
    message(start ? 'Batch started. You can close this tab; keep the server running.' : 'Batch saved paused.');
    await refresh();
  } catch (error) { message(error.message, true); await refresh(); }
}
document.getElementById('save').addEventListener('click', () => create(false));
document.getElementById('start').addEventListener('click', () => create(true));
for (const button of document.querySelectorAll('[data-action]')) button.addEventListener('click', async () => {
  try { await api('/api/batches/action', {id: selectedId, action: button.dataset.action}); message('Action accepted.'); await refresh(); }
  catch (error) { message(error.message, true); }
});
function showJobs(batch) {
  document.getElementById('batch-jobs').replaceChildren(...batch.jobs.map(job => {
    const p = job.params;
    const node = row([job.id, `gap ${p.gap}; feather ${p.feather}; steps ${p.steps}; guidance ${p.guidance_scale}; seed ${p.seed}`, job.status, '']);
    const details = node.lastElementChild;
    for (const [index, attempt] of job.attempts.entries()) {
      const entry = document.createElement('div');
      entry.textContent = `Attempt ${index + 1}: ${attempt.status} ${attempt.error || ''}`;
      details.append(entry);
      if (attempt.status === 'completed') {
        const link = document.createElement('a');
        link.href = `/api/batches/output?batch=${encodeURIComponent(batch.id)}&job=${encodeURIComponent(job.id)}&attempt=${index}`;
        link.textContent = 'Open output'; link.target = '_blank'; link.rel = 'noopener';
        details.append(link);
        const review = document.createElement('a');
        review.href = '/?run=' + encodeURIComponent(attempt.run_dir.split(/[\\/]/).pop());
        review.textContent = ' · Review (if pending)'; details.append(review);
      }
      if (attempt.warnings) {
        const warning = document.createElement('small');
        warning.textContent = JSON.stringify(attempt.warnings);
        details.append(warning);
      }
    }
    return node;
  }));
}
async function refresh() {
  if (polling) return;
  polling = true;
  try {
    let data = await api('/api/batches?batch=' + encodeURIComponent(selectedId));
    history = data.batches;
    if (!selectedId && history.length) {
      selectedId = history.at(-1).id;
      data = await api('/api/batches?batch=' + encodeURIComponent(selectedId));
      history = data.batches;
    }
    const list = document.getElementById('history');
    list.replaceChildren(...[...history].reverse().map(batch => {
      const button = document.createElement('button');
      button.className = 'history-item';
      const done = batch.jobs.filter(job => job.status === 'completed').length;
      button.textContent = `${batch.name} — ${batch.status} — ${done}/${batch.jobs.length} complete — ${batch.created_at}`;
      button.setAttribute('aria-pressed', String(batch.id === selectedId));
      button.addEventListener('click', () => { selectedId = batch.id; refresh(); });
      return button;
    }));
    if (!history.length) list.textContent = 'No saved batches yet.';
    const batch = history.find(item => item.id === selectedId);
    document.getElementById('batch-detail').hidden = !batch;
    if (batch) {
      const completed = batch.jobs.filter(job => job.status === 'completed').length;
      const failed = batch.jobs.filter(job => ['failed', 'interrupted'].includes(job.status)).length;
      const current = batch.jobs.find(job => job.status === 'running');
      document.getElementById('batch-title').textContent = batch.name;
      document.getElementById('batch-summary').textContent = `Status: ${batch.status}; completed ${completed}/${batch.jobs.length}; failed/interrupted ${failed}\nMeasured execution time (pauses excluded): ${duration(batch.elapsed_seconds + batch.running_elapsed_seconds)}\n${current ? 'Current configuration: ' + current.id : ''}\n${estimateText(data.estimate)}`;
      document.getElementById('progress').max = batch.jobs.length;
      document.getElementById('progress').value = completed + failed;
      document.getElementById('definition').textContent = JSON.stringify(batch.definition, null, 2);
      showJobs(batch);
      document.querySelector('[data-action="pause"]').disabled = batch.status !== 'running';
      document.querySelector('[data-action="resume"]').disabled = !batch.jobs.some(job => job.status === 'pending') || ['running', 'pause_requested'].includes(batch.status);
      document.querySelector('[data-action="retry"]').disabled = !failed || ['running', 'pause_requested'].includes(batch.status);
    }
  } catch (error) { message(error.message, true); }
  finally { polling = false; }
}
document.getElementById('download').addEventListener('click', () => {
  const batch = history.find(item => item.id === selectedId);
  const url = URL.createObjectURL(new Blob([JSON.stringify(batch, null, 2)], {type: 'application/json'}));
  const link = document.createElement('a'); link.href = url; link.download = `${batch.id}.json`; link.click();
  URL.revokeObjectURL(url);
});
async function initialize() {
  const catalog = await api('/api/batches/catalog');
  for (const side of ['left', 'right']) {
    const select = form.elements[side];
    for (const filename of catalog.images) select.add(new Option(filename, filename));
    select.selectedIndex = Math.min(side === 'right' ? 1 : 0, catalog.images.length - 1);
    select.dispatchEvent(new Event('change', {bubbles: true}));
  }
  for (const name of catalog.models) form.elements.model.add(new Option(name, name));
  for (const key of ['model', 'prompt', 'negative_prompt']) form.elements[key].value = catalog.defaults[key];
  for (const [name, label, minimum, maximum, step] of specifications) {
    const node = row([label, '', '', '']);
    const input = document.createElement('input'); Object.assign(input, {type: 'number', name, min: minimum, max: maximum, step, value: catalog.defaults[name], required: true});
    input.setAttribute('aria-label', `${label} baseline`); node.children[1].append(input);
    const mode = document.createElement('select'); mode.name = `${name}_mode`; mode.add(new Option('Fixed', 'fixed')); mode.add(new Option('Sweep', 'sweep'));
    mode.setAttribute('aria-label', `${label} mode`); node.children[2].append(mode);
    const values = document.createElement('input'); Object.assign(values, {name: `${name}_values`, type: 'text', disabled: true, placeholder: name === 'guidance_scale' ? '3, 5, 7.5' : 'Comma-separated values'});
    values.setAttribute('aria-label', `${label} sweep values`); node.children[3].append(values);
    document.getElementById('parameters').append(node);
  }
  await refresh();
  setInterval(refresh, 3000);
}
initialize().catch(error => message(error.message, true));
