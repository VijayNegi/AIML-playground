// Review page: renders recorded scores and highlight regions supplied by /api/review/*.
// Measurement regions only: overlays do not locate defects or rate severity.
import {api, message, element} from '/static/common.js';

const SVG = 'http://www.w3.org/2000/svg';
const $ = id => document.getElementById(id);
const image = $('comparison-image');
const overlay = $('metric-overlay');
const status = $('highlight-status');
const state = {runs: [], ratings: {}, run: null, regions: {}, selected: null, preview: null, showInput: false};

const finite = value => typeof value === 'number' && Number.isFinite(value);
const number = (value, percent = false) => !finite(value) ? 'Not available' : percent ? `${(value * 100).toFixed(2)}%` : value.toFixed(3);
const significant = value => String(Number(value.toPrecision(3)));

function validBox(box, size) {
  if (!Array.isArray(box) || box.length !== 4 || !box.every(Number.isInteger)) return null;
  const [width, height] = size;
  return 0 <= box[0] && box[0] < box[2] && box[2] <= width && 0 <= box[1] && box[1] < box[3] && box[3] <= height ? box : null;
}

function highlightButton(metric, label, attributes = {}) {
  if (!state.regions[metric]) return element('small', {text: 'Area unavailable: this run has no recorded geometry for this metric.'});
  const {className, ...rest} = attributes;
  return element('button', {type: 'button', class: `metric-highlight${className ? ' ' + className : ''}`,
    'aria-pressed': 'false', 'data-metric': metric, text: label, ...rest});
}

// --- Range rulers and tile rail around the image ---
function ruler(metrics, group, key, label) {
  const box = validBox(metrics[key], metrics.canvas_size);
  if (!box) return null;
  const width = metrics.canvas_size[0];
  const controls = [];
  for (const [edge, side] of [['left', 'L'], ['right', 'R']]) {
    for (const [suffix, short] of [['delta_e76', 'color'], ['gradient_excess', 'extra edge']]) {
      const name = `quality_${group}_${edge}_${suffix}`;
      if (state.regions[name] && finite(metrics[name])) {
        controls.push(' ', highlightButton(name, `${side} ${short} ${significant(metrics[name])}`, {title: name}));
      }
    }
  }
  return element('div', {class: 'range-row'},
    element('div', {}, element('strong', {text: label}), ' ', element('small', {text: `x ${box[0]}–${box[2]}`}), ...controls),
    element('div', {class: 'range-track', 'aria-label': `${label} horizontal range`},
      element('span', {style: `left:${box[0] / width * 100}%;width:${(box[2] - box[0]) / width * 100}%`})));
}

function tileRail(metrics) {
  const tiles = Array.isArray(metrics.quality_maniqa_tiles) ? metrics.quality_maniqa_tiles : [];
  const height = metrics.canvas_size[1];
  const svg = document.createElementNS(SVG, 'svg');
  svg.setAttribute('class', 'tile-connectors');
  svg.setAttribute('viewBox', `0 0 100 ${height}`);
  svg.setAttribute('preserveAspectRatio', 'none');
  svg.setAttribute('aria-hidden', 'true');
  const buttons = [];
  tiles.forEach((tile, index) => {
    const regions = state.regions[`quality_maniqa_tile_${index}`];
    if (!regions || !finite(tile?.score)) return;
    const [, top, , bottom] = regions[0];
    const labelY = (index + 0.5) / tiles.length;
    const line = document.createElementNS(SVG, 'line');
    for (const [key, value] of Object.entries({x1: 0, y1: (top + bottom) / 2, x2: 22, y2: labelY * height})) line.setAttribute(key, value);
    svg.append(line);
    buttons.push(highlightButton(`quality_maniqa_tile_${index}`, `${index + 1}: ${tile.score.toFixed(3)}`, {
      className: 'tile-value', style: `top:${labelY * 100}%`, title: `Tile ${index + 1}; y ${top}–${bottom}; higher is better`}));
  });
  return buttons.length ? [svg, ...buttons] : [element('p', {text: 'No tiled MANIQA scores.'})];
}

function renderSurroundings(metrics) {
  const size = metrics?.canvas_size;
  if (!Array.isArray(size) || size.length !== 2 || !size.every(value => Number.isInteger(value) && value > 0)) {
    $('top-ranges').replaceChildren();
    $('bottom-ranges').replaceChildren();
    $('tile-rail').replaceChildren(element('p', {text: 'No recorded spatial metrics.'}));
    return;
  }
  $('top-ranges').replaceChildren(...[ruler(metrics, 'gap', 'quality_gap_bbox', 'Gap'),
    ruler(metrics, 'mask', 'quality_effective_mask_bbox', 'Model mask')].filter(Boolean));
  $('bottom-ranges').replaceChildren(...[ruler(metrics, 'editable', 'quality_editable_bbox', 'Outer feather'),
    ruler(metrics, 'patch', 'crop_bbox', 'Pasted crop')].filter(Boolean));
  $('tile-rail').replaceChildren(...tileRail(metrics));
}

// --- Automatic assessment panel ---
function boundaryTable(metrics, boundaries) {
  const columns = [['delta_e76', 'Color mismatch'], ['gradient_excess', 'Extra edge strength'],
    ['gradient', 'Edge strength'], ['nearby_gradient', 'Nearby edges']];
  const rows = [];
  for (const [boundary, label] of boundaries) {
    if (!columns.some(([suffix]) => `quality_${boundary}_${suffix}` in metrics)) continue;
    rows.push(element('tr', {}, element('th', {scope: 'row', text: label}), ...columns.map(([suffix, title]) => {
      const name = `quality_${boundary}_${suffix}`;
      const value = number(metrics[name]);
      let cell;
      if (value === 'Not available') cell = element('span', {title: 'Not recorded or not measurable for this run', text: 'Unavailable'});
      else if (!state.regions[name]) cell = element('span', {title: `${label}: ${title}; ${name}; measured area unavailable`, text: value});
      else cell = highlightButton(name, value, {'aria-label': `${label}: ${title}: ${value}`, title: name});
      return element('td', {}, cell);
    })));
  }
  if (!rows.length) return element('p', {text: 'No boundary scores recorded for this run.'});
  return element('div', {class: 'metric-table'}, element('table', {},
    element('thead', {}, element('tr', {}, element('th', {scope: 'col', text: 'Measured edge'}),
      ...columns.map(([, title]) => element('th', {scope: 'col', text: title})))),
    element('tbody', {}, ...rows)));
}

function maniqaHelp(metrics, score) {
  if (score !== 'Not available') return 'Higher suggests better visual quality. Usually around 0–1; not a percentage or a human 1–5 rating. Compare the same input pair.';
  const error = String(metrics.quality_maniqa_error || '');
  if (error) {
    const install = error.includes('pyiqa') && error.includes('No module named')
      ? ' Activate your virtual environment, install requirements.txt, then start a new collage run.' : '';
    return 'Scoring failed for this run. See the assessment warning below.' + install;
  }
  if (metrics.quality_error) return 'Quality assessment failed before a MANIQA score was recorded.';
  return 'No MANIQA score was recorded for this run. Older runs are not scored retroactively.';
}

function renderScores(payload) {
  const metrics = payload.metrics || {};
  const tiles = Array.isArray(metrics.quality_maniqa_tiles) ? metrics.quality_maniqa_tiles : [];
  const score = number(metrics.quality_maniqa);
  const scope = tiles.length
    ? 'Native-resolution tiles follow the gap down its full height. No overview resizing is used.'
    : 'This run has no tiled scores. Any recorded MANIQA score uses the older context-overview assessment.';
  const tileDetails = tiles.length ? [
    element('p', {}, element('strong', {text: `Lowest tile: ${number(metrics.quality_maniqa_min)}`}), ' ',
      highlightButton('quality_maniqa_min', 'Show lowest tile')),
    element('p', {text: `${tiles.length} overlapping 224×224 tiles, top to bottom. Summary is the median; the lowest tile is a possible weak spot, not an automatic failure.`}),
    element('p', {class: 'tile-scores'}, ...tiles.flatMap((tile, index) => [
      highlightButton(`quality_maniqa_tile_${index}`, `Tile ${index + 1}: ${number(tile?.score)}`), ' '])),
  ] : [];
  const warnings = ['quality_error', 'quality_maniqa_error'].filter(key => key in metrics).map(key => String(metrics[key])).join(' ');
  $('scores').replaceChildren(
    element('h2', {id: 'automatic-heading', text: 'Automatic assessment'}),
    element('p', {text: 'Priority: gap quality, then seams, then patch checks. Select a value to highlight its area. No calibrated pass/fail thresholds.'}),
    element('section', {class: 'quality-group gap-quality'},
      element('h3', {text: '1. Gap quality · most important'}),
      element('p', {}, element('strong', {title: 'quality_maniqa', text: `Visual quality · MANIQA: ${score}`}), ' ',
        highlightButton('quality_maniqa', 'Show area')),
      element('p', {text: `${maniqaHelp(metrics, score)} ${scope}`}),
      ...tileDetails),
    element('section', {class: 'quality-group'},
      element('h3', {text: '2. Seams · does the join look continuous?'}),
      boundaryTable(metrics, [['mask_left', 'Left edge of model mask'], ['mask_right', 'Right edge of model mask'],
        ['gap_left', 'Left edge of gap'], ['gap_right', 'Right edge of gap'],
        ['editable_left', 'Left outer feather edge'], ['editable_right', 'Right outer feather edge']]),
      element('p', {text: 'The gap, outer feather boundary, and model mask are different edges. SDXL thresholds feather gray values at 50%, so its model-mask edge falls inside the feather. Older runs may only have nominal outer feather bounds.'}),
      element('p', {text: 'Color mismatch: Lab distance, 0 upward; lower means closer colors. Extra edge strength: −1 to 1; positive means sharper than nearby edges. Edge strength and nearby edges: 0–1. Real shadows or object edges can legitimately score high.'})),
    element('section', {class: 'quality-group'},
      element('h3', {text: '3. Patch · unwanted changes and pasted-crop edges'}),
      element('p', {},
        element('strong', {text: `Original pixels changed: ${number(metrics.quality_preservation_changed_fraction, true)}`}), ' ',
        highlightButton('quality_preservation_changed_fraction', 'Show area'), element('br'),
        element('strong', {text: `Size of unwanted changes: ${number(metrics.quality_preservation_mae)}`}), ' ',
        highlightButton('quality_preservation_mae', 'Show area')),
      element('p', {text: 'Outside the gap and feather zone. Lower is better: 0% means unchanged; average change uses 0–255 color levels. Even invisible pixel changes count.'}),
      boundaryTable(metrics, [['patch_left', 'Left edge of pasted crop'], ['patch_right', 'Right edge of pasted crop'],
        ['patch_top', 'Top edge of pasted crop'], ['patch_bottom', 'Bottom edge of pasted crop']]),
      element('p', {text: 'Same seam scales above; checks where the full model-generated crop was pasted back. Canvas-border edges have no outside neighbor and are omitted.'})),
    payload.metrics_error ? element('p', {role: 'status', text: payload.metrics_error}) : null,
    warnings ? element('p', {role: 'status', text: `Assessment warning: ${warnings}`}) : null);
}

// --- Highlight overlay ---
function draw(button) {
  if (!image.naturalWidth) return;
  overlay.replaceChildren();
  overlay.setAttribute('viewBox', `0 0 ${image.naturalWidth} ${image.naturalHeight}`);
  for (const item of document.querySelectorAll('.metric-highlight')) item.setAttribute('aria-pressed', String(item === state.selected));
  if (state.showInput) {
    status.textContent = 'Input canvas shown. Switch to output to see metric highlights.';
    return;
  }
  if (!button) {
    status.textContent = 'Hover or focus a metric button to preview its measured area. Click to keep it selected.';
    return;
  }
  const colors = ['#d35400', '#0077bb', '#9b3fa8'];
  const regions = state.regions[button.dataset.metric] || [];
  // Broad context goes behind the narrower measured boundary.
  regions.map((box, index) => ({box, index})).reverse().forEach(({box, index}) => {
    const [left, top, right, bottom] = box;
    const rect = document.createElementNS(SVG, 'rect');
    const color = colors[index % colors.length];
    for (const [key, value] of Object.entries({x: left, y: top, width: right - left, height: bottom - top,
      fill: color, 'fill-opacity': 0.18, stroke: color, 'stroke-width': 2, 'vector-effect': 'non-scaling-stroke'})) {
      rect.setAttribute(key, value);
    }
    overlay.append(rect);
  });
  status.textContent = `${button.dataset.metric}: outlined regions are measurement areas on the output, not detected defects. Orange is the first region; blue/purple show additional bands. For extra edge strength, orange marks the boundary and blue/purple mark nearby context.`;
}

function bindHighlights() {
  for (const button of document.querySelectorAll('.metric-highlight')) {
    for (const event of ['mouseenter', 'focus']) button.addEventListener(event, () => { state.preview = button; draw(button); });
    for (const event of ['mouseleave', 'blur']) button.addEventListener(event, () => { state.preview = null; draw(state.selected); });
    button.addEventListener('click', () => {
      state.selected = state.selected === button ? null : button;
      state.preview = null;
      draw(state.selected);
      if (state.selected && !button.closest('.metric-workspace')) image.scrollIntoView({behavior: 'smooth', block: 'center'});
    });
  }
}

function setView(showInput) {
  state.showInput = showInput;
  const view = showInput ? 'input' : 'output';
  image.src = `/image?run=${encodeURIComponent(state.run.name)}&view=${view}`;
  image.alt = showInput ? 'Input canvas before inpainting' : 'Inpainted output';
  $('full-image-link').href = image.src;
  $('image-view-label').textContent = showInput ? 'Input canvas' : 'Output';
  $('flip-image').textContent = showInput ? 'Show output' : 'Show input';
  draw(state.preview || state.selected);
}

// --- Run selection and review form ---
function renderRatingFields() {
  $('rating-fields').replaceChildren(...Object.entries(state.ratings).map(([key, label]) =>
    element('fieldset', {}, element('legend', {text: label}), ...[1, 2, 3, 4, 5].flatMap(value => [
      element('label', {}, element('input', {type: 'radio', name: key, value, required: true}), ` ${value}`), ' ']))));
}

async function loadRun(name) {
  const payload = await api(`/api/review/run?run=${encodeURIComponent(name)}`);
  Object.assign(state, {run: payload, regions: payload.regions || {}, selected: null, preview: null});
  history.replaceState(null, '', `/review?run=${encodeURIComponent(payload.name)}`);
  $('run-picker').elements.run.value = payload.name;
  $('run-name').textContent = payload.name;
  $('details-link').href = `/?run=${encodeURIComponent(payload.name)}`;
  $('settings').textContent = JSON.stringify(payload.params, null, 2);
  renderSurroundings(payload.metrics);
  renderScores(payload);
  bindHighlights();
  $('review-form').reset();
  const index = state.runs.indexOf(payload.name);
  $('skip-link').href = `/review?run=${encodeURIComponent(state.runs[(index + 1) % state.runs.length])}`;
  setView(false);
}

async function refreshRuns(preferred) {
  const data = await api('/api/review/runs');
  state.runs = data.runs;
  state.ratings = data.ratings;
  $('all-done').hidden = state.runs.length > 0;
  $('review').hidden = state.runs.length === 0;
  if (!state.runs.length) return;
  $('pending-count').textContent = `${state.runs.length} unreviewed runs`;
  $('run-picker').elements.run.replaceChildren(...state.runs.map(name => element('option', {value: name, text: name})));
  renderRatingFields();
  if (preferred && !state.runs.includes(preferred)) message('Run is missing or already reviewed; opened the first pending run.', true);
  await loadRun(state.runs.includes(preferred) ? preferred : state.runs[0]);
}

image.addEventListener('load', () => {
  document.querySelector('.metric-workspace').style.setProperty('--image-aspect', image.naturalWidth / image.naturalHeight);
  draw(state.preview || state.selected);
});
$('flip-image').addEventListener('click', () => setView(!state.showInput));
$('clear-highlight').addEventListener('click', () => { state.selected = null; state.preview = null; draw(null); });
$('run-picker').addEventListener('submit', event => {
  event.preventDefault();
  loadRun($('run-picker').elements.run.value).catch(error => message(error.message, true));
});
$('skip-link').addEventListener('click', event => {
  event.preventDefault();
  const next = new URL(event.currentTarget.href).searchParams.get('run');
  loadRun(next).catch(error => message(error.message, true));
});
$('review-form').addEventListener('submit', async event => {
  event.preventDefault();
  const form = event.currentTarget;
  const review = {run: state.run.name, notes: form.elements.notes.value};
  for (const key of Object.keys(state.ratings)) review[key] = Number(form.elements[key].value);
  const index = state.runs.indexOf(state.run.name);
  const next = state.runs[index + 1];
  try {
    const result = await api('/api/review/save', review);
    message(result.message);
    await refreshRuns(next);
  } catch (error) { message(error.message, true); }
});

refreshRuns(new URLSearchParams(location.search).get('run')).catch(error => message(error.message, true));
