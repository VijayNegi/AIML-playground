// Measurement regions only: these overlays do not locate defects or rate severity.
const image = document.getElementById('comparison-image');
const overlay = document.getElementById('metric-overlay');
const status = document.getElementById('highlight-status');
const buttons = [...document.querySelectorAll('.metric-highlight')];
let selected = null;
let preview = null;
let showInput = false;

function draw(button) {
  if (!image || !overlay || !image.naturalWidth) return;
  overlay.replaceChildren();
  overlay.setAttribute('viewBox', `0 0 ${image.naturalWidth} ${image.naturalHeight}`);
  for (const item of buttons) item.setAttribute('aria-pressed', String(item === selected));
  if (showInput) {
    status.textContent = 'Input canvas shown. Switch to output to see metric highlights.';
    return;
  }
  if (!button) {
    status.textContent = 'Hover or focus a metric button to preview its measured area. Click to keep it selected.';
    return;
  }
  const regions = JSON.parse(button.dataset.regions);
  const colors = ['#d35400', '#0077bb', '#9b3fa8'];
  // Broad context goes behind the narrower measured boundary.
  regions.map((box, index) => ({box, index})).reverse().forEach(({box, index}) => {
    const [left, top, right, bottom] = box;
    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    for (const [key, value] of Object.entries({x: left, y: top, width: right - left, height: bottom - top})) {
      rect.setAttribute(key, value);
    }
    rect.setAttribute('fill', colors[index % colors.length]);
    rect.setAttribute('fill-opacity', '0.18');
    rect.setAttribute('stroke', colors[index % colors.length]);
    rect.setAttribute('stroke-width', '2');
    rect.setAttribute('vector-effect', 'non-scaling-stroke');
    overlay.append(rect);
  });
  status.textContent = `${button.dataset.metric}: outlined regions are measurement areas on the output, not detected defects. Orange is the first region; blue/purple show additional bands. For extra edge strength, orange marks the boundary and blue/purple mark nearby context.`;
}

for (const button of buttons) {
  for (const event of ['mouseenter', 'focus']) button.addEventListener(event, () => {
    preview = button;
    draw(button);
  });
  for (const event of ['mouseleave', 'blur']) button.addEventListener(event, () => {
    preview = null;
    draw(selected);
  });
  button.addEventListener('click', () => {
    selected = selected === button ? null : button;
    preview = null;
    draw(selected);
    if (selected && !button.closest('.metric-workspace')) image?.scrollIntoView({behavior: 'smooth', block: 'center'});
  });
}
document.getElementById('clear-highlight')?.addEventListener('click', () => {
  selected = null;
  preview = null;
  draw(null);
});
function imageLoaded() {
  document.querySelector('.metric-workspace')?.style.setProperty('--image-aspect', image.naturalWidth / image.naturalHeight);
  draw(preview || selected);
}
image?.addEventListener('load', imageLoaded);
if (image?.complete && image.naturalWidth) imageLoaded();
document.getElementById('flip-image')?.addEventListener('click', (event) => {
  showInput = !showInput;
  image.src = showInput ? image.dataset.inputSrc : image.dataset.outputSrc;
  image.alt = showInput ? 'Input canvas before inpainting' : 'Inpainted output';
  document.getElementById('full-image-link').href = image.src;
  document.getElementById('image-view-label').textContent = showInput ? 'Input canvas' : 'Output';
  event.currentTarget.textContent = showInput ? 'Show output' : 'Show input';
  draw(preview || selected);
});
