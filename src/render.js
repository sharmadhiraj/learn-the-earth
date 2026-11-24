const WATER_COLOR = [24, 52, 96];
const LAND_COLOR = [214, 196, 150];
const FALSE_LAND_COLOR = [235, 70, 70];
const MISSED_LAND_COLOR = [80, 170, 255];

function mixColor(landShare) {
  return WATER_COLOR.map((water, i) => water + (LAND_COLOR[i] - water) * landShare);
}

function paintPixels(canvas, width, height, colorAt) {
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  const image = context.createImageData(width, height);
  for (let i = 0; i < width * height; i++) {
    const [r, g, b, a = 255] = colorAt(i);
    image.data.set([r, g, b, a], i * 4);
  }
  context.putImageData(image, 0, 0);
}

function trueLandAt(mask, index, width, height) {
  const lat = 90 - (((Math.floor(index / width)) + 0.5) * 180) / height;
  const lon = -180 + (((index % width) + 0.5) * 360) / width;
  return mask.isLand(lat, lon);
}

export function paintLearnedMap(canvas, grid, width, height) {
  paintPixels(canvas, width, height, (i) => mixColor(grid[i]));
}

export function paintTruthMap(canvas, mask, width, height) {
  paintPixels(canvas, width, height, (i) => mixColor(trueLandAt(mask, i, width, height) ? 1 : 0));
}

export function paintErrorMap(canvas, grid, mask, width, height) {
  paintPixels(canvas, width, height, (i) => {
    const predictedLand = grid[i] > 0.5;
    const isLand = trueLandAt(mask, i, width, height);
    if (predictedLand === isLand) return [0, 0, 0, 0];
    return predictedLand ? FALSE_LAND_COLOR : MISSED_LAND_COLOR;
  });
}

export function drawChart(canvas, values, { color, min, max }) {
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth * ratio;
  const height = canvas.clientHeight * ratio;
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, width, height);
  if (values.length < 2) return;

  const low = min ?? Math.min(...values);
  const high = max ?? Math.max(...values);
  const span = high - low || 1;
  context.lineWidth = 2 * ratio;
  context.strokeStyle = color;
  context.beginPath();
  values.forEach((value, i) => {
    const x = (i / (values.length - 1)) * width;
    const y = height - ((value - low) / span) * (height - 4 * ratio) - 2 * ratio;
    if (i === 0) context.moveTo(x, y);
    else context.lineTo(x, y);
  });
  context.stroke();
}
