import { drawChart, paintErrorMap, paintLearnedMap, paintTruthMap } from "./render.js";
import { loadLandMask } from "./sampler.js";

const NORMAL_GRID = { width: 180, height: 90 };
const SHARP_GRID = { width: 360, height: 180 };
const MAX_HISTORY = 400;

const elements = {
  learnedMap: document.getElementById("learned-map"),
  truthMap: document.getElementById("truth-map"),
  errorMap: document.getElementById("error-map"),
  accuracyChart: document.getElementById("accuracy-chart"),
  lossChart: document.getElementById("loss-chart"),
  step: document.getElementById("step"),
  accuracy: document.getElementById("accuracy"),
  loss: document.getElementById("loss"),
  play: document.getElementById("play"),
  reset: document.getElementById("reset"),
  fourier: document.getElementById("fourier"),
  fourierValue: document.getElementById("fourier-value"),
  hiddenSize: document.getElementById("hidden-size"),
  hiddenLayers: document.getElementById("hidden-layers"),
  learningRate: document.getElementById("learning-rate"),
  sharp: document.getElementById("sharp"),
};

const worker = new Worker(new URL("./train.worker.js", import.meta.url), { type: "module" });
let mask = null;
let runId = 0;
let isPlaying = false;
let history = { accuracy: [], loss: [] };
let paintedTruthSize = "";

function readConfig() {
  return {
    hiddenSize: Number(elements.hiddenSize.value),
    hiddenLayers: Number(elements.hiddenLayers.value),
    fourierBands: Number(elements.fourier.value),
    learningRate: Number(elements.learningRate.value),
  };
}

function setPlaying(playing) {
  isPlaying = playing;
  elements.play.textContent = playing ? "Pause" : "Play";
  worker.postMessage({ type: playing ? "play" : "pause" });
}

function restartRun() {
  runId++;
  history = { accuracy: [], loss: [] };
  worker.postMessage({ type: "reset", runId, config: readConfig() });
}

function remember(list, value) {
  list.push(value);
  if (list.length > MAX_HISTORY) list.shift();
}

function showSnapshot({ step, loss, accuracy, grid, width, height }) {
  elements.step.textContent = step.toLocaleString();
  elements.accuracy.textContent = `${(accuracy * 100).toFixed(1)}%`;
  elements.loss.textContent = loss.toFixed(3);

  remember(history.accuracy, accuracy);
  remember(history.loss, loss);
  const styles = getComputedStyle(document.documentElement);
  drawChart(elements.accuracyChart, history.accuracy, { color: styles.getPropertyValue("--accuracy-line"), min: 0.5, max: 1 });
  drawChart(elements.lossChart, history.loss, { color: styles.getPropertyValue("--loss-line"), min: 0 });

  paintLearnedMap(elements.learnedMap, grid, width, height);
  paintErrorMap(elements.errorMap, grid, mask, width, height);
  if (paintedTruthSize !== `${width}x${height}`) {
    paintTruthMap(elements.truthMap, mask, width, height);
    paintedTruthSize = `${width}x${height}`;
  }
}

worker.onmessage = ({ data }) => {
  if (data.type === "snapshot" && data.runId === runId && mask) showSnapshot(data);
};

elements.play.addEventListener("click", () => setPlaying(!isPlaying));
elements.reset.addEventListener("click", restartRun);
elements.sharp.addEventListener("change", () => {
  const { width, height } = elements.sharp.checked ? SHARP_GRID : NORMAL_GRID;
  worker.postMessage({ type: "render", width, height });
});
elements.fourier.addEventListener("input", () => {
  elements.fourierValue.textContent = elements.fourier.value;
});
for (const control of [elements.fourier, elements.hiddenSize, elements.hiddenLayers, elements.learningRate]) {
  control.addEventListener("change", restartRun);
}

mask = await loadLandMask(new URL("../data/land-mask.png", import.meta.url));
worker.postMessage({ type: "init", runId, config: readConfig() });
