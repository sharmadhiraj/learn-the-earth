import { loadLandMask } from "./sampler.js";
import { Trainer } from "./trainer.js";

const MASK_URL = new URL("../data/land-mask.png", import.meta.url);
const WORK_SLICE_MS = 40;
const SNAPSHOT_INTERVAL_MS = 250;

let mask = null;
let trainer = null;
let runId = 0;
let isPlaying = false;
let isLoopActive = false;
let lastSnapshotAt = 0;
let grid = { width: 180, height: 90 };

function postSnapshot() {
  const { loss, accuracy } = trainer.evaluate();
  const probabilities = trainer.predictGrid(grid.width, grid.height);
  lastSnapshotAt = performance.now();
  postMessage(
    { type: "snapshot", runId, step: trainer.stepCount, loss, accuracy, width: grid.width, height: grid.height, grid: probabilities },
    [probabilities.buffer],
  );
}

function ensureTrainLoop() {
  if (isPlaying && trainer && !isLoopActive) {
    isLoopActive = true;
    trainLoop();
  }
}

function trainLoop() {
  if (!isPlaying) {
    isLoopActive = false;
    return;
  }
  const sliceEnd = performance.now() + WORK_SLICE_MS;
  while (performance.now() < sliceEnd) trainer.step();
  if (performance.now() - lastSnapshotAt >= SNAPSHOT_INTERVAL_MS) postSnapshot();
  setTimeout(trainLoop, 0);
}

async function buildTrainer(config, newRunId) {
  mask ??= await loadLandMask(MASK_URL);
  runId = newRunId;
  trainer = new Trainer(mask, config);
  postMessage({ type: "ready", runId });
  postSnapshot();
  ensureTrainLoop();
}

self.onmessage = async ({ data }) => {
  switch (data.type) {
    case "init":
    case "reset":
      await buildTrainer(data.config, data.runId);
      break;
    case "play":
      isPlaying = true;
      ensureTrainLoop();
      break;
    case "pause":
      isPlaying = false;
      if (trainer) postSnapshot();
      break;
    case "render":
      grid = { width: data.width, height: data.height };
      if (trainer) postSnapshot();
      break;
  }
};
