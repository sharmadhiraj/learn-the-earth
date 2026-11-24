import { encodeInto, featureCount } from "./features.js";
import { Mlp } from "./net.js";
import { createRng } from "./random.js";
import { fillSphereBatch } from "./sampler.js";

const BATCH_SIZE = 256;
const HELD_OUT_SIZE = 5000;
const HELD_OUT_SEED = 12345;
const LOSS_FLOOR = 1e-7;

export const DEFAULT_CONFIG = {
  hiddenSize: 64,
  hiddenLayers: 2,
  fourierBands: 6,
  learningRate: 0.003,
};

export class Trainer {
  constructor(mask, config = DEFAULT_CONFIG, seed = 1) {
    this.mask = mask;
    this.config = config;
    this.nextUniform = createRng(seed);
    this.stepCount = 0;

    const inputSize = featureCount(config.fourierBands);
    const hidden = Array(config.hiddenLayers).fill(config.hiddenSize);
    this.net = new Mlp([inputSize, ...hidden, 1], this.nextUniform);

    this.batchInputs = new Float32Array(BATCH_SIZE * inputSize);
    this.batchLabels = new Float32Array(BATCH_SIZE);
    this.#buildHeldOutSet(inputSize);
  }

  #buildHeldOutSet(inputSize) {
    const nextUniform = createRng(HELD_OUT_SEED);
    this.heldOutInputs = new Float32Array(HELD_OUT_SIZE * inputSize);
    this.heldOutLabels = new Float32Array(HELD_OUT_SIZE);
    this.heldOutPredictions = new Float32Array(HELD_OUT_SIZE);
    fillSphereBatch(this.mask, nextUniform, this.config.fourierBands, this.heldOutInputs, this.heldOutLabels, HELD_OUT_SIZE);
  }

  step() {
    fillSphereBatch(this.mask, this.nextUniform, this.config.fourierBands, this.batchInputs, this.batchLabels, BATCH_SIZE);
    this.net.trainBatch(this.batchInputs, this.batchLabels, BATCH_SIZE, this.config.learningRate);
    this.stepCount++;
  }

  evaluate() {
    this.net.predict(this.heldOutInputs, HELD_OUT_SIZE, this.heldOutPredictions);
    let loss = 0;
    let correct = 0;
    for (let i = 0; i < HELD_OUT_SIZE; i++) {
      const p = Math.min(1 - LOSS_FLOOR, Math.max(LOSS_FLOOR, this.heldOutPredictions[i]));
      const label = this.heldOutLabels[i];
      loss -= label * Math.log(p) + (1 - label) * Math.log(1 - p);
      if (p > 0.5 === label > 0.5) correct++;
    }
    return { loss: loss / HELD_OUT_SIZE, accuracy: correct / HELD_OUT_SIZE };
  }

  predictGrid(width, height) {
    const inputSize = featureCount(this.config.fourierBands);
    const inputs = new Float32Array(width * height * inputSize);
    for (let y = 0; y < height; y++) {
      const lat = 90 - ((y + 0.5) * 180) / height;
      for (let x = 0; x < width; x++) {
        const lon = -180 + ((x + 0.5) * 360) / width;
        encodeInto(inputs, (y * width + x) * inputSize, lat, lon, this.config.fourierBands);
      }
    }
    const grid = new Float32Array(width * height);
    this.net.predict(inputs, width * height, grid);
    return grid;
  }
}
