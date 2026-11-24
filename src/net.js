import { randomNormal } from "./random.js";

const ADAM_BETA1 = 0.9;
const ADAM_BETA2 = 0.999;
const ADAM_EPSILON = 1e-8;
const PROBABILITY_FLOOR = 1e-7;

export class Mlp {
  constructor(layerSizes, nextUniform, { capacity = 1024, ArrayType = Float32Array } = {}) {
    this.layerSizes = layerSizes;
    this.layerCount = layerSizes.length - 1;
    this.capacity = capacity;
    this.adamStep = 0;

    this.weights = [];
    this.biases = [];
    this.weightGradients = [];
    this.biasGradients = [];
    this.adamState = [];
    this.activations = [null];
    this.deltas = [];

    for (let layer = 0; layer < this.layerCount; layer++) {
      const inputSize = layerSizes[layer];
      const outputSize = layerSizes[layer + 1];
      const weights = new ArrayType(inputSize * outputSize);
      const heScale = Math.sqrt(2 / inputSize);
      for (let i = 0; i < weights.length; i++) weights[i] = randomNormal(nextUniform) * heScale;

      this.weights.push(weights);
      this.biases.push(new ArrayType(outputSize));
      this.weightGradients.push(new ArrayType(weights.length));
      this.biasGradients.push(new ArrayType(outputSize));
      this.adamState.push({
        weightMean: new ArrayType(weights.length),
        weightVariance: new ArrayType(weights.length),
        biasMean: new ArrayType(outputSize),
        biasVariance: new ArrayType(outputSize),
      });
      this.activations.push(new ArrayType(capacity * outputSize));
      this.deltas.push(new ArrayType(capacity * outputSize));
    }
  }

  forward(inputs, count) {
    for (let layer = 0; layer < this.layerCount; layer++) {
      const inputSize = this.layerSizes[layer];
      const outputSize = this.layerSizes[layer + 1];
      const layerInput = layer === 0 ? inputs : this.activations[layer];
      const layerOutput = this.activations[layer + 1];
      const weights = this.weights[layer];
      const biases = this.biases[layer];
      const isLast = layer === this.layerCount - 1;

      for (let sample = 0; sample < count; sample++) {
        const inputBase = sample * inputSize;
        for (let unit = 0; unit < outputSize; unit++) {
          const weightBase = unit * inputSize;
          let sum = biases[unit];
          for (let i = 0; i < inputSize; i++) sum += weights[weightBase + i] * layerInput[inputBase + i];
          layerOutput[sample * outputSize + unit] = isLast ? 1 / (1 + Math.exp(-sum)) : sum > 0 ? sum : 0;
        }
      }
    }
  }

  computeGradients(inputs, labels, count) {
    this.forward(inputs, count);
    const probabilities = this.activations[this.layerCount];
    const outputDelta = this.deltas[this.layerCount - 1];
    let totalLoss = 0;
    for (let sample = 0; sample < count; sample++) {
      const p = Math.min(1 - PROBABILITY_FLOOR, Math.max(PROBABILITY_FLOOR, probabilities[sample]));
      const label = labels[sample];
      totalLoss -= label * Math.log(p) + (1 - label) * Math.log(1 - p);
      outputDelta[sample] = (probabilities[sample] - label) / count;
    }
    this.backward(inputs, count);
    return totalLoss / count;
  }

  backward(inputs, count) {
    for (let layer = this.layerCount - 1; layer >= 0; layer--) {
      const inputSize = this.layerSizes[layer];
      const outputSize = this.layerSizes[layer + 1];
      const layerInput = layer === 0 ? inputs : this.activations[layer];
      const delta = this.deltas[layer];
      const weights = this.weights[layer];
      const weightGradient = this.weightGradients[layer];
      const biasGradient = this.biasGradients[layer];
      const previousDelta = layer > 0 ? this.deltas[layer - 1] : null;

      weightGradient.fill(0);
      biasGradient.fill(0);
      if (previousDelta) previousDelta.fill(0, 0, count * inputSize);

      for (let sample = 0; sample < count; sample++) {
        const inputBase = sample * inputSize;
        for (let unit = 0; unit < outputSize; unit++) {
          const d = delta[sample * outputSize + unit];
          if (d === 0) continue;
          const weightBase = unit * inputSize;
          biasGradient[unit] += d;
          for (let i = 0; i < inputSize; i++) {
            weightGradient[weightBase + i] += d * layerInput[inputBase + i];
            if (previousDelta) previousDelta[inputBase + i] += weights[weightBase + i] * d;
          }
        }
      }

      if (previousDelta) {
        for (let i = 0; i < count * inputSize; i++) {
          if (layerInput[i] <= 0) previousDelta[i] = 0;
        }
      }
    }
  }

  applyAdam(learningRate) {
    this.adamStep++;
    const correction1 = 1 - ADAM_BETA1 ** this.adamStep;
    const correction2 = 1 - ADAM_BETA2 ** this.adamStep;
    for (let layer = 0; layer < this.layerCount; layer++) {
      const state = this.adamState[layer];
      this.#adamUpdate(this.weights[layer], this.weightGradients[layer], state.weightMean, state.weightVariance, learningRate, correction1, correction2);
      this.#adamUpdate(this.biases[layer], this.biasGradients[layer], state.biasMean, state.biasVariance, learningRate, correction1, correction2);
    }
  }

  #adamUpdate(params, gradients, mean, variance, learningRate, correction1, correction2) {
    for (let i = 0; i < params.length; i++) {
      const g = gradients[i];
      mean[i] = ADAM_BETA1 * mean[i] + (1 - ADAM_BETA1) * g;
      variance[i] = ADAM_BETA2 * variance[i] + (1 - ADAM_BETA2) * g * g;
      params[i] -= (learningRate * (mean[i] / correction1)) / (Math.sqrt(variance[i] / correction2) + ADAM_EPSILON);
    }
  }

  trainBatch(inputs, labels, count, learningRate) {
    const loss = this.computeGradients(inputs, labels, count);
    this.applyAdam(learningRate);
    return loss;
  }

  predict(inputs, count, output) {
    const inputSize = this.layerSizes[0];
    const probabilities = this.activations[this.layerCount];
    for (let start = 0; start < count; start += this.capacity) {
      const chunk = Math.min(this.capacity, count - start);
      this.forward(inputs.subarray(start * inputSize, (start + chunk) * inputSize), chunk);
      for (let i = 0; i < chunk; i++) output[start + i] = probabilities[i];
    }
  }
}
