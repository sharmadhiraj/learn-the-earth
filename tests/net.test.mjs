import assert from "node:assert/strict";
import { Mlp } from "../src/net.js";
import { createRng } from "../src/random.js";

function testGradientCheck() {
  const nextUniform = createRng(7);
  const count = 8;
  const net = new Mlp([4, 5, 3, 1], nextUniform, { capacity: 16, ArrayType: Float64Array });
  const inputs = Float64Array.from({ length: count * 4 }, () => nextUniform() * 2 - 1);
  const labels = Float64Array.from({ length: count }, () => (nextUniform() > 0.5 ? 1 : 0));

  net.computeGradients(inputs, labels, count);
  const analytic = net.weights.map((_, layer) => ({
    weights: Float64Array.from(net.weightGradients[layer]),
    biases: Float64Array.from(net.biasGradients[layer]),
  }));

  const step = 1e-6;
  let worstError = 0;
  for (let layer = 0; layer < net.layerCount; layer++) {
    for (const [params, expected] of [
      [net.weights[layer], analytic[layer].weights],
      [net.biases[layer], analytic[layer].biases],
    ]) {
      for (let i = 0; i < params.length; i++) {
        const original = params[i];
        params[i] = original + step;
        const lossUp = net.computeGradients(inputs, labels, count);
        params[i] = original - step;
        const lossDown = net.computeGradients(inputs, labels, count);
        params[i] = original;
        worstError = Math.max(worstError, Math.abs((lossUp - lossDown) / (2 * step) - expected[i]));
      }
    }
  }
  assert.ok(worstError < 1e-6, `gradient mismatch ${worstError}`);
  console.log(`gradient check ok, worst error ${worstError.toExponential(2)}`);
}

function testOverfitsTenPoints() {
  const nextUniform = createRng(3);
  const net = new Mlp([2, 16, 16, 1], nextUniform, { capacity: 16 });
  const inputs = Float32Array.from({ length: 20 }, () => nextUniform() * 2 - 1);
  const labels = Float32Array.from({ length: 10 }, (_, i) => i % 2);

  let loss = Infinity;
  for (let step = 0; step < 2000; step++) loss = net.trainBatch(inputs, labels, 10, 0.01);
  assert.ok(loss < 0.01, `did not overfit, loss ${loss}`);
  console.log(`overfit ok, final loss ${loss.toExponential(2)}`);
}

testGradientCheck();
testOverfitsTenPoints();
