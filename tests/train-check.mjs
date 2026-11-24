import { DEFAULT_CONFIG, Trainer } from "../src/trainer.js";
import { loadLandMaskFromDisk } from "./load-mask.mjs";

const mask = loadLandMaskFromDisk();
const seconds = Number(process.argv[2] ?? 30);
const fourierBandOptions = (process.argv[3] ?? "0,6").split(",").map(Number);

for (const fourierBands of fourierBandOptions) {
  const trainer = new Trainer(mask, { ...DEFAULT_CONFIG, fourierBands });
  const startedAt = performance.now();
  let nextReport = 5;
  console.log(`fourier bands ${fourierBands}`);
  while ((performance.now() - startedAt) / 1000 < seconds) {
    trainer.step();
    const elapsed = (performance.now() - startedAt) / 1000;
    if (elapsed >= nextReport) {
      const { loss, accuracy } = trainer.evaluate();
      console.log(`  ${elapsed.toFixed(0)}s step ${trainer.stepCount} loss ${loss.toFixed(3)} accuracy ${(accuracy * 100).toFixed(1)}%`);
      nextReport += 5;
    }
  }
}
