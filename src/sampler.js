import { encodeInto, featureCount } from "./features.js";

const LAND_THRESHOLD = 127;

export class LandMask {
  constructor(pixels, width, height) {
    this.pixels = pixels;
    this.width = width;
    this.height = height;
  }

  isLand(lat, lon) {
    const x = Math.min(this.width - 1, Math.floor(((lon + 180) / 360) * this.width));
    const y = Math.min(this.height - 1, Math.floor(((90 - lat) / 180) * this.height));
    return this.pixels[y * this.width + x] > LAND_THRESHOLD;
  }
}

export async function loadLandMask(url) {
  const response = await fetch(url);
  const bitmap = await createImageBitmap(await response.blob());
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const context = canvas.getContext("2d");
  context.drawImage(bitmap, 0, 0);
  const rgba = context.getImageData(0, 0, bitmap.width, bitmap.height).data;
  const pixels = new Uint8Array(bitmap.width * bitmap.height);
  for (let i = 0; i < pixels.length; i++) pixels[i] = rgba[i * 4];
  return new LandMask(pixels, bitmap.width, bitmap.height);
}

export function fillSphereBatch(mask, nextUniform, fourierBands, inputs, labels, count) {
  const size = featureCount(fourierBands);
  for (let sample = 0; sample < count; sample++) {
    const lat = (Math.asin(2 * nextUniform() - 1) * 180) / Math.PI;
    const lon = 360 * nextUniform() - 180;
    encodeInto(inputs, sample * size, lat, lon, fourierBands);
    labels[sample] = mask.isLand(lat, lon) ? 1 : 0;
  }
}
