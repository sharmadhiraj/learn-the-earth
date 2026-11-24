import { readFileSync } from "node:fs";
import { inflateSync } from "node:zlib";
import { LandMask } from "../src/sampler.js";

export function loadLandMaskFromDisk(path = new URL("../data/land-mask.png", import.meta.url)) {
  const png = readFileSync(path);
  const width = png.readUInt32BE(16);
  const height = png.readUInt32BE(20);
  const dataChunks = [];
  for (let offset = 8; offset < png.length; ) {
    const length = png.readUInt32BE(offset);
    const type = png.toString("ascii", offset + 4, offset + 8);
    if (type === "IDAT") dataChunks.push(png.subarray(offset + 8, offset + 8 + length));
    offset += 12 + length;
  }
  const rows = inflateSync(Buffer.concat(dataChunks));
  const pixels = new Uint8Array(width * height);
  for (let y = 0; y < height; y++) {
    rows.copy(pixels, y * width, y * (width + 1) + 1, (y + 1) * (width + 1));
  }
  return new LandMask(pixels, width, height);
}
