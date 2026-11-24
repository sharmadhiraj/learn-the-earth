import { readFileSync, writeFileSync } from "node:fs";
import { deflateSync } from "node:zlib";

const WIDTH = 360;
const HEIGHT = 180;

function isInsideRing(lon, lat, ring) {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [xi, yi] = ring[i];
    const [xj, yj] = ring[j];
    if (yi > lat !== yj > lat && lon < ((xj - xi) * (lat - yi)) / (yj - yi) + xi) {
      inside = !inside;
    }
  }
  return inside;
}

function isInsidePolygon(lon, lat, rings) {
  let inside = false;
  for (const ring of rings) {
    if (isInsideRing(lon, lat, ring)) inside = !inside;
  }
  return inside;
}

function collectPolygons(geojson) {
  const polygons = [];
  for (const feature of geojson.features) {
    const { type, coordinates } = feature.geometry;
    if (type === "Polygon") polygons.push(coordinates);
    if (type === "MultiPolygon") polygons.push(...coordinates);
  }
  return polygons;
}

function rasterize(polygons) {
  const pixels = new Uint8Array(WIDTH * HEIGHT);
  for (let y = 0; y < HEIGHT; y++) {
    const lat = 90 - y - 0.5;
    for (let x = 0; x < WIDTH; x++) {
      const lon = -180 + x + 0.5;
      if (polygons.some((rings) => isInsidePolygon(lon, lat, rings))) {
        pixels[y * WIDTH + x] = 255;
      }
    }
  }
  return pixels;
}

const crcTable = Array.from({ length: 256 }, (_, n) => {
  let c = n;
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  return c >>> 0;
});

function crc32(buffer) {
  let c = 0xffffffff;
  for (const byte of buffer) c = crcTable[(c ^ byte) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function pngChunk(type, data) {
  const length = Buffer.alloc(4);
  length.writeUInt32BE(data.length);
  const body = Buffer.concat([Buffer.from(type), data]);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(body));
  return Buffer.concat([length, body, crc]);
}

function encodeGrayscalePng(pixels) {
  const header = Buffer.alloc(13);
  header.writeUInt32BE(WIDTH, 0);
  header.writeUInt32BE(HEIGHT, 4);
  header[8] = 8;
  header[9] = 0;
  const rows = Buffer.alloc((WIDTH + 1) * HEIGHT);
  for (let y = 0; y < HEIGHT; y++) {
    Buffer.from(pixels.subarray(y * WIDTH, (y + 1) * WIDTH)).copy(rows, y * (WIDTH + 1) + 1);
  }
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    pngChunk("IHDR", header),
    pngChunk("IDAT", deflateSync(rows)),
    pngChunk("IEND", Buffer.alloc(0)),
  ]);
}

const [inputPath, outputPath] = process.argv.slice(2);
const geojson = JSON.parse(readFileSync(inputPath, "utf8"));
const pixels = rasterize(collectPolygons(geojson));
writeFileSync(outputPath, encodeGrayscalePng(pixels));

const landShare = pixels.filter((p) => p === 255).length / pixels.length;
console.log(`wrote ${outputPath}, land share ${(landShare * 100).toFixed(1)}%`);
