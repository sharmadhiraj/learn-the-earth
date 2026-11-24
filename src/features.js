export function featureCount(fourierBands) {
  return 2 + 4 * fourierBands;
}

export function encodeInto(out, offset, lat, lon, fourierBands) {
  const x = lon / 180;
  const y = lat / 90;
  out[offset] = x;
  out[offset + 1] = y;
  let next = offset + 2;
  for (let band = 0; band < fourierBands; band++) {
    const frequency = 2 ** band * Math.PI;
    out[next++] = Math.sin(frequency * x);
    out[next++] = Math.cos(frequency * x);
    out[next++] = Math.sin(frequency * y);
    out[next++] = Math.cos(frequency * y);
  }
}
