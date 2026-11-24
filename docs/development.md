# Development

Plain JavaScript (ES modules), no framework, no build step, no runtime dependencies.

## Code map

| File                       | What it does                                                         |
|----------------------------|----------------------------------------------------------------------|
| `src/net.js`               | The network: forward pass, backpropagation, Adam                     |
| `src/features.js`          | Turns latitude and longitude into inputs, including Fourier features |
| `src/sampler.js`           | Land mask lookup and sphere-uniform sampling                         |
| `src/trainer.js`           | One training step, held-out evaluation, predicting a whole grid      |
| `src/train.worker.js`      | Runs the trainer in a Web Worker so the page stays smooth            |
| `src/render.js`            | Paints the maps and charts on canvases                               |
| `src/main.js`              | Wires the controls to the worker                                     |
| `tools/make-land-mask.mjs` | Builds the land mask from Natural Earth                              |

Start with `src/net.js` (about 150 lines).

## Tests

```
npm test
```

Checks that backpropagation matches a numerical gradient and that the network can memorise 10 points. To check the real
trainer on the real map (30 seconds each with 0 and 6 Fourier bands):

```
node tests/train-check.mjs 30 0,6
```

## Land mask

`data/land-mask.png` is 360x180, one pixel per degree, white is land. Rebuild it from Natural Earth 110m land:

```
curl -L -o ne_110m_land.geojson https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_land.geojson
node tools/make-land-mask.mjs ne_110m_land.geojson data/land-mask.png
```
