# Implementation plan

Goal: a static web page where a small neural network learns land vs water from (lat, lon) and the viewer watches the
map form live. Effort: about 5 to 6 hours in total. No backend, no build step, no runtime dependencies.

## Scope

In v1:

- Train an MLP from scratch in a Web Worker, paint the learned map, truth map and error map live.
- Controls for size, depth, Fourier features, learning rate, play, pause, reset.
- Loss and accuracy curves.
- GitHub Pages deploy and a README with a GIF.

Out of v1 (see "Later"): guessing game, 3D globe, browser LLM mode, shareable URLs.

## Design decisions

| Topic            | Decision                                                                                                                              |
|------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| Task             | Binary classification: land (1) or water (0) for a (lat, lon)                                                                         |
| Ground truth     | Natural Earth 110m land polygons, rasterized to a 360x180 mask (1 degree per pixel), stored as `data/land-mask.png`                   |
| Sampling         | Uniform over the sphere: lon uniform in [-180, 180], lat = asin(uniform(-1, 1)). Plain uniform lat would oversample the poles         |
| Input encoding   | Normalize lat to [-1, 1] and lon to [-1, 1]. Optional Fourier features: for k in 0..L-1 add sin(2^k * pi * x) and cos(2^k * pi * x) |
| Network          | MLP: input -> H -> H -> 1, ReLU hidden layers, sigmoid output. Defaults H=64, 2 hidden layers, L=6 Fourier bands                      |
| Loss             | Binary cross-entropy                                                                                                                  |
| Optimizer        | Adam, lr 0.003, batch 256                                                                                                             |
| Compute          | Float32Array, hand-written forward and backward pass, in a Web Worker. TF.js only if CPU speed is not enough                          |
| Render grid      | Predict a 180x90 grid for the live view (fast), 360x180 on demand                                                                     |
| Evaluation       | Accuracy and loss on a fixed held-out set of 5000 sphere-uniform points, evaluated every N steps                                      |
| Error map        | Pixel is red if predicted land but truly water, blue if predicted water but truly land, transparent if correct                        |

## Repo layout

```
learn-the-earth/
  index.html
  styles.css
  src/
    main.js            page wiring, controls, worker messaging
    net.js             MLP, Adam, forward and backward
    features.js        normalization and Fourier features
    sampler.js         sphere-uniform sampling and mask lookup
    render.js          canvas painting for learned, truth and error maps, charts
    train.worker.js    training loop, posts snapshots to the page
  data/
    land-mask.png      360x180, white = land, black = water
  tools/
    make-land-mask.md  how the mask was produced (exact commands and source URL)
  README.md
  PLAN.md
  LICENSE
```

## Worker protocol

Page to worker:

- `{ type: "init", config }` where config holds hidden size, layers, fourier bands, learning rate, batch size
- `{ type: "play" }`, `{ type: "pause" }`, `{ type: "reset", config }`, `{ type: "render", width, height }`

Worker to page:

- `{ type: "snapshot", step, loss, accuracy, grid }` where grid is a Float32Array of land probabilities, transferred not
  copied
- `{ type: "ready" }` after the mask is loaded and the network is built

The worker loads the mask itself (fetch the PNG, draw to OffscreenCanvas, read pixels) so the page thread never blocks.

## Milestones

Each milestone is small enough to finish and check on its own.

### M1: Data (about 45 min)

- Download the Natural Earth 110m land shapefile, rasterize it to 360x180 (any tool, document the exact commands in
  `tools/make-land-mask.md`), save `data/land-mask.png`.
- Check: opening the PNG shows a recognizable world map, continents and large islands present.

### M2: Network core (about 1.5 h)

- `net.js`: dense layers, ReLU, sigmoid, BCE, backprop, Adam. `features.js`: normalization and Fourier features.
- Check with a tiny Node script (not shipped): the net overfits 10 fixed points to near-zero loss, and a numerical
  gradient check matches backprop within 1e-3 on a small net.

### M3: Training in a worker (about 1 h)

- `sampler.js`, `train.worker.js`: load mask, sample sphere-uniform batches, train, post snapshots every ~20 steps.
- Check: in the browser console, loss falls and held-out accuracy rises above 85 percent within a minute on defaults.

### M4: Rendering and UI (about 1.5 h)

- `render.js` and `index.html`: three canvases (learned, truth, error), loss and accuracy chart, controls bound to the
  worker.
- Check: pressing play shows the map sharpening live, reset restarts cleanly, changing a control rebuilds the network.

### M5: Polish and ship (about 1 h)

- Responsive layout (works at phone width), labelled controls, sensible defaults, short in-page explainer.
- Record a GIF of a run from blur to continents and put it at the top of the README.
- Enable GitHub Pages from `main`, add the live link to the README and the repo "website" field.

## Acceptance for v1

- Opens from a static server with no install and no console errors.
- With Fourier features on, defaults reach at least 90 percent held-out accuracy and clear continent shapes in under two
  minutes on a laptop. With them off, the map stays visibly blurry (the contrast is the point of the demo).
- Page stays responsive while training.
- README has the GIF, the live link, and credits for Natural Earth.

## Risks and notes

- **Speed:** a pure-JS MLP is fine at this size (about 16k pixels, 64 wide). If slow, reduce the live render grid first,
  then consider TF.js.
- **Dateline and poles:** lat/lon inputs wrap at the dateline. Fine for v1, and the reason for the 3D-input option below.
- **Mask data:** Natural Earth is public domain. Keep the source and process documented in `tools/make-land-mask.md`.
- **No secrets, no accounts, no personal data** anywhere in this project. Everything runs client side.

## Later (v2 ideas, in order of value)

1. **Input encoding toggle:** (lat, lon) vs 3D unit vector (x, y, z), which removes the dateline seam. Compare maps.
2. **Plain vs Fourier split view:** two networks training side by side on the same data.
3. **Guessing game:** "land or water?" for random coordinates, you against the network, with a score.
4. **Shareable URL:** encode the settings in the hash so a run can be shared.
5. **More layers:** elevation bands or country regions as extra targets from the same engine.
6. **Browser LLM mode:** a tiny WebGPU language model paints its own "blind Earth" with the original probe, as a bonus
   mode (large download, needs WebGPU).
7. **Globe view:** render the learned map on a rotating sphere.
