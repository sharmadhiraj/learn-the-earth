# learn-the-earth

A tiny neural network learns where land and water are, live in your browser.

Give the network only a latitude and a longitude. Ask it one question: land or water? Train it on random points of the
real Earth and watch the map sharpen from a grey blur into continents, coastlines and islands, pixel by pixel.

> Status: planned. Nothing is built yet, see [PLAN.md](PLAN.md) for the full implementation plan.

## Why

- A neural net that has never seen a picture of the Earth redraws it from coordinates alone.
- It shows, visually, why plain networks produce blurry maps (spectral bias) and how Fourier features fix it.
- It is built from scratch in plain JavaScript, no framework, no backend, no install. Open the page and press play.

Inspired by [How does a blind model see the earth?](https://www.lesswrong.com/posts/xwdRzJxyqFqgXTWbH/how-does-a-blind-model-see-the-earth),
where a language model is asked "land or water?" for every coordinate and the answers are painted as a map. This
project does the same with a small network you can train and tweak yourself.

## What you will see

- **Learned map:** the network's prediction for every pixel of the world, updating while it trains.
- **Truth map:** the real land mask for comparison.
- **Error map:** where the network is wrong, so you can see which coastlines it finds hardest.
- **Loss and accuracy curves** that move as the map sharpens.
- **Controls:** hidden size, layers, Fourier features on or off, learning rate, play, pause, reset.

## How it works

1. A 360x180 land mask (Natural Earth, public domain) is the ground truth.
2. Random (lat, lon) points are sampled uniformly over the sphere and labelled land or water from the mask.
3. A small MLP, written by hand in JavaScript, trains on those points in a Web Worker so the page stays smooth.
4. Every few steps the worker predicts the whole grid and the page paints it on a canvas.

## Stack

- Plain JavaScript (ES modules), HTML and CSS. No framework, no build step.
- Hand-written MLP with the Adam optimizer, running in a Web Worker.
- Canvas rendering.
- Hosted on GitHub Pages.

## Run locally

Once built, there is nothing to install. Serve the folder with any static server and open it:

```
python3 -m http.server 8000
```

Then visit `http://localhost:8000`.

## Credits

- Land mask derived from [Natural Earth](https://www.naturalearthdata.com/) (public domain).

## License

MIT, see [LICENSE](LICENSE).
