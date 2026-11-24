# learn-the-earth

A tiny neural network learns where land and water are, live in your browser.

It gets only a latitude and a longitude and answers one question: land or water? Train it on random points of the
real Earth and watch the map sharpen from a grey blur into continents and coastlines.

**[Try the live demo](https://sharmadhiraj.github.io/learn-the-earth/)**

Inspired by [How does a blind model see the earth?](https://www.lesswrong.com/posts/xwdRzJxyqFqgXTWbH/how-does-a-blind-model-see-the-earth).

## How it works

1. Pick random places on Earth and look up whether each is land or water.
2. The network guesses, and a loss number says how wrong it was.
3. Backpropagation and the Adam optimizer nudge its weights to guess better. Repeat about 300 times a second.
4. Every moment, ask it about every pixel of the world and paint the answers as a map.

Plain networks draw blurry blobs. Adding sine and cosine waves of the coordinates (Fourier features) lets it draw
sharp coastlines. Set **Fourier bands** to 0, then 6, to see the difference.

Full walkthrough: [docs/how-it-works.md](docs/how-it-works.md).

## Run

No install, no build. Serve the folder and open it:

```
python3 -m http.server 8000
```

Then visit `http://localhost:8000`. Code map, tests and how the land mask is made: [docs/development.md](docs/development.md).

## Credits

Land mask derived from [Natural Earth](https://www.naturalearthdata.com/) (public domain).

## License

MIT, see [LICENSE](LICENSE).
