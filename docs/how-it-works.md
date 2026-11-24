# How it works

## The task

The network is a function. You give it two numbers, a latitude and a longitude. It gives back one number between 0 and
1: how likely that spot is land. Near 1 means land, near 0 means water, 0.5 means "no idea".

## What a network is

Layers of tiny calculations. Each unit multiplies its inputs by numbers called **weights**, adds them up, and bends the
result (a ReLU, which replaces negatives with 0). Stack a few layers and you can draw complicated shapes. The weights
start random, so the first map is a grey smear. Training is finding better weights.

## One training step

Repeated about 300 times a second:

1. **Sample.** Pick 256 random places on Earth and look each up in the land mask: land is 1, water is 0.
2. **Guess.** Run them through the network.
3. **Score.** Compare guesses with the truth using cross-entropy loss. Confident and wrong is punished hard.
4. **Blame.** Backpropagation works out, for every weight, whether nudging it up or down lowers the loss. It is the
   chain rule applied layer by layer, backwards from the output.
5. **Nudge.** The Adam optimizer moves each weight a small step in the helpful direction. The step size is the
   **learning rate**.

The network never sees a picture, only coordinates and answers.

## Drawing the map

Every few hundred milliseconds the page asks the network about every pixel of the world and colours each pixel by its
answer. The error map compares that picture with the truth.

## Why Fourier features matter

A plain network finds smooth shapes easy and sharp detail hard (spectral bias). Raw latitude and longitude change
slowly, so a coastline is a very sharp feature to build from them.

Fourier features give the network extra inputs: sine and cosine waves of the coordinates at doubling frequencies. A
sharp edge is then easy to assemble from the right waves. Each band adds one frequency.

In a 30 second headless test, the plain network reached about 94 percent held-out accuracy and drew vague polygons.
With 6 bands it reached about 97.5 percent and drew recognizable continents.

## Two details that matter

- **Sampling is uniform over the sphere, not over the map.** Uniform latitude would crowd points near the poles, where
  the flat map stretches. The sampler uses `lat = asin(uniform(-1, 1))`.
- **Accuracy is measured on places the network never trains on.** A fixed set of 5000 held-out points is scored every
  snapshot. That separates "learned the Earth" from "memorised the training points".

## Things to try

- Fourier bands 0, 2, 6, 8. When do extra bands stop helping?
- Hidden size 16 versus 128. Better, or just slower?
- One layer versus four.
- Learning rate 0.01. What does a too-large step do?
- Which places stay red or blue longest? Small islands, narrow straits and the dateline are good guesses.
