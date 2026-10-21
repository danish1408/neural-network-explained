# Neural Network, from scratch

**A tiny neural network built with nothing but NumPy — so you can see what "a model thinks" actually means.**

Everyone talks about neural networks like they're a mystery. They aren't. A neural
network is just layers of tiny units. Each unit does one small calculation, passes its
signal forward, and during training the network adjusts its weights until the output
matches reality.

This repo lets you watch that happen, in under 60 lines.

> 🎥 Companion code to my 60-second explainer video.

---

## What it does

It teaches a small network the **XOR problem**:

```
0,0 -> 0    0,1 -> 1    1,0 -> 1    1,1 -> 0
```

XOR is the classic example for a reason: a **single neuron cannot solve it**. You need
a hidden layer. So this proves *why layers matter*, not just one unit.

No rules for XOR are ever written in the code. The network **learns** it.

---

## Run it

```bash
pip install -r requirements.txt
python neural_network.py
```

You'll see the loss fall as it learns, then its final predictions:

```
  [0 0] -> 0.02  (should be 0)
  [0 1] -> 0.97  (should be 1)
  [1 0] -> 0.98  (should be 1)
  [1 1] -> 0.02  (should be 0)
```

---

## What's actually inside (the whole of deep learning, in miniature)

- **Weighted sum + activation** — what every single neuron computes.
- **Forward pass** — signals flowing left to right through the layers.
- **Backward pass** — sending the error back and nudging each weight. This is the
  chain rule and gradient descent from my earlier videos, doing the actual learning.
- **Weights** — the numbers the network tunes. That's where the "knowledge" lives.

Bigger networks are just this, scaled up massively: more layers, more units, millions
of weights. The idea doesn't change.

---

## Try this

Open `neural_network.py` and remove the hidden layer (go straight from input to
output). Watch it fail to learn XOR, no matter how long you train. That failure is the
whole reason deep networks have hidden layers.

---

*Part of a series where I break down and build one AI concept at a time.*

**Danish Shabbir** — Test Automation → AI / ML
