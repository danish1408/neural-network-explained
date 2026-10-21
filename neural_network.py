"""
Neural Network, from scratch — no framework, just NumPy.
Watch a tiny network learn, so you can see what "a model thinks" really means.

Companion code to my 60-second explainer video.

The idea:
  A neural network is just layers of tiny units. Each unit does one small
  calculation (a weighted sum + an activation), passes its signal forward,
  and the network adjusts millions of these weights during training until
  its output matches reality.

This demo teaches a network the XOR problem:
    0,0 -> 0 | 0,1 -> 1 | 1,0 -> 1 | 1,1 -> 0
XOR is the classic example because a single neuron CANNOT solve it — you
need a hidden layer. So it proves why "layers" matter, not just one unit.

Run:  pip install -r requirements.txt  &&  python neural_network.py
Author: Danish Shabbir  |  Test Automation -> AI / ML
"""

import numpy as np

# the data: 4 examples, each with 2 inputs, and the XOR answer
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])


def sigmoid(z):
    """Activation: squashes any number into a 0..1 signal."""
    return 1 / (1 + np.exp(-z))

def sigmoid_deriv(a):
    """Slope of sigmoid — needed to send the error backwards."""
    return a * (1 - a)


def main(epochs=10000, lr=0.5, seed=42):
    rng = np.random.default_rng(seed)

    # a 2 -> 2 -> 1 network: 2 inputs, a hidden layer of 2 units, 1 output.
    # every connection has a WEIGHT the network will learn.
    W1 = rng.normal(0, 1, (2, 2)); b1 = np.zeros((1, 2))
    W2 = rng.normal(0, 1, (2, 1)); b2 = np.zeros((1, 1))

    print("Teaching a tiny network the XOR problem...\n")
    for epoch in range(epochs):
        # ---- FORWARD PASS: signals flow left to right ----
        h = sigmoid(X @ W1 + b1)          # hidden layer: weighted sum + activation
        out = sigmoid(h @ W2 + b2)        # output layer

        # ---- how wrong are we? ----
        loss = np.mean((out - y) ** 2)

        # ---- BACKWARD PASS: send the error back, nudge the weights ----
        # (this is the chain rule + gradient descent from my earlier videos)
        d_out = (out - y) * sigmoid_deriv(out)
        d_h = (d_out @ W2.T) * sigmoid_deriv(h)
        W2 -= lr * h.T @ d_out;  b2 -= lr * d_out.sum(0, keepdims=True)
        W1 -= lr * X.T @ d_h;    b1 -= lr * d_h.sum(0, keepdims=True)

        if (epoch + 1) % 2000 == 0:
            print(f"epoch {epoch+1:5d} | loss = {loss:.4f}")

    # ---- did it learn? ----
    print("\nWhat the trained network predicts:")
    final = sigmoid(sigmoid(X @ W1 + b1) @ W2 + b2)
    for inp, pred, true in zip(X, final, y):
        print(f"  {inp} -> {pred[0]:.2f}  (should be {true[0]})")
    print("\nNo rules were written for XOR. The network LEARNED it,")
    print("just by adjusting weights until its output matched reality.")


if __name__ == "__main__":
    main()
