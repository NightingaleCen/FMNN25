"""
Script for random-start search for the two-neuron Task 8 network.
Most code is copied from `task8.ipynb`.
"""

import argparse
import csv
import gzip
import pickle
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
with gzip.open(ROOT / "mnist.pkl.gz", "rb") as file:
    (images, labels), _, _ = pickle.load(file, encoding="latin1")
X = np.c_[images[:50].astype(float), np.ones(50)]
Y = np.eye(10)[labels[:50].astype(int)]


def sigmoid(z):
    return 0.5 * (1 + np.tanh(z / 2))


def activate(z, name):
    if name == "sigmoid":
        a = sigmoid(z)
        return a, a * (1 - a)
    if name == "softplus":
        return np.logaddexp(0, z), sigmoid(z)
    if name == "tanh":
        a = np.tanh(z)
        return a, 1 - a * a
    if name == "relu":
        return np.maximum(z, 0), (z > 0).astype(float)
    if name == "leaky_relu":
        return np.maximum(z, 0.1 * z), np.where(z > 0, 1.0, 0.1)
    raise ValueError(name)


def initial_weights(width, seed, hidden, output, scale, output_bias=None):
    rng = np.random.default_rng(seed)
    hidden_sd = (np.sqrt(2) if hidden in ("relu", "leaky_relu") else 1) / np.sqrt(785)
    w1 = rng.normal(0, scale * hidden_sd, (width, 785))
    w2 = rng.normal(0, 1 / np.sqrt(width + 1), (10, width + 1))
    if output_bias is not None:
        w2[:, -1] = output_bias
    elif output == "sigmoid":
        w2[:, -1] = np.log(0.1 / 0.9)
    elif output == "softplus":
        w2[:, -1] = np.log(np.expm1(0.1))
    else:
        w2[:, -1] = 0.1
    return w1, w2


def loss_gradient(w1, w2, hidden, output, indices=None):
    x, y = (X, Y) if indices is None else (X[indices], Y[indices])
    h, dh = activate(x @ w1.T, hidden)
    hb = np.c_[h, np.ones(len(x))]
    pred, dpred = activate(hb @ w2.T, output)
    error = pred - y
    loss = np.sum(error * error) / (2 * len(x))
    delta2 = error * dpred / len(x)
    delta1 = (delta2 @ w2[:, :-1]) * dh
    return loss, delta1.T @ x, delta2.T @ hb


def train(hidden, output, seed, lr, epochs, scale=1, method="Adam", batch=50, output_bias=None):
    w1, w2 = initial_weights(2, seed, hidden, output, scale, output_bias)
    params = [w1, w2]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    best = np.inf
    first = None
    rng = np.random.default_rng(seed + 1000)
    step = 0
    for epoch in range(1, epochs + 1):
        for ids in np.array_split(rng.permutation(50), 50 // batch):
            step += 1
            _, g1, g2 = loss_gradient(w1, w2, hidden, output, ids)
            for k, (p, g) in enumerate(zip(params, (g1, g2))):
                if method == "Adam":
                    m[k] = .9 * m[k] + .1 * g
                    v[k] = .999 * v[k] + .001 * g * g
                    p -= lr * (m[k] / (1 - .9**step)) / (np.sqrt(v[k] / (1 - .999**step)) + 1e-8)
                else:
                    p -= lr * g
        loss, _, _ = loss_gradient(w1, w2, hidden, output)
        if loss < best:
            best = loss
        if loss < 1e-8:
            first = epoch
            bias_name = "" if output_bias is None else f"_bias{output_bias}"
            np.savez(OUT / f"common_h2_{hidden}_{output}_s{seed}_lr{lr}_sc{scale}_{method}{bias_name}.npz",
                     w1=w1, w2=w2, loss=loss, epoch=epoch)
            break
    return first, best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--pairs", default="")
    parser.add_argument("--lrs", default="0.01")
    parser.add_argument("--scales", default="1")
    parser.add_argument("--method", default="Adam")
    parser.add_argument("--batch", type=int, default=50)
    parser.add_argument("--output-bias", type=float)
    args = parser.parse_args()
    pairs = ([(h, o) for h in ("sigmoid", "tanh", "softplus", "relu", "leaky_relu")
              for o in ("sigmoid", "softplus", "relu")]
             if not args.pairs else [tuple(pair.split(":")) for pair in args.pairs.split(",")])
    configs = [(h, o, s, float(lr), float(sc)) for h, o in pairs
               for s in range(args.seed_start, args.seed_start + args.seeds)
               for lr in args.lrs.split(",") for sc in args.scales.split(",")]
    path = OUT / ("common_search.csv" if args.output_bias is None else "bias_search.csv")
    exists = path.exists()
    with path.open("a", newline="") as file:
        writer = csv.writer(file)
        if not exists:
            writer.writerow(("hidden", "output", "seed", "method", "lr", "scale", "batch", "epochs", "first_below", "best_loss", "output_bias") if args.output_bias is not None else ("hidden", "output", "seed", "method", "lr", "scale", "batch", "epochs", "first_below", "best_loss"))
        for i, (h, o, seed, lr, scale) in enumerate(configs, 1):
            first, best = train(h, o, seed, lr, args.epochs, scale, args.method, args.batch, args.output_bias)
            row = (h, o, seed, args.method, lr, scale, args.batch, args.epochs, first, best)
            writer.writerow((*row, args.output_bias) if args.output_bias is not None else row)
            file.flush()
            print(i, len(configs), h, o, seed, lr, scale, first, f"{best:.3e}", flush=True)


if __name__ == "__main__":
    main()
