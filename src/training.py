# -*- coding: utf-8 -*-
"""학습 루프와 평가. test 분할은 모델 선택에 쓰지 않는다."""

from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from checkpoint import load_model, save_model
from data import load_mnist
from network import NeuralNetwork
from optimizers import Adam, SGD
from paths import DATA_DIR

BATCH_SIZE = 128


def split_training(size, seed=42):
    if size < 2:
        raise ValueError("at least two training examples required")
    order = np.random.default_rng(seed).permutation(size)
    validation = min(10000, max(1, size // 6))
    return order[validation:], order[:validation]

def predictions(model, x):
    return np.concatenate(
        [model.predict(x[i : i + 256]) for i in range(0, len(x), 256)]
    )


def train(args):
    np.random.seed(args.seed)
    (x, y), _ = load_mnist(DATA_DIR)
    train_ids, val_ids = split_training(len(x), args.seed)
    if args.limit:
        train_ids = train_ids[: args.limit]
    model = NeuralNetwork(
        hidden_sizes=args.hidden,
        use_batchnorm=True,
        use_dropout=True,
        dropout_ratio=0.1,
    )
    optimizer = Adam(lr=args.lr) if args.optimizer == "adam" else SGD(lr=args.lr)
    best = -1.0
    history = []
    start = time.monotonic()
    for epoch in range(args.epochs):
        order = np.random.permutation(train_ids)
        losses = []
        for begin in range(0, len(order), BATCH_SIZE):
            ids = order[begin : begin + BATCH_SIZE]
            losses.append(float(model.gradient(x[ids], y[ids])))
            optimizer.update(model.params, model.grads)
        accuracy = float(
            np.mean(predictions(model, x[val_ids]).argmax(1) == y[val_ids])
        )
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "validation_accuracy": accuracy,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if accuracy > best:
            best = accuracy
            metadata = {
                "seed": args.seed,
                "training_count": len(train_ids),
                "validation_count": len(val_ids),
                "epoch": epoch + 1,
                "optimizer": args.optimizer,
                "learning_rate": args.lr,
                "validation_accuracy": accuracy,
                "data_sha256": sha256(
                    (DATA_DIR / "mnist.npz").read_bytes()
                ).hexdigest(),
                "selection": "best validation accuracy; official test partition unused for selection",
            }
            save_model(model, args.output, metadata)
    print(
        json.dumps(
            {
                "saved": str(args.output),
                "training_seconds": time.monotonic() - start,
                "best_validation_accuracy": best,
            }
        )
    )


def evaluate(args):
    model, metadata = load_model(args.model)
    _, (x, y) = load_mnist(DATA_DIR)
    probabilities = predictions(model, x)
    predicted = probabilities.argmax(1)
    confusion = np.zeros((10, 10), dtype=int)
    np.add.at(confusion, (y, predicted), 1)
    wrong = np.flatnonzero(predicted != y)
    result = {
        "test_accuracy": float(np.mean(predicted == y)),
        "test_count": len(y),
        "confusion_matrix": confusion.tolist(),
        "model_sha256": sha256(Path(args.model).read_bytes()).hexdigest(),
        "training": metadata,
        "errors": [
            {
                "index": int(i),
                "expected": int(y[i]),
                "predicted": int(predicted[i]),
                "scores": probabilities[i].tolist(),
            }
            for i in wrong[:20]
        ],
    }
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2) + "\n")
        for i in wrong[:12]:
            Image.fromarray((x[i].reshape(28, 28) * 255).astype("uint8")).save(
                target.parent / f"error-{i}-true-{y[i]}-pred-{predicted[i]}.png"
            )
    print(json.dumps(result, ensure_ascii=False))
