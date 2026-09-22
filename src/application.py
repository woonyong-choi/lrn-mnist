"""Runnable single-digit recognizer built on the lab's NumPy layers."""

from __future__ import annotations
import argparse
import base64
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import threading
import time

import numpy as np
from PIL import Image

from data import load_mnist
from network import NeuralNetwork
from optimizers import Adam, SGD

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "models/reference.npz"


def split_training(size, seed=42):
    if size < 2:
        raise ValueError("at least two training examples required")
    order = np.random.default_rng(seed).permutation(size)
    validation = min(10000, max(1, size // 6))
    return order[validation:], order[:validation]


def save_model(model, path, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"param_{key}": value for key, value in model.params.items()}
    for name, layer in model.layers.items():
        if name.startswith("BatchNorm"):
            arrays[name + "_mean"] = layer.running_mean
            arrays[name + "_var"] = layer.running_var
    config = {
        key: getattr(model, key)
        for key in [
            "hidden_sizes",
            "use_batchnorm",
            "use_dropout",
            "dropout_ratio",
            "batchnorm_momentum",
        ]
    }
    arrays["manifest"] = np.array(
        json.dumps({"format": 1, "config": config, "metadata": metadata})
    )
    temp = path.with_suffix(".tmp")
    with temp.open("wb") as f:
        np.savez_compressed(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def load_model(path):
    with np.load(path, allow_pickle=False) as data:
        manifest = json.loads(str(data["manifest"]))
        if manifest["format"] != 1:
            raise ValueError("unsupported model format")
        model = NeuralNetwork(**manifest["config"])
        for key, value in model.params.items():
            saved = data["param_" + key]
            if saved.shape != value.shape or not np.isfinite(saved).all():
                raise ValueError("invalid model weights")
            value[:] = saved
        for name, layer in model.layers.items():
            if name.startswith("BatchNorm"):
                mean, var = data[name + "_mean"], data[name + "_var"]
                if (
                    mean.shape != layer.running_mean.shape
                    or var.shape != layer.running_var.shape
                    or not np.isfinite(mean).all()
                    or not (np.isfinite(var).all() and (var >= 0).all())
                ):
                    raise ValueError("invalid batchnorm statistics")
                layer.running_mean = mean.copy()
                layer.running_var = var.copy()
    return model, manifest["metadata"]


def preprocess_image(image):
    image = image.convert("L")
    if image.width > 4096 or image.height > 4096:
        raise ValueError("image exceeds 4096 pixels per side")
    array = np.asarray(image, dtype=np.uint8)
    border = np.concatenate((array[0], array[-1], array[:, 0], array[:, -1]))
    if np.median(border) > 127:
        array = 255 - array
    ys, xs = np.nonzero(array > 25)
    if not len(xs):
        raise ValueError("blank input: draw one digit")
    crop = Image.fromarray(array[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1])
    scale = 20 / max(crop.size)
    crop = crop.resize(
        (max(1, round(crop.width * scale)), max(1, round(crop.height * scale))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("L", (28, 28))
    canvas.paste(crop, ((28 - crop.width) // 2, (28 - crop.height) // 2))
    pixels = np.asarray(canvas, dtype=np.float32)
    y, x = np.indices(pixels.shape)
    mass = pixels.sum()
    dy = round(13.5 - (y * pixels).sum() / mass)
    dx = round(13.5 - (x * pixels).sum() / mass)
    centered = Image.new("L", (28, 28))
    centered.paste(canvas, (dx, dy))
    return np.asarray(centered, dtype=np.float32).reshape(1, 784) / 255.0


def predictions(model, x):
    return np.concatenate(
        [model.predict(x[i : i + 256]) for i in range(0, len(x), 256)]
    )


def train(args):
    np.random.seed(args.seed)
    (x, y), _ = load_mnist(ROOT / "data")
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
        for begin in range(0, len(order), 128):
            ids = order[begin : begin + 128]
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
                    (ROOT / "data/mnist.npz").read_bytes()
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
    _, (x, y) = load_mnist(ROOT / "data")
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


def recognize(model, image):
    pixels = preprocess_image(image)
    scores = model.predict(pixels)[0]
    return {
        "digit": int(scores.argmax()),
        "scores": scores.tolist(),
        "pixels": np.rint(pixels.reshape(28, 28) * 255).astype(int).tolist(),
        "note": "Class probabilities are not calibrated confidence; drawings differ from MNIST.",
    }


def build_server(model, port):
    """추론 HTTP 서버를 만든다. port=0 이면 빈 포트를 OS 가 고른다(테스트용).

    층이 forward 중간 상태를 self 에 저장하므로 모델은 **재진입 불가**다.
    ThreadingHTTPServer 는 요청마다 스레드를 만들기 때문에 추론 구간 전체를
    하나의 lock 으로 직렬화한다.
    """
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/":
                self.send_error(404)
                return
            data = (ROOT / "web/index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            if self.path != "/predict":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 1_000_000:
                    raise ValueError("invalid input size")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict) or not isinstance(
                    payload.get("image"), str
                ):
                    raise ValueError("image must be a base64 string")
                raw = base64.b64decode(payload["image"].split(",")[-1], validate=True)
                with Image.open(io.BytesIO(raw)) as im:
                    with lock:
                        result = recognize(model, im)
                status = 200
            except (ValueError, KeyError, OSError) as error:
                result = {"error": str(error)}
                status = 400
            data = json.dumps(result).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def serve(args):
    model, _ = load_model(args.model)
    server = build_server(model, args.port)
    print(f"http://127.0.0.1:{server.server_address[1]}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description="NumPy single-digit recognizer")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("train")
    p.add_argument("--output", type=Path, default=ROOT / ".artifacts/model.npz")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--hidden", nargs="+", type=int, default=[128, 64])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    p.set_defaults(run=train)
    p = commands.add_parser("evaluate")
    p.add_argument("--model", type=Path, default=REFERENCE)
    p.add_argument("--output")
    p.set_defaults(run=evaluate)
    p = commands.add_parser("predict")
    p.add_argument("image")
    p.add_argument("--model", type=Path, default=REFERENCE)
    p = commands.add_parser("serve")
    p.add_argument("--model", type=Path, default=REFERENCE)
    p.add_argument("--port", type=int, default=8765)
    p.set_defaults(run=serve)
    p = commands.add_parser("demo")
    p.add_argument("--model", type=Path, default=REFERENCE)
    args = parser.parse_args()
    try:
        if args.command in ("predict", "demo"):
            model, _ = load_model(args.model)
            image = (
                args.image
                if args.command == "predict"
                else ROOT / "examples/digit-7.png"
            )
            print(json.dumps(recognize(model, Image.open(image))))
        else:
            args.run(args)
    except (FileNotFoundError, ValueError) as error:
        parser.exit(
            2,
            f"{error}\nModel preparation: make train; use --model .artifacts/model.npz\n",
        )


if __name__ == "__main__":
    main()
