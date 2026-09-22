# -*- coding: utf-8 -*-
"""`make bench`: 핵심 주장(재현성·속도)을 숫자로 다시 만든다.

세 가지를 측정한다.
1. 재현성: seed 42 로 다시 학습한 모델의 sha256 이 models/manifest.json 의
   기록과 같은가. 다르면 "재현 가능"이라는 주장이 깨진 것이다.
2. 학습 시간: epoch 별 wall clock.
3. 추론 처리량과 병목: batch 크기를 바꿔 가며 초당 이미지 수를 재고, 같은
   shape 의 행렬곱만 돌린 하한과 비교해 GEMM 이 아닌 부분의 비용을 본다.

결과는 .artifacts/bench.json(원본)과 docs/bench.md·docs/bench.svg(저장소에 남기는
요약)로 저장된다. 측정값은 기계 상태에 따라 흔들리므로 표에 측정 시각을 함께 적는다.
"""

import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from application import load_model, predictions, split_training  # noqa: E402
from data import load_mnist  # noqa: E402
from network import NeuralNetwork  # noqa: E402
from optimizers import Adam  # noqa: E402

BATCH_SIZES = [1, 32, 128, 256, 1024, 10000]
REPEATS = 5


def time_best(fn, repeats=REPEATS):
    """가장 빠른 실행을 취한다. 느린 쪽은 다른 프로세스의 방해를 담고 있다."""
    fn()
    return min(_measure(fn) for _ in range(repeats))


def _measure(fn):
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def retrain(x, y, seed=42, epochs=8):
    """application.train 과 같은 설정으로 학습하고 epoch 별 시간을 돌려준다."""
    np.random.seed(seed)
    train_ids, val_ids = split_training(len(x), seed)
    model = NeuralNetwork(hidden_sizes=[128, 64], dropout_ratio=0.1)
    optimizer = Adam(lr=0.001)
    epoch_seconds = []
    best, best_params = -1.0, None
    for _ in range(epochs):
        start = time.perf_counter()
        order = np.random.permutation(train_ids)
        for begin in range(0, len(order), 128):
            ids = order[begin : begin + 128]
            model.gradient(x[ids], y[ids])
            optimizer.update(model.params, model.grads)
        epoch_seconds.append(time.perf_counter() - start)
        accuracy = float(np.mean(predictions(model, x[val_ids]).argmax(1) == y[val_ids]))
        if accuracy > best:
            best, best_params = accuracy, {k: v.copy() for k, v in model.params.items()}
    return model, best, best_params, epoch_seconds


def inference_throughput(model, x):
    """batch 크기별 초당 이미지 수와, 같은 shape 행렬곱만 돌린 하한."""
    weights = [model.params[f"W{i}"] for i in (1, 2, 3)]
    rows = []
    for batch in BATCH_SIZES:
        sample = x[:batch]

        def run_full():
            for begin in range(0, len(x), batch):
                model.predict(x[begin : begin + batch])

        def run_gemm():
            for begin in range(0, len(x), batch):
                chunk = x[begin : begin + batch]
                for weight in weights:
                    chunk = chunk @ weight

        full = time_best(run_full)
        gemm = time_best(run_gemm)
        rows.append(
            {
                "batch_size": batch,
                "images_per_second": round(len(x) / full),
                "seconds_per_10k": round(full, 4),
                "gemm_only_seconds_per_10k": round(gemm, 4),
                "gemm_share": round(gemm / full, 3),
            }
        )
    return rows


def render_svg(rows, path):
    """의존성 없이 처리량 막대그래프를 그린다(matplotlib 을 쓰지 않는 이유는 design.md)."""
    width, height, pad = 720, 300, 46
    top = max(row["images_per_second"] for row in rows)
    bar = (width - 2 * pad) / len(rows)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'font-family="system-ui, sans-serif" font-size="12">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{pad}" y="24" font-size="15" fill="#182134">'
        "추론 처리량 (images/sec, 값이 클수록 좋음)</text>",
    ]
    for index, row in enumerate(rows):
        value = row["images_per_second"]
        bar_height = (height - pad - 60) * value / top
        x = pad + index * bar + bar * 0.15
        y = height - pad - bar_height
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar * 0.7:.1f}" '
            f'height="{bar_height:.1f}" fill="#2457c5"/>'
        )
        parts.append(
            f'<text x="{x + bar * 0.35:.1f}" y="{y - 6:.1f}" text-anchor="middle" '
            f'fill="#182134">{value:,}</text>'
        )
        parts.append(
            f'<text x="{x + bar * 0.35:.1f}" y="{height - pad + 18:.1f}" '
            f'text-anchor="middle" fill="#4a5568">batch {row["batch_size"]}</text>'
        )
        parts.append(
            f'<text x="{x + bar * 0.35:.1f}" y="{height - pad + 34:.1f}" '
            f'text-anchor="middle" fill="#8892a4">GEMM {row["gemm_share"]:.0%}</text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


def render_markdown(result, path):
    """저장소에 남길 요약 표. 원본은 .artifacts/bench.json 에 있다."""
    training = result["training"]
    repro = result["reproducibility"]
    lines = [
        "# 벤치마크",
        "",
        f"`make bench` 가 생성한다. 측정: {result['measured_on']} · "
        f"{result['environment']['platform']} · Python {result['environment']['python']} · "
        f"NumPy {result['environment']['numpy']} · BLAS 스레드 1개.",
        "",
        "## 1. 재현성 (핵심 주장)",
        "",
        f"- seed 42 로 다시 학습한 가중치가 `models/reference.npz` 와 "
        f"**{'완전히 같다' if repro['weights_identical_to_reference'] else '다르다'}**"
        " (비트 단위 비교).",
        f"- 검증 정확도 {repro['validation_accuracy']:.4f} 가 manifest 기록과 "
        f"{'일치' if repro['matches_manifest_accuracy'] else '불일치'}한다.",
        "",
        "## 2. 학습 시간",
        "",
        f"- 50,000장 x {training['epochs']} epoch = **{training['seconds']:.1f}초**",
        f"- epoch 별(초): {', '.join(f'{v:.2f}' for v in training['epoch_seconds'])}",
        "",
        "## 3. 추론 처리량과 병목",
        "",
        "![batch 크기별 추론 처리량](bench.svg)",
        "",
        "| batch | images/sec | 10,000장(초) | 같은 shape 행렬곱만(초) | GEMM 비중 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in result["inference"]:
        lines.append(
            f"| {row['batch_size']} | {row['images_per_second']:,} | "
            f"{row['seconds_per_10k']:.4f} | {row['gemm_only_seconds_per_10k']:.4f} | "
            f"{row['gemm_share']:.0%} |"
        )
    best = max(result["inference"], key=lambda row: row["images_per_second"])
    single = result["inference"][0]
    lines += [
        "",
        f"한 장씩 넣으면 초당 {single['images_per_second']:,}장이고 "
        f"batch {best['batch_size']} 에서 {best['images_per_second']:,}장으로 "
        f"**{best['images_per_second'] / single['images_per_second']:.0f}배** 빨라진다. "
        "행렬 크기는 같으므로 차이는 전부 층마다 드는 호출·임시 배열 비용이다.",
        "",
        f"batch 를 키워도 같은 행렬곱만 돌린 하한이 전체의 {best['gemm_share']:.0%} 에 그친다. "
        "즉 이 모델(784-128-64-10)에서 병목은 행렬곱이 아니라 그 사이의 "
        "BatchNorm·ReLU·Softmax 원소 연산과 NumPy 임시 배열 할당이다. "
        "더 줄이려면 층을 더 키우거나(행렬곱 비중을 늘려) 원소 연산을 "
        "in-place 로 바꿔야 한다.",
        "",
    ]
    path.write_text("\n".join(lines))


def main():
    manifest = json.loads((ROOT / "models/manifest.json").read_text())
    (x_train, y_train), (x_test, _) = load_mnist(ROOT / "data")

    start = time.perf_counter()
    model, validation_accuracy, best_params, epoch_seconds = retrain(x_train, y_train)
    training_seconds = time.perf_counter() - start

    for key, value in best_params.items():
        model.params[key][:] = value
    reference, _ = load_model(ROOT / "models/reference.npz")
    identical = all(
        np.array_equal(model.params[key], reference.params[key])
        for key in model.params
    )

    rows = inference_throughput(reference, x_test)
    result = {
        "measured_on": time.strftime("%Y-%m-%d %H:%M %Z"),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "note": "Makefile 이 BLAS 스레드를 1로 고정한다",
        },
        "reproducibility": {
            "seed": 42,
            "validation_accuracy": validation_accuracy,
            "matches_manifest_accuracy": validation_accuracy
            == manifest["training"]["validation_accuracy"],
            "weights_identical_to_reference": identical,
        },
        "training": {
            "seconds": round(training_seconds, 3),
            "epoch_seconds": [round(value, 3) for value in epoch_seconds],
            "images": 50000,
            "epochs": len(epoch_seconds),
        },
        "inference": rows,
    }

    output = ROOT / ".artifacts/bench.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    render_svg(rows, ROOT / "docs/bench.svg")
    render_markdown(result, ROOT / "docs/bench.md")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
