"""CLI 배선과 학습·평가 루프. MNIST 다운로드 없이 합성 데이터로 돈다."""

import json

import numpy as np
import pytest

import training
from application import build_parser, main
from data import file_sha256


def fake_dataset(count=320):
    """레이블이 픽셀에서 실제로 예측 가능한 합성 데이터(학습이 진행되는지 보려고)."""
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 10, count)
    x = rng.random((count, 784)).astype(np.float32) * 0.1
    x[np.arange(count), labels * 70] += 1.0
    return x, labels


@pytest.fixture
def offline(monkeypatch, tmp_path):
    x, y = fake_dataset()
    monkeypatch.setattr(training, "load_mnist", lambda *a, **k: ((x, y), (x, y)))
    (tmp_path / "mnist.npz").write_bytes(b"synthetic")
    monkeypatch.setattr(training, "DATA_DIR", tmp_path)
    return tmp_path


def test_demo_and_predict_share_one_entry_point():
    """demo 는 예시 이미지를 미리 채운 predict 다(특례 분기가 아니다)."""
    demo = build_parser().parse_args(["demo"])
    predict = build_parser().parse_args(["predict", "examples/digit-3.png"])
    assert demo.run is predict.run
    assert demo.image.name == "digit-7.png"


def test_predict_prints_one_json_line(capsys):
    main_args = ["predict", "examples/digit-3.png"]
    build_parser().parse_args(main_args)
    import sys

    sys.argv = ["application.py"] + main_args
    main()
    result = json.loads(capsys.readouterr().out)
    assert result["digit"] == 3 and len(result["scores"]) == 10
    assert len(result["pixels"]) == 28


def test_missing_model_exits_with_guidance(monkeypatch, capsys, tmp_path):
    import sys

    sys.argv = ["application.py", "predict", "examples/digit-3.png", "--model", str(tmp_path / "none.npz")]
    with pytest.raises(SystemExit) as exit_info:
        main()
    assert exit_info.value.code == 2
    assert "make train" in capsys.readouterr().err


def test_training_selects_on_validation_and_never_reads_test(offline, tmp_path, capsys):
    """핵심 주장: 저장되는 모델은 validation 정확도가 가장 높았던 epoch 의 것이다."""
    output = tmp_path / "model.npz"
    args = build_parser().parse_args(
        ["train", "--epochs", "3", "--hidden", "16", "--output", str(output)]
    )
    training.train(args)
    rows = [json.loads(line) for line in capsys.readouterr().out.strip().split("\n")]

    history = rows[:-1]
    assert len(history) == 3
    assert history[-1]["train_loss"] < history[0]["train_loss"]
    best = max(history, key=lambda row: row["validation_accuracy"])
    assert rows[-1]["best_validation_accuracy"] == best["validation_accuracy"]

    from checkpoint import load_model

    _, metadata = load_model(output)
    assert metadata["epoch"] == best["epoch"]
    assert metadata["validation_accuracy"] == best["validation_accuracy"]
    assert metadata["training_count"] + metadata["validation_count"] == 320
    assert metadata["data_sha256"] == file_sha256(offline / "mnist.npz")
    assert "test" in metadata["selection"]


def test_evaluate_writes_a_confusion_matrix_that_sums_to_the_test_set(
    offline, tmp_path, capsys
):
    model_path = tmp_path / "model.npz"
    training.train(
        build_parser().parse_args(
            ["train", "--epochs", "1", "--hidden", "16", "--output", str(model_path)]
        )
    )
    metrics = tmp_path / "out/metrics.json"
    training.evaluate(
        build_parser().parse_args(
            ["evaluate", "--model", str(model_path), "--output", str(metrics)]
        )
    )
    capsys.readouterr()

    result = json.loads(metrics.read_text())
    assert np.array(result["confusion_matrix"]).sum() == result["test_count"] == 320
    assert np.trace(result["confusion_matrix"]) / 320 == result["test_accuracy"]
    assert result["model_sha256"] == file_sha256(model_path)
    assert len(result["errors"]) <= 20
