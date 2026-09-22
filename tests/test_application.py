"""학습·저장·전처리 경로의 불변식과 실패 경로를 검증한다."""

import numpy as np
import pytest
from PIL import Image, ImageDraw

from checkpoint import load_model, save_model
from images import preprocess_image
from network import NeuralNetwork
from training import predictions, split_training


def train_briefly(seed, steps=6):
    """학습 경로(init → gradient → Adam → Dropout RNG)를 짧게 한 번 돌린다."""
    from optimizers import Adam

    np.random.seed(seed)
    model = NeuralNetwork(hidden_sizes=[16, 8], dropout_ratio=0.3)
    optimizer = Adam(lr=0.01)
    rng = np.random.default_rng(0)
    x, y = rng.random((steps * 4, 784)), rng.integers(0, 10, steps * 4)
    for step in range(steps):
        ids = slice(step * 4, step * 4 + 4)
        model.gradient(x[ids], y[ids])
        optimizer.update(model.params, model.grads)
    return model


def test_training_is_bitwise_reproducible_for_a_fixed_seed():
    """핵심 주장의 근거: 같은 seed 면 가중치·BatchNorm 통계가 비트 단위로 같다."""
    first, second = train_briefly(42), train_briefly(42)
    for key, value in first.params.items():
        np.testing.assert_array_equal(value, second.params[key], err_msg=key)
    np.testing.assert_array_equal(
        first.layers["BatchNorm1"].running_var, second.layers["BatchNorm1"].running_var
    )
    # seed 가 다르면 실제로 달라야 한다(위 비교가 공허하지 않다는 확인).
    assert not np.array_equal(first.params["W1"], train_briefly(43).params["W1"])


@pytest.mark.parametrize("use_batchnorm", [False, True])
def test_full_network_gradient_matches_finite_difference(use_batchnorm):
    """BatchNorm 유무 양쪽에서 역전파 전체가 유한 차분과 일치한다."""
    np.random.seed(9)
    model = NeuralNetwork(
        hidden_sizes=[4], use_batchnorm=use_batchnorm, use_dropout=False
    )
    x = np.random.random((3, 784))
    y = np.array([1, 3, 5])
    model.gradient(x, y)

    keys = ["W1", "b1", "W2", "b2"] + (["gamma1", "beta1"] if use_batchnorm else [])
    for key in keys:
        index = np.ndindex(model.params[key].shape).__next__()
        analytical = float(model.grads[key][index])
        old, eps = model.params[key][index], 1e-5
        model.params[key][index] = old + eps
        plus = model.loss(x, y)
        model.params[key][index] = old - eps
        minus = model.loss(x, y)
        model.params[key][index] = old
        assert analytical == pytest.approx((plus - minus) / (2 * eps), abs=2e-6), key


def test_checkpoint_preserves_batchnorm_and_predictions(tmp_path):
    np.random.seed(7)
    model = NeuralNetwork(
        hidden_sizes=[8], use_batchnorm=True, use_dropout=True, dropout_ratio=0.2
    )
    x = np.random.random((6, 784))
    model.forward(x, train=True)
    before = model.predict(x).copy()
    path = tmp_path / "model.npz"
    save_model(model, path, {"seed": 7})
    restored, metadata = load_model(path)
    np.testing.assert_array_equal(restored.predict(x), before)
    assert metadata["seed"] == 7


def test_load_model_rejects_corrupted_checkpoints(tmp_path):
    """실패 경로: 값이 깨졌거나 shape 이 다르거나 format 이 낯선 파일은 거절한다."""
    np.random.seed(1)
    model = NeuralNetwork(hidden_sizes=[8])
    path = tmp_path / "model.npz"
    save_model(model, path, {})

    def rewrite(**changes):
        target = tmp_path / "broken.npz"
        with np.load(path, allow_pickle=False) as data:
            arrays = dict(data.items())
        arrays.update(changes)
        np.savez_compressed(target, **arrays)
        return target

    with pytest.raises(ValueError, match="invalid model weights"):
        load_model(rewrite(param_W1=np.full_like(model.params["W1"], np.nan)))
    with pytest.raises(ValueError, match="invalid model weights"):
        load_model(rewrite(param_W1=np.zeros((784, 7))))
    with pytest.raises(ValueError, match="unsupported model format"):
        load_model(rewrite(manifest=np.array('{"format": 2, "config": {}}')))


def test_batched_inference_equals_single_image_inference():
    """predictions() 의 256장 묶음이 결과를 바꾸지 않는다(추론은 배치 독립적)."""
    np.random.seed(5)
    model = NeuralNetwork(hidden_sizes=[8])
    model.forward(np.random.random((32, 784)), train=True)
    x = np.random.random((300, 784))
    np.testing.assert_allclose(
        predictions(model, x),
        np.concatenate([model.predict(row.reshape(1, -1)) for row in x]),
        atol=1e-12,
    )


def test_split_is_fixed_disjoint_and_has_no_test_dependency():
    a, b = split_training(60000, seed=42)
    c, d = split_training(60000, seed=42)
    np.testing.assert_array_equal(a, c)
    np.testing.assert_array_equal(b, d)
    assert len(a) == 50000 and len(b) == 10000
    assert not set(a) & set(b) and len(set(a) | set(b)) == 60000
    with pytest.raises(ValueError, match="at least two"):
        split_training(1)


def draw_digit(size, position, width=10):
    image = Image.new("L", size, 0)
    ImageDraw.Draw(image).line(position, fill=255, width=width)
    return image


def test_preprocessing_rejects_blank_and_oversized_input():
    with pytest.raises(ValueError, match="blank"):
        preprocess_image(Image.new("L", (100, 100), 0))
    with pytest.raises(ValueError, match="blank"):
        preprocess_image(Image.new("L", (100, 100), 255))
    with pytest.raises(ValueError, match="4096"):
        preprocess_image(Image.new("L", (5000, 10), 0))


def test_preprocessing_is_invariant_to_polarity_and_position():
    """전처리의 계약: 배경색과 그린 위치가 달라도 같은 28x28 입력이 된다."""
    image = draw_digit((100, 100), (50, 10, 50, 90))
    pixels = preprocess_image(image)
    assert pixels.shape == (1, 784) and 0 <= pixels.min() and pixels.max() <= 1

    inverted = Image.fromarray(255 - np.asarray(image))
    np.testing.assert_array_equal(pixels, preprocess_image(inverted))
    moved = draw_digit((100, 100), (18, 5, 18, 85))
    np.testing.assert_array_equal(pixels, preprocess_image(moved))


def test_preprocessing_fits_the_digit_into_the_mnist_20_pixel_box():
    """MNIST 규약: 긴 변 20px, 28x28 캔버스 중앙(무게중심 기준) 정렬."""
    pixels = preprocess_image(draw_digit((400, 400), (200, 40, 200, 360), width=40))
    grid = pixels.reshape(28, 28)
    ys, xs = np.nonzero(grid > 0)
    assert ys.max() - ys.min() + 1 <= 20 and xs.max() - xs.min() + 1 <= 20
    y, x = np.indices(grid.shape)
    assert abs((y * grid).sum() / grid.sum() - 13.5) <= 0.5
    assert abs((x * grid).sum() / grid.sum() - 13.5) <= 0.5
