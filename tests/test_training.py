"""층·optimizer 의 수학적 계약을 검증한다.

전략: 한 사례를 고정 seed 로 못 박는 대신, **무작위 shape·값 여러 벌에 대해 성질이
항상 성립하는지**(property) 확인한다. seed 목록만 고정해 실패를 재현 가능하게 둔다.
hypothesis 를 쓰지 않은 이유는 docs/design.md "테스트 전략" 참조.
"""

import numpy as np
import pytest

from activations import Softmax
from layers import BatchNorm, Dropout
from losses import cross_entropy_gradient, cross_entropy_loss
from optimizers import Adam, SGD

CASES = [(3, 2), (8, 5), (17, 1), (2, 64)]


def numeric_gradient(fn, value, eps=1e-5):
    """value 를 제자리에서 흔들어 중앙차분으로 d(fn)/d(value) 를 구한다."""
    grad = np.zeros_like(value)
    for idx in np.ndindex(value.shape):
        old = value[idx]
        value[idx] = old + eps
        plus = fn()
        value[idx] = old - eps
        minus = fn()
        value[idx] = old
        grad[idx] = (plus - minus) / (2 * eps)
    return grad


@pytest.mark.parametrize("shape", CASES)
def test_batchnorm_backward_matches_numeric_gradient(shape):
    """dx·dgamma·dbeta 가 모두 유한 차분과 일치한다(무작위 입력 4벌)."""
    rng = np.random.default_rng(shape[0])
    x = rng.normal(size=shape)
    gamma, beta = rng.normal(size=shape[1]) + 1.0, rng.normal(size=shape[1])
    layer = BatchNorm(gamma, beta)
    dout = rng.normal(size=shape)

    layer.forward(x)
    analytical = {
        "x": layer.backward(dout).copy(),
        "gamma": layer.dgamma.copy(),
        "beta": layer.dbeta.copy(),
    }
    # 유한 차분은 forward 를 다시 부르므로 running 통계를 오염시키지 않도록 사본 층을 쓴다.
    probe = BatchNorm(gamma, beta)
    for name, value in [("x", x), ("gamma", gamma), ("beta", beta)]:
        expected = numeric_gradient(lambda: np.sum(probe.forward(x) * dout), value)
        # batch=2 케이스는 std 가 작아 std**-3 항이 커지므로 상대 오차로 비교한다.
        np.testing.assert_allclose(analytical[name], expected, atol=1e-6, rtol=1e-4)


@pytest.mark.parametrize("shape", CASES)
def test_batchnorm_forward_normalizes_each_feature(shape):
    """학습 모드 출력은 feature 마다 평균 beta, 표준편차 |gamma| 를 갖는다."""
    rng = np.random.default_rng(shape[1])
    x = rng.normal(loc=5.0, scale=3.0, size=shape)
    gamma, beta = rng.normal(size=shape[1]), rng.normal(size=shape[1])
    out = BatchNorm(gamma, beta).forward(x)
    np.testing.assert_allclose(out.mean(axis=0), beta, atol=1e-6)
    np.testing.assert_allclose(out.std(axis=0), np.abs(gamma), atol=1e-3)


def test_batchnorm_inference_uses_running_statistics_and_does_not_update_them():
    """추론 모드는 배치 통계 대신 running 통계를 쓰고, 그 값을 갱신하지 않는다."""
    rng = np.random.default_rng(0)
    gamma, beta = np.ones(4), np.zeros(4)
    layer = BatchNorm(gamma, beta)
    for _ in range(3):
        layer.forward(rng.normal(size=(16, 4)))
    mean, var = layer.running_mean.copy(), layer.running_var.copy()

    x = rng.normal(size=(5, 4))
    np.testing.assert_allclose(
        layer.forward(x, train=False),
        gamma * (x - mean) / np.sqrt(var + layer.eps) + beta,
    )
    np.testing.assert_array_equal(layer.running_mean, mean)
    np.testing.assert_array_equal(layer.running_var, var)


def test_untrained_batchnorm_inference_is_not_amplified():
    """한 번도 학습하지 않은 층도 추론에서 입력 크기를 유지해야 한다.

    running_var 를 0 으로 초기화하면 1/sqrt(eps) = 3162 배로 증폭된다(회귀 방지).
    """
    x = np.linspace(-1, 1, 12).reshape(3, 4)
    out = BatchNorm(np.ones(4), np.zeros(4)).forward(x, train=False)
    np.testing.assert_allclose(out, x, atol=1e-3)


def test_dropout_mask_gates_forward_and_backward(monkeypatch):
    """꺼진 뉴런은 출력도 gradient 도 0 이고, 추론은 (1-p) 로 보정한다."""
    monkeypatch.setattr(np.random, "rand", lambda *shape: np.array([[0.2, 0.8]]))
    layer = Dropout(0.5)
    np.testing.assert_array_equal(layer.forward(np.array([[2.0, 4.0]])), [[0.0, 4.0]])
    np.testing.assert_array_equal(layer.backward(np.array([[3.0, 5.0]])), [[0.0, 5.0]])
    np.testing.assert_array_equal(
        layer.forward(np.array([[2.0, 4.0]]), train=False), [[1.0, 2.0]]
    )


@pytest.mark.parametrize("ratio", [0.1, 0.5, 0.9])
def test_dropout_training_and_inference_agree_on_average(ratio):
    """vanilla dropout 의 계약: 학습 출력의 기댓값 == 추론 출력."""
    np.random.seed(3)
    x = np.full((4096, 16), 2.0)
    layer = Dropout(ratio)
    train_mean = np.mean([layer.forward(x).mean() for _ in range(8)])
    np.testing.assert_allclose(train_mean, layer.forward(x, train=False).mean(), rtol=0.02)


@pytest.mark.parametrize("shape", CASES)
def test_softmax_rows_are_a_distribution_and_shift_invariant(shape):
    """행 합이 1이고, 상수를 더해도 결과가 같다(최댓값 빼기의 목적)."""
    rng = np.random.default_rng(shape[0] + shape[1])
    x = rng.normal(scale=4.0, size=shape)
    out = Softmax().forward(x)
    np.testing.assert_allclose(out.sum(axis=1), 1.0)
    assert (out > 0).all()
    np.testing.assert_allclose(Softmax().forward(x + 1000.0), out, atol=1e-12)


def test_softmax_does_not_overflow_on_extreme_logits():
    """exp overflow 로 nan 이 나오지 않는다."""
    out = Softmax().forward(np.array([[1e5, 0.0, -1e5], [-1e5, -1e5, -1e5]]))
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out.sum(axis=1), 1.0)
    np.testing.assert_allclose(out[0], [1.0, 0.0, 0.0])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_softmax_cross_entropy_gradient_matches_finite_difference(seed):
    """합쳐서 미분한 dL/dlogits(= p - onehot)가 실제 loss 의 기울기와 같다."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(scale=2.0, size=(6, 10))
    labels = rng.integers(0, 10, size=6)

    analytical = cross_entropy_gradient(Softmax().forward(logits), labels)
    expected = numeric_gradient(
        lambda: cross_entropy_loss(Softmax().forward(logits), labels), logits
    )
    np.testing.assert_allclose(analytical, expected, atol=1e-8)


@pytest.mark.parametrize("optimizer", [SGD, Adam])
def test_optimizer_takes_lr_sized_steps_on_a_constant_gradient(optimizer):
    """회귀 테스트: Adam 의 bias correction 이 빠지면 첫 스텝이 3.16배 커진다."""
    params = {"w": np.array([1.0, -1.0])}
    opt = optimizer(lr=0.1)
    for _ in range(2):
        opt.update(params, {"w": np.array([1.0, -1.0])})
    np.testing.assert_allclose(params["w"], [0.8, -0.8], atol=1e-7)


def test_adam_normalizes_step_size_across_parameter_scales():
    """Adam 의 존재 이유: gradient 크기가 1000배 달라도 스텝은 같다(SGD 는 비례)."""
    grads = {"big": np.array([1.0]), "small": np.array([1e-3])}
    adam = {"big": np.zeros(1), "small": np.zeros(1)}
    sgd = {"big": np.zeros(1), "small": np.zeros(1)}
    Adam(lr=0.1).update(adam, grads)
    SGD(lr=0.1).update(sgd, grads)

    assert abs(adam["big"][0] / adam["small"][0]) == pytest.approx(1.0, abs=1e-3)
    assert abs(sgd["big"][0] / sgd["small"][0]) == pytest.approx(1000.0, rel=1e-6)


def test_optimizer_updates_parameters_in_place():
    """불변식: optimizer 는 배열을 교체하지 않고 제자리 갱신한다.

    층이 params dict 의 배열 객체를 그대로 참조하므로(network.py), 재바인딩하면
    가중치가 갱신돼도 층은 옛 배열을 계속 쓰게 된다.
    """
    for optimizer in (SGD, Adam):
        params = {"w": np.ones(3)}
        shared = params["w"]
        optimizer(lr=0.1).update(params, {"w": np.ones(3)})
        assert params["w"] is shared
        assert not np.array_equal(shared, np.ones(3))
