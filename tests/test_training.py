import numpy as np
import pytest
from layers import BatchNorm, Dropout
from optimizers import Adam, SGD


def test_batchnorm_backward_matches_numeric_gradient_and_inference_uses_running_stats():
    x = np.array([[1.0, 2.0], [3.0, -1.0], [2.0, 4.0]])
    gamma, beta = np.array([1.2, 0.8]), np.array([0.1, -0.2])
    layer = BatchNorm(gamma, beta)
    dout = np.array([[0.2, -0.5], [0.7, 0.3], [-0.1, 0.9]])
    layer.forward(x)
    dx = layer.backward(dout)
    for value, analytical in [
        (x, dx.copy()),
        (gamma, layer.dgamma.copy()),
        (beta, layer.dbeta.copy()),
    ]:
        for idx in np.ndindex(value.shape):
            old, eps = value[idx], 1e-5
            value[idx] = old + eps
            plus = np.sum(layer.forward(x) * dout)
            value[idx] = old - eps
            minus = np.sum(layer.forward(x) * dout)
            value[idx] = old
            assert analytical[idx] == pytest.approx(
                (plus - minus) / (2 * eps), abs=1e-7
            )
    mean, var = layer.running_mean.copy(), layer.running_var.copy()
    np.testing.assert_allclose(
        layer.forward(x, train=False), gamma * (x - mean) / np.sqrt(var + 1e-7) + beta
    )
    np.testing.assert_array_equal(layer.running_mean, mean)


def test_untrained_batchnorm_inference_is_not_amplified():
    """한 번도 학습하지 않은 층도 추론에서 입력 크기를 유지해야 한다.

    running_var 를 0 으로 초기화하면 1/sqrt(eps) = 3162 배로 증폭된다(회귀 방지).
    """
    x = np.linspace(-1, 1, 12).reshape(3, 4)
    out = BatchNorm(np.ones(4), np.zeros(4)).forward(x, train=False)
    np.testing.assert_allclose(out, x, atol=1e-3)


def test_dropout_train_backward_and_inference(monkeypatch):
    monkeypatch.setattr(np.random, "rand", lambda *shape: np.array([[0.2, 0.8]]))
    layer = Dropout(0.5)
    np.testing.assert_array_equal(layer.forward(np.array([[2.0, 4.0]])), [[0.0, 4.0]])
    np.testing.assert_array_equal(layer.backward(np.array([[3.0, 5.0]])), [[0.0, 5.0]])
    np.testing.assert_array_equal(
        layer.forward(np.array([[2.0, 4.0]]), train=False), [[1.0, 2.0]]
    )


@pytest.mark.parametrize("optimizer", [SGD, Adam])
def test_optimizer_constant_gradient_two_steps(optimizer):
    params = {"w": np.array([1.0, -1.0])}
    opt = optimizer(lr=0.1)
    for _ in range(2):
        opt.update(params, {"w": np.array([1.0, -1.0])})
    np.testing.assert_allclose(params["w"], [0.8, -0.8], atol=1e-7)
