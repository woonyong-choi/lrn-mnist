# -*- coding: utf-8 -*-
"""
MNIST 분류용 신경망 조립 모듈.

개별 layer를 OrderedDict에 쌓아 forward/backward 순서를 명확히 유지합니다.
"""

from collections import OrderedDict

import numpy as np

from activations import ReLU, Softmax
from layers import Affine, BatchNorm, Dropout
from losses import cross_entropy_gradient, cross_entropy_loss


class NeuralNetwork:
    """
    MNIST 분류용 신경망.
    입력 784 -> 은닉층(들) -> 출력 10 (Softmax).
    은닉층 구성: Affine -> BatchNorm -> ReLU -> Dropout (모두 필수)
    가중치 초기화: He 또는 Xavier 중 선택.
    """

    def __init__(
        self,
        hidden_sizes,
        use_batchnorm=True,
        use_dropout=True,
        dropout_ratio=0.5,
        batchnorm_momentum=0.9,
    ):
        """
        Args:
            hidden_sizes: 은닉층 뉴런 수 목록(예: [128, 64])
            use_batchnorm: 은닉층마다 BatchNorm을 넣을지 여부
            use_dropout: 은닉층마다 Dropout을 넣을지 여부
            dropout_ratio: Dropout에서 끌 뉴런 비율
            batchnorm_momentum: BatchNorm running 통계 이동평균 비율
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout
        self.dropout_ratio = dropout_ratio
        self.batchnorm_momentum = batchnorm_momentum
        self.params = {}
        self.hidden_sizes = list(hidden_sizes)
        layer_sizes = [784] + self.hidden_sizes + [10]
        for idx in range(1, len(layer_sizes)):
            fan_in = layer_sizes[idx - 1]
            fan_out = layer_sizes[idx]
            self.params[f"W{idx}"] = np.random.randn(fan_in, fan_out) * np.sqrt(
                2.0 / fan_in
            )
            self.params[f"b{idx}"] = np.zeros(fan_out)

            if use_batchnorm and idx < len(layer_sizes) - 1:
                self.params[f"gamma{idx}"] = np.ones(fan_out)
                self.params[f"beta{idx}"] = np.zeros(fan_out)

        self.layers = OrderedDict()
        # 학습되는 층마다 (층, 그 층의 gradient 속성 -> params 키)를 미리 묶어 둔다.
        # backward 가 매번 층 이름을 문자열로 파싱해 키를 되짚을 필요가 없다.
        self.gradient_sources = []
        for idx in range(1, len(layer_sizes)):
            affine = Affine(self.params[f"W{idx}"], self.params[f"b{idx}"])
            self.layers[f"Affine{idx}"] = affine
            self.gradient_sources.append((affine, {"dW": f"W{idx}", "db": f"b{idx}"}))

            if idx < len(layer_sizes) - 1:
                if use_batchnorm:
                    batchnorm = BatchNorm(
                        self.params[f"gamma{idx}"],
                        self.params[f"beta{idx}"],
                        momentum=batchnorm_momentum,
                    )
                    self.layers[f"BatchNorm{idx}"] = batchnorm
                    self.gradient_sources.append(
                        (batchnorm, {"dgamma": f"gamma{idx}", "dbeta": f"beta{idx}"})
                    )
                self.layers[f"ReLU{idx}"] = ReLU()
                if use_dropout:
                    self.layers[f"Dropout{idx}"] = Dropout(dropout_ratio)

        self.layers["Softmax"] = Softmax()
        self.grads = {key: np.zeros_like(value) for key, value in self.params.items()}

    def forward(self, x, train=True):
        """
        Args:
            x: (batch_size, 784) 정규화된 MNIST 이미지
            train: BatchNorm/Dropout의 학습 모드 여부

        Returns:
            (batch_size, 10) 각 숫자 클래스의 확률
        """
        out = x
        for layer in self.layers.values():
            if isinstance(layer, (BatchNorm, Dropout)):
                out = layer.forward(out, train=train)
            else:
                out = layer.forward(out)
        return out

    def backward(self, dout):
        """
        네트워크 전체 역전파를 수행하고 self.grads를 채웁니다.

        Args:
            dout: Softmax+CrossEntropy를 합친 출력층 gradient
        """
        for layer in reversed(self.layers.values()):
            dout = layer.backward(dout)

        for layer, attributes in self.gradient_sources:
            for attribute, key in attributes.items():
                self.grads[key] = getattr(layer, attribute)

        return dout

    def gradient(self, x, y):
        """
        학습 1회분 gradient를 계산하고 self.grads를 채웁니다.

        순전파로 loss를 구하고, softmax + cross entropy의 미분값을 시작점으로
        backward를 돌립니다.
        """
        y_pred = self.forward(x, train=True)
        loss = cross_entropy_loss(y_pred, y)
        dout = cross_entropy_gradient(y_pred, y)
        self.backward(dout)
        return loss

    def loss(self, x, y, train=True):
        """현재 모델의 예측 확률을 만든 뒤 cross entropy loss를 반환합니다."""
        y_pred = self.forward(x, train=train)
        return cross_entropy_loss(y_pred, y)

    def predict(self, x):
        """추론 모드로 확률을 예측합니다. BatchNorm/Dropout은 train=False로 동작합니다."""
        return self.forward(x, train=False)
