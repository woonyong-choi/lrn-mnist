# -*- coding: utf-8 -*-
"""신경망 layer 모음. 입력은 모두 (batch_size, feature_dim) 2차원이다."""

import numpy as np


class Affine:
    """
    완전연결층(Fully Connected Layer).

    수식은 y = xW + b 입니다.
    MNIST에서는 784개 픽셀 입력을 은닉층/출력층 차원으로 선형 변환하는 역할을 합니다.
    """

    def __init__(self, W, b):
        """가중치 W와 편향 b를 외부 params dict와 같은 배열 객체로 공유합니다."""
        self.W = W
        self.b = b

    def forward(self, x):
        """
        Args:
            x: (batch_size, input_dim)

        Returns:
            (batch_size, output_dim)
        """
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout):
        """
        Args:
            dout: (batch_size, output_dim)

        Returns:
            dx: (batch_size, input_dim)

        Side effects:
            self.dW, self.db에 optimizer가 사용할 gradient를 저장합니다.
        """
        self.dW = self.x.T @ dout
        self.db = np.sum(dout, axis=0)
        return dout @ self.W.T


class BatchNorm:
    """
    Batch Normalization.

    미니배치 단위로 각 feature의 평균과 분산을 맞춰 학습을 안정화합니다.
    train=True일 때는 현재 배치 통계를 쓰고, 추론 때는 누적 running_mean/running_var를 사용합니다.
    """

    def __init__(self, gamma, beta, momentum=0.9):
        """
        Args:
            gamma: 정규화된 값을 다시 scale하는 학습 파라미터
            beta: 정규화된 값에 더하는 shift 학습 파라미터
            momentum: running_mean/running_var 이동평균 비율
        """
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.running_mean = np.zeros_like(beta)
        # 분산의 초깃값은 1이다. 0으로 두면 학습 전 추론이 1/sqrt(eps) = 3162배로
        # 증폭되고, 이동평균이 0에서 출발해 초기 running_var 가 실제보다 작아진다.
        self.running_var = np.ones_like(beta)
        self.eps = 1e-7

    def forward(self, x, train=True):
        """
        Args:
            x: (batch_size, feature_dim)
            train: True면 배치 통계, False면 running 통계 사용

        Returns:
            정규화 후 gamma, beta가 적용된 배열
        """
        if train:
            mu = np.mean(x, axis=0)
            var = np.var(x, axis=0)
            self.x_centered = x - mu
            self.std = np.sqrt(var + self.eps)
            self.x_norm = self.x_centered / self.std

            self.running_mean = (
                self.momentum * self.running_mean + (1 - self.momentum) * mu
            )
            self.running_var = (
                self.momentum * self.running_var + (1 - self.momentum) * var
            )
        else:
            self.x_centered = x - self.running_mean
            self.std = np.sqrt(self.running_var + self.eps)
            self.x_norm = self.x_centered / self.std

        return self.gamma * self.x_norm + self.beta

    def backward(self, dout):
        """
        BatchNorm 입력 x, scale gamma, shift beta에 대한 gradient를 계산합니다.

        Args:
            dout: 다음 층에서 넘어온 gradient

        Returns:
            dx: BatchNorm 입력 x에 대한 gradient
        """
        batch_size = dout.shape[0]

        self.dbeta = np.sum(dout, axis=0)
        self.dgamma = np.sum(self.x_norm * dout, axis=0)

        dx_norm = self.gamma * dout
        dvar = np.sum(
            dx_norm * self.x_centered * -0.5 * (self.std**-3), axis=0
        )
        dmu = np.sum(dx_norm * -1 / self.std, axis=0)
        dmu += dvar * np.mean(-2 * self.x_centered, axis=0)
        dx = dx_norm / self.std
        dx += dvar * 2 * self.x_centered / batch_size
        dx += dmu / batch_size
        return dx


class Dropout:
    """
    Dropout.

    학습 중 일부 뉴런 출력을 무작위로 0으로 만들어 과적합을 줄입니다.
    이 구현은 추론 시 출력에 (1 - drop_ratio)를 곱하는 기본 dropout 방식을 사용합니다.
    """

    def __init__(self, drop_ratio=0.5):
        """Args: drop_ratio: 학습 중 0으로 만들 뉴런 비율."""
        self.drop_ratio = drop_ratio

    def forward(self, x, train=True):
        """
        Args:
            x: 입력 배열
            train: True면 무작위 mask 적용, False면 평균적인 출력 크기로 scale
        """
        if train:
            self.mask = np.random.rand(*x.shape) > self.drop_ratio
            return x * self.mask

        return x * (1 - self.drop_ratio)

    def backward(self, dout):
        """forward에서 꺼졌던 뉴런 위치에는 gradient도 흘리지 않습니다."""
        return dout * self.mask
