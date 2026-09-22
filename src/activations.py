# -*- coding: utf-8 -*-
"""은닉층 활성화(ReLU)와 출력층 Softmax."""

import numpy as np


class ReLU:
    """
    ReLU(Rectified Linear Unit) 활성화 함수.

    은닉층에서 음수 값은 0으로 막고, 양수 값은 그대로 통과시킵니다.
    forward에서 만든 mask는 backward 때 "어느 위치로 gradient를 흘릴지" 결정하는 데 사용됩니다.
    """

    def forward(self, x):
        """
        Args:
            x: 임의 shape의 입력 배열

        Returns:
            x와 같은 shape. x > 0인 위치만 원래 값을 유지합니다.
        """
        self.mask = x <= 0
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        """
        Args:
            dout: 다음 층에서 넘어온 gradient

        Returns:
            ReLU 입력 x에 대한 gradient. forward 때 x <= 0이었던 위치는 0입니다.
        """
        dx = dout.copy()
        dx[self.mask] = 0
        return dx


class Softmax:
    """
    Softmax 출력층.

    각 샘플의 로짓(logit)을 클래스별 확률로 바꿉니다.
    exp 계산 전에 행별 최댓값을 빼면 큰 숫자에서 overflow가 나는 것을 줄일 수 있습니다.
    """

    def forward(self, x):
        """
        Args:
            x: (batch_size, num_classes) 로짓

        Returns:
            (batch_size, num_classes) 확률. 각 행의 합은 1입니다.
        """
        shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shifted)
        self.out = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.out

    def backward(self, dout):
        """
        항등 함수다. 출력층 gradient 는 Softmax 와 Cross Entropy 를 합쳐
        losses.cross_entropy_gradient 가 만들기 때문에(수치적으로 안정적이고 식이 짧다),
        이 층은 역전파에서 받은 값을 그대로 넘긴다.
        """
        return dout
