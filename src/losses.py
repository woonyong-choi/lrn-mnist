# -*- coding: utf-8 -*-
"""Cross Entropy 손실과 Softmax 를 합친 출력층 gradient.

두 함수 모두 계약이 같다: `y_pred` 는 (batch, 10) softmax 확률,
`y_true` 는 (batch,) 정수 레이블이다. one-hot 이나 1차원 입력은 받지 않는다.
"""

import numpy as np


def cross_entropy_loss(y_pred, y_true):
    """배치 평균 cross entropy. log(0) 을 막기 위해 확률을 1e-7 로 clip 한다."""
    batch_size = y_pred.shape[0]
    clipped = np.clip(y_pred, 1e-7, 1.0)
    return -np.sum(np.log(clipped[np.arange(batch_size), y_true])) / batch_size


def cross_entropy_gradient(y_pred, y_true):
    """Softmax + Cross Entropy 를 함께 미분한 출력층 gradient.

    y_pred 가 softmax 결과일 때 dL/dlogits 는 (y_pred - one-hot) 이다.
    배치 평균 loss 를 쓰므로 batch_size 로 나눈다.
    """
    batch_size = y_pred.shape[0]
    dout = y_pred.copy()
    dout[np.arange(batch_size), y_true] -= 1
    return dout / batch_size
