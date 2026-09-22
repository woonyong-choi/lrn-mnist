# -*- coding: utf-8 -*-
"""MNIST 데이터 로드 유틸리티.

로컬에 파일이 없으면 공개 URL 에서 내려받고, **항상 sha256 을 검증한다.**
models/manifest.json 이 학습에 쓴 데이터의 digest 를 기록하고 있으므로,
검증이 없으면 "같은 데이터로 재현했다"는 주장이 성립하지 않는다.
"""

from hashlib import sha256
from pathlib import Path
import shutil
import urllib.request

import numpy as np

MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
# models/manifest.json 의 data.sha256 과 같은 값(11,490,434 바이트).
MNIST_SHA256 = "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"
DOWNLOAD_TIMEOUT = 60


def file_sha256(path):
    """큰 파일을 통째로 메모리에 올리지 않고 digest 를 계산한다."""
    digest = sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url, path, timeout=DOWNLOAD_TIMEOUT):
    """임시 파일에 받은 뒤 원자적으로 옮긴다.

    곧바로 목적지에 쓰면 중간에 끊긴 파일이 캐시로 남아 이후 실행에서
    계속 신뢰된다. 부분 파일은 .part 로 남고 목적지는 만들어지지 않는다.
    """
    partial = path.with_name(path.name + ".part")
    with urllib.request.urlopen(url, timeout=timeout) as response:
        with partial.open("wb") as f:
            shutil.copyfileobj(response, f)
    partial.replace(path)


def load_mnist(data_dir="data", expected_sha256=MNIST_SHA256):
    """MNIST 손글씨 숫자 데이터셋을 로드한다.

    Args:
        data_dir: mnist.npz 를 두는 폴더
        expected_sha256: 기대하는 digest. None 이면 검증을 건너뛴다.

    Returns:
        (x_train, y_train), (x_test, y_test)
        - x: (N, 784) float32, 0~1 정규화
        - y: (N,) uint8 레이블 0~9

    Raises:
        ValueError: 파일 digest 가 기대값과 다를 때(손상·중간 버전 교체)
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    local_path = data_dir / "mnist.npz"

    if not local_path.is_file():
        download(MNIST_URL, local_path)

    if expected_sha256 is not None:
        digest = file_sha256(local_path)
        if digest != expected_sha256:
            raise ValueError(
                f"{local_path} sha256 {digest} != {expected_sha256}. "
                "손상되었거나 다른 파일입니다. 지우고 다시 실행하세요."
            )

    with np.load(local_path, allow_pickle=False) as data:
        x_train = data["x_train"].astype(np.float32).reshape(-1, 784) / 255.0
        x_test = data["x_test"].astype(np.float32).reshape(-1, 784) / 255.0
        y_train = data["y_train"]
        y_test = data["y_test"]

    return (x_train, y_train), (x_test, y_test)
