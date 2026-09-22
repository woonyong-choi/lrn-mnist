"""실패 모드: 손상된 데이터·체크포인트와 동시 요청."""

import base64
import json
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import pytest

from application import build_server, load_model, recognize, ROOT
from data import file_sha256, load_mnist
from network import NeuralNetwork


def write_dataset(path, count=4):
    rng = np.random.default_rng(0)
    images = rng.integers(0, 256, size=(count, 28, 28), dtype=np.uint8)
    labels = rng.integers(0, 10, size=count).astype(np.uint8)
    np.savez(path, x_train=images, y_train=labels, x_test=images, y_test=labels)


def test_load_mnist_rejects_a_file_with_an_unexpected_digest(tmp_path):
    """digest 검증이 없으면 '같은 데이터로 재현했다'가 성립하지 않는다."""
    write_dataset(tmp_path / "mnist.npz")
    digest = file_sha256(tmp_path / "mnist.npz")

    (x_train, _), (x_test, y_test) = load_mnist(tmp_path, expected_sha256=digest)
    assert x_train.shape == (4, 784) and 0 <= x_train.min() and x_train.max() <= 1
    assert x_test.shape == (4, 784) and y_test.shape == (4,)

    with pytest.raises(ValueError, match="sha256"):
        load_mnist(tmp_path, expected_sha256="0" * 64)


def test_interrupted_download_is_not_cached(tmp_path, monkeypatch):
    """중간에 끊긴 다운로드는 .part 로 남고 목적지 파일이 되지 않는다."""
    import data as data_module

    def fail(url, timeout=None):
        raise OSError("connection reset")

    monkeypatch.setattr(data_module.urllib.request, "urlopen", fail)
    with pytest.raises(OSError):
        load_mnist(tmp_path)
    assert not (tmp_path / "mnist.npz").exists()


def test_load_model_rejects_corrupted_batchnorm_statistics(tmp_path):
    from application import save_model

    np.random.seed(2)
    model = NeuralNetwork(hidden_sizes=[8])
    path = tmp_path / "model.npz"
    save_model(model, path, {})
    with np.load(path, allow_pickle=False) as data:
        arrays = dict(data.items())

    for change in [
        {"BatchNorm1_var": np.full(8, -1.0)},
        {"BatchNorm1_mean": np.full(8, np.nan)},
        {"BatchNorm1_var": np.ones(3)},
    ]:
        broken = tmp_path / "broken.npz"
        np.savez_compressed(broken, **{**arrays, **change})
        with pytest.raises(ValueError, match="invalid batchnorm statistics"):
            load_model(broken)


def post_digit(url, digit):
    image = (ROOT / f"examples/digit-{digit}.png").read_bytes()
    payload = json.dumps({"image": base64.b64encode(image).decode()}).encode()
    request = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())


class CountingModel:
    """추론이 겹치는지 세는 대역 모델. 겹치면 동시 진입 수가 1을 넘는다."""

    def __init__(self):
        self.inside = 0
        self.peak = 0
        self.guard = threading.Lock()

    def predict(self, pixels):
        with self.guard:
            self.inside += 1
            self.peak = max(self.peak, self.inside)
        time.sleep(0.02)
        with self.guard:
            self.inside -= 1
        return np.full((1, 10), 0.1)


def test_server_serializes_inference():
    """층이 forward 중간 상태를 self 에 저장해 재진입 불가이므로,
    서버는 추론을 반드시 직렬화해야 한다. lock 을 빼면 peak 가 1을 넘는다."""
    model = CountingModel()
    server = build_server(model, 0)
    url = f"http://127.0.0.1:{server.server_address[1]}/predict"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda d: post_digit(url, d), range(8)))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert model.peak == 1


def test_concurrent_requests_agree_with_single_threaded_inference():
    """통합: 그림 -> HTTP -> 전처리 -> 모델 -> JSON 응답이 10개 숫자 모두에서
    단일 스레드 결과와 같다."""
    from PIL import Image

    model, _ = load_model(ROOT / "models/reference.npz")
    digits = list(range(10))
    expected = {}
    for d in digits:
        with Image.open(ROOT / f"examples/digit-{d}.png") as image:
            expected[d] = recognize(model, image)["scores"]

    server = build_server(model, 0)
    url = f"http://127.0.0.1:{server.server_address[1]}/predict"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with ThreadPoolExecutor(max_workers=10) as pool:
            results = list(pool.map(lambda d: post_digit(url, d), digits * 3))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    for digit, result in zip(digits * 3, results, strict=True):
        np.testing.assert_allclose(result["scores"], expected[digit], atol=1e-12)


def test_server_rejects_malformed_requests():
    model, _ = load_model(ROOT / "models/reference.npz")
    server = build_server(model, 0)
    base = f"http://127.0.0.1:{server.server_address[1]}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        for payload, expected in [
            (b"{}", 400),
            (b'{"image": 3}', 400),
            (b'{"image": "not-base64!!"}', 400),
            (b"", 400),
        ]:
            request = urllib.request.Request(base + "/predict", data=payload or b" ")
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(request, timeout=30)
            assert error.value.code == expected
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(base + "/missing", timeout=30)
        assert error.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
