# ✍️ lrn-mnist

손글씨 숫자 하나를 0~9로 분류하는 NumPy 신경망입니다. 직접 구현한 MLP와 backward로 모델을 학습하고, 이미지 파일이나 브라우저에서 그린 숫자를 판별합니다.

[딥러닝 Wiki](https://docs.woonyong.com/wiki/deep-learning/) · [모델·혼동행렬·학습 조건](models/manifest.json)

## 데모

![lrn-mnist 데모: 캔버스에 그린 7과 3을 판별하고 28×28 입력과 클래스별 점수를 표시](docs/demo.gif)

`make demo` 후 `make serve` → http://127.0.0.1:8765 에서 캔버스에 7과 3을 마우스로 그리고 판별한 화면입니다. 제공 모델(`models/reference.npz`)의 실제 출력이며 두 경우 모두 해당 숫자가 100.0%로 표시됐습니다. 예시로 고른 두 장이므로 정확도를 뜻하지 않고, 점수는 보정된 confidence가 아닙니다. 모델 정확도는 [manifest](models/manifest.json)를 봅니다.

## 실행

Python 3.12와 [uv](https://docs.astral.sh/uv/getting-started/installation/)가 필요합니다. 의존성은 `requirements.lock`으로 고정합니다.

```sh
make setup
make demo
make serve
# 브라우저: http://127.0.0.1:8765
```

데모와 손그림 화면은 포함된 모델을 사용하므로 MNIST 다운로드나 재학습이 필요 없습니다. 예측 숫자·클래스별 점수와 실제 모델에 입력된 28×28 이미지를 확인합니다. 서버는 Ctrl-C로 종료합니다.

```sh
make test
.venv/bin/python src/application.py predict examples/digit-7.png
# 새 모델 학습은 별도 실행: 제공 모델을 덮어쓰지 않음
make train
.venv/bin/python src/application.py evaluate --model .artifacts/model.npz --output .artifacts/my-evaluation/metrics.json
```

## 구현과 설계

이미지 반전·crop·비율 유지·중심 정렬 → 28×28 입력 → MLP → 클래스별 점수로 이어집니다.

- [network.py](src/network.py)·[layers.py](src/layers.py): Affine, ReLU, Softmax, Cross Entropy와 backward, BatchNorm·Dropout, SGD·Adam.
- [application.py](src/application.py): 전처리·학습·CLI·모델 저장. weights와 BatchNorm 통계를 NumPy 배열로 저장하며 pickle을 사용하지 않습니다.
- [web/index.html](web/index.html): loopback에서 실행하는 손그림 화면. 빈 입력을 거절하고 전처리 결과를 보여 줍니다.

MNIST training 60,000개를 seed 42로 train 50,000 / validation 10,000으로 분리하고 validation으로 모델을 선택합니다. test 10,000개는 학습 중 선택에 사용하지 않습니다. `make test`는 수치 미분, optimizer, 저장 전후 예측, 데이터 분리와 입력 처리를 확인합니다.

## 현재 범위

이미지 한 장에 숫자 하나를 입력하는 MLP입니다. 손그림과 MNIST의 굵기·위치 차이로 오분류할 수 있으며 점수는 보정된 confidence가 아닙니다. CNN·OCR·여러 숫자 인식은 포함하지 않습니다.

학습·평가 시 데이터가 없으면 공개 Keras 배포의 `mnist.npz` 약 11 MiB를 내려받습니다. 제공 모델의 측정 정확도·데이터 해시·준비 비용은 [manifest](models/manifest.json)에 있습니다. 모델 복원은 추론용 weights·BN 통계까지이며 optimizer를 포함한 학습 재개는 지원하지 않습니다.

## 출처와 기여

`krafton-jungle/mnist-lab` 과제와 `Jungle-12-303/wk13_6_mnist` 팀 구현을 거친 `woonyong-kr/SW_AI-W13-mnist`에서 이어 받은 학습용 파생본이다. 기준 원본 revision은 `6a7e4511eb6b97b1762d32648d638c60b5ab9668`이다. 과제 제공물·팀 구현·이후 확장은 Git author와 diff로 구분하며, 기존 저작권 표시는 유지한다. 원본 주소의 공개 접근이 제한돼 있어 자료는 아래 이력 링크로 확인할 수 있다.

기존 신경망 구현 위에 데이터 분리, 모델 저장·복원, 이미지 CLI와 손그림 화면을 연결했다. 과제 설명과 이전 실험은 [정리 전 이력](https://github.com/woonyong-kr/lrn-mnist/tree/1c28de6983671e891faac453e392f68b354bdb53)에 남아 있다.
