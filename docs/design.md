# 설계 노트

이 문서는 "무엇을 만들었나"가 아니라 **왜 이 형태인가, 무엇을 버렸나, 무엇이 아직 틀렸나**를
적는다. 구조와 실행 방법은 [README](../README.md)에 있다.

## 1. 왜 이 형태로 구현했나

### Softmax 와 Cross Entropy 를 합쳐서 미분한다

`Softmax.backward` 는 항등 함수다([activations.py](../src/activations.py)). 출력층 gradient 는
`losses.cross_entropy_gradient` 가 `p - onehot` 한 줄로 만든다.

- Softmax 의 야코비안은 (batch, 10, 10) 이고, Cross Entropy 의 미분은 `-1/p` 라서 `p` 가 작을 때
  두 큰 값을 곱해 상쇄시키는 계산이 된다. 합쳐서 미분하면 상쇄가 **해석적으로 먼저 끝나** 있어
  gradient 경로에는 clip 이 필요 없다(loss 값 계산에는 `log(0)` 방지용 clip 이 남아 있다).
- 대가: 층 하나가 "아무것도 하지 않는" 모양이 되어 읽는 사람이 버그로 오해할 수 있다. 그래서
  `Softmax.backward` 의 docstring 에 이유를 적고, 합친 gradient 가 실제 loss 의 유한 차분과
  같은지 테스트로 고정했다(`test_softmax_cross_entropy_gradient_matches_finite_difference`).

### BatchNorm backward 를 dvar·dmu 로 분해해서 쓴다

한 줄로 접은 식(`dx = gamma/(N*std) * (N*dout - sum(dout) - x_norm*sum(dout*x_norm))`)이 더 짧고
빠르다. 그런데도 계산 그래프를 그대로 따라가는 형태를 골랐다([layers.py](../src/layers.py)).

- 이 저장소의 목적은 역전파를 **직접 유도할 수 있음을 보이는 것**이고, 분해형은 틀렸을 때 어느
  항이 틀렸는지 유한 차분으로 항별 추적이 가능하다. 접은 식은 값만 틀리고 위치를 알려주지 않는다.
- 대가: 임시 배열이 늘어난다. 실제로 [벤치마크](bench.md)에서 batch 를 키워도 행렬곱이
  절반을 넘지 못하는 이유 중 하나가 이런 원소 연산·임시 배열이다.

### params dict 와 층이 같은 배열 객체를 공유한다

`NeuralNetwork` 는 `self.params["W1"]` 배열을 그대로 `Affine` 에 넘긴다. optimizer 는 dict 만
보고 갱신하면 되고, 층을 다시 만들 필요가 없다.

- 대가: **"optimizer 는 in-place 로 갱신한다"가 불변식이 된다.** `params[key] = params[key] - lr*g`
  처럼 재바인딩하면 dict 값만 바뀌고 층은 옛 배열을 계속 써서, 예외 없이 조용히 학습이 멈춘다.
  실제로 팀 과제 시절 Adam 에서 비슷한 버그를 겪었다. 지금은
  `test_optimizer_updates_parameters_in_place` 가 이 불변식을 고정한다.

### 무작위성을 한 군데로 모은다

`np.random.seed(seed)` 하나로 가중치 초기화·배치 셔플·Dropout mask 가 모두 결정된다. 덕분에
같은 seed 로 다시 학습하면 **가중치가 비트 단위로 같다**(`make bench` 가 매번 확인한다). 이
저장소에서 가장 강한 주장이 이것이고, 그래서 검증을 문서가 아니라 스크립트에 넣었다.

- 대가: 전역 RNG 라서 라이브러리처럼 쓰면 호출자와 상태를 공유한다. 단일 실행 파일이라
  받아들였다. 검증 분할(`split_training`)만은 전역과 무관해야 해서 `default_rng` 를 따로 쓴다.

## 2. 버린 대안과 이유

### (1) inverted dropout — 표준이지만 쓰지 않았다

지금은 vanilla dropout 이다(학습에서 mask, 추론에서 `1-p` 곱). 표준은 inverted dropout 으로,
학습에서 `1/(1-p)` 로 미리 나눠 추론을 그대로 둔다. inverted 가 나은 점은 분명하다: 추론 경로에
dropout 의 흔적이 남지 않아 `p` 를 바꿔도 저장된 모델의 추론 코드가 그대로다.

바꾸지 않은 이유는 **출하된 `models/reference.npz` 의 예측을 바꾸지 않기 위해서**다. 이 저장소가
주장하는 것은 "97.80% 를 냈다"가 아니라 "그 97.80% 가 재현된다"이고, 지금 dropout 규약을 바꾸면
문서·manifest·sha256 이 전부 한 번에 움직인다. 이득(추론 코드 1줄)과 비교해 값이 맞지 않는다.
**다음에 재학습할 일이 생기면 그때 함께 바꾼다** — 알고 남겨 둔 빚이다.

### (2) `src` 를 패키지로 만들고 상대 import — 실행 한 줄을 지키려고 버렸다

`src/` 안의 모듈은 서로를 `from layers import Affine` 처럼 최상위 이름으로 import 한다. 패키지로
만들면(`from .layers import`) `python -m src.application` 로만 실행할 수 있다. README 의
`python src/application.py predict examples/digit-7.png` 한 줄이 복사해서 바로 도는 것이
이 저장소에서는 더 중요하다고 판단했다.

- 대가: `tests/conftest.py` 와 `bench.py` 가 `sys.path` 를 건드린다. 두 군데뿐이고 이유가
  주석에 있지만, 모듈이 더 늘면 패키지로 바꾸는 게 맞다.

### (3) 도구: hypothesis 와 matplotlib

- **hypothesis**: 성질 테스트는 쓰지만 라이브러리는 넣지 않았다. 부동소수 배열은 shrink 가
  의미 있는 최소 반례를 거의 못 만들고, 유한 차분 비교는 입력 범위를 좁게 제한해야 해서
  전략 정의가 테스트보다 길어진다. 대신 **shape·seed 목록을 고정한 무작위 스윕**으로
  같은 성질을 확인하고, 실패는 seed 로 그대로 재현된다.
- **matplotlib**: [bench.svg](bench.svg) 는 [bench.py](../bench.py)가 직접 SVG 를 쓴다.
  막대그래프 하나 때문에 30 MB 짜리 의존성과 백엔드 설정을 CI 에 들이지 않는다.

## 3. 알려진 한계와 다음 병목

| 항목 | 현재 | 다음에 할 일 |
|---|---|---|
| 추론 병목 | 직전 측정에서 행렬곱이 전체의 22~43%. 나머지는 원소 연산과 NumPy 임시 배열([bench.md](bench.md)) | 원소 연산을 in-place(`out=`)로 바꾸고 BatchNorm 추론을 `gamma/std`·`beta - mean*gamma/std` 로 미리 접기 |
| 한 장 추론 | batch 1 은 가장 빠른 batch 대비 10배 이상 느리다. 층당 고정 비용이 지배한다 | 단건 요청이 많아지면 위와 같은 접기가 그대로 효과를 낸다 |
| BatchNorm eps | `1e-7` (표준은 `1e-5`). 분산이 아주 작은 feature 에서 backward 의 `std**-3` 항이 커진다. batch 2 에서 유한 차분과의 상대 오차가 3e-6 까지 벌어져, 성질 테스트는 rtol 1e-4 로 비교한다 | 재학습 시 `1e-5` 로 올린다. 지금 바꾸면 출하 모델이 달라진다 |
| Dropout | vanilla. 위 (1) 참조 | 재학습 시 inverted 로 |
| 학습 재개 | 저장하는 것은 추론용 가중치와 BatchNorm 통계뿐이고 optimizer 상태(m, v, t)는 버린다 | 중단·재개가 필요해지면 같은 `.npz` 에 `opt_*` 로 추가 |
| 메모리 | 학습 데이터 전체를 메모리에 올린다(60,000 × 784 float32 = 188 MB) | MNIST 규모에서는 문제가 아니다. 더 큰 데이터면 배치 단위 로딩이 필요하다 |
| 서버 | 층이 forward 중간 상태를 `self` 에 저장해 **재진입 불가**다. 그래서 추론 전체를 lock 하나로 직렬화한다(`build_server`). 처리량 상한은 단일 스레드 추론 속도다 | 동시성이 필요하면 모델을 요청마다 복제하지 말고, 층에서 상태를 걷어내 함수형 forward 로 바꿔야 한다 |
| 손그림 | MNIST 와 굵기·필압 분포가 달라 오분류가 난다. 점수는 보정된 confidence 가 아니다 | 이 저장소 범위 밖이다. 고치려면 학습 데이터 증강이 필요하고, 그러면 "MNIST 재현"이라는 주장이 흐려진다 |
