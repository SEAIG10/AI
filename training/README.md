# 모듈러 학습 파이프라인 (Modular Training Pipeline)

이 폴더는 AttentionContextEncoder와 GRU 모델을 별도로 학습시키기 위한 모듈러 학습 파이프라인을 제공합니다.

## 왜 모듈러 학습인가?

**기존 방식 (End-to-End)**:
- 센서 특징 → AttentionEncoder → GRU를 한번에 학습
- YOLO 클래스 수 변경 시 전체 모델 재학습 필요
- 디버깅 및 튜닝이 어려움

**모듈러 방식 (현재)**:
- AttentionEncoder와 GRU를 별도로 학습
- Projection Layer가 차원 변화를 흡수
- YOLO 클래스 수 변경 시 Projection Layer만 재학습
- 각 모듈을 독립적으로 튜닝 가능
- Transfer Learning 지원

## 파일 구조

```
training/
├── __init__.py                 # 패키지 초기화
├── README.md                   # 이 파일
├── config.py                   # 학습 설정 (하이퍼파라미터, 경로 등)
├── data_generator.py           # 합성 센서 데이터 생성
├── train_encoder.py            # AttentionEncoder 학습
├── train_gru.py                # GRU 모델 학습
└── train_pipeline.py           # 전체 파이프라인 통합 실행
```

## 학습 흐름

### 1단계: 합성 데이터 생성
```bash
python training/data_generator.py
```

**출력**: `data/raw_features.npz`
- visual: (N, 30, 14) - YOLO 14-class
- audio: (N, 30, 17) - YAMNet 17-class
- pose: (N, 30, 51) - 17 관절 × 3
- spatial: (N, 30, 7) - GPS 구역
- time: (N, 30, 10) - 시간 특징
- labels: (N, 7) - 청소 필요 구역

### 2단계: AttentionEncoder 학습
```bash
python training/train_encoder.py
```

**목표**: 다중 모달 센서 특징 → 160차원 컨텍스트 벡터 변환
**출력**:
- `models/attention_encoder.keras` (학습된 인코더)
- `results/encoder_training_history.png` (학습 그래프)

**학습 메커니즘**:
- AttentionEncoder + 임시 예측 헤드를 함께 학습
- 예측 헤드는 인코더가 의미 있는 특징을 학습하도록 유도
- 학습 후 AttentionEncoder만 저장, 예측 헤드는 버림

### 3단계: GRU 모델 학습
```bash
python training/train_gru.py
```

**목표**: 160차원 컨텍스트 벡터 시퀀스 → 청소 필요 구역 예측
**출력**:
- `models/gru/gru_model.keras` (학습된 GRU)
- `results/gru_training_history.png` (학습 그래프)

**학습 메커니즘**:
- 사전 학습된 AttentionEncoder 로드 (Frozen)
- 원본 센서 특징 → AttentionEncoder → 160차원 컨텍스트
- 컨텍스트 시퀀스로 GRU 학습

### 전체 파이프라인 실행
```bash
python training/train_pipeline.py
```

위의 3단계를 자동으로 순차 실행합니다.

## 설정 변경

`training/config.py`에서 모든 하이퍼파라미터를 변경할 수 있습니다:

```python
# 센서 차원 변경 (예: YOLO 클래스 수 증가)
SENSOR_DIMS = {
    'visual': 20,  # 14 → 20으로 변경
    ...
}

# 학습 하이퍼파라미터
ENCODER_TRAINING = {
    'epochs': 100,
    'batch_size': 64,
    'learning_rate': 0.001,
    ...
}
```

## 센서 차원 변경 시 재학습 전략

### 시나리오 1: YOLO 클래스 수 변경 (14 → 20)

**옵션 A: Projection Layer만 재학습 (권장)**
1. `config.py`에서 `SENSOR_DIMS['visual'] = 20` 변경
2. `train_encoder.py` 실행 (빠름, 약 10분)
3. `train_gru.py`는 재학습 불필요 (AttentionEncoder 출력은 여전히 160차원)

**옵션 B: 전체 재학습**
1. 모든 설정 변경
2. `train_pipeline.py` 실행

### 시나리오 2: 새로운 환경에서 Transfer Learning

**방법**:
1. 기존 AttentionEncoder 가중치 로드
2. Projection Layer만 freeze 해제
3. 새 데이터로 Fine-tuning

```python
encoder = tf.keras.models.load_model('models/attention_encoder.keras')
# Projection Layer만 학습 가능하게 설정
for layer in encoder.layers:
    if 'projection' not in layer.name:
        layer.trainable = False
```

## 데이터 증강

`data_generator.py`는 자동으로 데이터 증강을 수행합니다:

```python
# 시나리오당 200개의 변형 샘플 생성
DATA_CONFIG = {
    'num_samples_per_scenario': 200,
    'noise_level': 0.1,  # 노이즈 강도
}
```

**증강 기법**:
- 객체 신뢰도 변경
- 무작위 객체 제거
- 시간적 변동

## 모델 크기

**AttentionEncoder**:
- 파라미터 수: ~50K
- 모델 크기: ~0.5MB (FP32), ~0.13MB (INT8)
- 추론 시간: ~50ms (CPU)

**GRU 모델**:
- Base Layer: ~42.8K params (공유)
- Head Layer: ~0.6K params (개인화)
- 모델 크기: ~0.5MB

**총 크기**: ~1MB (온디바이스 환경에 적합)

## 학습 모니터링

학습 중 TensorBoard를 사용하여 모니터링할 수 있습니다:

```bash
# 향후 추가 예정
tensorboard --logdir=results/logs
```

## 문제 해결

### 에러: AttentionEncoder를 찾을 수 없습니다

```
FileNotFoundError: AttentionEncoder를 찾을 수 없습니다: models/attention_encoder.keras
```

**해결책**: 먼저 `train_encoder.py`를 실행하세요.

### 에러: 메모리 부족

**해결책**: `config.py`에서 배치 크기를 줄이세요:
```python
ENCODER_TRAINING['batch_size'] = 32  # 64 → 32
GRU_TRAINING['batch_size'] = 16      # 32 → 16
```

### 에러: 학습이 수렴하지 않음

**해결책**: 학습률을 조정하거나 데이터 증강을 늘리세요:
```python
ENCODER_TRAINING['learning_rate'] = 0.0001  # 0.001 → 0.0001
DATA_CONFIG['num_samples_per_scenario'] = 300  # 200 → 300
```

## 다음 단계

학습이 완료되면:

1. **실시간 시스템에서 사용**:
   ```python
   # realtime/predictor.py에서 사용
   encoder = tf.keras.models.load_model('models/attention_encoder.keras')
   gru_model = tf.keras.models.load_model('models/gru/gru_model.keras')
   ```

2. **Webots 시뮬레이션에서 테스트**:
   ```bash
   # Webots에서 worlds/complete_apartment.wbt 실행
   ```

3. **FedPer 연합학습 적용**:
   - Base Layer (GRU): 여러 가정에서 공유
   - Head Layer: 각 가정별로 개인화

## 참고 자료

- **AttentionContextEncoder**: `src/context_fusion/attention_context_encoder.py`
- **GRU 모델**: `src/model/gru_model.py`
- **시나리오 생성기**: `src/dataset/scenario_generator.py`
- **프로젝트 README**: `../README.md`
