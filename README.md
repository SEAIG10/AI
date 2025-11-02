# LOCUS - Personalized Robot Cleaning System

FedPer 기반 개인화 로봇 청소 시스템 (Webots 시뮬레이션)

## Project Structure

```
SE_G10/
├── config/                      # 시스템 설정 파일 (존 정의 등)
├── controllers/                 # Webots 로봇 컨트롤러
│   └── robot_controller/        # 메인 로봇 제어 로직 및 YOLOv8 모델
├── data/                        # 학습 데이터셋 및 컨텍스트 DB
├── libraries/                   # 외부 라이브러리
├── models/                      # 학습된 GRU 모델 저장소
├── plugins/                     # Webots 플러그인 (physics, remote_controls 등)
├── protos/                      # Webots PROTO 정의 파일
├── results/                     # 학습 결과 (그래프, 메트릭 등)
├── src/                         # 메인 소스코드
│   ├── audio_recognition/       # FR2: Yamnet 기반 오디오 인식
│   ├── context_fusion/          # FR2: 멀티모달 컨텍스트 융합 및 벡터 인코딩
│   ├── dataset/                 # FR3: 시나리오 기반 Mock 데이터 생성
│   ├── model/                   # FR3: FedPer 기반 GRU 모델
│   ├── spatial_mapping/         # FR1: 의미론적 공간 매핑
│   └── train_gru.py             # FR3: GRU 모델 학습 스크립트
├── tests/                       # 유닛 테스트 및 통합 테스트
├── worlds/                      # Webots 시뮬레이션 월드 파일
└── venv/                        # Python 가상환경
```

## Functional Requirements

- **FR1**: Semantic Spatial Mapping (의미론적 공간 매핑)
- **FR2**: Multimodal Context Awareness (멀티모달 컨텍스트 인식)
- **FR3**: Sequential Pattern Learning (GRU 기반 청소 필요 예측)
- **FR4**: Personalized Federated Learning (FedPer 연합학습)

## Quick Start

```bash
# 1. 가상환경 활성화
source venv/bin/activate

# 2. GRU 모델 학습
python src/train_gru.py

# 3. Webots 시뮬레이션 실행
# Webots에서 worlds/complete_apartment.wbt 열기
```

## Model Architecture

**FedPer GRU Model**:
- Base Layer (공유): GRU(64) → GRU(32) [42.8K params]
- Head Layer (개인화): Dense(16) → Dense(7) [0.6K params]
- Input: (30, 108) - 30 timesteps of 108-dim context vectors
- Output: (7,) - Pollution probability for 7 semantic zones

## Technologies

- Python 3.11
- TensorFlow/Keras (GRU model)
- Webots (Robot simulation)
- YOLOv8 (Object detection)
- Yamnet (Audio recognition)
- SQLite (Context database)
