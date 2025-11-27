"""
학습 설정 파일
모든 학습 하이퍼파라미터와 경로를 중앙에서 관리합니다.
"""

import os

# 프로젝트 루트 경로
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ========== 센서 차원 설정 ==========
# 이 값들을 변경하면 Projection Layer만 재학습하면 됩니다.
SENSOR_DIMS = {
    'visual': 14,      # YOLO 클래스 수 (확장 가능: 15, 20, 80 등)
    'audio': 17,       # YAMNet 17-class
    'pose': 51,        # 17 관절 × 3 좌표
    'spatial': 4,      # GPS 구역 수 (balcony, bedroom, kitchen, living_room)
    'time': 10         # 시간 특징 차원
}

# ========== 컨텍스트 인코더 설정 ==========
ENCODER_CONFIG = {
    'embed_dim': 64,        # 어텐션을 위한 공통 임베딩 차원
    'num_heads': 4,         # 어텐션 헤드 수
    'context_dim': 160,     # 출력 컨텍스트 차원 (GRU 입력)
}

# ========== GRU 모델 설정 ==========
GRU_CONFIG = {
    'num_zones': 4,         # 예측할 구역 수 (balcony, bedroom, kitchen, living_room)
    'context_dim': 160,     # AttentionEncoder 출력 차원과 일치해야 함
    'sequence_length': 30,  # 시퀀스 길이 (30 타임스텝)
}

# ========== 데이터 생성 설정 ==========
DATA_CONFIG = {
    'num_samples_per_scenario': 200,  # 시나리오당 샘플 수
    'noise_level': 0.1,               # 데이터 증강 노이즈 강도
    'train_split': 0.8,               # 훈련/검증 분할 비율
    'sequence_length': 30,            # 시퀀스 길이
}

# ========== AttentionEncoder 학습 설정 ==========
ENCODER_TRAINING = {
    'epochs': 150,              # 충분한 에포크 (Early Stopping으로 자동 종료)
    'batch_size': 256,          # 대규모 데이터셋용 (56K 샘플)
    'learning_rate': 0.001,     # Adam 기본값
    'validation_split': 0.2,
    'early_stopping_patience': 20,  # 충분히 여유있게
    'reduce_lr_patience': 8,    # LR 감소 전 대기
    'min_lr': 1e-7,             # 최소 학습률
}

# ========== GRU 학습 설정 ==========
GRU_TRAINING = {
    'epochs': 100,              # 충분한 에포크 (Early Stopping으로 자동 종료)
    'batch_size': 128,          # 시퀀스 데이터용 (메모리 고려)
    'learning_rate': 0.0005,    # GRU는 조금 낮게
    'early_stopping_patience': 15,  # 충분히 여유있게
    'reduce_lr_patience': 6,    # LR 감소 전 대기
    'min_lr': 1e-7,             # 최소 학습률
}

# ========== 경로 설정 ==========
PATHS = {
    # 데이터 경로
    'data_dir': os.path.join(PROJECT_ROOT, 'data'),
    'raw_features': os.path.join(PROJECT_ROOT, 'data', 'raw_features.npz'),
    'encoder_dataset': os.path.join(PROJECT_ROOT, 'data', 'encoder_dataset.npz'),
    'gru_dataset': os.path.join(PROJECT_ROOT, 'data', 'gru_dataset.npz'),

    # 모델 저장 경로
    'models_dir': os.path.join(PROJECT_ROOT, 'models'),
    'encoder_model': os.path.join(PROJECT_ROOT, 'models', 'attention_encoder.keras'),
    'gru_model': os.path.join(PROJECT_ROOT, 'models', 'gru', 'gru_model.keras'),

    # 결과 저장 경로
    'results_dir': os.path.join(PROJECT_ROOT, 'results'),
    'encoder_history': os.path.join(PROJECT_ROOT, 'results', 'encoder_training_history.png'),
    'gru_history': os.path.join(PROJECT_ROOT, 'results', 'gru_training_history.png'),
}

# 필요한 디렉토리 생성
for path_key in ['data_dir', 'models_dir', 'results_dir']:
    os.makedirs(PATHS[path_key], exist_ok=True)

# GRU 모델 디렉토리
os.makedirs(os.path.dirname(PATHS['gru_model']), exist_ok=True)


def print_config():
    """현재 설정을 출력합니다."""
    print("\n" + "=" * 70)
    print("학습 설정 (Training Configuration)")
    print("=" * 70)

    print("\n[센서 차원]")
    for key, value in SENSOR_DIMS.items():
        print(f"  {key:10s}: {value:3d} dim")

    print("\n[컨텍스트 인코더]")
    for key, value in ENCODER_CONFIG.items():
        print(f"  {key:15s}: {value}")

    print("\n[GRU 모델]")
    for key, value in GRU_CONFIG.items():
        print(f"  {key:15s}: {value}")

    print("\n[데이터 생성]")
    for key, value in DATA_CONFIG.items():
        print(f"  {key:25s}: {value}")

    print("\n[인코더 학습]")
    for key, value in ENCODER_TRAINING.items():
        print(f"  {key:25s}: {value}")

    print("\n[GRU 학습]")
    for key, value in GRU_TRAINING.items():
        print(f"  {key:25s}: {value}")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    print_config()
