"""
GRU 모델 학습 스크립트 (FedPer Structure)

현실적인 일상 루틴 데이터셋으로 GRU 모델을 학습시킵니다.
학습 목표: 160차원 context 시퀀스 (30 timesteps) → 4개 구역 오염도 예측 (regression)

FedPer 구조:
- Base Layer (Shared): AttentionEncoder (99dim → 160dim)  # 사전 학습됨, frozen
- Head Layer (Personalized): GRU (160dim seq → 4-zone prediction)  # 이번에 학습
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks
import matplotlib.pyplot as plt

# 프로젝트 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model.gru_model import FedPerGRUModel
from src.context_fusion.attention_context_encoder import AttentionContextEncoder
from training.config import GRU_CONFIG, GRU_TRAINING, PATHS


def create_sequences_from_features(features_dict, labels, encoder, sequence_length=30):
    """
    딕셔너리 형태의 features를 AttentionEncoder를 거쳐 시퀀스로 변환합니다.

    Args:
        features_dict: {
            'time': (N, 10),
            'spatial': (N, 4),
            'visual': (N, 14),
            'audio': (N, 17),
            'pose': (N, 51)
        }
        labels: (N, 4)
        encoder: 사전 학습된 AttentionEncoder
        sequence_length: 시퀀스 길이

    Returns:
        X_seq: (N-sequence_length, sequence_length, 160)
        y_seq: (N-sequence_length, 4)
    """
    print("  [1/3] AttentionEncoder를 통과시켜 context vectors 생성 중...")

    # AttentionEncoder를 거쳐 160dim context vectors 생성
    context_vectors = encoder.predict(features_dict, batch_size=512, verbose=0)
    print(f"       Context vectors shape: {context_vectors.shape}")

    print("  [2/3] Context vectors를 시퀀스로 변환 중...")
    X_sequences = []
    y_sequences = []

    for i in range(len(context_vectors) - sequence_length):
        X_sequences.append(context_vectors[i:i+sequence_length])
        y_sequences.append(labels[i+sequence_length])  # 마지막 타임스텝의 라벨

    X_seq = np.array(X_sequences)
    y_seq = np.array(y_sequences)

    print(f"  [3/3] 생성된 시퀀스 shape: {X_seq.shape}")

    return X_seq, y_seq


class GRUTrainer:
    """GRU 모델 학습 클래스 (FedPer Head Layer)"""

    def __init__(self, encoder_path=None, num_zones=4, context_dim=160):
        """
        Args:
            encoder_path: 사전 학습된 AttentionEncoder 경로
            num_zones: 구역 개수 (balcony, bedroom, kitchen, living_room)
            context_dim: Context 벡터 차원 (AttentionEncoder 출력)
        """
        self.num_zones = num_zones
        self.context_dim = context_dim
        self.gru_model = None

        # AttentionEncoder 로드 (Frozen Base Layer)
        if encoder_path is None:
            encoder_path = PATHS['encoder_model']

        print("\n" + "=" * 70)
        print("사전 학습된 AttentionEncoder 로드 중...")
        print("=" * 70)

        if not os.path.exists(encoder_path):
            raise FileNotFoundError(
                f"AttentionEncoder를 찾을 수 없습니다: {encoder_path}\n"
                f"먼저 'python training/train_encoder.py'를 실행하세요."
            )

        self.encoder = tf.keras.models.load_model(
            encoder_path,
            custom_objects={'AttentionContextEncoder': AttentionContextEncoder}
        )
        print(f"✅ 로드 완료: {encoder_path}\n")

        # Encoder를 frozen (학습 안 함)
        self.encoder.trainable = False
        print("🔒 AttentionEncoder frozen (Base Layer - 학습 안 함)\n")

    def build_gru_model(self, sequence_length=30):
        """GRU 모델을 구축합니다 (Head Layer)."""
        print("\n" + "=" * 70)
        print("GRU 모델 구축 (FedPer Head Layer)")
        print("=" * 70 + "\n")

        self.gru_model = FedPerGRUModel(
            num_zones=self.num_zones,
            context_dim=self.context_dim  # 160차원 context 입력
        )

        # 더미 데이터로 모델 빌드
        dummy_input = tf.random.normal((1, sequence_length, self.context_dim))
        _ = self.gru_model.model(dummy_input)

        print("[GRU 모델 아키텍처]")
        self.gru_model.summary()
        print("\n" + "=" * 70 + "\n")

    def compile_model(self):
        """GRU 모델을 컴파일합니다 (Regression 용)."""
        self.gru_model.compile_model(
            learning_rate=GRU_TRAINING['learning_rate'],
            loss='mse',  # Regression: Mean Squared Error
            metrics=['mae', 'mse']  # Mean Absolute Error, MSE
        )
        print("GRU 모델 컴파일 완료 (Regression)\n")

    def train(self, X_train, y_train, X_val, y_val):
        """
        GRU 모델을 학습시킵니다.

        Args:
            X_train: (N_train, 30, 160) - Context vector sequences
            y_train: (N_train, 4)
            X_val: (N_val, 30, 160)
            y_val: (N_val, 4)

        Returns:
            history: 학습 기록
        """
        print("\n" + "=" * 70)
        print("GRU 모델 학습 시작 (FedPer Head Layer)")
        print("=" * 70 + "\n")

        print(f"훈련 샘플: {X_train.shape[0]:,}개")
        print(f"검증 샘플: {X_val.shape[0]:,}개")
        print(f"입력 shape: {X_train.shape}")
        print(f"출력 shape: {y_train.shape}\n")

        # 콜백 설정
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=GRU_TRAINING['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=GRU_TRAINING['reduce_lr_patience'],
                min_lr=GRU_TRAINING['min_lr'],
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                filepath=PATHS['gru_model'],
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]

        # 학습
        history = self.gru_model.train(
            X_train, y_train,
            X_val, y_val,
            epochs=GRU_TRAINING['epochs'],
            batch_size=GRU_TRAINING['batch_size'],
            callbacks=callback_list,
            verbose=1
        )

        print("\n" + "=" * 70)
        print("학습 완료!")
        print("=" * 70 + "\n")

        return history

    def evaluate(self, X_val, y_val, zone_names):
        """
        GRU 모델을 평가합니다.

        Args:
            X_val: 검증 입력
            y_val: 검증 라벨
            zone_names: 구역 이름 리스트
        """
        print("\n" + "=" * 70)
        print("GRU 모델 평가")
        print("=" * 70 + "\n")

        # 예측
        y_pred = self.gru_model.predict(X_val)

        # 구역별 평가
        print("구역별 성능 (Regression):")
        print("-" * 70)
        for i, zone in enumerate(zone_names):
            mae = np.mean(np.abs(y_val[:, i] - y_pred[:, i]))
            mse = np.mean((y_val[:, i] - y_pred[:, i]) ** 2)
            rmse = np.sqrt(mse)

            print(f"{zone:15s}: MAE={mae:.4f}, MSE={mse:.4f}, RMSE={rmse:.4f}")

        # 전체 평가
        overall_mae = np.mean(np.abs(y_val - y_pred))
        overall_rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))

        print("-" * 70)
        print(f"{'Overall':15s}: MAE={overall_mae:.4f}, RMSE={overall_rmse:.4f}")
        print("=" * 70 + "\n")

    def save_model(self, save_path=None):
        """
        학습된 GRU 모델을 저장합니다.

        Args:
            save_path: 저장 경로 (기본값: config)
        """
        if save_path is None:
            save_path = PATHS['gru_model']

        self.gru_model.save(save_path)
        print(f"✅ GRU 모델이 저장되었습니다: {save_path}")

        # 모델 크기 정보
        file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
        print(f"  파일 크기: {file_size_mb:.2f} MB")


def plot_training_history(history, save_path=None):
    """
    학습 기록을 시각화합니다.

    Args:
        history: Keras 학습 기록
        save_path: 저장 경로
    """
    if save_path is None:
        save_path = PATHS['gru_history']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 손실
    axes[0].plot(history.history['loss'], label='Train Loss (MSE)')
    axes[0].plot(history.history['val_loss'], label='Val Loss (MSE)')
    axes[0].set_title('GRU Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE')
    axes[0].legend()
    axes[0].grid(True)

    # MAE
    if 'mae' in history.history:
        axes[1].plot(history.history['mae'], label='Train MAE')
        axes[1].plot(history.history['val_mae'], label='Val MAE')
        axes[1].set_title('GRU Training MAE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].legend()
        axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 학습 기록 그래프가 저장되었습니다: {save_path}")


def save_training_metrics(history, save_path: str = None):
    """
    학습 지표를 텍스트 파일로 저장합니다.

    Args:
        history: Keras 학습 기록
        save_path: 저장 경로
    """
    if save_path is None:
        save_path = os.path.join(PATHS['results_dir'], 'gru_metrics.txt')

    with open(save_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("GRU 모델 학습 결과 (FedPer Head Layer)\n")
        f.write("=" * 70 + "\n\n")

        # 최종 에포크 정보
        final_epoch = len(history.history['loss'])
        f.write(f"총 학습 에포크: {final_epoch}\n\n")

        # 최고 성능
        f.write("최고 성능 (Validation):\n")
        f.write("-" * 70 + "\n")
        best_epoch = np.argmin(history.history['val_loss']) + 1
        best_val_loss = min(history.history['val_loss'])
        best_val_mae = history.history['val_mae'][best_epoch - 1] if 'val_mae' in history.history else None

        f.write(f"  Best Epoch: {best_epoch}\n")
        f.write(f"  Val Loss (MSE): {best_val_loss:.6f}\n")
        if best_val_mae:
            f.write(f"  Val MAE: {best_val_mae:.6f}\n")
        f.write("\n")

        # 최종 성능
        f.write("최종 성능:\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Train Loss: {history.history['loss'][-1]:.6f}\n")
        f.write(f"  Val Loss: {history.history['val_loss'][-1]:.6f}\n")
        if 'mae' in history.history:
            f.write(f"  Train MAE: {history.history['mae'][-1]:.6f}\n")
            f.write(f"  Val MAE: {history.history['val_mae'][-1]:.6f}\n")
        f.write("\n")

        # 에포크별 상세 기록
        f.write("에포크별 상세 기록:\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>12} {'Train MAE':>12} {'Val MAE':>12}\n")
        f.write("-" * 70 + "\n")

        for i in range(final_epoch):
            train_loss = history.history['loss'][i]
            val_loss = history.history['val_loss'][i]
            train_mae = history.history['mae'][i] if 'mae' in history.history else 0
            val_mae = history.history['val_mae'][i] if 'val_mae' in history.history else 0

            f.write(f"{i+1:6d} {train_loss:12.6f} {val_loss:12.6f} {train_mae:12.6f} {val_mae:12.6f}\n")

        f.write("\n" + "=" * 70 + "\n")

    print(f"📝 학습 지표가 저장되었습니다: {save_path}")


def main():
    """메인 학습 파이프라인"""
    print("\n" + "=" * 70)
    print("GRU 모델 학습 파이프라인 (FedPer Head Layer)")
    print("=" * 70)

    # ===== 단계 1: 데이터 로드 =====
    print("\n[단계 1] 데이터 로드 중...")

    data_path = os.path.join(PATHS['data_dir'], 'realistic_training_dataset.npz')

    if not os.path.exists(data_path):
        print(f"\n⚠️  오류: 학습 데이터를 찾을 수 없습니다: {data_path}")
        print("먼저 다음 명령을 실행하여 데이터를 생성하세요:")
        print("  python training/prepare_data.py")
        return

    data = np.load(data_path, allow_pickle=True)

    # Features 로드
    features_all = {
        'time': data['time'],
        'spatial': data['spatial'],
        'visual': data['visual'],
        'audio': data['audio'],
        'pose': data['pose']
    }
    labels_all = data['y']
    metadata = data['metadata'].item()

    print(f"  ✅ 데이터 로드 완료")
    print(f"  Total timesteps: {len(labels_all):,}")
    print(f"  Zones: {metadata['zones']}")

    # ===== 단계 2: GRU Trainer 초기화 (AttentionEncoder 로드) =====
    print("\n[단계 2] GRU Trainer 초기화 중 (AttentionEncoder 로드)...")
    trainer = GRUTrainer(num_zones=4, context_dim=160)

    # ===== 단계 3: Train/Val Split =====
    print("\n[단계 3] Train/Val Split...")

    train_split = 0.8
    n_train = int(len(labels_all) * train_split)

    features_train = {key: value[:n_train] for key, value in features_all.items()}
    features_val = {key: value[n_train:] for key, value in features_all.items()}
    labels_train = labels_all[:n_train]
    labels_val = labels_all[n_train:]

    print(f"  훈련 데이터: {len(labels_train):,}개")
    print(f"  검증 데이터: {len(labels_val):,}개")

    # ===== 단계 4: 시퀀스 생성 (AttentionEncoder 통과) =====
    print("\n[단계 4] 시퀀스 생성 (AttentionEncoder 통과)...")

    sequence_length = 30

    print("\n훈련 데이터:")
    X_train, y_train = create_sequences_from_features(
        features_train, labels_train, trainer.encoder, sequence_length
    )

    print("\n검증 데이터:")
    X_val, y_val = create_sequences_from_features(
        features_val, labels_val, trainer.encoder, sequence_length
    )

    print(f"\n최종 shape:")
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  X_val: {X_val.shape}")
    print(f"  y_val: {y_val.shape}")

    # ===== 단계 5: GRU 모델 구축 =====
    print("\n[단계 5] GRU 모델 구축 중...")
    trainer.build_gru_model(sequence_length=sequence_length)

    # ===== 단계 6: 모델 컴파일 =====
    print("\n[단계 6] 모델 컴파일 중...")
    trainer.compile_model()

    # ===== 단계 7: 학습 =====
    print("\n[단계 7] 학습 시작...")
    history = trainer.train(X_train, y_train, X_val, y_val)

    # ===== 단계 8: 평가 =====
    print("\n[단계 8] 모델 평가...")
    trainer.evaluate(X_val, y_val, zone_names=metadata['zones'])

    # ===== 단계 9: 모델 저장 =====
    print("\n[단계 9] GRU 모델 저장...")
    trainer.save_model()

    # ===== 단계 10: 학습 기록 시각화 및 저장 =====
    print("\n[단계 10] 학습 기록 시각화 및 저장...")
    plot_training_history(history)
    save_training_metrics(history)

    print("\n" + "=" * 70)
    print("✅ GRU 모델 학습 파이프라인 완료!")
    print("=" * 70)
    print("\n저장된 파일:")
    print(f"  📁 {PATHS['gru_model']} (학습된 GRU 모델 - Head Layer)")
    print(f"  📊 {PATHS['gru_history']} (학습 그래프)")
    print(f"  📝 {os.path.join(PATHS['results_dir'], 'gru_metrics.txt')} (성능 지표)")
    print("\nFedPer 구조 완성:")
    print(f"  🔒 Base Layer (Shared): {PATHS['encoder_model']}")
    print(f"  🎯 Head Layer (Personalized): {PATHS['gru_model']}")
    print("\n학습 완료! 이제 실시간 시스템에서 모델을 사용할 수 있습니다.")


if __name__ == "__main__":
    main()
