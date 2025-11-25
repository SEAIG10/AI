"""
GRU 모델 학습 스크립트

이 스크립트는 사전 학습된 AttentionContextEncoder를 사용하여 GRU 모델을 학습시킵니다.
학습 목표: 160차원 컨텍스트 벡터 시퀀스 → 청소 필요 구역 예측
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks
import matplotlib.pyplot as plt

# 프로젝트 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.context_fusion.attention_context_encoder import create_attention_encoder, AttentionContextEncoder
from src.model.gru_model import FedPerGRUModel
from training.data_generator import SyntheticDataGenerator
from training.config import (
    SENSOR_DIMS, ENCODER_CONFIG, GRU_CONFIG, GRU_TRAINING, PATHS
)


class GRUTrainer:
    """
    GRU 모델 학습 클래스

    학습 목표:
    - 입력: 160차원 컨텍스트 벡터 시퀀스 (30 타임스텝)
    - 출력: 7개 구역의 청소 필요 확률
    """

    def __init__(self, encoder_path: str = None):
        """
        학습기를 초기화합니다.

        Args:
            encoder_path: 사전 학습된 AttentionEncoder 경로 (기본값: config)
        """
        if encoder_path is None:
            encoder_path = PATHS['encoder_model']

        # 사전 학습된 AttentionEncoder 로드
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
        print(f"  로드 완료: {encoder_path}\n")

        # GRU 모델 초기화
        self.gru_model = None

    def generate_context_sequences(
        self,
        features: dict,
        labels: np.ndarray
    ) -> tuple:
        """
        원본 센서 특징으로부터 컨텍스트 벡터 시퀀스를 생성합니다.

        Args:
            features: 센서 특징 (각 키: (N, 30, dim))
            labels: 레이블 (N, 7)

        Returns:
            X: 컨텍스트 시퀀스 (N, 30, 160)
            y: 레이블 (N, 7)
        """
        N, T, _ = features['visual'].shape

        # 배치 처리를 위해 시퀀스를 평탄화
        # (N, 30, dim) → (N*30, dim)
        batched_features = {}
        for key, value in features.items():
            batched_features[key] = value.reshape(N * T, -1)

        # AttentionEncoder를 통과시켜 컨텍스트 벡터 생성
        # (N*30, dim) → (N*30, 160)
        print(f"  AttentionEncoder를 통과시키는 중... ({N*T}개 타임스텝)")
        context_vectors = self.encoder.predict(
            batched_features,
            batch_size=256,
            verbose=0
        )

        # 시퀀스 형태로 재구성
        # (N*30, 160) → (N, 30, 160)
        X = context_vectors.reshape(N, T, -1)

        print(f"  생성된 컨텍스트 시퀀스 shape: {X.shape}")

        return X, labels

    def build_gru_model(self):
        """GRU 모델을 구축합니다."""
        print("\n" + "=" * 70)
        print("GRU 모델 구축")
        print("=" * 70 + "\n")

        self.gru_model = FedPerGRUModel(
            num_zones=GRU_CONFIG['num_zones'],
            context_dim=GRU_CONFIG['context_dim']
        )

        # 더미 데이터로 모델 빌드
        dummy_input = tf.random.normal(
            (1, GRU_CONFIG['sequence_length'], GRU_CONFIG['context_dim'])
        )
        _ = self.gru_model.model(dummy_input)

        print("[GRU 모델 아키텍처]")
        self.gru_model.summary()
        print("\n" + "=" * 70 + "\n")

    def compile_model(self):
        """GRU 모델을 컴파일합니다."""
        self.gru_model.compile_model(
            learning_rate=GRU_TRAINING['learning_rate'],
            loss='binary_crossentropy',
            metrics=['accuracy', 'AUC', 'Precision', 'Recall']
        )
        print("GRU 모델 컴파일 완료\n")

    def train(
        self,
        features_train: dict,
        labels_train: np.ndarray,
        features_val: dict,
        labels_val: np.ndarray
    ):
        """
        GRU 모델을 학습시킵니다.

        Args:
            features_train: 훈련 특징 (각 키: (N_train, 30, dim))
            labels_train: 훈련 레이블 (N_train, 7)
            features_val: 검증 특징 (각 키: (N_val, 30, dim))
            labels_val: 검증 레이블 (N_val, 7)

        Returns:
            history: 학습 기록
        """
        print("\n" + "=" * 70)
        print("GRU 모델 학습 시작")
        print("=" * 70 + "\n")

        # ===== 단계 1: 컨텍스트 벡터 시퀀스 생성 =====
        print("[1] 훈련 데이터를 컨텍스트 시퀀스로 변환 중...")
        X_train, y_train = self.generate_context_sequences(
            features_train, labels_train
        )

        print("\n[2] 검증 데이터를 컨텍스트 시퀀스로 변환 중...")
        X_val, y_val = self.generate_context_sequences(
            features_val, labels_val
        )

        print(f"\n[3] 학습 데이터:")
        print(f"  훈련 샘플: {X_train.shape[0]}개")
        print(f"  검증 샘플: {X_val.shape[0]}개")

        # ===== 단계 2: 콜백 설정 =====
        print("\n[4] 콜백 설정 중...")
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

        # ===== 단계 3: 학습 =====
        print("\n[5] 학습 진행 중...\n")
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

    def evaluate(
        self,
        features_val: dict,
        labels_val: np.ndarray,
        zone_names: list
    ):
        """
        GRU 모델을 평가합니다.

        Args:
            features_val: 검증 특징
            labels_val: 검증 레이블
            zone_names: 구역 이름 리스트
        """
        print("\n" + "=" * 70)
        print("GRU 모델 평가")
        print("=" * 70 + "\n")

        # 컨텍스트 시퀀스 생성
        X_val, y_val = self.generate_context_sequences(features_val, labels_val)

        # 평가
        results = self.gru_model.evaluate(X_val, y_val, zone_names=zone_names)

        print("\n" + "=" * 70 + "\n")

    def save_model(self, save_path: str = None):
        """
        학습된 GRU 모델을 저장합니다.

        Args:
            save_path: 저장 경로 (기본값: config)
        """
        if save_path is None:
            save_path = PATHS['gru_model']

        self.gru_model.save(save_path)
        print(f"GRU 모델이 저장되었습니다: {save_path}")

        # 모델 크기 정보
        file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
        print(f"  파일 크기: {file_size_mb:.2f} MB")


def plot_training_history(history, save_path: str = None):
    """
    학습 기록을 시각화합니다.

    Args:
        history: Keras 학습 기록
        save_path: 저장 경로 (기본값: config)
    """
    if save_path is None:
        save_path = PATHS['gru_history']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 손실
    axes[0, 0].plot(history.history['loss'], label='Train Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # 정확도
    axes[0, 1].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0, 1].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0, 1].set_title('Model Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # AUC
    if 'auc' in history.history:
        axes[1, 0].plot(history.history['auc'], label='Train AUC')
        axes[1, 0].plot(history.history['val_auc'], label='Val AUC')
        axes[1, 0].set_title('Model AUC')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

    # 정밀도
    if 'precision' in history.history:
        axes[1, 1].plot(history.history['precision'], label='Train Precision')
        axes[1, 1].plot(history.history['val_precision'], label='Val Precision')
        axes[1, 1].set_title('Model Precision')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Precision')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n학습 기록 그래프가 저장되었습니다: {save_path}")


def main():
    """메인 학습 파이프라인"""
    print("\n" + "=" * 70)
    print("GRU 모델 학습 파이프라인")
    print("=" * 70)

    # ===== 단계 1: 데이터 로드 =====
    print("\n[단계 1] 데이터 로드 중...")

    if not os.path.exists(PATHS['raw_features']):
        print(f"\n오류: 학습 데이터를 찾을 수 없습니다: {PATHS['raw_features']}")
        print("먼저 다음 명령을 실행하여 데이터를 생성하세요:")
        print("  python training/prepare_data.py")
        return

    generator = SyntheticDataGenerator()
    features_train, features_val, labels_train, labels_val = \
        generator.load_data()

    # ===== 단계 2: GRU Trainer 초기화 =====
    print("\n[단계 2] GRU Trainer 초기화 중...")
    trainer = GRUTrainer()

    # ===== 단계 3: GRU 모델 구축 =====
    print("[단계 3] GRU 모델 구축 중...")
    trainer.build_gru_model()

    # ===== 단계 4: 모델 컴파일 =====
    print("[단계 4] 모델 컴파일 중...")
    trainer.compile_model()

    # ===== 단계 5: 학습 =====
    print("[단계 5] 학습 시작...")
    history = trainer.train(
        features_train, labels_train,
        features_val, labels_val
    )

    # ===== 단계 6: 평가 =====
    print("[단계 6] 모델 평가...")
    trainer.evaluate(features_val, labels_val, zone_names=generator.zones)

    # ===== 단계 7: GRU 모델 저장 =====
    print("[단계 7] GRU 모델 저장...")
    trainer.save_model()

    # ===== 단계 8: 학습 기록 시각화 =====
    print("[단계 8] 학습 기록 시각화...")
    plot_training_history(history)

    print("\n" + "=" * 70)
    print("GRU 모델 학습 파이프라인 완료!")
    print("=" * 70)
    print("\n저장된 파일:")
    print(f"  - {PATHS['gru_model']} (학습된 GRU 모델)")
    print(f"  - {PATHS['gru_history']} (학습 그래프)")
    print("\n학습 완료! 이제 실시간 시스템에서 모델을 사용할 수 있습니다.")


if __name__ == "__main__":
    main()
