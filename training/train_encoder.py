"""
AttentionContextEncoder 학습 스크립트

이 스크립트는 AttentionContextEncoder를 별도로 학습시킵니다.
학습 목표: 다중 모달 센서 특징 → 160차원 컨텍스트 벡터 변환
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks
import matplotlib.pyplot as plt

# 프로젝트 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.context_fusion.attention_context_encoder import create_attention_encoder
from training.data_generator import SyntheticDataGenerator
from training.config import (
    SENSOR_DIMS, ENCODER_CONFIG, ENCODER_TRAINING, PATHS, GRU_CONFIG
)


class EncoderTrainer:
    """
    AttentionContextEncoder 학습 클래스

    학습 목표:
    - 입력: 다중 모달 센서 특징 (visual, audio, pose, spatial, time)
    - 출력: 160차원 컨텍스트 벡터
    - 손실: 컨텍스트 벡터가 청소 필요 여부를 예측할 수 있도록 학습
    """

    def __init__(self):
        """학습기를 초기화합니다."""
        self.encoder = None
        self.prediction_head = None
        self.full_model = None

    def build_model(self):
        """
        학습을 위한 모델을 구축합니다.

        AttentionEncoder + 예측 헤드 조합:
        - AttentionEncoder: 특징 융합 (학습 대상)
        - 예측 헤드: 임시 분류 레이어 (인코더 학습용, 나중에 버림)
        """
        print("\n" + "=" * 70)
        print("AttentionContextEncoder 모델 구축")
        print("=" * 70)

        # AttentionEncoder 생성
        self.encoder = create_attention_encoder(
            visual_dim=SENSOR_DIMS['visual'],
            audio_dim=SENSOR_DIMS['audio'],
            pose_dim=SENSOR_DIMS['pose'],
            spatial_dim=SENSOR_DIMS['spatial'],
            time_dim=SENSOR_DIMS['time'],
            context_dim=ENCODER_CONFIG['context_dim']
        )

        print("\n[AttentionContextEncoder]")
        self.encoder.summary()

        # 예측 헤드 생성 (학습용)
        # 이 레이어는 인코더가 의미 있는 특징을 학습하도록 돕습니다.
        context_input = tf.keras.Input(
            shape=(ENCODER_CONFIG['context_dim'],),
            name='context_vector'
        )
        x = tf.keras.layers.Dense(64, activation='relu', name='pred_hidden')(context_input)
        x = tf.keras.layers.Dropout(0.3, name='pred_dropout')(x)
        output = tf.keras.layers.Dense(GRU_CONFIG['num_zones'], activation='sigmoid', name='pred_output')(x)

        self.prediction_head = tf.keras.Model(
            inputs=context_input,
            outputs=output,
            name='prediction_head'
        )

        print("\n[예측 헤드 (학습용)]")
        self.prediction_head.summary()

        # 전체 모델: Encoder + 예측 헤드
        # 입력: 다중 모달 특징
        # 출력: 7개 구역의 청소 필요 확률
        inputs = {
            'visual': tf.keras.Input(shape=(SENSOR_DIMS['visual'],), name='visual'),
            'audio': tf.keras.Input(shape=(SENSOR_DIMS['audio'],), name='audio'),
            'pose': tf.keras.Input(shape=(SENSOR_DIMS['pose'],), name='pose'),
            'spatial': tf.keras.Input(shape=(SENSOR_DIMS['spatial'],), name='spatial'),
            'time': tf.keras.Input(shape=(SENSOR_DIMS['time'],), name='time'),
        }

        context = self.encoder(inputs)
        predictions = self.prediction_head(context)

        self.full_model = tf.keras.Model(
            inputs=inputs,
            outputs=predictions,
            name='encoder_training_model'
        )

        print("\n[전체 학습 모델]")
        self.full_model.summary()

        print("\n" + "=" * 70 + "\n")

    def compile_model(self):
        """모델을 컴파일합니다."""
        self.full_model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=ENCODER_TRAINING['learning_rate']
            ),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
        print("모델 컴파일 완료\n")

    def prepare_training_data(self, features: dict, labels: np.ndarray):
        """
        학습 데이터를 준비합니다.

        Args:
            features: 특징 딕셔너리 (각 키: (N, 30, dim))
            labels: 레이블 (N, 7)

        Returns:
            prepared_features: 타임스텝별로 평탄화된 특징
            prepared_labels: 반복된 레이블
        """
        # 시퀀스를 타임스텝별로 평탄화
        # (N, 30, dim) → (N*30, dim)
        prepared_features = {}
        for key, value in features.items():
            N, T, dim = value.shape
            prepared_features[key] = value.reshape(N * T, dim)

        # 레이블도 반복
        # (N, 7) → (N*30, 7)
        N = labels.shape[0]
        prepared_labels = np.repeat(labels, 30, axis=0)

        return prepared_features, prepared_labels

    def train(
        self,
        features_train: dict,
        labels_train: np.ndarray,
        features_val: dict,
        labels_val: np.ndarray
    ):
        """
        모델을 학습시킵니다.

        Args:
            features_train: 훈련 특징 (각 키: (N_train, 30, dim))
            labels_train: 훈련 레이블 (N_train, 7)
            features_val: 검증 특징 (각 키: (N_val, 30, dim))
            labels_val: 검증 레이블 (N_val, 7)

        Returns:
            history: 학습 기록
        """
        print("\n" + "=" * 70)
        print("AttentionContextEncoder 학습 시작")
        print("=" * 70 + "\n")

        # 데이터 준비
        print("[1] 학습 데이터 준비 중...")
        X_train, y_train = self.prepare_training_data(features_train, labels_train)
        X_val, y_val = self.prepare_training_data(features_val, labels_val)

        print(f"  훈련 샘플: {y_train.shape[0]}개 (원본: {labels_train.shape[0]}개 × 30)")
        print(f"  검증 샘플: {y_val.shape[0]}개 (원본: {labels_val.shape[0]}개 × 30)")

        # 콜백 설정
        print("\n[2] 콜백 설정 중...")
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=ENCODER_TRAINING['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=ENCODER_TRAINING['reduce_lr_patience'],
                min_lr=ENCODER_TRAINING['min_lr'],
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                filepath=PATHS['encoder_model'],
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]

        # 학습
        print("\n[3] 학습 진행 중...\n")
        history = self.full_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=ENCODER_TRAINING['epochs'],
            batch_size=ENCODER_TRAINING['batch_size'],
            callbacks=callback_list,
            verbose=1
        )

        print("\n" + "=" * 70)
        print("학습 완료!")
        print("=" * 70 + "\n")

        return history

    def save_encoder(self, save_path: str = None):
        """
        학습된 AttentionEncoder만 저장합니다.
        (예측 헤드는 버립니다)

        Args:
            save_path: 저장 경로 (기본값: config)
        """
        if save_path is None:
            save_path = PATHS['encoder_model']

        self.encoder.save(save_path)
        print(f"AttentionEncoder가 저장되었습니다: {save_path}")

        # 모델 크기 정보
        file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
        print(f"  파일 크기: {file_size_mb:.2f} MB")

    def evaluate(self, features_val: dict, labels_val: np.ndarray):
        """
        모델을 평가합니다.

        Args:
            features_val: 검증 특징
            labels_val: 검증 레이블
        """
        print("\n" + "=" * 70)
        print("모델 평가")
        print("=" * 70 + "\n")

        X_val, y_val = self.prepare_training_data(features_val, labels_val)

        results = self.full_model.evaluate(X_val, y_val, verbose=0)

        print("[평가 결과]")
        for metric_name, value in zip(self.full_model.metrics_names, results):
            print(f"  {metric_name:12s}: {value:.4f}")

        print("\n" + "=" * 70 + "\n")


def plot_training_history(history, save_path: str = None):
    """
    학습 기록을 시각화합니다.

    Args:
        history: Keras 학습 기록
        save_path: 저장 경로 (기본값: config)
    """
    if save_path is None:
        save_path = PATHS['encoder_history']

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
    print("AttentionContextEncoder 학습 파이프라인")
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

    # ===== 단계 2: 모델 구축 =====
    print("\n[단계 2] 모델 구축 중...")
    trainer = EncoderTrainer()
    trainer.build_model()

    # ===== 단계 3: 모델 컴파일 =====
    print("[단계 3] 모델 컴파일 중...")
    trainer.compile_model()

    # ===== 단계 4: 학습 =====
    print("[단계 4] 학습 시작...")
    history = trainer.train(
        features_train, labels_train,
        features_val, labels_val
    )

    # ===== 단계 5: 평가 =====
    print("[단계 5] 모델 평가...")
    trainer.evaluate(features_val, labels_val)

    # ===== 단계 6: AttentionEncoder 저장 =====
    print("[단계 6] AttentionEncoder 저장...")
    trainer.save_encoder()

    # ===== 단계 7: 학습 기록 시각화 =====
    print("[단계 7] 학습 기록 시각화...")
    plot_training_history(history)

    print("\n" + "=" * 70)
    print("AttentionContextEncoder 학습 파이프라인 완료!")
    print("=" * 70)
    print("\n저장된 파일:")
    print(f"  - {PATHS['encoder_model']} (학습된 인코더)")
    print(f"  - {PATHS['encoder_history']} (학습 그래프)")
    print("\n다음 단계:")
    print("  python training/train_gru.py")


if __name__ == "__main__":
    main()
