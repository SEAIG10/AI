"""
AttentionContextEncoder 학습 스크립트 (Updated for Realistic Dataset)

현실적인 일상 루틴 데이터셋으로 AttentionEncoder를 학습시킵니다.
학습 목표: 다중 모달 센서 특징 (visual, audio, pose, spatial, time) → 160차원 컨텍스트 벡터 변환
손실: 컨텍스트 벡터가 4개 구역의 오염도를 예측할 수 있도록 학습 (regression)
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
from training.config import SENSOR_DIMS, ENCODER_CONFIG, ENCODER_TRAINING, PATHS


class EncoderTrainer:
    """AttentionContextEncoder 학습 클래스"""

    def __init__(self):
        """학습기를 초기화합니다."""
        self.encoder = None
        self.prediction_head = None
        self.full_model = None

    def build_model(self, num_zones=4):
        """
        학습을 위한 모델을 구축합니다.

        AttentionEncoder + 예측 헤드 조합:
        - AttentionEncoder: 특징 융합 (학습 대상, Base Layer for FedPer)
        - 예측 헤드: 임시 regression 레이어 (인코더 학습용, 나중에 버림)
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

        # 예측 헤드 생성 (학습용, regression)
        # 이 레이어는 인코더가 의미 있는 특징을 학습하도록 돕습니다.
        context_input = tf.keras.Input(
            shape=(ENCODER_CONFIG['context_dim'],),
            name='context_vector'
        )
        x = tf.keras.layers.Dense(64, activation='relu', name='pred_hidden')(context_input)
        x = tf.keras.layers.Dropout(0.3, name='pred_dropout')(x)
        output = tf.keras.layers.Dense(num_zones, activation='sigmoid', name='pred_output')(x)

        self.prediction_head = tf.keras.Model(
            inputs=context_input,
            outputs=output,
            name='prediction_head'
        )

        print("\n[예측 헤드 (학습용, Regression)]")
        self.prediction_head.summary()

        # 전체 모델: Encoder + 예측 헤드
        # 입력: 다중 모달 특징
        # 출력: 4개 구역의 오염도 (0~1)
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
        """모델을 컴파일합니다 (Regression)."""
        self.full_model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=ENCODER_TRAINING['learning_rate']
            ),
            loss='mse',  # Regression: Mean Squared Error
            metrics=['mae', 'mse']
        )
        print("모델 컴파일 완료 (Regression MSE)\n")

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
            features_train: 훈련 특징 딕셔너리
                - 'time': (N, 10)
                - 'spatial': (N, 4)
                - 'visual': (N, 14)
                - 'audio': (N, 17)
                - 'pose': (N, 51)
            labels_train: 훈련 레이블 (N, 4)
            features_val: 검증 특징 (동일 구조)
            labels_val: 검증 레이블 (N_val, 4)

        Returns:
            history: 학습 기록
        """
        print("\n" + "=" * 70)
        print("AttentionContextEncoder 학습 시작")
        print("=" * 70 + "\n")

        print(f"훈련 샘플: {labels_train.shape[0]:,}개")
        print(f"검증 샘플: {labels_val.shape[0]:,}개")
        print(f"입력 features:")
        for key, value in features_train.items():
            print(f"  {key:10s}: {value.shape}")
        print(f"출력 labels: {labels_train.shape}\n")

        # 콜백 설정
        print("[콜백 설정]")
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
        print("\n[학습 진행 중...]\n")
        history = self.full_model.fit(
            features_train, labels_train,
            validation_data=(features_val, labels_val),
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
        (예측 헤드는 버립니다 - GRU가 새로운 Head가 될 것)

        Args:
            save_path: 저장 경로 (기본값: config)
        """
        if save_path is None:
            save_path = PATHS['encoder_model']

        self.encoder.save(save_path)
        print(f"✅ AttentionEncoder가 저장되었습니다: {save_path}")

        # 모델 크기 정보
        file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
        print(f"  파일 크기: {file_size_mb:.2f} MB")

    def evaluate(self, features_val: dict, labels_val: np.ndarray, zone_names: list):
        """
        모델을 평가합니다.

        Args:
            features_val: 검증 특징
            labels_val: 검증 레이블
            zone_names: 구역 이름 리스트
        """
        print("\n" + "=" * 70)
        print("AttentionEncoder 평가")
        print("=" * 70 + "\n")

        # 예측
        y_pred = self.full_model.predict(features_val, verbose=0)

        # 구역별 평가
        print("구역별 성능 (Regression):")
        print("-" * 70)
        for i, zone in enumerate(zone_names):
            mae = np.mean(np.abs(labels_val[:, i] - y_pred[:, i]))
            mse = np.mean((labels_val[:, i] - y_pred[:, i]) ** 2)
            rmse = np.sqrt(mse)

            print(f"{zone:15s}: MAE={mae:.4f}, MSE={mse:.4f}, RMSE={rmse:.4f}")

        # 전체 평가
        overall_mae = np.mean(np.abs(labels_val - y_pred))
        overall_rmse = np.sqrt(np.mean((labels_val - y_pred) ** 2))

        print("-" * 70)
        print(f"{'Overall':15s}: MAE={overall_mae:.4f}, RMSE={overall_rmse:.4f}")
        print("=" * 70 + "\n")


def plot_training_history(history, save_path: str = None):
    """
    학습 기록을 시각화합니다.

    Args:
        history: Keras 학습 기록
        save_path: 저장 경로
    """
    if save_path is None:
        save_path = PATHS['encoder_history']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 손실
    axes[0].plot(history.history['loss'], label='Train Loss (MSE)')
    axes[0].plot(history.history['val_loss'], label='Val Loss (MSE)')
    axes[0].set_title('Encoder Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE')
    axes[0].legend()
    axes[0].grid(True)

    # MAE
    if 'mae' in history.history:
        axes[1].plot(history.history['mae'], label='Train MAE')
        axes[1].plot(history.history['val_mae'], label='Val MAE')
        axes[1].set_title('Encoder Training MAE')
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
        save_path = os.path.join(PATHS['results_dir'], 'encoder_metrics.txt')

    with open(save_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("AttentionEncoder 학습 결과\n")
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
    print("AttentionContextEncoder 학습 파이프라인 (Realistic Dataset)")
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

    # ===== 단계 2: Train/Val Split =====
    print("\n[단계 2] Train/Val Split...")

    train_split = 0.8
    n_train = int(len(labels_all) * train_split)

    features_train = {key: value[:n_train] for key, value in features_all.items()}
    features_val = {key: value[n_train:] for key, value in features_all.items()}
    labels_train = labels_all[:n_train]
    labels_val = labels_all[n_train:]

    print(f"  훈련 데이터: {len(labels_train):,}개")
    print(f"  검증 데이터: {len(labels_val):,}개")

    # ===== 단계 3: Encoder Trainer 초기화 =====
    print("\n[단계 3] Encoder Trainer 초기화 중...")
    trainer = EncoderTrainer()

    # ===== 단계 4: 모델 구축 =====
    print("\n[단계 4] 모델 구축 중...")
    trainer.build_model(num_zones=4)

    # ===== 단계 5: 모델 컴파일 =====
    print("\n[단계 5] 모델 컴파일 중...")
    trainer.compile_model()

    # ===== 단계 6: 학습 =====
    print("\n[단계 6] 학습 시작...")
    history = trainer.train(
        features_train, labels_train,
        features_val, labels_val
    )

    # ===== 단계 7: 평가 =====
    print("\n[단계 7] 모델 평가...")
    trainer.evaluate(features_val, labels_val, zone_names=metadata['zones'])

    # ===== 단계 8: Encoder 저장 =====
    print("\n[단계 8] AttentionEncoder 저장...")
    trainer.save_encoder()

    # ===== 단계 9: 학습 기록 시각화 및 저장 =====
    print("\n[단계 9] 학습 기록 시각화 및 저장...")
    plot_training_history(history)
    save_training_metrics(history)

    print("\n" + "=" * 70)
    print("✅ AttentionEncoder 학습 파이프라인 완료!")
    print("=" * 70)
    print("\n저장된 파일:")
    print(f"  📁 {PATHS['encoder_model']} (학습된 AttentionEncoder - Base Layer)")
    print(f"  📊 {PATHS['encoder_history']} (학습 그래프)")
    print(f"  📝 {os.path.join(PATHS['results_dir'], 'encoder_metrics.txt')} (성능 지표)")
    print("\n다음 단계:")
    print("  python training/train_gru.py  # GRU Head Layer 학습")


if __name__ == "__main__":
    main()
