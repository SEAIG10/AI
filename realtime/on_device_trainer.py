"""
온디바이스 Head Layer 학습
실시간으로 수집된 데이터로 GRU Head를 지속적으로 학습시킵니다.
"""

import sys
import os
import threading
import time
import numpy as np
from collections import deque
from typing import Optional, Tuple

# 프로젝트 루트 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model.gru_model import FedPerGRUModel


class OnDeviceTrainer:
    """
    온디바이스 Head Layer 학습

    역할:
    1. 실시간 예측 데이터와 실제 오염도 피드백을 버퍼에 저장
    2. 버퍼가 일정 크기 이상 쌓이면 백그라운드에서 Head 학습
    3. Base Layer는 frozen, Head Layer만 학습
    4. 학습된 모델 자동 저장
    """

    def __init__(
        self,
        gru_model: FedPerGRUModel,
        buffer_size: int = 300,
        min_samples_for_training: int = 100,
        batch_size: int = 16,
        epochs_per_update: int = 5,
        learning_rate: float = 0.0005,
        auto_save_path: Optional[str] = None,
        mqtt_client = None
    ):
        """
        OnDeviceTrainer를 초기화합니다.

        Args:
            gru_model: GRU 모델 인스턴스
            buffer_size: 데이터 버퍼 최대 크기
            min_samples_for_training: 학습을 시작하는 최소 샘플 수
            batch_size: 학습 배치 크기
            epochs_per_update: 학습 에포크 수
            learning_rate: 학습률
            auto_save_path: 모델 자동 저장 경로
            mqtt_client: MQTT 클라이언트 (학습 상태 전송용, 선택)
        """
        self.gru_model = gru_model
        self.buffer_size = buffer_size
        self.min_samples_for_training = min_samples_for_training
        self.batch_size = batch_size
        self.epochs_per_update = epochs_per_update
        self.learning_rate = learning_rate
        self.auto_save_path = auto_save_path
        self.mqtt_client = mqtt_client

        # 데이터 버퍼 (X: context sequences, y: pollution labels)
        self.X_buffer = deque(maxlen=buffer_size)
        self.y_buffer = deque(maxlen=buffer_size)

        # 학습 통계
        self.total_samples_collected = 0
        self.total_training_runs = 0
        self.last_training_time = None

        # 백그라운드 학습 스레드
        self.training_thread = None
        self.is_training = False
        self.stop_flag = False

        # Base Layer를 frozen으로 설정
        self._freeze_base_layer()

        print(f"\n{'='*60}")
        print(f"OnDeviceTrainer Initialized")
        print(f"{'='*60}")
        print(f"  Buffer size: {buffer_size}")
        print(f"  Min samples for training: {min_samples_for_training}")
        print(f"  Batch size: {batch_size}")
        print(f"  Epochs per update: {epochs_per_update}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Auto-save: {auto_save_path or 'Disabled'}")
        print(f"{'='*60}\n")

    def _freeze_base_layer(self):
        """Base Layer를 frozen으로 설정합니다."""
        if self.gru_model.base_model is not None:
            self.gru_model.base_model.trainable = False
            print("Base Layer frozen (학습 안 함)")

        if self.gru_model.head_model is not None:
            self.gru_model.head_model.trainable = True
            print("Head Layer trainable (학습 대상)")

    def add_sample(self, context_sequence: np.ndarray, pollution_label: np.ndarray):
        """
        학습 샘플을 버퍼에 추가합니다.

        Args:
            context_sequence: (30, 160) - 30 타임스텝의 context vector
            pollution_label: (num_zones,) - 실제 오염도
        """
        # Shape 검증
        if context_sequence.shape != (30, self.gru_model.context_dim):
            print(f"Warning: Invalid context_sequence shape: {context_sequence.shape}, expected: (30, {self.gru_model.context_dim})")
            return

        if pollution_label.shape[0] != self.gru_model.num_zones:
            print(f"Warning: Invalid pollution_label shape: {pollution_label.shape}, expected: ({self.gru_model.num_zones},)")
            return

        # 버퍼에 추가
        self.X_buffer.append(context_sequence)
        self.y_buffer.append(pollution_label)
        self.total_samples_collected += 1

        # 학습 트리거 확인
        if len(self.X_buffer) >= self.min_samples_for_training and not self.is_training:
            print(f"\nTraining triggered: {len(self.X_buffer)} samples collected")
            self.start_background_training()

    def start_background_training(self):
        """백그라운드 스레드에서 학습을 시작합니다."""
        if self.is_training:
            print("Warning: Training already in progress, skipping...")
            return

        # 백그라운드 스레드 시작
        self.training_thread = threading.Thread(target=self._training_worker, daemon=True)
        self.training_thread.start()

    def _training_worker(self):
        """백그라운드 학습 워커 (스레드)"""
        self.is_training = True

        try:
            print(f"\n{'='*60}")
            print(f"On-Device Head Training Started")
            print(f"{'='*60}")

            # 버퍼에서 데이터 추출
            X_train = np.array(list(self.X_buffer))
            y_train = np.array(list(self.y_buffer))

            print(f"  Training samples: {len(X_train)}")
            print(f"  X shape: {X_train.shape}")
            print(f"  y shape: {y_train.shape}")

            # Train/Val Split (80/20)
            split_idx = int(len(X_train) * 0.8)
            X_train_split = X_train[:split_idx]
            y_train_split = y_train[:split_idx]
            X_val_split = X_train[split_idx:]
            y_val_split = y_train[split_idx:]

            print(f"\n  Train: {len(X_train_split)} samples")
            print(f"  Val: {len(X_val_split)} samples")

            # 모델 재컴파일 (Head만 학습)
            self.gru_model.compile_model(
                learning_rate=self.learning_rate,
                loss='mse',
                metrics=['mae']
            )

            # 학습
            print(f"\n  Training for {self.epochs_per_update} epochs...\n")
            history = self.gru_model.model.fit(
                X_train_split, y_train_split,
                validation_data=(X_val_split, y_val_split) if len(X_val_split) > 0 else None,
                epochs=self.epochs_per_update,
                batch_size=self.batch_size,
                verbose=1
            )

            # 통계 업데이트
            self.total_training_runs += 1
            self.last_training_time = time.time()

            # 최종 손실
            final_loss = history.history['loss'][-1]
            final_val_loss = history.history['val_loss'][-1] if 'val_loss' in history.history else None

            print(f"\n{'='*60}")
            print(f"Training Complete")
            print(f"{'='*60}")
            print(f"  Final train loss: {final_loss:.6f}")
            if final_val_loss:
                print(f"  Final val loss: {final_val_loss:.6f}")
            print(f"  Total training runs: {self.total_training_runs}")
            print(f"  Total samples collected: {self.total_samples_collected}")
            print(f"{'='*60}\n")

            # 모델 자동 저장
            if self.auto_save_path:
                self.save_model()

            # MQTT로 학습 완료 상태 전송
            if self.mqtt_client:
                self.mqtt_client.publish_training_status(
                    "completed",
                    samples_used=len(X_train),
                    epochs=self.epochs_per_update,
                    final_train_loss=float(final_loss),
                    final_val_loss=float(final_val_loss) if final_val_loss else None,
                    total_training_runs=self.total_training_runs
                )

        except Exception as e:
            print(f"\nError during on-device training: {e}")
            import traceback
            traceback.print_exc()

            # MQTT로 학습 실패 상태 전송
            if self.mqtt_client:
                self.mqtt_client.publish_training_status(
                    "failed",
                    error=str(e)
                )

        finally:
            self.is_training = False

    def save_model(self, save_path: Optional[str] = None):
        """
        학습된 모델을 저장합니다.

        Args:
            save_path: 저장 경로 (None이면 auto_save_path 사용)
        """
        path = save_path or self.auto_save_path
        if path is None:
            print("Warning: No save path specified, skipping model save")
            return

        try:
            self.gru_model.save(path)
            file_size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"Model saved: {path} ({file_size_mb:.2f} MB)")
        except Exception as e:
            print(f"Error: Failed to save model: {e}")

    def get_statistics(self) -> dict:
        """학습 통계를 반환합니다."""
        return {
            'buffer_size': len(self.X_buffer),
            'total_samples_collected': self.total_samples_collected,
            'total_training_runs': self.total_training_runs,
            'last_training_time': self.last_training_time,
            'is_training': self.is_training
        }

    def clear_buffer(self):
        """데이터 버퍼를 초기화합니다."""
        self.X_buffer.clear()
        self.y_buffer.clear()
        print("Training buffer cleared")

    def stop(self):
        """학습을 중지하고 리소스를 정리합니다."""
        self.stop_flag = True
        if self.training_thread and self.training_thread.is_alive():
            self.training_thread.join(timeout=5.0)
        print("\nOnDeviceTrainer stopped")


def test_on_device_trainer():
    """OnDeviceTrainer 테스트"""
    print("\n" + "="*60)
    print("OnDeviceTrainer Test")
    print("="*60)

    # GRU 모델 생성
    print("\n1. Creating GRU model...")
    from src.model.gru_model import FedPerGRUModel

    gru_model = FedPerGRUModel(num_zones=4, context_dim=160)
    gru_model.compile_model()

    # OnDeviceTrainer 초기화
    print("\n2. Initializing OnDeviceTrainer...")
    trainer = OnDeviceTrainer(
        gru_model=gru_model,
        buffer_size=300,
        min_samples_for_training=50,
        batch_size=16,
        epochs_per_update=3
    )

    # 더미 데이터 추가
    print("\n3. Adding dummy training samples...")
    for i in range(60):
        context_seq = np.random.randn(30, 160).astype(np.float32)
        pollution = np.random.rand(4).astype(np.float32)
        trainer.add_sample(context_seq, pollution)

        if i % 10 == 0:
            print(f"   Added {i+1}/60 samples, buffer: {len(trainer.X_buffer)}")

    # 학습 완료 대기
    print("\n4. Waiting for training to complete...")
    time.sleep(10)

    # 통계 확인
    print("\n5. Training statistics:")
    stats = trainer.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    print("\nTest complete!\n")


if __name__ == "__main__":
    test_on_device_trainer()
