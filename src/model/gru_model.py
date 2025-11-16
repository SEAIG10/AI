"""
순차 패턴 학습을 위한 GRU 모델
행동 패턴의 인과 관계를 기반으로 청소 필요성을 예측합니다.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Tuple, Optional


class FedPerGRUModel:
    """
    어텐션 컨텍스트 인코더와 함께 사용하는 FedPer 아키텍처 기반의 GRU 모델입니다.

    아키텍처:
    - 기반 레이어 (공유): GRU(64) → GRU(32)
    - 헤드 레이어 (개인화): Dense(16) → Dense(7, sigmoid)

    입력: (배치, 30, 160) - 160차원 어텐션 컨텍스트 벡터의 30개 타임스텝
    출력: (배치, 7) - 7개 구역에 대한 오염 확률
    """

    def __init__(self, num_zones: int = 7, context_dim: int = 160):
        """
        GRU 모델을 초기화합니다.

        Args:
            num_zones: 의미론적 공간의 수 (기본값: 7)
            context_dim: 컨텍스트 벡터의 차원 (기본값: 160, AttentionContextEncoder 출력)
        """
        self.num_zones = num_zones
        self.context_dim = context_dim
        self.model = self._build_model()
        self.base_model = None  # FedPer용: 공유 기반 레이어
        self.head_model = None  # FedPer용: 개인화된 헤드 레이어

    def _build_model(self) -> keras.Model:
        """
        전체 GRU 모델을 구성합니다.

        Returns:
            컴파일된 Keras 모델
        """
        # 입력: 30 타임스텝의 어텐션 컨텍스트 벡터
        inputs = layers.Input(shape=(30, self.context_dim), name='context_sequence')

        # ===== 기반 레이어 (공유 특징 추출) =====
        # GRU 레이어 1: 64 유닛, 다음 GRU를 위해 시퀀스 반환
        x = layers.GRU(64, return_sequences=True, name='base_gru1')(inputs)

        # GRU 레이어 2: 32 유닛, 최종 표현
        base_output = layers.GRU(32, return_sequences=False, name='base_gru2')(x)

        # ===== 헤드 레이어 (개인화된 예측) =====
        # Dense 레이어 1: 16 유닛, ReLU 활성화 함수
        x = layers.Dense(16, activation='relu', name='head_dense1')(base_output)

        # 드롭아웃: 0.3 (과적합 방지)
        x = layers.Dropout(0.3, name='head_dropout')(x)

        # 출력 레이어: 7개 구역, Sigmoid 활성화 함수 (다중 레이블 이진 분류)
        outputs = layers.Dense(self.num_zones, activation='sigmoid', name='head_output')(x)

        # 모델 구성
        model = keras.Model(inputs=inputs, outputs=outputs, name='FedPer_GRU')

        return model

    def compile_model(
        self,
        learning_rate: float = 0.001,
        loss: str = 'binary_crossentropy',
        metrics: list = None
    ):
        """
        옵티마이저와 손실 함수로 모델을 컴파일합니다.

        Args:
            learning_rate: Adam 옵티마이저의 학습률
            loss: 손실 함수 (기본값: 다중 레이블 분류를 위한 binary_crossentropy)
            metrics: 추적할 평가지표 리스트
        """
        if metrics is None:
            metrics = ['accuracy', 'AUC', 'Precision', 'Recall']

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss,
            metrics=metrics
        )

        print("모델이 성공적으로 컴파일되었습니다.")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        verbose: int = 1
    ) -> keras.callbacks.History:
        """
        GRU 모델을 학습합니다.

        Args:
            X_train: 훈련 시퀀스 (N, 30, 160) - 어텐션 컨텍스트 벡터
            y_train: 훈련 레이블 (N, 7)
            X_val: 검증 시퀀스 (N, 30, 160) - 어텐션 컨텍스트 벡터
            y_val: 검증 레이블 (N, 7)
            epochs: 학습 에포크 수
            batch_size: 배치 크기
            verbose: 상세 정보 출력 수준

        Returns:
            학습 기록
        """
        print("=" * 70)
        print("GRU 모델 학습")
        print("=" * 70)
        print(f"훈련 샘플: {X_train.shape[0]}")
        print(f"검증 샘플: {X_val.shape[0]}")
        print(f"에포크: {epochs}, 배치 크기: {batch_size}")
        print()

        # 콜백 설정
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]

        # 학습
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )

        print("\n" + "=" * 70)
        print("학습 완료!")
        print("=" * 70)

        return history

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        오염 확률을 예측합니다.

        Args:
            X: 입력 시퀀스 (N, 30, 160) - 어텐션 컨텍스트 벡터
            threshold: 이진 분류를 위한 임계값

        Returns:
            (N, 7) 크기의 예측 확률 배열
        """
        return self.model.predict(X, verbose=0)

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        zone_names: list = None
    ) -> dict:
        """
        모델 성능을 평가합니다.

        Args:
            X_test: 테스트 시퀀스 (N, 30, 160) - 어텐션 컨텍스트 벡터
            y_test: 테스트 레이블 (N, 7)
            zone_names: 리포팅을 위한 구역 이름 리스트

        Returns:
            평가 지표를 담은 딕셔너리
        """
        print("\n" + "=" * 70)
        print("모델 평가")
        print("=" * 70)

        # 전체 평가지표
        results = self.model.evaluate(X_test, y_test, verbose=0)
        metric_names = self.model.metrics_names

        print("\n[전체 평가지표]")
        for name, value in zip(metric_names, results):
            print(f"  {name}: {value:.4f}")

        # 구역별 분석
        y_pred = self.predict(X_test)
        y_pred_binary = (y_pred > 0.5).astype(int)

        if zone_names is None:
            zone_names = [f"Zone_{i}" for i in range(self.num_zones)]

        print("\n[구역별 분석]")
        for i, zone in enumerate(zone_names):
            # 양성 샘플이 있는 구역만 분석
            if y_test[:, i].sum() > 0:
                true_pos = np.sum((y_test[:, i] == 1) & (y_pred_binary[:, i] == 1))
                false_pos = np.sum((y_test[:, i] == 0) & (y_pred_binary[:, i] == 1))
                false_neg = np.sum((y_test[:, i] == 1) & (y_pred_binary[:, i] == 0))

                precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
                recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                print(f"  {zone:15s}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
            else:
                print(f"  {zone:15s}: 양성 샘플 없음 (건너뜀)")

        print("=" * 70)

        return {
            'overall': dict(zip(metric_names, results)),
            'predictions': y_pred
        }

    def save(self, save_path: str):
        """모델을 파일에 저장합니다."""
        self.model.save(save_path)
        print(f"모델이 저장되었습니다: {save_path}")

    def load(self, load_path: str):
        """파일로부터 모델을 로드합니다."""
        self.model = keras.models.load_model(load_path)
        print(f"모델을 로드했습니다: {load_path}")

    def summary(self):
        """모델 요약 정보를 출력합니다."""
        print("\n" + "=" * 70)
        print("GRU 모델 아키텍처")
        print("=" * 70)
        self.model.summary()
        print("\n[파라미터 수]")

        # 기반 레이어와 헤드 레이어의 파라미터 수 계산
        base_params = 0
        head_params = 0

        for layer in self.model.layers:
            params = layer.count_params()
            if 'base' in layer.name:
                base_params += params
            elif 'head' in layer.name:
                head_params += params

        total_params = self.model.count_params()

        print(f"  기반 레이어:  {base_params:,} 파라미터 (~{base_params/1000:.1f}K)")
        print(f"  헤드 레이어:  {head_params:,} 파라미터 (~{head_params/1000:.1f}K)")
        print(f"  총합:       {total_params:,} 파라미터 (~{total_params/1000:.1f}K)")
        print("=" * 70)


def test_gru_model():
    """더미 데이터로 GRU 모델을 테스트합니다."""
    print("\n" + "=" * 70)
    print("어텐션 컨텍스트(160차원)를 사용한 GRU 모델 테스트")
    print("=" * 70 + "\n")

    # 더미 데이터 생성 (어텐션 컨텍스트 벡터)
    context_dim = 160
    X_train = np.random.randn(100, 30, context_dim).astype(np.float32)
    y_train = np.random.randint(0, 2, (100, 7)).astype(np.float32)
    X_val = np.random.randn(20, 30, context_dim).astype(np.float32)
    y_val = np.random.randint(0, 2, (20, 7)).astype(np.float32)

    print("[1] 모델 생성 중...")
    model = FedPerGRUModel(num_zones=7, context_dim=context_dim)
    model.summary()

    print("\n[2] 모델 컴파일 중...")
    model.compile_model(learning_rate=0.001)

    print("\n[3] 모델 학습 중 (테스트용 5 에포크)...")
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=5,
        batch_size=16,
        verbose=1
    )

    print("\n[4] 예측 테스트 중...")
    X_test = np.random.randn(10, 30, context_dim).astype(np.float32)
    y_pred = model.predict(X_test)
    print(f"  입력 shape: {X_test.shape}")
    print(f"  출력 shape: {y_pred.shape}")
    print(f"  샘플 예측: {y_pred[0]}")

    print("\n모든 테스트 통과!")


if __name__ == "__main__":
    test_gru_model()