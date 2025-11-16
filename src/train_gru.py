"""
GRU 모델 학습 스크립트
행동 패턴 데이터를 사용하여 GRU 모델을 학습합니다.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# src 경로 추가
sys.path.append(os.path.dirname(__file__))

from dataset.dataset_builder import DatasetBuilder
from model.gru_model import FedPerGRUModel


def plot_training_history(history, save_path='training_history.png'):
    """
    학습 기록을 시각화합니다.

    Args:
        history: Keras 학습 기록 객체
        save_path: 그래프를 저장할 경로
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 손실
    axes[0, 0].plot(history.history['loss'], label='훈련 손실')
    axes[0, 0].plot(history.history['val_loss'], label='검증 손실')
    axes[0, 0].set_title('모델 손실')
    axes[0, 0].set_xlabel('에포크')
    axes[0, 0].set_ylabel('손실')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # 정확도
    axes[0, 1].plot(history.history['accuracy'], label='훈련 정확도')
    axes[0, 1].plot(history.history['val_accuracy'], label='검증 정확도')
    axes[0, 1].set_title('모델 정확도')
    axes[0, 1].set_xlabel('에포크')
    axes[0, 1].set_ylabel('정확도')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # AUC
    if 'auc' in history.history:
        axes[1, 0].plot(history.history['auc'], label='훈련 AUC')
        axes[1, 0].plot(history.history['val_auc'], label='검증 AUC')
        axes[1, 0].set_title('모델 AUC')
        axes[1, 0].set_xlabel('에포크')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

    # 정밀도
    if 'precision' in history.history:
        axes[1, 1].plot(history.history['precision'], label='훈련 정밀도')
        axes[1, 1].plot(history.history['val_precision'], label='검증 정밀도')
        axes[1, 1].set_title('모델 정밀도')
        axes[1, 1].set_xlabel('에포크')
        axes[1, 1].set_ylabel('정밀도')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n학습 기록 그래프가 저장되었습니다: {save_path}")


def main():
    """메인 학습 파이프라인"""
    print("\n" + "=" * 70)
    print("GRU 모델 학습 파이프라인")
    print("=" * 70 + "\n")

    # ===== 단계 1: 데이터셋 구축 =====
    print("[단계 1] AttentionContextEncoder(160차원)를 사용하여 데이터셋 구축 중...")
    builder = DatasetBuilder(use_attention=True)  # 160차원 어텐션 컨텍스트 사용

    # 데이터셋 생성 (7개 시나리오 × 143개 샘플 = 약 1000개 샘플)
    X_train, y_train, X_val, y_val = builder.build_dataset(
        num_samples_per_scenario=143,
        noise_level=0.1,
        train_split=0.8
    )

    # 나중에 재사용을 위해 데이터셋 저장
    print("\n[단계 1.1] 데이터셋 저장 중...")
    builder.save_dataset(X_train, y_train, X_val, y_val, "data/training_dataset.npz")

    # ===== 단계 2: 모델 생성 =====
    print("\n[단계 2] 160차원 컨텍스트 입력을 받는 GRU 모델 생성 중...")
    model = FedPerGRUModel(num_zones=7, context_dim=160)  # 160차원 어텐션 컨텍스트
    model.summary()

    # ===== 단계 3: 모델 컴파일 =====
    print("\n[단계 3] 모델 컴파일 중...")
    model.compile_model(
        learning_rate=0.001,
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC', 'Precision', 'Recall']
    )

    # ===== 단계 4: 모델 학습 =====
    print("\n[단계 4] 모델 학습 중...")
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=50,
        batch_size=32,
        verbose=1
    )

    # ===== 단계 5: 모델 평가 =====
    print("\n[단계 5] 모델 평가 중...")
    zone_names = builder.generator.zones
    results = model.evaluate(X_val, y_val, zone_names=zone_names)

    # ===== 단계 6: 모델 저장 =====
    print("\n[단계 6] 모델 저장 중...")
    os.makedirs("models/gru", exist_ok=True)
    model.save("models/gru/gru_model.keras")

    # ===== 단계 7: 학습 기록 시각화 =====
    print("\n[단계 7] 학습 기록 시각화 중...")
    os.makedirs("results", exist_ok=True)
    plot_training_history(history, save_path="results/training_history.png")

    # ===== 단계 8: 예측 테스트 =====
    print("\n[단계 8] 샘플 데이터로 예측 테스트 중...")
    sample_idx = 0
    X_sample = X_val[sample_idx:sample_idx+1]
    y_true = y_val[sample_idx]
    y_pred = model.predict(X_sample)[0]

    print(f"\n샘플 예측 결과:")
    print(f"{'구역':<15} {'실제 레이블':<12} {'예측 확률':<12} {'일치'}")
    print("-" * 55)
    for i, zone in enumerate(zone_names):
        match = "O" if (y_true[i] > 0.5) == (y_pred[i] > 0.5) else "X"
        print(f"{zone:<15} {y_true[i]:<12.0f} {y_pred[i]:<12.3f} {match}")

    print("\n" + "=" * 70)
    print("학습 파이프라인 완료!")
    print("=" * 70)
    print("\n저장된 파일:")
    print("  - data/training_dataset.npz (데이터셋)")
    print("  - models/gru/gru_model.keras (학습된 모델)")
    print("  - results/training_history.png (학습 그래프)")


if __name__ == "__main__":
    main()
