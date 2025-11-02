"""
FR3 Training Script: Train GRU model on behavioral pattern data
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.dirname(__file__))

from dataset.dataset_builder import DatasetBuilder
from model.gru_model import FedPerGRUModel


def plot_training_history(history, save_path='training_history.png'):
    """
    Plot training history

    Args:
        history: Keras training history
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Loss
    axes[0, 0].plot(history.history['loss'], label='Train Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Accuracy
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

    # Precision
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
    print(f"\n✓ Training history plot saved to: {save_path}")


def main():
    """Main training pipeline"""
    print("\n" + "=" * 70)
    print("FR3: GRU Model Training Pipeline")
    print("=" * 70 + "\n")

    # ===== Step 1: Build Dataset =====
    print("[Step 1] Building dataset...")
    builder = DatasetBuilder()

    # Generate dataset (주방/거실 위주 - 7개 시나리오)
    X_train, y_train, X_val, y_val = builder.build_dataset(
        num_samples_per_scenario=100,  # 각 시나리오당 100개
        noise_level=0.1,
        train_split=0.8
    )

    # Save dataset for future use
    print("\n[Step 1.1] Saving dataset...")
    builder.save_dataset(X_train, y_train, X_val, y_val, "data/training_dataset.npz")

    # ===== Step 2: Create Model =====
    print("\n[Step 2] Creating GRU model...")
    model = FedPerGRUModel(num_zones=7)
    model.summary()

    # ===== Step 3: Compile Model =====
    print("\n[Step 3] Compiling model...")
    model.compile_model(
        learning_rate=0.001,
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC', 'Precision', 'Recall']
    )

    # ===== Step 4: Train Model =====
    print("\n[Step 4] Training model...")
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=50,
        batch_size=32,
        verbose=1
    )

    # ===== Step 5: Evaluate Model =====
    print("\n[Step 5] Evaluating model...")
    zone_names = builder.generator.zones
    results = model.evaluate(X_val, y_val, zone_names=zone_names)

    # ===== Step 6: Save Model =====
    print("\n[Step 6] Saving model...")
    os.makedirs("models", exist_ok=True)
    model.save("models/gru_model.keras")

    # ===== Step 7: Plot Training History =====
    print("\n[Step 7] Plotting training history...")
    os.makedirs("results", exist_ok=True)
    plot_training_history(history, save_path="results/training_history.png")

    # ===== Step 8: Test Prediction =====
    print("\n[Step 8] Testing prediction on sample...")
    sample_idx = 0
    X_sample = X_val[sample_idx:sample_idx+1]
    y_true = y_val[sample_idx]
    y_pred = model.predict(X_sample)[0]

    print(f"\nSample Prediction:")
    print(f"{'Zone':<15} {'True Label':<12} {'Predicted':<12} {'Match'}")
    print("-" * 55)
    for i, zone in enumerate(zone_names):
        match = "✓" if (y_true[i] > 0.5) == (y_pred[i] > 0.5) else "✗"
        print(f"{zone:<15} {y_true[i]:<12.0f} {y_pred[i]:<12.3f} {match}")

    print("\n" + "=" * 70)
    print("✓ Training Pipeline Complete!")
    print("=" * 70)
    print("\nSaved files:")
    print("  - data/training_dataset.npz (dataset)")
    print("  - models/gru_model.keras (trained model)")
    print("  - results/training_history.png (plots)")


if __name__ == "__main__":
    main()
