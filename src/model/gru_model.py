"""
FR3: GRU Model for Sequential Pattern Learning
Predicts cleaning needs based on behavioral pattern causality
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Tuple, Optional


class FedPerGRUModel:
    """
    FR3.2: FedPer Architecture GRU Model with Attention Context Encoder

    Architecture:
    - Base Layer (shared): GRU(64) → GRU(32)
    - Head Layer (personalized): Dense(16) → Dense(7, sigmoid)

    Input: (batch, 30, 160) - 30 timesteps of 160-dim attention context vectors
    Output: (batch, 7) - Pollution probability for 7 zones
    """

    def __init__(self, num_zones: int = 7, context_dim: int = 160):
        """
        Initialize GRU model

        Args:
            num_zones: Number of semantic zones (default: 7)
            context_dim: Context vector dimension (default: 160 from AttentionContextEncoder)
        """
        self.num_zones = num_zones
        self.context_dim = context_dim
        self.model = self._build_model()
        self.base_model = None  # For FedPer: shared base layers
        self.head_model = None  # For FedPer: personalized head layers

    def _build_model(self) -> keras.Model:
        """
        FR3.2: Build complete GRU model

        Returns:
            Compiled Keras model
        """
        # Input: 30 timesteps of attention context vectors
        inputs = layers.Input(shape=(30, self.context_dim), name='context_sequence')

        # ===== FR3.2a: Base Layer (Shared Feature Extraction) =====
        # GRU Layer 1: 64 units, return sequences for next GRU
        x = layers.GRU(64, return_sequences=True, name='base_gru1')(inputs)

        # GRU Layer 2: 32 units, final representation
        base_output = layers.GRU(32, return_sequences=False, name='base_gru2')(x)

        # ===== FR3.2b: Head Layer (Personalized Prediction) =====
        # Dense Layer 1: 16 units, ReLU
        x = layers.Dense(16, activation='relu', name='head_dense1')(base_output)

        # Dropout: 0.3 (prevent overfitting)
        x = layers.Dropout(0.3, name='head_dropout')(x)

        # Output Layer: 7 zones, Sigmoid (multi-label binary classification)
        outputs = layers.Dense(self.num_zones, activation='sigmoid', name='head_output')(x)

        # Build model
        model = keras.Model(inputs=inputs, outputs=outputs, name='FedPer_GRU')

        return model

    def compile_model(
        self,
        learning_rate: float = 0.001,
        loss: str = 'binary_crossentropy',
        metrics: list = None
    ):
        """
        Compile model with optimizer and loss

        Args:
            learning_rate: Learning rate for Adam optimizer
            loss: Loss function (default: binary_crossentropy for multi-label)
            metrics: List of metrics to track
        """
        if metrics is None:
            metrics = ['accuracy', 'AUC', 'Precision', 'Recall']

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss,
            metrics=metrics
        )

        print("✓ Model compiled successfully")

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
        Train GRU model

        Args:
            X_train: Training sequences (N, 30, 160) - attention context vectors
            y_train: Training labels (N, 7)
            X_val: Validation sequences (N, 30, 160) - attention context vectors
            y_val: Validation labels (N, 7)
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Verbosity level

        Returns:
            Training history
        """
        print("=" * 70)
        print("FR3: Training GRU Model")
        print("=" * 70)
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Validation samples: {X_val.shape[0]}")
        print(f"Epochs: {epochs}, Batch size: {batch_size}")
        print()

        # Callbacks
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

        # Train
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )

        print("\n" + "=" * 70)
        print("✓ Training Complete!")
        print("=" * 70)

        return history

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict pollution probabilities

        Args:
            X: Input sequences (N, 30, 160) - attention context vectors
            threshold: Threshold for binary classification

        Returns:
            Predictions (N, 7) with probabilities
        """
        return self.model.predict(X, verbose=0)

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        zone_names: list = None
    ) -> dict:
        """
        Evaluate model performance

        Args:
            X_test: Test sequences (N, 30, 160) - attention context vectors
            y_test: Test labels (N, 7)
            zone_names: List of zone names for reporting

        Returns:
            Evaluation metrics dict
        """
        print("\n" + "=" * 70)
        print("FR3: Model Evaluation")
        print("=" * 70)

        # Overall metrics
        results = self.model.evaluate(X_test, y_test, verbose=0)
        metric_names = self.model.metrics_names

        print("\n[Overall Metrics]")
        for name, value in zip(metric_names, results):
            print(f"  {name}: {value:.4f}")

        # Per-zone analysis
        y_pred = self.predict(X_test)
        y_pred_binary = (y_pred > 0.5).astype(int)

        if zone_names is None:
            zone_names = [f"Zone_{i}" for i in range(self.num_zones)]

        print("\n[Per-Zone Analysis]")
        for i, zone in enumerate(zone_names):
            # Only analyze zones with positive samples
            if y_test[:, i].sum() > 0:
                true_pos = np.sum((y_test[:, i] == 1) & (y_pred_binary[:, i] == 1))
                false_pos = np.sum((y_test[:, i] == 0) & (y_pred_binary[:, i] == 1))
                false_neg = np.sum((y_test[:, i] == 1) & (y_pred_binary[:, i] == 0))

                precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
                recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                print(f"  {zone:15s}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
            else:
                print(f"  {zone:15s}: No positive samples (skipped)")

        print("=" * 70)

        return {
            'overall': dict(zip(metric_names, results)),
            'predictions': y_pred
        }

    def save(self, save_path: str):
        """Save model to file"""
        self.model.save(save_path)
        print(f"✓ Model saved to: {save_path}")

    def load(self, load_path: str):
        """Load model from file"""
        self.model = keras.models.load_model(load_path)
        print(f"✓ Model loaded from: {load_path}")

    def summary(self):
        """Print model summary"""
        print("\n" + "=" * 70)
        print("FR3.2: GRU Model Architecture")
        print("=" * 70)
        self.model.summary()
        print("\n[Parameter Count]")

        # Calculate base vs head parameters
        base_params = 0
        head_params = 0

        for layer in self.model.layers:
            params = layer.count_params()
            if 'base' in layer.name:
                base_params += params
            elif 'head' in layer.name:
                head_params += params

        total_params = self.model.count_params()

        print(f"  Base Layer:  {base_params:,} parameters (~{base_params/1000:.1f}K)")
        print(f"  Head Layer:  {head_params:,} parameters (~{head_params/1000:.1f}K)")
        print(f"  Total:       {total_params:,} parameters (~{total_params/1000:.1f}K)")
        print("=" * 70)


def test_gru_model():
    """Test GRU model with dummy data"""
    print("\n" + "=" * 70)
    print("Testing GRU Model with Attention Context (160-dim)")
    print("=" * 70 + "\n")

    # Create dummy data (attention context vectors)
    context_dim = 160
    X_train = np.random.randn(100, 30, context_dim).astype(np.float32)
    y_train = np.random.randint(0, 2, (100, 7)).astype(np.float32)
    X_val = np.random.randn(20, 30, context_dim).astype(np.float32)
    y_val = np.random.randint(0, 2, (20, 7)).astype(np.float32)

    print("[1] Creating model...")
    model = FedPerGRUModel(num_zones=7, context_dim=context_dim)
    model.summary()

    print("\n[2] Compiling model...")
    model.compile_model(learning_rate=0.001)

    print("\n[3] Training model (5 epochs for testing)...")
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=5,
        batch_size=16,
        verbose=1
    )

    print("\n[4] Testing prediction...")
    X_test = np.random.randn(10, 30, context_dim).astype(np.float32)
    y_pred = model.predict(X_test)
    print(f"  Input shape: {X_test.shape}")
    print(f"  Output shape: {y_pred.shape}")
    print(f"  Sample prediction: {y_pred[0]}")

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_gru_model()