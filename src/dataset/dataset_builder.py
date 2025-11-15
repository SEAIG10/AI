"""
FR6.3: Dataset Builder with Attention Context Encoder
Converts scenarios to training-ready (X, y) numerical datasets
"""

import numpy as np
import os
import sys
import tensorflow as tf
from typing import List, Dict, Tuple

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from context_fusion.context_encoder import ContextEncoder
from context_fusion.attention_context_encoder import create_attention_encoder
from dataset.scenario_generator import ScenarioGenerator


class DatasetBuilder:
    """
    FR6.3: Build training datasets from scenarios with Attention Context Encoder

    Converts JSON scenarios → (X, y) numerical data
    X: (N, 30, 160) - N sequences, each 30 timesteps of 160-dim attention context vectors
    y: (N, 7) - N labels, 7 zones (binary: dirty or clean)
    """

    def __init__(self, use_attention: bool = True):
        """
        Initialize dataset builder

        Args:
            use_attention: Use AttentionContextEncoder (160-dim) vs legacy ContextEncoder (338-dim)
        """
        self.use_attention = use_attention
        self.legacy_encoder = ContextEncoder()  # For extracting raw features
        self.generator = ScenarioGenerator()

        if use_attention:
            # Create attention encoder
            self.attention_encoder = create_attention_encoder(
                visual_dim=14,
                context_dim=160
            )

    def scenario_to_training_sample(self, scenario: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert single scenario to (X, y) training sample

        Args:
            scenario: Scenario dict from ScenarioGenerator

        Returns:
            X: (30, 160) or (30, 338) sequence depending on use_attention
            y: (7,) label
        """
        if not self.use_attention:
            # Legacy mode: use ContextEncoder (338-dim)
            sequence_vectors = []
            for timestep in scenario["sequence"]:
                vector = self.legacy_encoder.encode(timestep)
                sequence_vectors.append(vector)

            X = np.array(sequence_vectors, dtype=np.float32)  # (30, 338)
            y = np.array(scenario["label"], dtype=np.float32)  # (7,)

            assert X.shape == (30, 338), f"Expected X shape (30, 338), got {X.shape}"
            assert y.shape == (7,), f"Expected y shape (7,), got {y.shape}"

            return X, y

        # Attention mode: use AttentionContextEncoder (160-dim)
        # First, prepare raw features for each timestep
        batch_features = {
            'visual': [],
            'audio': [],
            'pose': [],
            'spatial': [],
            'time': []
        }

        for timestep in scenario["sequence"]:
            # Extract raw features using legacy encoder's helper methods
            visual_vec = self.legacy_encoder._encode_visual(timestep)     # (14,)
            spatial_vec = self.legacy_encoder._encode_spatial(timestep)   # (7,)
            time_vec = self.legacy_encoder._encode_time(timestep.get("timestamp", 0))  # (10,)

            # Mock audio classification and pose keypoints (for mock data)
            # In production, these would come from YAMNet (17-class) and YOLO-Pose
            audio_vec = np.random.uniform(0, 0.3, 17).astype(np.float32)  # (17,) - 17-class probabilities
            pose_vec = np.zeros(51, dtype=np.float32)  # (51,) - no person detected in mock

            batch_features['visual'].append(visual_vec)
            batch_features['audio'].append(audio_vec)
            batch_features['pose'].append(pose_vec)
            batch_features['spatial'].append(spatial_vec)
            batch_features['time'].append(time_vec)

        # Convert to tensors (batch_size=30)
        batch_tensors = {
            'visual': tf.constant(np.array(batch_features['visual']), dtype=tf.float32),    # (30, 14)
            'audio': tf.constant(np.array(batch_features['audio']), dtype=tf.float32),      # (30, 17)
            'pose': tf.constant(np.array(batch_features['pose']), dtype=tf.float32),        # (30, 51)
            'spatial': tf.constant(np.array(batch_features['spatial']), dtype=tf.float32),  # (30, 7)
            'time': tf.constant(np.array(batch_features['time']), dtype=tf.float32),        # (30, 10)
        }

        # Pass through attention encoder
        context_vectors = self.attention_encoder(batch_tensors, training=False)  # (30, 160)

        X = context_vectors.numpy().astype(np.float32)  # (30, 160)
        y = np.array(scenario["label"], dtype=np.float32)  # (7,)

        assert X.shape == (30, 160), f"Expected X shape (30, 160), got {X.shape}"
        assert y.shape == (7,), f"Expected y shape (7,), got {y.shape}"

        return X, y

    def build_dataset(
        self,
        num_samples_per_scenario: int = 100,
        noise_level: float = 0.1,
        train_split: float = 0.8
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Build complete training dataset with data augmentation

        FR6.3a, FR6.3b, FR6.3c implementation

        Args:
            num_samples_per_scenario: Number of variations per scenario
            noise_level: Noise intensity for augmentation (0.0 - 1.0)
            train_split: Train/val split ratio

        Returns:
            X_train: (N_train, 30, 160) - attention context vectors
            y_train: (N_train, 7)
            X_val: (N_val, 30, 160) - attention context vectors
            y_val: (N_val, 7)
        """
        print("=" * 70)
        print("FR6.3: Building Training Dataset")
        print("=" * 70)

        # Generate base scenarios
        scenarios = self.generator.generate_all_scenarios()
        print(f"\n[1] Base scenarios: {len(scenarios)}")

        X_all = []
        y_all = []

        # Generate samples with augmentation
        print(f"[2] Generating {num_samples_per_scenario} samples per scenario...")

        for scenario_idx, scenario in enumerate(scenarios):
            print(f"    - {scenario['name']}: ", end="")

            for sample_idx in range(num_samples_per_scenario):
                # Add noise for augmentation (except first sample = clean)
                if sample_idx == 0:
                    augmented_scenario = scenario
                else:
                    augmented_scenario = self.generator.add_noise(scenario, noise_level)

                # Convert to numerical
                X, y = self.scenario_to_training_sample(augmented_scenario)
                X_all.append(X)
                y_all.append(y)

            print(f"{num_samples_per_scenario} samples ✓")

        # Convert to numpy arrays
        X_all = np.array(X_all, dtype=np.float32)
        y_all = np.array(y_all, dtype=np.float32)

        print(f"\n[3] Total dataset size:")
        print(f"    - X: {X_all.shape}")
        print(f"    - y: {y_all.shape}")

        # Train/val split
        num_train = int(len(X_all) * train_split)
        indices = np.random.permutation(len(X_all))

        train_indices = indices[:num_train]
        val_indices = indices[num_train:]

        X_train = X_all[train_indices]
        y_train = y_all[train_indices]
        X_val = X_all[val_indices]
        y_val = y_all[val_indices]

        print(f"\n[4] Split:")
        print(f"    - Train: {X_train.shape[0]} samples")
        print(f"    - Val:   {X_val.shape[0]} samples")

        # Statistics
        print(f"\n[5] Label statistics:")
        for zone_idx, zone_name in enumerate(self.generator.zones):
            positive_train = np.sum(y_train[:, zone_idx])
            positive_val = np.sum(y_val[:, zone_idx])
            print(f"    - {zone_name:15s}: Train={int(positive_train):3d}, Val={int(positive_val):3d}")

        print("\n" + "=" * 70)
        print("✓ FR6.3 Dataset Building Complete!")
        print("=" * 70)

        return X_train, y_train, X_val, y_val

    def save_dataset(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        save_path: str = "data/training_dataset.npz"
    ):
        """
        FR6.3b: Save dataset to .npz file

        Args:
            X_train, y_train, X_val, y_val: Dataset arrays
            save_path: Path to save .npz file
        """
        # Create directory if needed
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"Created directory: {save_dir}")

        # Save
        np.savez_compressed(
            save_path,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val
        )

        file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
        print(f"\n✓ Dataset saved to: {save_path}")
        print(f"  File size: {file_size_mb:.2f} MB")

    def load_dataset(self, load_path: str = "data/training_dataset.npz"):
        """
        Load dataset from .npz file

        Args:
            load_path: Path to .npz file

        Returns:
            X_train, y_train, X_val, y_val
        """
        data = np.load(load_path)

        X_train = data['X_train']
        y_train = data['y_train']
        X_val = data['X_val']
        y_val = data['y_val']

        print(f"✓ Dataset loaded from: {load_path}")
        print(f"  Train: {X_train.shape[0]} samples")
        print(f"  Val:   {X_val.shape[0]} samples")

        return X_train, y_train, X_val, y_val


def test_dataset_builder():
    """Test dataset builder"""
    print("\n" + "=" * 70)
    print("Testing Dataset Builder")
    print("=" * 70 + "\n")

    builder = DatasetBuilder()

    # Build dataset (small for testing)
    X_train, y_train, X_val, y_val = builder.build_dataset(
        num_samples_per_scenario=50,  # 50 samples per scenario
        noise_level=0.1,
        train_split=0.8
    )

    # Verify shapes
    print(f"\n[Verification]")
    print(f"X_train shape: {X_train.shape} (expected: (N, 30, 160))")
    print(f"y_train shape: {y_train.shape} (expected: (N, 7))")
    print(f"X_val shape: {X_val.shape}")
    print(f"y_val shape: {y_val.shape}")

    # Sample inspection
    print(f"\n[Sample Inspection]")
    print(f"X_train[0][0][:10] = {X_train[0][0][:10]}")
    print(f"y_train[0] = {y_train[0]} (zones: {builder.generator.zones})")

    # Save test
    print(f"\n[Save Test]")
    builder.save_dataset(X_train, y_train, X_val, y_val, "data/test_dataset.npz")

    # Load test
    print(f"\n[Load Test]")
    X_train_loaded, y_train_loaded, X_val_loaded, y_val_loaded = builder.load_dataset("data/test_dataset.npz")

    # Verify loaded data matches
    assert np.allclose(X_train, X_train_loaded), "Loaded X_train doesn't match!"
    assert np.allclose(y_train, y_train_loaded), "Loaded y_train doesn't match!"
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_dataset_builder()
