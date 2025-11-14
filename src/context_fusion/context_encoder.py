"""
FR2.3.3: Context Encoder
Converts JSON Context Vectors to 108-dimensional Numerical Vectors for GRU training
"""

import numpy as np
from datetime import datetime
from typing import Dict, List, Optional


class ContextEncoder:
    """
    FR2.3.3: Encodes JSON Context Vectors into fixed 338-dimensional numerical vectors

    Vector composition:
    - Time features: 10 dimensions
    - Spatial features: 7 dimensions (zone one-hot)
    - Visual features: 14 dimensions (Custom YOLO 14 classes, multi-hot)
    - Audio features: 256 dimensions (YAMNet-256 embedding)
    - Keypoints features: 51 dimensions (raw normalized pose)
    Total: 338 dimensions
    """

    def __init__(self):
        """Initialize encoder with feature mappings"""

        # FR1.2: 7 semantic zones (alphabetical order for consistency)
        self.zone_labels = [
            "bathroom",      # 0
            "bedroom_1",     # 1
            "bedroom_2",     # 2
            "corridor",      # 3
            "garden_balcony",# 4
            "kitchen",       # 5
            "living_room"    # 6
        ]
        self.zone_to_idx = {zone: idx for idx, zone in enumerate(self.zone_labels)}

        # Custom YOLO 14 classes (HomeObjects-3K + HD10K)
        # See: /datasets/vacuum_cleaner.yaml
        self.visual_classes = [
            "bed",           # 0 - furniture
            "sofa",          # 1 - furniture
            "chair",         # 2 - furniture
            "table",         # 3 - furniture
            "lamp",          # 4 - object
            "tv",            # 5 - electronics
            "laptop",        # 6 - electronics
            "wardrobe",      # 7 - furniture
            "window",        # 8 - structure
            "door",          # 9 - structure
            "potted plant",  # 10 - decoration
            "photo frame",   # 11 - decoration
            "solid_waste",   # 12 - dirt (HD10K: crumbs, dust, trash)
            "liquid_stain"   # 13 - dirt (HD10K: spills, water stains)
        ]
        self.visual_to_idx = {cls: idx for idx, cls in enumerate(self.visual_classes)}

        # Audio: YAMNet-256 embedding (256 dimensions)
        self.audio_embedding_dim = 256

        # Time of day categories (5 categories)
        self.time_of_day_labels = ["dawn", "morning", "afternoon", "evening", "night"]

    def encode(self, context: Dict) -> np.ndarray:
        """
        Encode JSON context vector to 338-dimensional numerical vector

        Args:
            context: JSON context dict from ContextVector.create_context()

        Returns:
            338-dimensional numpy array [time(10) + spatial(7) + visual(14) + audio(256) + keypoints(51)]
        """
        # Extract timestamp
        timestamp = context.get("timestamp", 0)

        # Encode each component
        time_features = self._encode_time(timestamp)        # 10 dim
        spatial_features = self._encode_spatial(context)    # 7 dim
        visual_features = self._encode_visual(context)      # 14 dim (Custom YOLO)
        audio_features = self._encode_audio(context)        # 256 dim (YAMNet-256 embedding)
        keypoints_features = self._encode_keypoints(context)  # 51 dim

        # Concatenate all features
        vector = np.concatenate([
            time_features,
            spatial_features,
            visual_features,
            audio_features,
            keypoints_features
        ])

        assert vector.shape == (338,), f"Expected shape (338,), got {vector.shape}"
        return vector.astype(np.float32)

    def _encode_time(self, timestamp: float) -> np.ndarray:
        """
        FR2.3.3(a): Encode time features (10 dimensions)

        Returns:
            [hour_sin, hour_cos, dow_sin, dow_cos, is_weekend,
             is_meal_time, is_work_time, tod_dawn, tod_morning, tod_afternoon, tod_evening, tod_night]

        Wait, that's 12 dimensions. Let me fix:
        Actually time_of_day should use 5 dimensions but we need to make it 10 total.
        Let me recalculate:
        - hour_sin, hour_cos: 2
        - dow_sin, dow_cos: 2
        - is_weekend: 1
        - is_meal_time: 1
        - is_work_time: 1
        - time_of_day one-hot: 5
        Total = 12 dimensions

        But doc says 10 dimensions. Let me check...
        Looking at the doc again, it says "~10차원" so approximately 10.
        But let me be more precise. The doc lists:
        - hour: cyclic (2)
        - day_of_week: cyclic (2)
        - is_weekend: (1)
        - is_meal_time: (1)
        - is_work_time: (1)
        - time_of_day: one-hot [5] (5)
        Total = 2+2+1+1+1+5 = 12

        Hmm, but the total should be 108. Let me recalculate:
        10 + 7 + 85 + 6 = 108 ✓

        So time features should be exactly 10. Let me adjust:
        Option 1: Remove time_of_day one-hot (keep it simpler)
        - hour_sin, hour_cos: 2
        - dow_sin, dow_cos: 2
        - is_weekend: 1
        - is_meal_time: 1
        - is_work_time: 1
        - hour_of_day (normalized): 1
        - day_of_week (normalized): 1
        - month (normalized): 1
        Total = 10 ✓

        Let me do this version for simplicity.
        """
        dt = datetime.fromtimestamp(timestamp)

        # Cyclic encoding for hour (0-23)
        hour = dt.hour
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)

        # Cyclic encoding for day of week (0=Monday, 6=Sunday)
        dow = dt.weekday()
        dow_sin = np.sin(2 * np.pi * dow / 7)
        dow_cos = np.cos(2 * np.pi * dow / 7)

        # Binary features
        is_weekend = 1.0 if dow >= 5 else 0.0

        # Meal time: 07-09, 12-14, 18-20
        is_meal_time = 1.0 if (7 <= hour <= 9) or (12 <= hour <= 14) or (18 <= hour <= 20) else 0.0

        # Work time: 09-18, weekdays only
        is_work_time = 1.0 if (9 <= hour <= 18 and dow < 5) else 0.0

        # Normalized features (0-1 range)
        hour_normalized = hour / 24.0
        dow_normalized = dow / 7.0
        month_normalized = dt.month / 12.0

        return np.array([
            hour_sin, hour_cos,
            dow_sin, dow_cos,
            is_weekend,
            is_meal_time,
            is_work_time,
            hour_normalized,
            dow_normalized,
            month_normalized
        ], dtype=np.float32)

    def _encode_spatial(self, context: Dict) -> np.ndarray:
        """
        FR2.3.3(b): Encode spatial features (7 dimensions)
        One-hot encoding of zone_id

        Returns:
            7-dimensional one-hot vector
        """
        zone_id = context.get("zone_id", "unknown")

        # One-hot encoding
        vector = np.zeros(7, dtype=np.float32)
        if zone_id in self.zone_to_idx:
            vector[self.zone_to_idx[zone_id]] = 1.0

        return vector

    def _encode_visual(self, context: Dict) -> np.ndarray:
        """
        FR2.3.3(c): Encode visual features (14 dimensions)
        Multi-hot encoding of detected object classes from Custom YOLO

        Returns:
            14-dimensional multi-hot vector (multiple 1s possible)
        """
        visual_events = context.get("visual_events", [])

        # Multi-hot encoding
        vector = np.zeros(14, dtype=np.float32)

        for event in visual_events:
            class_name = event.get("class", "")
            if class_name in self.visual_to_idx:
                vector[self.visual_to_idx[class_name]] = 1.0

        return vector

    def _encode_audio(self, context: Dict) -> np.ndarray:
        """
        FR2.3.3(d): Encode audio features (256 dimensions)
        Direct pass-through of YAMNet-256 embedding

        Returns:
            256-dimensional audio embedding vector (continuous values)
        """
        audio_embedding = context.get("audio_embedding", None)

        # If no audio embedding, return zeros (silence)
        if audio_embedding is None:
            return np.zeros(self.audio_embedding_dim, dtype=np.float32)

        # Convert to numpy array if needed
        if isinstance(audio_embedding, list):
            embedding_array = np.array(audio_embedding, dtype=np.float32)
        else:
            embedding_array = audio_embedding.astype(np.float32)

        # Ensure correct shape
        if embedding_array.shape != (self.audio_embedding_dim,):
            print(f"Warning: Expected audio embedding shape ({self.audio_embedding_dim},), got {embedding_array.shape}. Using zeros.")
            return np.zeros(self.audio_embedding_dim, dtype=np.float32)

        return embedding_array

    def _encode_keypoints(self, context: Dict) -> np.ndarray:
        """
        Encode raw pose keypoints (51 dimensions)
        Direct pass-through of normalized keypoint data

        Returns:
            51-dimensional vector (17 keypoints × 3 values each)
        """
        keypoints_data = context.get("keypoints", None)

        # If no keypoints, return zeros
        if keypoints_data is None:
            return np.zeros(51, dtype=np.float32)

        # Convert to numpy array if needed
        if isinstance(keypoints_data, list):
            keypoints_array = np.array(keypoints_data, dtype=np.float32)
        else:
            keypoints_array = keypoints_data.astype(np.float32)

        # Ensure correct shape
        if keypoints_array.shape != (51,):
            print(f"Warning: Expected keypoints shape (51,), got {keypoints_array.shape}. Using zeros.")
            return np.zeros(51, dtype=np.float32)

        return keypoints_array

    def encode_batch(self, contexts: List[Dict]) -> np.ndarray:
        """
        Encode a batch of contexts

        Args:
            contexts: List of JSON context dicts

        Returns:
            (N, 338) numpy array
        """
        return np.array([self.encode(ctx) for ctx in contexts], dtype=np.float32)

    def get_feature_names(self) -> List[str]:
        """
        Get human-readable feature names for debugging

        Returns:
            List of 338 feature names
        """
        names = []

        # Time features (10)
        names.extend([
            "hour_sin", "hour_cos",
            "dow_sin", "dow_cos",
            "is_weekend", "is_meal_time", "is_work_time",
            "hour_norm", "dow_norm", "month_norm"
        ])

        # Spatial features (7)
        names.extend([f"zone_{zone}" for zone in self.zone_labels])

        # Visual features (14) - Custom YOLO
        names.extend([f"obj_{cls}" for cls in self.visual_classes])

        # Audio features (256) - YAMNet-256 embedding
        names.extend([f"audio_emb_{i}" for i in range(256)])

        # Keypoints features (51)
        keypoint_names = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]
        for kpt_name in keypoint_names:
            names.extend([f"kpt_{kpt_name}_x", f"kpt_{kpt_name}_y", f"kpt_{kpt_name}_conf"])

        return names

    def decode_debug(self, vector: np.ndarray) -> Dict:
        """
        Convert numerical vector back to human-readable format (for debugging)

        Args:
            vector: 338-dimensional numpy array

        Returns:
            Dict with decoded features
        """
        feature_names = self.get_feature_names()

        # Time features (0-9)
        time_features = {
            "hour_sin": vector[0],
            "hour_cos": vector[1],
            "estimated_hour": int((np.arctan2(vector[0], vector[1]) / (2 * np.pi) * 24) % 24),
            "is_weekend": bool(vector[4] > 0.5),
            "is_meal_time": bool(vector[5] > 0.5),
            "is_work_time": bool(vector[6] > 0.5)
        }

        # Spatial features (10-16)
        zone_vector = vector[10:17]
        zone_idx = np.argmax(zone_vector) if np.max(zone_vector) > 0 else None
        spatial_features = {
            "zone": self.zone_labels[zone_idx] if zone_idx is not None else "unknown"
        }

        # Visual features (17-30)
        visual_vector = vector[17:31]
        detected_objects = [self.visual_classes[i] for i, v in enumerate(visual_vector) if v > 0.5]
        visual_features = {
            "objects": detected_objects,
            "count": len(detected_objects)
        }

        # Audio features (31-286)
        audio_embedding = vector[31:287]
        has_audio = np.any(audio_embedding != 0)
        audio_features = {
            "has_audio": has_audio,
            "embedding_norm": float(np.linalg.norm(audio_embedding)),
            "embedding_mean": float(np.mean(audio_embedding)),
            "embedding_shape": "(256,)"
        }

        # Keypoints features (287-337)
        keypoints_vector = vector[287:338]
        has_pose = np.any(keypoints_vector != 0)
        keypoints_features = {
            "has_pose": has_pose,
            "non_zero_values": int(np.count_nonzero(keypoints_vector)),
            "raw_data_shape": "(51,)"
        }

        return {
            "time": time_features,
            "spatial": spatial_features,
            "visual": visual_features,
            "audio": audio_features,
            "keypoints": keypoints_features
        }


def test_context_encoder():
    """Test context encoder"""
    print("=" * 70)
    print("FR2.3.3: Context Encoder Test")
    print("=" * 70)

    encoder = ContextEncoder()

    # Test context from our previous example
    test_context = {
        "timestamp": 1678886400.0,  # 2023-03-15 12:00:00 UTC
        "position": {"x": -5.0, "y": -3.0},
        "zone": "거실 (Living Room)",
        "zone_id": "living_room",
        "visual_events": [
            {"class": "person", "confidence": 0.92, "bbox": (100, 150, 200, 400)},
            {"class": "couch", "confidence": 0.85, "bbox": (50, 200, 300, 450)},
            {"class": "remote", "confidence": 0.78, "bbox": (180, 350, 220, 380)},
        ],
        "audio_events": [
            {"event": "Television", "confidence": 0.88}
        ],
        "context_summary": "living_room | 3 objects | Television"
    }

    print("\n[1] Original JSON Context:")
    print(f"  Zone: {test_context['zone_id']}")
    print(f"  Visual: {[e['class'] for e in test_context['visual_events']]}")
    print(f"  Audio: {test_context['audio_events'][0]['event']}")

    print("\n[2] Encoding to Numerical Vector...")
    vector = encoder.encode(test_context)
    print(f"  Shape: {vector.shape}")
    print(f"  Dtype: {vector.dtype}")
    print(f"  First 20 values: {vector[:20]}")

    print("\n[3] Feature Breakdown:")
    print(f"  Time features (0-9): {vector[0:10]}")
    print(f"  Spatial features (10-16): {vector[10:17]}")
    print(f"  Visual features (17-30): {vector[17:31].sum()} objects detected")
    print(f"  Audio features (31-286): {vector[31:287].sum():.2f} embedding norm")
    print(f"  Keypoints features (287-337): {vector[287:338].sum():.2f} pose sum")

    print("\n[4] Decode Debug:")
    decoded = encoder.decode_debug(vector)
    print(f"  Estimated hour: {decoded['time']['estimated_hour']}")
    print(f"  Zone: {decoded['spatial']['zone']}")
    print(f"  Objects: {decoded['visual']['objects']}")
    print(f"  Has audio: {decoded['audio']['has_audio']}")
    print(f"  Has pose: {decoded['keypoints']['has_pose']}")

    print("\n[5] Batch Encoding Test:")
    contexts = [test_context] * 5
    batch = encoder.encode_batch(contexts)
    print(f"  Batch shape: {batch.shape}")
    print(f"  All vectors identical: {np.allclose(batch[0], batch[1])}")

    print("\n" + "=" * 70)
    print("✓ FR2.3.3 Context Encoder Test Complete!")
    print("=" * 70)


if __name__ == "__main__":
    test_context_encoder()
