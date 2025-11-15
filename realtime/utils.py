"""
Realtime Demo - Common Utilities
공통 함수 및 상수 정의
"""

import numpy as np
from datetime import datetime

# Zone 정의
ZONES = [
    "bathroom",
    "bedroom_1",
    "bedroom_2",
    "corridor",
    "garden_balcony",
    "kitchen",
    "living_room"
]

# YOLO class names (14 classes)
YOLO_CLASSES = [
    "bed",           # 0
    "sofa",          # 1
    "chair",         # 2
    "table",         # 3
    "lamp",          # 4
    "tv",            # 5
    "laptop",        # 6
    "wardrobe",      # 7
    "window",        # 8
    "door",          # 9
    "potted plant",  # 10
    "photo frame",   # 11
    "solid_waste",   # 12
    "liquid_stain"   # 13
]


def zone_to_onehot(zone_name: str) -> np.ndarray:
    """
    Zone name을 one-hot vector로 변환

    Args:
        zone_name: Zone 이름 (예: "kitchen")

    Returns:
        (7,) one-hot vector
    """
    vector = np.zeros(7, dtype=np.float32)
    if zone_name in ZONES:
        idx = ZONES.index(zone_name)
        vector[idx] = 1.0
    return vector


def get_time_features(dt: datetime = None) -> np.ndarray:
    """
    시간 정보를 10-dim feature vector로 변환

    Args:
        dt: datetime object (None이면 현재 시간)

    Returns:
        (10,) time feature vector
    """
    if dt is None:
        dt = datetime.now()

    # Cyclic encoding
    hour = dt.hour
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    dow = dt.weekday()  # 0=Monday, 6=Sunday
    dow_sin = np.sin(2 * np.pi * dow / 7)
    dow_cos = np.cos(2 * np.pi * dow / 7)

    # Binary features
    is_weekend = 1.0 if dow >= 5 else 0.0
    is_meal_time = 1.0 if (7 <= hour <= 9) or (12 <= hour <= 14) or (18 <= hour <= 20) else 0.0
    is_work_time = 1.0 if (9 <= hour <= 18 and dow < 5) else 0.0

    # Normalized features
    hour_norm = hour / 24.0
    dow_norm = dow / 7.0
    month_norm = dt.month / 12.0

    return np.array([
        hour_sin, hour_cos,
        dow_sin, dow_cos,
        is_weekend,
        is_meal_time,
        is_work_time,
        hour_norm,
        dow_norm,
        month_norm
    ], dtype=np.float32)


def yolo_results_to_14dim(results) -> np.ndarray:
    """
    YOLO detection results를 14-dim multi-hot vector로 변환

    Args:
        results: YOLO results object

    Returns:
        (14,) multi-hot vector
    """
    vector = np.zeros(14, dtype=np.float32)

    if len(results) > 0 and hasattr(results[0], 'boxes'):
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            if 0 <= cls_id < 14:
                vector[cls_id] = 1.0

    return vector


def extract_pose_keypoints(results) -> np.ndarray:
    """
    YOLO-Pose results에서 keypoints 추출

    Args:
        results: YOLO results object

    Returns:
        (51,) keypoints vector (17 joints × 3 values)
    """
    pose_vec = np.zeros(51, dtype=np.float32)

    # YOLO-Pose가 활성화되어 있고 사람이 감지된 경우
    if len(results) > 0 and hasattr(results[0], 'keypoints'):
        keypoints_data = results[0].keypoints
        if keypoints_data is not None and len(keypoints_data) > 0:
            # 첫 번째 사람의 keypoints 사용
            kpts = keypoints_data[0].data.cpu().numpy().flatten()

            # 51-dim으로 맞추기 (17 joints × 3 = 51)
            if len(kpts) >= 51:
                pose_vec = kpts[:51].astype(np.float32)
            else:
                pose_vec[:len(kpts)] = kpts.astype(np.float32)

    return pose_vec


def print_prediction_result(prediction: np.ndarray, zones: list = None):
    """
    GRU 예측 결과를 예쁘게 출력

    Args:
        prediction: (7,) prediction array
        zones: Zone 이름 리스트
    """
    if zones is None:
        zones = ZONES

    print("\n" + "="*60)
    print("🎯 Pollution Prediction (15 minutes later)")
    print("="*60 + "\n")

    for zone, prob in zip(zones, prediction):
        # Progress bar
        bar_length = int(prob * 20)
        bar = "█" * bar_length + "░" * (20 - bar_length)

        # Emoji
        emoji = "🔴" if prob > 0.5 else "✅"

        # Print
        print(f"  {emoji} {zone:15s} [{bar}] {prob*100:5.1f}%")

    print("\n" + "="*60)


if __name__ == "__main__":
    # Test utilities
    print("Testing utils...")

    # Test zone_to_onehot
    zone_vec = zone_to_onehot("kitchen")
    print(f"Zone vector: {zone_vec}")

    # Test get_time_features
    time_vec = get_time_features()
    print(f"Time vector shape: {time_vec.shape}")
    print(f"Time vector: {time_vec}")

    # Test print_prediction_result
    mock_prediction = np.array([0.1, 0.05, 0.03, 0.02, 0.01, 0.85, 0.12])
    print_prediction_result(mock_prediction)

    print("\n✓ All utils tested successfully!")
