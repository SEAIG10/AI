"""
실시간 데모 - 공용 유틸리티
실시간 데모에서 공통으로 사용되는 함수 및 상수를 정의합니다.
"""

import numpy as np
from datetime import datetime

# Zone 정의 (4개 구역)
ZONES = [
    "balcony",
    "bedroom",
    "kitchen",
    "living_room"
]

# YOLO 클래스 이름 (14개)
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
    Zone 이름을 원-핫 벡터로 변환합니다.

    Args:
        zone_name: Zone 이름 (예: "kitchen")

    Returns:
        (4,) 크기의 원-핫 벡터
    """
    vector = np.zeros(4, dtype=np.float32)
    if zone_name in ZONES:
        idx = ZONES.index(zone_name)
        vector[idx] = 1.0
    return vector


def get_time_features(dt: datetime = None) -> np.ndarray:
    """
    시간 정보를 10차원 특징 벡터로 변환합니다.

    Args:
        dt: datetime 객체 (None일 경우 현재 시간 사용)

    Returns:
        (10,) 크기의 시간 특징 벡터
    """
    if dt is None:
        dt = datetime.now()

    # 주기적 특징 인코딩
    hour = dt.hour
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    dow = dt.weekday()  # 0=월요일, 6=일요일
    dow_sin = np.sin(2 * np.pi * dow / 7)
    dow_cos = np.cos(2 * np.pi * dow / 7)

    # 이진 특징
    is_weekend = 1.0 if dow >= 5 else 0.0
    is_meal_time = 1.0 if (7 <= hour <= 9) or (12 <= hour <= 14) or (18 <= hour <= 20) else 0.0
    is_work_time = 1.0 if (9 <= hour <= 18 and dow < 5) else 0.0

    # 정규화된 특징
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
    YOLO 감지 결과를 14차원 멀티-핫 벡터로 변환합니다.

    Args:
        results: YOLO results 객체

    Returns:
        (14,) 크기의 멀티-핫 벡터
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
    YOLO-Pose 결과에서 키포인트를 추출합니다.

    Args:
        results: YOLO results 객체

    Returns:
        (51,) 크기의 키포인트 벡터 (17개 관절 × 3개 값)
    """
    pose_vec = np.zeros(51, dtype=np.float32)

    # YOLO-Pose가 활성화되어 있고 사람이 감지된 경우
    if len(results) > 0 and hasattr(results[0], 'keypoints'):
        keypoints_data = results[0].keypoints
        if keypoints_data is not None and len(keypoints_data) > 0:
            # 첫 번째로 감지된 사람의 키포인트를 사용
            kpts = keypoints_data[0].data.cpu().numpy().flatten()

            # 51차원으로 크기 맞춤 (17 관절 × 3 = 51)
            if len(kpts) >= 51:
                pose_vec = kpts[:51].astype(np.float32)
            else:
                pose_vec[:len(kpts)] = kpts.astype(np.float32)

    return pose_vec


def print_prediction_result(prediction: np.ndarray, zones: list = None):
    """
    GRU 예측 결과를 형식에 맞게 출력합니다.

    Args:
        prediction: (4,) 크기의 예측 확률 배열
        zones: Zone 이름 리스트
    """
    if zones is None:
        zones = ZONES

    print("\n" + "="*60)
    print("Pollution Prediction (15 minutes later)")
    print("="*60 + "\n")

    for zone, prob in zip(zones, prediction):
        # 진행률 표시줄
        bar_length = int(prob * 20)
        bar = "█" * bar_length + "░" * (20 - bar_length)

        # 상태 표시
        status = "High" if prob > 0.5 else "Low"

        # 출력
        print(f"  - {zone:15s} [{bar}] {prob*100:5.1f}% ({status})")

    print("\n" + "="*60)


if __name__ == "__main__":
    # 유틸리티 함수 테스트
    print("Testing utils...")

    # zone_to_onehot 테스트
    zone_vec = zone_to_onehot("kitchen")
    print(f"Zone vector: {zone_vec}")

    # get_time_features 테스트
    time_vec = get_time_features()
    print(f"Time vector shape: {time_vec.shape}")
    print(f"Time vector: {time_vec}")

    # print_prediction_result 테스트
    mock_prediction = np.array([0.1, 0.05, 0.85, 0.12])  # 4 zones
    print_prediction_result(mock_prediction)

    print("\nAll utils tested successfully!")
