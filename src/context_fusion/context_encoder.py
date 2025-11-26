"""
컨텍스트 인코더
JSON 형식의 컨텍스트 벡터를 GRU 학습에 사용될 수 있는 고정된 크기의 수치 벡터로 변환합니다.
"""

import numpy as np
from datetime import datetime
from typing import Dict, List, Optional


class ContextEncoder:
    """
    JSON 형식의 컨텍스트 벡터를 고정된 335차원의 수치 벡터로 인코딩합니다.

    벡터 구성:
    - 시간 특징: 10차원
    - 공간 특징: 4차원 (구역 원-핫 인코딩)
    - 시각 특징: 14차원 (사용자 정의 YOLO 14개 클래스, 멀티-핫 인코딩)
    - 오디오 특징: 256차원 (YAMNet-256 임베딩)
    - 키포인트 특징: 51차원 (정규화된 원본 자세 정보)
    총: 335차원 (기존 338 → 4개 구역으로 변경)
    """

    def __init__(self):
        """특징 매핑과 함께 인코더를 초기화합니다."""

        # 4개의 의미론적 공간 (일관성을 위해 알파벳 순서로 정렬)
        self.zone_labels = [
            "balcony",       # 0
            "bedroom",       # 1
            "kitchen",       # 2
            "living_room"    # 3
        ]
        self.zone_to_idx = {zone: idx for idx, zone in enumerate(self.zone_labels)}

        # 사용자 정의 YOLO 14개 클래스 (HomeObjects-3K + HD10K 데이터셋 기반)
        self.visual_classes = [
            "bed",           # 0 - 가구
            "sofa",          # 1 - 가구
            "chair",         # 2 - 가구
            "table",         # 3 - 가구
            "lamp",          # 4 - 사물
            "tv",            # 5 - 전자제품
            "laptop",        # 6 - 전자제품
            "wardrobe",      # 7 - 가구
            "window",        # 8 - 구조물
            "door",          # 9 - 구조물
            "potted plant",  # 10 - 장식
            "photo frame",   # 11 - 장식
            "solid_waste",   # 12 - 오염 (부스러기, 먼지 등)
            "liquid_stain"   # 13 - 오염 (액체 얼룩 등)
        ]
        self.visual_to_idx = {cls: idx for idx, cls in enumerate(self.visual_classes)}

        # 오디오: YAMNet-256 임베딩 (256 차원)
        self.audio_embedding_dim = 256

    def encode(self, context: Dict) -> np.ndarray:
        """
        JSON 컨텍스트 벡터를 335차원의 수치 벡터로 인코딩합니다.

        Args:
            context: ContextVector.create_context()로부터 생성된 JSON 컨텍스트 딕셔너리

        Returns:
            335차원 numpy 배열 [시간(10) + 공간(4) + 시각(14) + 오디오(256) + 키포인트(51)]
        """
        # 타임스탬프 추출
        timestamp = context.get("timestamp", 0)

        # 각 구성 요소 인코딩
        time_features = self._encode_time(timestamp)        # 10차원
        spatial_features = self._encode_spatial(context)    # 4차원
        visual_features = self._encode_visual(context)      # 14차원 (사용자 정의 YOLO)
        audio_features = self._encode_audio(context)        # 256차원 (YAMNet-256 임베딩)
        keypoints_features = self._encode_keypoints(context)  # 51차원

        # 모든 특징 연결
        vector = np.concatenate([
            time_features,
            spatial_features,
            visual_features,
            audio_features,
            keypoints_features
        ])

        assert vector.shape == (335,), f"Expected shape (335,), got {vector.shape}"
        return vector.astype(np.float32)

    def _encode_time(self, timestamp: float) -> np.ndarray:
        """
        시간 특징을 10차원 벡터로 인코딩합니다.
        - 시간(hour)과 요일(day of week)을 주기적(cyclic)으로 인코딩하고,
        - 주말, 식사 시간, 근무 시간 여부를 이진(binary) 특징으로,
        - 시간, 요일, 월을 정규화(normalized)하여 구성합니다.
        """
        dt = datetime.fromtimestamp(timestamp)

        # 시간을 주기적(cyclic)으로 인코딩 (0-23)
        hour = dt.hour
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)

        # 요일을 주기적(cyclic)으로 인코딩 (0=월요일, 6=일요일)
        dow = dt.weekday()
        dow_sin = np.sin(2 * np.pi * dow / 7)
        dow_cos = np.cos(2 * np.pi * dow / 7)

        # 이진(binary) 특징
        is_weekend = 1.0 if dow >= 5 else 0.0
        is_meal_time = 1.0 if (7 <= hour <= 9) or (12 <= hour <= 14) or (18 <= hour <= 20) else 0.0
        is_work_time = 1.0 if (9 <= hour <= 18 and dow < 5) else 0.0

        # 정규화된 특징 (0-1 범위)
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
        공간 특징을 4차원 벡터로 인코딩합니다 (zone_id의 원-핫 인코딩).

        Returns:
            4차원 원-핫 벡터
        """
        zone_id = context.get("zone_id", "unknown")

        # 원-핫 인코딩
        vector = np.zeros(4, dtype=np.float32)
        if zone_id in self.zone_to_idx:
            vector[self.zone_to_idx[zone_id]] = 1.0

        return vector

    def _encode_visual(self, context: Dict) -> np.ndarray:
        """
        시각 특징을 14차원 벡터로 인코딩합니다 (사용자 정의 YOLO 클래스의 멀티-핫 인코딩).

        Returns:
            14차원 멀티-핫 벡터 (여러 개의 1을 가질 수 있음)
        """
        visual_events = context.get("visual_events", [])

        # 멀티-핫 인코딩
        vector = np.zeros(14, dtype=np.float32)

        for event in visual_events:
            class_name = event.get("class", "")
            if class_name in self.visual_to_idx:
                vector[self.visual_to_idx[class_name]] = 1.0

        return vector

    def _encode_audio(self, context: Dict) -> np.ndarray:
        """
        오디오 특징을 256차원 벡터로 인코딩합니다 (YAMNet-256 임베딩 직접 사용).

        Returns:
            256차원 오디오 임베딩 벡터 (연속적인 값)
        """
        audio_embedding = context.get("audio_embedding", None)

        # 오디오 임베딩이 없으면 0으로 채워진 벡터(무음)를 반환합니다.
        if audio_embedding is None:
            return np.zeros(self.audio_embedding_dim, dtype=np.float32)

        # 필요시 numpy 배열로 변환합니다.
        if isinstance(audio_embedding, list):
            embedding_array = np.array(audio_embedding, dtype=np.float32)
        else:
            embedding_array = audio_embedding.astype(np.float32)

        # 올바른 shape인지 확인합니다.
        if embedding_array.shape != (self.audio_embedding_dim,):
            print(f"Warning: Expected audio embedding shape ({self.audio_embedding_dim},), got {embedding_array.shape}. Using zeros.")
            return np.zeros(self.audio_embedding_dim, dtype=np.float32)

        return embedding_array

    def _encode_keypoints(self, context: Dict) -> np.ndarray:
        """
        원본 자세 키포인트를 51차원 벡터로 인코딩합니다 (정규화된 데이터 직접 사용).

        Returns:
            51차원 벡터 (17개 키포인트 × 3개 값)
        """
        keypoints_data = context.get("keypoints", None)

        # 키포인트가 없으면 0으로 채워진 벡터를 반환합니다.
        if keypoints_data is None:
            return np.zeros(51, dtype=np.float32)

        # 필요시 numpy 배열로 변환합니다.
        if isinstance(keypoints_data, list):
            keypoints_array = np.array(keypoints_data, dtype=np.float32)
        else:
            keypoints_array = keypoints_data.astype(np.float32)

        # 올바른 shape인지 확인합니다.
        if keypoints_array.shape != (51,):
            print(f"Warning: Expected keypoints shape (51,), got {keypoints_array.shape}. Using zeros.")
            return np.zeros(51, dtype=np.float32)

        return keypoints_array

    def encode_batch(self, contexts: List[Dict]) -> np.ndarray:
        """
        컨텍스트 배치를 인코딩합니다.

        Args:
            contexts: JSON 컨텍스트 딕셔너리 리스트

        Returns:
            (N, 335) 크기의 numpy 배열
        """
        return np.array([self.encode(ctx) for ctx in contexts], dtype=np.float32)

    def get_feature_names(self) -> List[str]:
        """
        디버깅을 위해 사람이 읽을 수 있는 특징 이름을 가져옵니다.

        Returns:
            335개 특징 이름의 리스트
        """
        names = []

        # 시간 특징 (10)
        names.extend([
            "hour_sin", "hour_cos",
            "dow_sin", "dow_cos",
            "is_weekend", "is_meal_time", "is_work_time",
            "hour_norm", "dow_norm", "month_norm"
        ])

        # 공간 특징 (4)
        names.extend([f"zone_{zone}" for zone in self.zone_labels])

        # 시각 특징 (14) - 사용자 정의 YOLO
        names.extend([f"obj_{cls}" for cls in self.visual_classes])

        # 오디오 특징 (256) - YAMNet-256 임베딩
        names.extend([f"audio_emb_{i}" for i in range(256)])

        # 키포인트 특징 (51)
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
        수치 벡터를 디버깅을 위해 사람이 읽을 수 있는 형식으로 다시 변환합니다.

        Args:
            vector: 335차원 numpy 배열

        Returns:
            디코딩된 특징을 담은 딕셔너리
        """
        feature_names = self.get_feature_names()

        # 시간 특징 (0-9)
        time_features = {
            "hour_sin": vector[0],
            "hour_cos": vector[1],
            "estimated_hour": int((np.arctan2(vector[0], vector[1]) / (2 * np.pi) * 24) % 24),
            "is_weekend": bool(vector[4] > 0.5),
            "is_meal_time": bool(vector[5] > 0.5),
            "is_work_time": bool(vector[6] > 0.5)
        }

        # 공간 특징 (10-13)
        zone_vector = vector[10:14]
        zone_idx = np.argmax(zone_vector) if np.max(zone_vector) > 0 else None
        spatial_features = {
            "zone": self.zone_labels[zone_idx] if zone_idx is not None else "unknown"
        }

        # 시각 특징 (14-27)
        visual_vector = vector[14:28]
        detected_objects = [self.visual_classes[i] for i, v in enumerate(visual_vector) if v > 0.5]
        visual_features = {
            "objects": detected_objects,
            "count": len(detected_objects)
        }

        # 오디오 특징 (28-283)
        audio_embedding = vector[28:284]
        has_audio = np.any(audio_embedding != 0)
        audio_features = {
            "has_audio": has_audio,
            "embedding_norm": float(np.linalg.norm(audio_embedding)),
            "embedding_mean": float(np.mean(audio_embedding)),
            "embedding_shape": "(256,)"
        }

        # 키포인트 특징 (284-334)
        keypoints_vector = vector[284:335]
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
    """컨텍스트 인코더 테스트"""
    print("=" * 70)
    print("컨텍스트 인코더 테스트")
    print("=" * 70)

    encoder = ContextEncoder()

    # 이전 예제의 테스트 컨텍스트
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

    print("\n[1] 원본 JSON 컨텍스트:")
    print(f"  구역: {test_context['zone_id']}")
    print(f"  시각: {[e['class'] for e in test_context['visual_events']]}")
    print(f"  오디오: {test_context['audio_events'][0]['event']}")

    print("\n[2] 수치 벡터로 인코딩 중...")
    vector = encoder.encode(test_context)
    print(f"  Shape: {vector.shape}")
    print(f"  Dtype: {vector.dtype}")
    print(f"  처음 20개 값: {vector[:20]}")

    print("\n[3] 특징 분해:")
    print(f"  시간 특징 (0-9): {vector[0:10]}")
    print(f"  공간 특징 (10-16): {vector[10:17]}")
    print(f"  시각 특징 (17-30): {vector[17:31].sum()} 객체 감지됨")
    print(f"  오디오 특징 (31-286): {vector[31:287].sum():.2f} 임베딩 norm")
    print(f"  키포인트 특징 (287-337): {vector[287:338].sum():.2f} 자세 합계")

    print("\n[4] 디버그용 디코딩:")
    decoded = encoder.decode_debug(vector)
    print(f"  추정 시간: {decoded['time']['estimated_hour']}")
    print(f"  구역: {decoded['spatial']['zone']}")
    print(f"  객체: {decoded['visual']['objects']}")
    print(f"  오디오 유무: {decoded['audio']['has_audio']}")
    print(f"  자세 유무: {decoded['keypoints']['has_pose']}")

    print("\n[5] 배치 인코딩 테스트:")
    contexts = [test_context] * 5
    batch = encoder.encode_batch(contexts)
    print(f"  배치 Shape: {batch.shape}")
    print(f"  모든 벡터가 동일한가: {np.allclose(batch[0], batch[1])}")

    print("\n" + "=" * 70)
    print("컨텍스트 인코더 테스트 완료!")
    print("=" * 70)


if __name__ == "__main__":
    test_context_encoder()
