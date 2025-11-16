"""
컨텍스트 벡터 생성
다중 모드 센서 데이터(GPS, YOLO, Audio)를 구조화된 컨텍스트 벡터로 융합합니다.
"""

import time
import json
import numpy as np
from typing import Dict, List, Optional, Tuple


class ContextVector:
    """
    다중 모드 컨텍스트 융합
    공간(GPS), 시각(YOLO), 오디오(Yamnet) 데이터를 결합합니다.
    """

    def __init__(self):
        """ContextVector 프로세서를 초기화합니다."""
        self.last_context = None

    def create_context(
        self,
        timestamp: Optional[float] = None,
        position: Optional[Tuple[float, float]] = None,
        zone_info: Optional[Dict] = None,
        visual_detections: Optional[List[Dict]] = None,
        audio_embedding: Optional[np.ndarray] = None,
        keypoints: Optional[np.ndarray] = None
    ) -> Dict:
        """
        다중 모드 입력으로부터 통합된 컨텍스트 벡터를 생성합니다.

        Args:
            timestamp: Unix 타임스탬프 (기본값: 현재 시간)
            position: (x, y) GPS 좌표
            zone_info: FloorPlanManager로부터 받은 의미론적 공간 정보
            visual_detections: YOLO 객체 탐지 결과
            audio_embedding: YAMNet-256 오디오 임베딩 (256차원 numpy 배열)
            keypoints: 정규화된 자세 키포인트 (51차원 numpy 배열)

        Returns:
            JSON으로 직렬화 가능한 딕셔너리 형태의 컨텍스트 벡터
        """
        if timestamp is None:
            timestamp = time.time()

        # 위치 데이터 생성
        position_data = None
        if position:
            position_data = {
                "x": round(position[0], 2),
                "y": round(position[1], 2)
            }

        # Zone 데이터 생성
        zone_label = "Unknown"
        zone_id = "unknown"
        if zone_info:
            zone_label = zone_info.get("label", "Unknown")
            zone_id = zone_info.get("id", "unknown")

        # 시각 이벤트 생성 (YOLO 감지 결과)
        visual_events = []
        if visual_detections:
            for det in visual_detections:
                visual_events.append({
                    "class": det["class"],
                    "confidence": round(det["confidence"], 2),
                    "bbox": det["bbox"]  # (x1, y1, x2, y2)
                })

        # 오디오 임베딩 처리
        audio_embedding_data = None
        has_audio = False
        if audio_embedding is not None:
            # JSON 직렬화를 위해 numpy 배열을 리스트로 변환
            audio_embedding_data = audio_embedding.tolist() if isinstance(audio_embedding, np.ndarray) else list(audio_embedding)
            has_audio = True

        # 키포인트 처리
        keypoints_data = None
        has_pose = False
        if keypoints is not None:
            # JSON 직렬화를 위해 numpy 배열을 리스트로 변환
            keypoints_data = keypoints.tolist() if isinstance(keypoints, np.ndarray) else list(keypoints)
            has_pose = True

        # 사람이 읽을 수 있는 컨텍스트 요약 생성
        visual_count = len(visual_events)
        audio_label = "audio:yes" if has_audio else "audio:no"
        pose_label = "pose:yes" if has_pose else "pose:no"
        context_summary = f"{zone_id} | {visual_count} objects | {audio_label} | {pose_label}"

        # 전체 컨텍스트 벡터 조립
        context_vector = {
            "timestamp": round(timestamp, 3),
            "position": position_data,
            "zone": zone_label,
            "zone_id": zone_id,
            "visual_events": visual_events,
            "audio_embedding": audio_embedding_data,  # 256차원 YAMNet 임베딩 (또는 None)
            "keypoints": keypoints_data,  # 51차원 정규화된 자세 (또는 None)
            "context_summary": context_summary
        }

        self.last_context = context_vector
        return context_vector

    def context_to_json(self, context: Dict) -> str:
        """
        컨텍스트 벡터를 JSON 문자열로 직렬화합니다.

        Args:
            context: 컨텍스트 벡터 딕셔너리

        Returns:
            JSON 문자열
        """
        return json.dumps(context, ensure_ascii=False, indent=2)

    def json_to_context(self, json_str: str) -> Dict:
        """
        JSON 문자열로부터 컨텍스트 벡터를 역직렬화합니다.

        Args:
            json_str: JSON 문자열

        Returns:
            컨텍스트 벡터 딕셔너리
        """
        return json.loads(json_str)

    def get_visual_class_counts(self, context: Dict) -> Dict[str, int]:
        """
        컨텍스트 벡터에서 객체 클래스별 개수를 추출합니다.

        Args:
            context: 컨텍스트 벡터

        Returns:
            클래스 이름을 키로, 개수를 값으로 하는 딕셔너리
            (예: {"person": 2, "couch": 1, "cup": 3})
        """
        counts = {}
        for event in context.get("visual_events", []):
            class_name = event["class"]
            counts[class_name] = counts.get(class_name, 0) + 1
        return counts

    def has_audio_embedding(self, context: Dict) -> bool:
        """
        컨텍스트에 오디오 임베딩이 있는지 확인합니다.

        Args:
            context: 컨텍스트 벡터

        Returns:
            오디오 임베딩이 존재하면 True, 그렇지 않으면 False
        """
        audio_embedding = context.get("audio_embedding", None)
        return audio_embedding is not None and len(audio_embedding) > 0

    def compare_contexts(self, context1: Dict, context2: Dict) -> Dict:
        """
        두 컨텍스트 벡터를 비교하여 변경 사항을 감지합니다.

        Args:
            context1: 이전 컨텍스트 벡터
            context2: 이후 컨텍스트 벡터

        Returns:
            차이점을 설명하는 딕셔너리
        """
        # Zone 변경
        zone_changed = (
            context1.get("zone_id") != context2.get("zone_id")
        )

        # 시각적 변경
        classes1 = set(e["class"] for e in context1.get("visual_events", []))
        classes2 = set(e["class"] for e in context2.get("visual_events", []))
        new_objects = list(classes2 - classes1)
        removed_objects = list(classes1 - classes2)

        # 오디오 변경 (임베딩 비교)
        audio1 = context1.get("audio_embedding")
        audio2 = context2.get("audio_embedding")

        audio_changed = False
        audio_distance = 0.0

        if audio1 is not None and audio2 is not None:
            # 임베딩 간의 L2 거리 계산
            emb1 = np.array(audio1) if isinstance(audio1, list) else audio1
            emb2 = np.array(audio2) if isinstance(audio2, list) else audio2
            audio_distance = float(np.linalg.norm(emb1 - emb2))
            # 거리가 임계값보다 크면 변경된 것으로 간주
            audio_changed = audio_distance > 0.5
        elif audio1 != audio2:  # 하나는 None이고 다른 하나는 아닌 경우
            audio_changed = True

        return {
            "zone_changed": zone_changed,
            "new_objects": new_objects,
            "removed_objects": removed_objects,
            "audio_changed": audio_changed,
            "audio_distance": audio_distance,
            "time_delta": context2["timestamp"] - context1["timestamp"]
        }


def test_context_vector():
    """컨텍스트 벡터 생성 테스트"""
    print("=" * 60)
    print("컨텍스트 벡터 생성 테스트")
    print("=" * 60)

    cv = ContextVector()

    # 센서 데이터 시뮬레이션
    position = (-5.0, -3.0)
    zone_info = {"id": "living_room", "label": "거실 (Living Room)"}
    visual_detections = [
        {"class": "person", "confidence": 0.92, "bbox": (100, 150, 200, 400)},
        {"class": "couch", "confidence": 0.85, "bbox": (50, 200, 300, 450)},
    ]
    # 256차원 오디오 임베딩 시뮬레이션
    audio_embedding = np.random.randn(256).astype(np.float32)

    # 컨텍스트 벡터 생성
    context = cv.create_context(
        position=position,
        zone_info=zone_info,
        visual_detections=visual_detections,
        audio_embedding=audio_embedding
    )

    print("\n생성된 컨텍스트 벡터 (가독성을 위해 일부 생략):")
    context_display = context.copy()
    if context_display.get("audio_embedding"):
        context_display["audio_embedding"] = f"[256차원 배열, 처음 5개: {context_display['audio_embedding'][:5]}...]"
    print(json.dumps(context_display, ensure_ascii=False, indent=2))

    print("\n시각적 클래스 개수:")
    print(cv.get_visual_class_counts(context))

    print("\n오디오 임베딩 존재 여부:")
    print(cv.has_audio_embedding(context))

    print("\n" + "=" * 60)
    print("컨텍스트 벡터 테스트 완료!")
    print("=" * 60)


if __name__ == "__main__":
    test_context_vector()
