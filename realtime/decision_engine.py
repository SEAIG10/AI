"""
Decision Engine - Rule-based Cleaning Decision
GRU 예측 결과를 바탕으로 청소가 필요한 구역을 결정합니다.
"""

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class CleaningDecision:
    """청소 결정 결과"""
    zones_to_clean: List[str]  # 청소해야 할 구역 리스트
    priority_order: List[float]  # 각 구역의 우선순위 (오염 확률)
    estimated_time: int  # 예상 청소 시간 (분)
    path: List[str]  # 청소 경로 (순서대로)
    threshold_used: float  # 사용된 임계값

    def __str__(self):
        """사람이 읽기 쉬운 형태로 출력"""
        if not self.zones_to_clean:
            return " No cleaning needed (all zones below threshold)"

        result = f"\n{'='*60}\n"
        result += f" CLEANING REQUIRED\n"
        result += f"{'='*60}\n"
        result += f"Zones to clean: {len(self.zones_to_clean)}\n"
        result += f"Estimated time: {self.estimated_time} minutes\n"
        result += f"Threshold: {self.threshold_used}\n\n"
        result += f"Cleaning Path (priority order):\n"

        for i, (zone, priority) in enumerate(zip(self.path, self.priority_order), 1):
            result += f"  {i}. {zone:15s} (pollution: {priority:.2%})\n"

        result += f"{'='*60}\n"
        return result


class LocalDecisionEngine:
    """
    로컬 의사결정 엔진 (Rule-based)

    오프라인 작동을 보장하는 단순한 규칙 기반 청소 결정.
    복잡한 경로 계획은 나중에 추가 가능.
    """

    def __init__(self,
                 pollution_threshold: float = 0.5,
                 zone_names: List[str] = None,
                 time_per_zone: int = 10):
        """
        Args:
            pollution_threshold: 청소가 필요하다고 판단하는 오염 확률 임계값 (0~1)
            zone_names: 구역 이름 리스트
            time_per_zone: 구역당 예상 청소 시간 (분)
        """
        self.threshold = pollution_threshold
        self.zone_names = zone_names or ['balcony', 'bedroom', 'kitchen', 'living_room']
        self.time_per_zone = time_per_zone
        self.current_position = None  # TODO: iPhone ARKit에서 현재 위치 받기

        print(f"\n{'='*60}")
        print(f"Decision Engine Initialized")
        print(f"{'='*60}")
        print(f"Pollution Threshold: {self.threshold}")
        print(f"Zones: {self.zone_names}")
        print(f"Time per zone: {self.time_per_zone} min")
        print(f"{'='*60}\n")

    def decide(self, prediction: np.ndarray) -> CleaningDecision:
        """
        GRU 예측 결과를 바탕으로 청소 결정을 내립니다.

        Args:
            prediction: GRU 모델의 출력 (4,) numpy array
                        [balcony, bedroom, kitchen, living_room] 각 구역의 오염 확률

        Returns:
            CleaningDecision: 청소 결정 결과
        """
        # 1. 임계값 이상인 구역 필터링
        zones_above_threshold = [
            (zone, float(prob))
            for zone, prob in zip(self.zone_names, prediction)
            if prob > self.threshold
        ]

        # 2. 확률 높은 순으로 정렬 (우선순위)
        zones_above_threshold.sort(key=lambda x: x[1], reverse=True)

        # 3. 경로 계획 (현재는 단순 우선순위 순서, 나중에 A* 등으로 개선 가능)
        if self.current_position:
            # TODO: 실제 위치 기반 경로 계획
            path = self._plan_path_with_position(
                [z[0] for z in zones_above_threshold],
                self.current_position
            )
        else:
            # 위치 정보 없으면 단순히 우선순위 순서대로
            path = [z[0] for z in zones_above_threshold]

        # 4. 예상 청소 시간 계산
        estimated_time = len(path) * self.time_per_zone

        # 5. CleaningDecision 객체 생성
        decision = CleaningDecision(
            zones_to_clean=[z[0] for z in zones_above_threshold],
            priority_order=[z[1] for z in zones_above_threshold],
            estimated_time=estimated_time,
            path=path,
            threshold_used=self.threshold
        )

        return decision

    def _plan_path_with_position(self, zones: List[str], current_pos: dict) -> List[str]:
        """
        현재 위치를 고려한 경로 계획 (간단한 최근접 이웃 - Greedy)

        TODO: A* 알고리즘으로 개선

        Args:
            zones: 청소해야 할 구역 리스트
            current_pos: 현재 위치 {'x': float, 'y': float, 'z': float}

        Returns:
            최적화된 경로 (구역 이름 리스트)
        """
        # 현재는 단순히 입력 순서 반환 (나중에 구현)
        # 실제로는 각 구역의 중심 좌표를 알아야 하고,
        # 현재 위치에서 가장 가까운 구역부터 방문하도록 계획
        return zones

    def update_position(self, position: dict):
        """
        현재 위치 업데이트 (iPhone ARKit에서 받은 위치)

        Args:
            position: {'x': float, 'y': float, 'z': float}
        """
        self.current_position = position


# 간단한 테스트
if __name__ == "__main__":
    print("Testing Decision Engine...\n")

    # Decision Engine 생성
    engine = LocalDecisionEngine(
        pollution_threshold=0.5,
        zone_names=['balcony', 'bedroom', 'kitchen', 'living_room']
    )

    # 테스트 예측 결과 (GRU 모델 출력 시뮬레이션)
    test_predictions = [
        np.array([0.85, 0.12, 0.65, 0.23]),  # balcony, kitchen 청소 필요
        np.array([0.30, 0.45, 0.20, 0.15]),  # 모두 임계값 이하
        np.array([0.95, 0.88, 0.72, 0.91]),  # 모두 청소 필요
    ]

    for i, pred in enumerate(test_predictions, 1):
        print(f"\n{'#'*60}")
        print(f"Test Case {i}")
        print(f"{'#'*60}")
        print(f"Prediction: {pred}")

        decision = engine.decide(pred)
        print(decision)
