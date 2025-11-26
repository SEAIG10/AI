"""
시나리오 기반 모의 데이터 생성
GRU 학습을 위해 현실적인 가정 내 시나리오를 정의합니다.
"""

import random
from typing import List, Dict, Any
from datetime import datetime, timedelta


class ScenarioGenerator:
    """
    학습 데이터를 위한 모의 시나리오를 생성합니다.

    각 시나리오는 30분 길이의 가정 내 활동 시퀀스를 나타내며,
    이 활동이 청소 필요 여부로 이어지는지를 모델링합니다.
    """

    def __init__(self):
        """시나리오 생성기를 초기화합니다."""
        self.zones = ["balcony", "bedroom", "kitchen", "living_room"]

    def generate_all_scenarios(self) -> List[Dict]:
        """
        미리 정의된 모든 시나리오를 생성합니다.

        Returns:
            시나리오 딕셔너리의 리스트
        """
        return [
            self.scenario_dinner_floor_mess(),
            self.scenario_tv_snack_floor_mess(),
            self.scenario_cooking_floor_spill(),
            self.scenario_bedroom_sleeping_clean(),
            self.scenario_work_from_home_clean(),
            self.scenario_late_night_ramen_spill(),
            self.scenario_living_drink_spill(),
        ]

    def scenario_dinner_floor_mess(self) -> Dict:
        """
        시나리오 1: 저녁 식사 중 바닥에 부스러기 발생

        패턴:
        - 19:00-19:15: 주방에서 요리
        - 19:15-19:30: 거실 식탁에서 식사 (이때 바닥에 음식 부스러기 발생)

        예측: 15분 후 living_room 바닥 청소 필요
        """
        base_time = datetime(2024, 1, 15, 19, 0, 0)  # 19:00

        sequence = []

        # t-30분 ~ t-15분: 주방에서 요리
        for i in range(6):
            sequence.append({
                "timestamp": (base_time + timedelta(minutes=i*2.5)).timestamp(),
                "zone_id": "kitchen",
                "visual_events": [
                    {"class": "person", "confidence": 0.92},
                    {"class": "bowl", "confidence": 0.85},
                    {"class": "spoon", "confidence": 0.78}
                ],
                "audio_events": [{"event": "Cooking", "confidence": 0.88}]
            })

        # t-15분 ~ t-0분: 거실 식탁에서 식사
        for i in range(6):
            sequence.append({
                "timestamp": (base_time + timedelta(minutes=15 + i*2.5)).timestamp(),
                "zone_id": "living_room",
                "visual_events": [
                    {"class": "person", "confidence": 0.90},
                    {"class": "chair", "confidence": 0.95},
                    {"class": "dining table", "confidence": 0.98},
                    {"class": "plate", "confidence": 0.87},
                    {"class": "fork", "confidence": 0.82},
                    {"class": "cup", "confidence": 0.85}
                    # 부스러기(crumbs)는 예측 시점에서는 보이지 않으므로 제외
                ],
                "audio_events": [{"event": "Dishes", "confidence": 0.80}]
            })

        # 30 타임스텝으로 패딩
        while len(sequence) < 30:
            sequence.append(sequence[-1])

        return {
            "name": "저녁 식사 중 바닥 부스러기",
            "sequence": sequence[:30],
            "label": [0, 0, 0, 1]  # 15분 후 living_room 바닥 오염 예측
        }

    def scenario_tv_snack_floor_mess(self) -> Dict:
        """
        시나리오 2: TV 시청 중 과자를 먹으며 바닥에 부스러기 발생

        패턴:
        - 21:00-21:10: 거실에서 TV 시청
        - 21:10-21:30: 과자 섭취 (이때 바닥에 부스러기 발생)

        예측: 15분 후 living_room 바닥 청소 필요
        """
        base_time = datetime(2024, 1, 15, 21, 0, 0)

        sequence = []

        # t-30분 ~ t-20분: TV 시청 시작
        for i in range(4):
            sequence.append({
                "timestamp": (base_time + timedelta(minutes=i*2.5)).timestamp(),
                "zone_id": "living_room",
                "visual_events": [
                    {"class": "person", "confidence": 0.93},
                    {"class": "couch", "confidence": 0.96},
                    {"class": "tv", "confidence": 0.92}
                ],
                "audio_events": [{"event": "Television", "confidence": 0.90}]
            })

        # t-20분 ~ t-0분: 과자 섭취
        for i in range(8):
            sequence.append({
                "timestamp": (base_time + timedelta(minutes=10 + i*2.5)).timestamp(),
                "zone_id": "living_room",
                "visual_events": [
                    {"class": "person", "confidence": 0.92},
                    {"class": "couch", "confidence": 0.96},
                    {"class": "tv", "confidence": 0.91},
                    {"class": "bowl", "confidence": 0.85}  # 과자 그릇
                    # 부스러기(crumbs)는 예측 시점에서는 보이지 않으므로 제외
                ],
                "audio_events": [{"event": "Television", "confidence": 0.88}]
            })

        while len(sequence) < 30:
            sequence.append(sequence[-1])

        return {
            "name": "TV 시청 중 과자 섭취",
            "sequence": sequence[:30],
            "label": [0, 0, 0, 1]  # 15분 후 living_room 바닥 오염
        }

    def scenario_cooking_floor_spill(self) -> Dict:
        """
        시나리오 3: 요리 중 주방 바닥에 기름이나 재료가 떨어짐

        패턴:
        - 12:00-12:30: 점심 요리 (기름 튐, 재료 떨어짐)

        예측: 15분 후 kitchen 바닥 청소 필요
        """
        base_time = datetime(2024, 1, 16, 12, 0, 0)

        sequence = []

        # t-30분 ~ t-0분: 요리 중
        for i in range(12):
            sequence.append({
                "timestamp": (base_time + timedelta(minutes=i*2.5)).timestamp(),
                "zone_id": "kitchen",
                "visual_events": [
                    {"class": "person", "confidence": 0.93},
                    {"class": "oven", "confidence": 0.88},
                    {"class": "bowl", "confidence": 0.82}
                    # 액체 얼룩(liquid_spill)은 예측 시점에서는 보이지 않으므로 제외
                ],
                "audio_events": [{"event": "Cooking", "confidence": 0.90}]
            })

        while len(sequence) < 30:
            sequence.append(sequence[-1])

        return {
            "name": "요리 중 주방 바닥 오염",
            "sequence": sequence[:30],
            "label": [0, 0, 1, 0]  # 15분 후 kitchen 바닥 오염
        }

    def scenario_bedroom_sleeping_clean(self) -> Dict:
        """
        시나리오 4: 침실에서 취침 (오염 없음, 방해 금지)

        패턴:
        - 23:00-23:30: 침실에서 수면

        예측: 모든 구역 바닥 깨끗함
        """
        base_time = datetime(2024, 1, 16, 23, 0, 0)

        sequence = []

        for i in range(12):
            sequence.append({
                "timestamp": (base_time + timedelta(minutes=i*2.5)).timestamp(),
                "zone_id": "bedroom",
                "visual_events": [
                    {"class": "person", "confidence": 0.88},
                    {"class": "bed", "confidence": 0.95}
                ],
                "audio_events": [{"event": "Silence", "confidence": 0.98}]
            })

        while len(sequence) < 30:
            sequence.append(sequence[-1])

        return {
            "name": "침실에서 취침",
            "sequence": sequence[:30],
            "label": [0, 0, 0, 0]  # 오염 없음
        }

    def scenario_work_from_home_clean(self) -> Dict:
        """
        시나리오 5: 재택근무 중 (오염 없음)

        패턴:
        - 14:00-14:30: 침실에서 컴퓨터 작업

        예측: 모든 구역 바닥 깨끗함
        """
        base_time = datetime(2024, 1, 17, 14, 0, 0)

        sequence = []

        for i in range(12):
            sequence.append({
                "timestamp": (base_time + timedelta(minutes=i*2.5)).timestamp(),
                "zone_id": "bedroom",
                "visual_events": [
                    {"class": "person", "confidence": 0.91},
                    {"class": "chair", "confidence": 0.90},
                    {"class": "laptop", "confidence": 0.88},
                    {"class": "keyboard", "confidence": 0.85}
                ],
                "audio_events": [{"event": "Silence", "confidence": 0.80}]
            })

        while len(sequence) < 30:
            sequence.append(sequence[-1])

        return {
            "name": "재택근무 중",
            "sequence": sequence[:30],
            "label": [0, 0, 0, 0]  # 오염 없음
        }

    def scenario_late_night_ramen_spill(self) -> Dict:
        """
        시나리오 6: 야식으로 라면 섭취 중 주방 바닥에 국물 흘림

        패턴:
        - 01:00-01:20: 주방에서 라면 조리
        - 01:20-01:30: 섭취 중 (국물 흘림)

        예측: 15분 후 kitchen 바닥 청소 필요
        """
        base_time = datetime(2024, 1, 18, 1, 0, 0)

        sequence = []

        # 라면 조리
        for i in range(8):
            sequence.append({
                "timestamp": (base_time + timedelta(minutes=i*2.5)).timestamp(),
                "zone_id": "kitchen",
                "visual_events": [
                    {"class": "person", "confidence": 0.89},
                    {"class": "oven", "confidence": 0.85}
                ],
                "audio_events": [{"event": "Cooking", "confidence": 0.82}]
            })

        # 섭취 중
        for i in range(4):
            sequence.append({
                "timestamp": (base_time + timedelta(minutes=20 + i*2.5)).timestamp(),
                "zone_id": "kitchen",
                "visual_events": [
                    {"class": "person", "confidence": 0.87},
                    {"class": "bowl", "confidence": 0.88},
                    {"class": "spoon", "confidence": 0.80}
                    # 액체 얼룩(liquid_spill)은 예측 시점에서는 보이지 않으므로 제외
                ],
                "audio_events": [{"event": "Silence", "confidence": 0.92}]
            })

        while len(sequence) < 30:
            sequence.append(sequence[-1])

        return {
            "name": "야식 라면 섭취",
            "sequence": sequence[:30],
            "label": [0, 0, 1, 0]  # 15분 후 kitchen 바닥 오염
        }

    def scenario_living_drink_spill(self) -> Dict:
        """
        시나리오 7: 거실에서 음료수를 마시다 바닥에 흘림

        패턴:
        - 15:00-15:30: 거실 소파에서 음료 섭취 (바닥에 액체 흘림)

        예측: 15분 후 living_room 바닥 청소 필요
        """
        base_time = datetime(2024, 1, 18, 15, 0, 0)

        sequence = []

        # 거실에서 음료 섭취 중
        for i in range(12):
            sequence.append({
                "timestamp": (base_time + timedelta(minutes=i*2.5)).timestamp(),
                "zone_id": "living_room",
                "visual_events": [
                    {"class": "person", "confidence": 0.91},
                    {"class": "couch", "confidence": 0.94},
                    {"class": "cup", "confidence": 0.86}
                    # 액체 얼룩(liquid_spill)은 예측 시점에서는 보이지 않으므로 제외
                ],
                "audio_events": [{"event": "Television", "confidence": 0.82}]
            })

        while len(sequence) < 30:
            sequence.append(sequence[-1])

        return {
            "name": "거실 음료수 섭취",
            "sequence": sequence[:30],
            "label": [0, 0, 0, 1]  # 15분 후 living_room 바닥 오염
        }

    def add_noise(self, scenario: Dict, noise_level: float = 0.1) -> Dict:
        """
        데이터 증강을 위해 시나리오에 무작위 노이즈를 추가합니다.

        Args:
            scenario: 원본 시나리오
            noise_level: 노이즈 강도 (0.0 - 1.0)

        Returns:
            노이즈가 추가된 시나리오
        """
        noisy_scenario = {
            "name": scenario["name"] + " (noisy)",
            "sequence": [],
            "label": scenario["label"]
        }

        for step in scenario["sequence"]:
            noisy_step = step.copy()

            # 객체를 무작위로 제거
            if random.random() < noise_level and noisy_step["visual_events"]:
                if len(noisy_step["visual_events"]) > 1:
                    noisy_step["visual_events"] = noisy_step["visual_events"][:-1]

            # 신뢰도를 약간 변경
            for event in noisy_step["visual_events"]:
                event["confidence"] += random.uniform(-noise_level, noise_level)
                event["confidence"] = max(0.5, min(1.0, event["confidence"]))

            for event in noisy_step["audio_events"]:
                event["confidence"] += random.uniform(-noise_level, noise_level)
                event["confidence"] = max(0.5, min(1.0, event["confidence"]))

            noisy_scenario["sequence"].append(noisy_step)

        return noisy_scenario


def test_scenario_generator():
    """시나리오 생성기 테스트"""
    print("=" * 70)
    print("시나리오 생성기 테스트")
    print("=" * 70)

    generator = ScenarioGenerator()
    scenarios = generator.generate_all_scenarios()

    print(f"\n총 {len(scenarios)}개 시나리오 생성:\n")

    for i, scenario in enumerate(scenarios):
        print(f"{i+1}. {scenario['name']}")
        print(f"   - 시퀀스 길이: {len(scenario['sequence'])} 타임스텝")
        print(f"   - 오염 구역: {[generator.zones[i] for i, v in enumerate(scenario['label']) if v == 1] or ['없음']}")

        # 첫 타임스텝 샘플
        first_step = scenario['sequence'][0]
        print(f"   - 시작: {first_step['zone_id']}, "
              f"{len(first_step['visual_events'])} 객체, "
              f"{first_step['audio_events'][0]['event']}")
        print()

    print("=" * 70)
    print("시나리오 생성기 테스트 완료!")
    print("=" * 70)


if __name__ == "__main__":
    test_scenario_generator()
