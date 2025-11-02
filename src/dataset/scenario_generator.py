"""
FR6.2: Scenario-based Mock Data Generation
Defines realistic household scenarios for GRU training
"""

import random
from typing import List, Dict, Any
from datetime import datetime, timedelta


class ScenarioGenerator:
    """
    FR6.2: Generate mock scenarios for training data

    Each scenario represents a 30-minute sequence of household activities
    that leads to cleaning needs (or not)
    """

    def __init__(self):
        """Initialize scenario generator"""
        self.zones = ["bathroom", "bedroom_1", "bedroom_2", "corridor",
                      "garden_balcony", "kitchen", "living_room"]

    def generate_all_scenarios(self) -> List[Dict]:
        """
        Generate all predefined scenarios

        Returns:
            List of scenario dicts
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
        시나리오 1: 저녁 식사 중 바닥에 부스러기 떨어짐

        패턴:
        - 19:00-19:15: 주방에서 요리
        - 19:15-19:30: 거실 식탁에서 식사 (바닥에 음식 부스러기 떨어짐)

        예측: living_room 바닥 청소 필요 (15분 후)
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

        # t-15분 ~ t-0분: 거실 식탁에서 식사 (오염물 제외!)
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
                    # crumbs 제거! 아직 안 보임
                ],
                "audio_events": [{"event": "Dishes", "confidence": 0.80}]
            })

        # Padding to 30 timesteps
        while len(sequence) < 30:
            sequence.append(sequence[-1])

        return {
            "name": "저녁 식사 중 바닥 부스러기",
            "sequence": sequence[:30],
            "label": [0, 0, 0, 0, 0, 0, 1]  # 15분 후 living_room 바닥 오염 예측
        }

    def scenario_tv_snack_floor_mess(self) -> Dict:
        """
        시나리오 2: TV 시청 중 과자 먹으며 바닥에 부스러기

        패턴:
        - 21:00-21:10: 거실에서 TV 시청
        - 21:10-21:30: 과자 먹기 (바닥에 부스러기 떨어짐)

        예측: living_room 바닥 청소 필요 (15분 후)
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

        # t-20분 ~ t-0분: 과자 먹기 (오염물 제외!)
        for i in range(8):
            sequence.append({
                "timestamp": (base_time + timedelta(minutes=10 + i*2.5)).timestamp(),
                "zone_id": "living_room",
                "visual_events": [
                    {"class": "person", "confidence": 0.92},
                    {"class": "couch", "confidence": 0.96},
                    {"class": "tv", "confidence": 0.91},
                    {"class": "bowl", "confidence": 0.85}  # 과자 그릇
                    # crumbs 제거! 바닥은 아직 안 보임
                ],
                "audio_events": [{"event": "Television", "confidence": 0.88}]
            })

        while len(sequence) < 30:
            sequence.append(sequence[-1])

        return {
            "name": "TV 시청 중 과자 → 바닥 부스러기",
            "sequence": sequence[:30],
            "label": [0, 0, 0, 0, 0, 0, 1]  # 15분 후 living_room 바닥 오염
        }

    def scenario_cooking_floor_spill(self) -> Dict:
        """
        시나리오 3: 요리 중 주방 바닥에 기름/재료 떨어짐

        패턴:
        - 12:00-12:30: 점심 요리 (기름 튐, 재료 떨어짐)

        예측: kitchen 바닥 청소 필요 (15분 후)
        """
        base_time = datetime(2024, 1, 16, 12, 0, 0)

        sequence = []

        # t-30분 ~ t-0분: 요리 중 (오염물 제외!)
        for i in range(12):
            sequence.append({
                "timestamp": (base_time + timedelta(minutes=i*2.5)).timestamp(),
                "zone_id": "kitchen",
                "visual_events": [
                    {"class": "person", "confidence": 0.93},
                    {"class": "oven", "confidence": 0.88},
                    {"class": "bowl", "confidence": 0.82}
                    # liquid_spill 제거! 바닥 아직 안 보임
                ],
                "audio_events": [{"event": "Cooking", "confidence": 0.90}]
            })

        while len(sequence) < 30:
            sequence.append(sequence[-1])

        return {
            "name": "요리 중 주방 바닥 기름/재료",
            "sequence": sequence[:30],
            "label": [0, 0, 0, 0, 0, 1, 0]  # 15분 후 kitchen 바닥 오염
        }

    def scenario_bedroom_sleeping_clean(self) -> Dict:
        """
        시나리오 4: 침실에서 취침 → 바닥 오염 없음 (DND)

        패턴:
        - 23:00-23:30: 침실에서 잠자기

        예측: 모든 구역 바닥 깨끗함
        """
        base_time = datetime(2024, 1, 16, 23, 0, 0)

        sequence = []

        for i in range(12):
            sequence.append({
                "timestamp": (base_time + timedelta(minutes=i*2.5)).timestamp(),
                "zone_id": "bedroom_1",
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
            "label": [0, 0, 0, 0, 0, 0, 0]  # 오염 없음 (DND)
        }

    def scenario_work_from_home_clean(self) -> Dict:
        """
        시나리오 5: 재택근무 중 → 바닥 오염 없음

        패턴:
        - 14:00-14:30: 침실에서 컴퓨터 작업

        예측: 모든 구역 바닥 깨끗함
        """
        base_time = datetime(2024, 1, 17, 14, 0, 0)

        sequence = []

        for i in range(12):
            sequence.append({
                "timestamp": (base_time + timedelta(minutes=i*2.5)).timestamp(),
                "zone_id": "bedroom_2",
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
            "label": [0, 0, 0, 0, 0, 0, 0]  # 오염 없음
        }

    def scenario_late_night_ramen_spill(self) -> Dict:
        """
        시나리오 6: 야식 라면 → 주방 바닥 국물 흘림

        패턴:
        - 01:00-01:20: 주방에서 라면 끓임
        - 01:20-01:30: 먹는 중 (국물 흘림)

        예측: kitchen 바닥 청소 필요 (15분 후)
        """
        base_time = datetime(2024, 1, 18, 1, 0, 0)

        sequence = []

        # 라면 끓임
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

        # 먹는 중 (오염물 제외!)
        for i in range(4):
            sequence.append({
                "timestamp": (base_time + timedelta(minutes=20 + i*2.5)).timestamp(),
                "zone_id": "kitchen",
                "visual_events": [
                    {"class": "person", "confidence": 0.87},
                    {"class": "bowl", "confidence": 0.88},
                    {"class": "spoon", "confidence": 0.80}
                    # liquid_spill 제거! 바닥 아직 안 보임
                ],
                "audio_events": [{"event": "Silence", "confidence": 0.92}]
            })

        while len(sequence) < 30:
            sequence.append(sequence[-1])

        return {
            "name": "야식 라면 → 주방 바닥 국물",
            "sequence": sequence[:30],
            "label": [0, 0, 0, 0, 0, 1, 0]  # 15분 후 kitchen 바닥 오염
        }

    def scenario_living_drink_spill(self) -> Dict:
        """
        시나리오 7: 거실에서 음료수 마시다 바닥에 흘림

        패턴:
        - 15:00-15:30: 거실 소파에서 음료 마심 (바닥에 액체 흘림)

        예측: living_room 바닥 청소 필요 (15분 후)
        """
        base_time = datetime(2024, 1, 18, 15, 0, 0)

        sequence = []

        # 거실에서 음료 마시는 중 (오염물 제외!)
        for i in range(12):
            sequence.append({
                "timestamp": (base_time + timedelta(minutes=i*2.5)).timestamp(),
                "zone_id": "living_room",
                "visual_events": [
                    {"class": "person", "confidence": 0.91},
                    {"class": "couch", "confidence": 0.94},
                    {"class": "cup", "confidence": 0.86}
                    # liquid_spill 제거! 바닥 아직 안 보임
                ],
                "audio_events": [{"event": "Television", "confidence": 0.82}]
            })

        while len(sequence) < 30:
            sequence.append(sequence[-1])

        return {
            "name": "거실 음료수 → 바닥 액체",
            "sequence": sequence[:30],
            "label": [0, 0, 0, 0, 0, 0, 1]  # 15분 후 living_room 바닥 오염
        }

    def add_noise(self, scenario: Dict, noise_level: float = 0.1) -> Dict:
        """
        Add random noise to scenario for data augmentation

        Args:
            scenario: Original scenario
            noise_level: Noise intensity (0.0 - 1.0)

        Returns:
            Noisy scenario
        """
        noisy_scenario = {
            "name": scenario["name"] + " (noisy)",
            "sequence": [],
            "label": scenario["label"]
        }

        for step in scenario["sequence"]:
            noisy_step = step.copy()

            # Randomly drop/add objects
            if random.random() < noise_level and noisy_step["visual_events"]:
                # Drop one object
                if len(noisy_step["visual_events"]) > 1:
                    noisy_step["visual_events"] = noisy_step["visual_events"][:-1]

            # Slightly change confidence
            for event in noisy_step["visual_events"]:
                event["confidence"] += random.uniform(-noise_level, noise_level)
                event["confidence"] = max(0.5, min(1.0, event["confidence"]))

            for event in noisy_step["audio_events"]:
                event["confidence"] += random.uniform(-noise_level, noise_level)
                event["confidence"] = max(0.5, min(1.0, event["confidence"]))

            noisy_scenario["sequence"].append(noisy_step)

        return noisy_scenario


def test_scenario_generator():
    """Test scenario generator"""
    print("=" * 70)
    print("FR6.2: Scenario Generator Test")
    print("=" * 70)

    generator = ScenarioGenerator()
    scenarios = generator.generate_all_scenarios()

    print(f"\n총 {len(scenarios)}개 시나리오 생성:\n")

    for i, scenario in enumerate(scenarios):
        print(f"{i+1}. {scenario['name']}")
        print(f"   - 시퀀스 길이: {len(scenario['sequence'])} timesteps")
        print(f"   - 오염 구역: {[generator.zones[i] for i, v in enumerate(scenario['label']) if v == 1] or ['없음']}")

        # 첫 timestep 샘플
        first_step = scenario['sequence'][0]
        print(f"   - 시작: {first_step['zone_id']}, "
              f"{len(first_step['visual_events'])} objects, "
              f"{first_step['audio_events'][0]['event']}")
        print()

    print("=" * 70)
    print("✓ FR6.2 Scenario Generator Test Complete!")
    print("=" * 70)


if __name__ == "__main__":
    test_scenario_generator()
