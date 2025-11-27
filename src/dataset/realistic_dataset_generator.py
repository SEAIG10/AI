"""
대규모 현실적 데이터셋 생성기
realistic_action_patterns.py의 패턴들을 사용하여 일상 루틴을 시뮬레이션합니다.

목표:
- 50,000개 이상의 타임스텝 생성
- 실제 학습된 YOLO(14) + YAMNet(17) 클래스만 사용
- 인간의 행동 루틴 → 오염 발생의 인과관계 반영
- 시간대별 현실적인 행동 패턴 (아침, 점심, 저녁, 밤)
"""

import numpy as np
import os
from typing import Dict, List, Tuple
from collections import defaultdict
from .realistic_action_patterns import RealisticActionPattern, YOLO_CLASSES, YAMNET_CLASSES

# 시간 인코딩 (10차원: hour_sin, hour_cos, day_sin, day_cos, is_weekday, is_morning, is_afternoon, is_evening, is_night, is_weekend)
def encode_time(hour: int, day_of_week: int) -> np.ndarray:
    """시간 정보를 10차원 벡터로 인코딩"""
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    day_sin = np.sin(2 * np.pi * day_of_week / 7)
    day_cos = np.cos(2 * np.pi * day_of_week / 7)

    is_weekday = 1.0 if day_of_week < 5 else 0.0
    is_weekend = 1.0 - is_weekday

    # 시간대 분류
    is_morning = 1.0 if 6 <= hour < 12 else 0.0
    is_afternoon = 1.0 if 12 <= hour < 18 else 0.0
    is_evening = 1.0 if 18 <= hour < 22 else 0.0
    is_night = 1.0 if hour >= 22 or hour < 6 else 0.0

    return np.array([hour_sin, hour_cos, day_sin, day_cos, is_weekday,
                    is_morning, is_afternoon, is_evening, is_night, is_weekend])


# 공간 인코딩 (4차원: 4개 zone one-hot)
ZONE_TO_INDEX = {'balcony': 0, 'bedroom': 1, 'kitchen': 2, 'living_room': 3}

def encode_spatial(zone: str) -> np.ndarray:
    """구역을 4차원 one-hot 벡터로 인코딩"""
    vec = np.zeros(4)
    if zone in ZONE_TO_INDEX:
        vec[ZONE_TO_INDEX[zone]] = 1.0
    return vec


# YOLO 인코딩 (14차원: 각 객체 존재 여부)
def encode_visual(objects: List[str]) -> np.ndarray:
    """YOLO 객체들을 14차원 벡터로 인코딩 (multi-hot)"""
    vec = np.zeros(14)
    for obj in objects:
        if obj in YOLO_CLASSES:
            idx = YOLO_CLASSES.index(obj)
            vec[idx] = 1.0
    return vec


# YAMNet 인코딩 (17차원: 각 소리 발생 여부)
def encode_audio(sounds: List[str]) -> np.ndarray:
    """YAMNet 소리들을 17차원 벡터로 인코딩 (multi-hot)"""
    vec = np.zeros(17)
    for sound in sounds:
        if sound in YAMNET_CLASSES:
            idx = YAMNET_CLASSES.index(sound)
            vec[idx] = 1.0
    return vec


# Pose 인코딩 (51차원: 17 keypoints * 3 coordinates)
def encode_pose_realistic(action_name: str, zone: str) -> np.ndarray:
    """
    행동에 따른 현실적인 포즈 생성 (랜덤 노이즈 추가)

    간단한 휴리스틱:
    - sleeping, resting_on_sofa: 누워있는 포즈 (y좌표 낮음)
    - sitting (eating, working): 앉은 포즈
    - standing (cooking, entering_room): 서있는 포즈
    - vacuuming: 서서 약간 굽은 포즈
    """
    # 기본 서있는 포즈 (normalized 0~1)
    base_pose = np.random.uniform(0.3, 0.7, 51)

    # 행동별 포즈 조정
    if 'sleeping' in action_name or 'resting' in action_name:
        # 누워있는 포즈: y좌표 낮게
        base_pose[1::3] = np.random.uniform(0.1, 0.3, 17)  # y coordinates
    elif 'eating' in action_name or 'working' in action_name or 'watching' in action_name:
        # 앉은 포즈: y좌표 중간
        base_pose[1::3] = np.random.uniform(0.3, 0.5, 17)
    elif 'cooking' in action_name or 'washing' in action_name or 'vacuuming' in action_name:
        # 서있는 포즈: y좌표 높게
        base_pose[1::3] = np.random.uniform(0.5, 0.8, 17)

    # 노이즈 추가 (자연스러운 움직임)
    noise = np.random.normal(0, 0.05, 51)
    base_pose = np.clip(base_pose + noise, 0.0, 1.0)

    return base_pose


class RealisticRoutineGenerator:
    """
    현실적인 일상 루틴 생성기

    하루 24시간을 시뮬레이션하며, 시간대별로 적절한 행동 패턴을 선택합니다.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.patterns = RealisticActionPattern.PATTERNS

        # 시간대별 행동 확률 (현실적인 루틴)
        self.time_based_activities = {
            'morning': {  # 6-12시
                'sleeping': 0.3,
                'eating_at_table': 0.2,
                'cooking': 0.15,
                'washing_dishes': 0.1,
                'working_on_laptop': 0.1,
                'phone_call': 0.05,
                'using_bathroom': 0.05,
                'entering_room': 0.05
            },
            'afternoon': {  # 12-18시
                'eating_at_table': 0.2,
                'cooking': 0.15,
                'working_on_laptop': 0.2,
                'watching_tv': 0.1,
                'snacking': 0.1,
                'phone_call': 0.1,
                'washing_dishes': 0.05,
                'resting_on_sofa': 0.05,
                'vacuuming': 0.05
            },
            'evening': {  # 18-22시
                'cooking': 0.2,
                'eating_at_table': 0.25,
                'washing_dishes': 0.15,
                'watching_tv': 0.15,
                'snacking': 0.1,
                'phone_call': 0.05,
                'resting_on_sofa': 0.05,
                'using_bathroom': 0.05
            },
            'night': {  # 22-6시
                'sleeping': 0.7,
                'watching_tv': 0.1,
                'working_on_laptop': 0.05,
                'phone_call': 0.05,
                'resting_on_sofa': 0.05,
                'using_bathroom': 0.05
            }
        }

    def get_time_period(self, hour: int) -> str:
        """시간대 반환"""
        if 6 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 18:
            return 'afternoon'
        elif 18 <= hour < 22:
            return 'evening'
        else:
            return 'night'

    def sample_action(self, hour: int, day_of_week: int) -> str:
        """시간대에 맞는 행동 샘플링"""
        period = self.get_time_period(hour)
        activities = self.time_based_activities[period]

        # 주말에는 청소 확률 높이기
        if day_of_week >= 5:  # 토, 일
            if 'vacuuming' in activities:
                activities = activities.copy()
                activities['vacuuming'] *= 3.0

        # 확률 정규화
        total = sum(activities.values())
        probs = {k: v/total for k, v in activities.items()}

        # 샘플링
        action_names = list(probs.keys())
        action_probs = list(probs.values())
        return self.rng.choice(action_names, p=action_probs)

    def generate_timestep(self,
                         action_name: str,
                         hour: int,
                         day_of_week: int,
                         current_pollution: Dict[str, float]) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        단일 타임스텝 생성

        Returns:
            (features_dict, pollution_label_4dim)
            features_dict = {
                'time': (10,),
                'spatial': (4,),
                'visual': (14,),
                'audio': (17,),
                'pose': (51,)
            }
        """
        pattern = self.patterns[action_name]

        # 구역 선택 (패턴에서 허용하는 구역 중 랜덤)
        zone = self.rng.choice(pattern['zones'])

        # 1. Time encoding (10dim)
        time_vec = encode_time(hour, day_of_week)

        # 2. Spatial encoding (4dim)
        spatial_vec = encode_spatial(zone)

        # 3. Visual encoding (14dim)
        visual_vec = encode_visual(pattern['visual_objects'])

        # 4. Audio encoding (17dim)
        audio_vec = encode_audio(pattern['audio_events'])

        # 5. Pose encoding (51dim)
        pose_vec = encode_pose_realistic(action_name, zone)

        # 딕셔너리 형태로 features 구성 (AttentionEncoder 입력 형식)
        features_dict = {
            'time': time_vec,
            'spatial': spatial_vec,
            'visual': visual_vec,
            'audio': audio_vec,
            'pose': pose_vec
        }

        # 오염도 업데이트
        pollution_delta = pattern['pollution_result']
        for zone_name, delta in pollution_delta.items():
            if zone_name in current_pollution:
                current_pollution[zone_name] += delta
                # 오염도는 0~1 범위로 클리핑
                current_pollution[zone_name] = np.clip(current_pollution[zone_name], 0.0, 1.0)

        # 라벨: 4개 zone의 오염도
        label = np.array([
            current_pollution['balcony'],
            current_pollution['bedroom'],
            current_pollution['kitchen'],
            current_pollution['living_room']
        ])

        return features_dict, label

    def generate_day(self,
                    day_of_week: int,
                    timesteps_per_hour: int = 4) -> Tuple[Dict[str, List], List[np.ndarray]]:
        """
        하루치 데이터 생성 (24시간)

        Args:
            day_of_week: 0=월요일, 6=일요일
            timesteps_per_hour: 시간당 타임스텝 수 (기본 4 = 15분마다)

        Returns:
            (features_dict, labels_list)
            features_dict = {
                'time': list of (10,),
                'spatial': list of (4,),
                ...
            }
        """
        # 각 modality별 리스트 초기화
        features_dict = {
            'time': [],
            'spatial': [],
            'visual': [],
            'audio': [],
            'pose': []
        }
        labels_list = []

        # 초기 오염도 (청소 후 상태)
        current_pollution = {
            'balcony': 0.1,
            'bedroom': 0.1,
            'kitchen': 0.1,
            'living_room': 0.1
        }

        # 24시간 시뮬레이션
        for hour in range(24):
            for _ in range(timesteps_per_hour):
                # 행동 샘플링
                action = self.sample_action(hour, day_of_week)

                # 타임스텝 생성
                features, label = self.generate_timestep(
                    action, hour, day_of_week, current_pollution
                )

                # 각 modality별로 추가
                for key in features_dict.keys():
                    features_dict[key].append(features[key])
                labels_list.append(label)

                # 가끔 자동 청소 (vacuuming)
                # 오염도가 0.7 넘으면 50% 확률로 청소
                if any(p > 0.7 for p in current_pollution.values()) and self.rng.rand() < 0.5:
                    # 청소 타임스텝 추가
                    vacuum_features, vacuum_label = self.generate_timestep(
                        'vacuuming', hour, day_of_week, current_pollution
                    )
                    for key in features_dict.keys():
                        features_dict[key].append(vacuum_features[key])
                    labels_list.append(vacuum_label)

        return features_dict, labels_list

    def generate_dataset(self,
                        num_days: int = 100,
                        timesteps_per_hour: int = 4,
                        output_path: str = None) -> Dict:
        """
        대규모 데이터셋 생성

        Args:
            num_days: 생성할 날짜 수 (100일 = ~10,000 타임스텝)
            timesteps_per_hour: 시간당 타임스텝 수
            output_path: 저장 경로 (None이면 저장 안 함)

        Returns:
            {'X': features, 'y': labels, 'metadata': {...}}
        """
        print("=" * 60)
        print("Realistic Dataset Generation")
        print("=" * 60)
        print(f"Target days: {num_days}")
        print(f"Timesteps per hour: {timesteps_per_hour}")
        print(f"Expected timesteps: ~{num_days * 24 * timesteps_per_hour}")
        print("=" * 60 + "\n")

        # 각 modality별 리스트 초기화
        all_features = {
            'time': [],
            'spatial': [],
            'visual': [],
            'audio': [],
            'pose': []
        }
        all_labels = []

        for day in range(num_days):
            day_of_week = day % 7

            if day % 10 == 0:
                print(f"[{day}/{num_days}] Generating day {day}...")

            features_dict, labels_list = self.generate_day(day_of_week, timesteps_per_hour)

            # 각 modality별로 extend
            for key in all_features.keys():
                all_features[key].extend(features_dict[key])
            all_labels.extend(labels_list)

        # numpy 배열로 변환
        features = {
            key: np.array(values, dtype=np.float32)
            for key, values in all_features.items()
        }
        y = np.array(all_labels, dtype=np.float32)

        print(f"\n{'=' * 60}")
        print(f"Dataset Generation Complete!")
        print(f"{'=' * 60}")
        print(f"Total timesteps: {len(y)}")
        print("Features shapes:")
        for key, value in features.items():
            print(f"  {key:10s}: {value.shape}")
        print(f"Labels shape: {y.shape}")
        print(f"{'=' * 60}\n")

        # 통계 출력
        print("Pollution Statistics:")
        for i, zone in enumerate(['balcony', 'bedroom', 'kitchen', 'living_room']):
            print(f"  {zone:15s}: mean={y[:, i].mean():.3f}, std={y[:, i].std():.3f}, "
                  f"min={y[:, i].min():.3f}, max={y[:, i].max():.3f}")
        print()

        # 데이터셋 객체 생성
        dataset = {
            'time': features['time'],
            'spatial': features['spatial'],
            'visual': features['visual'],
            'audio': features['audio'],
            'pose': features['pose'],
            'y': y,
            'metadata': {
                'num_days': num_days,
                'timesteps_per_hour': timesteps_per_hour,
                'total_timesteps': len(y),
                'num_zones': y.shape[1],
                'zones': ['balcony', 'bedroom', 'kitchen', 'living_room'],
                'yolo_classes': YOLO_CLASSES,
                'yamnet_classes': YAMNET_CLASSES,
                'generation_seed': self.rng.get_state()[1][0],
                'feature_dims': {
                    'time': 10,
                    'spatial': 4,
                    'visual': 14,
                    'audio': 17,
                    'pose': 51
                }
            }
        }

        # 저장
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            np.savez_compressed(output_path, **dataset)
            print(f"✅ Dataset saved to: {output_path}\n")

        return dataset


# 테스트 및 실행
if __name__ == "__main__":
    print("Testing Realistic Dataset Generator...\n")

    # 작은 데이터셋으로 테스트
    print("1️⃣  Small Test (3 days):")
    generator = RealisticRoutineGenerator(seed=42)
    test_dataset = generator.generate_dataset(
        num_days=3,
        timesteps_per_hour=4,
        output_path=None
    )

    print(f"\n2️⃣  Generating Full Dataset (100 days = ~10,000 timesteps):")
    output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'generated')
    output_path = os.path.join(output_dir, 'realistic_dataset_100days.npz')

    full_dataset = generator.generate_dataset(
        num_days=100,
        timesteps_per_hour=4,
        output_path=output_path
    )

    print("\n✅ All done!")
