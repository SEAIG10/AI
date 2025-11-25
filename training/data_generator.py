"""
합성 센서 데이터 생성기

시나리오 기반으로 다중 모달 센서 데이터를 생성합니다.
이 데이터는 AttentionContextEncoder와 GRU 모델 학습에 사용됩니다.

데이터 흐름:
1. 시나리오 → 원본 특징 (visual, audio, pose, spatial, time)
2. 원본 특징 → AttentionEncoder 학습용 데이터
3. AttentionEncoder → 160차원 컨텍스트 벡터
4. 컨텍스트 벡터 → GRU 학습용 데이터
"""

import os
import sys
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple

# 프로젝트 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.dataset.scenario_generator import ScenarioGenerator
from src.context_fusion.context_encoder import ContextEncoder
from training.config import (
    SENSOR_DIMS, DATA_CONFIG, PATHS
)


class SyntheticDataGenerator:
    """
    합성 센서 데이터 생성기

    시나리오로부터 5가지 센서 모달리티의 원본 특징을 생성합니다:
    - visual: YOLO 객체 탐지 (14차원)
    - audio: YAMNet 오디오 분류 (17차원)
    - pose: 사람 자세 키포인트 (51차원)
    - spatial: GPS 구역 (7차원)
    - time: 시간 특징 (10차원)
    """

    def __init__(self):
        """데이터 생성기를 초기화합니다."""
        self.scenario_gen = ScenarioGenerator()
        self.legacy_encoder = ContextEncoder()  # 특징 추출용
        self.zones = self.scenario_gen.zones

    def _extract_raw_features(self, scenario: Dict) -> Dict[str, np.ndarray]:
        """
        시나리오로부터 원본 센서 특징을 추출합니다.

        Args:
            scenario: ScenarioGenerator에서 생성된 시나리오

        Returns:
            각 모달리티별 특징 딕셔너리:
            - visual: (30, 14)
            - audio: (30, 17)
            - pose: (30, 51)
            - spatial: (30, 7)
            - time: (30, 10)
        """
        batch_features = {
            'visual': [],
            'audio': [],
            'pose': [],
            'spatial': [],
            'time': []
        }

        for timestep in scenario["sequence"]:
            # Visual: YOLO 객체 탐지 (14 클래스)
            visual_vec = self.legacy_encoder._encode_visual(timestep)
            batch_features['visual'].append(visual_vec)

            # Spatial: GPS 구역 (7 구역)
            spatial_vec = self.legacy_encoder._encode_spatial(timestep)
            batch_features['spatial'].append(spatial_vec)

            # Time: 시간 특징 (10차원)
            time_vec = self.legacy_encoder._encode_time(
                timestep.get("timestamp", 0)
            )
            batch_features['time'].append(time_vec)

            # Audio: YAMNet 17-class 확률
            # 실제 환경에서는 YAMNet에서 추출하지만, 여기서는 합성 데이터 생성
            audio_events = timestep.get("audio_events", [])
            audio_vec = self._synthesize_audio_features(audio_events)
            batch_features['audio'].append(audio_vec)

            # Pose: 사람 자세 키포인트 (51차원 = 17관절 × 3좌표)
            # 실제 환경에서는 YOLO-Pose에서 추출하지만, 여기서는 합성 데이터 생성
            pose_vec = self._synthesize_pose_features(timestep)
            batch_features['pose'].append(pose_vec)

        # numpy 배열로 변환
        features = {
            key: np.array(values, dtype=np.float32)
            for key, values in batch_features.items()
        }

        # Shape 검증
        assert features['visual'].shape == (30, SENSOR_DIMS['visual'])
        assert features['audio'].shape == (30, SENSOR_DIMS['audio'])
        assert features['pose'].shape == (30, SENSOR_DIMS['pose'])
        assert features['spatial'].shape == (30, SENSOR_DIMS['spatial'])
        assert features['time'].shape == (30, SENSOR_DIMS['time'])

        return features

    def _synthesize_audio_features(self, audio_events: List[Dict]) -> np.ndarray:
        """
        오디오 이벤트로부터 YAMNet 17-class 확률 벡터를 합성합니다.

        실제 환경에서는 YAMNet 모델에서 직접 추출하지만,
        학습 데이터 생성을 위해 합성 데이터를 생성합니다.

        Args:
            audio_events: 오디오 이벤트 리스트

        Returns:
            audio_vec: (17,) YAMNet 클래스 확률
        """
        # YAMNet 17-class 매핑 (간단한 버전)
        yamnet_class_map = {
            "Silence": 0,
            "Cooking": 1,
            "Dishes": 2,
            "Television": 3,
            "Music": 4,
            "Conversation": 5,
            "Vacuum_cleaner": 6,
            "Water": 7,
        }

        # 기본값: 낮은 확률의 배경 노이즈
        audio_vec = np.random.uniform(0.0, 0.1, 17).astype(np.float32)

        # 이벤트가 있으면 해당 클래스의 확률을 높임
        if audio_events:
            event = audio_events[0]
            event_name = event.get("event", "Silence")
            confidence = event.get("confidence", 0.5)

            if event_name in yamnet_class_map:
                class_idx = yamnet_class_map[event_name]
                audio_vec[class_idx] = confidence

        # 확률 합이 1이 되도록 정규화
        audio_vec = audio_vec / np.sum(audio_vec)

        return audio_vec

    def _synthesize_pose_features(self, timestep: Dict) -> np.ndarray:
        """
        타임스텝으로부터 사람 자세 키포인트를 합성합니다.

        실제 환경에서는 YOLO-Pose에서 직접 추출하지만,
        학습 데이터 생성을 위해 합성 데이터를 생성합니다.

        Args:
            timestep: 시나리오의 타임스텝

        Returns:
            pose_vec: (51,) 키포인트 좌표
        """
        visual_events = timestep.get("visual_events", [])

        # 사람이 있는지 확인
        has_person = any(
            event.get("class") == "person"
            for event in visual_events
        )

        if has_person:
            # 사람이 있으면 정규화된 키포인트 생성
            # 실제로는 0.0-1.0 범위의 정규화된 좌표
            pose_vec = np.random.uniform(0.3, 0.7, 51).astype(np.float32)
        else:
            # 사람이 없으면 0벡터 (키포인트 없음)
            pose_vec = np.zeros(51, dtype=np.float32)

        return pose_vec

    def generate_training_data(
        self,
        num_samples_per_scenario: int = None,
        noise_level: float = None,
        train_split: float = None
    ) -> Tuple[Dict, Dict, np.ndarray, np.ndarray]:
        """
        전체 학습 데이터를 생성합니다.

        Args:
            num_samples_per_scenario: 시나리오당 샘플 수 (기본값: config)
            noise_level: 데이터 증강 노이즈 강도 (기본값: config)
            train_split: 훈련/검증 분할 비율 (기본값: config)

        Returns:
            features_train: 훈련 특징 딕셔너리 (각 키: (N_train, 30, dim))
            features_val: 검증 특징 딕셔너리 (각 키: (N_val, 30, dim))
            labels_train: 훈련 레이블 (N_train, 7)
            labels_val: 검증 레이블 (N_val, 7)
        """
        # 기본값 설정
        if num_samples_per_scenario is None:
            num_samples_per_scenario = DATA_CONFIG['num_samples_per_scenario']
        if noise_level is None:
            noise_level = DATA_CONFIG['noise_level']
        if train_split is None:
            train_split = DATA_CONFIG['train_split']

        print("\n" + "=" * 70)
        print("합성 센서 데이터 생성")
        print("=" * 70)

        # 시나리오 생성
        scenarios = self.scenario_gen.generate_all_scenarios()
        print(f"\n[1] 기본 시나리오: {len(scenarios)}개")

        # 모든 샘플 저장
        all_features = {key: [] for key in ['visual', 'audio', 'pose', 'spatial', 'time']}
        all_labels = []

        # 데이터 증강을 통해 샘플 생성
        print(f"[2] 시나리오당 {num_samples_per_scenario}개의 샘플 생성 중...\n")

        for scenario_idx, scenario in enumerate(scenarios):
            print(f"  [{scenario_idx+1}/{len(scenarios)}] {scenario['name']}: ", end="")

            for sample_idx in range(num_samples_per_scenario):
                # 데이터 증강 (첫 샘플은 원본 유지)
                if sample_idx == 0:
                    augmented_scenario = scenario
                else:
                    augmented_scenario = self.scenario_gen.add_noise(
                        scenario, noise_level
                    )

                # 원본 특징 추출
                features = self._extract_raw_features(augmented_scenario)
                label = np.array(scenario["label"], dtype=np.float32)

                # 저장
                for key in all_features.keys():
                    all_features[key].append(features[key])
                all_labels.append(label)

            print(f"{num_samples_per_scenario}개 완료")

        # numpy 배열로 변환
        for key in all_features.keys():
            all_features[key] = np.array(all_features[key], dtype=np.float32)
        all_labels = np.array(all_labels, dtype=np.float32)

        print(f"\n[3] 전체 데이터셋 크기:")
        for key, value in all_features.items():
            print(f"  {key:10s}: {value.shape}")
        print(f"  labels    : {all_labels.shape}")

        # 훈련/검증 분할
        num_samples = len(all_labels)
        num_train = int(num_samples * train_split)
        indices = np.random.permutation(num_samples)

        train_indices = indices[:num_train]
        val_indices = indices[num_train:]

        features_train = {
            key: values[train_indices]
            for key, values in all_features.items()
        }
        features_val = {
            key: values[val_indices]
            for key, values in all_features.items()
        }
        labels_train = all_labels[train_indices]
        labels_val = all_labels[val_indices]

        print(f"\n[4] 데이터 분할:")
        print(f"  훈련 데이터: {len(train_indices)}개 샘플")
        print(f"  검증 데이터: {len(val_indices)}개 샘플")

        # 레이블 통계
        print(f"\n[5] 레이블 통계:")
        for zone_idx, zone_name in enumerate(self.zones):
            positive_train = np.sum(labels_train[:, zone_idx])
            positive_val = np.sum(labels_val[:, zone_idx])
            print(f"  {zone_name:15s}: 훈련={int(positive_train):3d}, "
                  f"검증={int(positive_val):3d}")

        print("\n" + "=" * 70)
        print("데이터 생성 완료!")
        print("=" * 70 + "\n")

        return features_train, features_val, labels_train, labels_val

    def save_data(
        self,
        features_train: Dict,
        features_val: Dict,
        labels_train: np.ndarray,
        labels_val: np.ndarray,
        save_path: str = None
    ):
        """
        생성된 데이터를 .npz 파일로 저장합니다.

        Args:
            features_train: 훈련 특징 딕셔너리
            features_val: 검증 특징 딕셔너리
            labels_train: 훈련 레이블
            labels_val: 검증 레이블
            save_path: 저장 경로 (기본값: config)
        """
        if save_path is None:
            save_path = PATHS['raw_features']

        # 저장할 데이터 준비
        save_dict = {
            'labels_train': labels_train,
            'labels_val': labels_val,
        }

        # 훈련 특징 추가
        for key, value in features_train.items():
            save_dict[f'{key}_train'] = value

        # 검증 특징 추가
        for key, value in features_val.items():
            save_dict[f'{key}_val'] = value

        # 저장
        np.savez_compressed(save_path, **save_dict)

        file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
        print(f"데이터가 저장되었습니다: {save_path}")
        print(f"  파일 크기: {file_size_mb:.2f} MB")

    def load_data(self, load_path: str = None):
        """
        저장된 데이터를 로드합니다.

        Args:
            load_path: 로드 경로 (기본값: config)

        Returns:
            features_train, features_val, labels_train, labels_val
        """
        if load_path is None:
            load_path = PATHS['raw_features']

        data = np.load(load_path)

        # 특징 딕셔너리 재구성
        features_train = {
            'visual': data['visual_train'],
            'audio': data['audio_train'],
            'pose': data['pose_train'],
            'spatial': data['spatial_train'],
            'time': data['time_train'],
        }

        features_val = {
            'visual': data['visual_val'],
            'audio': data['audio_val'],
            'pose': data['pose_val'],
            'spatial': data['spatial_val'],
            'time': data['time_val'],
        }

        labels_train = data['labels_train']
        labels_val = data['labels_val']

        print(f"데이터를 로드했습니다: {load_path}")
        print(f"  훈련 데이터: {len(labels_train)}개 샘플")
        print(f"  검증 데이터: {len(labels_val)}개 샘플")

        return features_train, features_val, labels_train, labels_val


if __name__ == "__main__":
    """데이터 생성 테스트"""
    print("\n" + "=" * 70)
    print("합성 데이터 생성기 테스트")
    print("=" * 70)

    # 데이터 생성
    generator = SyntheticDataGenerator()
    features_train, features_val, labels_train, labels_val = \
        generator.generate_training_data(num_samples_per_scenario=50)

    # 저장
    print("\n[저장 테스트]")
    generator.save_data(features_train, features_val, labels_train, labels_val)

    # 로드
    print("\n[로드 테스트]")
    loaded_train, loaded_val, loaded_labels_train, loaded_labels_val = \
        generator.load_data()

    # 검증
    print("\n[검증]")
    assert np.allclose(features_train['visual'], loaded_train['visual'])
    assert np.allclose(labels_train, loaded_labels_train)
    print("모든 테스트 통과!")
