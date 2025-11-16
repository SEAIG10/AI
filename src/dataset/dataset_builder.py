"""
어텐션 컨텍스트 인코더를 사용한 데이터셋 빌더
시나리오를 학습에 바로 사용할 수 있는 (X, y) 형태의 수치 데이터셋으로 변환합니다.
"""

import numpy as np
import os
import sys
import tensorflow as tf
from typing import List, Dict, Tuple

# src 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from context_fusion.context_encoder import ContextEncoder
from context_fusion.attention_context_encoder import create_attention_encoder
from dataset.scenario_generator import ScenarioGenerator


class DatasetBuilder:
    """
    시나리오로부터 어텐션 컨텍스트 인코더를 사용하여 학습 데이터셋을 구축합니다.

    JSON 시나리오 → (X, y) 수치 데이터로 변환
    X: (N, 30, 160) - N개의 시퀀스, 각 시퀀스는 30 타임스텝의 160차원 어텐션 컨텍스트 벡터로 구성
    y: (N, 7) - N개의 레이블, 7개 구역 (이진값: 오염 또는 청소)
    """

    def __init__(self, use_attention: bool = True):
        """
        데이터셋 빌더를 초기화합니다.

        Args:
            use_attention: AttentionContextEncoder(160차원) 또는 레거시 ContextEncoder(338차원) 사용 여부
        """
        self.use_attention = use_attention
        self.legacy_encoder = ContextEncoder()  # 원본 특징 추출용
        self.generator = ScenarioGenerator()

        if use_attention:
            # 어텐션 인코더 생성
            self.attention_encoder = create_attention_encoder(
                visual_dim=14,
                context_dim=160
            )

    def scenario_to_training_sample(self, scenario: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        단일 시나리오를 (X, y) 학습 샘플로 변환합니다.

        Args:
            scenario: ScenarioGenerator에서 생성된 시나리오 딕셔너리

        Returns:
            X: (30, 160) 또는 (30, 338) 크기의 시퀀스 (use_attention에 따라 다름)
            y: (7,) 크기의 레이블
        """
        if not self.use_attention:
            # 레거시 모드: ContextEncoder 사용 (338차원)
            sequence_vectors = []
            for timestep in scenario["sequence"]:
                vector = self.legacy_encoder.encode(timestep)
                sequence_vectors.append(vector)

            X = np.array(sequence_vectors, dtype=np.float32)  # (30, 338)
            y = np.array(scenario["label"], dtype=np.float32)  # (7,)

            assert X.shape == (30, 338), f"Expected X shape (30, 338), got {X.shape}"
            assert y.shape == (7,), f"Expected y shape (7,), got {y.shape}"

            return X, y

        # 어텐션 모드: AttentionContextEncoder 사용 (160차원)
        # 먼저, 각 타임스텝에 대한 원본 특징을 준비합니다.
        batch_features = {
            'visual': [],
            'audio': [],
            'pose': [],
            'spatial': [],
            'time': []
        }

        for timestep in scenario["sequence"]:
            # 레거시 인코더의 헬퍼 메서드를 사용하여 원본 특징을 추출합니다.
            visual_vec = self.legacy_encoder._encode_visual(timestep)     # (14,)
            spatial_vec = self.legacy_encoder._encode_spatial(timestep)   # (7,)
            time_vec = self.legacy_encoder._encode_time(timestep.get("timestamp", 0))  # (10,)

            # 오디오 분류 및 키포인트 모의 데이터 생성 (모의 데이터용)
            # 실제 환경에서는 YAMNet(17-class)과 YOLO-Pose에서 이 값들을 가져옵니다.
            audio_vec = np.random.uniform(0, 0.3, 17).astype(np.float32)  # (17,) - 17-class 확률
            pose_vec = np.zeros(51, dtype=np.float32)  # (51,) - 모의 데이터에서는 사람 없음

            batch_features['visual'].append(visual_vec)
            batch_features['audio'].append(audio_vec)
            batch_features['pose'].append(pose_vec)
            batch_features['spatial'].append(spatial_vec)
            batch_features['time'].append(time_vec)

        # 텐서로 변환 (배치 크기=30)
        batch_tensors = {
            'visual': tf.constant(np.array(batch_features['visual']), dtype=tf.float32),    # (30, 14)
            'audio': tf.constant(np.array(batch_features['audio']), dtype=tf.float32),      # (30, 17)
            'pose': tf.constant(np.array(batch_features['pose']), dtype=tf.float32),        # (30, 51)
            'spatial': tf.constant(np.array(batch_features['spatial']), dtype=tf.float32),  # (30, 7)
            'time': tf.constant(np.array(batch_features['time']), dtype=tf.float32),        # (30, 10)
        }

        # 어텐션 인코더를 통과시킵니다.
        context_vectors = self.attention_encoder(batch_tensors, training=False)  # (30, 160)

        X = context_vectors.numpy().astype(np.float32)  # (30, 160)
        y = np.array(scenario["label"], dtype=np.float32)  # (7,)

        assert X.shape == (30, 160), f"Expected X shape (30, 160), got {X.shape}"
        assert y.shape == (7,), f"Expected y shape (7,), got {y.shape}"

        return X, y

    def build_dataset(
        self,
        num_samples_per_scenario: int = 100,
        noise_level: float = 0.1,
        train_split: float = 0.8
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        데이터 증강을 포함하여 전체 학습 데이터셋을 구축합니다.

        Args:
            num_samples_per_scenario: 시나리오당 생성할 변형 샘플 수
            noise_level: 데이터 증강을 위한 노이즈 강도 (0.0 - 1.0)
            train_split: 훈련/검증 데이터 분할 비율

        Returns:
            X_train: (N_train, 30, 160) - 어텐션 컨텍스트 벡터
            y_train: (N_train, 7)
            X_val: (N_val, 30, 160) - 어텐션 컨텍스트 벡터
            y_val: (N_val, 7)
        """
        print("=" * 70)
        print("학습 데이터셋 구축")
        print("=" * 70)

        # 기본 시나리오 생성
        scenarios = self.generator.generate_all_scenarios()
        print(f"\n[1] 기본 시나리오: {len(scenarios)}개")

        X_all = []
        y_all = []

        # 데이터 증강을 통해 샘플 생성
        print(f"[2] 시나리오당 {num_samples_per_scenario}개의 샘플 생성 중...")

        for scenario_idx, scenario in enumerate(scenarios):
            print(f"    - {scenario['name']}: ", end="")

            for sample_idx in range(num_samples_per_scenario):
                # 데이터 증강을 위해 노이즈 추가 (첫 샘플은 원본 유지)
                if sample_idx == 0:
                    augmented_scenario = scenario
                else:
                    augmented_scenario = self.generator.add_noise(scenario, noise_level)

                # 수치 데이터로 변환
                X, y = self.scenario_to_training_sample(augmented_scenario)
                X_all.append(X)
                y_all.append(y)

            print(f"{num_samples_per_scenario}개 샘플 생성 완료")

        # numpy 배열로 변환
        X_all = np.array(X_all, dtype=np.float32)
        y_all = np.array(y_all, dtype=np.float32)

        print(f"\n[3] 전체 데이터셋 크기:")
        print(f"    - X: {X_all.shape}")
        print(f"    - y: {y_all.shape}")

        # 훈련/검증 데이터 분할
        num_train = int(len(X_all) * train_split)
        indices = np.random.permutation(len(X_all))

        train_indices = indices[:num_train]
        val_indices = indices[num_train:]

        X_train = X_all[train_indices]
        y_train = y_all[train_indices]
        X_val = X_all[val_indices]
        y_val = y_all[val_indices]

        print(f"\n[4] 데이터 분할:")
        print(f"    - 훈련 데이터: {X_train.shape[0]}개 샘플")
        print(f"    - 검증 데이터:   {X_val.shape[0]}개 샘플")

        # 통계
        print(f"\n[5] 레이블 통계:")
        for zone_idx, zone_name in enumerate(self.generator.zones):
            positive_train = np.sum(y_train[:, zone_idx])
            positive_val = np.sum(y_val[:, zone_idx])
            print(f"    - {zone_name:15s}: 훈련={int(positive_train):3d}, 검증={int(positive_val):3d}")

        print("\n" + "=" * 70)
        print("데이터셋 구축 완료!")
        print("=" * 70)

        return X_train, y_train, X_val, y_val

    def save_dataset(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        save_path: str = "data/training_dataset.npz"
    ):
        """
        데이터셋을 .npz 파일로 저장합니다.

        Args:
            X_train, y_train, X_val, y_val: 데이터셋 배열
            save_path: .npz 파일을 저장할 경로
        """
        # 필요시 디렉토리 생성
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"디렉토리를 생성했습니다: {save_dir}")

        # 저장
        np.savez_compressed(
            save_path,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val
        )

        file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
        print(f"\n데이터셋이 저장되었습니다: {save_path}")
        print(f"  파일 크기: {file_size_mb:.2f} MB")

    def load_dataset(self, load_path: str = "data/training_dataset.npz"):
        """
        .npz 파일로부터 데이터셋을 로드합니다.

        Args:
            load_path: .npz 파일 경로

        Returns:
            X_train, y_train, X_val, y_val
        """
        data = np.load(load_path)

        X_train = data['X_train']
        y_train = data['y_train']
        X_val = data['X_val']
        y_val = data['y_val']

        print(f"데이터셋을 로드했습니다: {load_path}")
        print(f"  훈련 데이터: {X_train.shape[0]}개 샘플")
        print(f"  검증 데이터:   {X_val.shape[0]}개 샘플")

        return X_train, y_train, X_val, y_val


def test_dataset_builder():
    """데이터셋 빌더 테스트"""
    print("\n" + "=" * 70)
    print("데이터셋 빌더 테스트")
    print("=" * 70 + "\n")

    builder = DatasetBuilder()

    # 데이터셋 구축 (테스트용으로 작은 규모)
    X_train, y_train, X_val, y_val = builder.build_dataset(
        num_samples_per_scenario=50,  # 시나리오당 50개 샘플
        noise_level=0.1,
        train_split=0.8
    )

    # Shape 검증
    print(f"\n[검증]")
    print(f"X_train shape: {X_train.shape} (예상: (N, 30, 160))")
    print(f"y_train shape: {y_train.shape} (예상: (N, 7))")
    print(f"X_val shape: {X_val.shape}")
    print(f"y_val shape: {y_val.shape}")

    # 샘플 검사
    print(f"\n[샘플 검사]")
    print(f"X_train[0][0][:10] = {X_train[0][0][:10]}")
    print(f"y_train[0] = {y_train[0]} (구역: {builder.generator.zones})")

    # 저장 테스트
    print(f"\n[저장 테스트]")
    builder.save_dataset(X_train, y_train, X_val, y_val, "data/test_dataset.npz")

    # 로드 테스트
    print(f"\n[로드 테스트]")
    X_train_loaded, y_train_loaded, X_val_loaded, y_val_loaded = builder.load_dataset("data/test_dataset.npz")

    # 로드된 데이터가 일치하는지 검증
    assert np.allclose(X_train, X_train_loaded), "로드된 X_train이 일치하지 않습니다!"
    assert np.allclose(y_train, y_train_loaded), "로드된 y_train이 일치하지 않습니다!"
    print("\n모든 테스트 통과!")


if __name__ == "__main__":
    test_dataset_builder()
