"""
대규모 데이터셋 생성 스크립트 (옵션 2 자동 실행)
2000일 × 3 시드 = ~220,000 타임스텝
"""

import os
import sys
import numpy as np

# 프로젝트 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.dataset.realistic_dataset_generator import RealisticRoutineGenerator
from training.config import PATHS


def compute_pollution_occurred_labels(y: np.ndarray) -> np.ndarray:
    """오염 발생 여부 라벨 계산"""
    pollution_occurred = np.zeros(len(y), dtype=np.float32)

    for i in range(1, len(y)):
        if np.any(y[i] > y[i-1]):
            pollution_occurred[i] = 1.0

    if len(y) > 0 and np.any(y[0] > 0.1):
        pollution_occurred[0] = 1.0

    return pollution_occurred


def generate_multi_seed_dataset(num_days: int, num_seeds: int, timesteps_per_hour: int = 4):
    """여러 시드로 데이터 생성"""
    print("\n" + "=" * 80)
    print("대규모 다중 시드 데이터셋 생성")
    print("=" * 80)
    print(f"설정:")
    print(f"  - 시드당 날짜 수: {num_days}일")
    print(f"  - 시드 개수: {num_seeds}개")
    print(f"  - 시간당 타임스텝: {timesteps_per_hour}개 (15분 간격)")
    print(f"  - 예상 총 타임스텝: ~{num_days * num_seeds * 24 * timesteps_per_hour * 1.15:,.0f}개")
    print(f"    (자동 청소 트리거 포함)")
    print("=" * 80 + "\n")

    all_data = {
        'time': [],
        'spatial': [],
        'visual': [],
        'audio': [],
        'pose': [],
        'y': [],
        'pollution_occurred': []
    }

    for seed_idx in range(num_seeds):
        seed = 42 + seed_idx

        print(f"\n{'─' * 80}")
        print(f"시드 {seed_idx + 1}/{num_seeds} (seed={seed}) 생성 중...")
        print(f"{'─' * 80}")

        generator = RealisticRoutineGenerator(seed=seed)

        dataset = generator.generate_dataset(
            num_days=num_days,
            timesteps_per_hour=timesteps_per_hour,
            output_path=None
        )

        for key in ['time', 'spatial', 'visual', 'audio', 'pose', 'y']:
            all_data[key].append(dataset[key])

        pollution_labels = compute_pollution_occurred_labels(dataset['y'])
        all_data['pollution_occurred'].append(pollution_labels)

        print(f"✓ 시드 {seed} 완료: {len(dataset['y']):,} 타임스텝")

    print(f"\n{'=' * 80}")
    print("모든 시드 데이터 병합 중...")
    print(f"{'=' * 80}")

    merged_dataset = {}
    for key in ['time', 'spatial', 'visual', 'audio', 'pose', 'y', 'pollution_occurred']:
        merged_dataset[key] = np.concatenate(all_data[key], axis=0)
        print(f"  {key:20s}: {merged_dataset[key].shape}")

    merged_dataset['metadata'] = {
        'num_days_per_seed': num_days,
        'num_seeds': num_seeds,
        'timesteps_per_hour': timesteps_per_hour,
        'total_timesteps': len(merged_dataset['y']),
        'total_days_equivalent': num_days * num_seeds,
        'seeds_used': list(range(42, 42 + num_seeds)),
        'num_zones': 4,
        'zones': ['balcony', 'bedroom', 'kitchen', 'living_room'],
        'feature_dims': {
            'time': 10,
            'spatial': 4,
            'visual': 14,
            'audio': 17,
            'pose': 51
        },
        'new_labels': {
            'pollution_occurred': '오염 발생 여부 (1: 발생, 0: 미발생)'
        }
    }

    # 통계
    y = merged_dataset['y']
    pollution_occurred = merged_dataset['pollution_occurred']

    print(f"\n{'=' * 80}")
    print("데이터셋 생성 완료!")
    print(f"{'=' * 80}")
    print(f"총 타임스텝: {merged_dataset['metadata']['total_timesteps']:,}")
    print(f"등가 일수: {merged_dataset['metadata']['total_days_equivalent']:,}일")

    print(f"\n{'─' * 80}")
    print("구역별 오염도 통계")
    print(f"{'─' * 80}")
    zones = ['balcony', 'bedroom', 'kitchen', 'living_room']
    for i, zone in enumerate(zones):
        print(f"  {zone:15s}: mean={y[:, i].mean():.3f}, std={y[:, i].std():.3f}, "
              f"min={y[:, i].min():.3f}, max={y[:, i].max():.3f}")

    print(f"\n{'─' * 80}")
    print("오염 발생 여부 통계")
    print(f"{'─' * 80}")
    num_polluted = int(pollution_occurred.sum())
    num_clean = len(pollution_occurred) - num_polluted
    print(f"  오염 발생:     {num_polluted:,} 타임스텝 ({num_polluted/len(pollution_occurred)*100:.1f}%)")
    print(f"  오염 미발생:   {num_clean:,} 타임스텝 ({num_clean/len(pollution_occurred)*100:.1f}%)")
    print(f"  합계:          {len(pollution_occurred):,} 타임스텝")

    return merged_dataset


if __name__ == "__main__":
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 15 + "대규모 데이터셋 자동 생성" + " " * 30 + "║")
    print("╚" + "═" * 78 + "╝")
    print("\n옵션 2: 2000일 × 3 시드 (약 220,000 타임스텝)\n")

    # 옵션 2 자동 설정
    num_days = 2000
    num_seeds = 3
    output_filename = 'massive_dataset_2000days_3seeds.npz'

    output_path = os.path.join(PATHS['data_dir'], output_filename)

    if os.path.exists(output_path):
        print(f"⚠️  기존 데이터가 존재합니다: {output_path}")
        print(f"파일 크기: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
        print("\n기존 파일을 삭제하고 새로 생성합니다...\n")

    # 데이터 생성
    dataset = generate_multi_seed_dataset(
        num_days=num_days,
        num_seeds=num_seeds,
        timesteps_per_hour=4
    )

    # 저장
    print(f"\n{'=' * 80}")
    print("데이터셋 저장 중...")
    print(f"{'=' * 80}")
    print(f"경로: {output_path}")

    np.savez_compressed(output_path, **dataset)

    file_size_mb = os.path.getsize(output_path) / 1024 / 1024

    print("\n" + "=" * 80)
    print("✅ 데이터 준비 완료!")
    print("=" * 80)
    print(f"\n저장된 파일:")
    print(f"  📁 {output_path}")
    print(f"  📊 총 타임스텝: {len(dataset['y']):,}")
    print(f"  💾 파일 크기: {file_size_mb:.2f} MB")
    print(f"  🗜️  압축률: ~{len(dataset['y']) * 96 * 4 / 1024 / 1024 / file_size_mb:.1f}x")
    print(f"\n다음 단계:")
    print(f"  1. python training/train_encoder.py")
    print(f"  2. python training/train_gru.py")
