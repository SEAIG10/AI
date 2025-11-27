"""
데이터셋 검증 및 시각화
생성된 데이터셋의 품질을 확인합니다.
"""

import numpy as np
import os
import sys

def load_dataset(path: str):
    """데이터셋 로드"""
    data = np.load(path, allow_pickle=True)
    return data

def validate_dataset(dataset_path: str):
    """데이터셋 검증"""
    print("=" * 60)
    print("Dataset Validation Report")
    print("=" * 60 + "\n")

    # 데이터 로드
    data = load_dataset(dataset_path)
    X = data['X']
    y = data['y']
    metadata = data['metadata'].item() if 'metadata' in data else {}

    print(f"📁 Dataset: {os.path.basename(dataset_path)}")
    print(f"💾 File size: {os.path.getsize(dataset_path) / 1024 / 1024:.2f} MB\n")

    # 1. 기본 정보
    print("1️⃣  Basic Information")
    print("-" * 60)
    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Total timesteps: {len(X):,}")
    print(f"Feature dimension: {X.shape[1]}")
    print(f"Number of zones: {y.shape[1]}")
    if metadata:
        print(f"Generation days: {metadata.get('num_days', 'N/A')}")
        print(f"Timesteps per hour: {metadata.get('timesteps_per_hour', 'N/A')}")
    print()

    # 2. 특성 통계
    print("2️⃣  Feature Statistics")
    print("-" * 60)
    print(f"Mean: {X.mean():.4f}")
    print(f"Std: {X.std():.4f}")
    print(f"Min: {X.min():.4f}")
    print(f"Max: {X.max():.4f}")
    print(f"Non-zero ratio: {(X != 0).sum() / X.size:.2%}")
    print()

    # 3. 라벨 통계 (구역별)
    print("3️⃣  Label Statistics (Pollution by Zone)")
    print("-" * 60)
    zones = metadata.get('zones', ['zone_0', 'zone_1', 'zone_2', 'zone_3'])
    for i, zone in enumerate(zones):
        zone_pollution = y[:, i]
        print(f"{zone:15s}:")
        print(f"  Mean:   {zone_pollution.mean():.3f}")
        print(f"  Std:    {zone_pollution.std():.3f}")
        print(f"  Min:    {zone_pollution.min():.3f}")
        print(f"  Max:    {zone_pollution.max():.3f}")
        print(f"  Median: {np.median(zone_pollution):.3f}")

        # 분포 분석
        clean = (zone_pollution < 0.3).sum()
        moderate = ((zone_pollution >= 0.3) & (zone_pollution < 0.7)).sum()
        dirty = (zone_pollution >= 0.7).sum()
        print(f"  Clean (<0.3):    {clean:6d} ({clean/len(zone_pollution)*100:5.1f}%)")
        print(f"  Moderate (0.3-0.7): {moderate:6d} ({moderate/len(zone_pollution)*100:5.1f}%)")
        print(f"  Dirty (>=0.7):   {dirty:6d} ({dirty/len(zone_pollution)*100:5.1f}%)")
        print()

    # 4. 시간 시계열 검증 (첫 100개 샘플)
    print("4️⃣  Time Series Check (first 100 timesteps)")
    print("-" * 60)
    sample_size = min(100, len(y))

    # 각 구역의 오염도 변화 확인
    for i, zone in enumerate(zones):
        pollution_change = np.diff(y[:sample_size, i])
        increase_ratio = (pollution_change > 0).sum() / len(pollution_change)
        decrease_ratio = (pollution_change < 0).sum() / len(pollution_change)

        print(f"{zone:15s}: increase={increase_ratio:.1%}, decrease={decrease_ratio:.1%}, "
              f"stable={(pollution_change == 0).sum() / len(pollution_change):.1%}")
    print()

    # 5. 데이터 품질 체크
    print("5️⃣  Data Quality Checks")
    print("-" * 60)

    # NaN/Inf 체크
    has_nan_X = np.isnan(X).any()
    has_inf_X = np.isinf(X).any()
    has_nan_y = np.isnan(y).any()
    has_inf_y = np.isinf(y).any()

    print(f"❌ NaN in features: {'YES (ERROR!)' if has_nan_X else 'NO (OK)'}")
    print(f"❌ Inf in features: {'YES (ERROR!)' if has_inf_X else 'NO (OK)'}")
    print(f"❌ NaN in labels: {'YES (ERROR!)' if has_nan_y else 'NO (OK)'}")
    print(f"❌ Inf in labels: {'YES (ERROR!)' if has_inf_y else 'NO (OK)'}")

    # 범위 체크 (오염도는 0~1 사이여야 함)
    out_of_range = ((y < 0) | (y > 1)).sum()
    print(f"❌ Labels out of range [0,1]: {out_of_range} ({out_of_range/y.size*100:.2f}%)")
    print()

    # 6. 클래스 사용 현황 (YOLO + YAMNet)
    if metadata and 'yolo_classes' in metadata and 'yamnet_classes' in metadata:
        print("6️⃣  Class Usage (YOLO + YAMNet)")
        print("-" * 60)

        # YOLO 클래스 (visual features, 14dim)
        # Feature 구조: [time(10) + spatial(4) + visual(14) + audio(17) + pose(51) + padding(3)]
        visual_start = 14
        visual_end = 14 + 14
        visual_features = X[:, visual_start:visual_end]

        yolo_classes = metadata['yolo_classes']
        print("YOLO Classes:")
        for i, cls in enumerate(yolo_classes):
            usage = (visual_features[:, i] > 0).sum()
            print(f"  {cls:20s}: {usage:6d} timesteps ({usage/len(X)*100:5.1f}%)")
        print()

        # YAMNet 클래스 (audio features, 17dim)
        audio_start = 14 + 14
        audio_end = 14 + 14 + 17
        audio_features = X[:, audio_start:audio_end]

        yamnet_classes = metadata['yamnet_classes']
        print("YAMNet Classes:")
        for i, cls in enumerate(yamnet_classes):
            usage = (audio_features[:, i] > 0).sum()
            print(f"  {cls:20s}: {usage:6d} timesteps ({usage/len(X)*100:5.1f}%)")
        print()

    print("=" * 60)
    print("✅ Validation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    # 생성된 데이터셋 검증
    dataset_paths = [
        '/Users/inter4259/Projects/SE_G10/data/generated/realistic_dataset_100days.npz',
        '/Users/inter4259/Projects/SE_G10/data/generated/realistic_dataset_500days_50k.npz'
    ]

    for path in dataset_paths:
        if os.path.exists(path):
            validate_dataset(path)
            print("\n" + "="*60 + "\n")
        else:
            print(f"⚠️  Dataset not found: {path}\n")
