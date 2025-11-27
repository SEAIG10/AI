"""
대규모 데이터셋 생성 스크립트
50,000개 이상의 타임스텝을 생성합니다.
"""

import os
import sys
sys.path.append(os.path.dirname(__file__))

from realistic_dataset_generator import RealisticRoutineGenerator

if __name__ == "__main__":
    print("🚀 Generating Large-Scale Realistic Dataset")
    print("=" * 60)
    print("Target: 50,000+ timesteps")
    print("Strategy: 500 days × 24 hours × 4 timesteps/hour = ~48,000 timesteps")
    print("(with automatic vacuuming: ~55,000 timesteps expected)")
    print("=" * 60 + "\n")

    # 생성기 초기화
    generator = RealisticRoutineGenerator(seed=42)

    # 500일치 데이터 생성
    output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'generated')
    output_path = os.path.join(output_dir, 'realistic_dataset_500days_50k.npz')

    dataset = generator.generate_dataset(
        num_days=500,
        timesteps_per_hour=4,
        output_path=output_path
    )

    print("\n✅ Large-scale dataset generation completed!")
    print(f"📁 Saved to: {output_path}")
    print(f"📊 Total timesteps: {dataset['X'].shape[0]:,}")
    print(f"💾 File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
