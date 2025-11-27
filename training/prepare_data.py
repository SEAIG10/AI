"""
데이터 준비 스크립트

현실적인 행동 패턴 기반 대규모 데이터셋을 생성합니다.
이 스크립트를 먼저 실행한 후 train_gru.py를 실행해야 합니다.
"""

import os
import sys

# 프로젝트 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.dataset.realistic_dataset_generator import RealisticRoutineGenerator
from training.config import PATHS


def main():
    """현실적인 센서 데이터 생성 및 저장"""
    print("\n" + "=" * 70)
    print("현실적인 일상 루틴 데이터 준비")
    print("=" * 70)

    # 데이터 생성기 초기화
    generator = RealisticRoutineGenerator(seed=42)

    # 기존 데이터 확인
    output_path = os.path.join(PATHS['data_dir'], 'realistic_training_dataset.npz')

    if os.path.exists(output_path):
        print(f"\n⚠️  기존 데이터가 존재합니다: {output_path}")
        user_input = input("덮어쓰시겠습니까? (y/n): ")

        if user_input.lower() != 'y':
            print("데이터 생성을 취소합니다.")
            print(f"기존 데이터를 사용하려면 train_gru.py를 실행하세요.")
            return

    # 데이터 생성
    print("\n🚀 현실적인 일상 루틴 데이터 생성 중...")
    print("   - 500일 시뮬레이션")
    print("   - 시간당 4 타임스텝 (15분마다)")
    print("   - 실제 학습된 YOLO(14) + YAMNet(17) 클래스만 사용")
    print("   - 인과관계 반영 (행동 → 오염 발생)")
    print()

    dataset = generator.generate_dataset(
        num_days=500,
        timesteps_per_hour=4,
        output_path=output_path
    )

    print("\n" + "=" * 70)
    print("✅ 데이터 준비 완료!")
    print("=" * 70)
    print(f"\n저장된 파일:")
    print(f"  📁 {output_path}")
    print(f"  📊 Total timesteps: {len(dataset['y']):,}")
    print(f"  💾 File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    print(f"\n다음 단계:")
    print(f"  1. python training/train_encoder.py  # AttentionEncoder 학습 (Base Layer)")
    print(f"  2. python training/train_gru.py      # GRU 학습 (Head Layer)")


if __name__ == "__main__":
    main()
