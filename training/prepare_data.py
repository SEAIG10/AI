"""
데이터 준비 스크립트

합성 센서 데이터를 생성하고 저장합니다.
이 스크립트를 먼저 실행한 후 train_encoder.py, train_gru.py를 실행해야 합니다.
"""

import os
import sys

# 프로젝트 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from training.data_generator import SyntheticDataGenerator
from training.config import PATHS


def main():
    """합성 센서 데이터 생성 및 저장"""
    print("\n" + "=" * 70)
    print("합성 센서 데이터 준비")
    print("=" * 70)

    # 데이터 생성기 초기화
    generator = SyntheticDataGenerator()

    # 기존 데이터 확인
    if os.path.exists(PATHS['raw_features']):
        print(f"\n기존 데이터가 존재합니다: {PATHS['raw_features']}")
        user_input = input("덮어쓰시겠습니까? (y/n): ")

        if user_input.lower() != 'y':
            print("데이터 생성을 취소합니다.")
            print(f"기존 데이터를 사용하려면 train_encoder.py를 실행하세요.")
            return

    # 데이터 생성
    print("\n합성 센서 데이터 생성 중...")
    features_train, features_val, labels_train, labels_val = \
        generator.generate_training_data()

    # 데이터 저장
    print("\n데이터 저장 중...")
    generator.save_data(features_train, features_val, labels_train, labels_val)

    print("\n" + "=" * 70)
    print("데이터 준비 완료!")
    print("=" * 70)
    print(f"\n저장된 파일:")
    print(f"  - {PATHS['raw_features']}")
    print(f"\n다음 단계:")
    print(f"  1. python training/train_encoder.py")
    print(f"  2. python training/train_gru.py")


if __name__ == "__main__":
    main()
