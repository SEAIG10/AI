"""
구역 관리 및 GRU Head 동적 재생성
Backend로부터 구역 정보를 받아 GRU 모델의 Head를 교체합니다.
"""

import sys
import os
import threading
import time
from typing import List, Dict, Optional

# 프로젝트 루트 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model.gru_model import FedPerGRUModel


class ZoneManager:
    """
    구역 정보 관리 및 GRU Head 동적 교체

    역할:
    1. Backend로부터 MQTT를 통해 구역 정보 수신
    2. GRU Head를 해당 구역 개수로 재생성
    3. 온디바이스 학습 트리거 (향후 구현)
    """

    def __init__(self, gru_model: FedPerGRUModel, mqtt_client=None, on_device_trainer=None):
        """
        ZoneManager를 초기화합니다.

        Args:
            gru_model: GRU 모델 인스턴스
            mqtt_client: MQTT 클라이언트 (선택)
            on_device_trainer: OnDeviceTrainer 인스턴스 (선택)
        """
        self.gru_model = gru_model
        self.mqtt_client = mqtt_client
        self.on_device_trainer = on_device_trainer

        # 현재 구역 정보
        self.current_zones: List[Dict] = []
        self.zone_name_mapping: Dict[str, str] = {}  # {id: name}

        # 기본 4개 구역으로 시작
        self._initialize_default_zones()

        print(f"\n{'='*60}")
        print(f"ZoneManager Initialized")
        print(f"{'='*60}")
        print(f"  Current zones: {len(self.current_zones)}")
        print(f"  GRU output dim: {self.gru_model.num_zones}")
        print(f"{'='*60}\n")

    def _initialize_default_zones(self):
        """기본 4개 구역으로 초기화"""
        self.current_zones = [
            {"id": "1", "name": "거실", "name_en": "living_room"},
            {"id": "2", "name": "주방", "name_en": "kitchen"},
            {"id": "3", "name": "침실", "name_en": "bedroom"},
            {"id": "4", "name": "베란다", "name_en": "balcony"}
        ]
        self._update_zone_mapping()

    def _update_zone_mapping(self):
        """구역 ID → 이름 매핑 업데이트"""
        self.zone_name_mapping = {
            zone.get('id', str(i)): zone.get('name', f'Zone {i}')
            for i, zone in enumerate(self.current_zones)
        }

    def update_zones(self, home_id: str, zones: List[Dict]):
        """
        Backend로부터 받은 구역 정보로 업데이트합니다.

        Args:
            home_id: 집 ID
            zones: 구역 정보 리스트
                [
                    {"id": 1, "name": "거실", "points": [...]},
                    {"id": 2, "name": "주방", "points": [...]}
                ]
        """
        print(f"\n{'='*60}")
        print(f"🏠 Zone Update Request")
        print(f"{'='*60}")
        print(f"  Home ID: {home_id}")
        print(f"  Previous zones: {len(self.current_zones)}")
        print(f"  New zones: {len(zones)}")

        # 구역 목록 출력
        print(f"\n  Zone List:")
        for zone in zones:
            zone_name = zone.get('name', 'Unknown')
            zone_id = zone.get('id', '?')
            num_points = len(zone.get('points', []))
            print(f"    - [{zone_id}] {zone_name} ({num_points} points)")

        # 구역 정보 저장
        self.current_zones = zones
        self._update_zone_mapping()

        # GRU Head 재생성
        num_zones = len(zones)
        print(f"\n🔧 Replacing GRU Head...")
        print(f"   Old output dim: {self.gru_model.num_zones}")
        print(f"   New output dim: {num_zones}")

        self.replace_gru_head(num_zones)

        print(f"\n✅ Zone update complete!")
        print(f"{'='*60}\n")

        # MQTT로 상태 전송 (선택)
        if self.mqtt_client:
            self.mqtt_client.publish_device_status(
                "zones_updated",
                num_zones=num_zones,
                zones=[z.get('name') for z in zones]
            )

    def replace_gru_head(self, num_zones: int):
        """
        GRU Head를 새로운 구역 개수로 교체합니다.

        Args:
            num_zones: 새로운 구역 개수
        """
        try:
            # GRU Head 교체
            self.gru_model.replace_head(num_zones)

            # 모델 재컴파일
            self.gru_model.compile_model(
                learning_rate=0.0005,
                loss='mse',
                metrics=['mae', 'mse']
            )

            print(f"✅ GRU Head replaced successfully!")
            print(f"   New architecture:")
            print(f"     - Base: GRU(64) → GRU(32) [Frozen]")
            print(f"     - Head: Dense(16) → Dense({num_zones}) [Trainable]")

        except Exception as e:
            print(f"❌ Error replacing GRU Head: {e}")
            import traceback
            traceback.print_exc()

    def get_current_zones(self) -> List[Dict]:
        """현재 구역 정보 반환"""
        return self.current_zones

    def get_zone_name(self, zone_id: str) -> str:
        """구역 ID로 이름 조회"""
        return self.zone_name_mapping.get(zone_id, f"Unknown Zone ({zone_id})")

    def get_zone_name_by_index(self, index: int) -> str:
        """인덱스로 구역 이름 조회"""
        if 0 <= index < len(self.current_zones):
            return self.current_zones[index].get('name', f'Zone {index}')
        return f'Unknown Zone {index}'

    def start_on_device_training(self):
        """
        온디바이스 학습 시작 (백그라운드)
        """
        if self.on_device_trainer is None:
            print(f"\nWarning: On-device training requested but OnDeviceTrainer not available")
            return

        print(f"\nOn-device training requested")
        print(f"  Current buffer size: {len(self.on_device_trainer.X_buffer)}")
        print(f"  Min samples needed: {self.on_device_trainer.min_samples_for_training}")

        if len(self.on_device_trainer.X_buffer) >= self.on_device_trainer.min_samples_for_training:
            print(f"  Starting training...")
            self.on_device_trainer.start_background_training()
        else:
            print(f"  Not enough samples yet. Waiting for more data...")


def test_zone_manager():
    """ZoneManager 테스트"""
    print("\n" + "="*60)
    print("ZoneManager Test")
    print("="*60)

    # GRU 모델 로드 (테스트용)
    print("\n1. Loading GRU model...")
    from src.model.gru_model import FedPerGRUModel

    gru_model = FedPerGRUModel(num_zones=4, context_dim=160)
    print(f"   Initial GRU output dim: {gru_model.num_zones}")

    # ZoneManager 초기화
    print("\n2. Initializing ZoneManager...")
    zone_manager = ZoneManager(gru_model)

    # 구역 업데이트 시뮬레이션
    print("\n3. Simulating zone update (3 zones)...")
    test_zones = [
        {"id": "1", "name": "거실", "points": []},
        {"id": "2", "name": "주방", "points": []},
        {"id": "3", "name": "침실", "points": []}
    ]

    zone_manager.update_zones(
        home_id="test_home_123",
        zones=test_zones
    )

    # 결과 확인
    print("\n4. Verifying...")
    print(f"   Current zones: {len(zone_manager.get_current_zones())}")
    print(f"   GRU output dim: {gru_model.num_zones}")

    for i, zone in enumerate(zone_manager.get_current_zones()):
        print(f"   [{i}] {zone['name']}")

    # 다시 5개 구역으로 변경
    print("\n5. Changing to 5 zones...")
    test_zones_5 = [
        {"id": "1", "name": "거실", "points": []},
        {"id": "2", "name": "주방", "points": []},
        {"id": "3", "name": "침실", "points": []},
        {"id": "4", "name": "서재", "points": []},
        {"id": "5", "name": "베란다", "points": []}
    ]

    zone_manager.update_zones(
        home_id="test_home_123",
        zones=test_zones_5
    )

    print(f"\n6. Final verification:")
    print(f"   Current zones: {len(zone_manager.get_current_zones())}")
    print(f"   GRU output dim: {gru_model.num_zones}")

    print("\n✅ Test complete!\n")


if __name__ == "__main__":
    test_zone_manager()
