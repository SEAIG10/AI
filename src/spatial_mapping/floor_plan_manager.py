"""
의미론적 매핑 모듈
SLAM 없이 평면도와 구역(zone) 매핑을 관리합니다.
"""

import json
import numpy as np


class FloorPlanManager:
    """
    의미론적 공간 매핑을 관리합니다.
    - 사용자 제공 평면도 로드
    - 의미론적 구역(주방, 거실 등) 정의
    - Webots GPS로부터 실시간 (x, y) 좌표 수신
    - 좌표를 의미론적 레이블로 매핑
    """

    def __init__(self, config_path):
        """
        평면도 관리자를 초기화합니다.

        Args:
            config_path: zones_config.json 파일 경로
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        self.floor_plan_config = self.config['floor_plan']
        self.zones = self.config['zones']

        # 좌표 경계 (Webots 월드 좌표)
        bounds = self.floor_plan_config['real_world_bounds']
        self.x_min = bounds['x_min']
        self.x_max = bounds['x_max']
        self.y_min = bounds['y_min']
        self.y_max = bounds['y_max']

        print(f"FloorPlanManager가 초기화되었습니다.")
        print(f"  로드된 구역 수: {len(self.zones)}개")
        print(f"  월드 경계: ({self.x_min}, {self.y_min}) 에서 ({self.x_max}, {self.y_max}) 까지")

    def point_in_polygon(self, point, polygon):
        """
        광선 투사 알고리즘을 사용하여 점이 폴리곤 내부에 있는지 확인합니다.

        Args:
            point: (x, y) 튜플
            polygon: 폴리곤의 꼭짓점을 정의하는 (x, y) 튜플의 리스트

        Returns:
            bool: 점이 폴리곤 내부에 있으면 True
        """
        x, y = point
        n = len(polygon)
        inside = False

        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def get_zone(self, x, y):
        """
        로봇의 위치를 의미론적 구역 레이블로 매핑합니다.

        Args:
            x: Webots 월드의 X 좌표
            y: Webots 월드의 Y 좌표

        Returns:
            id, label, description을 포함하는 구역 정보 딕셔너리.
            위치가 모든 구역 밖에 있으면 None을 반환합니다.
        """
        for zone in self.zones:
            if zone['type'] == 'polygon':
                if self.point_in_polygon((x, y), zone['coordinates_world']):
                    return {
                        'id': zone['id'],
                        'label': zone['label'],
                        'description': zone.get('description', ''),
                        'coordinates': (x, y)
                    }

        return None

    def get_all_zones(self):
        """정의된 모든 구역의 리스트를 가져옵니다."""
        return [
            {
                'id': z['id'],
                'label': z['label'],
                'description': z.get('description', '')
            }
            for z in self.zones
        ]

    def is_valid_position(self, x, y):
        """위치가 월드 경계 내에 있는지 확인합니다."""
        return (self.x_min <= x <= self.x_max and
                self.y_min <= y <= self.y_max)


def test_spatial_mapping():
    """Webots 없이 공간 매핑을 테스트합니다."""
    print("=" * 60)
    print("공간 매핑 모듈 테스트")
    print("=" * 60)

    # 설정 파일 로드
    manager = FloorPlanManager('../../config/zones_config.json')

    # 테스트 케이스
    test_positions = [
        (1.5, 1.0, "거실 중앙"),
        (-1.5, 1.0, "침실1 중앙"),
        (2.7, -1.5, "침실2 중앙"),
        (-3.0, 0.0, "현관"),
        (0.0, 0.0, "경계 지점"),
        (5.0, 5.0, "범위 밖"),
    ]

    print(f"\n{len(test_positions)}개의 위치 테스트:")
    print("-" * 60)

    for x, y, description in test_positions:
        zone_info = manager.get_zone(x, y)

        if zone_info:
            print(f"  성공: ({x:5.1f}, {y:5.1f}) → {zone_info['label']:8s} ({description})")
        else:
            print(f"  실패: ({x:5.1f}, {y:5.1f}) → 구역 없음   ({description})")

    print("-" * 60)
    print(f"\n모든 구역:")
    for zone in manager.get_all_zones():
        print(f"  - {zone['id']:15s}: {zone['label']:10s} ({zone['description']})")

    print("\n" + "=" * 60)
    print("공간 매핑 테스트 완료!")
    print("=" * 60)


if __name__ == "__main__":
    test_spatial_mapping()
