"""
FR1: Semantic Mapping Module for LOCUS
Manages floor plan and zone mapping without SLAM
"""

import json
import numpy as np


class FloorPlanManager:
    """
    FR1: Semantic Spatial Mapping
    - FR1.1: Load user-provided floor plan
    - FR1.2: Define semantic zones (kitchen, living room, etc.)
    - FR1.3: Receive real-time (x, y) from Webots GPS
    - FR1.4: Map coordinates to semantic labels
    """

    def __init__(self, config_path):
        """
        Initialize floor plan manager

        Args:
            config_path: Path to zones_config.json
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        self.floor_plan_config = self.config['floor_plan']
        self.zones = self.config['zones']

        # Coordinate bounds (Webots world coordinates)
        bounds = self.floor_plan_config['real_world_bounds']
        self.x_min = bounds['x_min']
        self.x_max = bounds['x_max']
        self.y_min = bounds['y_min']
        self.y_max = bounds['y_max']

        print(f"✓ FloorPlanManager initialized")
        print(f"  Loaded {len(self.zones)} zones")
        print(f"  World bounds: ({self.x_min}, {self.y_min}) to ({self.x_max}, {self.y_max})")

    def point_in_polygon(self, point, polygon):
        """
        FR1.4: Check if point is inside polygon using ray casting algorithm

        Args:
            point: (x, y) tuple
            polygon: List of (x, y) tuples defining polygon vertices

        Returns:
            bool: True if point is inside polygon
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
        FR1.4: Map robot position to semantic zone label

        Args:
            x: X coordinate in Webots world
            y: Y coordinate in Webots world

        Returns:
            dict: Zone info including id, label, description
                  Returns None if position is outside all zones
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
        """Get list of all defined zones"""
        return [
            {
                'id': z['id'],
                'label': z['label'],
                'description': z.get('description', '')
            }
            for z in self.zones
        ]

    def is_valid_position(self, x, y):
        """Check if position is within world bounds"""
        return (self.x_min <= x <= self.x_max and
                self.y_min <= y <= self.y_max)


def test_spatial_mapping():
    """Test spatial mapping without Webots"""
    print("=" * 60)
    print("Testing FR1: Spatial Mapping Module")
    print("=" * 60)

    # Load configuration
    manager = FloorPlanManager('../../config/zones_config.json')

    # Test cases
    test_positions = [
        (1.5, 1.0, "거실 중앙"),
        (-1.5, 1.0, "침실1 중앙"),
        (2.7, -1.5, "침실2 중앙"),
        (-3.0, 0.0, "현관"),
        (0.0, 0.0, "경계 지점"),
        (5.0, 5.0, "범위 밖"),
    ]

    print(f"\nTesting {len(test_positions)} positions:")
    print("-" * 60)

    for x, y, description in test_positions:
        zone_info = manager.get_zone(x, y)

        if zone_info:
            print(f"✓ ({x:5.1f}, {y:5.1f}) → {zone_info['label']:8s} ({description})")
        else:
            print(f"✗ ({x:5.1f}, {y:5.1f}) → No zone   ({description})")

    print("-" * 60)
    print(f"\nAll zones:")
    for zone in manager.get_all_zones():
        print(f"  - {zone['id']:15s}: {zone['label']:10s} ({zone['description']})")

    print("\n" + "=" * 60)
    print("FR1 Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_spatial_mapping()
