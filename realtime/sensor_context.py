"""
Realtime Demo - Context Sensor (Spatial/Time/Pose)
공간, 시간, Pose 정보를 생성하여 ZeroMQ로 전송
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import zmq
import time
import numpy as np
from datetime import datetime
from realtime.utils import zone_to_onehot, get_time_features, ZONES

# ZeroMQ 설정
ZMQ_ENDPOINT = "ipc:///tmp/locus_sensors.ipc"


class ContextSensor:
    """
    Context Sensor (Spatial, Time, Pose)
    공간 정보, 시간 정보, Pose 정보를 생성하여 ZeroMQ로 전송
    """

    def __init__(self, default_zone="living_room"):
        """
        Initialize Context Sensor

        Args:
            default_zone: 기본 Zone (GPS가 없으면 수동 입력)
        """
        print("="*60)
        print("📍 Context Sensor (Spatial/Time/Pose) Initializing...")
        print("="*60)

        # ZeroMQ Publisher 설정
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.PUB)
        self.zmq_socket.connect(ZMQ_ENDPOINT)
        print(f"✓ ZeroMQ connected to {ZMQ_ENDPOINT}")

        # 현재 Zone (실제로는 GPS로 판단, 데모에서는 수동 입력)
        self.current_zone = default_zone
        print(f"✓ Default zone set to: {self.current_zone}")

        print("\n✅ Context Sensor ready!\n")

    def set_zone(self, zone_name):
        """
        현재 Zone 설정

        Args:
            zone_name: Zone 이름
        """
        if zone_name not in ZONES:
            print(f"⚠ Warning: Invalid zone '{zone_name}', keeping '{self.current_zone}'")
            return

        self.current_zone = zone_name
        print(f"✓ Zone changed to: {self.current_zone}")

    def run(self, interval=1.0):
        """
        센서 실행 (메인 루프)

        Args:
            interval: 전송 주기 (초)
        """
        print("🚀 Starting Context Sensor loop...")
        print(f"  - Interval: {interval}s")
        print(f"  - Current zone: {self.current_zone}")
        print("  - Press Ctrl+C to quit")
        print("\nCommands (type during running):")
        print("  - Type zone name to change (e.g., 'kitchen', 'bedroom_1')")
        print("  - Available zones:", ", ".join(ZONES))
        print()

        sample_count = 0

        try:
            while True:
                # Spatial (7-dim)
                spatial_vec = zone_to_onehot(self.current_zone)

                # Time (10-dim)
                now = datetime.now()
                time_vec = get_time_features(now)

                # Pose (51-dim) - Mock for demo
                # 실제로는 sensor_visual에서 YOLO-Pose로 추출
                pose_vec = np.zeros(51, dtype=np.float32)

                # ZeroMQ 전송 - Spatial
                message_spatial = {
                    'type': 'spatial',
                    'data': spatial_vec,
                    'timestamp': time.time(),
                    'sample_count': sample_count,
                    'zone_name': self.current_zone
                }
                self.zmq_socket.send_pyobj(message_spatial)

                # ZeroMQ 전송 - Time
                message_time = {
                    'type': 'time',
                    'data': time_vec,
                    'timestamp': time.time(),
                    'sample_count': sample_count,
                    'datetime': now.isoformat()
                }
                self.zmq_socket.send_pyobj(message_time)

                # ZeroMQ 전송 - Pose
                message_pose = {
                    'type': 'pose',
                    'data': pose_vec,
                    'timestamp': time.time(),
                    'sample_count': sample_count
                }
                self.zmq_socket.send_pyobj(message_pose)

                # 로그 출력
                print(f"[{sample_count:04d}] 📍 Context → ZMQ: "
                      f"zone={self.current_zone}, "
                      f"hour={now.hour:02d}:{now.minute:02d}, "
                      f"pose=mock")

                sample_count += 1
                time.sleep(interval)

        except KeyboardInterrupt:
            print("\n⚠ Keyboard interrupt, stopping...")

        finally:
            self.cleanup()

    def cleanup(self):
        """리소스 정리"""
        print("\n🧹 Cleaning up Context Sensor...")
        self.zmq_socket.close()
        self.zmq_context.term()
        print("✓ Context Sensor stopped!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Context Sensor (Spatial/Time/Pose)")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Sensing interval in seconds (default: 1.0)")
    parser.add_argument("--zone", type=str, default="living_room",
                        choices=ZONES,
                        help=f"Initial zone (default: living_room)")

    args = parser.parse_args()

    # 센서 시작
    sensor = ContextSensor(default_zone=args.zone)
    sensor.run(interval=args.interval)
