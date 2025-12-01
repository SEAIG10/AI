"""
실시간  - 컨텍스트 센서 (공간/시간)
공간, 시간 정보를 생성하여 ZeroMQ로 전송합니다.
Pose 정보는 sensor_visual.py의 YOLOv11n-pose에서 전송합니다.
"""

import sys
import os
#TODO: LocusMobileTracker 재연결

# 프로젝트 루트 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import zmq
import time
import numpy as np
from datetime import datetime
from realtime.utils import zone_to_onehot, get_time_features, ZONES
import asyncio
import websockets
import json
import threading

# ZeroMQ 설정
ZMQ_ENDPOINT = "ipc:///tmp/locus_sensors.ipc"


class ContextSensor:
    """
    컨텍스트 센서 (공간, 시간)
    공간, 시간 정보를 생성하여 ZeroMQ로 전송합니다.
    Pose는 sensor_visual.py에서 YOLOv11n-pose로 처리됩니다.
    """

    def __init__(self, default_zone="living_room", enable_location_tracker=False, tracker_ws_uri=None):
        """
        컨텍스트 센서를 초기화합니다.
        Args:
            default_zone: 기본 Zone (GPS 부재 시 수동으로 입력)
            enable_location_tracker: LocationTracker WebSocket 사용 여부
            tracker_ws_uri: LocationTracker 서버 주소 (예: ws://192.168.43.1:8080)
        """
        print("="*60)
        print("Context Sensor (Spatial/Time) Initializing...")
        print("="*60)

        # ZeroMQ Publisher 설정
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.PUB)
        self.zmq_socket.connect(ZMQ_ENDPOINT)
        print(f"ZeroMQ connected to {ZMQ_ENDPOINT}")

        # 현재 Zone (실제 환경에서는 GPS 등으로 판단, 데모에서는 수동 입력)
        self.current_zone = default_zone
        print(f"Default zone set to: {self.current_zone}")

        # LocationTracker WebSocket 설정
        self.enable_location_tracker = enable_location_tracker
        self.tracker_ws_uri = tracker_ws_uri or "ws://192.168.43.1:8080"

        if self.enable_location_tracker:
            print(f"\n[LocationTracker] Enabled")
            print(f"[LocationTracker] Connecting to: {self.tracker_ws_uri}")
            # 백그라운드 스레드로 WebSocket 리스너 시작
            self.ws_thread = threading.Thread(target=self._start_ws_listener, daemon=True)
            self.ws_thread.start()
        else:
            print(f"\n[LocationTracker] Disabled (manual zone control)")

        print("\nContext Sensor ready!\n")

    def _start_ws_listener(self):
        """백그라운드에서 WebSocket 리스너 실행"""
        asyncio.run(self._listen_zone_updates())

    async def _listen_zone_updates(self):
        """LocationTracker로부터 zone 업데이트 수신"""
        try:
            async with websockets.connect(self.tracker_ws_uri) as ws:
                # viewer로 식별
                await ws.send(json.dumps({
                    'type': 'identify',
                    'clientType': 'viewer'
                }))

                print(f"[LocationTracker] Connected successfully")

                while True:
                    data = await ws.recv()
                    message = json.loads(data)

                    if message['type'] == 'location_update':
                        new_zone = message['data'].get('zone', 'unknown')

                        if new_zone != self.current_zone and new_zone != 'unknown':
                            print(f"\n[LocationTracker] Zone update: {self.current_zone} -> {new_zone}\n")
                            self.current_zone = new_zone

        except Exception as e:
            print(f"[LocationTracker] Connection failed: {e}")
            print(f"[LocationTracker] Using default zone: {self.current_zone}")

    def set_zone(self, zone_name):
        """
        현재 Zone을 설정합니다.

        Args:
            zone_name: Zone 이름
        """
        if zone_name not in ZONES:
            print(f"Warning: Invalid zone '{zone_name}', keeping '{self.current_zone}'")
            return

        self.current_zone = zone_name
        print(f"Zone changed to: {self.current_zone}")

    def run(self, interval=1.0):
        """
        센서의 메인 루프를 실행합니다.

        Args:
            interval: 데이터 전송 주기 (초)
        """
        print("Starting Context Sensor loop...")
        print(f"  - Interval: {interval}s")
        print(f"  - Current zone: {self.current_zone}")
        print("  - Press Ctrl+C to quit")

        if not self.enable_location_tracker:
            print("\n[Manual Mode]")
            print("  - Zone can be changed using set_zone() method")
            print("  - Available zones:", ", ".join(ZONES))
        else:
            print("\n[LocationTracker Mode]")
            print("  - Zone will update automatically from iPhone ARKit")

        print()

        sample_count = 0

        try:
            while True:
                # 모든 컨텍스트 데이터는 동일한 시점을 기준으로 측정
                start_timestamp = time.time()

                # 공간 정보 (7차원)
                spatial_vec = zone_to_onehot(self.current_zone)

                # 시간 정보 (10차원)
                now = datetime.now()
                time_vec = get_time_features(now)

                # ZeroMQ 전송 - 공간 정보 (측정 시작 시점의 타임스탬프 사용)
                message_spatial = {
                    'type': 'spatial',
                    'data': spatial_vec,
                    'timestamp': start_timestamp,
                    'sample_count': sample_count,
                    'zone_name': self.current_zone
                }
                self.zmq_socket.send_pyobj(message_spatial)

                # ZeroMQ 전송 - 시간 정보 (동일 타임스탬프)
                message_time = {
                    'type': 'time',
                    'data': time_vec,
                    'timestamp': start_timestamp,
                    'sample_count': sample_count,
                    'datetime': now.isoformat()
                }
                self.zmq_socket.send_pyobj(message_time)

                # 로그 출력
                print(f"[{sample_count:04d}] Context → ZMQ: "
                      f"zone={self.current_zone}, "
                      f"hour={now.hour:02d}:{now.minute:02d}")

                sample_count += 1
                time.sleep(interval)

        except KeyboardInterrupt:
            print("\nKeyboard interrupt, stopping...")

        finally:
            self.cleanup()

    def cleanup(self):
        """사용한 리소스를 정리합니다."""
        print("\nCleaning up Context Sensor...")
        self.zmq_socket.close()
        self.zmq_context.term()
        print("Context Sensor stopped!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Context Sensor (Spatial/Time)")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Sensing interval in seconds (default: 1.0)")
    parser.add_argument("--zone", type=str, default="living_room",
                        choices=ZONES,
                        help=f"Initial zone (default: living_room)")
    parser.add_argument("--enable-tracker", action="store_true",
                        help="Enable LocationTracker WebSocket integration")
    parser.add_argument("--tracker-uri", type=str, default="ws://192.168.43.1:8080",
                        help="LocationTracker WebSocket URI (default: ws://192.168.43.1:8080)")

    args = parser.parse_args()

    # 센서 시작
    sensor = ContextSensor(
        default_zone=args.zone,
        enable_location_tracker=args.enable_tracker,
        tracker_ws_uri=args.tracker_uri
    )
    sensor.run(interval=args.interval)
