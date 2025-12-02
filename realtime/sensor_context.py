"""
실시간 데모 - 컨텍스트 센서 (공간/시간)
공간, 시간 정보를 생성하여 ZeroMQ로 전송합니다.
Zone 정보는 LocusBackend의 MQTT 메시지로부터 받아옵니다.
Pose 정보는 sensor_visual.py의 YOLOv11n-pose에서 전송합니다.
"""

import sys
import os

# 프로젝트 루트 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import zmq
import time
import numpy as np
from datetime import datetime
from realtime.utils import zone_to_onehot, get_time_features, ZONES
import json
import paho.mqtt.client as mqtt

# ZeroMQ 설정
ZMQ_ENDPOINT = "ipc:///tmp/locus_sensors.ipc"


class ContextSensor:
    """
    컨텍스트 센서 (공간, 시간)
    공간, 시간 정보를 생성하여 ZeroMQ로 전송합니다.
    Pose는 sensor_visual.py에서 YOLOv11n-pose로 처리됩니다.
    """

    def __init__(self, home_id, default_zone="living_room", mqtt_broker="43.200.178.189", mqtt_port=1883):
        """
        컨텍스트 센서를 초기화합니다.
        Args:
            home_id: Home ID (필수) - LocusBackend에서 사용하는 집 ID
            default_zone: 기본 Zone (MQTT 연결 전까지 사용)
            mqtt_broker: MQTT 브로커 주소 (기본값: mqtt.eclipseprojects.io)
            mqtt_port: MQTT 브로커 포트 (기본값: 1883)
        """
        print("="*60)
        print("Context Sensor (Spatial/Time) Initializing...")
        print("="*60)

        # ZeroMQ Publisher 설정
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.PUB)
        self.zmq_socket.connect(ZMQ_ENDPOINT)
        print(f"ZeroMQ connected to {ZMQ_ENDPOINT}")

        # 현재 Zone (MQTT로부터 업데이트됨)
        self.current_zone = default_zone
        print(f"Default zone set to: {self.current_zone}")

        # MQTT 클라이언트 설정 (LocusBackend 연동)
        self.mqtt_broker = mqtt_broker
        self.mqtt_port = mqtt_port
        self.home_id = home_id
        self.mqtt_client = None

        print(f"\n[MQTT] Connecting to LocusBackend...")
        print(f"[MQTT] Broker: {self.mqtt_broker}:{self.mqtt_port}")
        print(f"[MQTT] Topic: home/{self.home_id}/robot/location")
        self._setup_mqtt_client()

        print("\nContext Sensor ready!\n")

    def _setup_mqtt_client(self):
        """MQTT 클라이언트를 설정하고 연결합니다."""
        self.mqtt_client = mqtt.Client(client_id=f"context_sensor_{int(time.time())}")
        self.mqtt_client.on_connect = self._on_mqtt_connect
        self.mqtt_client.on_message = self._on_mqtt_message
        self.mqtt_client.on_disconnect = self._on_mqtt_disconnect

        try:
            self.mqtt_client.connect(self.mqtt_broker, self.mqtt_port, 60)
            self.mqtt_client.loop_start()
            print(f"[MQTT] Connection initiated...")
        except Exception as e:
            print(f"[MQTT] Connection failed: {e}")
            print(f"[MQTT] Using default zone: {self.current_zone}")

    def _on_mqtt_connect(self, client, userdata, flags, rc):
        """MQTT 연결 성공 시 호출되는 콜백"""
        if rc == 0:
            print(f"[MQTT] Connected successfully!")
            topic = f"home/{self.home_id}/robot/location"
            client.subscribe(topic)
            print(f"[MQTT] Subscribed to: {topic}")
        else:
            print(f"[MQTT] Connection failed with code {rc}")

    def _on_mqtt_message(self, client, userdata, msg):
        """MQTT 메시지 수신 시 호출되는 콜백"""
        try:
            payload = json.loads(msg.payload.decode())
            new_zone = payload.get('zone')

            # zone이 null이 아니고, 유효한 zone이며, 현재 zone과 다를 때만 업데이트
            if new_zone and new_zone != 'null' and new_zone in ZONES and new_zone != self.current_zone:
                print(f"\n[MQTT] Zone update: {self.current_zone} -> {new_zone}")
                print(f"[MQTT] Position: x={payload.get('x'):.2f}, z={payload.get('z'):.2f}\n")
                self.current_zone = new_zone
        except Exception as e:
            print(f"[MQTT] Error parsing message: {e}")

    def _on_mqtt_disconnect(self, client, userdata, rc):
        """MQTT 연결 해제 시 호출되는 콜백"""
        if rc != 0:
            print(f"[MQTT] Unexpected disconnect (code {rc}), attempting reconnect...")

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

        print("\n[MQTT Mode]")
        print(f"  - Zone updates automatically from LocusBackend")
        print(f"  - Listening to: home/{self.home_id}/robot/location")
        print(f"  - Manual override available via set_zone() method")
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

        # MQTT 클라이언트 정리
        if self.mqtt_client:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
            print("MQTT client disconnected")

        # ZeroMQ 정리
        self.zmq_socket.close()
        self.zmq_context.term()
        print("Context Sensor stopped!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Context Sensor (Spatial/Time) with MQTT")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Sensing interval in seconds (default: 1.0)")
    parser.add_argument("--zone", type=str, default="living_room",
                        choices=ZONES,
                        help=f"Initial zone before MQTT connection (default: living_room)")
    parser.add_argument("--mqtt-broker", type=str, default="43.200.178.189",
                        help="MQTT broker address (default: 43.200.178.189)")
    parser.add_argument("--mqtt-port", type=int, default=1883,
                        help="MQTT broker port (default: 1883)")
    parser.add_argument("--home-id", type=int, required=True,
                        help="Home ID for MQTT topic (required)")

    args = parser.parse_args()

    # 센서 시작
    sensor = ContextSensor(
        home_id=args.home_id,
        default_zone=args.zone,
        mqtt_broker=args.mqtt_broker,
        mqtt_port=args.mqtt_port
    )
    sensor.run(interval=args.interval)
