"""
Realtime Demo - Visual Sensor (YOLO)
웹캠으로 객체 감지 후 MQTT로 전송
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import cv2
from ultralytics import YOLO
import paho.mqtt.client as mqtt
import pickle
import time
import numpy as np
from realtime.utils import yolo_results_to_14dim, YOLO_CLASSES

# MQTT 설정
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "sensor/visual"

# YOLO 모델 경로
YOLO_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'yolo', 'best.pt')


class VisualSensor:
    """
    YOLO 기반 Visual Sensor
    웹캠에서 프레임을 읽고 YOLO로 객체 감지 후 MQTT로 전송
    """

    def __init__(self, camera_id=0):
        """
        Initialize Visual Sensor

        Args:
            camera_id: 웹캠 ID (기본값: 0)
        """
        print("="*60)
        print("🎥 Visual Sensor (YOLO) Initializing...")
        print("="*60)

        # MQTT 클라이언트
        self.mqtt_client = mqtt.Client("visual_sensor")
        self.mqtt_client.connect(MQTT_BROKER, MQTT_PORT)
        print(f"✓ MQTT connected to {MQTT_BROKER}:{MQTT_PORT}")

        # YOLO 모델 로드
        print(f"Loading YOLO model from: {YOLO_MODEL_PATH}")
        self.yolo_model = YOLO(YOLO_MODEL_PATH)
        print("✓ YOLO model loaded!")

        # 웹캠 초기화
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_id}")
        print(f"✓ Camera {camera_id} opened!")

        print("\n✅ Visual Sensor ready!\n")

    def run(self, interval=1.0, show_window=False):
        """
        센서 실행 (메인 루프)

        Args:
            interval: 전송 주기 (초)
            show_window: 웹캠 화면 표시 여부
        """
        print("🚀 Starting Visual Sensor loop...")
        print(f"  - Interval: {interval}s")
        print(f"  - Show window: {show_window}")
        print("  - Press 'q' to quit\n")

        frame_count = 0

        try:
            while True:
                # 프레임 읽기
                ret, frame = self.cap.read()
                if not ret:
                    print("⚠ Failed to read frame, retrying...")
                    time.sleep(0.1)
                    continue

                # YOLO 추론
                results = self.yolo_model(frame, verbose=False)

                # 14-dim 벡터 생성
                visual_vec = yolo_results_to_14dim(results)

                # 감지된 객체 확인
                detected_indices = np.where(visual_vec > 0)[0]
                detected_objects = [YOLO_CLASSES[i] for i in detected_indices]

                # MQTT 전송
                payload = pickle.dumps({
                    'visual': visual_vec,
                    'timestamp': time.time(),
                    'frame_count': frame_count
                })
                self.mqtt_client.publish(MQTT_TOPIC, payload)

                # 로그 출력
                frame_count += 1
                print(f"[{frame_count:04d}] 📹 Visual → MQTT: {len(detected_objects)} objects detected", end="")
                if detected_objects:
                    print(f" ({', '.join(detected_objects[:3])}{'...' if len(detected_objects) > 3 else ''})")
                else:
                    print(" (none)")

                # 화면 표시 (옵션)
                if show_window:
                    annotated = results[0].plot()
                    cv2.imshow("YOLO Visual Sensor", annotated)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\n⚠ User pressed 'q', stopping...")
                        break

                # 대기
                time.sleep(interval)

        except KeyboardInterrupt:
            print("\n⚠ Keyboard interrupt, stopping...")

        finally:
            self.cleanup()

    def cleanup(self):
        """리소스 정리"""
        print("\n🧹 Cleaning up Visual Sensor...")
        self.cap.release()
        cv2.destroyAllWindows()
        self.mqtt_client.disconnect()
        print("✓ Visual Sensor stopped!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visual Sensor (YOLO)")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Sensing interval in seconds (default: 1.0)")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera ID (default: 0)")
    parser.add_argument("--show", action="store_true",
                        help="Show webcam window")

    args = parser.parse_args()

    # 센서 시작
    sensor = VisualSensor(camera_id=args.camera)
    sensor.run(interval=args.interval, show_window=args.show)
