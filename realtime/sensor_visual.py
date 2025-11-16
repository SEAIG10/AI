"""
실시간 데모 - 비주얼 센서 (YOLO)
웹캠으로 객체를 감지한 후 ZeroMQ로 전송합니다.
"""

import sys
import os

# 프로젝트 루트 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import cv2
from ultralytics import YOLO
import zmq
import time
import numpy as np
from realtime.utils import yolo_results_to_14dim, YOLO_CLASSES

# ZeroMQ 설정
ZMQ_ENDPOINT = "ipc:///tmp/locus_sensors.ipc"

# YOLO 모델 경로
YOLO_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'yolo', 'best.pt')


class VisualSensor:
    """
    YOLO 기반 비주얼 센서
    웹캠에서 프레임을 읽고 YOLO로 객체를 감지한 후, 결과를 ZeroMQ로 전송합니다.
    """

    def __init__(self, camera_id=0):
        """
        비주얼 센서를 초기화합니다.

        Args:
            camera_id: 사용할 웹캠 ID (기본값: 0)
        """
        print("="*60)
        print("Visual Sensor (YOLO) Initializing...")
        print("="*60)

        # ZeroMQ Publisher 설정
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.PUB)
        self.zmq_socket.connect(ZMQ_ENDPOINT)
        print(f"ZeroMQ connected to {ZMQ_ENDPOINT}")

        # YOLO 모델 로드
        print(f"Loading YOLO model from: {YOLO_MODEL_PATH}")
        self.yolo_model = YOLO(YOLO_MODEL_PATH)
        print("YOLO model loaded!")

        # 웹캠 초기화
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_id}")
        print(f"Camera {camera_id} opened!")

        print("\nVisual Sensor ready!\n")

    def run(self, interval=1.0, show_window=False):
        """
        센서의 메인 루프를 실행합니다.

        Args:
            interval: 데이터 전송 주기 (초)
            show_window: 웹캠 화면을 창으로 표시할지 여부
        """
        print("Starting Visual Sensor loop...")
        print(f"  - Interval: {interval}s")
        print(f"  - Show window: {show_window}")
        print("  - Press 'q' to quit\n")

        frame_count = 0

        try:
            while True:
                # 측정 시작 시점의 타임스탬프
                start_timestamp = time.time()

                # 프레임 읽기
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame, retrying...")
                    time.sleep(0.1)
                    continue

                # YOLO 추론
                results = self.yolo_model(frame, verbose=False)

                # 14차원 벡터 생성
                visual_vec = yolo_results_to_14dim(results)

                # 감지된 객체 확인
                detected_indices = np.where(visual_vec > 0)[0]
                detected_objects = [YOLO_CLASSES[i] for i in detected_indices]

                # ZeroMQ로 전송 (측정 시작 시점의 타임스탬프 사용)
                message = {
                    'type': 'visual',
                    'data': visual_vec,
                    'timestamp': start_timestamp,
                    'frame_count': frame_count
                }
                self.zmq_socket.send_pyobj(message)

                # 로그 출력
                frame_count += 1
                print(f"[{frame_count:04d}] Visual → ZMQ: {len(detected_objects)} objects detected", end="")
                if detected_objects:
                    print(f" ({', '.join(detected_objects[:3])}{'...' if len(detected_objects) > 3 else ''})")
                else:
                    print(" (none)")

                # 화면 표시 (옵션)
                if show_window:
                    annotated = results[0].plot()
                    cv2.imshow("YOLO Visual Sensor", annotated)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\nUser pressed 'q', stopping...")
                        break

                # 다음 주기까지 대기
                time.sleep(interval)

        except KeyboardInterrupt:
            print("\nKeyboard interrupt, stopping...")

        finally:
            self.cleanup()

    def cleanup(self):
        """사용한 리소스를 정리합니다."""
        print("\nCleaning up Visual Sensor...")
        self.cap.release()
        cv2.destroyAllWindows()
        self.zmq_socket.close()
        self.zmq_context.term()
        print("Visual Sensor stopped!")


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
