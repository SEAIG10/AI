"""
실시간 데모 - 비주얼 센서 (YOLO)
웹캠으로 객체를 감지한 후 ZeroMQ로 전송합니다.
YOLOv11n: 객체 감지 (14 classes)
YOLOv11n-pose: 사람 자세 추정 (17 keypoints × 3 = 51 dims)
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
from realtime.utils import yolo_results_to_14dim, extract_pose_keypoints, YOLO_CLASSES

# ZeroMQ 설정
ZMQ_ENDPOINT = "ipc:///tmp/locus_sensors.ipc"

# YOLO 모델 경로
YOLO_DETECT_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'yolo', 'best.pt')
YOLO_POSE_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'yolo', 'yolo11n-pose.pt')


class VisualSensor:
    """
    YOLO 기반 비주얼 센서
    - YOLOv11n: 객체 감지 (가구, 오염물 등)
    - YOLOv11n-pose: 사람 자세 추정 (17 keypoints)
    웹캠에서 프레임을 읽고 YOLO로 감지한 후, 결과를 ZeroMQ로 전송합니다.
    """

    def __init__(self, camera_id=0, enable_pose=True):
        """
        비주얼 센서를 초기화합니다.

        Args:
            camera_id: 사용할 웹캠 ID (기본값: 0)
            enable_pose: YOLOv11n-pose 활성화 여부 (기본값: True)
        """
        print("="*60)
        print("Visual Sensor (YOLOv11n + YOLOv11n-pose) Initializing...")
        print("="*60)

        self.enable_pose = enable_pose

        # ZeroMQ Publisher 설정
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.PUB)
        self.zmq_socket.connect(ZMQ_ENDPOINT)
        print(f"ZeroMQ connected to {ZMQ_ENDPOINT}")

        # YOLOv11n 객체 감지 모델 로드
        print(f"\nLoading YOLOv11n (object detection) from: {YOLO_DETECT_PATH}")
        self.yolo_detect = YOLO(YOLO_DETECT_PATH)
        print("  ✓ YOLOv11n loaded! (14 classes: furniture & pollution)")

        # YOLOv11n-pose 모델 로드
        if self.enable_pose:
            print(f"\nLoading YOLOv11n-pose (pose estimation)...")
            self.yolo_pose = YOLO(YOLO_POSE_PATH)
            print("  ✓ YOLOv11n-pose loaded! (17 keypoints)")
        else:
            self.yolo_pose = None
            print("\n  ⚠️  Pose estimation disabled")

        # 웹캠 초기화
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_id}")
        print(f"\nCamera {camera_id} opened!")

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

                # YOLOv11n 객체 감지
                detect_results = self.yolo_detect(frame, verbose=False)

                # 14차원 벡터 생성
                visual_vec = yolo_results_to_14dim(detect_results)

                # 감지된 객체 확인
                detected_indices = np.where(visual_vec > 0)[0]
                detected_objects = [YOLO_CLASSES[i] for i in detected_indices]

                # ZeroMQ로 visual 데이터 전송
                message_visual = {
                    'type': 'visual',
                    'data': visual_vec,
                    'timestamp': start_timestamp,
                    'frame_count': frame_count
                }
                self.zmq_socket.send_pyobj(message_visual)

                # YOLOv11n-pose로 자세 추정
                pose_vec = np.zeros(51, dtype=np.float32)
                if self.enable_pose and self.yolo_pose is not None:
                    pose_results = self.yolo_pose(frame, verbose=False)
                    pose_vec = extract_pose_keypoints(pose_results)

                # ZeroMQ로 pose 데이터 전송
                message_pose = {
                    'type': 'pose',
                    'data': pose_vec,
                    'timestamp': start_timestamp,
                    'frame_count': frame_count
                }
                self.zmq_socket.send_pyobj(message_pose)

                # 로그 출력
                frame_count += 1
                pose_detected = np.any(pose_vec > 0)
                print(f"[{frame_count:04d}] Visual → ZMQ: {len(detected_objects)} objects", end="")
                if detected_objects:
                    print(f" ({', '.join(detected_objects[:2])})", end="")
                print(f" | Pose: {'✓' if pose_detected else '✗'}")

                # 화면 표시 (옵션)
                if show_window:
                    # 객체 감지 결과 표시
                    annotated = detect_results[0].plot()

                    # Pose가 활성화되어 있고 사람이 감지되면 keypoints도 표시
                    if self.enable_pose and pose_detected:
                        annotated = pose_results[0].plot(img=annotated)

                    cv2.imshow("YOLO Visual Sensor (Object + Pose)", annotated)

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

    parser = argparse.ArgumentParser(description="Visual Sensor (YOLOv11n + YOLOv11n-pose)")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Sensing interval in seconds (default: 1.0)")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera ID (default: 0)")
    parser.add_argument("--show", action="store_true",
                        help="Show webcam window")
    parser.add_argument("--no-pose", action="store_true",
                        help="Disable pose estimation (YOLOv11n only)")

    args = parser.parse_args()

    # 센서 시작
    sensor = VisualSensor(camera_id=args.camera, enable_pose=not args.no_pose)
    sensor.run(interval=args.interval, show_window=args.show)
