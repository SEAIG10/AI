"""
Test Sensor Publisher - Send dummy sensor data to WebSocket bridge
This script simulates sensor data to test the WebSocket bridge.
"""

import zmq
import time
import numpy as np

# ZeroMQ 설정
ZMQ_ENDPOINT = "ipc:///tmp/locus_sensors.ipc"


def main():
    print("="*60)
    print("Test Sensor Publisher")
    print("="*60)

    # ZeroMQ Publisher 설정
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.connect(ZMQ_ENDPOINT)
    print(f"Connected to {ZMQ_ENDPOINT}")

    # 초기 대기 (subscriber가 연결될 때까지)
    print("Waiting 1 second for subscribers to connect...")
    time.sleep(1)

    print("\nSending test sensor data...")
    print("Press Ctrl+C to stop\n")

    count = 0

    try:
        while True:
            timestamp = time.time()

            # Visual 데이터 (14차원 - YOLO 객체 감지)
            visual_data = np.random.rand(14).astype(np.float32) * 0.5
            message_visual = {
                'type': 'visual',
                'data': visual_data,
                'timestamp': timestamp,
                'frame_count': count
            }
            socket.send_pyobj(message_visual)

            # Audio 데이터 (17차원 - YAMNet 분류)
            audio_data = np.random.rand(17).astype(np.float32)
            message_audio = {
                'type': 'audio',
                'data': audio_data,
                'timestamp': timestamp,
                'sample_count': count
            }
            socket.send_pyobj(message_audio)

            # Pose 데이터 (51차원 - YOLO Pose)
            pose_data = np.random.rand(51).astype(np.float32)
            message_pose = {
                'type': 'pose',
                'data': pose_data,
                'timestamp': timestamp,
                'frame_count': count
            }
            socket.send_pyobj(message_pose)

            # Synced 데이터 (동기화 완료 신호)
            if count % 5 == 0:  # 5회마다 동기화 신호
                message_synced = {
                    'type': 'synced',
                    'timestamp': timestamp,
                    'sample_count': count
                }
                socket.send_pyobj(message_synced)

            count += 1
            print(f"[{count:04d}] Sent: visual, audio, pose" +
                  (" + synced" if count % 5 == 0 else ""))

            # 1초 대기
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n\nStopping...")

    finally:
        socket.close()
        context.term()
        print("Test publisher stopped!")


if __name__ == "__main__":
    main()
