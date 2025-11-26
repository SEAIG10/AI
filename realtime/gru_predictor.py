"""
실시간 데모 - GRU 예측기
ZeroMQ로 센서 데이터를 수집 후 GRU 모델로 오염도를 예측합니다.
"""

import sys
import os

# 프로젝트 루트 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import zmq
import time
import numpy as np
import tensorflow as tf
from collections import deque, defaultdict

# 내부 모듈 임포트
from src.context_fusion.attention_context_encoder import create_attention_encoder
from src.model.gru_model import FedPerGRUModel
from realtime.utils import print_prediction_result, ZONES

# ZeroMQ 설정
ZMQ_ENDPOINT = "ipc:///tmp/locus_sensors.ipc"

# 모델 경로
GRU_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'gru', 'gru_model.keras')

# 컨텍스트 버퍼 설정
CONTEXT_BUFFER_SIZE = 30  # 30 타임스텝

# ROS ApproximateTimeSynchronizer 방식의 동기화 설정
QUEUE_SIZE = 10  # 각 센서별로 저장할 메시지 수
SLOP = 0.5  # 허용 오차 (초) - 센서 간 최대 시간 차이


class GRUPredictor:
    """
    GRU Predictor
    ZeroMQ로 센서 데이터를 수신하여 AttentionContextEncoder를 거친 후, GRU 모델로 예측을 수행합니다.
    """

    def __init__(self):
        """GRU 예측기를 초기화합니다."""
        print("="*60)
        print("GRU Predictor Initializing...")
        print("="*60)

        # ZeroMQ Subscriber 설정 (BIND - 구독자가 바인드하고, 발행자가 연결)
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.SUB)
        self.zmq_socket.bind(ZMQ_ENDPOINT)
        self.zmq_socket.setsockopt_string(zmq.SUBSCRIBE, "")  # 모든 메시지 구독
        print(f"ZeroMQ bound to {ZMQ_ENDPOINT}")
        print("Subscribed to all sensor messages")

        # 모델 로드
        print("\nLoading models...")
        print("  1. AttentionContextEncoder...")
        self.attention_encoder = create_attention_encoder(
            visual_dim=14,
            audio_dim=17,
            pose_dim=51,
            spatial_dim=4,  # 4 zones: balcony, bedroom, kitchen, living_room
            time_dim=10,
            context_dim=160
        )
        print("     AttentionContextEncoder loaded!")

        print(f"  2. GRU Model from {GRU_MODEL_PATH}...")
        self.gru_model = FedPerGRUModel(num_zones=4, context_dim=160)  # 4 zones
        self.gru_model.load(GRU_MODEL_PATH)
        print("     GRU Model loaded!")

        # ROS ApproximateTimeSynchronizer 방식: 센서별 큐
        self.sensor_queues = {
            'visual': deque(maxlen=QUEUE_SIZE),
            'audio': deque(maxlen=QUEUE_SIZE),
            'pose': deque(maxlen=QUEUE_SIZE),
            'spatial': deque(maxlen=QUEUE_SIZE),
            'time': deque(maxlen=QUEUE_SIZE)
        }

        # 컨텍스트 버퍼 (30 타임스텝)
        self.context_buffer = deque(maxlen=CONTEXT_BUFFER_SIZE)

        # 통계 정보
        self.timestep_count = 0
        self.prediction_count = 0
        self.sync_dropped = 0  # 동기화 실패로 폐기된 데이터 수

        print("\nGRU Predictor ready!\n")

    def receive_messages(self):
        """
        ZeroMQ 메시지를 ROS ApproximateTimeSynchronizer 방식으로 수신합니다.
        """
        try:
            # 논블로킹 방식으로 타임아웃과 함께 메시지 수신
            if self.zmq_socket.poll(timeout=100):  # 100ms 타임아웃
                message = self.zmq_socket.recv_pyobj()

                # 메시지에서 데이터 추출
                sensor_type = message.get('type')
                timestamp = message.get('timestamp')
                data = message.get('data')

                if sensor_type is None or timestamp is None or data is None:
                    return

                # 해당 센서의 큐에 메시지 추가
                if sensor_type in self.sensor_queues:
                    self.sensor_queues[sensor_type].append({
                        'timestamp': timestamp,
                        'data': data
                    })

                # 동기화 시도
                self.try_synchronize()

        except Exception as e:
            print(f"Error in receive_messages: {e}")

    def try_synchronize(self):
        """
        ROS ApproximateTimeSynchronizer 알고리즘을 사용하여 동기화를 시도합니다.
        모든 큐에 메시지가 있을 경우에만 동기화를 진행합니다.
        """
        # 모든 큐에 최소 1개 이상의 메시지가 있는지 확인
        if not all(len(q) > 0 for q in self.sensor_queues.values()):
            return

        # 1. 피벗(pivot) 탐색: 모든 큐의 첫 번째 메시지 중 가장 최신 타임스탬프
        pivot = max(q[0]['timestamp'] for q in self.sensor_queues.values())

        # 2. 각 큐에서 피벗에 가장 가까운 메시지 탐색
        candidates = {}
        for sensor_type, queue in self.sensor_queues.items():
            # 큐에서 피벗에 가장 가까운 메시지를 찾음
            best_msg = min(queue, key=lambda msg: abs(msg['timestamp'] - pivot))
            candidates[sensor_type] = best_msg

        # 3. 모든 후보 메시지가 허용 오차(slop) 이내에 있는지 확인
        timestamps = [msg['timestamp'] for msg in candidates.values()]
        time_span = max(timestamps) - min(timestamps)

        if time_span <= SLOP:
            # 동기화 성공: 매칭된 메시지 추출
            sensor_data = {sensor_type: msg['data']
                          for sensor_type, msg in candidates.items()}

            # 사용된 메시지를 큐에서 제거
            for sensor_type, matched_msg in candidates.items():
                queue = self.sensor_queues[sensor_type]
                # 타임스탬프를 기준으로 정확히 일치하는 메시지를 제거
                self.sensor_queues[sensor_type] = deque(
                    (msg for msg in queue if msg['timestamp'] != matched_msg['timestamp']),
                    maxlen=QUEUE_SIZE
                )

            # 컨텍스트 생성
            avg_timestamp = sum(timestamps) / len(timestamps)
            self.process_context(sensor_data, avg_timestamp)
        else:
            # 동기화 실패: 가장 오래된 메시지를 큐에서 제거
            oldest_sensor = min(self.sensor_queues.items(),
                               key=lambda x: x[1][0]['timestamp'] if len(x[1]) > 0 else float('inf'))
            if len(oldest_sensor[1]) > 0:
                oldest_sensor[1].popleft()
                self.sync_dropped += 1

    def process_context(self, sensor_data, timestamp_bucket):
        """
        동기화된 센서 데이터로 컨텍스트를 생성합니다.
        AttentionContextEncoder를 사용하여 160차원 벡터를 생성 후 버퍼에 추가합니다.

        Args:
            sensor_data: {'visual': data, 'audio': data, ...} 형식의 센서 데이터
            timestamp_bucket: 타임스탬프 버킷
        """
        try:
            # TensorFlow 텐서로 변환
            context_dict = {
                'visual': tf.constant([sensor_data['visual']], dtype=tf.float32),
                'audio': tf.constant([sensor_data['audio']], dtype=tf.float32),
                'pose': tf.constant([sensor_data['pose']], dtype=tf.float32),
                'spatial': tf.constant([sensor_data['spatial']], dtype=tf.float32),
                'time': tf.constant([sensor_data['time']], dtype=tf.float32)
            }

            # AttentionContextEncoder를 통해 160차원 컨텍스트 벡터 생성
            context_160 = self.attention_encoder(context_dict, training=False)[0].numpy()

            # 버퍼에 추가
            self.context_buffer.append(context_160)
            self.timestep_count += 1

            print(f"[{self.timestep_count:04d}] Synced timestep @ {timestamp_bucket:.2f}s → Buffer: {len(self.context_buffer)}/{CONTEXT_BUFFER_SIZE}")

            # 버퍼가 가득 차면 예측 수행
            if len(self.context_buffer) == CONTEXT_BUFFER_SIZE:
                self.predict()

        except Exception as e:
            print(f"Error in process_context: {e}")
            import traceback
            traceback.print_exc()

    def predict(self):
        """
        GRU 모델을 사용하여 예측을 수행합니다.
        """
        try:
            print("\n" + "="*60)
            print(f"Running GRU Prediction #{self.prediction_count + 1}...")
            print("="*60)

            # 버퍼를 numpy 배열로 변환
            X = np.array(self.context_buffer).reshape(1, CONTEXT_BUFFER_SIZE, 160)

            # GRU 모델로 예측
            prediction = self.gru_model.predict(X)[0]

            # 예측 결과 출력
            print_prediction_result(prediction, ZONES)

            self.prediction_count += 1

            # 버퍼 초기화
            self.context_buffer.clear()
            print(f"\nBuffer cleared. Collecting next {CONTEXT_BUFFER_SIZE} timesteps...")
            print("="*60 + "\n")

        except Exception as e:
            print(f"Error in predict: {e}")
            import traceback
            traceback.print_exc()

    def run(self):
        """
        예측기를 실행합니다 (ZeroMQ 폴링 루프).
        """
        print("GRU Predictor started!")
        print(f"  - Waiting for {CONTEXT_BUFFER_SIZE} timesteps of sensor data...")
        print("  - Press Ctrl+C to quit\n")

        try:
            while True:
                self.receive_messages()

        except KeyboardInterrupt:
            print("\nKeyboard interrupt, stopping...")

        finally:
            self.cleanup()

    def cleanup(self):
        """사용한 리소스를 정리합니다."""
        print("\nCleaning up GRU Predictor...")
        self.zmq_socket.close()
        self.zmq_context.term()
        print("GRU Predictor stopped!")
        print(f"\nStatistics:")
        print(f"  - Total timesteps collected: {self.timestep_count}")
        print(f"  - Total predictions made: {self.prediction_count}")
        print(f"  - Sync failures (dropped): {self.sync_dropped}")


if __name__ == "__main__":
    predictor = GRUPredictor()
    predictor.run()
