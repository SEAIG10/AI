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
from realtime.cleaning_executor import CleaningExecutor
from realtime.mqtt_client import EdgeMQTTClient
from realtime.zone_manager import ZoneManager
from realtime.on_device_trainer import OnDeviceTrainer

# ZeroMQ 설정
ZMQ_ENDPOINT = "ipc:///tmp/locus_sensors.ipc"
ZMQ_BRIDGE_ENDPOINT = "ipc:///tmp/locus_bridge.ipc"  # 브릿지 전용

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

    def __init__(
        self,
        enable_cleaning: bool = True,
        enable_on_device_training: bool = True,
        backend_url: str = "https://a599ad257210.ngrok-free.app",
        home_id: str = "1",
        mqtt_broker: str = "43.200.178.189",
        mqtt_port: int = 1883
    ):
        """
        GRU 예측기를 초기화합니다.

        Args:
            enable_cleaning: 청소 실행 기능 활성화 여부
            enable_on_device_training: 온디바이스 Head 학습 활성화 여부
            backend_url: LocusBackend API URL
            home_id: 집 ID
            mqtt_broker: MQTT Broker 주소
            mqtt_port: MQTT Broker 포트
        """
        print("="*60)
        print("GRU Predictor Initializing...")
        print("="*60)

        self.home_id = home_id

        # ZeroMQ Subscriber 설정 (BIND - 구독자가 바인드하고, 발행자가 연결)
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.SUB)
        self.zmq_socket.bind(ZMQ_ENDPOINT)
        self.zmq_socket.setsockopt_string(zmq.SUBSCRIBE, "")  # 모든 메시지 구독
        print(f"ZeroMQ bound to {ZMQ_ENDPOINT}")
        print("Subscribed to all sensor messages")

        # ZeroMQ Publisher for Bridge (센서 데이터를 브릿지로 재전송)
        self.zmq_bridge_socket = self.zmq_context.socket(zmq.PUB)
        self.zmq_bridge_socket.bind(ZMQ_BRIDGE_ENDPOINT)
        print(f"ZeroMQ bridge publisher bound to {ZMQ_BRIDGE_ENDPOINT}")

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

        # MQTT 클라이언트 초기화
        print("\n  3. MQTT Client...")
        self.mqtt_client = EdgeMQTTClient(
            home_id=home_id,
            device_id="edge_device_001",
            broker_host=mqtt_broker,
            broker_port=mqtt_port
        )

        # MQTT 연결
        if self.mqtt_client.connect():
            print("     MQTT Client connected!")
        else:
            print("     ⚠️  MQTT Client connection failed (continuing without MQTT)")
            self.mqtt_client = None

        # OnDeviceTrainer 초기화 (ZoneManager보다 먼저)
        self.enable_on_device_training = enable_on_device_training
        if self.enable_on_device_training:
            print("\n  4. OnDeviceTrainer...")
            model_save_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'gru', 'gru_model_personalized.keras')
            self.on_device_trainer = OnDeviceTrainer(
                gru_model=self.gru_model,
                buffer_size=300,
                min_samples_for_training=100,
                batch_size=16,
                epochs_per_update=5,
                learning_rate=0.0005,
                auto_save_path=model_save_path,
                mqtt_client=self.mqtt_client
            )
            print("     OnDeviceTrainer initialized!")
        else:
            self.on_device_trainer = None
            print("\n  4. On-device training disabled")

        # ZoneManager 초기화
        print("\n  5. ZoneManager...")
        self.zone_manager = ZoneManager(
            gru_model=self.gru_model,
            mqtt_client=self.mqtt_client,
            on_device_trainer=self.on_device_trainer
        )
        print("     ZoneManager initialized!")

        # MQTT 핸들러 등록
        if self.mqtt_client:
            self.mqtt_client.set_zone_update_handler(
                lambda zones: self.zone_manager.update_zones(self.home_id, zones)
            )

            # 온디바이스 학습 핸들러 등록
            if self.enable_on_device_training:
                self.mqtt_client.set_training_start_handler(
                    lambda force=False: self._handle_training_command(force)
                )

            print("     MQTT handlers registered!")

        # Cleaning Executor 초기화
        self.enable_cleaning = enable_cleaning
        if self.enable_cleaning:
            print("\n  6. Cleaning Executor...")
            self.cleaning_executor = CleaningExecutor(
                backend_url=backend_url,
                device_id="robot_001",
                enable_backend=True,
                mqtt_client=self.mqtt_client,
                feedback_callback=self._handle_cleaning_feedback if self.enable_on_device_training else None,
                zmq_bridge_socket=self.zmq_bridge_socket  # WebSocket Bridge로 청소 상태 전송
            )
            print("     Cleaning Executor initialized!")
        else:
            self.cleaning_executor = None
            print("\nCleaning execution disabled (prediction only mode)")

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

        # 피드백을 위한 임시 저장소
        self.last_context_sequence = None  # 마지막 예측에 사용된 context sequence (30, 160)
        self.last_prediction = None  # 마지막 예측 결과

        print("\nGRU Predictor ready!\n")

    def receive_messages(self):
        """
        ZeroMQ 메시지를 ROS ApproximateTimeSynchronizer 방식으로 수신합니다.
        """
        try:
            # 논블로킹 방식으로 타임아웃과 함께 메시지 수신
            if self.zmq_socket.poll(timeout=100):  # 100ms 타임아웃
                message = self.zmq_socket.recv_pyobj()

                # 브릿지로 재전송 (websocket_bridge가 받을 수 있도록)
                self.zmq_bridge_socket.send_pyobj(message)

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

            # WebSocket Bridge: 버퍼 상태 전송
            self.zmq_bridge_socket.send_pyobj({
                'type': 'buffer_status',
                'timestamp': time.time(),
                'buffer_size': len(self.context_buffer),
                'buffer_capacity': CONTEXT_BUFFER_SIZE
            })

            # 버퍼가 가득 차면 예측 수행
            if len(self.context_buffer) == CONTEXT_BUFFER_SIZE:
                self.predict()

        except Exception as e:
            print(f"Error in process_context: {e}")
            import traceback
            traceback.print_exc()

    def predict(self):
        """
        GRU 모델을 사용하여 예측을 수행하고, 청소 결정을 내립니다.
        """
        try:
            print("\n" + "="*60)
            print(f"Running GRU Prediction #{self.prediction_count + 1}...")
            print("="*60)

            # 버퍼를 numpy 배열로 변환
            X = np.array(self.context_buffer).reshape(1, CONTEXT_BUFFER_SIZE, 160)

            # 온디바이스 학습을 위해 context sequence 저장
            if self.enable_on_device_training:
                self.last_context_sequence = X[0].copy()  # (30, 160)

            # GRU 모델로 예측
            prediction = self.gru_model.predict(X)[0]
            self.last_prediction = prediction.copy()

            # 예측 결과 출력
            print_prediction_result(prediction, ZONES)

            self.prediction_count += 1

            # MQTT로 오염도 예측 발행
            zone_names = self.zone_manager.get_current_zones()
            predictions_dict = {}
            for i, zone in enumerate(zone_names):
                # DB와 매칭되도록 영어 이름 사용
                zone_name = zone.get('name_en', zone.get('name', f'Zone {i}'))
                if i < len(prediction):
                    predictions_dict[zone_name] = float(prediction[i])

            # WebSocket Bridge로 예측 결과 전송 (대시보드용)
            self.zmq_bridge_socket.send_pyobj({
                'type': 'prediction',
                'timestamp': time.time(),
                'prediction': predictions_dict
            })
            print(f"📡 [Bridge] Sent prediction to WebSocket: {predictions_dict}")

            if self.mqtt_client:
                self.mqtt_client.publish_pollution_prediction(predictions_dict)

            # 청소 실행 (활성화된 경우)
            if self.enable_cleaning and self.cleaning_executor:
                print("\nTriggering Cleaning Executor...")
                self.cleaning_executor.handle_prediction_sync(prediction)

            # 버퍼 초기화
            self.context_buffer.clear()

            # WebSocket Bridge: 버퍼 리셋 & 센서 카운터 리셋 신호
            self.zmq_bridge_socket.send_pyobj({
                'type': 'buffer_reset',
                'timestamp': time.time()
            })

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

    def _handle_cleaning_feedback(self, actual_pollution: np.ndarray):
        """
        CleaningExecutor로부터 청소 전 실제 오염도 피드백을 받아 OnDeviceTrainer에 전달합니다.

        Args:
            actual_pollution: 청소 전 YOLO로 측정된 실제 오염도 (num_zones,)
        """
        if not self.enable_on_device_training or self.on_device_trainer is None:
            return

        if self.last_context_sequence is None:
            print("Warning: No context sequence saved for feedback")
            return

        print(f"\n[OnDevice Learning] Feedback received")
        print(f"  Context sequence shape: {self.last_context_sequence.shape}")
        print(f"  Actual pollution: {actual_pollution}")

        # OnDeviceTrainer에 샘플 추가
        self.on_device_trainer.add_sample(
            context_sequence=self.last_context_sequence,
            pollution_label=actual_pollution
        )

    def _handle_training_command(self, force: bool = False):
        """
        MQTT를 통해 받은 학습 시작 명령을 처리합니다.

        Args:
            force: True면 버퍼 크기 무시하고 강제 학습
        """
        print(f"\n[MQTT] Training command received (force={force})")

        if not self.enable_on_device_training or self.on_device_trainer is None:
            print("Warning: On-device training is disabled")
            if self.mqtt_client:
                self.mqtt_client.publish_training_status(
                    "failed",
                    reason="training_disabled"
                )
            return

        # MQTT로 학습 시작 상태 전송
        if self.mqtt_client:
            buffer_size = len(self.on_device_trainer.X_buffer)
            self.mqtt_client.publish_training_status(
                "started",
                buffer_size=buffer_size,
                min_samples=self.on_device_trainer.min_samples_for_training,
                force=force
            )

        # ZoneManager를 통해 학습 시작
        if force:
            # 강제 학습: 버퍼에 데이터가 있으면 무조건 학습
            if len(self.on_device_trainer.X_buffer) > 0:
                print(f"  Forcing training with {len(self.on_device_trainer.X_buffer)} samples")
                self.on_device_trainer.start_background_training()
            else:
                print("  No samples in buffer, cannot train")
                if self.mqtt_client:
                    self.mqtt_client.publish_training_status(
                        "failed",
                        reason="no_samples"
                    )
        else:
            # 일반 학습: ZoneManager가 조건 체크
            self.zone_manager.start_on_device_training()

    def cleanup(self):
        """사용한 리소스를 정리합니다."""
        print("\nCleaning up GRU Predictor...")

        # OnDeviceTrainer 종료
        if self.on_device_trainer:
            self.on_device_trainer.stop()

        # MQTT 연결 종료
        if self.mqtt_client:
            self.mqtt_client.disconnect()

        # ZeroMQ 종료
        self.zmq_socket.close()
        self.zmq_bridge_socket.close()
        self.zmq_context.term()

        print("GRU Predictor stopped!")
        print(f"\nStatistics:")
        print(f"  - Total timesteps collected: {self.timestep_count}")
        print(f"  - Total predictions made: {self.prediction_count}")
        print(f"  - Sync failures (dropped): {self.sync_dropped}")

        # OnDeviceTrainer 통계
        if self.on_device_trainer:
            stats = self.on_device_trainer.get_statistics()
            print(f"\nOn-Device Training Statistics:")
            print(f"  - Buffer size: {stats['buffer_size']}")
            print(f"  - Total samples collected: {stats['total_samples_collected']}")
            print(f"  - Total training runs: {stats['total_training_runs']}")


if __name__ == "__main__":
    predictor = GRUPredictor()
    predictor.run()
