"""
Realtime Demo - GRU Predictor
MQTT로 센서 데이터 수집 후 GRU로 오염 예측
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import paho.mqtt.client as mqtt
import pickle
import time
import numpy as np
import tensorflow as tf
from collections import deque

# Import existing models
from src.context_fusion.attention_context_encoder import create_attention_encoder
from src.model.gru_model import FedPerGRUModel
from realtime.utils import print_prediction_result, ZONES

# MQTT 설정
MQTT_BROKER = "localhost"
MQTT_PORT = 1883

# 모델 경로
GRU_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'gru', 'gru_model.keras')

# Context buffer 설정
CONTEXT_BUFFER_SIZE = 30  # 30 timesteps


class GRUPredictor:
    """
    GRU Predictor
    MQTT로 센서 데이터를 받아서 AttentionContextEncoder → GRU로 예측
    """

    def __init__(self):
        """Initialize GRU Predictor"""
        print("="*60)
        print("🧠 GRU Predictor Initializing...")
        print("="*60)

        # MQTT 클라이언트
        self.mqtt_client = mqtt.Client("gru_predictor")
        self.mqtt_client.on_message = self.on_message
        self.mqtt_client.connect(MQTT_BROKER, MQTT_PORT)
        print(f"✓ MQTT connected to {MQTT_BROKER}:{MQTT_PORT}")

        # 모든 센서 구독
        self.mqtt_client.subscribe("sensor/#")
        print("✓ Subscribed to sensor/#")

        # 모델 로드
        print("\n📦 Loading models...")
        print("  1. AttentionContextEncoder...")
        self.attention_encoder = create_attention_encoder(
            visual_dim=14,
            audio_dim=17,
            pose_dim=51,
            spatial_dim=7,
            time_dim=10,
            context_dim=160
        )
        print("     ✓ AttentionContextEncoder loaded!")

        print(f"  2. GRU Model from {GRU_MODEL_PATH}...")
        self.gru_model = FedPerGRUModel(num_zones=7, context_dim=160)
        self.gru_model.load(GRU_MODEL_PATH)
        print("     ✓ GRU Model loaded!")

        # 센서 데이터 버퍼
        self.current_context = {
            'visual': None,
            'audio': None,
            'pose': None,
            'spatial': None,
            'time': None
        }

        # Context buffer (30 timesteps)
        self.context_buffer = deque(maxlen=CONTEXT_BUFFER_SIZE)

        # 통계
        self.timestep_count = 0
        self.prediction_count = 0

        print("\n✅ GRU Predictor ready!\n")

    def on_message(self, client, userdata, msg):
        """
        MQTT 메시지 수신 콜백

        Args:
            msg: MQTT message
        """
        try:
            # 메시지 파싱
            topic = msg.topic  # "sensor/visual"
            data = pickle.loads(msg.payload)

            sensor_type = topic.split('/')[-1]  # "visual"

            # 센서 데이터 저장
            if sensor_type in self.current_context:
                self.current_context[sensor_type] = data[sensor_type]

                # 로그 (간단하게)
                # print(f"  [MQTT] Received: {sensor_type}")

            # 모든 센서 데이터가 모였는지 확인
            if all(v is not None for v in self.current_context.values()):
                self.process_context()

        except Exception as e:
            print(f"⚠ Error in on_message: {e}")

    def process_context(self):
        """
        모든 센서 데이터가 모였을 때 처리
        AttentionContextEncoder로 160-dim 생성 후 버퍼에 추가
        """
        try:
            # TensorFlow 형식으로 변환
            context_dict = {
                'visual': tf.constant([self.current_context['visual']], dtype=tf.float32),
                'audio': tf.constant([self.current_context['audio']], dtype=tf.float32),
                'pose': tf.constant([self.current_context['pose']], dtype=tf.float32),
                'spatial': tf.constant([self.current_context['spatial']], dtype=tf.float32),
                'time': tf.constant([self.current_context['time']], dtype=tf.float32)
            }

            # AttentionContextEncoder로 160-dim 생성
            context_160 = self.attention_encoder(context_dict, training=False)[0].numpy()

            # Buffer에 추가
            self.context_buffer.append(context_160)
            self.timestep_count += 1

            print(f"[{self.timestep_count:04d}] ✅ Context created → Buffer: {len(self.context_buffer)}/{CONTEXT_BUFFER_SIZE} timesteps")

            # 30개 모이면 예측
            if len(self.context_buffer) == CONTEXT_BUFFER_SIZE:
                self.predict()

            # 센서 데이터 초기화
            self.current_context = {k: None for k in self.current_context}

        except Exception as e:
            print(f"⚠ Error in process_context: {e}")
            import traceback
            traceback.print_exc()

    def predict(self):
        """
        GRU로 예측 수행
        """
        try:
            print("\n" + "="*60)
            print(f"🧠 Running GRU Prediction #{self.prediction_count + 1}...")
            print("="*60)

            # Buffer를 numpy array로 변환
            X = np.array(self.context_buffer).reshape(1, CONTEXT_BUFFER_SIZE, 160)

            # GRU 예측
            prediction = self.gru_model.predict(X, verbose=0)[0]

            # 결과 출력
            print_prediction_result(prediction, ZONES)

            self.prediction_count += 1

            # Buffer 초기화 (다시 30개 모으기)
            self.context_buffer.clear()
            print(f"\n✓ Buffer cleared. Collecting next {CONTEXT_BUFFER_SIZE} timesteps...")
            print("="*60 + "\n")

        except Exception as e:
            print(f"⚠ Error in predict: {e}")
            import traceback
            traceback.print_exc()

    def run(self):
        """
        Predictor 실행 (MQTT loop)
        """
        print("🚀 GRU Predictor started!")
        print(f"  - Waiting for {CONTEXT_BUFFER_SIZE} timesteps of sensor data...")
        print("  - Press Ctrl+C to quit\n")

        try:
            self.mqtt_client.loop_forever()

        except KeyboardInterrupt:
            print("\n⚠ Keyboard interrupt, stopping...")

        finally:
            self.cleanup()

    def cleanup(self):
        """리소스 정리"""
        print("\n🧹 Cleaning up GRU Predictor...")
        self.mqtt_client.disconnect()
        print("✓ GRU Predictor stopped!")
        print(f"\nStatistics:")
        print(f"  - Total timesteps collected: {self.timestep_count}")
        print(f"  - Total predictions made: {self.prediction_count}")


if __name__ == "__main__":
    predictor = GRUPredictor()
    predictor.run()
