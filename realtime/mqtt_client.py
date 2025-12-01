"""
엣지 디바이스용 MQTT 클라이언트
Backend와 실시간 양방향 통신을 담당합니다.
"""

import json
import time
from datetime import datetime
from typing import Callable, Optional
import paho.mqtt.client as mqtt


class EdgeMQTTClient:
    """
    엣지 디바이스용 MQTT 클라이언트

    구독 토픽:
        - home/{home_id}/zones/update: 구역 정보 업데이트
        - home/{home_id}/control/clean: 수동 청소 명령
        - home/{home_id}/model/command: 모델 제어 명령
        - home/{home_id}/training/start: 온디바이스 학습 시작 명령

    발행 토픽:
        - home/{home_id}/cleaning/status: 청소 상태 (시작/진행/완료)
        - home/{home_id}/cleaning/result: 청소 결과
        - home/{home_id}/prediction/pollution: 오염도 예측
        - home/{home_id}/training/status: 학습 상태 (started, completed, failed)
        - edge/{device_id}/status: 디바이스 상태
    """

    def __init__(
        self,
        home_id: str,
        device_id: str = "edge_device_001",
        broker_host: str = "localhost",
        broker_port: int = 1883,
        username: Optional[str] = None,
        password: Optional[str] = None
    ):
        """
        MQTT 클라이언트를 초기화합니다.

        Args:
            home_id: 집 ID
            device_id: 디바이스 ID
            broker_host: MQTT Broker 주소 (기본값: localhost)
            broker_port: MQTT Broker 포트 (기본값: 1883)
            username: MQTT 인증 사용자명 (선택)
            password: MQTT 인증 비밀번호 (선택)
        """
        self.home_id = home_id
        self.device_id = device_id
        self.broker_host = broker_host
        self.broker_port = broker_port

        # MQTT 클라이언트 초기화
        self.client = mqtt.Client(client_id=device_id)

        # 인증 설정
        if username and password:
            self.client.username_pw_set(username, password)

        # 콜백 설정
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message

        # 메시지 핸들러 저장
        self.zone_update_handler: Optional[Callable] = None
        self.clean_command_handler: Optional[Callable] = None
        self.model_command_handler: Optional[Callable] = None
        self.training_start_handler: Optional[Callable] = None

        # 연결 상태
        self.is_connected = False

        print(f"\n{'='*60}")
        print(f"EdgeMQTTClient Initialized")
        print(f"{'='*60}")
        print(f"  Home ID: {home_id}")
        print(f"  Device ID: {device_id}")
        print(f"  Broker: {broker_host}:{broker_port}")
        print(f"{'='*60}\n")

    def _on_connect(self, client, userdata, flags, rc):
        """연결 성공 시 콜백"""
        if rc == 0:
            self.is_connected = True
            print(f"✅ Connected to MQTT Broker: {self.broker_host}:{self.broker_port}")

            # 토픽 구독
            self._subscribe_topics()

            # 디바이스 온라인 상태 전송
            self.publish_device_status("online")
        else:
            print(f"❌ Connection failed with code {rc}")
            self.is_connected = False

    def _on_disconnect(self, client, userdata, rc):
        """연결 끊김 시 콜백"""
        self.is_connected = False
        if rc != 0:
            print(f"⚠️  Unexpected disconnection (code: {rc}). Reconnecting...")
        else:
            print("Disconnected from MQTT Broker")

    def _on_message(self, client, userdata, msg):
        """메시지 수신 시 콜백"""
        try:
            topic = msg.topic
            payload = json.loads(msg.payload.decode())

            print(f"\n📨 MQTT Message Received")
            print(f"   Topic: {topic}")
            print(f"   Payload: {payload}")

            # 토픽별 처리
            if "/zones/update" in topic:
                self._handle_zone_update(payload)
            elif "/control/clean" in topic:
                self._handle_clean_command(payload)
            elif "/model/command" in topic:
                self._handle_model_command(payload)
            elif "/training/start" in topic:
                self._handle_training_start(payload)
            else:
                print(f"   Unknown topic: {topic}")

        except json.JSONDecodeError as e:
            print(f"❌ Failed to decode JSON: {e}")
        except Exception as e:
            print(f"❌ Error processing message: {e}")
            import traceback
            traceback.print_exc()

    def _subscribe_topics(self):
        """구독할 토픽 등록"""
        topics = [
            f"home/{self.home_id}/zones/update",
            f"home/{self.home_id}/control/clean",
            f"home/{self.home_id}/model/command",
            f"home/{self.home_id}/training/start"
        ]

        for topic in topics:
            self.client.subscribe(topic, qos=1)
            print(f"Subscribed: {topic}")

    def _handle_zone_update(self, payload):
        """구역 업데이트 처리"""
        if self.zone_update_handler:
            zones = payload.get('zones', [])
            self.zone_update_handler(zones)
        else:
            print("   ⚠️  No zone update handler registered")

    def _handle_clean_command(self, payload):
        """청소 명령 처리"""
        if self.clean_command_handler:
            zone_name = payload.get('zone')
            self.clean_command_handler(zone_name)
        else:
            print("   ⚠️  No clean command handler registered")

    def _handle_model_command(self, payload):
        """모델 명령 처리"""
        if self.model_command_handler:
            command = payload.get('command')
            self.model_command_handler(command)
        else:
            print("   Warning: No model command handler registered")

    def _handle_training_start(self, payload):
        """온디바이스 학습 시작 명령 처리"""
        if self.training_start_handler:
            # 옵션 파라미터 전달 (예: force=True)
            force = payload.get('force', False)
            self.training_start_handler(force=force)
        else:
            print("   Warning: No training start handler registered")

    # ==================== 핸들러 등록 ====================

    def set_zone_update_handler(self, handler: Callable):
        """구역 업데이트 핸들러 등록"""
        self.zone_update_handler = handler
        print(f"✅ Zone update handler registered")

    def set_clean_command_handler(self, handler: Callable):
        """청소 명령 핸들러 등록"""
        self.clean_command_handler = handler
        print(f"✅ Clean command handler registered")

    def set_model_command_handler(self, handler: Callable):
        """모델 명령 핸들러 등록"""
        self.model_command_handler = handler
        print(f"Model command handler registered")

    def set_training_start_handler(self, handler: Callable):
        """온디바이스 학습 시작 핸들러 등록"""
        self.training_start_handler = handler
        print(f"Training start handler registered")

    # ==================== 메시지 발행 ====================

    def publish_cleaning_status(self, status: str, zone: str, **kwargs):
        """
        청소 상태 발행

        Args:
            status: started, in_progress, completed
            zone: 청소 구역 이름
            **kwargs: 추가 정보 (progress, duration_seconds 등)
        """
        topic = f"home/{self.home_id}/cleaning/status"
        payload = {
            "status": status,
            "zone": zone,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }

        self.client.publish(topic, json.dumps(payload), qos=1)
        print(f"📤 Published: {topic}")
        print(f"   Status: {status}, Zone: {zone}")

    def publish_cleaning_result(self, zone: str, duration_seconds: float, **kwargs):
        """
        청소 완료 결과 발행

        Args:
            zone: 청소한 구역 이름
            duration_seconds: 청소 소요 시간 (초)
            **kwargs: 추가 정보 (area_cleaned_m2 등)
        """
        topic = f"home/{self.home_id}/cleaning/result"
        payload = {
            "zone": zone,
            "duration_seconds": duration_seconds,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }

        self.client.publish(topic, json.dumps(payload), qos=1)
        print(f"📤 Published: {topic}")
        print(f"   Zone: {zone}, Duration: {duration_seconds:.1f}s")

    def publish_pollution_prediction(self, predictions: dict):
        """
        오염도 예측 발행

        Args:
            predictions: {zone_name: pollution_score, ...}
        """
        topic = f"home/{self.home_id}/prediction/pollution"
        payload = {
            "predictions": predictions,
            "device_id": self.device_id,
            "timestamp": datetime.now().isoformat()
        }

        self.client.publish(topic, json.dumps(payload), qos=1)
        print(f"📤 Published: {topic}")
        print(f"   Predictions: {predictions}")

    def publish_device_status(self, status: str, **kwargs):
        """
        디바이스 상태 발행

        Args:
            status: online, offline, error
            **kwargs: 추가 정보
        """
        topic = f"edge/{self.device_id}/status"
        payload = {
            "status": status,
            "home_id": self.home_id,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }

        self.client.publish(topic, json.dumps(payload), qos=1, retain=True)
        print(f"Published: {topic} (status: {status})")

    def publish_training_status(self, status: str, **kwargs):
        """
        온디바이스 학습 상태 발행

        Args:
            status: started, completed, failed
            **kwargs: 추가 정보 (samples_used, epochs, loss 등)
        """
        topic = f"home/{self.home_id}/training/status"
        payload = {
            "status": status,
            "device_id": self.device_id,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }

        self.client.publish(topic, json.dumps(payload), qos=1)
        print(f"Published: {topic} (status: {status})")
        if kwargs:
            print(f"   Details: {kwargs}")

    # ==================== 연결 관리 ====================

    def connect(self):
        """MQTT Broker에 연결"""
        try:
            print(f"\nConnecting to {self.broker_host}:{self.broker_port}...")
            self.client.connect(self.broker_host, self.broker_port, keepalive=60)
            self.client.loop_start()  # 백그라운드 스레드 시작

            # 연결 대기 (최대 5초)
            timeout = 5
            start = time.time()
            while not self.is_connected and (time.time() - start) < timeout:
                time.sleep(0.1)

            if not self.is_connected:
                print(f"⚠️  Connection timeout after {timeout}s")
                return False

            return True

        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False

    def disconnect(self):
        """연결 종료"""
        if self.is_connected:
            # 오프라인 상태 전송
            self.publish_device_status("offline")
            time.sleep(0.5)  # 메시지 전송 대기

        self.client.loop_stop()
        self.client.disconnect()
        print("Disconnected from MQTT Broker")


def test_mqtt_client():
    """MQTT 클라이언트 테스트"""
    print("\n" + "="*60)
    print("EdgeMQTTClient Test")
    print("="*60)

    # 클라이언트 초기화
    client = EdgeMQTTClient(
        home_id="test_home_123",
        device_id="test_device",
        broker_host="localhost",
        broker_port=1883
    )

    # 핸들러 등록
    def on_zone_update(zones):
        print(f"\n✅ Zone update received: {len(zones)} zones")
        for zone in zones:
            print(f"   - {zone.get('name')}")

    client.set_zone_update_handler(on_zone_update)

    # 연결
    if client.connect():
        print("\n✅ Test successful! Press Ctrl+C to quit\n")

        try:
            # 테스트 메시지 발행
            time.sleep(2)
            client.publish_pollution_prediction({
                "거실": 0.85,
                "주방": 0.32
            })

            # 무한 대기
            while True:
                time.sleep(1)

        except KeyboardInterrupt:
            print("\n\nTest interrupted by user")
    else:
        print("\n❌ Test failed: Could not connect to broker")

    # 정리
    client.disconnect()


if __name__ == "__main__":
    print("\n⚠️  Make sure Mosquitto is running:")
    print("   $ mosquitto -v")
    print("   or")
    print("   $ docker run -it -p 1883:1883 eclipse-mosquitto\n")

    test_mqtt_client()
