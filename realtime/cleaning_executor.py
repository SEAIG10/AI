"""
Cleaning Executor - Action Execution & Backend Sync
청소 결정을 실행하고, Backend에 비동기로 전송합니다.
"""

import asyncio
import aiohttp
import time
import json
import zmq
from typing import Optional
from decision_engine import LocalDecisionEngine, CleaningDecision
import numpy as np


class CleaningExecutor:
    """
    청소 실행 및 Backend 통신

    - 로컬 우선: 즉시 청소 실행 (오프라인 모드 OK)
    - Backend 동기화: 비동기로 예측 결과 전송 (non-blocking)
    - WebSocket 오버라이드: Backend에서 긴급 명령 수신 (TODO)
    """

    def __init__(self,
                 backend_url: str = "https://a599ad257210.ngrok-free.app",
                 device_id: str = "robot_001",
                 enable_backend: bool = True,
                 mqtt_client=None,
                 feedback_callback=None,
                 zmq_bridge_socket=None):
        """
        Args:
            backend_url: LocusBackend API URL
            device_id: 로봇 디바이스 ID
            enable_backend: Backend 통신 활성화 여부 (False면 완전 오프라인)
            mqtt_client: MQTT 클라이언트 (선택)
            feedback_callback: 청소 후 오염도 피드백을 받을 콜백 함수 (선택)
                              signature: callback(actual_pollution: np.ndarray)
            zmq_bridge_socket: WebSocket Bridge로 메시지를 보낼 ZeroMQ 소켓 (선택)
        """
        self.backend_url = backend_url
        self.device_id = device_id
        self.enable_backend = enable_backend
        self.mqtt_client = mqtt_client
        self.feedback_callback = feedback_callback
        self.zmq_bridge_socket = zmq_bridge_socket

        # Decision Engine 생성
        self.decision_engine = LocalDecisionEngine(
            pollution_threshold=0.5,
            zone_names=['balcony', 'bedroom', 'kitchen', 'living_room']
        )

        # 상태 관리
        self.is_cleaning = False
        self.current_override = None  # Backend에서 받은 오버라이드 명령
        self.cleaning_count = 0

        print(f"\n{'='*60}")
        print(f"Cleaning Executor Initialized")
        print(f"{'='*60}")
        print(f"Device ID: {self.device_id}")
        print(f"Backend URL: {self.backend_url}")
        print(f"Backend Sync: {'Enabled' if self.enable_backend else 'Disabled (Offline Mode)'}")
        print(f"WebSocket Bridge: {'Enabled' if self.zmq_bridge_socket else 'Disabled'}")
        print(f"{'='*60}\n")

    async def handle_prediction(self, prediction: np.ndarray):
        """
        GRU 예측 결과를 받아서 처리합니다.

        1. 로컬 결정 엔진으로 청소 결정
        2. 청소 실행 (백그라운드)
        3. Backend에 전송 (비동기, non-blocking)

        Args:
            prediction: GRU 모델 출력 (4,) numpy array
        """
        print(f"\n{'#'*60}")
        print(f"New Prediction Received")
        print(f"{'#'*60}")
        print(f"Raw Prediction: {prediction}")

        # 1. 로컬 결정 (즉시)
        decision = self.decision_engine.decide(prediction)
        print(decision)

        # 2. 청소 실행 (백그라운드)
        if decision.zones_to_clean:
            asyncio.create_task(self._execute_cleaning(decision))
        else:
            print("✅ No action needed - all zones clean!\n")

        # 3. Backend에 전송 (비동기, non-blocking)
        if self.enable_backend:
            asyncio.create_task(self._send_to_backend(prediction, decision))

    async def _execute_cleaning(self, decision: CleaningDecision):
        """
        실제 청소 로직 실행 (개선된 피드백 학습)

        새로운 플로우:
        1. 각 구역으로 이동
        2. 청소 전 YOLO 실측 -> Ground Truth 확보
        3. GRU 예측 vs 실측 비교 -> 피드백 학습 데이터 수집
        4. 오염물 있으면 청소, 없으면 스킵

        Args:
            decision: CleaningDecision 객체
        """
        self.is_cleaning = True
        self.cleaning_count += 1

        print(f"\n[Robot] Starting Cleaning Session #{self.cleaning_count}")
        print(f"{'='*60}")

        # 피드백 학습용 데이터 수집
        feedback_data = []

        for i, zone in enumerate(decision.path, 1):
            # 오버라이드 체크
            if self.current_override:
                print(f"\n[Warning] Override Command Received: {self.current_override}")
                print(f"   Stopping current cleaning session...")
                break

            print(f"\n[{i}/{len(decision.path)}] Processing zone: {zone}")
            gru_prediction = decision.priority_order[i-1]
            print(f"   GRU predicted: {gru_prediction:.3f}")

            # Step 1: 청소 전 YOLO 실측 (Ground Truth)
            print(f"   [Measuring] actual pollution (pre-cleaning)...")
            actual_pollution = self._measure_pollution_now(zone)

            # Step 2: GRU 예측 vs 실측 비교
            error = gru_prediction - actual_pollution
            print(f"   [Comparison] GRU: {gru_prediction:.3f}, Actual: {actual_pollution:.3f}, Error: {error:+.3f}")

            # 피드백 데이터 수집
            zone_idx = self.decision_engine.zone_names.index(zone)
            feedback_data.append((zone_idx, actual_pollution))

            # Step 3: 실제 오염물이 있으면 청소, 없으면 스킵
            cleaning_threshold = 0.15

            if actual_pollution > cleaning_threshold:
                print(f"   [Action] Pollution detected -> Cleaning")

                # WebSocket Bridge: 청소 시작 알림
                if self.zmq_bridge_socket:
                    self.zmq_bridge_socket.send_pyobj({
                        'type': 'cleaning_started',
                        'timestamp': time.time(),
                        'zone': zone,
                        'priority': float(gru_prediction),
                        'total_zones': len(decision.path),
                        'current_index': i
                    })

                # MQTT: 청소 시작 알림 (Backend 통신 - 필수!)
                if self.mqtt_client:
                    self.mqtt_client.publish_cleaning_status(
                        status="started",
                        zone=zone,
                        priority=float(gru_prediction)
                    )

                # 청소 수행
                start_time = time.time()
                await asyncio.sleep(10)
                duration = time.time() - start_time

                print(f"   [Done] Zone '{zone}' cleaned!")

                # WebSocket Bridge: 청소 완료 알림
                if self.zmq_bridge_socket:
                    self.zmq_bridge_socket.send_pyobj({
                        'type': 'cleaning_completed',
                        'timestamp': time.time(),
                        'zone': zone,
                        'duration_seconds': duration
                    })

                # MQTT: 청소 완료 알림 (Backend 통신 - 필수!)
                if self.mqtt_client:
                    self.mqtt_client.publish_cleaning_result(
                        zone=zone,
                        duration_seconds=duration
                    )
            else:
                print(f"   [Skip] No cleaning needed (false positive)")

        self.is_cleaning = False

        # Step 4: 모든 구역 완료 후 피드백 학습
        if not self.current_override and self.feedback_callback and feedback_data:
            print(f"\n{'='*60}")
            print(f"Cleaning Session #{self.cleaning_count} Completed!")
            print(f"{'='*60}")

            # 구역별 실측값을 배열로 변환
            num_zones = len(self.decision_engine.zone_names)
            actual_pollution_array = np.zeros(num_zones, dtype=np.float32)

            for zone_idx, pollution in feedback_data:
                actual_pollution_array[zone_idx] = pollution

            print(f"\n[Feedback] Ground truth: {actual_pollution_array}")

            # 피드백 학습 실행
            self.feedback_callback(actual_pollution_array)
            print(f"{'='*60}\n")
        elif self.current_override:
            self.current_override = None

    async def _send_to_backend(self, prediction: np.ndarray, decision: CleaningDecision):
        """
        Backend에 예측 결과 및 결정 전송 (비동기)

        Args:
            prediction: GRU 예측 결과
            decision: 청소 결정
        """
        try:
            payload = {
                "device_id": self.device_id,
                "timestamp": time.time(),
                "prediction": {
                    "balcony": float(prediction[0]),
                    "bedroom": float(prediction[1]),
                    "kitchen": float(prediction[2]),
                    "living_room": float(prediction[3])
                },
                "decision": {
                    "zones_to_clean": decision.zones_to_clean,
                    "priority_order": [float(p) for p in decision.priority_order],
                    "estimated_time": decision.estimated_time,
                    "path": decision.path,
                    "threshold": decision.threshold_used
                }
            }

            async with aiohttp.ClientSession() as session:
                url = f"{self.backend_url}/api/predictions"
                async with session.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    if resp.status == 200 or resp.status == 201:
                        print(f"✅ [Backend] Prediction sent successfully")
                    else:
                        text = await resp.text()
                        print(f"⚠️  [Backend] Failed to send prediction: {resp.status} - {text}")

        except asyncio.TimeoutError:
            print(f"⚠️  [Backend] Timeout (continuing anyway - offline mode)")
        except aiohttp.ClientError as e:
            print(f"⚠️  [Backend] Network error (continuing anyway): {e}")
        except Exception as e:
            print(f"⚠️  [Backend] Unexpected error: {e}")

    def _measure_pollution_now(self, zone: str = None) -> float:
        """
        현재 오염도를 YOLO로 측정합니다 (Confidence 기반).

        YOLO로 웹캠 캡처 후 오염물(solid_waste, liquid_stain) 탐지하여
        실제 오염도를 측정합니다.

        Args:
            zone: 측정할 구역 이름 (선택, 현재는 미사용)

        Returns:
            측정된 오염도 (0.0~1.0)
        """
        try:
            import cv2
            from ultralytics import YOLO
            import os

            # YOLO 모델 경로
            yolo_model_path = os.path.join(
                os.path.dirname(__file__), '..', 'models', 'yolo', 'best.pt'
            )

            # YOLO 모델 로드
            yolo_model = YOLO(yolo_model_path)

            # 웹캠에서 프레임 캡처
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("      ⚠️  Camera not available, using fallback")
                return self._fallback_measurement_single()

            ret, frame = cap.read()
            cap.release()

            if not ret:
                print("      ⚠️  Failed to capture frame, using fallback")
                return self._fallback_measurement_single()

            # YOLO로 오염물 탐지
            results = yolo_model(frame, verbose=False)

            # Confidence 기반 오염도 계산
            pollution_score = 0.1  # 기본값 (깨끗함)
            num_solid_waste = 0
            num_liquid_stain = 0

            if len(results) > 0 and hasattr(results[0], 'boxes'):
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    confidence = float(box.conf[0])  # YOLO 신뢰도 (0~1)

                    if cls_id == 12:  # solid_waste
                        pollution_score += confidence * 0.15
                        num_solid_waste += 1
                    elif cls_id == 13:  # liquid_stain
                        pollution_score += confidence * 0.20
                        num_liquid_stain += 1

            # 최대 1.0으로 제한
            pollution_score = min(pollution_score, 1.0)

            # 결과 출력
            print(f"      [YOLO] solid_waste: {num_solid_waste}, liquid_stain: {num_liquid_stain}")
            print(f"      [YOLO] Pollution score: {pollution_score:.3f}")

            return pollution_score

        except Exception as e:
            print(f"      ⚠️  YOLO error: {e}")
            return self._fallback_measurement_single()

    def _fallback_measurement_single(self) -> float:
        """
        YOLO 측정 실패 시 폴백 (단일 값)

        Returns:
            랜덤 오염도 값 (0.1~0.3)
        """
        return float(np.random.uniform(0.1, 0.3))

    def handle_prediction_sync(self, prediction: np.ndarray):
        """
        동기식 래퍼 (asyncio 이벤트 루프가 없는 경우)

        Args:
            prediction: GRU 예측 결과
        """
        # 새 이벤트 루프 생성 및 실행
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.handle_prediction(prediction))
        finally:
            # 실행 중인 태스크 정리
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()


# 테스트
if __name__ == "__main__":
    print("Testing Cleaning Executor...\n")

    # Executor 생성 (Backend 비활성화 - 로컬 테스트)
    executor = CleaningExecutor(
        backend_url="https://a599ad257210.ngrok-free.app",
        device_id="test_robot_001",
        enable_backend=False  # 오프라인 모드 테스트
    )

    # 테스트 예측 결과
    test_prediction = np.array([0.85, 0.12, 0.65, 0.23])  # balcony, kitchen 청소 필요

    print("Simulating GRU prediction...\n")

    # 동기식으로 실행
    executor.handle_prediction_sync(test_prediction)

    print("\n✅ Test completed!")
