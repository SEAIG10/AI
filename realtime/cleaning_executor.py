"""
Cleaning Executor - Action Execution & Backend Sync
청소 결정을 실행하고, Backend에 비동기로 전송합니다.
"""

import asyncio
import aiohttp
import time
import json
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
                 backend_url: str = "http://localhost:4000",
                 device_id: str = "robot_001",
                 enable_backend: bool = True,
                 mqtt_client=None,
                 feedback_callback=None):
        """
        Args:
            backend_url: LocusBackend API URL
            device_id: 로봇 디바이스 ID
            enable_backend: Backend 통신 활성화 여부 (False면 완전 오프라인)
            mqtt_client: MQTT 클라이언트 (선택)
            feedback_callback: 청소 후 오염도 피드백을 받을 콜백 함수 (선택)
                              signature: callback(actual_pollution: np.ndarray)
        """
        self.backend_url = backend_url
        self.device_id = device_id
        self.enable_backend = enable_backend
        self.mqtt_client = mqtt_client
        self.feedback_callback = feedback_callback

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
        실제 청소 로직 실행

        TODO: 실제 로봇 모터 제어 API 연동
        현재는 시뮬레이션 (각 구역당 대기)

        Args:
            decision: CleaningDecision 객체
        """
        self.is_cleaning = True
        self.cleaning_count += 1

        print(f"\n🤖 Starting Cleaning Session #{self.cleaning_count}")
        print(f"{'='*60}")

        for i, zone in enumerate(decision.path, 1):
            # 오버라이드 체크
            if self.current_override:
                print(f"\n⚠️  Override Command Received: {self.current_override}")
                print(f"   Stopping current cleaning session...")
                break

            print(f"\n[{i}/{len(decision.path)}] 🧹 Cleaning zone: {zone}")
            print(f"   Priority: {decision.priority_order[i-1]:.2%}")

            # MQTT: 청소 시작 알림
            if self.mqtt_client:
                self.mqtt_client.publish_cleaning_status(
                    status="started",
                    zone=zone,
                    priority=float(decision.priority_order[i-1])
                )

            # TODO: 실제 로봇 모터 제어
            # robot_controller.move_to_zone(zone)
            # robot_controller.start_cleaning()

            # 시뮬레이션: 구역당 10초 (실제로는 10분)
            start_time = time.time()
            await asyncio.sleep(10)
            duration = time.time() - start_time

            print(f"   ✅ Zone '{zone}' cleaned!")

            # MQTT: 청소 완료 알림
            if self.mqtt_client:
                self.mqtt_client.publish_cleaning_result(
                    zone=zone,
                    duration_seconds=duration
                )

        self.is_cleaning = False

        if not self.current_override:
            print(f"\n{'='*60}")
            print(f"Cleaning Session #{self.cleaning_count} Completed!")
            print(f"   Total zones cleaned: {len(decision.path)}")
            print(f"   Total time: {decision.estimated_time} minutes (simulated)")
            print(f"{'='*60}\n")

            # 청소 후 오염도 측정 및 피드백
            if self.feedback_callback:
                actual_pollution = self._measure_pollution_after_cleaning()
                print(f"\n[Feedback] Measured pollution after cleaning: {actual_pollution}")
                self.feedback_callback(actual_pollution)
        else:
            # 오버라이드로 중단됨
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

    def _measure_pollution_after_cleaning(self) -> np.ndarray:
        """
        청소 후 실제 오염도를 측정합니다.

        TODO: 실제 센서로부터 오염도 측정
        현재는 시뮬레이션 (청소 후 낮은 값 반환)

        Returns:
            측정된 오염도 (num_zones,) numpy array
        """
        # 시뮬레이션: 청소 후 0.1~0.3 범위의 낮은 오염도
        num_zones = len(self.decision_engine.zone_names)
        actual_pollution = np.random.uniform(0.1, 0.3, num_zones).astype(np.float32)

        # TODO: 실제 구현
        # sensors = collect_sensor_data()
        # actual_pollution = compute_pollution_from_sensors(sensors)

        return actual_pollution

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
        backend_url="http://localhost:4000",
        device_id="test_robot_001",
        enable_backend=False  # 오프라인 모드 테스트
    )

    # 테스트 예측 결과
    test_prediction = np.array([0.85, 0.12, 0.65, 0.23])  # balcony, kitchen 청소 필요

    print("Simulating GRU prediction...\n")

    # 동기식으로 실행
    executor.handle_prediction_sync(test_prediction)

    print("\n✅ Test completed!")
