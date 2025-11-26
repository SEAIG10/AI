#!/usr/bin/env python3
"""
LocationTracker 서버로 zone 업데이트를 테스트하는 스크립트
iPhone 없이 직접 WebSocket 메시지를 전송합니다.
"""
import asyncio
import websockets
import json
import time

async def test_zone_updates():
    # LocationTracker 서버 주소
    uri = "ws://172.30.1.40:8080"

    print("=" * 60)
    print("LocationTracker Zone Update Test")
    print("=" * 60)

    try:
        async with websockets.connect(uri) as ws:
            print(f"\n✅ Connected to {uri}\n")

            # 1. Welcome 메시지 받기
            welcome = await ws.recv()
            print(f"📥 Welcome: {welcome}\n")

            # 2. Identify as tracker
            identify_msg = {
                'type': 'identify',
                'clientType': 'tracker'
            }
            await ws.send(json.dumps(identify_msg))
            print(f"📤 Sent identify message\n")

            # 3. 여러 zone으로 테스트
            test_cases = [
                {"x": 0.0, "y": 0.0, "z": 0.0, "zone": "living_room"},
                {"x": 1.5, "y": 0.0, "z": 1.0, "zone": "kitchen"},
                {"x": -1.0, "y": 0.0, "z": 2.0, "zone": "bedroom"},
                {"x": 0.5, "y": 0.0, "z": -1.0, "zone": "balcony"},
            ]

            for i, test in enumerate(test_cases, 1):
                print(f"[Test {i}] Sending position → Expected zone: {test['zone']}")

                location_msg = {
                    'type': 'arkit_location',
                    'data': {
                        'position3D': {
                            'x': test['x'],
                            'y': test['y'],
                            'z': test['z']
                        },
                        'accuracy': 0.01,
                        'timestamp': int(time.time() * 1000)
                    }
                }

                await ws.send(json.dumps(location_msg))
                print(f"  📤 Sent: ({test['x']}, {test['y']}, {test['z']})")
                print(f"  ⏳ Waiting 5 seconds...\n")

                await asyncio.sleep(5)

            print("=" * 60)
            print("✅ All tests completed!")
            print("=" * 60)
            print("\nCheck SE_G10 terminal for zone updates!")

    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_zone_updates())
