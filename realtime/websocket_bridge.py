"""
WebSocket Bridge - ZeroMQ to WebSocket
ZeroMQ 메시지를 받아서 브라우저(WebSocket)로 중계합니다.
"""

import asyncio
import json
import zmq
import zmq.asyncio
from datetime import datetime
from typing import Set
import websockets
from websockets.server import WebSocketServerProtocol
import numpy as np

# ZeroMQ 설정 - 브릿지 전용 엔드포인트 (gru_predictor가 재전송)
ZMQ_ENDPOINT = "ipc:///tmp/locus_bridge.ipc"

# WebSocket 설정
WEBSOCKET_HOST = "0.0.0.0"
WEBSOCKET_PORT = 8080

# 연결된 클라이언트들
connected_clients: Set[WebSocketServerProtocol] = set()


class WebSocketBridge:
    """ZeroMQ → WebSocket 브릿지"""

    def __init__(self):
        self.zmq_context = zmq.asyncio.Context()
        self.zmq_socket = None
        self.is_running = False

        # 통계
        self.message_count = 0
        self.client_count = 0

    async def setup_zmq(self):
        """ZeroMQ 소켓 설정"""
        self.zmq_socket = self.zmq_context.socket(zmq.SUB)
        self.zmq_socket.connect(ZMQ_ENDPOINT)
        self.zmq_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        print(f"[ZMQ] Connected to {ZMQ_ENDPOINT}")

    async def zmq_listener(self):
        """ZeroMQ 메시지를 받아서 모든 WebSocket 클라이언트에게 전송"""
        print("[ZMQ] Listener started")

        while self.is_running:
            try:
                # ZeroMQ 메시지 수신
                message = await self.zmq_socket.recv_pyobj()
                self.message_count += 1

                # 타임스탬프 추가
                message['bridge_timestamp'] = datetime.now().isoformat()

                # numpy array를 list로 변환 (JSON serialization을 위해)
                if 'data' in message and isinstance(message['data'], np.ndarray):
                    message['data'] = message['data'].tolist()

                # JSON으로 변환
                json_message = json.dumps(message)

                # 모든 연결된 클라이언트에게 브로드캐스트
                if connected_clients:
                    disconnected = set()
                    for client in connected_clients:
                        try:
                            await client.send(json_message)
                        except websockets.exceptions.ConnectionClosed:
                            disconnected.add(client)
                        except Exception as e:
                            print(f"[WS] Error sending to client: {e}")
                            disconnected.add(client)

                    # 연결 끊긴 클라이언트 제거
                    for client in disconnected:
                        connected_clients.discard(client)

                # 로그 (10개마다)
                if self.message_count % 10 == 0:
                    print(f"[Bridge] Messages: {self.message_count}, Clients: {len(connected_clients)}")

            except Exception as e:
                print(f"[ZMQ] Error receiving message: {e}")
                await asyncio.sleep(0.1)

    async def websocket_handler(self, websocket: WebSocketServerProtocol):
        """WebSocket 클라이언트 연결 핸들러"""
        # 클라이언트 등록
        connected_clients.add(websocket)
        self.client_count += 1
        client_id = self.client_count

        print(f"[WS] Client #{client_id} connected from {websocket.remote_address}")

        # 환영 메시지
        welcome = {
            "type": "welcome",
            "client_id": client_id,
            "message": "Connected to LOCUS WebSocket Bridge",
            "zmq_endpoint": ZMQ_ENDPOINT,
            "timestamp": datetime.now().isoformat()
        }
        await websocket.send(json.dumps(welcome))

        try:
            # 클라이언트로부터 메시지를 받기 (Ping/Pong)
            async for message in websocket:
                # 클라이언트가 보낸 메시지 처리 (필요시)
                try:
                    data = json.loads(message)
                    if data.get('type') == 'ping':
                        await websocket.send(json.dumps({
                            'type': 'pong',
                            'timestamp': datetime.now().isoformat()
                        }))
                except json.JSONDecodeError:
                    pass

        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            # 클라이언트 제거
            connected_clients.discard(websocket)
            print(f"[WS] Client #{client_id} disconnected")

    async def start(self):
        """브릿지 시작"""
        self.is_running = True

        print("="*60)
        print("WebSocket Bridge Starting...")
        print("="*60)

        # ZeroMQ 설정
        await self.setup_zmq()

        # WebSocket 서버 시작
        print(f"[WS] Starting WebSocket server on {WEBSOCKET_HOST}:{WEBSOCKET_PORT}")

        async with websockets.serve(
            self.websocket_handler,
            WEBSOCKET_HOST,
            WEBSOCKET_PORT
        ):
            print(f"[WS] WebSocket server running on ws://{WEBSOCKET_HOST}:{WEBSOCKET_PORT}")
            print("="*60)
            print("Ready to bridge ZeroMQ → WebSocket")
            print("Press Ctrl+C to stop")
            print("="*60)

            # ZeroMQ 리스너 시작
            listener_task = asyncio.create_task(self.zmq_listener())

            try:
                # 서버 실행 유지
                await asyncio.Future()
            except asyncio.CancelledError:
                pass
            finally:
                listener_task.cancel()
                await self.cleanup()

    async def cleanup(self):
        """리소스 정리"""
        print("\n[Bridge] Shutting down...")
        self.is_running = False

        if self.zmq_socket:
            self.zmq_socket.close()

        self.zmq_context.term()

        print("[Bridge] Cleanup complete")
        print(f"Total messages bridged: {self.message_count}")
        print(f"Total clients served: {self.client_count}")


async def main():
    """메인 함수"""
    bridge = WebSocketBridge()
    try:
        await bridge.start()
    except KeyboardInterrupt:
        print("\n[Main] Keyboard interrupt")
    except Exception as e:
        print(f"[Main] Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
