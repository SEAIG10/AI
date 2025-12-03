"""
통합 런처 - 전체 시스템 실행
모든 센서, 예측기, WebSocket 브릿지, YOLO 비디오 서버를 한 번에 실행합니다.
"""

import subprocess
import sys
import os
import time
import signal

# 프로세스 리스트
processes = []


def start_process(script_name, args=None, name=None):
    """
    지정된 파이썬 스크립트를 별도의 프로세스로 시작합니다.

    Args:
        script_name: 실행할 파이썬 스크립트 이름
        args: 스크립트에 전달할 추가 인자 리스트
        name: 표시용 이름 (None이면 script_name 사용)
    """
    realtime_dir = os.path.dirname(__file__)
    script_path = os.path.join(realtime_dir, script_name)

    cmd = [sys.executable, script_path]
    if args:
        cmd.extend(args)

    display_name = name or script_name
    print(f"  Starting: {display_name}")
    process = subprocess.Popen(cmd)
    processes.append((display_name, process))

    return process


def cleanup():
    """실행 중인 모든 자식 프로세스를 종료합니다."""
    print("\n" + "="*60)
    print("Shutting down all processes...")
    print("="*60)

    for name, process in processes:
        if process.poll() is None:  # 프로세스가 아직 실행 중인 경우
            print(f"  ⏹  Stopping: {name}")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(f"  ⚠️  Force killing: {name}")
                process.kill()

    print("\n✅ All processes stopped!")


def signal_handler(sig, frame):
    """Ctrl+C 인터럽트 신호를 처리하는 핸들러입니다."""
    print("\n\n⚠️  Received interrupt signal (Ctrl+C)...")
    cleanup()
    sys.exit(0)


def main():
    """메인 실행 함수"""
    import argparse

    parser = argparse.ArgumentParser(description="LOCUS Full System Launcher")
    parser.add_argument("--enable-tracker", action="store_true",
                        help="Enable LocationTracker WebSocket integration")
    parser.add_argument("--tracker-uri", type=str, default="ws://192.168.43.1:8080",
                        help="LocationTracker WebSocket URI (default: ws://192.168.43.1:8080)")
    parser.add_argument("--zone", type=str, default="living_room",
                        help="Default zone (default: living_room)")
    args = parser.parse_args()

    print("="*60)
    print("🚀 LOCUS AI Cleaning System - Full Launcher")
    print("="*60)
    print("\nThis script will start 5 processes:")
    print("  1. GRU Predictor (ML inference)")
    print("  2. Visual Sensor (YOLOv11n + YOLOv11n-pose)")
    print("     └─ Video stream: http://localhost:5001/video_feed")
    print("  3. Audio Sensor (YAMNet)")
    print("  4. Context Sensor (Spatial/Time/Pose)")
    print("  5. WebSocket Bridge (Dashboard communication)")
    print("     └─ WebSocket server: ws://localhost:8080")
    print("\nProcesses communicate via ZeroMQ (IPC):")
    print("  - Sensors → GRU: ipc:///tmp/locus_sensors.ipc")
    print("  - GRU → Bridge: ipc:///tmp/locus_bridge.ipc")

    if args.enable_tracker:
        print(f"\n📍 LocationTracker: Enabled ({args.tracker_uri})")
    else:
        print(f"\n📍 LocationTracker: Disabled (using default zone: {args.zone})")

    print("\n⚠️  Press Ctrl+C to stop all processes.\n")
    print("="*60)

    # Ctrl+C 핸들러 등록
    signal.signal(signal.SIGINT, signal_handler)

    try:
        print("\n🔧 Starting processes...\n")

        # 1. GRU 예측기를 먼저 시작 (ZeroMQ BIND - 구독자가 먼저 바인드)
        print("[1/5] GRU Predictor")
        start_process("gru_predictor.py", name="GRU Predictor")
        time.sleep(3)  # 모델 로딩 및 ZeroMQ BIND 대기

        # 2. Visual Sensor (YOLO + Flask 비디오 서버)
        print("\n[2/5] Visual Sensor (YOLO)")
        start_process("sensor_visual.py", ["--interval", "1.0"], name="Visual Sensor (YOLO + Flask)")
        time.sleep(2)

        # 3. Audio Sensor
        print("\n[3/5] Audio Sensor (YAMNet)")
        start_process("sensor_audio.py", ["--interval", "1.0", "--duration", "0.975"], name="Audio Sensor (YAMNet)")
        time.sleep(2)

        # 4. Context Sensor
        print("\n[4/5] Context Sensor")
        context_args = [
            "--interval", "1.0",
            "--zone", args.zone,
            "--home-id", "1",
            "--mqtt-broker", "43.200.178.189"
        ]

        if args.enable_tracker:
            context_args.extend(["--enable-tracker", "--tracker-uri", args.tracker_uri])

        start_process("sensor_context.py", context_args, name="Context Sensor")
        time.sleep(2)

        # 5. WebSocket Bridge
        print("\n[5/5] WebSocket Bridge")
        start_process("websocket_bridge.py", name="WebSocket Bridge (ZMQ→WS)")
        time.sleep(2)

        print("\n" + "="*60)
        print("✅ All processes started successfully!")
        print("="*60)
        print("\n📊 Dashboard: http://localhost:3001")
        print("📹 Video Feed: http://localhost:5001/video_feed")
        print("🔌 WebSocket: ws://localhost:8080")
        print("\n⏳ Collecting 30 timesteps before first GRU prediction...")
        print("\n⚠️  Press Ctrl+C to stop all processes.\n")

        # 프로세스 모니터링
        while True:
            time.sleep(1)

            # 프로세스가 비정상적으로 종료되었는지 확인
            for name, process in processes:
                if process.poll() is not None:
                    print(f"\n❌ Warning: {name} stopped unexpectedly!")
                    cleanup()
                    sys.exit(1)

    except KeyboardInterrupt:
        pass

    finally:
        cleanup()


if __name__ == "__main__":
    main()
