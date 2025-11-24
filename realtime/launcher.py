"""
실시간 데모 - 런처
모든 센서와 예측기를 한 번에 실행하는 통합 스크립트입니다.
"""

import subprocess
import sys
import os
import time
import signal

# 프로세스 리스트
processes = []


def start_process(script_name, args=None):
    """
    지정된 파이썬 스크립트를 별도의 프로세스로 시작합니다.

    Args:
        script_name: 실행할 파이썬 스크립트 이름
        args: 스크립트에 전달할 추가 인자 리스트
    """
    realtime_dir = os.path.dirname(__file__)
    script_path = os.path.join(realtime_dir, script_name)

    cmd = [sys.executable, script_path]
    if args:
        cmd.extend(args)

    print(f"Starting: {script_name}")
    process = subprocess.Popen(cmd)
    processes.append((script_name, process))

    return process


def cleanup():
    """실행 중인 모든 자식 프로세스를 종료합니다."""
    print("\nCleaning up processes...")

    for name, process in processes:
        if process.poll() is None:  # 프로세스가 아직 실행 중인 경우
            print(f"  Terminating: {name}")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(f"  Force killing: {name}")
                process.kill()

    print("All processes stopped!")


def signal_handler(sig, frame):
    """Ctrl+C 인터럽트 신호를 처리하는 핸들러입니다."""
    print("\nReceived interrupt signal...")
    cleanup()
    sys.exit(0)


def main():
    """메인 실행 함수"""
    import argparse

    parser = argparse.ArgumentParser(description="Smart Vacuum Cleaner - Realtime Demo Launcher")
    parser.add_argument("--enable-tracker", action="store_true",
                        help="Enable LocationTracker WebSocket integration")
    parser.add_argument("--tracker-uri", type=str, default="ws://192.168.43.1:8080",
                        help="LocationTracker WebSocket URI (default: ws://192.168.43.1:8080)")
    args = parser.parse_args()

    print("="*60)
    print("Smart Vacuum Cleaner - Realtime Demo Launcher")
    print("="*60)
    print("\nThis script will start 4 processes:")
    print("  1. Visual Sensor (YOLO)")
    print("  2. Audio Sensor (YAMNet)")
    print("  3. Context Sensor (Spatial/Time/Pose)")
    print("  4. GRU Predictor")
    print("\nProcesses communicate via ZeroMQ (IPC):")
    print("  - Endpoint: ipc:///tmp/locus_sensors.ipc")
    print("  - Pattern: PUB/SUB (sensors publish, predictor subscribes)")

    if args.enable_tracker:
        print("\n[LocationTracker] Enabled")
        print(f"  - URI: {args.tracker_uri}")
        print("  - Zone will update automatically from iPhone ARKit")
    else:
        print("\n[LocationTracker] Disabled (manual zone control)")

    print("\nPress Ctrl+C to stop all processes.\n")

    # Ctrl+C 핸들러 등록
    signal.signal(signal.SIGINT, signal_handler)

    input("Press ENTER to start...")

    try:
        # 1. GRU 예측기를 먼저 시작합니다 (ZeroMQ BIND - 구독자가 먼저 바인드해야 함).
        print("\n[1/4] Starting GRU Predictor...")
        start_process("gru_predictor.py")
        time.sleep(3)  # 모델 로딩 및 ZeroMQ BIND 대기

        # 2. Visual Sensor
        print("\n[2/4] Starting Visual Sensor (YOLO)...")
        start_process("sensor_visual.py", ["--interval", "1.0"])
        time.sleep(2)

        # 3. Audio Sensor
        print("\n[3/4] Starting Audio Sensor (YAMNet)...")
        start_process("sensor_audio.py", ["--interval", "1.0", "--duration", "0.975"])
        time.sleep(2)

        # 4. Context Sensor
        print("\n[4/4] Starting Context Sensor...")
        zone = "living_room"  # 기본값 자동 설정
        context_args = ["--interval", "1.0", "--zone", zone]

        if args.enable_tracker:
            context_args.extend(["--enable-tracker", "--tracker-uri", args.tracker_uri])
            print(f"LocationTracker enabled: {args.tracker_uri}")
        else:
            print(f"Using default zone: {zone}")

        start_process("sensor_context.py", context_args)

        print("\n" + "="*60)
        print("All processes started successfully!")
        print("="*60)
        print("\nCollecting 30 timesteps of sensor data...")
        print("GRU prediction will run automatically after 30 timesteps.\n")
        print("Press Ctrl+C to stop all processes.\n")

        # 프로세스 모니터링
        while True:
            time.sleep(1)

            # 프로세스가 비정상적으로 종료되었는지 확인
            for name, process in processes:
                if process.poll() is not None:
                    print(f"\nWarning: {name} stopped unexpectedly!")
                    cleanup()
                    sys.exit(1)

    except KeyboardInterrupt:
        pass

    finally:
        cleanup()


if __name__ == "__main__":
    main()
