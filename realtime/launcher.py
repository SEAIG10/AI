"""
Realtime Demo - Launcher
모든 센서와 예측기를 한 번에 실행하는 통합 스크립트
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
    센서 프로세스 시작

    Args:
        script_name: Python 스크립트 이름
        args: 추가 인자
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
    """모든 프로세스 종료"""
    print("\n🧹 Cleaning up processes...")

    for name, process in processes:
        if process.poll() is None:  # Still running
            print(f"  Terminating: {name}")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(f"  Force killing: {name}")
                process.kill()

    print("✓ All processes stopped!")


def signal_handler(sig, frame):
    """Ctrl+C 핸들러"""
    print("\n⚠ Received interrupt signal...")
    cleanup()
    sys.exit(0)


def main():
    """메인 실행 함수"""
    print("="*60)
    print("🚀 Smart Vacuum Cleaner - Realtime Demo Launcher")
    print("="*60)
    print("\nThis script will start 4 processes:")
    print("  1. Visual Sensor (YOLO)")
    print("  2. Audio Sensor (YAMNet)")
    print("  3. Context Sensor (Spatial/Time/Pose)")
    print("  4. GRU Predictor")
    print("\nMake sure MQTT broker (mosquitto) is running!")
    print("  - macOS: brew services start mosquitto")
    print("  - Linux: sudo systemctl start mosquitto")
    print("\nPress Ctrl+C to stop all processes.\n")

    # Ctrl+C 핸들러 등록
    signal.signal(signal.SIGINT, signal_handler)

    input("Press ENTER to start...")

    try:
        # 1. GRU Predictor 먼저 시작 (MQTT 메시지 수신 준비)
        print("\n[1/4] Starting GRU Predictor...")
        start_process("gru_predictor.py")
        time.sleep(3)  # 모델 로딩 대기

        # 2. Visual Sensor
        print("\n[2/4] Starting Visual Sensor (YOLO)...")
        start_process("sensor_visual.py", ["--interval", "1.0"])
        time.sleep(2)

        # 3. Audio Sensor
        print("\n[3/4] Starting Audio Sensor (YAMNet)...")
        start_process("sensor_audio.py", ["--interval", "1.0", "--duration", "1.0"])
        time.sleep(2)

        # 4. Context Sensor
        print("\n[4/4] Starting Context Sensor...")
        zone = input("Enter initial zone (default: living_room): ").strip()
        if not zone:
            zone = "living_room"
        start_process("sensor_context.py", ["--interval", "1.0", "--zone", zone])

        print("\n" + "="*60)
        print("✅ All processes started successfully!")
        print("="*60)
        print("\n🎥 Collecting 30 timesteps of sensor data...")
        print("🧠 GRU prediction will run automatically after 30 timesteps.\n")
        print("Press Ctrl+C to stop all processes.\n")

        # 프로세스 모니터링
        while True:
            time.sleep(1)

            # 프로세스가 비정상 종료되었는지 확인
            for name, process in processes:
                if process.poll() is not None:
                    print(f"\n⚠ Warning: {name} stopped unexpectedly!")
                    cleanup()
                    sys.exit(1)

    except KeyboardInterrupt:
        pass

    finally:
        cleanup()


if __name__ == "__main__":
    main()
