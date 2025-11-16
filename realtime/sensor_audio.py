"""
Realtime Demo - Audio Sensor (YAMNet + 17-class Head)
마이크로 소리 녹음 후 YAMNet으로 17-class 분류하여 ZeroMQ로 전송
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import sounddevice as sd
import zmq
import time
import numpy as np

# Import YamnetProcessor from src
from src.audio_recognition.yamnet_processor import YamnetProcessor, AUDIO_CLASSES

# ZeroMQ 설정
ZMQ_ENDPOINT = "ipc:///tmp/locus_sensors.ipc"


class AudioSensor:
    """
    YAMNet + 17-class Head 기반 Audio Sensor
    마이크로 소리를 녹음하고 YAMNet으로 17-class 분류 후 ZeroMQ로 전송
    """

    def __init__(self, sample_rate=16000):
        """
        Initialize Audio Sensor

        Args:
            sample_rate: 샘플링 레이트 (기본값: 16000Hz)
        """
        print("="*60)
        print("🎤 Audio Sensor (YAMNet 17-class) Initializing...")
        print("="*60)

        self.sample_rate = sample_rate

        # ZeroMQ Publisher 설정
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.PUB)
        self.zmq_socket.connect(ZMQ_ENDPOINT)
        print(f"✓ ZeroMQ connected to {ZMQ_ENDPOINT}")

        # YAMNet 프로세서 로드 (src/audio_recognition에서 임포트)
        print("Loading YAMNet processor...")
        self.yamnet_processor = YamnetProcessor()
        print("✓ YAMNet processor ready!")

        # 마이크 테스트
        print("\n🎤 Testing microphone...")
        try:
            test_audio = sd.rec(int(0.1 * sample_rate),
                               samplerate=sample_rate,
                               channels=1,
                               blocking=True)
            print("✓ Microphone working!")
        except Exception as e:
            raise RuntimeError(f"Microphone test failed: {e}")

        print("\n✅ Audio Sensor ready!\n")

    def run(self, interval=1.0, duration=1.0):
        """
        센서 실행 (메인 루프)

        Args:
            interval: 전송 주기 (초)
            duration: 녹음 길이 (초)
        """
        print("🚀 Starting Audio Sensor loop...")
        print(f"  - Interval: {interval}s")
        print(f"  - Duration: {duration}s per recording")
        print("  - Press Ctrl+C to quit\n")

        sample_count = 0

        try:
            while True:
                # 오디오 녹음
                print(f"[{sample_count:04d}] 🎤 Recording {duration}s audio...", end=" ", flush=True)

                audio = sd.rec(
                    int(duration * self.sample_rate),
                    samplerate=self.sample_rate,
                    channels=1,
                    blocking=True
                )

                # Flatten to 1D
                audio = audio.flatten()

                # YAMNet 17-class 분류
                try:
                    # get_audio_embedding()은 이제 17-class 확률 벡터를 반환
                    probs = self.yamnet_processor.get_audio_embedding(audio, self.sample_rate)

                    # 상위 클래스 확인
                    top_sounds = self.yamnet_processor.get_top_sounds(
                        audio,
                        self.sample_rate,
                        top_k=3,
                        threshold=0.3
                    )

                    # ZeroMQ 전송
                    message = {
                        'type': 'audio',
                        'data': probs,  # (17,) 확률 벡터
                        'timestamp': time.time(),
                        'sample_count': sample_count
                    }
                    self.zmq_socket.send_pyobj(message)

                    # 로그 출력
                    if top_sounds:
                        sounds_str = ", ".join([f"{name}({prob:.2f})" for name, prob in top_sounds])
                        print(f"→ ZMQ: {sounds_str}")
                    else:
                        print(f"→ ZMQ: (no significant sounds)")

                except Exception as e:
                    print(f"⚠ Error: {e}")

                sample_count += 1

                # 대기 (interval - duration)
                wait_time = max(0, interval - duration)
                if wait_time > 0:
                    time.sleep(wait_time)

        except KeyboardInterrupt:
            print("\n⚠ Keyboard interrupt, stopping...")

        finally:
            self.cleanup()

    def cleanup(self):
        """리소스 정리"""
        print("\n🧹 Cleaning up Audio Sensor...")
        self.zmq_socket.close()
        self.zmq_context.term()
        print("✓ Audio Sensor stopped!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Audio Sensor (YAMNet 17-class)")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Sensing interval in seconds (default: 1.0)")
    parser.add_argument("--duration", type=float, default=1.0,
                        help="Recording duration in seconds (default: 1.0)")
    parser.add_argument("--sample-rate", type=int, default=16000,
                        help="Sample rate in Hz (default: 16000)")

    args = parser.parse_args()

    # 센서 시작
    sensor = AudioSensor(sample_rate=args.sample_rate)
    sensor.run(interval=args.interval, duration=args.duration)
