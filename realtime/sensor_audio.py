"""
실시간 데모 - 오디오 센서 (YAMNet + 17-class Head)
마이크로 소리를 녹음 후 YAMNet으로 17-class 분류하여 ZeroMQ로 전송합니다.
"""

import sys
import os

# 프로젝트 루트 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import sounddevice as sd
import zmq
import time
import numpy as np

# src로부터 YamnetProcessor 임포트
from src.audio_recognition.yamnet_processor import YamnetProcessor, AUDIO_CLASSES

# ZeroMQ 설정
ZMQ_ENDPOINT = "ipc:///tmp/locus_sensors.ipc"


class AudioSensor:
    """
    YAMNet + 17-class Head 기반 오디오 센서
    마이크로 소리를 녹음하고 YAMNet으로 17-class 분류 후 ZeroMQ로 전송합니다.
    """

    def __init__(self, sample_rate=16000):
        """
        오디오 센서를 초기화합니다.

        Args:
            sample_rate: 샘플링 레이트 (기본값: 16000Hz)
        """
        print("="*60)
        print("Audio Sensor (YAMNet 17-class) Initializing...")
        print("="*60)

        self.sample_rate = sample_rate

        # ZeroMQ Publisher 설정
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.PUB)
        self.zmq_socket.connect(ZMQ_ENDPOINT)
        print(f"ZeroMQ connected to {ZMQ_ENDPOINT}")

        # YAMNet 프로세서 로드
        print("Loading YAMNet processor...")
        self.yamnet_processor = YamnetProcessor()
        print("YAMNet processor ready!")

        # 마이크 테스트
        print("\nTesting microphone...")
        try:
            test_audio = sd.rec(int(0.1 * sample_rate),
                               samplerate=sample_rate,
                               channels=1,
                               blocking=True)
            print("Microphone working!")
        except Exception as e:
            raise RuntimeError(f"Microphone test failed: {e}")

        print("\nAudio Sensor ready!\n")

    def run(self, interval=1.0, duration=0.975):
        """
        센서의 메인 루프를 실행합니다.

        Args:
            interval: 데이터 전송 주기 (초)
            duration: 녹음 길이 (초). YAMNet은 0.975초(16kHz에서 15600 샘플)가 필요합니다.
        """
        print("Starting Audio Sensor loop...")
        print(f"  - Interval: {interval}s")
        print(f"  - Duration: {duration}s per recording (15600 samples for YAMNet)")
        print("  - Press Ctrl+C to quit\n")

        sample_count = 0

        try:
            while True:
                # 측정 시작 시점의 타임스탬프
                start_timestamp = time.time()

                # 오디오 녹음
                print(f"[{sample_count:04d}] Recording {duration}s audio...", end=" ", flush=True)

                audio = sd.rec(
                    int(duration * self.sample_rate),
                    samplerate=self.sample_rate,
                    channels=1,
                    blocking=True
                )

                # 1차원 배열로 변환
                audio = audio.flatten()

                # YAMNet 17-class 분류
                try:
                    # get_audio_embedding()은 17-class 확률 벡터를 반환
                    probs = self.yamnet_processor.get_audio_embedding(audio, self.sample_rate)

                    # 확률이 높은 상위 클래스 확인
                    top_sounds = self.yamnet_processor.get_top_sounds(
                        audio,
                        self.sample_rate,
                        top_k=3,
                        threshold=0.3
                    )

                    # ZeroMQ로 전송 (측정 시작 시점의 타임스탬프 사용)
                    message = {
                        'type': 'audio',
                        'data': probs,  # (17,) 크기의 확률 벡터
                        'timestamp': start_timestamp,
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
                    print(f"Error: {e}")

                sample_count += 1

                # 다음 주기까지 대기
                wait_time = max(0, interval - duration)
                if wait_time > 0:
                    time.sleep(wait_time)

        except KeyboardInterrupt:
            print("\nKeyboard interrupt, stopping...")

        finally:
            self.cleanup()

    def cleanup(self):
        """사용한 리소스를 정리합니다."""
        print("\nCleaning up Audio Sensor...")
        self.zmq_socket.close()
        self.zmq_context.term()
        print("Audio Sensor stopped!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Audio Sensor (YAMNet 17-class)")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Sensing interval in seconds (default: 1.0)")
    parser.add_argument("--duration", type=float, default=0.975,
                        help="Recording duration in seconds (default: 0.975 for YAMNet)")
    parser.add_argument("--sample-rate", type=int, default=16000,
                        help="Sample rate in Hz (default: 16000)")

    args = parser.parse_args()

    # 센서 시작
    sensor = AudioSensor(sample_rate=args.sample_rate)
    sensor.run(interval=args.interval, duration=args.duration)
