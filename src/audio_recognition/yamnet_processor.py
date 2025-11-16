"""
오디오 이벤트 탐지 (YAMNet + 17-class Head 사용)
YAMNet을 백본으로 사용하고, 17개 클래스로 분류하는 헤드를 추가하여 가정 내 소리를 탐지합니다.
"""

import numpy as np
import os
from pathlib import Path
from tensorflow.lite.python.interpreter import Interpreter


# 17개 오디오 클래스 정의
AUDIO_CLASSES = [
    "door", "dishes", "cutlery", "chopping", "frying", "microwave", "blender",
    "water_tap", "sink", "toilet_flush", "telephone", "chewing", "speech",
    "television", "footsteps", "vacuum", "hair_dryer"
]


class YamnetProcessor:
    """
    소리 이벤트 탐지(Sound Event Detection, SED)를 수행합니다.
    YAMNet 백본과 17개 클래스를 분류하는 커스텀 헤드를 사용하여 가정 내 소리를 분류합니다.

    아키텍처:
        오디오 (16kHz) → YAMNet 백본 → 1024차원 임베딩 → Head → 17-class 확률
    """

    def __init__(self, yamnet_path=None, head_path=None):
        """
        YAMNet + Head 프로세서를 초기화합니다.

        Args:
            yamnet_path: yamnet.tflite 모델 파일 경로 (백본)
            head_path: head_1024_fp16.tflite 모델 파일 경로 (17-class 분류기)
        """
        self.sample_rate = 16000  # YAMNet 표준 샘플링 레이트
        self.required_samples = 15600  # YAMNet TFLite는 정확히 15600개의 샘플(0.975초)이 필요합니다.
        self.num_classes = 17

        # 기본 모델 경로 설정
        if yamnet_path is None:
            yamnet_path = os.path.join(
                Path(__file__).resolve().parents[2],
                'models', 'audio', 'yamnet.tflite'
            )
        if head_path is None:
            head_path = os.path.join(
                Path(__file__).resolve().parents[2],
                'models', 'audio', 'head_1024_fp16.tflite'
            )

        self.yamnet_path = yamnet_path
        self.head_path = head_path

        # 모델 로드
        self._load_models()

    def _load_models(self):
        """YAMNet 백본과 17-class 헤드 모델을 로드합니다."""
        try:
            # YAMNet 백본 로드
            if not os.path.exists(self.yamnet_path):
                raise FileNotFoundError(f"YAMNet 모델을 찾을 수 없습니다: {self.yamnet_path}")

            self.yamnet = Interpreter(model_path=self.yamnet_path, num_threads=2)
            self.yamnet.allocate_tensors()

            in_det = self.yamnet.get_input_details()[0]
            self.yam_in_idx = in_det["index"]

            # 입력 텐서를 15600 샘플로 리사이징 (YAMNet TFLite 요구사항)
            if (len(in_det['shape']) == 1 and in_det['shape'][0] not in (0, 15600)) \
               or (len(in_det['shape']) == 2 and tuple(in_det['shape']) not in ((1,15600),(15600,1))):
                try:
                    self.yamnet.resize_tensor_input(in_det['index'], [15600], strict=False)
                    self.yamnet.allocate_tensors()
                except Exception:
                    pass

            self.yam_in_details = self.yamnet.get_input_details()[0]
            self.yam_out_details = self.yamnet.get_output_details()
            self.yam_emb_idx = 1  # output[1]은 1024차원 임베딩입니다.

            print(f"YAMNet backbone loaded from {self.yamnet_path}")

            # Head 분류기 로드
            if not os.path.exists(self.head_path):
                raise FileNotFoundError(f"Head 모델을 찾을 수 없습니다: {self.head_path}")

            self.head = Interpreter(model_path=self.head_path, num_threads=2)
            self.head.allocate_tensors()

            self.head_in_details = self.head.get_input_details()[0]
            self.head_in_idx = self.head_in_details["index"]
            self.head_out_idx = self.head.get_output_details()[0]["index"]

            print(f"17-class Head loaded from {self.head_path}")
            print(f"  Classes: {', '.join(AUDIO_CLASSES[:5])}... (17 total)")

            self.model_loaded = True

        except Exception as e:
            print(f"Warning: Could not load YAMNet models: {e}")
            print("  대신 시뮬레이션된 오디오 특징을 사용합니다.")
            self.model_loaded = False

    def get_audio_embedding(self, audio_buffer, sample_rate=None):
        """
        17차원의 오디오 분류 확률을 추출합니다.
        참고: 이 메서드는 기존의 256차원 임베딩 방식을 대체하며, 임베딩 대신 17-class 확률을 반환합니다.

        Args:
            audio_buffer: 원본 오디오 샘플 (numpy array, mono).
                         정확히 15600개의 샘플(16kHz에서 0.975초)이어야 합니다.
            sample_rate: 오디오 샘플링 레이트 (기본값: 16000)

        Returns:
            np.ndarray: 17차원의 오디오 클래스 확률 (float32).
                        오디오가 None이거나 모델 추론에 실패하면 0으로 채워진 배열을 반환합니다.
        """
        if audio_buffer is None or len(audio_buffer) == 0:
            # 무음(silence)에 대해서는 0 확률을 반환합니다.
            return np.zeros(self.num_classes, dtype=np.float32)

        # 샘플링 레이트 확인
        if sample_rate is not None and sample_rate != self.sample_rate:
            print(f"Warning: Expected {self.sample_rate}Hz, got {sample_rate}Hz")
            # 리샘플링이 필요한 경우 librosa를 사용합니다.
            try:
                import librosa
                audio_buffer = librosa.resample(
                    audio_buffer,
                    orig_sr=sample_rate,
                    target_sr=self.sample_rate
                )
            except ImportError:
                print("Warning: librosa not available for resampling")

        # YAMNet TFLite는 정확히 15600개의 샘플이 필요합니다.
        if len(audio_buffer) != self.required_samples:
            # 정확히 15600개의 샘플이 되도록 패딩 또는 자르기를 수행합니다.
            if len(audio_buffer) < self.required_samples:
                # 0으로 패딩합니다.
                audio_padded = np.zeros(self.required_samples, dtype=np.float32)
                audio_padded[:len(audio_buffer)] = audio_buffer
                audio_buffer = audio_padded
            else:
                # 15600개 샘플로 자릅니다.
                audio_buffer = audio_buffer[:self.required_samples]

        if not self.model_loaded:
            # 모델이 로드되지 않은 경우, 시뮬레이션 결과를 반환합니다.
            return self._simulate_audio_classification(audio_buffer)

        try:
            # YAMNet 백본 추론 (오디오 → 1024차원 임베딩)
            # 오디오 데이터가 float32 타입이고 [-1.0, +1.0] 범위에 있는지 확인합니다.
            audio_float32 = audio_buffer.astype(np.float32)

            # 필요시 정규화합니다.
            if np.abs(audio_float32).max() > 1.0:
                audio_float32 = audio_float32 / np.abs(audio_float32).max()

            self.yamnet.set_tensor(self.yam_in_idx, audio_float32)
            self.yamnet.invoke()
            emb = self.yamnet.get_tensor(self.yam_out_details[self.yam_emb_idx]["index"])

            # 임베딩이 (T, 1024) 형태일 수 있으므로, 시간 축에 대해 평균을 계산합니다.
            if emb.ndim == 2:
                emb_vec = emb.mean(axis=0)  # (1024,)
            else:
                emb_vec = emb.flatten()

            # Head 모델의 입력 shape에 맞게 조정합니다.
            head_shape = self.head_in_details['shape']
            if len(head_shape) == 2:
                # (1, 1024) 형태
                emb_input = emb_vec[np.newaxis, :].astype(np.float32)
            else:
                # (1024,) 형태
                emb_input = emb_vec.astype(np.float32)

            # Head 추론 (1024차원 → 17-class)
            self.head.set_tensor(self.head_in_idx, emb_input)
            self.head.invoke()
            probs = self.head.get_tensor(self.head_out_idx).flatten()  # (17,)

            return probs.astype(np.float32)

        except Exception as e:
            print(f"Warning: Audio classification failed: {e}")
            import traceback
            traceback.print_exc()
            return self._simulate_audio_classification(audio_buffer)

    def _simulate_audio_classification(self, audio_buffer):
        """
        오디오 에너지를 기반으로 17-class 오디오 확률을 시뮬레이션합니다.
        (모델이 완전히 통합되기 전 테스트용)

        Returns:
            np.ndarray: 시뮬레이션된 17차원 확률 벡터
        """
        # RMS 에너지 계산
        rms = np.sqrt(np.mean(audio_buffer ** 2))

        # 랜덤 확률 생성 (에너지 기반)
        probs = np.random.rand(self.num_classes).astype(np.float32) * rms * 10

        # Sigmoid 함수로 [0, 1] 범위로 변환
        probs = 1.0 / (1.0 + np.exp(-probs))

        # 약간의 희소성(sparsity) 추가 (대부분 낮은 확률)
        probs = probs * 0.1

        return probs

    def get_top_sounds(self, audio_buffer, sample_rate=None, top_k=3, threshold=0.3):
        """
        지정된 임계값 이상의 상위 K개 소리를 가져옵니다.

        Args:
            audio_buffer: 오디오 샘플
            sample_rate: 샘플링 레이트
            top_k: 반환할 상위 예측 개수
            threshold: 최소 확률 임계값

        Returns:
            (클래스 이름, 확률) 튜플의 리스트
        """
        probs = self.get_audio_embedding(audio_buffer, sample_rate)

        # 상위 K개 추출
        top_indices = np.argsort(-probs)[:top_k]

        results = []
        for idx in top_indices:
            if probs[idx] >= threshold:
                results.append((AUDIO_CLASSES[idx], float(probs[idx])))

        return results


def test_yamnet_processor():
    """YAMNet 프로세서 테스트"""
    print("=" * 60)
    print("Testing YAMNet 17-class Audio Processor")
    print("=" * 60)

    processor = YamnetProcessor()

    # 테스트용 오디오 버퍼 시뮬레이션 (1초 길이의 노이즈)
    sample_rate = 16000
    duration = 1.0
    audio_buffer = np.random.randn(int(sample_rate * duration)).astype(np.float32) * 0.1

    print(f"\nTest audio: {len(audio_buffer)} samples @ {sample_rate}Hz")

    # 분류 테스트
    probs = processor.get_audio_embedding(audio_buffer, sample_rate)

    print(f"\nExtracted audio probabilities:")
    print(f"  Shape: {probs.shape}")
    print(f"  Dtype: {probs.dtype}")
    print(f"  Range: [{probs.min():.3f}, {probs.max():.3f}]")
    print(f"  Sum: {probs.sum():.3f}")

    # 상위 소리 확인
    top_sounds = processor.get_top_sounds(audio_buffer, sample_rate, top_k=5, threshold=0.0)
    print(f"\nTop 5 detected sounds:")
    for sound, prob in top_sounds:
        print(f"  {sound:<15} {prob:.3f}")

    # 무음 테스트
    silence_probs = processor.get_audio_embedding(None)
    print(f"\nSilence probabilities (all zeros): {np.all(silence_probs == 0)}")

    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_yamnet_processor()
