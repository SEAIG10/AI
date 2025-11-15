"""
FR2.2: Audio Event Detection using YAMNet + 17-class Head
Based on friend's YAMNet implementation for home sound classification
"""

import numpy as np
import os
from pathlib import Path
from tensorflow.lite.python.interpreter import Interpreter


# 17개 오디오 클래스 (친구의 YAMNet Head 기준)
AUDIO_CLASSES = [
    "door", "dishes", "cutlery", "chopping", "frying", "microwave", "blender",
    "water_tap", "sink", "toilet_flush", "telephone", "chewing", "speech",
    "television", "footsteps", "vacuum", "hair_dryer"
]


class YamnetProcessor:
    """
    FR2.2: Sound Event Detection (SED)
    Uses YAMNet backbone + 17-class custom head for home sound classification

    Architecture:
        Audio (16kHz) → YAMNet backbone → 1024-dim embedding → Head → 17-class probabilities
    """

    def __init__(self, yamnet_path=None, head_path=None):
        """
        Initialize YAMNet + Head processor

        Args:
            yamnet_path: Path to yamnet.tflite (백본)
            head_path: Path to head_1024_fp16.tflite (17-class 분류기)
        """
        self.sample_rate = 16000  # YAMNet standard
        self.num_classes = 17

        # 기본 경로 설정
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
        """Load YAMNet backbone and 17-class head"""
        try:
            # YAMNet 백본 로드
            if not os.path.exists(self.yamnet_path):
                raise FileNotFoundError(f"YAMNet model not found: {self.yamnet_path}")

            self.yamnet = Interpreter(model_path=self.yamnet_path, num_threads=2)
            self.yamnet.allocate_tensors()

            self.yam_in_details = self.yamnet.get_input_details()[0]
            self.yam_in_idx = self.yam_in_details["index"]
            self.yam_out_details = self.yamnet.get_output_details()
            self.yam_emb_idx = 1  # output[1] = 1024-dim embedding

            print(f"✓ YAMNet backbone loaded from {self.yamnet_path}")

            # Head 분류기 로드
            if not os.path.exists(self.head_path):
                raise FileNotFoundError(f"Head model not found: {self.head_path}")

            self.head = Interpreter(model_path=self.head_path, num_threads=2)
            self.head.allocate_tensors()

            self.head_in_details = self.head.get_input_details()[0]
            self.head_in_idx = self.head_in_details["index"]
            self.head_out_idx = self.head.get_output_details()[0]["index"]

            print(f"✓ 17-class Head loaded from {self.head_path}")
            print(f"  Classes: {', '.join(AUDIO_CLASSES[:5])}... (17 total)")

            self.model_loaded = True

        except Exception as e:
            print(f"Warning: Could not load YAMNet models: {e}")
            print("  Will use simulated audio features instead.")
            self.model_loaded = False

    def get_audio_embedding(self, audio_buffer, sample_rate=None):
        """
        FR2.2: Extract 17-dimensional audio classification probabilities

        NOTE: This replaces the old 256-dim embedding approach!
        Now returns 17-class probabilities instead of embeddings.

        Args:
            audio_buffer: Raw audio samples (numpy array, mono)
            sample_rate: Sample rate (default: 16000)

        Returns:
            np.ndarray: 17-dimensional audio class probabilities (float32)
                        Returns zeros if audio is None or model fails
        """
        if audio_buffer is None or len(audio_buffer) == 0:
            # Return zero probabilities for silence
            return np.zeros(self.num_classes, dtype=np.float32)

        # 샘플레이트 체크
        if sample_rate is not None and sample_rate != self.sample_rate:
            print(f"Warning: Expected {self.sample_rate}Hz, got {sample_rate}Hz")
            # 리샘플링 필요하면 librosa 사용
            try:
                import librosa
                audio_buffer = librosa.resample(
                    audio_buffer,
                    orig_sr=sample_rate,
                    target_sr=self.sample_rate
                )
            except ImportError:
                print("Warning: librosa not available for resampling")

        if not self.model_loaded:
            # 모델 없으면 시뮬레이션
            return self._simulate_audio_classification(audio_buffer)

        try:
            # YAMNet 백본 추론 (audio → 1024-dim embedding)
            audio_float32 = audio_buffer.astype(np.float32)

            self.yamnet.set_tensor(self.yam_in_idx, audio_float32)
            self.yamnet.invoke()
            emb = self.yamnet.get_tensor(self.yam_out_details[self.yam_emb_idx]["index"])

            # Embedding이 (T, 1024) 형태일 수 있으므로 평균
            if emb.ndim == 2:
                emb_vec = emb.mean(axis=0)  # (1024,)
            else:
                emb_vec = emb.flatten()

            # Head 입력 shape에 맞게 조정
            head_shape = self.head_in_details['shape']
            if len(head_shape) == 2:
                # (1, 1024) 형태
                emb_input = emb_vec[np.newaxis, :].astype(np.float32)
            else:
                # (1024,) 형태
                emb_input = emb_vec.astype(np.float32)

            # Head 추론 (1024-dim → 17-class)
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
        Simulate 17-class audio probabilities based on audio energy
        (For testing before models are fully integrated)

        Returns:
            np.ndarray: Simulated 17-dimensional probabilities
        """
        # RMS 에너지 계산
        rms = np.sqrt(np.mean(audio_buffer ** 2))

        # 랜덤 확률 생성 (에너지 기반)
        probs = np.random.rand(self.num_classes).astype(np.float32) * rms * 10

        # Sigmoid로 [0, 1] 범위로 변환
        probs = 1.0 / (1.0 + np.exp(-probs))

        # 약간의 sparsity 추가 (대부분 낮은 확률)
        probs = probs * 0.1

        return probs

    def get_top_sounds(self, audio_buffer, sample_rate=None, top_k=3, threshold=0.3):
        """
        Get top-k detected sounds above threshold

        Args:
            audio_buffer: Audio samples
            sample_rate: Sample rate
            top_k: Number of top predictions to return
            threshold: Minimum probability threshold

        Returns:
            list of (class_name, probability) tuples
        """
        probs = self.get_audio_embedding(audio_buffer, sample_rate)

        # Top-k 추출
        top_indices = np.argsort(-probs)[:top_k]

        results = []
        for idx in top_indices:
            if probs[idx] >= threshold:
                results.append((AUDIO_CLASSES[idx], float(probs[idx])))

        return results


def test_yamnet_processor():
    """Test YAMNet processor"""
    print("=" * 60)
    print("Testing FR2.2: YAMNet 17-class Audio Processor")
    print("=" * 60)

    processor = YamnetProcessor()

    # Simulate audio buffer (1 second of noise)
    sample_rate = 16000
    duration = 1.0
    audio_buffer = np.random.randn(int(sample_rate * duration)).astype(np.float32) * 0.1

    print(f"\nTest audio: {len(audio_buffer)} samples @ {sample_rate}Hz")

    # Test classification
    probs = processor.get_audio_embedding(audio_buffer, sample_rate)

    print(f"\nExtracted audio probabilities:")
    print(f"  Shape: {probs.shape}")
    print(f"  Dtype: {probs.dtype}")
    print(f"  Range: [{probs.min():.3f}, {probs.max():.3f}]")
    print(f"  Sum: {probs.sum():.3f}")

    # Top sounds
    top_sounds = processor.get_top_sounds(audio_buffer, sample_rate, top_k=5, threshold=0.0)
    print(f"\nTop 5 detected sounds:")
    for sound, prob in top_sounds:
        print(f"  {sound:<15} {prob:.3f}")

    # Test silence
    silence_probs = processor.get_audio_embedding(None)
    print(f"\nSilence probabilities (all zeros): {np.all(silence_probs == 0)}")

    print("\n" + "=" * 60)
    print("FR2.2 Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_yamnet_processor()
