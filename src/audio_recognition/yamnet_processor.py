"""
FR2.2: Audio Event Detection using Yamnet-256
Lightweight audio classification for on-device processing
"""

import numpy as np
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("Warning: librosa not installed. Audio processing will be limited.")


class YamnetProcessor:
    """
    FR2.2: Sound Event Detection (SED)
    Uses Yamnet-256 for lightweight audio event classification
    """

    def __init__(self, model_path=None):
        """
        Initialize Yamnet-256 processor

        Args:
            model_path: Path to Yamnet-256 TFLite model (optional)
        """
        self.model_path = model_path
        self.model = None
        self.sample_rate = 16000  # YAMNet standard
        self.embedding_dim = 256  # YAMNet-256 embedding dimension

        # Load model if path provided
        if model_path:
            self._load_model(model_path)
        else:
            print("Warning: No model path provided. Using simulated audio embeddings.")

    def _load_model(self, model_path):
        """Load Yamnet-256 TFLite model"""
        try:
            import tensorflow as tf
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()

            # Get input/output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

            # Find 256-dim embedding tensor (not the classification output)
            # The embedding is at an intermediate layer before final classification
            tensor_details = self.interpreter.get_tensor_details()
            self.embedding_tensor_index = None

            for tensor in tensor_details:
                shape = tensor['shape']
                # Look for (1, 256) shaped tensor - this is the embedding
                if len(shape) == 2 and shape[0] == 1 and shape[1] == 256:
                    self.embedding_tensor_index = tensor['index']
                    self.embedding_tensor_details = tensor
                    break

            # Re-create interpreter with custom output for embedding extraction
            if self.embedding_tensor_index is not None:
                # Create signature runner for flexible output access
                # Note: TFLite doesn't easily allow intermediate tensor access,
                # so we'll use a workaround with tensors allocated after invoke()
                pass

            print(f"✓ Yamnet-256 model loaded from {model_path}")
            print(f"  Input shape: {self.input_details[0]['shape']}")
            print(f"  Output shape: {self.output_details[0]['shape']}")
            if self.embedding_tensor_index is not None:
                print(f"  Embedding tensor index: {self.embedding_tensor_index}")
                print(f"  Embedding shape: {self.embedding_tensor_details['shape']}")
            else:
                print(f"  Warning: Could not find 256-dim embedding tensor!")

            self.model = True
        except Exception as e:
            print(f"Warning: Could not load Yamnet-256 model: {e}")
            self.model = None

    def preprocess_audio(self, audio_buffer, sample_rate=None):
        """
        Convert audio buffer to mel-spectrogram for YAMNet-256

        YAMNet-256 spec (STMicroelectronics):
        - 64 mels, 96 frames
        - FFT window: 25ms
        - Hop length: 10ms
        - Frequency range: 125-7500 Hz

        Args:
            audio_buffer: Raw audio samples (numpy array)
            sample_rate: Sample rate of audio (default: 16000)

        Returns:
            mel_spectrogram: (1, 64, 96, 1) mel-spectrogram patch (with batch dim)
        """
        if not LIBROSA_AVAILABLE:
            return None

        if sample_rate is None:
            sample_rate = self.sample_rate

        # Resample if necessary
        if sample_rate != self.sample_rate:
            audio_buffer = librosa.resample(
                audio_buffer,
                orig_sr=sample_rate,
                target_sr=self.sample_rate
            )

        # YAMNet-256 parameters
        n_fft = int(0.025 * self.sample_rate)  # 25ms window = 400 samples @ 16kHz
        hop_length = int(0.010 * self.sample_rate)  # 10ms hop = 160 samples @ 16kHz

        # Compute mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio_buffer,
            sr=self.sample_rate,
            n_mels=64,
            n_fft=n_fft,
            hop_length=hop_length,
            fmin=125,  # Min frequency
            fmax=7500  # Max frequency
        )

        # Convert to log scale (dB)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        # Extract 96-frame patch (center crop if longer, pad if shorter)
        if log_mel_spec.shape[1] >= 96:
            start_frame = (log_mel_spec.shape[1] - 96) // 2
            mel_patch = log_mel_spec[:, start_frame:start_frame + 96]
        else:
            # Pad with zeros if too short
            pad_width = ((0, 0), (0, 96 - log_mel_spec.shape[1]))
            mel_patch = np.pad(log_mel_spec, pad_width, mode='constant')

        # Normalize to [0, 1] range (typical for neural networks)
        mel_patch = (mel_patch - mel_patch.min()) / (mel_patch.max() - mel_patch.min() + 1e-6)

        # Add channel dimension: (64, 96) → (64, 96, 1)
        mel_patch = mel_patch[..., np.newaxis]

        return mel_patch.astype(np.float32)

    def get_audio_embedding(self, audio_buffer, sample_rate=None):
        """
        FR2.2: Extract 256-dimensional audio embedding from YAMNet-256

        Args:
            audio_buffer: Raw audio samples (numpy array)
            sample_rate: Sample rate (default: 16000)

        Returns:
            np.ndarray: 256-dimensional audio embedding (float32)
                        Returns zeros if audio is None or model fails
        """
        if audio_buffer is None or len(audio_buffer) == 0:
            # Return zero embedding for silence
            return np.zeros(self.embedding_dim, dtype=np.float32)

        # Preprocess audio
        mel_spec = self.preprocess_audio(audio_buffer, sample_rate)

        if mel_spec is None:
            # Fallback: Simulate audio embedding
            return self._simulate_audio_embedding(audio_buffer)

        if self.model is None:
            # No model loaded, use simulation
            return self._simulate_audio_embedding(audio_buffer)

        try:
            # Add batch dimension if not present: (64, 96, 1) → (1, 64, 96, 1)
            if mel_spec.ndim == 3:
                mel_spec = mel_spec[np.newaxis, ...]

            # Check if model expects INT8 input (quantized model)
            input_dtype = self.input_details[0]['dtype']
            if input_dtype == np.int8:
                # Quantize input: float32 [0, 1] → int8 [-128, 127]
                # Get quantization parameters
                input_scale = self.input_details[0]['quantization'][0]
                input_zero_point = self.input_details[0]['quantization'][1]

                # Quantize: q = f / scale + zero_point
                mel_spec_quantized = (mel_spec / input_scale + input_zero_point).astype(np.int8)
                input_data = mel_spec_quantized
            else:
                input_data = mel_spec

            # Run inference
            self.interpreter.set_tensor(
                self.input_details[0]['index'],
                input_data
            )
            self.interpreter.invoke()

            # Get 256-dim embedding from intermediate layer (not final classification output)
            if self.embedding_tensor_index is None:
                raise ValueError("Embedding tensor not found in model")

            # TFLite workaround: Access intermediate tensor data directly from buffer
            # After invoke(), the tensor buffer should contain the computed values
            try:
                embedding = self.interpreter.get_tensor(self.embedding_tensor_index)
            except ValueError:
                # If get_tensor doesn't work, try accessing via internal API
                tensor = self.interpreter._get_tensor_details(self.embedding_tensor_index)
                embedding = self.interpreter._interpreter.tensor(self.interpreter._interpreter, self.embedding_tensor_index)

            # Dequantize output if needed
            embedding_dtype = self.embedding_tensor_details['dtype']
            if embedding_dtype == np.int8:
                # Dequantize: f = (q - zero_point) * scale
                embedding_scale = self.embedding_tensor_details['quantization'][0]
                embedding_zero_point = self.embedding_tensor_details['quantization'][1]
                embedding = (embedding.astype(np.float32) - embedding_zero_point) * embedding_scale

            # Remove batch dimension and return: (1, 256) → (256,)
            if embedding.ndim == 2:
                embedding = embedding[0]

            return embedding.astype(np.float32)

        except Exception as e:
            print(f"Warning: Audio embedding extraction failed: {e}")
            import traceback
            traceback.print_exc()
            return self._simulate_audio_embedding(audio_buffer)

    def _simulate_audio_embedding(self, audio_buffer):
        """
        Simulate 256-dimensional audio embedding based on audio energy
        (For testing before YAMNet-256 model is fully integrated)

        Returns:
            np.ndarray: Simulated 256-dimensional embedding
        """
        # Calculate RMS energy
        rms = np.sqrt(np.mean(audio_buffer ** 2))

        # Create a pseudo-embedding based on audio characteristics
        # First 64 dims: energy-based features
        # Next 64 dims: frequency-based (simulate low/mid/high)
        # Last 128 dims: random noise (simulate learned features)

        embedding = np.zeros(256, dtype=np.float32)

        # Energy features (0-63)
        embedding[0:64] = np.clip(rms * 10, 0, 1) + np.random.randn(64) * 0.1

        # Frequency features (64-127) - simulate spectral content
        if len(audio_buffer) > 512:
            fft = np.abs(np.fft.rfft(audio_buffer[:512]))
            low_freq = np.mean(fft[:50])
            mid_freq = np.mean(fft[50:150])
            high_freq = np.mean(fft[150:])
            embedding[64:96] = low_freq * 0.01 + np.random.randn(32) * 0.05
            embedding[96:128] = mid_freq * 0.01 + np.random.randn(32) * 0.05

        # Context features (128-255) - random learned-like features
        embedding[128:256] = np.random.randn(128) * 0.1

        # Normalize to reasonable range [-1, 1]
        embedding = np.clip(embedding, -1, 1)

        return embedding.astype(np.float32)


def test_yamnet_processor():
    """Test Yamnet processor without Webots"""
    print("=" * 60)
    print("Testing FR2.2: Yamnet-256 Audio Processor")
    print("=" * 60)

    processor = YamnetProcessor()

    # Simulate audio buffer (1 second of noise)
    sample_rate = 16000
    duration = 1.0
    audio_buffer = np.random.randn(int(sample_rate * duration)) * 0.1

    print(f"\nTest audio: {len(audio_buffer)} samples @ {sample_rate}Hz")

    # Test embedding extraction
    embedding = processor.get_audio_embedding(audio_buffer, sample_rate)

    print(f"\nExtracted audio embedding:")
    print(f"  Shape: {embedding.shape}")
    print(f"  Dtype: {embedding.dtype}")
    print(f"  Range: [{embedding.min():.3f}, {embedding.max():.3f}]")
    print(f"  Mean: {embedding.mean():.3f}")
    print(f"  First 10 values: {embedding[:10]}")

    # Test silence (zero audio)
    silence_embedding = processor.get_audio_embedding(None)
    print(f"\nSilence embedding (all zeros): {np.all(silence_embedding == 0)}")

    print("\n" + "=" * 60)
    print("FR2.2 Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_yamnet_processor()
