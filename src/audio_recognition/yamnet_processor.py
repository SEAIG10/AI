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

        # FR2.2.2: Target audio events for context awareness
        self.target_events = {
            'Speech': ['Speech', 'Conversation', 'Narration'],
            'Television': ['Television', 'Video'],
            'Music': ['Music', 'Musical instrument'],
            'Dishes': ['Dishes, pots, and pans', 'Cutlery', 'silverware'],
            'Cooking': ['Frying', 'Food', 'Boiling'],
            'Silence': ['Silence', 'White noise']
        }

        # Load model if path provided
        if model_path:
            self._load_model(model_path)
        else:
            print("Warning: No model path provided. Using simulated audio detection.")

    def _load_model(self, model_path):
        """Load Yamnet-256 TFLite model"""
        try:
            import tensorflow as tf
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()

            # Get input/output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

            print(f"✓ Yamnet-256 model loaded from {model_path}")
            print(f"  Input shape: {self.input_details[0]['shape']}")
            print(f"  Output shape: {self.output_details[0]['shape']}")
            self.model = True
        except Exception as e:
            print(f"Warning: Could not load Yamnet-256 model: {e}")
            self.model = None

    def preprocess_audio(self, audio_buffer, sample_rate=None):
        """
        Convert audio buffer to mel-spectrogram

        Args:
            audio_buffer: Raw audio samples (numpy array)
            sample_rate: Sample rate of audio (default: 16000)

        Returns:
            mel_spectrogram: (64, 96, 1) mel-spectrogram patch
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

        # Compute mel-spectrogram
        # Yamnet-256 expects: (64 mels, 96 frames, 1 channel)
        mel_spec = librosa.feature.melspectrogram(
            y=audio_buffer,
            sr=self.sample_rate,
            n_mels=64,
            hop_length=160,  # 10ms hop
            n_fft=512
        )

        # Convert to log scale
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        # Extract 96-frame patch (center crop if longer, pad if shorter)
        if log_mel_spec.shape[1] >= 96:
            start_frame = (log_mel_spec.shape[1] - 96) // 2
            mel_patch = log_mel_spec[:, start_frame:start_frame + 96]
        else:
            # Pad with zeros if too short
            pad_width = ((0, 0), (0, 96 - log_mel_spec.shape[1]))
            mel_patch = np.pad(log_mel_spec, pad_width, mode='constant')

        # Add channel dimension: (64, 96) → (64, 96, 1)
        mel_patch = mel_patch[..., np.newaxis]

        return mel_patch.astype(np.float32)

    def classify_audio(self, audio_buffer, sample_rate=None):
        """
        FR2.2.1 & FR2.2.2: Classify audio events

        Args:
            audio_buffer: Raw audio samples
            sample_rate: Sample rate

        Returns:
            list: Detected audio events with confidence scores
                  [{"event": "Speech", "confidence": 0.85}, ...]
        """
        if audio_buffer is None or len(audio_buffer) == 0:
            return [{"event": "Silence", "confidence": 1.0}]

        # Preprocess audio
        mel_spec = self.preprocess_audio(audio_buffer, sample_rate)

        if mel_spec is None:
            # Fallback: Simulate audio detection based on RMS energy
            return self._simulate_audio_detection(audio_buffer)

        if self.model is None:
            # No model loaded, use simulation
            return self._simulate_audio_detection(audio_buffer)

        try:
            # Run inference
            self.interpreter.set_tensor(
                self.input_details[0]['index'],
                mel_spec[np.newaxis, ...]  # Add batch dimension
            )
            self.interpreter.invoke()

            # Get embedding (256-dimensional)
            embedding = self.interpreter.get_tensor(self.output_details[0]['index'])

            # TODO: Add classification head for our 5 target classes
            # For now, return simulated results
            return self._simulate_audio_detection(audio_buffer)

        except Exception as e:
            print(f"Warning: Audio classification failed: {e}")
            return self._simulate_audio_detection(audio_buffer)

    def _simulate_audio_detection(self, audio_buffer):
        """
        Simulate audio event detection based on audio energy
        (For testing before model is fully integrated)
        """
        # Calculate RMS energy
        rms = np.sqrt(np.mean(audio_buffer ** 2))

        # Simple heuristic classification
        if rms < 0.01:
            return [{"event": "Silence", "confidence": 0.95}]
        elif rms < 0.05:
            return [{"event": "Speech", "confidence": 0.7}]
        elif rms < 0.1:
            return [{"event": "Television", "confidence": 0.6}]
        else:
            return [{"event": "Music", "confidence": 0.8}]

    def filter_target_events(self, events):
        """
        FR2.2.2: Filter and map to target event categories

        Args:
            events: List of detected events

        Returns:
            Filtered list focusing on context-relevant events
        """
        filtered = []
        for event in events:
            event_name = event['event']

            # Check if event matches any target category
            for category, keywords in self.target_events.items():
                if any(keyword.lower() in event_name.lower() for keyword in keywords):
                    filtered.append({
                        'event': category,
                        'confidence': event['confidence']
                    })
                    break

        return filtered if filtered else [{"event": "Unknown", "confidence": 0.0}]


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

    # Test classification
    events = processor.classify_audio(audio_buffer, sample_rate)

    print("\nDetected audio events:")
    for event in events:
        print(f"  - {event['event']}: {event['confidence']:.2f}")

    print("\n" + "=" * 60)
    print("FR2.2 Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_yamnet_processor()
