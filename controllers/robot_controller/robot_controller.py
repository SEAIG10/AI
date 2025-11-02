"""RVC (Robot Vacuum Cleaner) Controller with AI Vision and Audio"""

from controller import Robot
import numpy as np
import cv2
from ultralytics import YOLO
import sys
import os
from collections import deque

# Add src directory to path for module imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from spatial_mapping.floor_plan_manager import FloorPlanManager
from audio_recognition.yamnet_processor import YamnetProcessor
from context_fusion.context_vector import ContextVector
from context_fusion.context_database import ContextDatabase
from context_fusion.context_encoder import ContextEncoder

# Import TensorFlow for GRU model
import tensorflow as tf
from tensorflow import keras

class RVCRobot:
    def __init__(self):
        # Initialize robot
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())

        # Initialize camera
        self.camera = self.robot.getDevice('camera')
        self.camera.enable(self.timestep)
        # Enable recognition for object detection (will be used with YOLO)
        if self.camera.hasRecognition():
            self.camera.recognitionEnable(self.timestep)

        # Initialize microphone (optional - not all Webots versions support it)
        self.microphone = self.robot.getDevice('microphone')
        if self.microphone is not None:
            self.microphone.enable(self.timestep)
            print("Microphone enabled")
        else:
            print("Microphone not available (audio sensing will be simulated)")

        # Initialize GPS sensor (FR1.3: Real-time position tracking)
        self.gps = self.robot.getDevice('gps')
        if self.gps is not None:
            self.gps.enable(self.timestep)
            print("GPS sensor enabled")
        else:
            print("GPS sensor not available")

        # Initialize FloorPlanManager (FR1: Semantic Spatial Mapping)
        config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'zones_config.json')
        try:
            self.floor_plan = FloorPlanManager(config_path)
            print("FloorPlanManager initialized")
        except Exception as e:
            print(f"Warning: Could not load FloorPlanManager: {e}")
            self.floor_plan = None

        # Initialize YamnetProcessor (FR2.2: Audio Event Detection)
        try:
            self.audio_processor = YamnetProcessor()
            print("YamnetProcessor initialized")
        except Exception as e:
            print(f"Warning: Could not load YamnetProcessor: {e}")
            self.audio_processor = None

        # Initialize motors (try different naming conventions)
        self.left_motor = self.robot.getDevice('left_motor')
        self.right_motor = self.robot.getDevice('right_motor')

        # Fallback to iRobot Create names
        if self.left_motor is None:
            self.left_motor = self.robot.getDevice('left wheel motor')
        if self.right_motor is None:
            self.right_motor = self.robot.getDevice('right wheel motor')

        # Check if motors were found
        if self.left_motor is None or self.right_motor is None:
            print("ERROR: Motors not found!")
            print("Available devices:")
            for i in range(self.robot.getNumberOfDevices()):
                device = self.robot.getDeviceByIndex(i)
                print(f"  - {device.getName()}")
        else:
            # Set motors to velocity control mode
            self.left_motor.setPosition(float('inf'))
            self.right_motor.setPosition(float('inf'))
            self.left_motor.setVelocity(0.0)
            self.right_motor.setVelocity(0.0)
            print("Motors initialized successfully!")

        # Initialize YOLO model
        print("Loading YOLOv8 model...")
        try:
            self.yolo_model = YOLO('yolov8n.pt')  # Use nano model for speed
            print("YOLOv8 model loaded successfully!")
        except Exception as e:
            print(f"Warning: Could not load YOLO model: {e}")
            self.yolo_model = None

        # Initialize Context Fusion (FR2.3: Multimodal Context Fusion)
        try:
            self.context_vector = ContextVector()
            db_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'context_vectors.db')
            self.context_db = ContextDatabase(db_path)
            print("Context Fusion system initialized")
        except Exception as e:
            print(f"Warning: Could not initialize Context Fusion: {e}")
            self.context_vector = None
            self.context_db = None

        # FR3: Initialize Context Encoder for real-time vectorization
        try:
            self.context_encoder = ContextEncoder()
            print("Context Encoder initialized")
        except Exception as e:
            print(f"Warning: Could not initialize Context Encoder: {e}")
            self.context_encoder = None

        # FR3: Initialize GRU Model for pollution prediction
        model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'gru_model.keras')
        try:
            self.gru_model = keras.models.load_model(model_path)
            print(f"GRU Model loaded from: {model_path}")
        except Exception as e:
            print(f"Warning: Could not load GRU model: {e}")
            self.gru_model = None

        # FR3: Initialize 30-timestep buffer for sequential prediction
        self.context_buffer = deque(maxlen=30)  # Stores last 30 context vectors (108-dim each)
        self.prediction_interval = 10  # Run prediction every N steps (to avoid too frequent predictions)
        self.last_prediction = None  # Store last prediction result
        print(f"Context buffer initialized (size: 30 timesteps)")

        # Zone names for prediction output
        self.zone_names = [
            "bathroom", "bedroom_1", "bedroom_2", "corridor",
            "garden_balcony", "kitchen", "living_room"
        ]

        print("RVC Robot initialized successfully!")
        print(f"Camera resolution: {self.camera.getWidth()}x{self.camera.getHeight()}")
        if self.microphone is not None:
            print(f"Microphone sampling rate: {self.microphone.getSamplingRate()} Hz")

    def get_camera_image(self):
        """Get current camera image as numpy array"""
        # Get image from camera
        image = self.camera.getImage()
        if image is None:
            return None

        # Convert to numpy array (RGB format)
        width = self.camera.getWidth()
        height = self.camera.getHeight()

        # Webots returns BGRA format, convert to RGB
        image_array = np.frombuffer(image, np.uint8).reshape((height, width, 4))
        rgb_image = image_array[:, :, [2, 1, 0]]  # Convert BGRA to RGB

        return rgb_image

    def get_audio_sample(self):
        """Get current audio sample from microphone"""
        if self.microphone is None:
            return None

        # Get audio buffer
        buffer = self.microphone.getSampleBuffer()
        if buffer is None:
            return None

        # Convert to numpy array
        audio_array = np.array(buffer)
        return audio_array

    def get_position(self):
        """FR1.3: Get current GPS position (x, y, z)"""
        if self.gps is None:
            return None

        values = self.gps.getValues()
        if values is None or len(values) < 2:
            return None

        # Return (x, y) - we ignore z for floor plan mapping
        return (values[0], values[1])

    def get_current_zone(self):
        """FR1.4: Get semantic zone label for current position"""
        if self.floor_plan is None:
            return None

        position = self.get_position()
        if position is None:
            return None

        x, y = position
        zone_info = self.floor_plan.get_zone(x, y)
        return zone_info

    def move_forward(self, speed=2.0):
        """Move robot forward"""
        if self.left_motor and self.right_motor:
            self.left_motor.setVelocity(speed)
            self.right_motor.setVelocity(speed)

    def move_backward(self, speed=2.0):
        """Move robot backward"""
        if self.left_motor and self.right_motor:
            self.left_motor.setVelocity(-speed)
            self.right_motor.setVelocity(-speed)

    def turn_left(self, speed=2.0):
        """Turn robot left"""
        if self.left_motor and self.right_motor:
            self.left_motor.setVelocity(-speed)
            self.right_motor.setVelocity(speed)

    def turn_right(self, speed=2.0):
        """Turn robot right"""
        if self.left_motor and self.right_motor:
            self.left_motor.setVelocity(speed)
            self.right_motor.setVelocity(-speed)

    def stop(self):
        """Stop robot"""
        if self.left_motor and self.right_motor:
            self.left_motor.setVelocity(0.0)
            self.right_motor.setVelocity(0.0)

    def display_camera_view(self, image, detections=None):
        """Display camera image in OpenCV window with YOLO detections"""
        if image is not None:
            # Add text overlay showing resolution
            display_img = image.copy()

            # Draw YOLO detections
            if detections:
                for det in detections:
                    x1, y1, x2, y2 = det['bbox']
                    conf = det['confidence']
                    class_name = det['class']

                    # Draw bounding box
                    color = (0, 255, 0)  # Green
                    cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 2)

                    # Draw label
                    label = f"{class_name} {conf:.2f}"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(display_img, (x1, y1 - label_size[1] - 10),
                                (x1 + label_size[0], y1), color, -1)
                    cv2.putText(display_img, label, (x1, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            # Add info overlay
            cv2.putText(display_img, f"Objects: {len(detections) if detections else 0}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display_img, "RVC Robot - YOLOv8",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Show image
            cv2.imshow("RVC Robot - Camera", display_img)
            cv2.waitKey(1)  # Required for cv2.imshow to work

    def process_vision(self, image):
        """Process camera image with YOLO model"""
        if image is None or self.yolo_model is None:
            return None

        # Run YOLO inference
        results = self.yolo_model(image, verbose=False)  # verbose=False to reduce console spam

        # Extract detection results
        detections = []
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes

            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                # Get confidence
                conf = float(box.conf[0])
                # Get class
                cls = int(box.cls[0])
                class_name = result.names[cls]

                detections.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': conf,
                    'class': class_name
                })

        return detections

    def process_audio(self, audio):
        """
        FR2.2: Process audio with Yamnet-256 for Sound Event Detection

        Args:
            audio: Raw audio buffer from microphone

        Returns:
            list: Detected audio events [{"event": "Speech", "confidence": 0.85}, ...]
        """
        if self.audio_processor is None:
            return None

        if audio is None or len(audio) == 0:
            return [{"event": "Silence", "confidence": 1.0}]

        # Classify audio events
        events = self.audio_processor.classify_audio(audio)

        # Filter to target events (FR2.2.2)
        filtered_events = self.audio_processor.filter_target_events(events)

        return filtered_events

    def predict_pollution(self):
        """
        FR3: Predict pollution probability for all zones using GRU model

        Requires: context_buffer to have 30 timesteps (30, 108)

        Returns:
            dict: Prediction results with probabilities for each zone
        """
        if self.gru_model is None:
            return None

        if len(self.context_buffer) < 30:
            return None  # Not enough data yet

        # Convert buffer to numpy array (30, 108)
        sequence = np.array(list(self.context_buffer), dtype=np.float32)

        # Add batch dimension: (30, 108) → (1, 30, 108)
        sequence = np.expand_dims(sequence, axis=0)

        # Run GRU prediction
        predictions = self.gru_model.predict(sequence, verbose=0)[0]  # (7,)

        # Create result dict
        result = {
            'timestamp': self.robot.getTime(),
            'zones': {}
        }

        for i, zone_name in enumerate(self.zone_names):
            result['zones'][zone_name] = float(predictions[i])

        # Find zones that need cleaning (threshold: 0.5)
        result['needs_cleaning'] = [
            zone for zone, prob in result['zones'].items() if prob > 0.5
        ]

        return result

    def run(self):
        """Main control loop"""
        step_count = 0

        print("\n=== RVC Robot Starting ===")
        print("Controls will be implemented for autonomous navigation")
        print("Camera and microphone data will be processed for AI learning\n")

        # Simple test pattern: move forward for a bit
        self.move_forward(1.0)

        while self.robot.step(self.timestep) != -1:
            step_count += 1

            # Get sensor data
            image = self.get_camera_image()
            audio = self.get_audio_sample()

            # FR2: Multimodal sensing
            detections = self.process_vision(image)  # FR2.1: Visual
            audio_events = self.process_audio(audio)  # FR2.2: Audio

            # Display camera view with detections
            self.display_camera_view(image, detections)

            # Print sensor data periodically (FR2: Multimodal Context Awareness)
            if step_count % 50 == 0:
                print(f"\n=== Step {step_count} ===")

                # FR1.3 & FR1.4: GPS position and semantic zone
                position = self.get_position()
                zone_info = self.get_current_zone()

                if position:
                    print(f"Position: ({position[0]:.2f}, {position[1]:.2f})")
                    if zone_info:
                        print(f"Zone: {zone_info['label']}")
                    else:
                        print("Zone: Unknown (outside defined zones)")
                else:
                    print("Position: GPS not available")

                # FR2.1: YOLO visual detections
                if detections:
                    print(f"Visual: {len(detections)} objects")
                    for det in detections[:3]:  # Show top 3
                        print(f"  - {det['class']}: {det['confidence']:.2f}")
                else:
                    print("Visual: No objects detected")

                # FR2.2: Audio events
                if audio_events:
                    print(f"Audio: {audio_events[0]['event']} ({audio_events[0]['confidence']:.2f})")
                else:
                    print("Audio: N/A")

                # FR2.3: Create and store context vector
                if self.context_vector and self.context_db:
                    context = self.context_vector.create_context(
                        position=position,
                        zone_info=zone_info,
                        visual_detections=detections,
                        audio_events=audio_events
                    )
                    row_id = self.context_db.insert_context(context)
                    print(f"Context: [{row_id}] {context['context_summary']}")

                    # FR3: Encode context to 108-dim vector and add to buffer
                    if self.context_encoder:
                        context_vector = self.context_encoder.encode(context)
                        self.context_buffer.append(context_vector)
                        print(f"  → Vector buffer: {len(self.context_buffer)}/30 timesteps")

                # FR3: Run GRU prediction every N steps
                if step_count % self.prediction_interval == 0 and len(self.context_buffer) >= 30:
                    prediction = self.predict_pollution()
                    if prediction:
                        self.last_prediction = prediction
                        print(f"\n{'='*60}")
                        print(f"FR3: Pollution Prediction @ t={prediction['timestamp']:.1f}s")
                        print(f"{'='*60}")
                        for zone, prob in prediction['zones'].items():
                            status = "🔴 NEEDS CLEANING" if prob > 0.5 else "✅ Clean"
                            print(f"  {zone:15s}: {prob:.3f} {status}")
                        if prediction['needs_cleaning']:
                            print(f"\n⚠️  Zones needing attention: {', '.join(prediction['needs_cleaning'])}")
                        print(f"{'='*60}\n")

            # TODO: Add AI decision making logic here
            # - YOLO for object detection (crumbs, people, furniture)
            # - SED for sound classification (TV, conversation)
            # - LSTM for pattern learning and prediction

            # Simple behavior for testing
            if step_count == 100:
                print("Turning right...")
                self.turn_right(1.0)
            elif step_count == 150:
                print("Moving forward...")
                self.move_forward(1.0)

        print("\n=== RVC Robot Stopped ===")
        cv2.destroyAllWindows()

# Main execution
if __name__ == "__main__":
    rvc = RVCRobot()
    rvc.run()
