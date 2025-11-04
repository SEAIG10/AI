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

        # Initialize YOLO models
        print("Loading YOLOv8 models...")
        try:
            self.yolo_model = YOLO('yolov8n.pt')  # Object detection
            print("✓ YOLOv8n (object detection) loaded")
        except Exception as e:
            print(f"Warning: Could not load YOLO model: {e}")
            self.yolo_model = None

        try:
            self.yolo_pose_model = YOLO('yolov8n-pose.pt')  # Pose estimation
            print("✓ YOLOv8n-pose (human pose) loaded")
        except Exception as e:
            print(f"Warning: Could not load YOLO-pose model: {e}")
            self.yolo_pose_model = None

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
        """
        Process camera image with YOLO models

        Returns:
            tuple: (detections, pose_data)
                detections: list of object detections
                pose_data: dict with human pose keypoints or None
        """
        if image is None or self.yolo_model is None:
            return None, None

        # Run YOLOv8n for object detection
        results = self.yolo_model(image, verbose=False)

        # Extract detection results
        detections = []
        person_detected = False

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

                # Check if person detected
                if class_name == 'person':
                    person_detected = True

        # Run YOLOv8n-pose if person detected (Option 3: Conditional execution)
        pose_data = None
        if person_detected and self.yolo_pose_model is not None:
            try:
                pose_results = self.yolo_pose_model(image, verbose=False)
                if len(pose_results) > 0:
                    pose_result = pose_results[0]
                    if hasattr(pose_result, 'keypoints') and pose_result.keypoints is not None:
                        # Extract keypoints (17 points x, y, confidence)
                        kpts = pose_result.keypoints.data.cpu().numpy()
                        if len(kpts) > 0:
                            pose_data = {
                                'keypoints': kpts[0],  # First person's keypoints (17, 3)
                                'confidence': float(pose_result.boxes.conf[0]) if len(pose_result.boxes.conf) > 0 else 0.0
                            }
            except Exception as e:
                print(f"Pose estimation error: {e}")
                pose_data = None

        return detections, pose_data

    def normalize_keypoints(self, keypoints):
        """
        Normalize keypoints to be body-centric and scale-invariant

        Args:
            keypoints: (17, 3) numpy array [x, y, confidence]

        Returns:
            (51,) flattened normalized keypoints, or None if invalid
        """
        import numpy as np

        # Confidence threshold
        confidence_threshold = 0.3

        # Get hip center as reference point
        left_hip = keypoints[11]
        right_hip = keypoints[12]

        # Check if hips are visible
        if left_hip[2] < confidence_threshold or right_hip[2] < confidence_threshold:
            # Can't normalize without hips, return zeros
            return np.zeros(51, dtype=np.float32)

        # Hip center
        hip_center_x = (left_hip[0] + right_hip[0]) / 2
        hip_center_y = (left_hip[1] + right_hip[1]) / 2

        # Body scale: distance from hip to nose
        nose = keypoints[0]
        if nose[2] > confidence_threshold:
            body_height = np.sqrt(
                (nose[0] - hip_center_x)**2 +
                (nose[1] - hip_center_y)**2
            )
        else:
            # Fallback: use shoulder-hip distance
            left_shoulder = keypoints[5]
            right_shoulder = keypoints[6]
            if left_shoulder[2] > confidence_threshold:
                body_height = np.sqrt(
                    (left_shoulder[0] - hip_center_x)**2 +
                    (left_shoulder[1] - hip_center_y)**2
                )
            else:
                body_height = 100.0  # Default scale

        # Avoid division by zero
        body_height = max(body_height, 1.0)

        # Normalize all keypoints
        normalized = []
        for i in range(17):
            x, y, conf = keypoints[i]

            if conf < confidence_threshold:
                # Low confidence: set to zero
                normalized.extend([0.0, 0.0, 0.0])
            else:
                # Normalize: center at hip, scale by body height
                x_norm = (x - hip_center_x) / body_height
                y_norm = (y - hip_center_y) / body_height
                normalized.extend([x_norm, y_norm, conf])

        return np.array(normalized, dtype=np.float32)

    def infer_activity_from_pose(self, pose_data):
        """
        Infer human activity from pose keypoints

        Keypoints (COCO 17-point format):
        0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
        5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
        9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
        13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle

        Returns:
            str: Activity label ("standing", "sitting", "cooking", "eating", "unknown")
        """
        if pose_data is None:
            return "unknown"

        keypoints = pose_data['keypoints']  # Shape: (17, 3) - (x, y, confidence)

        # Extract key body parts with confidence threshold
        confidence_threshold = 0.5

        def get_point(idx):
            if keypoints[idx][2] > confidence_threshold:
                return (keypoints[idx][0], keypoints[idx][1])
            return None

        # Get critical points
        left_shoulder = get_point(5)
        right_shoulder = get_point(6)
        left_hip = get_point(11)
        right_hip = get_point(12)
        left_knee = get_point(13)
        right_knee = get_point(14)
        left_wrist = get_point(9)
        right_wrist = get_point(10)
        nose = get_point(0)

        # Check if we have enough points
        if not all([left_hip, right_hip, left_knee, right_knee]):
            return "unknown"

        # Calculate hip and knee positions
        hip_y = (left_hip[1] + right_hip[1]) / 2
        knee_y = (left_knee[1] + right_knee[1]) / 2

        # Sitting detection: knees are close to or above hip level (smaller y = higher in image)
        hip_knee_ratio = abs(knee_y - hip_y) / max(abs(hip_y), 1)

        if hip_knee_ratio < 0.3:  # Knees very close to hips
            activity = "sitting"
        else:
            activity = "standing"

        # Cooking detection: hands near hip level (near counter/sink)
        if left_wrist and right_wrist and left_hip and right_hip:
            avg_hip_y = (left_hip[1] + right_hip[1]) / 2
            left_wrist_near_hip = abs(left_wrist[1] - avg_hip_y) < 100
            right_wrist_near_hip = abs(right_wrist[1] - avg_hip_y) < 100

            if (left_wrist_near_hip or right_wrist_near_hip) and activity == "standing":
                activity = "cooking"

        # Eating detection: hands near head level
        if left_wrist and nose:
            if abs(left_wrist[1] - nose[1]) < 80:
                activity = "eating"
        if right_wrist and nose:
            if abs(right_wrist[1] - nose[1]) < 80:
                activity = "eating"

        return activity

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
            detections, pose_data = self.process_vision(image)  # FR2.1: Visual (objects + pose)
            audio_events = self.process_audio(audio)  # FR2.2: Audio

            # Normalize keypoints if available (Option 1: Raw keypoints to GRU)
            keypoints_normalized = None
            if pose_data:
                keypoints_normalized = self.normalize_keypoints(pose_data['keypoints'])

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

                # Pose keypoints (raw data to GRU)
                if pose_data:
                    print(f"Pose: Detected (conf: {pose_data['confidence']:.2f})")
                    if keypoints_normalized is not None:
                        non_zero = (keypoints_normalized != 0).sum()
                        print(f"  Keypoints: {non_zero}/51 values (normalized)")

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
                        audio_events=audio_events,
                        keypoints=keypoints_normalized  # Add normalized pose keypoints (51 dim)
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
