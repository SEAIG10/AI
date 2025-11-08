"""
FR2.3: Context Vector Generation
Fuses multimodal sensor data (GPS, YOLO, Audio) into structured context vectors
"""

import time
import json
import numpy as np
from typing import Dict, List, Optional, Tuple


class ContextVector:
    """
    FR2.3: Multimodal Context Fusion
    Combines spatial (GPS), visual (YOLO), and audio (Yamnet) data
    """

    def __init__(self):
        """Initialize ContextVector processor"""
        self.last_context = None

    def create_context(
        self,
        timestamp: Optional[float] = None,
        position: Optional[Tuple[float, float]] = None,
        zone_info: Optional[Dict] = None,
        visual_detections: Optional[List[Dict]] = None,
        audio_embedding: Optional[np.ndarray] = None,
        keypoints: Optional[np.ndarray] = None
    ) -> Dict:
        """
        FR2.3: Create unified context vector from multimodal inputs

        Args:
            timestamp: Unix timestamp (default: current time)
            position: (x, y) GPS coordinates
            zone_info: Semantic zone information from FloorPlanManager
            visual_detections: YOLO object detections
            audio_embedding: YAMNet-256 audio embedding (256-dim numpy array)
            keypoints: Normalized pose keypoints (51-dim numpy array)

        Returns:
            Context vector as JSON-serializable dict:
            {
                "timestamp": 1678886400.123,
                "position": {"x": -5.0, "y": -3.0},
                "zone": "거실 (Living Room)",
                "zone_id": "living_room",
                "visual_events": [
                    {"class": "person", "confidence": 0.92, "bbox": [...]},
                    {"class": "couch", "confidence": 0.85, "bbox": [...]}
                ],
                "audio_embedding": [0.12, -0.34, ..., 0.56],  # 256-dim
                "keypoints": [x0, y0, c0, ..., x16, y16, c16],  # 51-dim
                "context_summary": "living_room | 2 objects | audio:yes | pose:yes"
            }
        """
        if timestamp is None:
            timestamp = time.time()

        # Build position data
        position_data = None
        if position:
            position_data = {
                "x": round(position[0], 2),
                "y": round(position[1], 2)
            }

        # Build zone data
        zone_label = "Unknown"
        zone_id = "unknown"
        if zone_info:
            zone_label = zone_info.get("label", "Unknown")
            zone_id = zone_info.get("id", "unknown")

        # Build visual events (YOLO detections)
        visual_events = []
        if visual_detections:
            for det in visual_detections:
                visual_events.append({
                    "class": det["class"],
                    "confidence": round(det["confidence"], 2),
                    "bbox": det["bbox"]  # (x1, y1, x2, y2)
                })

        # Process audio embedding
        audio_embedding_data = None
        has_audio = False
        if audio_embedding is not None:
            # Convert numpy array to list for JSON serialization
            audio_embedding_data = audio_embedding.tolist() if isinstance(audio_embedding, np.ndarray) else list(audio_embedding)
            has_audio = True

        # Process keypoints
        keypoints_data = None
        has_pose = False
        if keypoints is not None:
            # Convert numpy array to list for JSON serialization
            keypoints_data = keypoints.tolist() if isinstance(keypoints, np.ndarray) else list(keypoints)
            has_pose = True

        # Create context summary (human-readable)
        visual_count = len(visual_events)
        audio_label = "audio:yes" if has_audio else "audio:no"
        pose_label = "pose:yes" if has_pose else "pose:no"
        context_summary = f"{zone_id} | {visual_count} objects | {audio_label} | {pose_label}"

        # Assemble full context vector
        context_vector = {
            "timestamp": round(timestamp, 3),
            "position": position_data,
            "zone": zone_label,
            "zone_id": zone_id,
            "visual_events": visual_events,
            "audio_embedding": audio_embedding_data,  # 256-dim YAMNet embedding (or None)
            "keypoints": keypoints_data,  # 51-dim normalized pose (or None)
            "context_summary": context_summary
        }

        self.last_context = context_vector
        return context_vector

    def context_to_json(self, context: Dict) -> str:
        """
        Serialize context vector to JSON string

        Args:
            context: Context vector dict

        Returns:
            JSON string
        """
        return json.dumps(context, ensure_ascii=False, indent=2)

    def json_to_context(self, json_str: str) -> Dict:
        """
        Deserialize context vector from JSON string

        Args:
            json_str: JSON string

        Returns:
            Context vector dict
        """
        return json.loads(json_str)

    def get_visual_class_counts(self, context: Dict) -> Dict[str, int]:
        """
        Extract object class counts from context vector

        Args:
            context: Context vector

        Returns:
            Dict mapping class names to counts
            e.g., {"person": 2, "couch": 1, "cup": 3}
        """
        counts = {}
        for event in context.get("visual_events", []):
            class_name = event["class"]
            counts[class_name] = counts.get(class_name, 0) + 1
        return counts

    def has_audio_embedding(self, context: Dict) -> bool:
        """
        Check if context has audio embedding

        Args:
            context: Context vector

        Returns:
            True if audio embedding exists, False otherwise
        """
        audio_embedding = context.get("audio_embedding", None)
        return audio_embedding is not None and len(audio_embedding) > 0

    def compare_contexts(self, context1: Dict, context2: Dict) -> Dict:
        """
        Compare two context vectors to detect changes

        Args:
            context1: Earlier context vector
            context2: Later context vector

        Returns:
            Dict describing differences:
            {
                "zone_changed": True/False,
                "new_objects": ["person", "cup"],
                "removed_objects": ["dog"],
                "audio_changed": True/False,
                "audio_distance": 0.5  # L2 distance between embeddings
            }
        """
        # Zone change
        zone_changed = (
            context1.get("zone_id") != context2.get("zone_id")
        )

        # Visual changes
        classes1 = set(e["class"] for e in context1.get("visual_events", []))
        classes2 = set(e["class"] for e in context2.get("visual_events", []))
        new_objects = list(classes2 - classes1)
        removed_objects = list(classes1 - classes2)

        # Audio change (compare embeddings)
        audio1 = context1.get("audio_embedding")
        audio2 = context2.get("audio_embedding")

        audio_changed = False
        audio_distance = 0.0

        if audio1 is not None and audio2 is not None:
            # Calculate L2 distance between embeddings
            emb1 = np.array(audio1) if isinstance(audio1, list) else audio1
            emb2 = np.array(audio2) if isinstance(audio2, list) else audio2
            audio_distance = float(np.linalg.norm(emb1 - emb2))
            # Consider changed if distance > threshold
            audio_changed = audio_distance > 0.5
        elif audio1 != audio2:  # One is None, the other isn't
            audio_changed = True

        return {
            "zone_changed": zone_changed,
            "new_objects": new_objects,
            "removed_objects": removed_objects,
            "audio_changed": audio_changed,
            "audio_distance": audio_distance,
            "time_delta": context2["timestamp"] - context1["timestamp"]
        }


def test_context_vector():
    """Test context vector creation"""
    print("=" * 60)
    print("Testing FR2.3: Context Vector")
    print("=" * 60)

    cv = ContextVector()

    # Simulate sensor data
    position = (-5.0, -3.0)
    zone_info = {"id": "living_room", "label": "거실 (Living Room)"}
    visual_detections = [
        {"class": "person", "confidence": 0.92, "bbox": (100, 150, 200, 400)},
        {"class": "couch", "confidence": 0.85, "bbox": (50, 200, 300, 450)},
    ]
    # Simulate 256-dim audio embedding
    audio_embedding = np.random.randn(256).astype(np.float32)

    # Create context vector
    context = cv.create_context(
        position=position,
        zone_info=zone_info,
        visual_detections=visual_detections,
        audio_embedding=audio_embedding
    )

    print("\nContext Vector (truncated for readability):")
    context_display = context.copy()
    if context_display.get("audio_embedding"):
        context_display["audio_embedding"] = f"[256-dim array, first 5: {context_display['audio_embedding'][:5]}...]"
    print(json.dumps(context_display, ensure_ascii=False, indent=2))

    print("\nVisual class counts:")
    print(cv.get_visual_class_counts(context))

    print("\nHas audio embedding:")
    print(cv.has_audio_embedding(context))

    print("\n" + "=" * 60)
    print("FR2.3 Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_context_vector()
