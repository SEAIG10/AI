"""
FR2.3: Context Vector Generation
Fuses multimodal sensor data (GPS, YOLO, Audio) into structured context vectors
"""

import time
import json
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
        audio_events: Optional[List[Dict]] = None
    ) -> Dict:
        """
        FR2.3: Create unified context vector from multimodal inputs

        Args:
            timestamp: Unix timestamp (default: current time)
            position: (x, y) GPS coordinates
            zone_info: Semantic zone information from FloorPlanManager
            visual_detections: YOLO object detections
            audio_events: Yamnet audio event classifications

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
                "audio_events": [
                    {"event": "Television", "confidence": 0.78}
                ],
                "context_summary": "living_room | 2 objects | Television"
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

        # Build audio events
        audio_events_data = []
        if audio_events:
            for event in audio_events:
                audio_events_data.append({
                    "event": event["event"],
                    "confidence": round(event["confidence"], 2)
                })

        # Create context summary (human-readable)
        visual_count = len(visual_events)
        audio_label = audio_events_data[0]["event"] if audio_events_data else "None"
        context_summary = f"{zone_id} | {visual_count} objects | {audio_label}"

        # Assemble full context vector
        context_vector = {
            "timestamp": round(timestamp, 3),
            "position": position_data,
            "zone": zone_label,
            "zone_id": zone_id,
            "visual_events": visual_events,
            "audio_events": audio_events_data,
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

    def get_dominant_audio_event(self, context: Dict) -> Optional[str]:
        """
        Get the audio event with highest confidence

        Args:
            context: Context vector

        Returns:
            Audio event name or None
        """
        audio_events = context.get("audio_events", [])
        if not audio_events:
            return None

        # Find event with max confidence
        dominant = max(audio_events, key=lambda e: e["confidence"])
        return dominant["event"]

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
                "audio_changed": True/False
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

        # Audio change
        audio1 = self.get_dominant_audio_event(context1)
        audio2 = self.get_dominant_audio_event(context2)
        audio_changed = (audio1 != audio2)

        return {
            "zone_changed": zone_changed,
            "new_objects": new_objects,
            "removed_objects": removed_objects,
            "audio_changed": audio_changed,
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
    audio_events = [
        {"event": "Television", "confidence": 0.78}
    ]

    # Create context vector
    context = cv.create_context(
        position=position,
        zone_info=zone_info,
        visual_detections=visual_detections,
        audio_events=audio_events
    )

    print("\nContext Vector:")
    print(cv.context_to_json(context))

    print("\nVisual class counts:")
    print(cv.get_visual_class_counts(context))

    print("\nDominant audio event:")
    print(cv.get_dominant_audio_event(context))

    print("\n" + "=" * 60)
    print("FR2.3 Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_context_vector()
