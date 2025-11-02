"""
Test script for FR2.3: Context Fusion
Tests context vector creation and database operations
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from context_fusion.context_vector import ContextVector
from context_fusion.context_database import ContextDatabase


def test_context_fusion():
    """Test complete FR2.3 context fusion pipeline"""
    print("=" * 70)
    print("FR2.3: Context Fusion Test")
    print("=" * 70)

    # Initialize components
    cv = ContextVector()
    db = ContextDatabase("data/test_context_fusion.db")

    print("\n[1] Testing Context Vector Creation...")
    print("-" * 70)

    # Simulate multimodal sensor data
    test_scenarios = [
        {
            "name": "Living Room - Watching TV",
            "position": (-5.0, -3.0),
            "zone_info": {"id": "living_room", "label": "거실 (Living Room)"},
            "visual_detections": [
                {"class": "person", "confidence": 0.92, "bbox": (100, 150, 200, 400)},
                {"class": "couch", "confidence": 0.85, "bbox": (50, 200, 300, 450)},
                {"class": "remote", "confidence": 0.78, "bbox": (180, 350, 220, 380)},
            ],
            "audio_events": [
                {"event": "Television", "confidence": 0.88}
            ]
        },
        {
            "name": "Kitchen - Cooking",
            "position": (-1.5, -2.5),
            "zone_info": {"id": "kitchen", "label": "주방 (Kitchen)"},
            "visual_detections": [
                {"class": "person", "confidence": 0.89, "bbox": (120, 160, 220, 420)},
                {"class": "bowl", "confidence": 0.81, "bbox": (200, 300, 280, 360)},
            ],
            "audio_events": [
                {"event": "Cooking", "confidence": 0.85}
            ]
        },
        {
            "name": "Bedroom - Sleeping",
            "position": (-7.0, -10.0),
            "zone_info": {"id": "bedroom_1", "label": "침실1 (Bedroom 1)"},
            "visual_detections": [
                {"class": "bed", "confidence": 0.95, "bbox": (50, 100, 400, 450)},
            ],
            "audio_events": [
                {"event": "Silence", "confidence": 0.92}
            ]
        }
    ]

    contexts = []
    for i, scenario in enumerate(test_scenarios):
        print(f"\nScenario {i+1}: {scenario['name']}")

        context = cv.create_context(
            position=scenario["position"],
            zone_info=scenario["zone_info"],
            visual_detections=scenario["visual_detections"],
            audio_events=scenario["audio_events"]
        )

        contexts.append(context)

        print(f"  Position: {context['position']}")
        print(f"  Zone: {context['zone']}")
        print(f"  Visual objects: {len(context['visual_events'])}")
        print(f"  Audio: {context['audio_events'][0]['event']}")
        print(f"  Summary: {context['context_summary']}")

    print("\n[2] Testing Database Storage...")
    print("-" * 70)

    for i, context in enumerate(contexts):
        row_id = db.insert_context(context)
        print(f"  Stored context {i+1} with ID: {row_id}")

    print("\n[3] Testing Database Queries...")
    print("-" * 70)

    # Recent contexts
    print("\nRecent contexts:")
    recent = db.get_recent_contexts(limit=5)
    for ctx in recent:
        print(f"  [{ctx['id']}] {ctx['context_summary']}")

    # Zone-specific query
    print("\nLiving room contexts:")
    living_room = db.get_contexts_by_zone("living_room")
    for ctx in living_room:
        print(f"  [{ctx['id']}] {ctx['context_summary']}")

    # Statistics
    print("\nDatabase statistics:")
    stats = db.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n[4] Testing Context Comparison...")
    print("-" * 70)

    if len(contexts) >= 2:
        comparison = cv.compare_contexts(contexts[0], contexts[1])
        print(f"\nComparing '{test_scenarios[0]['name']}' vs '{test_scenarios[1]['name']}':")
        print(f"  Zone changed: {comparison['zone_changed']}")
        print(f"  New objects: {comparison['new_objects']}")
        print(f"  Removed objects: {comparison['removed_objects']}")
        print(f"  Audio changed: {comparison['audio_changed']}")

    print("\n[5] Testing JSON Serialization...")
    print("-" * 70)

    json_str = cv.context_to_json(contexts[0])
    print("\nContext as JSON:")
    print(json_str[:300] + "...")

    print("\n" + "=" * 70)
    print("✓ FR2.3 Context Fusion Test Complete!")
    print("=" * 70)


if __name__ == "__main__":
    test_context_fusion()
