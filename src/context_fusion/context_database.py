"""
FR2.3: Context Vector Database
SQLite storage for time-series context vectors
"""

import sqlite3
import json
import os
from typing import List, Dict, Optional
from datetime import datetime


class ContextDatabase:
    """
    FR2.3: SQLite database for storing context vectors
    Provides time-series storage and querying of multimodal context data
    """

    def __init__(self, db_path: str = "data/context_vectors.db"):
        """
        Initialize context database

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path

        # Create data directory if needed
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
            print(f"Created database directory: {db_dir}")

        # Initialize database
        self._init_database()
        print(f"Context database initialized: {db_path}")

    def _init_database(self):
        """Create database schema if not exists"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Context vectors table (time-series data)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS context_vectors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                position_x REAL,
                position_y REAL,
                zone TEXT,
                zone_id TEXT,
                visual_events TEXT,
                audio_events TEXT,
                context_summary TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Index for fast time-based queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON context_vectors(timestamp)
        """)

        # Index for zone-based queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_zone_id
            ON context_vectors(zone_id)
        """)

        conn.commit()
        conn.close()

    def insert_context(self, context: Dict) -> int:
        """
        Insert context vector into database

        Args:
            context: Context vector dict

        Returns:
            Row ID of inserted context
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Extract position
        position = context.get("position")
        position_x = position["x"] if position else None
        position_y = position["y"] if position else None

        # Serialize JSON fields
        visual_events_json = json.dumps(context.get("visual_events", []))
        audio_events_json = json.dumps(context.get("audio_events", []))

        cursor.execute("""
            INSERT INTO context_vectors (
                timestamp, position_x, position_y, zone, zone_id,
                visual_events, audio_events, context_summary
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            context["timestamp"],
            position_x,
            position_y,
            context.get("zone"),
            context.get("zone_id"),
            visual_events_json,
            audio_events_json,
            context.get("context_summary")
        ))

        row_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return row_id

    def get_recent_contexts(self, limit: int = 100) -> List[Dict]:
        """
        Get most recent context vectors

        Args:
            limit: Maximum number of contexts to retrieve

        Returns:
            List of context vectors (newest first)
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM context_vectors
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))

        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_context(row) for row in rows]

    def get_contexts_by_zone(self, zone_id: str, limit: int = 100) -> List[Dict]:
        """
        Get context vectors for a specific zone

        Args:
            zone_id: Zone identifier (e.g., "living_room")
            limit: Maximum number of contexts to retrieve

        Returns:
            List of context vectors (newest first)
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM context_vectors
            WHERE zone_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (zone_id, limit))

        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_context(row) for row in rows]

    def get_contexts_by_timerange(
        self,
        start_time: float,
        end_time: float
    ) -> List[Dict]:
        """
        Get context vectors within time range

        Args:
            start_time: Start timestamp (Unix time)
            end_time: End timestamp (Unix time)

        Returns:
            List of context vectors (oldest first)
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM context_vectors
            WHERE timestamp BETWEEN ? AND ?
            ORDER BY timestamp ASC
        """, (start_time, end_time))

        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_context(row) for row in rows]

    def get_statistics(self) -> Dict:
        """
        Get database statistics

        Returns:
            Dict with database stats:
            {
                "total_contexts": 1234,
                "earliest_timestamp": 1678886400.0,
                "latest_timestamp": 1678972800.0,
                "zones": ["living_room", "kitchen", ...],
                "duration_hours": 24.0
            }
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Total contexts
        cursor.execute("SELECT COUNT(*) FROM context_vectors")
        total = cursor.fetchone()[0]

        # Time range
        cursor.execute("""
            SELECT MIN(timestamp), MAX(timestamp)
            FROM context_vectors
        """)
        earliest, latest = cursor.fetchone()

        # Unique zones
        cursor.execute("SELECT DISTINCT zone_id FROM context_vectors")
        zones = [row[0] for row in cursor.fetchall()]

        conn.close()

        duration_hours = 0
        if earliest and latest:
            duration_hours = (latest - earliest) / 3600.0

        return {
            "total_contexts": total,
            "earliest_timestamp": earliest,
            "latest_timestamp": latest,
            "zones": zones,
            "duration_hours": round(duration_hours, 2)
        }

    def clear_database(self):
        """Delete all context vectors (use with caution!)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM context_vectors")
        conn.commit()
        conn.close()
        print("Database cleared!")

    def _row_to_context(self, row: sqlite3.Row) -> Dict:
        """
        Convert database row to context vector dict

        Args:
            row: SQLite row

        Returns:
            Context vector dict
        """
        position = None
        if row["position_x"] is not None and row["position_y"] is not None:
            position = {
                "x": row["position_x"],
                "y": row["position_y"]
            }

        return {
            "id": row["id"],
            "timestamp": row["timestamp"],
            "position": position,
            "zone": row["zone"],
            "zone_id": row["zone_id"],
            "visual_events": json.loads(row["visual_events"]),
            "audio_events": json.loads(row["audio_events"]),
            "context_summary": row["context_summary"]
        }


def test_context_database():
    """Test context database operations"""
    print("=" * 60)
    print("Testing FR2.3: Context Database")
    print("=" * 60)

    # Create test database
    db = ContextDatabase("data/test_context.db")

    # Create test contexts
    contexts = [
        {
            "timestamp": 1678886400.0,
            "position": {"x": -5.0, "y": -3.0},
            "zone": "거실 (Living Room)",
            "zone_id": "living_room",
            "visual_events": [
                {"class": "person", "confidence": 0.92, "bbox": [100, 150, 200, 400]}
            ],
            "audio_events": [
                {"event": "Television", "confidence": 0.78}
            ],
            "context_summary": "living_room | 1 objects | Television"
        },
        {
            "timestamp": 1678886410.0,
            "position": {"x": -2.5, "y": -2.0},
            "zone": "주방 (Kitchen)",
            "zone_id": "kitchen",
            "visual_events": [
                {"class": "person", "confidence": 0.88, "bbox": [120, 160, 210, 420]}
            ],
            "audio_events": [
                {"event": "Cooking", "confidence": 0.85}
            ],
            "context_summary": "kitchen | 1 objects | Cooking"
        }
    ]

    # Insert contexts
    print("\nInserting test contexts...")
    for ctx in contexts:
        row_id = db.insert_context(ctx)
        print(f"  Inserted context {row_id}: {ctx['context_summary']}")

    # Query recent contexts
    print("\nRecent contexts:")
    recent = db.get_recent_contexts(limit=5)
    for ctx in recent:
        print(f"  [{ctx['id']}] {ctx['context_summary']}")

    # Query by zone
    print("\nLiving room contexts:")
    living_room = db.get_contexts_by_zone("living_room")
    for ctx in living_room:
        print(f"  [{ctx['id']}] {ctx['context_summary']}")

    # Statistics
    print("\nDatabase statistics:")
    stats = db.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print("FR2.3 Database Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_context_database()
