"""
컨텍스트 벡터 데이터베이스
시계열 컨텍스트 벡터를 저장하기 위한 SQLite 데이터베이스입니다.
"""

import sqlite3
import json
import os
from typing import List, Dict, Optional
from datetime import datetime


class ContextDatabase:
    """
    컨텍스트 벡터 저장을 위한 SQLite 데이터베이스 클래스입니다.
    다중 모드 컨텍스트 데이터의 시계열 저장 및 조회를 제공합니다.
    """

    def __init__(self, db_path: str = "data/context_vectors.db"):
        """
        컨텍스트 데이터베이스를 초기화합니다.

        Args:
            db_path: SQLite 데이터베이스 파일 경로
        """
        self.db_path = db_path

        # 필요시 데이터 디렉토리를 생성합니다.
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
            print(f"데이터베이스 디렉토리를 생성했습니다: {db_dir}")

        # 데이터베이스 초기화
        self._init_database()
        print(f"컨텍스트 데이터베이스가 초기화되었습니다: {db_path}")

    def _init_database(self):
        """데이터베이스 스키마가 존재하지 않으면 생성합니다."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 컨텍스트 벡터 테이블 (시계열 데이터)
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

        # 시간 기반의 빠른 조회를 위한 인덱스
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON context_vectors(timestamp)
        """)

        # Zone 기반의 빠른 조회를 위한 인덱스
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_zone_id
            ON context_vectors(zone_id)
        """)

        conn.commit()
        conn.close()

    def insert_context(self, context: Dict) -> int:
        """
        컨텍스트 벡터를 데이터베이스에 삽입합니다.

        Args:
            context: 컨텍스트 벡터 딕셔너리

        Returns:
            삽입된 행의 ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 위치 정보 추출
        position = context.get("position")
        position_x = position["x"] if position else None
        position_y = position["y"] if position else None

        # JSON 필드 직렬화
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
        가장 최근의 컨텍스트 벡터를 가져옵니다.

        Args:
            limit: 검색할 최대 컨텍스트 수

        Returns:
            컨텍스트 벡터의 리스트 (최신순)
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
        특정 구역의 컨텍스트 벡터를 가져옵니다.

        Args:
            zone_id: 구역 식별자 (예: "living_room")
            limit: 검색할 최대 컨텍스트 수

        Returns:
            컨텍스트 벡터의 리스트 (최신순)
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
        지정된 시간 범위 내의 컨텍스트 벡터를 가져옵니다.

        Args:
            start_time: 시작 타임스탬프 (Unix 시간)
            end_time: 종료 타임스탬프 (Unix 시간)

        Returns:
            컨텍스트 벡터의 리스트 (오래된순)
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
        데이터베이스 통계를 가져옵니다.

        Returns:
            데이터베이스 통계를 담은 딕셔너리
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 전체 컨텍스트 수
        cursor.execute("SELECT COUNT(*) FROM context_vectors")
        total = cursor.fetchone()[0]

        # 시간 범위
        cursor.execute("""
            SELECT MIN(timestamp), MAX(timestamp)
            FROM context_vectors
        """)
        earliest, latest = cursor.fetchone()

        # 고유한 구역
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
        """모든 컨텍스트 벡터를 삭제합니다 (주의해서 사용)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM context_vectors")
        conn.commit()
        conn.close()
        print("데이터베이스가 초기화되었습니다!")

    def _row_to_context(self, row: sqlite3.Row) -> Dict:
        """
        데이터베이스 행을 컨텍스트 벡터 딕셔너리로 변환합니다.

        Args:
            row: SQLite 행 객체

        Returns:
            컨텍스트 벡터 딕셔너리
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
    """컨텍스트 데이터베이스 작업 테스트"""
    print("=" * 60)
    print("컨텍스트 데이터베이스 테스트")
    print("=" * 60)

    # 테스트 데이터베이스 생성
    db = ContextDatabase("data/test_context.db")

    # 테스트 컨텍스트 생성
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

    # 컨텍스트 삽입
    print("\n테스트 컨텍스트 삽입 중...")
    for ctx in contexts:
        row_id = db.insert_context(ctx)
        print(f"  삽입된 컨텍스트 {row_id}: {ctx['context_summary']}")

    # 최근 컨텍스트 조회
    print("\n최근 컨텍스트:")
    recent = db.get_recent_contexts(limit=5)
    for ctx in recent:
        print(f"  [{ctx['id']}] {ctx['context_summary']}")

    # 구역별 조회
    print("\n거실 컨텍스트:")
    living_room = db.get_contexts_by_zone("living_room")
    for ctx in living_room:
        print(f"  [{ctx['id']}] {ctx['context_summary']}")

    # 통계
    print("\n데이터베이스 통계:")
    stats = db.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print("데이터베이스 테스트 완료!")
    print("=" * 60)


if __name__ == "__main__":
    test_context_database()
