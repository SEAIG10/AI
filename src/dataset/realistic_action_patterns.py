"""
현실적인 행동 패턴 라이브러리
실제 학습된 YOLO(14 classes) + YAMNet(17 classes)만 사용
"""

import numpy as np
from typing import Dict, List, Tuple

# 실제 학습된 클래스
YOLO_CLASSES = [
    'bed', 'sofa', 'chair', 'table', 'lamp', 'tv', 'laptop',
    'wardrobe', 'window', 'door', 'potted_plant', 'photo_frame',
    'solid_waste', 'liquid_stain'
]

YAMNET_CLASSES = [
    'door', 'dishes', 'cutlery', 'chopping', 'frying', 'microwave',
    'blender', 'water_tap', 'sink', 'toilet_flush', 'telephone',
    'chewing', 'speech', 'television', 'footsteps', 'vacuum', 'hair_dryer'
]

# COCO pretrained (fine-tuning 후에도 사용 가능, 성능 약간 감소)
COCO_PERSON_AVAILABLE = True  # person 클래스는 사용 가능


class RealisticActionPattern:
    """
    실제 생활에서 발생하는 행동 패턴 정의

    각 패턴은:
    - visual_objects: 감지되는 YOLO 객체들
    - audio_events: 감지되는 YAMNet 소리들
    - typical_duration: 보통 지속 시간 (분)
    - pollution_result: 이 행동 후 발생하는 오염 (zone: probability)
    """

    PATTERNS = {
        # ═══════════════════════════════════════
        # 식사 관련 (높은 오염 확률)
        # ═══════════════════════════════════════

        'cooking': {
            'description': '주방에서 요리하기',
            'zones': ['kitchen'],
            'visual_objects': ['table', 'chair'],  # 주방 테이블
            'audio_events': ['chopping', 'frying', 'water_tap', 'sink'],
            'duration_min': (10, 30),
            'pollution_result': {
                'kitchen': 0.7,  # 요리 후 주방 바닥 오염 70%
                'living_room': 0.1
            },
            'notes': '칼질, 볶음 소리 → 음식물 튀김 → 바닥 오염'
        },

        'eating_at_table': {
            'description': '식탁에서 식사하기',
            'zones': ['kitchen', 'living_room'],
            'visual_objects': ['table', 'chair'],
            'audio_events': ['chewing', 'dishes', 'cutlery', 'speech'],
            'duration_min': (15, 45),
            'pollution_result': {
                'living_room': 0.85,  # 식사 중 부스러기 떨어뜨림
                'kitchen': 0.4
            },
            'notes': '씹기 + 식기 소리 → 음식 섭취 중 → 부스러기 발생'
        },

        'snacking': {
            'description': '간식 먹기 (TV 보면서)',
            'zones': ['living_room'],
            'visual_objects': ['sofa', 'tv', 'table'],
            'audio_events': ['chewing', 'television'],
            'duration_min': (10, 30),
            'pollution_result': {
                'living_room': 0.6,  # 간식 부스러기
                'bedroom': 0.1
            },
            'notes': 'TV + 씹기 → 과자 부스러기'
        },

        'washing_dishes': {
            'description': '설거지하기',
            'zones': ['kitchen'],
            'visual_objects': ['sink'],  # 싱크대 (table 근처)
            'audio_events': ['water_tap', 'sink', 'dishes'],
            'duration_min': (5, 15),
            'pollution_result': {
                'kitchen': 0.3,  # 물 튐
                'living_room': 0.0
            },
            'notes': '물 소리 + 설거지 → 물 튐 가능'
        },

        # ═══════════════════════════════════════
        # 여가 활동 (중간~낮은 오염)
        # ═══════════════════════════════════════

        'watching_tv': {
            'description': 'TV 시청 (간식 없이)',
            'zones': ['living_room'],
            'visual_objects': ['sofa', 'tv'],
            'audio_events': ['television'],
            'duration_min': (30, 120),
            'pollution_result': {
                'living_room': 0.1,  # 거의 깨끗
                'kitchen': 0.0
            },
            'notes': 'TV만 보면 깨끗'
        },

        'working_on_laptop': {
            'description': '노트북 작업',
            'zones': ['bedroom', 'living_room'],
            'visual_objects': ['chair', 'table', 'laptop'],
            'audio_events': [],  # 조용함
            'duration_min': (60, 240),
            'pollution_result': {
                'bedroom': 0.05,  # 거의 깨끗
                'living_room': 0.05
            },
            'notes': '책상 작업은 깨끗함'
        },

        'phone_call': {
            'description': '전화 통화',
            'zones': ['living_room', 'bedroom', 'balcony'],
            'visual_objects': ['sofa', 'chair', 'bed'],
            'audio_events': ['telephone', 'speech'],
            'duration_min': (5, 30),
            'pollution_result': {
                'living_room': 0.0,
                'bedroom': 0.0,
                'balcony': 0.0
            },
            'notes': '통화만 하면 깨끗'
        },

        # ═══════════════════════════════════════
        # 수면/휴식 (오염 없음)
        # ═══════════════════════════════════════

        'sleeping': {
            'description': '잠자기',
            'zones': ['bedroom'],
            'visual_objects': ['bed', 'lamp'],
            'audio_events': [],  # 조용함
            'duration_min': (360, 540),  # 6-9시간
            'pollution_result': {
                'bedroom': 0.0,  # 깨끗
                'living_room': 0.0
            },
            'notes': '잠자면 오염 없음'
        },

        'resting_on_sofa': {
            'description': '소파에서 쉬기',
            'zones': ['living_room'],
            'visual_objects': ['sofa', 'lamp'],
            'audio_events': [],
            'duration_min': (15, 60),
            'pollution_result': {
                'living_room': 0.0,
                'bedroom': 0.0
            },
            'notes': '누워있기만 하면 깨끗'
        },

        # ═══════════════════════════════════════
        # 청소 활동 (오염 제거!)
        # ═══════════════════════════════════════

        'vacuuming': {
            'description': '청소기 돌리기',
            'zones': ['living_room', 'bedroom', 'kitchen'],
            'visual_objects': ['chair', 'table', 'sofa', 'bed'],
            'audio_events': ['vacuum', 'footsteps'],
            'duration_min': (10, 30),
            'pollution_result': {
                # 청소하면 오염 감소!
                'living_room': -0.8,  # 오염 80% 제거
                'bedroom': -0.8,
                'kitchen': -0.7
            },
            'notes': '청소기 소리 → 오염 제거'
        },

        # ═══════════════════════════════════════
        # 기타 활동
        # ═══════════════════════════════════════

        'entering_room': {
            'description': '방 들어가기',
            'zones': ['living_room', 'bedroom', 'kitchen', 'balcony'],
            'visual_objects': ['door'],
            'audio_events': ['door', 'footsteps'],
            'duration_min': (1, 3),
            'pollution_result': {
                'living_room': 0.05,  # 발자국 먼지
                'bedroom': 0.05,
                'kitchen': 0.05,
                'balcony': 0.1  # 밖에서 들어오면 더 오염
            },
            'notes': '문 열림 + 발자국 → 약간 먼지'
        },

        'using_bathroom': {
            'description': '화장실 사용',
            'zones': ['bathroom'],  # Note: bathroom이 4개 zone에 없음!
            'visual_objects': ['door'],
            'audio_events': ['toilet_flush', 'water_tap', 'sink', 'footsteps'],
            'duration_min': (3, 10),
            'pollution_result': {
                # 화장실은 4개 zone에 포함 안 됨
                'living_room': 0.0,
                'bedroom': 0.0,
                'kitchen': 0.0,
                'balcony': 0.0
            },
            'notes': '화장실은 학습 zone 밖'
        },

        'using_microwave': {
            'description': '전자레인지 사용',
            'zones': ['kitchen'],
            'visual_objects': ['table'],
            'audio_events': ['microwave', 'footsteps'],
            'duration_min': (2, 5),
            'pollution_result': {
                'kitchen': 0.2,  # 약간 오염 (음식 데우기)
                'living_room': 0.0
            },
            'notes': '전자레인지만 쓰면 오염 적음'
        },

        'using_blender': {
            'description': '믹서기 사용',
            'zones': ['kitchen'],
            'visual_objects': ['table'],
            'audio_events': ['blender', 'water_tap'],
            'duration_min': (3, 10),
            'pollution_result': {
                'kitchen': 0.4,  # 믹서기 튐
                'living_room': 0.0
            },
            'notes': '믹서기 → 액체/고체 튐'
        }
    }

    @classmethod
    def get_pattern(cls, pattern_name: str) -> Dict:
        """패턴 정보 반환"""
        return cls.PATTERNS.get(pattern_name)

    @classmethod
    def get_all_patterns(cls) -> List[str]:
        """모든 패턴 이름 반환"""
        return list(cls.PATTERNS.keys())

    @classmethod
    def get_high_pollution_patterns(cls, threshold: float = 0.5) -> List[str]:
        """높은 오염 확률을 가진 패턴 반환"""
        result = []
        for name, pattern in cls.PATTERNS.items():
            max_pollution = max(pattern['pollution_result'].values())
            if max_pollution >= threshold:
                result.append(name)
        return result

    @classmethod
    def get_patterns_by_zone(cls, zone: str) -> List[str]:
        """특정 구역에서 발생하는 패턴 반환"""
        result = []
        for name, pattern in cls.PATTERNS.items():
            if zone in pattern['zones']:
                result.append(name)
        return result


# 테스트
if __name__ == "__main__":
    print("=" * 60)
    print("Realistic Action Patterns")
    print("=" * 60)

    print(f"\n총 패턴 수: {len(RealisticActionPattern.get_all_patterns())}")

    print("\n높은 오염 패턴:")
    for pattern_name in RealisticActionPattern.get_high_pollution_patterns():
        pattern = RealisticActionPattern.get_pattern(pattern_name)
        print(f"  - {pattern_name}: {pattern['description']}")
        print(f"    오염 결과: {pattern['pollution_result']}")

    print("\n주방에서 발생하는 패턴:")
    for pattern_name in RealisticActionPattern.get_patterns_by_zone('kitchen'):
        print(f"  - {pattern_name}")
