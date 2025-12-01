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
        },

        # ═══════════════════════════════════════
        # 추가 오염 없는 활동
        # ═══════════════════════════════════════

        'reading_book': {
            'description': '책 읽기',
            'zones': ['bedroom', 'living_room', 'balcony'],
            'visual_objects': ['chair', 'sofa', 'lamp', 'bed'],
            'audio_events': [],  # 조용함
            'duration_min': (30, 120),
            'pollution_result': {
                'bedroom': 0.0,
                'living_room': 0.0,
                'balcony': 0.0,
                'kitchen': 0.0
            },
            'notes': '책 읽기는 완전 깨끗'
        },

        'meditating': {
            'description': '명상/요가',
            'zones': ['bedroom', 'living_room', 'balcony'],
            'visual_objects': ['lamp'],
            'audio_events': [],
            'duration_min': (15, 60),
            'pollution_result': {
                'bedroom': 0.0,
                'living_room': 0.0,
                'balcony': 0.0,
                'kitchen': 0.0
            },
            'notes': '명상은 움직임 없음'
        },

        'standing_by_window': {
            'description': '창문 앞에 서 있기',
            'zones': ['bedroom', 'living_room', 'balcony'],
            'visual_objects': ['window'],
            'audio_events': [],
            'duration_min': (5, 20),
            'pollution_result': {
                'bedroom': 0.0,
                'living_room': 0.0,
                'balcony': 0.0,
                'kitchen': 0.0
            },
            'notes': '창밖 보기만 함'
        },

        'listening_music': {
            'description': '음악 듣기',
            'zones': ['bedroom', 'living_room'],
            'visual_objects': ['sofa', 'bed', 'laptop'],
            'audio_events': [],  # 음악 자체는 YAMNet 범위 밖
            'duration_min': (20, 90),
            'pollution_result': {
                'bedroom': 0.0,
                'living_room': 0.0,
                'balcony': 0.0,
                'kitchen': 0.0
            },
            'notes': '음악만 듣기'
        },

        # ═══════════════════════════════════════
        # 추가 오염 발생 활동
        # ═══════════════════════════════════════

        'drinking_beverages': {
            'description': '음료 마시기',
            'zones': ['kitchen', 'living_room', 'bedroom'],
            'visual_objects': ['table', 'chair', 'sofa'],
            'audio_events': ['water_tap', 'dishes'],
            'duration_min': (3, 10),
            'pollution_result': {
                'kitchen': 0.15,  # 물 흘림 가능
                'living_room': 0.2,
                'bedroom': 0.1
            },
            'notes': '음료 흘림 가능'
        },

        'eating_snacks_in_bed': {
            'description': '침대에서 간식 먹기',
            'zones': ['bedroom'],
            'visual_objects': ['bed', 'lamp'],
            'audio_events': ['chewing'],
            'duration_min': (10, 30),
            'pollution_result': {
                'bedroom': 0.75,  # 침대에 부스러기
                'living_room': 0.0
            },
            'notes': '침대에서 먹으면 부스러기 많이 떨어짐'
        },

        'preparing_simple_food': {
            'description': '간단한 음식 준비 (샌드위치 등)',
            'zones': ['kitchen'],
            'visual_objects': ['table', 'chair'],
            'audio_events': ['chopping', 'dishes', 'cutlery'],
            'duration_min': (5, 15),
            'pollution_result': {
                'kitchen': 0.5,  # 요리보다 적음
                'living_room': 0.0
            },
            'notes': '간단한 조리는 중간 오염'
        },

        'opening_packages': {
            'description': '택배 개봉하기',
            'zones': ['living_room', 'bedroom', 'balcony'],
            'visual_objects': ['table', 'chair', 'sofa'],
            'audio_events': [],
            'duration_min': (5, 15),
            'pollution_result': {
                'living_room': 0.3,  # 포장재 부스러기
                'bedroom': 0.25,
                'balcony': 0.2
            },
            'notes': '포장재, 스티로폼 부스러기'
        },

        'exercising_indoor': {
            'description': '실내 운동',
            'zones': ['living_room', 'bedroom', 'balcony'],
            'visual_objects': ['lamp'],
            'audio_events': ['footsteps'],
            'duration_min': (20, 60),
            'pollution_result': {
                'living_room': 0.2,  # 땀, 발자국
                'bedroom': 0.15,
                'balcony': 0.1
            },
            'notes': '운동 중 땀, 발자국 먼지'
        },

        # ═══════════════════════════════════════
        # 추가 청소 활동
        # ═══════════════════════════════════════

        'wiping_table': {
            'description': '테이블 닦기',
            'zones': ['kitchen', 'living_room'],
            'visual_objects': ['table', 'chair'],
            'audio_events': ['water_tap', 'sink'],
            'duration_min': (3, 10),
            'pollution_result': {
                'kitchen': -0.4,  # 테이블 청소
                'living_room': -0.3
            },
            'notes': '테이블만 닦기'
        },

        'mopping_floor': {
            'description': '걸레질하기',
            'zones': ['kitchen', 'living_room', 'bedroom', 'balcony'],
            'visual_objects': ['door'],
            'audio_events': ['water_tap', 'sink', 'footsteps'],
            'duration_min': (15, 40),
            'pollution_result': {
                'kitchen': -0.9,  # 걸레질은 청소기보다 강력
                'living_room': -0.85,
                'bedroom': -0.85,
                'balcony': -0.7
            },
            'notes': '물걸레질은 오염 제거 효과 높음'
        },

        'organizing_items': {
            'description': '물건 정리하기',
            'zones': ['bedroom', 'living_room', 'kitchen'],
            'visual_objects': ['wardrobe', 'chair', 'table'],
            'audio_events': ['footsteps'],
            'duration_min': (10, 30),
            'pollution_result': {
                'bedroom': -0.2,  # 약간 청소 효과
                'living_room': -0.15,
                'kitchen': -0.1
            },
            'notes': '정리정돈 중 약간 청소'
        },

        # ═══════════════════════════════════════
        # 혼합 활동 (다양한 조합)
        # ═══════════════════════════════════════

        'hosting_guests': {
            'description': '손님 맞이',
            'zones': ['living_room', 'kitchen'],
            'visual_objects': ['sofa', 'chair', 'table', 'tv'],
            'audio_events': ['door', 'footsteps', 'speech', 'dishes'],
            'duration_min': (60, 180),
            'pollution_result': {
                'living_room': 0.7,  # 손님 많으면 오염 높음
                'kitchen': 0.5
            },
            'notes': '손님 방문 시 발자국, 음식 등'
        },

        'watering_plants': {
            'description': '화분에 물주기',
            'zones': ['balcony', 'living_room'],
            'visual_objects': ['potted_plant', 'window'],
            'audio_events': ['water_tap', 'footsteps'],
            'duration_min': (5, 15),
            'pollution_result': {
                'balcony': 0.3,  # 물 흘림
                'living_room': 0.15
            },
            'notes': '화분 물주기 시 물 흘림'
        },

        'pet_activities': {
            'description': '반려동물 활동 (먹이, 놀이)',
            'zones': ['kitchen', 'living_room', 'balcony'],
            'visual_objects': ['table', 'sofa', 'door'],
            'audio_events': ['dishes', 'footsteps'],
            'duration_min': (10, 30),
            'pollution_result': {
                'kitchen': 0.6,  # 사료 흘림
                'living_room': 0.55,  # 털, 발자국
                'balcony': 0.4
            },
            'notes': '반려동물 사료, 털, 발자국'
        },

        'crafting_hobbies': {
            'description': '공예/취미 활동',
            'zones': ['living_room', 'bedroom'],
            'visual_objects': ['table', 'chair'],
            'audio_events': [],
            'duration_min': (30, 120),
            'pollution_result': {
                'living_room': 0.45,  # 재료 부스러기
                'bedroom': 0.35
            },
            'notes': '공예 재료 부스러기'
        },

        'changing_clothes': {
            'description': '옷 갈아입기',
            'zones': ['bedroom'],
            'visual_objects': ['wardrobe', 'bed', 'chair'],
            'audio_events': ['footsteps'],
            'duration_min': (5, 15),
            'pollution_result': {
                'bedroom': 0.1,  # 먼지, 옷 섬유
                'living_room': 0.0
            },
            'notes': '옷 먼지, 섬유 약간'
        },

        'drying_laundry': {
            'description': '빨래 널기',
            'zones': ['balcony', 'living_room'],
            'visual_objects': ['window', 'door'],
            'audio_events': ['footsteps'],
            'duration_min': (10, 20),
            'pollution_result': {
                'balcony': 0.25,  # 물 떨어짐
                'living_room': 0.15
            },
            'notes': '젖은 빨래에서 물 떨어짐'
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
