# LOCUS AI Cleaning System - 진행상황

## 📅 마지막 업데이트: 2025-12-03

---

## ✅ 완료된 작업

### 1. Dashboard 통합 및 개선
- **FR1, FR4 페이지 제거** - 불필요한 페이지 삭제
- **FR2, FR3, FR5 통합** - 단일 스크롤 페이지로 통합
- **Sidebar 간소화** - 네비게이션 제거, 브랜딩만 유지
- **4개 센서 카드 레이아웃 (4x1)** 구현:
  1. YOLO 객체 감지
  2. Pose 추정
  3. YAMNet 오디오 분류
  4. 위치 → Zone 매핑
- **한글 UI 번역** - 모든 영어 레이블을 한글로 변경
- **이모티콘 제거** - 프로페셔널한 UI로 개선

**파일:**
- `dashboard/src/pages/UnifiedDashboard.tsx`
- `dashboard/src/pages/UnifiedDashboard.css`
- `dashboard/src/components/Sidebar.tsx`
- `dashboard/src/App.tsx`

---

### 2. 통합 런처 생성
**파일:** `realtime/full_launcher.py`

**기능:**
- 5개 프로세스 자동 시작:
  1. GRU Predictor (ML 추론)
  2. Visual Sensor (YOLO + Flask 비디오 서버)
  3. Audio Sensor (YAMNet)
  4. Context Sensor (시간/공간/자세)
  5. WebSocket Bridge (Dashboard 통신)

**실행:**
```bash
source venv/bin/activate
python realtime/full_launcher.py
```

**접속 주소:**
- 📊 Dashboard: http://localhost:3001
- 📹 Video Feed: http://localhost:5001/video_feed
- 🔌 WebSocket: ws://localhost:8080

---

## 🔄 현재 아키텍처

### 데이터 흐름:
```
[센서들]
  ├─ Visual (YOLO)
  ├─ Audio (YAMNet)
  ├─ Pose (YOLOv11n-pose)
  └─ Context (시간/공간/자세)
         ↓ ZeroMQ (ipc:///tmp/locus_sensors.ipc)
    [GRU Predictor]
         ↓ ZeroMQ (ipc:///tmp/locus_bridge.ipc)
    [WebSocket Bridge]
         ↓ WebSocket (ws://localhost:8080)
    [Dashboard (React)]
```

### 비디오 스트리밍:
```
[Visual Sensor] → Flask (:5001/video_feed) → [Dashboard]
```

---

## ⚠️ 현재 이슈

### MQTT 브로커 아키텍처 문제

**현재 상황:**
- 센서 코드가 **Public MQTT Broker** 사용 중:
  - `mqtt.eclipseprojects.io` (테스트용 공개 브로커)
- LocusBackend (EC2)에 **MQTT Broker가 설치되지 않음**
  - MQTT Client만 존재 (메시지 수신 역할만)

**문제점:**
- Public 브로커는 보안 취약 (누구나 접근 가능)
- 신뢰성 없음 (프로덕션 부적합)
- 친구가 EC2에 브로커 설치 예상했으나 실제로는 클라이언트만 있음

**영향받는 파일:**
- `realtime/sensor_context.py` (line 33, 218)
- `realtime/launcher.py` (line 122)
- `realtime/full_launcher.py` (line 129)

---

## 📋 다음 단계 (TODO)

### 1. MQTT 브로커 설정 (우선순위: 높음)

**옵션 A: EC2에 Mosquitto 설치 (권장)**

친구가 EC2에서 실행:
```bash
# Mosquitto 설치
sudo apt update
sudo apt install mosquitto mosquitto-clients -y

# 서비스 시작
sudo systemctl start mosquitto
sudo systemctl enable mosquitto

# 확인
sudo systemctl status mosquitto
```

AWS 보안그룹 설정:
- Inbound Rule 추가
- Type: Custom TCP
- Port: 1883
- Source: 0.0.0.0/0

코드 변경:
```python
# mqtt.eclipseprojects.io → EC2 주소로 변경
mqtt_broker="ec2-XX-XX-XX-XX.ap-northeast-2.compute.amazonaws.com"
```

**옵션 B: 환경변수로 관리 (개발/배포 분리)**

```python
# sensor_context.py
mqtt_broker = os.getenv("MQTT_BROKER", "mqtt.eclipseprojects.io")
```

실행:
```bash
# 개발
python full_launcher.py

# 배포
MQTT_BROKER=ec2-XX-XX.amazonaws.com python full_launcher.py
```

**옵션 C: 당분간 Public Broker 사용 (임시)**
- 현재 상태 유지
- 프로토타입/데모용으로만 사용
- 나중에 EC2로 전환

---

### 2. 시스템 통합 테스트

**테스트 항목:**
- [ ] full_launcher.py 실행 → 5개 프로세스 정상 시작
- [ ] Dashboard 접속 → WebSocket 연결 확인
- [ ] YOLO 비디오 스트림 표시
- [ ] 센서 데이터 실시간 업데이트 (4개 카드)
- [ ] GRU 예측 결과 표시
- [ ] 청소 실행 시뮬레이션

---

### 3. 배포 준비

**필요 작업:**
- [ ] EC2 Mosquitto 브로커 설정
- [ ] 환경변수 설정 (.env 파일)
- [ ] Docker Compose 작성 (선택사항)
- [ ] 배포 스크립트 작성
- [ ] 모니터링 설정

---

## 📂 주요 파일 구조

```
SE_G10/
├── dashboard/                    # React Dashboard
│   ├── src/
│   │   ├── pages/
│   │   │   ├── UnifiedDashboard.tsx    # 통합 대시보드 (FR2+FR3+FR5)
│   │   │   └── UnifiedDashboard.css
│   │   ├── components/
│   │   │   ├── Sidebar.tsx             # 간소화된 사이드바
│   │   │   └── Sidebar.css
│   │   └── App.tsx                     # 단일 라우트
│   └── package.json
│
├── realtime/                     # Edge 센서 & 예측
│   ├── full_launcher.py          # ✨ 통합 런처 (NEW)
│   ├── launcher.py               # 기존 런처 (센서 4개만)
│   ├── sensor_visual.py          # YOLO + Flask 비디오 서버
│   ├── sensor_audio.py           # YAMNet 오디오 분류
│   ├── sensor_context.py         # 시간/공간/자세 센서
│   ├── gru_predictor.py          # GRU 예측 모델
│   ├── websocket_bridge.py       # ZeroMQ → WebSocket 브릿지
│   ├── mqtt_client.py            # MQTT 클라이언트
│   ├── cleaning_executor.py      # 청소 실행 로직
│   └── zone_manager.py           # Zone 관리
│
├── models/                       # 학습된 모델
│   ├── yolo/
│   │   ├── best.pt               # YOLOv11n (객체 감지)
│   │   └── yolo11n-pose.pt       # YOLOv11n-pose (자세 추정)
│   └── gru/
│       └── gru_model.pth         # GRU 오염도 예측 모델
│
└── PROGRESS.md                   # 이 파일
```

---

## 🔌 포트 사용 현황

| 포트 | 서비스 | 설명 |
|------|--------|------|
| 3001 | Dashboard (Vite) | React 개발 서버 |
| 5001 | Visual Sensor (Flask) | YOLO 비디오 스트림 |
| 8080 | WebSocket Bridge | Dashboard ↔ Edge 통신 |
| 1883 | MQTT Broker | (EC2 설치 필요) |

---

## 📝 참고 사항

### WebSocket 메시지 타입:
- `type: 'visual'` - YOLO 감지 데이터
- `type: 'pose'` - Pose keypoint 데이터
- `type: 'audio'` - YAMNet 오디오 분류
- `type: 'location'` - 위치 (x, y) + zone
- `type: 'synced'` - 센서 동기화 완료
- `prediction: {...}` - GRU 예측 결과 (zone별 오염도)
- `type: 'cleaning_started'` - 청소 시작
- `type: 'cleaning_completed'` - 청소 완료

### MQTT 토픽 구조:
**구독 (Edge → Backend):**
- `home/{home_id}/zones/update`
- `home/{home_id}/control/clean`
- `home/{home_id}/model/command`
- `home/{home_id}/training/start`

**발행 (Backend → Edge):**
- `home/{home_id}/cleaning/status`
- `home/{home_id}/cleaning/result`
- `home/{home_id}/prediction/pollution`
- `home/{home_id}/training/status`
- `edge/{device_id}/status`

---

## 🚀 빠른 시작 가이드

### 1. Dashboard 실행
```bash
cd dashboard
npm run dev
# → http://localhost:3001
```

### 2. 전체 시스템 실행
```bash
cd ..
source venv/bin/activate
python realtime/full_launcher.py
```

### 3. 접속
- Dashboard: http://localhost:3001
- Video Feed: http://localhost:5001/video_feed

### 4. 종료
- `Ctrl+C` → 모든 프로세스 자동 종료

---

## 🐛 알려진 이슈

1. **MQTT Broker 미설정**
   - 현재: Public Broker 사용 중
   - 해결: EC2에 Mosquitto 설치 필요

2. **카메라 연결 실패 시**
   - Video Feed 플레이스홀더 표시
   - 웹캠 권한 확인 필요

3. **WebSocket 재연결**
   - 3초마다 자동 재연결 시도
   - 네트워크 상태 확인 필요

---

## 📞 연락처 & 협업

- **Frontend**: Dashboard (React + TypeScript)
- **Backend**: LocusBackend (Node.js + MQTT Client)
- **Edge**: realtime/ (Python + ZeroMQ + WebSocket)

**다음 미팅 전 준비사항:**
- [ ] EC2 Public DNS 주소 확인
- [ ] Mosquitto 설치 여부 확인
- [ ] 통합 테스트 결과 공유
