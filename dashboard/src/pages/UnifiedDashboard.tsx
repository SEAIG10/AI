import React, { useState, useEffect, useRef } from 'react';
import './UnifiedDashboard.css';

const WEBSOCKET_URL = 'ws://localhost:8080';

// YOLO 클래스 (14개) - realtime/utils.py와 동일
const YOLO_CLASSES = [
  "bed", "sofa", "chair", "table", "lamp", "tv", "laptop",
  "wardrobe", "window", "door", "potted plant", "photo frame",
  "solid_waste", "liquid_stain"
];

// Audio 클래스 (17개) - src/audio_recognition/yamnet_processor.py와 동일
const AUDIO_CLASSES = [
  "door", "dishes", "cutlery", "chopping", "frying", "microwave", "blender",
  "water_tap", "sink", "toilet_flush", "telephone", "chewing", "speech",
  "television", "footsteps", "vacuum", "hair_dryer"
];

// Type definitions
interface YoloDetection {
  label: string;
  confidence: number;
}

interface AudioClass {
  label: string;
  score: number;
}

interface PredictionResult {
  zone: string;
  score: number;
}

interface ZonePollution {
  zone: string;
  pollution: number;
  priority: number;
}

interface CleaningTask {
  time: string;
  zone: string;
  action: string;
  duration: number;
  status: 'completed' | 'in_progress' | 'pending';
}

const UnifiedDashboard: React.FC = () => {
  // WebSocket state
  const [wsConnected, setWsConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  // FR2 - Visual & Audio states
  const [yoloDetections, setYoloDetections] = useState<YoloDetection[]>([]);
  const [poseKeypoints, setPoseKeypoints] = useState<number>(0); // Number of detected keypoints
  const [audioClasses, setAudioClasses] = useState<AudioClass[]>([]);
  const [currentLocation, setCurrentLocation] = useState({ x: 0, y: 0, zone: 'unknown' });

  // Temporal filtering states
  const detectionMapRef = useRef<Map<string, { confidence: number, lastSeen: number }>>(new Map());
  const poseHistoryRef = useRef<number[]>([]);
  const audioHistoryRef = useRef<AudioClass[][]>([]);
  const [stats, setStats] = useState({
    visualMsgCount: 0,
    audioMsgCount: 0,
    poseMsgCount: 0,
    locationMsgCount: 0,
    syncedCount: 0,
    latencyMs: 0
  });

  // FR3 - GRU Prediction states
  const [predictions, setPredictions] = useState<PredictionResult[]>([]);
  const [explanation, setExplanation] = useState(
    'WebSocket 브릿지 연결 대기 중...'
  );

  // FR5 - Cleaning states
  const [currentAction, setCurrentAction] = useState(
    'WebSocket 연결 대기 중...'
  );
  const [zonePollutions, setZonePollutions] = useState<ZonePollution[]>([
    { zone: 'balcony', pollution: 0.65, priority: 1 },
    { zone: 'bedroom', pollution: 0.40, priority: 2 },
    { zone: 'kitchen', pollution: 0.25, priority: 3 },
    { zone: 'living_room', pollution: 0.15, priority: 4 }
  ]);
  const [cleaningTimeline, setCleaningTimeline] = useState<CleaningTask[]>([
    { time: '14:30', zone: 'balcony', action: 'Deep clean', duration: 15, status: 'in_progress' },
    { time: '14:45', zone: 'kitchen', action: 'Standard clean', duration: 12, status: 'pending' },
    { time: '14:57', zone: 'bedroom', action: 'Light clean', duration: 8, status: 'pending' }
  ]);
  const [estimatedTime, setEstimatedTime] = useState({
    total: 35,
    remaining: 30,
    currentTask: 5
  });

  // WebSocket connection
  useEffect(() => {
    const connectWebSocket = () => {
      try {
        const ws = new WebSocket(WEBSOCKET_URL);
        wsRef.current = ws;

        ws.onopen = () => {
          console.log('[Unified WS] Connected');
          setWsConnected(true);
          setExplanation('실시간 센서 데이터 수신 대기 중...');
          setCurrentAction('청소 명령 대기 중...');
        };

        ws.onmessage = (event) => {
          try {
            const message = JSON.parse(event.data);
            if (message.type === 'welcome') return;

            const timestamp = new Date().toLocaleTimeString();

            // Visual sensor - Parse 14-dim multi-hot vector with temporal filtering
            if (message.type === 'visual' && message.data) {
              setStats(prev => ({ ...prev, visualMsgCount: prev.visualMsgCount + 1 }));

              const visualVec = message.data as number[];
              const now = Date.now();
              const detectionMap = detectionMapRef.current;

              // Update detection map
              for (let i = 0; i < Math.min(visualVec.length, YOLO_CLASSES.length); i++) {
                const label = YOLO_CLASSES[i];
                if (visualVec[i] > 0) {
                  // 감지된 객체: 업데이트
                  detectionMap.set(label, {
                    confidence: visualVec[i],
                    lastSeen: now
                  });
                }
              }

              // Remove old detections (not seen for 3 seconds)
              for (const [label, info] of detectionMap.entries()) {
                if (now - info.lastSeen > 3000) {
                  detectionMap.delete(label);
                }
              }

              // Convert map to array for display
              const filteredDetections: YoloDetection[] = Array.from(detectionMap.entries()).map(([label, info]) => ({
                label,
                confidence: info.confidence
              }));

              setYoloDetections(filteredDetections);
            }

            // Pose sensor - Count non-zero keypoints with moving average smoothing
            if (message.type === 'pose' && message.data) {
              setStats(prev => ({ ...prev, poseMsgCount: prev.poseMsgCount + 1 }));

              const poseVec = message.data as number[];
              const detectedKeypoints = poseVec.filter(val => val !== 0).length;
              const keypointCount = Math.floor(detectedKeypoints / 3);

              // Add to history (max 5 values)
              const poseHistory = poseHistoryRef.current;
              poseHistory.push(keypointCount);
              if (poseHistory.length > 5) {
                poseHistory.shift();
              }

              // Calculate moving average
              const avgKeypoints = Math.round(
                poseHistory.reduce((sum, val) => sum + val, 0) / poseHistory.length
              );

              setPoseKeypoints(avgKeypoints);
            }

            // Audio sensor - Parse 17-dim probability vector with smoothing
            if (message.type === 'audio' && message.data) {
              setStats(prev => ({ ...prev, audioMsgCount: prev.audioMsgCount + 1 }));

              const audioProbs = message.data as number[];

              // Get top 3 classes
              const classesWithScores = audioProbs
                .map((prob, idx) => ({
                  label: AUDIO_CLASSES[idx] || `Class ${idx}`,
                  score: prob
                }))
                .sort((a, b) => b.score - a.score)
                .slice(0, 3); // Top 3

              // Add to history (max 3 samples)
              const audioHistory = audioHistoryRef.current;
              audioHistory.push(classesWithScores);
              if (audioHistory.length > 3) {
                audioHistory.shift();
              }

              // Calculate moving average for each class
              const labelMap = new Map<string, number[]>();
              for (const sample of audioHistory) {
                for (const cls of sample) {
                  if (!labelMap.has(cls.label)) {
                    labelMap.set(cls.label, []);
                  }
                  labelMap.get(cls.label)!.push(cls.score);
                }
              }

              // Average scores
              const smoothedClasses: AudioClass[] = Array.from(labelMap.entries())
                .map(([label, scores]) => ({
                  label,
                  score: scores.reduce((sum, s) => sum + s, 0) / scores.length
                }))
                .sort((a, b) => b.score - a.score)
                .slice(0, 3); // Top 3 after smoothing

              setAudioClasses(smoothedClasses);
            }

            // Location sensor - Use actual coordinates and zone
            if (message.type === 'location' || message.x !== undefined) {
              setStats(prev => ({ ...prev, locationMsgCount: prev.locationMsgCount + 1 }));
              setCurrentLocation({
                x: message.x ?? 0,
                y: message.y ?? 0,
                zone: message.zone || 'unknown'
              });
            }

            // Synced data
            if (message.type === 'synced') {
              setStats(prev => ({
                ...prev,
                syncedCount: prev.syncedCount + 1,
                latencyMs: message.latencyMs ?? prev.latencyMs
              }));
            }

            // GRU prediction result
            if (message.prediction) {
              const predResults: PredictionResult[] = Object.entries(message.prediction).map(([zone, score]) => ({
                zone,
                score: score as number
              })).sort((a, b) => b.score - a.score);

              setPredictions(predResults);

              // Update floor map
              const pollutions = predResults.map((p, idx) => ({
                zone: p.zone,
                pollution: p.score,
                priority: idx + 1
              }));
              setZonePollutions(pollutions);

              const highest = predResults[0];
              if (highest && highest.score > 0.5) {
                setExplanation(`${highest.zone}에서 높은 오염도(${(highest.score * 100).toFixed(0)}%)가 감지되었습니다.`);
              } else {
                setExplanation('모든 구역의 오염도가 낮은 상태입니다.');
              }
            }

            // Cleaning started
            if (message.type === 'cleaning_started' || message.zone) {
              const zone = message.zone || 'Unknown';
              setCurrentAction(`${zone} 청소 중 (${timestamp} 시작)`);
              setCleaningTimeline(prev => prev.map(task =>
                task.zone === zone ? { ...task, status: 'in_progress' as const } : task
              ));
            }

            // Cleaning completed
            if (message.type === 'cleaning_completed') {
              const zone = message.zone || 'Unknown';
              setCleaningTimeline(prev => prev.map(task =>
                task.zone === zone ? { ...task, status: 'completed' as const } : task
              ));
              setZonePollutions(prev => prev.map(zp =>
                zp.zone === zone ? { ...zp, pollution: 0 } : zp
              ));
            }

          } catch (err) {
            console.error('[Unified WS] Parsing error:', err);
          }
        };

        ws.onerror = (error) => {
          console.error('[Unified WS] Error:', error);
          setWsConnected(false);
          setExplanation('WebSocket 연결 오류');
        };

        ws.onclose = () => {
          console.log('[Unified WS] Disconnected');
          setWsConnected(false);
          setExplanation('WebSocket 연결이 끊어졌습니다. 재연결 시도 중...');
          setCurrentAction('WebSocket 연결 끊김 - 재연결 시도 중...');
          wsRef.current = null;
          setTimeout(connectWebSocket, 3000);
        };

      } catch (err) {
        console.error('[Unified WS] Connection error:', err);
        setExplanation('WebSocket 브릿지에 연결할 수 없습니다.');
      }
    };

    connectWebSocket();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  const getPollutionColor = (pollution: number) => {
    if (pollution >= 0.6) return 'var(--danger)';
    if (pollution >= 0.3) return 'var(--warning)';
    return 'var(--success)';
  };

  return (
    <div className="unified-dashboard">
      {/* Header */}
      <div className="dashboard-header">
        <div>
          <h1 className="page-title">LOCUS AI Cleaning Dashboard</h1>
          <p className="page-subtitle">
            실시간 센서 융합 · GRU 예측 · 정책 기반 청소 실행
          </p>
        </div>
        <div className="connection-status">
          <div style={{
            width: '12px',
            height: '12px',
            borderRadius: '50%',
            background: wsConnected ? 'var(--success)' : 'var(--danger)',
            animation: wsConnected ? 'pulse 2s ease-in-out infinite' : 'none'
          }}></div>
          <span>{wsConnected ? '연결됨' : '연결 끊김'}</span>
        </div>
      </div>

      {/* SECTION 1: Real-time Visual & Audio Context (FR2) */}
      <section className="section-visual-audio">
        <h2 className="section-title">FR2 · 시각 & 청각 컨텍스트 인식</h2>

        <div className="grid grid-4">
          {/* 1. YOLO Object Detection */}
          <div className="card">
            <div className="card-header">YOLO 객체 감지</div>
            <div className="video-container" style={{ marginBottom: '0.75rem' }}>
              <img
                src="http://localhost:5001/video_feed"
                alt="YOLO Live Stream"
                style={{ width: '100%', height: 'auto', borderRadius: '12px', maxHeight: '180px', objectFit: 'cover' }}
                onError={(e) => {
                  e.currentTarget.style.display = 'none';
                  e.currentTarget.nextElementSibling?.setAttribute('style', 'display: flex');
                }}
              />
              <div className="video-placeholder" style={{ display: 'none', minHeight: '180px' }}>
                <p style={{ fontSize: '0.85rem' }}>카메라 대기 중</p>
              </div>
            </div>
            <div className="detection-list">
              {yoloDetections.map((det, idx) => (
                <div key={idx} className="detection-item">
                  <span className="detection-label">{det.label}</span>
                  <div className="detection-bar-container">
                    <div
                      className="detection-bar"
                      style={{ width: `${det.confidence * 100}%` }}
                    />
                  </div>
                  <span className="detection-value">{(det.confidence * 100).toFixed(0)}%</span>
                </div>
              ))}
            </div>
          </div>

          {/* 2. Pose Estimation */}
          <div className="card">
            <div className="card-header">Pose 추정</div>
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '240px', gap: '1rem' }}>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: '3rem', fontWeight: '700', color: 'var(--accent)' }}>
                  {poseKeypoints}
                </div>
                <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '0.5rem' }}>
                  Keypoints 감지됨
                </div>
              </div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', textAlign: 'center', marginTop: '0.5rem' }}>
                {stats.poseMsgCount} 메시지 수신
              </div>
            </div>
          </div>

          {/* 3. YAMNet Audio Classification */}
          <div className="card">
            <div className="card-header">YAMNet 오디오 분류</div>
            <div className="audio-viz">
              {audioClasses.map((audio, idx) => (
                <div key={idx} className="audio-class-item" style={{ height: `${audio.score * 170}px` }}>
                  <div className="audio-bar" style={{ height: '100%' }} />
                  <div className="audio-label">{audio.label}</div>
                  <div className="audio-score">{(audio.score * 100).toFixed(0)}%</div>
                </div>
              ))}
            </div>
          </div>

          {/* 4. Location to Zone Mapping */}
          <div className="card">
            <div className="card-header">위치 → Zone 매핑</div>
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '240px', gap: '1rem' }}>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: '2rem', fontWeight: '700', color: 'var(--accent)', marginBottom: '0.75rem' }}>
                  {currentLocation.zone}
                </div>
                <div style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', fontFamily: 'monospace' }}>
                  x: {currentLocation.x.toFixed(2)}
                </div>
                <div style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', fontFamily: 'monospace' }}>
                  y: {currentLocation.y.toFixed(2)}
                </div>
              </div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', textAlign: 'center', marginTop: '0.5rem' }}>
                {stats.locationMsgCount} 위치 업데이트
              </div>
            </div>
          </div>
        </div>

        {/* ROS Sensor Sync */}
        <div className="card" style={{ marginTop: '1rem' }}>
          <div className="card-header">ROS 센서 동기화 - 센서 융합</div>
          <p style={{ color: 'var(--text-secondary)', marginBottom: '1rem', fontSize: '0.875rem' }}>
            시각, 청각, 자세 센서 스트림을 <span style={{ color: 'var(--accent)', fontWeight: '600' }}>±500ms 허용 오차</span> 윈도우로 동기화하여 Context Encoder로 전송
          </p>

          {/* Sensor sync timeline */}
          <div style={{ background: '#f8f9fa', padding: '1.25rem', borderRadius: '12px', marginBottom: '1rem' }}>
            <div style={{ fontFamily: 'monospace', fontSize: '0.85rem', lineHeight: '1.8' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <span style={{ width: '80px', color: 'var(--text-secondary)' }}>시각</span>
                <div style={{ flex: 1, height: '2px', background: '#e0e0e0', position: 'relative' }}>
                  <div style={{
                    position: 'absolute',
                    left: '40%',
                    top: '-6px',
                    width: '12px',
                    height: '12px',
                    borderRadius: '50%',
                    background: 'var(--accent)'
                  }}></div>
                </div>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>{stats.visualMsgCount} msgs</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <span style={{ width: '80px', color: 'var(--text-secondary)' }}>청각</span>
                <div style={{ flex: 1, height: '2px', background: '#e0e0e0', position: 'relative' }}>
                  <div style={{
                    position: 'absolute',
                    left: '45%',
                    top: '-6px',
                    width: '12px',
                    height: '12px',
                    borderRadius: '50%',
                    background: 'var(--accent)'
                  }}></div>
                </div>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>{stats.audioMsgCount} msgs</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <span style={{ width: '80px', color: 'var(--text-secondary)' }}>자세</span>
                <div style={{ flex: 1, height: '2px', background: '#e0e0e0', position: 'relative' }}>
                  <div style={{
                    position: 'absolute',
                    left: '42%',
                    top: '-6px',
                    width: '12px',
                    height: '12px',
                    borderRadius: '50%',
                    background: 'var(--accent)'
                  }}></div>
                </div>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>{stats.poseMsgCount} msgs</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <span style={{ width: '80px', color: 'var(--text-secondary)' }}>위치</span>
                <div style={{ flex: 1, height: '2px', background: '#e0e0e0', position: 'relative' }}>
                  <div style={{
                    position: 'absolute',
                    left: '44%',
                    top: '-6px',
                    width: '12px',
                    height: '12px',
                    borderRadius: '50%',
                    background: 'var(--accent)'
                  }}></div>
                </div>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>{stats.locationMsgCount} msgs</span>
              </div>
              <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                marginTop: '0.75rem',
                paddingTop: '0.75rem',
                borderTop: '2px dashed var(--accent)'
              }}>
                <span style={{ width: '80px', color: 'var(--accent)', fontWeight: '700' }}>동기화</span>
                <div style={{ flex: 1, height: '3px', background: 'var(--accent)', position: 'relative', borderRadius: '2px' }}>
                  <div style={{
                    position: 'absolute',
                    left: '43%',
                    top: '-7px',
                    width: '16px',
                    height: '16px',
                    borderRadius: '50%',
                    background: 'var(--accent)',
                    border: '3px solid white',
                    boxShadow: '0 2px 8px rgba(165, 0, 52, 0.3)'
                  }}></div>
                </div>
                <span style={{ fontSize: '0.8rem', color: 'var(--accent)', fontWeight: '600' }}>{stats.syncedCount} synced</span>
              </div>

              {/* Pipeline flow */}
              <div style={{ marginTop: '1.25rem', paddingTop: '1rem', borderTop: '1px solid #e0e0e0' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
                  <div style={{
                    padding: '0.75rem 1rem',
                    background: 'white',
                    border: '2px solid var(--accent)',
                    borderRadius: '10px',
                    textAlign: 'center',
                    minWidth: '110px'
                  }}>
                    <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', marginBottom: '0.2rem' }}>단계 1</div>
                    <div style={{ fontSize: '0.8rem', fontWeight: '600', color: 'var(--accent)' }}>동기화 데이터</div>
                    <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>4개 센서</div>
                  </div>

                  <div style={{ fontSize: '1.25rem', color: 'var(--accent)' }}>→</div>

                  <div style={{
                    padding: '0.75rem 1rem',
                    background: 'white',
                    border: '2px solid #e0e0e0',
                    borderRadius: '10px',
                    textAlign: 'center',
                    minWidth: '110px'
                  }}>
                    <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', marginBottom: '0.2rem' }}>단계 2</div>
                    <div style={{ fontSize: '0.8rem', fontWeight: '600', color: 'var(--text-primary)' }}>컨텍스트 인코더</div>
                    <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>어텐션 (160차원)</div>
                  </div>

                  <div style={{ fontSize: '1.25rem', color: 'var(--text-secondary)' }}>→</div>

                  <div style={{
                    padding: '0.75rem 1rem',
                    background: 'white',
                    border: '2px solid #e0e0e0',
                    borderRadius: '10px',
                    textAlign: 'center',
                    minWidth: '110px'
                  }}>
                    <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', marginBottom: '0.2rem' }}>단계 3</div>
                    <div style={{ fontSize: '0.8rem', fontWeight: '600', color: 'var(--text-primary)' }}>버퍼</div>
                    <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>30 타임스텝</div>
                  </div>

                  <div style={{ fontSize: '1.25rem', color: 'var(--text-secondary)' }}>→</div>

                  <div style={{
                    padding: '0.75rem 1rem',
                    background: 'white',
                    border: '2px solid #e0e0e0',
                    borderRadius: '10px',
                    textAlign: 'center',
                    minWidth: '110px'
                  }}>
                    <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', marginBottom: '0.2rem' }}>단계 4</div>
                    <div style={{ fontSize: '0.8rem', fontWeight: '600', color: 'var(--text-primary)' }}>GRU 모델</div>
                    <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>FedPer</div>
                  </div>

                  <div style={{ fontSize: '1.25rem', color: 'var(--accent)' }}>→</div>

                  <div style={{
                    padding: '0.75rem 1rem',
                    background: 'var(--accent)',
                    border: '2px solid var(--accent)',
                    borderRadius: '10px',
                    textAlign: 'center',
                    minWidth: '110px'
                  }}>
                    <div style={{ fontSize: '0.65rem', color: 'rgba(255,255,255,0.8)', marginBottom: '0.2rem' }}>단계 5</div>
                    <div style={{ fontSize: '0.8rem', fontWeight: '600', color: 'white' }}>오염도 예측</div>
                    <div style={{ fontSize: '0.65rem', color: 'rgba(255,255,255,0.8)', marginTop: '0.2rem' }}>4개 구역</div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Stats */}
          <div className="grid grid-4" style={{ gridTemplateColumns: 'repeat(6, 1fr)' }}>
            <div className="stat-card">
              <div className="stat-label">시각</div>
              <div className="stat-value-small">{stats.visualMsgCount}</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">자세</div>
              <div className="stat-value-small">{stats.poseMsgCount}</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">청각</div>
              <div className="stat-value-small">{stats.audioMsgCount}</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">위치</div>
              <div className="stat-value-small">{stats.locationMsgCount}</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">동기화</div>
              <div className="stat-value-small" style={{ color: 'var(--accent)' }}>{stats.syncedCount}</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">지연시간</div>
              <div className="stat-value-small">{stats.latencyMs}ms</div>
            </div>
          </div>
        </div>
      </section>

      {/* SECTION 2: GRU Prediction & Floor Map */}
      <section className="section-prediction-map">
        <h2 className="section-title">FR3 · GRU 예측 & FR5 · 청소 실행</h2>

        <div className="grid grid-2">
          {/* Left: Prediction Results (Large) */}
          <div className="card">
            <div className="card-header">오염도 예측 결과</div>
            <p style={{ color: 'var(--text-secondary)', marginBottom: '1.5rem', fontSize: '0.9rem' }}>
              구역별 오염도 예측 (0.0 ~ 1.0)
            </p>

            <div className="prediction-chart-large">
              {predictions.length > 0 ? (
                predictions.map((pred, idx) => (
                  <div key={idx} className="prediction-bar-item-large">
                    <div className="prediction-zone-label-large">{pred.zone}</div>
                    <div className="prediction-bar-bg-large">
                      <div
                        className="prediction-bar-fill-large"
                        style={{
                          width: `${pred.score * 100}%`,
                          background: pred.score > 0.5 ? 'var(--danger)' : pred.score > 0.3 ? 'var(--warning)' : 'var(--success)'
                        }}
                      />
                    </div>
                    <div className="prediction-score-large">{pred.score.toFixed(2)}</div>
                  </div>
                ))
              ) : (
                <div style={{ textAlign: 'center', padding: '3rem', color: 'var(--text-secondary)' }}>
                  <div style={{ fontSize: '1.1rem', marginBottom: '0.5rem' }}>WebSocket 연결 대기 중...</div>
                  <div style={{ fontSize: '0.9rem' }}>GRU 예측 결과가 여기 표시됩니다</div>
                </div>
              )}
            </div>

            <div className="prediction-explanation">
              <div className="explanation-text">{explanation}</div>
            </div>
          </div>

          {/* Right: Floor Map (Large) */}
          <div className="card">
            <div className="card-header">평면도 - 오염도 히트맵</div>
            <p style={{ color: 'var(--text-secondary)', marginBottom: '1.5rem', fontSize: '0.9rem' }}>
              구역별 오염도 시각화
            </p>

            <div className="floor-map-large">
              <div className="floor-grid-large">
                {zonePollutions.map((zone, idx) => (
                  <div
                    key={idx}
                    className="floor-zone-large"
                    style={{
                      background: `linear-gradient(135deg, ${getPollutionColor(zone.pollution)}33 0%, ${getPollutionColor(zone.pollution)}11 100%)`,
                      borderColor: getPollutionColor(zone.pollution)
                    }}
                  >
                    <div className="zone-name-large">{zone.zone}</div>
                    <div className="zone-pollution-large">
                      {(zone.pollution * 100).toFixed(0)}%
                    </div>
                    <div className="zone-priority-large">
                      우선순위 #{zone.priority}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Map Legend */}
            <div className="map-legend" style={{ marginTop: '1.5rem' }}>
              <div className="legend-item">
                <div className="legend-color" style={{ background: 'var(--danger)' }}></div>
                <span>높음 (&gt;60%)</span>
              </div>
              <div className="legend-item">
                <div className="legend-color" style={{ background: 'var(--warning)' }}></div>
                <span>중간 (30-60%)</span>
              </div>
              <div className="legend-item">
                <div className="legend-color" style={{ background: 'var(--success)' }}></div>
                <span>낮음 (&lt;30%)</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* SECTION 3: Cleaning Timeline */}
      <section className="section-cleaning-timeline">
        <h2 className="section-title">청소 실행</h2>

        {/* Current Action */}
        <div className="card" style={{ marginBottom: '1.5rem' }}>
          <div className="current-action">
            <div className="action-content">
              <div className="action-label">현재 작업</div>
              <div className="action-text">{currentAction}</div>
            </div>
            <div className="action-progress">
              <div className="progress-circle">
                <svg width="80" height="80">
                  <circle
                    cx="40"
                    cy="40"
                    r="35"
                    fill="none"
                    stroke="var(--bg-secondary)"
                    strokeWidth="8"
                  />
                  <circle
                    cx="40"
                    cy="40"
                    r="35"
                    fill="none"
                    stroke="var(--accent)"
                    strokeWidth="8"
                    strokeDasharray={`${((estimatedTime.total - estimatedTime.remaining) / estimatedTime.total) * 220} 220`}
                    strokeLinecap="round"
                    transform="rotate(-90 40 40)"
                  />
                </svg>
                <div className="progress-text">
                  {Math.round(((estimatedTime.total - estimatedTime.remaining) / estimatedTime.total) * 100)}%
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Cleaning Timeline */}
        <div className="card">
          <div className="card-header">청소 경로 타임라인</div>

          <div className="cleaning-timeline">
            {cleaningTimeline.map((task, idx) => (
              <div key={idx} className={`timeline-task status-${task.status}`}>
                <div className="task-time">{task.time}</div>
                <div className="task-details">
                  <div className="task-zone">{task.zone}</div>
                  <div className="task-action">{task.action}</div>
                  <div className="task-duration">{task.duration} min</div>
                </div>
                <div className={`task-status-badge ${task.status}`}>
                  {task.status === 'completed' && '✓'}
                  {task.status === 'in_progress' && '⟳'}
                  {task.status === 'pending' && '○'}
                </div>
              </div>
            ))}
          </div>

          <div className="timeline-summary">
            <div className="summary-item">
              <span className="summary-label">총 시간</span>
              <span className="summary-value">{estimatedTime.total} 분</span>
            </div>
            <div className="summary-item">
              <span className="summary-label">남은 시간</span>
              <span className="summary-value" style={{ color: 'var(--warning)' }}>
                {estimatedTime.remaining} 분
              </span>
            </div>
            <div className="summary-item">
              <span className="summary-label">현재 작업</span>
              <span className="summary-value" style={{ color: 'var(--accent)' }}>
                {estimatedTime.currentTask} 분
              </span>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};

export default UnifiedDashboard;
