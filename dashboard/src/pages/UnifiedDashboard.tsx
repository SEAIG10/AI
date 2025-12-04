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
  const [zonePollutions, setZonePollutions] = useState<ZonePollution[]>([]);
  const [cleaningTimeline, setCleaningTimeline] = useState<CleaningTask[]>([]);
  const [estimatedTime, setEstimatedTime] = useState({
    total: 0,
    remaining: 0,
    currentTask: 0
  });

  // Buffer status state
  const [bufferStatus, setBufferStatus] = useState({ size: 0, capacity: 30 });

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

            // Buffer status update
            if (message.type === 'buffer_status') {
              setBufferStatus({
                size: message.buffer_size || 0,
                capacity: message.buffer_capacity || 30
              });
            }

            // Buffer reset - 센서 카운터도 리셋
            if (message.type === 'buffer_reset') {
              setBufferStatus({ size: 0, capacity: 30 });
              setStats(prev => ({
                visualMsgCount: 0,
                audioMsgCount: 0,
                poseMsgCount: 0,
                locationMsgCount: 0,
                syncedCount: 0,
                latencyMs: prev.latencyMs
              }));
              console.log('[Buffer Reset] Buffer and sensor counters reset');
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
        <h2 className="section-title">실시간 센서 데이터 수집</h2>

        <div className="grid" style={{ display: 'grid', gridTemplateColumns: '2fr 1fr 1fr 1fr', gap: '1rem' }}>
          {/* 1. YOLO Object Detection */}
          <div className="card">
            <div className="card-header">YOLO 객체 감지</div>
            <div className="video-container" style={{ marginBottom: '0.75rem', width: '100%', height: '240px', overflow: 'hidden', borderRadius: '12px', background: '#000' }}>
              <img
                src="http://localhost:5001/video_feed"
                alt="YOLO Live Stream"
                style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                onError={(e) => {
                  e.currentTarget.style.display = 'none';
                  e.currentTarget.nextElementSibling?.setAttribute('style', 'display: flex');
                }}
              />
              <div className="video-placeholder" style={{ display: 'none', minHeight: '240px', alignItems: 'center', justifyContent: 'center' }}>
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
          <div className="card-header">AI 오염도 예측 파이프라인</div>

          {/* Pipeline flow */}
          <div style={{ background: '#f8f9fa', padding: '2.5rem 2rem', borderRadius: '12px', marginBottom: '1rem' }}>
            <div style={{ fontFamily: 'monospace', fontSize: '0.9rem', lineHeight: '1.8' }}>
              <div>
                <div style={{ display: 'flex', alignItems: 'stretch', justifyContent: 'space-between', gap: '2rem', maxWidth: '100%' }}>
                  {/* 단계 1: 병렬 센서 입력 (세로 스택) */}
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', justifyContent: 'center' }}>
                    <div style={{
                      padding: '0.75rem 1rem',
                      background: 'white',
                      border: '2px solid var(--accent)',
                      borderRadius: '10px',
                      minWidth: '120px'
                    }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.3rem' }}>
                        <div style={{ fontSize: '0.7rem', fontWeight: '600', color: 'var(--accent)' }}>시각</div>
                        <div style={{ fontSize: '0.75rem', fontWeight: '700', color: 'var(--text-primary)', background: '#f8f9fa', padding: '0.15rem 0.4rem', borderRadius: '4px' }}>{stats.visualMsgCount}</div>
                      </div>
                      <div style={{ width: '100%', height: '3px', background: '#f0f0f0', borderRadius: '2px', overflow: 'hidden' }}>
                        <div style={{ width: `${Math.min((stats.visualMsgCount / Math.max(stats.visualMsgCount, stats.poseMsgCount, stats.audioMsgCount, stats.locationMsgCount, 1)) * 100, 100)}%`, height: '100%', background: 'var(--accent)', transition: 'width 0.3s ease' }}></div>
                      </div>
                    </div>
                    <div style={{
                      padding: '0.75rem 1rem',
                      background: 'white',
                      border: '2px solid var(--accent)',
                      borderRadius: '10px',
                      minWidth: '120px'
                    }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.3rem' }}>
                        <div style={{ fontSize: '0.7rem', fontWeight: '600', color: 'var(--accent)' }}>자세</div>
                        <div style={{ fontSize: '0.75rem', fontWeight: '700', color: 'var(--text-primary)', background: '#f8f9fa', padding: '0.15rem 0.4rem', borderRadius: '4px' }}>{stats.poseMsgCount}</div>
                      </div>
                      <div style={{ width: '100%', height: '3px', background: '#f0f0f0', borderRadius: '2px', overflow: 'hidden' }}>
                        <div style={{ width: `${Math.min((stats.poseMsgCount / Math.max(stats.visualMsgCount, stats.poseMsgCount, stats.audioMsgCount, stats.locationMsgCount, 1)) * 100, 100)}%`, height: '100%', background: 'var(--accent)', transition: 'width 0.3s ease' }}></div>
                      </div>
                    </div>
                    <div style={{
                      padding: '0.75rem 1rem',
                      background: 'white',
                      border: '2px solid var(--accent)',
                      borderRadius: '10px',
                      minWidth: '120px'
                    }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.3rem' }}>
                        <div style={{ fontSize: '0.7rem', fontWeight: '600', color: 'var(--accent)' }}>청각</div>
                        <div style={{ fontSize: '0.75rem', fontWeight: '700', color: 'var(--text-primary)', background: '#f8f9fa', padding: '0.15rem 0.4rem', borderRadius: '4px' }}>{stats.audioMsgCount}</div>
                      </div>
                      <div style={{ width: '100%', height: '3px', background: '#f0f0f0', borderRadius: '2px', overflow: 'hidden' }}>
                        <div style={{ width: `${Math.min((stats.audioMsgCount / Math.max(stats.visualMsgCount, stats.poseMsgCount, stats.audioMsgCount, stats.locationMsgCount, 1)) * 100, 100)}%`, height: '100%', background: 'var(--accent)', transition: 'width 0.3s ease' }}></div>
                      </div>
                    </div>
                    <div style={{
                      padding: '0.75rem 1rem',
                      background: 'white',
                      border: '2px solid var(--accent)',
                      borderRadius: '10px',
                      minWidth: '120px'
                    }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.3rem' }}>
                        <div style={{ fontSize: '0.7rem', fontWeight: '600', color: 'var(--accent)' }}>위치</div>
                        <div style={{ fontSize: '0.75rem', fontWeight: '700', color: 'var(--text-primary)', background: '#f8f9fa', padding: '0.15rem 0.4rem', borderRadius: '4px' }}>{stats.locationMsgCount}</div>
                      </div>
                      <div style={{ width: '100%', height: '3px', background: '#f0f0f0', borderRadius: '2px', overflow: 'hidden' }}>
                        <div style={{ width: `${Math.min((stats.locationMsgCount / Math.max(stats.visualMsgCount, stats.poseMsgCount, stats.audioMsgCount, stats.locationMsgCount, 1)) * 100, 100)}%`, height: '100%', background: 'var(--accent)', transition: 'width 0.3s ease' }}></div>
                      </div>
                    </div>
                  </div>

                  <div style={{ fontSize: '1.25rem', color: 'var(--accent)', fontWeight: '700', display: 'flex', alignItems: 'center' }}>→</div>

                  {/* 단계 2: AI (원형 차트) */}
                  <div style={{
                    padding: '1.25rem 1.5rem',
                    background: 'white',
                    border: bufferStatus.size === bufferStatus.capacity ? '2px solid var(--accent)' : '2px solid #e0e0e0',
                    borderRadius: '12px',
                    textAlign: 'center',
                    minWidth: '140px',
                    transition: 'border-color 0.3s ease',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: '0.5rem'
                  }}>
                    <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)' }}>단계 2</div>
                    <div style={{ position: 'relative', width: '60px', height: '60px' }}>
                      <svg width="60" height="60" style={{ transform: 'rotate(-90deg)' }}>
                        {/* Background circle */}
                        <circle
                          cx="30"
                          cy="30"
                          r="24"
                          fill="none"
                          stroke="#f0f0f0"
                          strokeWidth="8"
                        />
                        {/* Progress circle */}
                        <circle
                          cx="30"
                          cy="30"
                          r="24"
                          fill="none"
                          stroke="var(--accent)"
                          strokeWidth="8"
                          strokeDasharray={`${2 * Math.PI * 24}`}
                          strokeDashoffset={`${2 * Math.PI * 24 * (1 - bufferStatus.size / bufferStatus.capacity)}`}
                          strokeLinecap="round"
                          style={{ transition: 'stroke-dashoffset 0.3s ease' }}
                        />
                      </svg>
                      <div style={{
                        position: 'absolute',
                        top: '50%',
                        left: '50%',
                        transform: 'translate(-50%, -50%)',
                        fontSize: '0.75rem',
                        fontWeight: '700',
                        color: bufferStatus.size === bufferStatus.capacity ? 'var(--accent)' : 'var(--text-primary)'
                      }}>
                        {bufferStatus.size}
                      </div>
                    </div>
                    <div style={{ fontSize: '0.8rem', fontWeight: '600', color: bufferStatus.size === bufferStatus.capacity ? 'var(--accent)' : 'var(--text-primary)' }}>
                      AI
                    </div>
                    <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)' }}>
                      / {bufferStatus.capacity}
                    </div>
                  </div>

                  <div style={{ fontSize: '1.25rem', color: 'var(--accent)', display: 'flex', alignItems: 'center' }}>→</div>

                  {/* 단계 3: 오염도 예측 결과 */}
                  <div style={{
                    padding: '1.25rem 2rem',
                    background: 'white',
                    border: '2px solid var(--accent)',
                    borderRadius: '12px',
                    flex: 1,
                    maxWidth: '500px',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '0.6rem'
                  }}>
                    <div style={{ fontSize: '0.75rem', fontWeight: '600', color: 'var(--accent)', marginBottom: '0.25rem', textAlign: 'center' }}>
                      오염도 예측 결과
                    </div>
                    {predictions.length > 0 ? (
                      predictions.map((pred, idx) => (
                        <div key={idx} style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                          <div style={{ fontSize: '0.8rem', fontWeight: '500', color: 'var(--text-primary)', minWidth: '90px' }}>
                            {pred.zone}
                          </div>
                          <div style={{ flex: 1, height: '8px', background: '#f0f0f0', borderRadius: '4px', overflow: 'hidden' }}>
                            <div style={{
                              width: `${pred.score * 100}%`,
                              height: '100%',
                              background: 'linear-gradient(90deg, var(--accent) 0%, #C4003C 100%)',
                              transition: 'width 0.5s ease'
                            }}></div>
                          </div>
                          <div style={{ fontSize: '0.85rem', fontWeight: '700', color: 'var(--text-primary)', minWidth: '40px', textAlign: 'right' }}>
                            {pred.score.toFixed(2)}
                          </div>
                        </div>
                      ))
                    ) : (
                      <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', textAlign: 'center', padding: '0.5rem' }}>
                        대기 중...
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* SECTION 2: GRU Prediction & Floor Map */}
      <section className="section-prediction-map">
        <h2 className="section-title">오염도 예측 & 청소 실행</h2>

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
                        style={{ width: `${pred.score * 100}%` }}
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
          </div>

          {/* Right: Cleaning Execution Status */}
          <div className="card">
            <div className="card-header">청소 실행 상태</div>
            <p style={{ color: 'var(--text-secondary)', marginBottom: '1.5rem', fontSize: '0.9rem' }}>
              오염도 &gt; 0.5 구역 자동 청소
            </p>

            {/* Current Action */}
            <div className="current-action" style={{ marginBottom: '1.5rem' }}>
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
                      strokeDasharray={`${estimatedTime.total > 0 ? ((estimatedTime.total - estimatedTime.remaining) / estimatedTime.total) * 220 : 0} 220`}
                      strokeLinecap="round"
                      transform="rotate(-90 40 40)"
                    />
                  </svg>
                  <div className="progress-text">
                    {estimatedTime.total > 0 ? Math.round(((estimatedTime.total - estimatedTime.remaining) / estimatedTime.total) * 100) : 0}%
                  </div>
                </div>
              </div>
            </div>

            {/* Cleaning Timeline */}
            <div className="cleaning-timeline">
              {cleaningTimeline.length > 0 ? (
                cleaningTimeline.map((task, idx) => (
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
                ))
              ) : (
                <div style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-secondary)' }}>
                  <div style={{ fontSize: '1rem', marginBottom: '0.5rem' }}>청소 작업 대기 중</div>
                  <div style={{ fontSize: '0.85rem' }}>오염도 예측 후 청소 작업이 시작됩니다</div>
                </div>
              )}
            </div>

            {/* Summary Stats */}
            <div className="timeline-summary" style={{ marginTop: '1.5rem' }}>
              <div className="summary-item">
                <span className="summary-label">완료</span>
                <span className="summary-value" style={{ color: 'var(--accent)' }}>
                  {cleaningTimeline.filter(t => t.status === 'completed').length}건
                </span>
              </div>
              <div className="summary-item">
                <span className="summary-label">진행 중</span>
                <span className="summary-value" style={{ color: 'var(--text-primary)' }}>
                  {cleaningTimeline.filter(t => t.status === 'in_progress').length}건
                </span>
              </div>
              <div className="summary-item">
                <span className="summary-label">대기</span>
                <span className="summary-value" style={{ color: 'var(--text-secondary)' }}>
                  {cleaningTimeline.filter(t => t.status === 'pending').length}건
                </span>
              </div>
            </div>
          </div>

        </div>
      </section>

    </div>
  );
};

export default UnifiedDashboard;
