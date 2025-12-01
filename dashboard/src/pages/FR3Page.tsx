import React, { useState, useEffect, useRef } from 'react';
import './FR3Page.css';

const WEBSOCKET_URL = 'ws://localhost:8080';

interface ContextPoint {
  timestamp: string;
  activity: string;
  zone: string;
  intensity: number;
}

interface PredictionResult {
  zone: string;
  score: number;
}

interface AttentionWeight {
  feature: string;
  weight: number;
}

const FR3Page: React.FC = () => {
  const [contextSequence, setContextSequence] = useState<ContextPoint[]>([]);
  const [wsConnected, setWsConnected] = useState(false);
  const [lastPrediction, setLastPrediction] = useState<any>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const [attentionWeights, setAttentionWeights] = useState<AttentionWeight[]>([
    { feature: 'Person movement pattern', weight: 0.92 },
    { feature: 'Object dropping sound', weight: 0.87 },
    { feature: 'Cooking activity', weight: 0.73 },
    { feature: 'Time of day', weight: 0.58 },
    { feature: 'Door events', weight: 0.45 }
  ]);

  const [predictions, setPredictions] = useState<PredictionResult[]>([]);

  const [explanation, setExplanation] = useState(
    'WebSocket 브릿지 연결 대기 중... (websocket_bridge.py를 실행해주세요)'
  );

  const [layerStats, setLayerStats] = useState({
    baseLayerParams: 124800,
    headLayerParams: 35200,
    lastMqttSync: 'Waiting...',
    flServerStatus: 'Disconnected'
  });

  // WebSocket 연결 및 실시간 데이터 수신
  useEffect(() => {
    const connectWebSocket = () => {
      try {
        const ws = new WebSocket(WEBSOCKET_URL);
        wsRef.current = ws;

        ws.onopen = () => {
          console.log('[WS] Connected to WebSocket bridge');
          setWsConnected(true);
          setExplanation('실시간 센서 데이터 수신 대기 중...');
        };

        ws.onmessage = (event) => {
          try {
            const message = JSON.parse(event.data);

            // 환영 메시지 무시
            if (message.type === 'welcome') {
              console.log('[WS] Welcome:', message.message);
              return;
            }

            // ZeroMQ 센서 데이터 처리
            if (message.type && message.timestamp && message.data) {
              // 컨텍스트 시퀀스에 추가
              const now = Date.now();
              const newPoint: ContextPoint = {
                timestamp: new Date(message.timestamp * 1000).toLocaleTimeString(),
                activity: `${message.type} sensor`,
                zone: 'Processing...',
                intensity: Array.isArray(message.data) ? message.data[0] : 0.5
              };

              setContextSequence(prev => {
                const updated = [...prev, newPoint];
                return updated.slice(-7); // 최근 7개만 유지
              });

              setLayerStats(prev => ({
                ...prev,
                lastMqttSync: 'Just now'
              }));
            }

            // GRU 예측 결과 처리 (예측 메시지가 오면)
            if (message.prediction) {
              setLastPrediction(message.prediction);

              const predResults: PredictionResult[] = Object.entries(message.prediction).map(([zone, score]) => ({
                zone,
                score: score as number
              })).sort((a, b) => b.score - a.score);

              setPredictions(predResults);

              const highest = predResults[0];
              if (highest && highest.score > 0.5) {
                setExplanation(`${highest.zone}에서 높은 오염도(${(highest.score * 100).toFixed(0)}%)가 감지되었습니다.`);
              } else {
                setExplanation('모든 구역의 오염도가 낮은 상태입니다.');
              }
            }

          } catch (err) {
            console.error('[WS] Message parsing error:', err);
          }
        };

        ws.onerror = (error) => {
          console.error('[WS] WebSocket error:', error);
          setWsConnected(false);
          setExplanation('WebSocket 연결 오류. 브릿지 서버를 확인해주세요.');
        };

        ws.onclose = () => {
          console.log('[WS] Disconnected');
          setWsConnected(false);
          setExplanation('WebSocket 연결이 끊어졌습니다. 재연결 시도 중...');
          wsRef.current = null;

          // 3초 후 재연결 시도
          setTimeout(connectWebSocket, 3000);
        };

      } catch (err) {
        console.error('[WS] Connection error:', err);
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

  return (
    <div className="page">
      <h1 className="page-title">FR3 · Sequential GRU Prediction</h1>
      <p className="page-subtitle">
        30×160 Context Sequence → Attention Mechanism → GRU → Pollution Prediction
      </p>

      {/* WebSocket 연결 상태 */}
      <div className="card" style={{ marginBottom: '1.5rem', background: wsConnected ? 'rgba(78, 204, 163, 0.1)' : 'rgba(255, 99, 72, 0.1)', borderLeft: `3px solid ${wsConnected ? 'var(--success)' : 'var(--danger)'}` }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <div style={{
            width: '12px',
            height: '12px',
            borderRadius: '50%',
            background: wsConnected ? 'var(--success)' : 'var(--danger)',
            animation: wsConnected ? 'pulse 2s ease-in-out infinite' : 'none'
          }}></div>
          <div>
            <div style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', marginBottom: '0.25rem' }}>
              WebSocket Bridge Status
            </div>
            <div style={{ fontSize: '1.1rem', fontWeight: '600', color: 'var(--text-primary)' }}>
              {wsConnected ? '🟢 Connected to ZeroMQ Bridge' : '🔴 Disconnected - Run websocket_bridge.py'}
            </div>
          </div>
        </div>
      </div>

      {/* 30×160 Context Sequence Timeline */}
      <div className="card">
        <div className="card-header">30×160 Context Sequence (Recent 30 minutes)</div>
        <p style={{ color: 'var(--text-secondary)', marginBottom: '1rem', fontSize: '0.95rem' }}>
          최근 30분 동안 집안 상황의 주요 변화 (30 timesteps × 160 features)
        </p>

        <div className="context-timeline">
          {contextSequence.map((point, idx) => (
            <div key={idx} className="context-point">
              <div className="context-timestamp">{point.timestamp}</div>
              <div className="context-details">
                <div className="context-activity">{point.activity}</div>
                <div className="context-zone">{point.zone}</div>
              </div>
              <div className="context-intensity-bar">
                <div
                  className="context-intensity-fill"
                  style={{ width: `${point.intensity * 100}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Attention Weights + Predictions */}
      <div className="grid grid-2" style={{ marginTop: '1.5rem' }}>
        {/* Attention Visualization */}
        <div className="card">
          <div className="card-header">Attention Weights</div>
          <p style={{ color: 'var(--text-secondary)', marginBottom: '1rem', fontSize: '0.9rem' }}>
            GRU가 예측에 가장 큰 영향을 준 특징들
          </p>

          <div className="attention-list">
            {attentionWeights.map((item, idx) => (
              <div key={idx} className="attention-item">
                <span className="attention-feature">{item.feature}</span>
                <div className="attention-bar-container">
                  <div
                    className="attention-bar"
                    style={{ width: `${item.weight * 100}%` }}
                  />
                </div>
                <span className="attention-value">{(item.weight * 100).toFixed(0)}%</span>
              </div>
            ))}
          </div>

          <div className="attention-summary">
            <strong>핵심 인사이트:</strong> 사람 이동 패턴이 예측에 가장 큰 영향을 주었습니다.
          </div>
        </div>

        {/* Prediction Results */}
        <div className="card">
          <div className="card-header">Pollution Prediction Results</div>
          <p style={{ color: 'var(--text-secondary)', marginBottom: '1rem', fontSize: '0.9rem' }}>
            구역별 오염도 예측 (0.0 ~ 1.0)
          </p>

          <div className="prediction-chart">
            {predictions.map((pred, idx) => (
              <div key={idx} className="prediction-bar-item">
                <div className="prediction-zone-label">{pred.zone}</div>
                <div className="prediction-bar-bg">
                  <div
                    className="prediction-bar-fill"
                    style={{
                      width: `${pred.score * 100}%`,
                      background: pred.score > 0.5 ? 'var(--danger)' : pred.score > 0.3 ? 'var(--warning)' : 'var(--success)'
                    }}
                  />
                </div>
                <div className="prediction-score">{pred.score.toFixed(2)}</div>
              </div>
            ))}
          </div>

          <div className="prediction-explanation">
            <div className="explanation-icon">💡</div>
            <div className="explanation-text">{explanation}</div>
          </div>
        </div>
      </div>

      {/* Base/Head Layer + MQTT to FL Server */}
      <div className="card" style={{ marginTop: '1.5rem' }}>
        <div className="card-header">FedPer Architecture: Base + Head Layers</div>

        <div className="fedper-architecture">
          <div className="layer-section base-layer">
            <div className="layer-title">Base Layer (Shared)</div>
            <div className="layer-params">{layerStats.baseLayerParams.toLocaleString()} params</div>
            <div className="layer-description">
              Visual, Audio, Pose 특징 추출<br />
              연합학습으로 공유됨
            </div>
            <div className="mqtt-status">
              <span className="status-dot-green"></span>
              <span>MQTT → FL Server</span>
            </div>
          </div>

          <div className="layer-arrow">→</div>

          <div className="layer-section head-layer">
            <div className="layer-title">Head Layer (Personalized)</div>
            <div className="layer-params">{layerStats.headLayerParams.toLocaleString()} params</div>
            <div className="layer-description">
              개인화된 오염도 예측<br />
              로컬에서만 학습됨
            </div>
          </div>
        </div>

        <div className="grid grid-3" style={{ marginTop: '1.5rem' }}>
          <div className="stat-card">
            <div className="stat-label">Base Layer Size</div>
            <div className="stat-value-small">{(layerStats.baseLayerParams / 1000).toFixed(1)}K</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Last MQTT Sync</div>
            <div className="stat-value-small" style={{ fontSize: '1.25rem', color: 'var(--success)' }}>
              {layerStats.lastMqttSync}
            </div>
          </div>
          <div className="stat-card">
            <div className="stat-label">FL Server Status</div>
            <div className="stat-value-small" style={{ fontSize: '1.25rem', color: 'var(--success)' }}>
              {layerStats.flServerStatus}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default FR3Page;
