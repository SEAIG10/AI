import React, { useState, useEffect, useRef } from 'react';
import './FR5Page.css';

const WEBSOCKET_URL = 'ws://localhost:8080';

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

const FR5Page: React.FC = () => {
  const [wsConnected, setWsConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const [currentAction, setCurrentAction] = useState(
    'WebSocket 연결 대기 중... (cleaning_executor.py 실행 필요)'
  );

  const [policyReason, setPolicyReason] = useState(
    '거실에서 물건 떨어지는 소리가 감지되었고, 현재 조용한 시간대가 아니므로 즉시 청소를 시작합니다. 사용자 활동 패턴 분석 결과 현재 시간대가 청소에 가장 적합합니다.'
  );

  const [zonePollutions, setZonePollutions] = useState<ZonePollution[]>([
    { zone: 'Living Room', pollution: 0.65, priority: 1 },
    { zone: 'Kitchen', pollution: 0.40, priority: 2 },
    { zone: 'Bedroom', pollution: 0.25, priority: 3 },
    { zone: 'Hallway', pollution: 0.15, priority: 4 }
  ]);

  const [cleaningTimeline, setCleaningTimeline] = useState<CleaningTask[]>([
    { time: '14:30', zone: 'Living Room', action: 'Deep clean', duration: 15, status: 'in_progress' },
    { time: '14:45', zone: 'Living Room', action: 'Spot clean (detected spill)', duration: 5, status: 'pending' },
    { time: '14:50', zone: 'Kitchen', action: 'Standard clean', duration: 12, status: 'pending' },
    { time: '15:02', zone: 'Bedroom', action: 'Light clean', duration: 8, status: 'pending' }
  ]);

  const [estimatedTime, setEstimatedTime] = useState({
    total: 40,
    remaining: 35,
    currentTask: 5
  });

  const [policyMetrics, setPolicyMetrics] = useState({
    quietHoursActive: false,
    userPresent: true,
    batteryLevel: 87,
    lastCleaning: '3 hours ago'
  });

  // WebSocket 연결 및 청소 데이터 수신
  useEffect(() => {
    const connectWebSocket = () => {
      try {
        const ws = new WebSocket(WEBSOCKET_URL);
        wsRef.current = ws;

        ws.onopen = () => {
          console.log('[FR5 WS] Connected to WebSocket bridge');
          setWsConnected(true);
          setCurrentAction('청소 명령 대기 중...');
        };

        ws.onmessage = (event) => {
          try {
            const message = JSON.parse(event.data);

            if (message.type === 'welcome') return;

            // 청소 시작 메시지
            if (message.type === 'cleaning_started' || message.zone) {
              const zone = message.zone || 'Unknown';
              setCurrentAction(`${zone} 청소 중 (${new Date().toLocaleTimeString()} 시작)`);

              // 청소 타임라인 업데이트
              setCleaningTimeline(prev => prev.map(task =>
                task.zone === zone ? { ...task, status: 'in_progress' as const } : task
              ));
            }

            // 청소 완료 메시지
            if (message.type === 'cleaning_completed') {
              const zone = message.zone || 'Unknown';
              setCleaningTimeline(prev => prev.map(task =>
                task.zone === zone ? { ...task, status: 'completed' as const } : task
              ));

              // 오염도 업데이트 (청소 후 0으로)
              setZonePollutions(prev => prev.map(zp =>
                zp.zone === zone ? { ...zp, pollution: 0 } : zp
              ));
            }

            // 예측 결과로 오염도 업데이트
            if (message.prediction) {
              const predictions = Object.entries(message.prediction).map(([zone, pollution]) => ({
                zone,
                pollution: pollution as number,
                priority: 1
              })).sort((a, b) => b.pollution - a.pollution);

              // 우선순위 재계산
              predictions.forEach((p, idx) => {
                p.priority = idx + 1;
              });

              setZonePollutions(predictions);
            }

          } catch (err) {
            console.error('[FR5 WS] Message parsing error:', err);
          }
        };

        ws.onerror = (error) => {
          console.error('[FR5 WS] WebSocket error:', error);
          setWsConnected(false);
        };

        ws.onclose = () => {
          console.log('[FR5 WS] Disconnected');
          setWsConnected(false);
          setCurrentAction('WebSocket 연결 끊김 - 재연결 시도 중...');
          wsRef.current = null;
          setTimeout(connectWebSocket, 3000);
        };

      } catch (err) {
        console.error('[FR5 WS] Connection error:', err);
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
    <div className="page">
      <h1 className="page-title">FR5 · Policy Engine & Cleaning Execution</h1>
      <p className="page-subtitle">
        상황 인식 기반 정책 결정 및 청소 로봇 제어
      </p>

      {/* Current Action Summary */}
      <div className="card" style={{ marginBottom: '1.5rem' }}>
        <div className="current-action">
          <div className="action-icon">🤖</div>
          <div className="action-content">
            <div className="action-label">Current Action</div>
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

      {/* Policy Decision Reason */}
      <div className="card" style={{ marginBottom: '1.5rem' }}>
        <div className="card-header">Policy Decision Reasoning</div>
        <div className="policy-reason">
          <div className="reason-icon">💡</div>
          <div className="reason-text">{policyReason}</div>
        </div>

        <div className="grid grid-4" style={{ marginTop: '1.5rem' }}>
          <div className="policy-metric">
            <div className="metric-label">Quiet Hours</div>
            <div className="metric-value" style={{ color: policyMetrics.quietHoursActive ? 'var(--warning)' : 'var(--success)' }}>
              {policyMetrics.quietHoursActive ? 'Active' : 'Inactive'}
            </div>
          </div>
          <div className="policy-metric">
            <div className="metric-label">User Present</div>
            <div className="metric-value" style={{ color: policyMetrics.userPresent ? 'var(--success)' : 'var(--text-secondary)' }}>
              {policyMetrics.userPresent ? 'Yes' : 'No'}
            </div>
          </div>
          <div className="policy-metric">
            <div className="metric-label">Battery Level</div>
            <div className="metric-value" style={{ color: policyMetrics.batteryLevel > 30 ? 'var(--success)' : 'var(--danger)' }}>
              {policyMetrics.batteryLevel.toFixed(0)}%
            </div>
          </div>
          <div className="policy-metric">
            <div className="metric-label">Last Cleaning</div>
            <div className="metric-value" style={{ color: 'var(--text-secondary)', fontSize: '1rem' }}>
              {policyMetrics.lastCleaning}
            </div>
          </div>
        </div>
      </div>

      {/* Floor Map + Cleaning Timeline */}
      <div className="grid grid-2" style={{ marginBottom: '1.5rem' }}>
        {/* Floor Map with Pollution Overlay */}
        <div className="card">
          <div className="card-header">Floor Map - Pollution Heatmap</div>
          <p style={{ color: 'var(--text-secondary)', marginBottom: '1rem', fontSize: '0.9rem' }}>
            구역별 오염도 시각화 (모바일 앱의 RoomPlan 기반)
          </p>

          <div className="floor-map">
            <div className="floor-grid">
              {zonePollutions.map((zone, idx) => (
                <div
                  key={idx}
                  className="floor-zone"
                  style={{
                    background: `linear-gradient(135deg, ${getPollutionColor(zone.pollution)}33 0%, ${getPollutionColor(zone.pollution)}11 100%)`,
                    borderColor: getPollutionColor(zone.pollution)
                  }}
                >
                  <div className="zone-name">{zone.zone}</div>
                  <div className="zone-pollution">
                    {(zone.pollution * 100).toFixed(0)}%
                  </div>
                  <div className="zone-priority">
                    Priority: #{zone.priority}
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="map-legend">
            <div className="legend-item">
              <div className="legend-color" style={{ background: 'var(--danger)' }}></div>
              <span>High (&gt;60%)</span>
            </div>
            <div className="legend-item">
              <div className="legend-color" style={{ background: 'var(--warning)' }}></div>
              <span>Medium (30-60%)</span>
            </div>
            <div className="legend-item">
              <div className="legend-color" style={{ background: 'var(--success)' }}></div>
              <span>Low (&lt;30%)</span>
            </div>
          </div>
        </div>

        {/* Cleaning Timeline */}
        <div className="card">
          <div className="card-header">Cleaning Path Timeline</div>
          <p style={{ color: 'var(--text-secondary)', marginBottom: '1rem', fontSize: '0.9rem' }}>
            우선순위 기반 청소 경로 및 예상 시간
          </p>

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
              <span className="summary-label">Total Time</span>
              <span className="summary-value">{estimatedTime.total} min</span>
            </div>
            <div className="summary-item">
              <span className="summary-label">Remaining</span>
              <span className="summary-value" style={{ color: 'var(--warning)' }}>
                {estimatedTime.remaining} min
              </span>
            </div>
            <div className="summary-item">
              <span className="summary-label">Current Task</span>
              <span className="summary-value" style={{ color: 'var(--accent)' }}>
                {estimatedTime.currentTask} min
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default FR5Page;
