import React, { useState, useEffect } from 'react';
import './FR4Page.css';

interface ClientStatus {
  id: number;
  name: string;
  status: 'active' | 'training' | 'uploading' | 'idle';
  lastUpdate: string;
  dataPoints: number;
  accuracy: number;
}

interface ServerLog {
  timestamp: string;
  message: string;
  type: 'info' | 'success' | 'warning';
}

const FR4Page: React.FC = () => {
  const [clients, setClients] = useState<ClientStatus[]>([
    { id: 1, name: 'Client 1 (Main)', status: 'active', lastUpdate: '5s ago', dataPoints: 1247, accuracy: 0.87 },
    { id: 2, name: 'Client 2 (Demo)', status: 'training', lastUpdate: '12s ago', dataPoints: 892, accuracy: 0.83 },
    { id: 3, name: 'Client 3 (Demo)', status: 'idle', lastUpdate: '45s ago', dataPoints: 654, accuracy: 0.79 }
  ]);

  const [serverLogs, setServerLogs] = useState<ServerLog[]>([
    { timestamp: 'T-15s', message: 'Client 1 weight update received (124.8K params)', type: 'info' },
    { timestamp: 'T-12s', message: 'Client 2 weight update received (124.8K params)', type: 'info' },
    { timestamp: 'T-8s', message: 'Aggregating weights from 2 clients...', type: 'info' },
    { timestamp: 'T-5s', message: 'Weight validation completed - Loss: 0.234', type: 'success' },
    { timestamp: 'T-2s', message: 'Global model updated and distributed to all clients', type: 'success' }
  ]);

  const [serverStats, setServerStats] = useState({
    totalRounds: 47,
    activeClients: 2,
    globalAccuracy: 0.85,
    aggregationTime: 3.2,
    modelSize: 124.8
  });

  const [aggregationProgress, setAggregationProgress] = useState({
    weightsReceived: 2,
    weightsNeeded: 3,
    validationStatus: 'completed',
    redistributionStatus: 'in_progress'
  });

  // 시뮬레이션: 클라이언트 상태 업데이트
  useEffect(() => {
    const interval = setInterval(() => {
      setClients(prev => prev.map(client => {
        if (client.id === 1) {
          return {
            ...client,
            status: ['active', 'training', 'uploading'][Math.floor(Math.random() * 3)] as any,
            dataPoints: client.dataPoints + Math.floor(Math.random() * 10)
          };
        }
        return client;
      }));

      setServerStats(prev => ({
        ...prev,
        globalAccuracy: Math.min(0.95, prev.globalAccuracy + 0.001),
        aggregationTime: 2.5 + Math.random() * 1.5
      }));
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'var(--success)';
      case 'training': return 'var(--warning)';
      case 'uploading': return 'var(--accent)';
      default: return 'var(--text-secondary)';
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'active': return 'badge-success';
      case 'training': return 'badge-warning';
      case 'uploading': return 'badge-warning';
      default: return '';
    }
  };

  return (
    <div className="page">
      <h1 className="page-title">FR4 · Federated Learning</h1>
      <p className="page-subtitle">
        FedPer 기반 분산 협업 학습 - Base Layer 공유 및 집계
      </p>

      {/* FL Server 연결 상태 */}
      <div className="card" style={{ marginBottom: '1.5rem', background: 'rgba(255, 165, 2, 0.1)', borderLeft: '3px solid var(--warning)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <div style={{
            width: '12px',
            height: '12px',
            borderRadius: '50%',
            background: 'var(--warning)'
          }}></div>
          <div>
            <div style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', marginBottom: '0.25rem' }}>
              FL Server Status
            </div>
            <div style={{ fontSize: '1.1rem', fontWeight: '600', color: 'var(--text-primary)' }}>
              🟡 FL Server 구현 진행 중 - MQTT 연결 대기
            </div>
            <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '0.5rem' }}>
              FL 서버가 실행되면 MQTT 토픽 (locus/fl/*)을 통해 자동 연결됩니다.
            </div>
          </div>
        </div>
      </div>

      {/* Server Stats Overview */}
      <div className="grid grid-4" style={{ marginBottom: '1.5rem' }}>
        <div className="stat-card">
          <div className="stat-label">Training Rounds</div>
          <div className="stat-value-small">{serverStats.totalRounds}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Active Clients</div>
          <div className="stat-value-small" style={{ color: 'var(--success)' }}>
            {serverStats.activeClients}/{clients.length}
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Global Accuracy</div>
          <div className="stat-value-small" style={{ color: 'var(--accent)' }}>
            {(serverStats.globalAccuracy * 100).toFixed(1)}%
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Model Size</div>
          <div className="stat-value-small" style={{ color: 'var(--warning)' }}>
            {serverStats.modelSize}K
          </div>
        </div>
      </div>

      {/* FL Architecture Visualization */}
      <div className="card" style={{ marginBottom: '1.5rem' }}>
        <div className="card-header">Federated Learning Architecture</div>

        <div className="fl-architecture">
          {/* Clients */}
          <div className="fl-clients">
            {clients.map(client => (
              <div key={client.id} className="fl-client">
                <div className="client-header">
                  <div className="client-name">{client.name}</div>
                  <span className={`badge ${getStatusBadge(client.status)}`}>
                    {client.status}
                  </span>
                </div>
                <div className="client-stats">
                  <div className="client-stat">
                    <span className="stat-label-small">Data Points</span>
                    <span className="stat-value-tiny">{client.dataPoints}</span>
                  </div>
                  <div className="client-stat">
                    <span className="stat-label-small">Accuracy</span>
                    <span className="stat-value-tiny">{(client.accuracy * 100).toFixed(0)}%</span>
                  </div>
                </div>
                <div className="client-update">{client.lastUpdate}</div>
                <div className="upload-arrow">↑</div>
              </div>
            ))}
          </div>

          {/* Central Server */}
          <div className="fl-server">
            <div className="server-header">
              <div className="server-icon">🖥️</div>
              <div className="server-title">FL Central Server</div>
            </div>

            <div className="server-process">
              <div className="process-step">
                <div className="step-number">1</div>
                <div className="step-content">
                  <div className="step-title">Weight Collection</div>
                  <div className="step-detail">
                    {aggregationProgress.weightsReceived}/{aggregationProgress.weightsNeeded} clients
                  </div>
                </div>
              </div>

              <div className="process-step">
                <div className="step-number">2</div>
                <div className="step-content">
                  <div className="step-title">Aggregation</div>
                  <div className="step-detail">
                    FedAvg ({serverStats.aggregationTime.toFixed(1)}s)
                  </div>
                </div>
              </div>

              <div className="process-step">
                <div className="step-number">3</div>
                <div className="step-content">
                  <div className="step-title">Validation</div>
                  <div className="step-detail completed">
                    {aggregationProgress.validationStatus}
                  </div>
                </div>
              </div>

              <div className="process-step">
                <div className="step-number">4</div>
                <div className="step-content">
                  <div className="step-title">Distribution</div>
                  <div className="step-detail in-progress">
                    {aggregationProgress.redistributionStatus}
                  </div>
                </div>
              </div>
            </div>

            <div className="download-arrows">↓ ↓ ↓</div>
          </div>
        </div>
      </div>

      {/* Server Activity Log */}
      <div className="card">
        <div className="card-header">Server Activity Log</div>
        <p style={{ color: 'var(--text-secondary)', marginBottom: '1rem', fontSize: '0.95rem' }}>
          Weight 수신 → 집계 → 검증 → 재배포 과정
        </p>

        <div className="server-log">
          {serverLogs.map((log, idx) => (
            <div key={idx} className={`log-entry log-${log.type}`}>
              <div className="log-timestamp">{log.timestamp}</div>
              <div className="log-message">{log.message}</div>
            </div>
          ))}
        </div>

        <div className="log-summary">
          <div className="summary-icon">✓</div>
          <div className="summary-text">
            <strong>Round {serverStats.totalRounds} 완료:</strong> 2개 클라이언트로부터 가중치 수신 및 집계 완료.
            Global model 업데이트 후 모든 클라이언트에 재배포 중.
          </div>
        </div>
      </div>
    </div>
  );
};

export default FR4Page;
