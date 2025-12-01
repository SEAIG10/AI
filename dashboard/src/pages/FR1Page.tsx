import React from 'react';

const FR1Page: React.FC = () => {
  return (
    <div className="page">
      <h1 className="page-title">FR1 · Home Structure & Location Intelligence</h1>
      <p className="page-subtitle">
        RoomPlan 기반 3D 구조 생성 및 실시간 위치 추적 (Mobile App에서 표시)
      </p>

      <div className="card">
        <div className="card-header">Mobile Tracker App</div>
        <div className="video-placeholder">
          <p>모바일 앱에서 RoomPlan 및 위치 추적 표시</p>
        </div>
      </div>

      <div className="grid grid-3" style={{ marginTop: '1.5rem' }}>
        <div className="card">
          <div className="stat-label">Tracked Zones</div>
          <div className="stat-value">4</div>
        </div>
        <div className="card">
          <div className="stat-label">Current Location</div>
          <div className="stat-value">Living Room</div>
        </div>
        <div className="card">
          <div className="stat-label">Mapping Accuracy</div>
          <div className="stat-value">98%</div>
        </div>
      </div>
    </div>
  );
};

export default FR1Page;
