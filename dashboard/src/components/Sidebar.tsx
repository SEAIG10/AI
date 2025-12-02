import React from 'react';
import './Sidebar.css';

const Sidebar: React.FC = () => {
  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <h1 className="sidebar-title">LOCUS</h1>
        <p className="sidebar-subtitle">AI Cleaning Dashboard</p>
      </div>

      <div className="sidebar-footer">
        <div className="status-indicator">
          <span className="status-dot"></span>
          <span>System Online</span>
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;
