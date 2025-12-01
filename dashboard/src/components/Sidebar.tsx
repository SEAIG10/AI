import React from 'react';
import { NavLink } from 'react-router-dom';
import './Sidebar.css';

const Sidebar: React.FC = () => {
  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <h1 className="sidebar-title">LOCUS</h1>
        <p className="sidebar-subtitle">AI Cleaning Dashboard</p>
      </div>

      <nav className="sidebar-nav">
        <NavLink to="/fr1" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
          <span className="nav-icon">🏠</span>
          <div>
            <div className="nav-title">FR1</div>
            <div className="nav-desc">Home Structure</div>
          </div>
        </NavLink>

        <NavLink to="/fr2" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
          <span className="nav-icon">👁️</span>
          <div>
            <div className="nav-title">FR2</div>
            <div className="nav-desc">Visual & Audio</div>
          </div>
        </NavLink>

        <NavLink to="/fr3" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
          <span className="nav-icon">🧠</span>
          <div>
            <div className="nav-title">FR3</div>
            <div className="nav-desc">GRU Prediction</div>
          </div>
        </NavLink>

        <NavLink to="/fr4" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
          <span className="nav-icon">🤝</span>
          <div>
            <div className="nav-title">FR4</div>
            <div className="nav-desc">Federated Learning</div>
          </div>
        </NavLink>

        <NavLink to="/fr5" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
          <span className="nav-icon">🧹</span>
          <div>
            <div className="nav-title">FR5</div>
            <div className="nav-desc">Policy & Cleaning</div>
          </div>
        </NavLink>
      </nav>

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
