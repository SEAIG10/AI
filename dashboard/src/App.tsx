import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import FR1Page from './pages/FR1Page';
import FR2Page from './pages/FR2Page';
import FR3Page from './pages/FR3Page';
import FR4Page from './pages/FR4Page';
import FR5Page from './pages/FR5Page';

const App: React.FC = () => {
  return (
    <div className="app">
      <Sidebar />
      <main className="main-content">
        <Routes>
          <Route path="/" element={<Navigate to="/fr1" replace />} />
          <Route path="/fr1" element={<FR1Page />} />
          <Route path="/fr2" element={<FR2Page />} />
          <Route path="/fr3" element={<FR3Page />} />
          <Route path="/fr4" element={<FR4Page />} />
          <Route path="/fr5" element={<FR5Page />} />
        </Routes>
      </main>
    </div>
  );
};

export default App;
