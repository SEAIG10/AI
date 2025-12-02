import React from 'react';
import { Routes, Route } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import UnifiedDashboard from './pages/UnifiedDashboard';

const App: React.FC = () => {
  return (
    <div className="app">
      <Sidebar />
      <main className="main-content">
        <Routes>
          <Route path="/" element={<UnifiedDashboard />} />
        </Routes>
      </main>
    </div>
  );
};

export default App;
