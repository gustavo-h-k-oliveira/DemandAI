import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Navigation } from './components/Navigation';
import Dashboard from './pages/Dashboard'; // Changed: removed destructuring
import { Login } from './pages/Login';

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-gray-50">
        <Navigation />
        <main>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/login" element={<Login />} />
            <Route path="/predict" element={<div>Predict Page - Coming Soon</div>} />
            <Route path="/history" element={<div>History Page - Coming Soon</div>} />
            <Route path="/register" element={<div>Register Page - Coming Soon</div>} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}