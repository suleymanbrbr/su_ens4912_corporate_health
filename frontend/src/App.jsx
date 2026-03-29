import React, { useState, useEffect } from 'react'
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import Login from './components/Login'
import Signup from './components/Signup'
import ChatDashboard from './components/ChatDashboard'
import AdminPanel from './components/AdminPanel'
import Profile from './components/Profile'
import Settings from './components/Settings'
import PolicyBrowser from './components/PolicyBrowser'

function App() {
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // Theme initialization
    const savedTheme = localStorage.getItem('theme')
    if (savedTheme) {
      document.documentElement.setAttribute('data-theme', savedTheme)
    } else if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
      document.documentElement.setAttribute('data-theme', 'dark')
      localStorage.setItem('theme', 'dark')
    }

    const token = localStorage.getItem('token')
    if (token) {
      fetch('/api/auth/me', {
        headers: { 'Authorization': `Bearer ${token}` }
      })
      .then(res => res.ok ? res.json() : null)
      .then(data => {
        if (data) setUser(data)
        setLoading(false)
      })
      .catch(() => setLoading(false))
    } else {
      setLoading(false)
    }
  }, [])

  if (loading) return <div className="loading">Yükleniyor...</div>

  return (
    <Router>
      <Routes>
        <Route path="/login" element={user ? <Navigate to="/" /> : <Login onLogin={setUser} />} />
        <Route path="/signup" element={user ? <Navigate to="/" /> : <Signup />} />
        
        {/* Protected routes */}
        <Route path="/" element={user ? <ChatDashboard user={user} onLogout={() => setUser(null)} /> : <Navigate to="/login" />} />
        <Route path="/profile" element={user ? <Profile user={user} onLogout={() => setUser(null)} /> : <Navigate to="/login" />} />
        <Route path="/settings" element={user ? <Settings user={user} onLogout={() => setUser(null)} /> : <Navigate to="/login" />} />
        <Route path="/admin" element={user?.role === 'admin' ? <AdminPanel user={user} onLogout={() => setUser(null)} /> : <Navigate to="/" />} />
        <Route path="/policies" element={user ? <PolicyBrowser user={user} /> : <Navigate to="/" />} />
      </Routes>
    </Router>
  )
}

export default App
