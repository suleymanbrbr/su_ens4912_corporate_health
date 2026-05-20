import React, { useState, useEffect, Suspense, lazy } from 'react'
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import Login from './components/Login'
import Signup from './components/Signup'
import ChatDashboard from './components/ChatDashboard'
import ErrorBoundary from './components/ErrorBoundary'

// Heavier / less-frequently-visited views split into their own chunks so the
// first paint on the auth + chat path stays lean.
const AdminPanel    = lazy(() => import('./components/AdminPanel'))
const Profile       = lazy(() => import('./components/Profile'))
const Settings      = lazy(() => import('./components/Settings'))
const PolicyBrowser = lazy(() => import('./components/PolicyBrowser'))

function RouteFallback() {
  return <div className="loading" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '100vh', color: 'var(--text-muted)' }}>Yükleniyor…</div>
}

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
    if (!token) {
      setLoading(false)
      return
    }
    let cancelled = false
    fetch('/api/auth/me', { headers: { Authorization: `Bearer ${token}` } })
      .then(res => {
        if (cancelled) return null
        if (res.status === 401) {
          // Token expired/invalid — clear and fall through to login.
          localStorage.removeItem('token')
          return null
        }
        return res.ok ? res.json() : null
      })
      .then(data => {
        if (cancelled) return
        if (data) setUser(data)
        setLoading(false)
      })
      .catch(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [])

  if (loading) return <div className="loading">Yükleniyor...</div>

  return (
    <ErrorBoundary>
      <Router>
        <Suspense fallback={<RouteFallback />}>
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
        </Suspense>
      </Router>
    </ErrorBoundary>
  )
}

export default App
