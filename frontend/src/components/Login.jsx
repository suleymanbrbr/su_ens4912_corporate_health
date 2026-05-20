import React, { useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'

function Login({ onLogin }) {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const navigate = useNavigate()

  const handleLogin = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError('')

    const formData = new URLSearchParams()
    formData.append('username', username)
    formData.append('password', password)

    try {
      const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: formData,
      })

      if (response.ok) {
        const data = await response.json()
        localStorage.setItem('token', data.access_token)
        
        const meRes = await fetch('/api/auth/me', {
          headers: { 'Authorization': `Bearer ${data.access_token}` }
        })
        const user = await meRes.json()
        onLogin(user)
        navigate('/')
      } else {
        const err = await response.json()
        setError(err.detail || 'Giriş Başarısız.')
      }
    } catch (err) {
      setError('Sistem hatası. Lütfen daha sonra deneyin.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="auth-container" style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '100vh' }}>
      <div className="premium-card glass" style={{ maxWidth: '400px', width: '90%' }}>
        <h2 className="text-gradient" style={{ textAlign: 'center', marginBottom: '1rem', fontSize: '2rem' }}>Tekrar Hoşgeldiniz</h2>
        <p style={{ color: 'var(--text-muted)', textAlign: 'center', marginBottom: '2rem' }}>SUT Kurumsal Sağlık Asistanına giriş yapın</p>
        
        {error && <div role="alert" style={{ color: '#ef4444', marginBottom: '1rem', textAlign: 'center', fontSize: '0.9rem' }}>{error}</div>}

        <form onSubmit={handleLogin} style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <div>
            <label htmlFor="login-username" style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 600 }}>Kullanıcı Adı</label>
            <input
              id="login-username"
              type="text"
              required
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="kullaniciadiniz"
              autoComplete="username"
              autoFocus
            />
          </div>
          <div>
            <label htmlFor="login-password" style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 600 }}>Şifre</label>
            <input
              id="login-password"
              type="password"
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••"
              autoComplete="current-password"
            />
          </div>
          <button type="submit" disabled={loading || !username.trim() || !password} className="btn-primary" style={{ marginTop: '1rem' }}>
            {loading ? 'Giriş yapılıyor...' : 'Giriş Yap'}
          </button>
        </form>
        
        <p style={{ marginTop: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>
          Hesabınız yok mu? <Link to="/signup" style={{ color: 'var(--accent)', fontWeight: 600, textDecoration: 'none' }}>Kayıt Olun</Link>
        </p>
      </div>
    </div>
  )
}

export default Login
