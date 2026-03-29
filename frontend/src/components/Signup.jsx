import React, { useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'

function Signup() {
  const [username, setUsername] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [role, setRole] = useState('user')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const navigate = useNavigate()

  const handleSignup = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError('')

    try {
      const response = await fetch('/api/auth/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, email, password, role }),
      })

      if (response.ok) {
        navigate('/login')
      } else {
        const err = await response.json()
        setError(err.detail || 'Kayıt Başarısız.')
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
        <h2 className="text-gradient" style={{ textAlign: 'center', marginBottom: '1rem', fontSize: '2rem' }}>Yeni Hesap Oluştur</h2>
        <p style={{ color: 'var(--text-muted)', textAlign: 'center', marginBottom: '2rem' }}>SUT Mevzuat Asistanına katılın</p>
        
        {error && <div style={{ color: '#ef4444', marginBottom: '1rem', textAlign: 'center', fontSize: '0.9rem' }}>{error}</div>}
        
        <form onSubmit={handleSignup} style={{ display: 'flex', flexDirection: 'column', gap: '1.2rem' }}>
          <div>
            <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 600 }}>Kullanıcı Adı</label>
            <input 
              type="text" 
              required 
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="kullaniciadiniz"
            />
          </div>
          <div>
            <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 600 }}>E-posta</label>
            <input 
              type="email" 
              required 
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="ornek@posta.com"
            />
          </div>
          <div>
            <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 600 }}>Şifre</label>
            <input 
              type="password" 
              required 
              minLength={6}
            />
          </div>
          <div>
            <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 600 }}>Rolünüz</label>
            <select 
              value={role} 
              onChange={(e) => setRole(e.target.value)}
              style={{ width: '100%', padding: '0.8rem', borderRadius: '12px', border: '1px solid var(--border)', background: 'white', color: 'var(--text-main)', fontSize: '1rem', cursor: 'pointer' }}
            >
              <option value="user">Standart Kullanıcı (Doktor/Eczacı)</option>
              <option value="admin">Yönetici (Sistem Admini)</option>
            </select>
          </div>
          <button type="submit" disabled={loading} className="btn-primary" style={{ marginTop: '1rem' }}>
            {loading ? 'Kaydediliyor...' : 'Kayıt Ol'}
          </button>
        </form>
        
        <p style={{ marginTop: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>
          Zaten hesabınız var mı? <Link to="/login" style={{ color: 'var(--accent)', fontWeight: 600, textDecoration: 'none' }}>Giriş Yapın</Link>
        </p>
      </div>
    </div>
  )
}

export default Signup
