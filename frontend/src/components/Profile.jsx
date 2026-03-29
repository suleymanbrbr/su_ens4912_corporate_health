import React, { useState, useEffect } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { ArrowLeft, User, Lock, Clock, Bookmark, ShieldCheck } from 'lucide-react'

function Profile({ user, onLogout }) {
  const [history, setHistory] = useState([])
  const [saved, setSaved] = useState([])
  const [loading, setLoading] = useState(true)
  
  // Password change state
  const [oldPassword, setOldPassword] = useState('')
  const [newPassword, setNewPassword] = useState('')
  const [pwdMsg, setPwdMsg] = useState('')

  useEffect(() => {
    fetch('/api/history', {
      headers: { 'Authorization': `Bearer ${localStorage.getItem('token')}` }
    })
    .then(res => res.ok ? res.json() : null)
    .then(data => {
      if (data) {
        setHistory(data.history)
        setSaved(data.saved)
      }
      setLoading(false)
    })
    .catch(() => setLoading(false))
  }, [])

  const handleChangePassword = async (e) => {
    e.preventDefault()
    setPwdMsg('')
    try {
      const res = await fetch('/api/auth/password', {
        method: 'PUT',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify({ old_password: oldPassword, new_password: newPassword })
      })
      const data = await res.json()
      if (res.ok) {
        setPwdMsg('Şifre başarıyla güncellendi.')
        setOldPassword('')
        setNewPassword('')
      } else {
        setPwdMsg(data.detail || 'Hata oluştu.')
      }
    } catch (err) {
      setPwdMsg('Sunucu hatası.')
    }
  }

  return (
    <div style={{ display: 'flex', height: '100vh', background: 'var(--bg)' }}>
      <aside className="sidebar glass" style={{ width: '280px', borderRight: '1px solid var(--border)', padding: '2rem', display: 'flex', flexDirection: 'column' }}>
        <h2 className="text-gradient" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '2rem' }}>
          <User size={24} /> Profilim
        </h2>
        
        <nav style={{ flex: 1 }}>
          <Link to="/" style={{ textDecoration: 'none', color: 'var(--text-main)', display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.75rem', borderRadius: '8px' }} onMouseOver={e => e.currentTarget.style.background = '#f1f5f9'} onMouseOut={e => e.currentTarget.style.background = 'transparent'}>
            <ArrowLeft size={18} /> Sohbet Ekranı
          </Link>
        </nav>
      </aside>

      <main style={{ flex: 1, padding: '3rem', overflowY: 'auto' }}>
        <header style={{ marginBottom: '2rem' }}>
          <h1 style={{ fontSize: '2rem', fontWeight: 700 }}>Hesap Bilgileri</h1>
          <p style={{ color: 'var(--text-muted)' }}>Profilinizi ve sistem etkinliklerinizi yönetin.</p>
        </header>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem' }}>
          
          {/* Account Details & Security */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
            <div className="premium-card" style={{ padding: '2rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '2rem' }}>
                 <div style={{ width: '64px', height: '64px', borderRadius: '50%', background: 'var(--accent)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'white' }}>
                  <User size={32} />
                </div>
                <div>
                  <h2 style={{ fontSize: '1.5rem', fontWeight: 700 }}>{user.username}</h2>
                  <p style={{ color: 'var(--text-muted)' }}>{user.email}</p>
                </div>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.85rem', color: user.role === 'admin' ? '#166534' : 'var(--primary)', background: user.role === 'admin' ? '#dcfce7' : '#e0f2fe', padding: '0.5rem 1rem', borderRadius: '20px', display: 'inline-flex' }}>
                <ShieldCheck size={16} /> Hesap Rolü: {user.role.toUpperCase()}
              </div>
            </div>

            <div className="premium-card" style={{ padding: '2rem' }}>
              <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.5rem', fontSize: '1.2rem' }}>
                <Lock size={20} color="var(--primary)" /> Şifre Değiştir
              </h3>
              <form onSubmit={handleChangePassword} style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                <div>
                  <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '0.5rem' }}>Mevcut Şifre</label>
                  <input type="password" value={oldPassword} onChange={e => setOldPassword(e.target.value)} required style={{ width: '100%', padding: '0.75rem', borderRadius: '8px', border: '1px solid var(--border)' }} />
                </div>
                <div>
                  <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '0.5rem' }}>Yeni Şifre</label>
                  <input type="password" value={newPassword} onChange={e => setNewPassword(e.target.value)} required style={{ width: '100%', padding: '0.75rem', borderRadius: '8px', border: '1px solid var(--border)' }} />
                </div>
                <button type="submit" className="btn-primary" style={{ marginTop: '0.5rem' }}>Güncelle</button>
                {pwdMsg && <p style={{ fontSize: '0.85rem', color: pwdMsg.includes('başarı') ? '#10b981' : '#ef4444', marginTop: '0.5rem' }}>{pwdMsg}</p>}
              </form>
            </div>
          </div>

          {/* History */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
            <div className="premium-card" style={{ padding: '2rem', height: '100%' }}>
              <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.5rem', fontSize: '1.2rem' }}>
                <Clock size={20} color="#f59e0b" /> Son Aramalarınız
              </h3>
              {loading ? <p>Yükleniyor...</p> : history.length === 0 ? <p style={{ color: 'var(--text-muted)' }}>Henüz arama yapmadınız.</p> : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', maxHeight: '500px', overflowY: 'auto', paddingRight: '0.5rem' }}>
                  {history.map((h, i) => (
                    <div key={i} style={{ padding: '1rem', background: '#f8fafc', borderRadius: '8px', borderLeft: '3px solid var(--accent)' }}>
                      <p style={{ fontWeight: 600, fontSize: '0.95rem', marginBottom: '0.25rem' }}>{h.query}</p>
                      <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{new Date(h.created_at).toLocaleString('tr-TR')}</p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

        </div>
      </main>
    </div>
  )
}

export default Profile
