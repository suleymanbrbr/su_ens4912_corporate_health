import React, { useState, useEffect } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { Users, LayoutDashboard, LogOut, ArrowLeft, Shield, Database, Activity, UserMinus, UserPlus } from 'lucide-react'

function AdminPanel({ user, onLogout }) {
  const [users, setUsers] = useState([])
  const [metrics, setMetrics] = useState({ users_count: 0, queries_count: 0, chunks_count: 0 })
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const navigate = useNavigate()

  useEffect(() => {
    fetchData()
  }, [])

  const fetchData = async () => {
    setLoading(true)
    try {
      const headers = { 'Authorization': `Bearer ${localStorage.getItem('token')}` }
      
      const [usersRes, metricsRes] = await Promise.all([
        fetch('/api/admin/users', { headers }),
        fetch('/api/admin/system', { headers })
      ])

      if (!usersRes.ok || !metricsRes.ok) throw new Error('Veri çekme hatası.')

      setUsers(await usersRes.json())
      setMetrics(await metricsRes.json())
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleRoleChange = async (targetId, currentRole) => {
    const newRole = currentRole === 'admin' ? 'user' : 'admin'
    try {
      const res = await fetch(`/api/admin/users/${targetId}/role`, {
        method: 'PUT',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}` 
        },
        body: JSON.stringify({ role: newRole })
      })
      if (res.ok) fetchData()
      else alert('Rol güncellenemedi.')
    } catch (err) {
      alert('Sunucu hatası.')
    }
  }

  const handleDeleteUser = async (targetId) => {
    if (!window.confirm('Bu kullanıcıyı tamamen silmek istediğinize emin misiniz?')) return
    try {
      const res = await fetch(`/api/admin/users/${targetId}`, {
        method: 'DELETE',
        headers: { 'Authorization': `Bearer ${localStorage.getItem('token')}` }
      })
      if (res.ok) fetchData()
      else alert(await res.json().then(d => d.detail || 'Silinemedi.'))
    } catch (err) {
      alert('Sunucu hatası.')
    }
  }

  const handleLogout = () => {
    localStorage.removeItem('token')
    onLogout()
    navigate('/login')
  }

  return (
    <div className="admin-layout" style={{ display: 'flex', height: '100vh' }}>
      <aside className="sidebar glass" style={{ width: '280px', borderRight: '1px solid var(--border)', padding: '2rem', display: 'flex', flexDirection: 'column' }}>
        <h2 className="text-gradient" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '2rem' }}>
          <Shield size={24} /> Admin Paneli
        </h2>
        
        <nav style={{ flex: 1 }}>
          <Link to="/" style={{ textDecoration: 'none', color: 'var(--text-main)', display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.75rem', borderRadius: '8px' }} onMouseOver={e => e.currentTarget.style.background = '#f1f5f9'} onMouseOut={e => e.currentTarget.style.background = 'transparent'}>
            <ArrowLeft size={18} /> Sohbet Ekranı
          </Link>
          <div style={{ background: '#f1f5f9', color: 'var(--accent)', display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.75rem', borderRadius: '8px', marginTop: '1rem', fontWeight: 600 }}>
            <LayoutDashboard size={18} /> Dashboard Özeti
          </div>
        </nav>

        <button onClick={handleLogout} className="btn-secondary" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', background: 'transparent', color: '#ef4444', padding: '1rem' }}>
          <LogOut size={18} /> Çıkış Yap
        </button>
      </aside>

      <main style={{ flex: 1, padding: '3rem', background: 'var(--bg)', overflowY: 'auto' }}>
        <header style={{ marginBottom: '2rem' }}>
          <h1 style={{ fontSize: '2rem', fontWeight: 700 }}>Sistem Yönetimi</h1>
          <p style={{ color: 'var(--text-muted)' }}>Mevcut sistem durumunu ve kullanıcıları yönetin.</p>
        </header>

        {loading ? (
          <div>Yükleniyor...</div>
        ) : error ? (
          <div style={{ color: '#ef4444' }}>{error}</div>
        ) : (
          <>
            {/* Metrics Dashboard */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1.5rem', marginBottom: '3rem' }}>
              <div className="premium-card" style={{ padding: '1.5rem', display: 'flex', alignItems: 'center', gap: '1rem' }}>
                <div style={{ width: '50px', height: '50px', background: '#dcfce7', color: '#16a34a', borderRadius: '12px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <Users size={24} />
                </div>
                <div>
                  <div style={{ fontSize: '1.8rem', fontWeight: 700 }}>{metrics.users_count}</div>
                  <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '1px' }}>Kayıtlı Kullanıcı</div>
                </div>
              </div>
              <div className="premium-card" style={{ padding: '1.5rem', display: 'flex', alignItems: 'center', gap: '1rem' }}>
                <div style={{ width: '50px', height: '50px', background: '#e0f2fe', color: '#0284c7', borderRadius: '12px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <Activity size={24} />
                </div>
                <div>
                  <div style={{ fontSize: '1.8rem', fontWeight: 700 }}>{metrics.queries_count}</div>
                  <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '1px' }}>Toplam Sorgu</div>
                </div>
              </div>
              <div className="premium-card" style={{ padding: '1.5rem', display: 'flex', alignItems: 'center', gap: '1rem' }}>
                <div style={{ width: '50px', height: '50px', background: '#fef3c7', color: '#d97706', borderRadius: '12px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <Database size={24} />
                </div>
                <div>
                  <div style={{ fontSize: '1.8rem', fontWeight: 700 }}>{metrics.chunks_count}</div>
                  <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '1px' }}>Veritabanı Belgesi</div>
                </div>
              </div>
            </div>

            {/* User Table */}
            <div style={{ marginBottom: '1rem' }}>
              <h2 style={{ fontSize: '1.2rem', fontWeight: 600 }}>Kullanıcı Yönetimi ({users.length})</h2>
            </div>
            <div className="premium-card" style={{ padding: 0, overflow: 'hidden' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
                <thead style={{ background: '#f8fafc', borderBottom: '1px solid var(--border)' }}>
                  <tr>
                    <th style={{ padding: '1rem 1.5rem' }}>E-posta</th>
                    <th style={{ padding: '1rem 1.5rem' }}>Rol</th>
                    <th style={{ padding: '1rem 1.5rem', textAlign: 'right' }}>Aksiyonlar</th>
                  </tr>
                </thead>
                <tbody>
                  {users.map(u => (
                    <tr key={u.id} style={{ borderBottom: '1px solid var(--border)' }}>
                      <td style={{ padding: '1rem 1.5rem' }}>
                        <div style={{ fontWeight: 600 }}>{u.username}</div>
                        <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{u.email}</div>
                      </td>
                      <td style={{ padding: '1rem 1.5rem' }}>
                        <span style={{ 
                          padding: '0.25rem 0.75rem', 
                          borderRadius: '20px', 
                          fontSize: '0.75rem', 
                          fontWeight: 700,
                          background: u.role === 'admin' ? '#dcfce7' : '#f1f5f9',
                          color: u.role === 'admin' ? '#166534' : 'var(--text-muted)',
                          textTransform: 'uppercase'
                        }}>
                          {u.role}
                        </span>
                      </td>
                      <td style={{ padding: '1rem 1.5rem', textAlign: 'right' }}>
                        {u.id !== user.id && (
                          <div style={{ display: 'flex', gap: '0.5rem', justifyContent: 'flex-end' }}>
                            <button 
                              onClick={() => handleRoleChange(u.id, u.role)}
                              style={{ background: '#f1f5f9', color: 'var(--text-main)', border: 'none', padding: '0.5rem', borderRadius: '8px', cursor: 'pointer' }}
                              title={u.role === 'admin' ? 'Kullanıcı Yap' : 'Admin Yap'}
                            >
                              {u.role === 'admin' ? <UserMinus size={16} /> : <UserPlus size={16} />}
                            </button>
                            <button 
                              onClick={() => handleDeleteUser(u.id)}
                              style={{ background: '#fef2f2', color: '#ef4444', border: 'none', padding: '0.5rem', borderRadius: '8px', cursor: 'pointer' }}
                              title="Sil"
                            >
                              <LogOut size={16} />
                            </button>
                          </div>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </main>
    </div>
  )
}

export default AdminPanel
