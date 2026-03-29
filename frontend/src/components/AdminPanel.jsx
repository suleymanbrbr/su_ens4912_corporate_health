import React, { useState, useEffect, useRef } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import {
  Users, LayoutDashboard, LogOut, ArrowLeft, Shield, Database,
  Activity, UserMinus, UserPlus, Megaphone, Clock, TrendingUp,
  BarChart2, BookOpen, Trash2, CheckCircle
} from 'lucide-react'

const API_HEADERS = () => ({ 'Authorization': `Bearer ${localStorage.getItem('token')}` })

function TabButton({ active, onClick, icon, label }) {
  return (
    <button
      onClick={onClick}
      style={{
        display: 'flex', alignItems: 'center', gap: '0.5rem',
        padding: '0.75rem 1.25rem', borderRadius: '8px', border: 'none',
        background: active ? 'var(--primary)' : 'transparent',
        color: active ? 'white' : 'var(--text-muted)',
        fontWeight: active ? 700 : 500, cursor: 'pointer', fontSize: '0.9rem',
        transition: 'all 0.2s'
      }}
    >
      {icon} {label}
    </button>
  )
}

function MetricCard({ icon, iconBg, iconColor, value, label }) {
  return (
    <div className="premium-card" style={{ padding: '1.5rem', display: 'flex', alignItems: 'center', gap: '1rem' }}>
      <div style={{ width: '50px', height: '50px', background: iconBg, color: iconColor, borderRadius: '12px', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
        {icon}
      </div>
      <div>
        <div style={{ fontSize: '1.8rem', fontWeight: 700 }}>{value}</div>
        <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '1px' }}>{label}</div>
      </div>
    </div>
  )
}

function AdminPanel({ user, onLogout }) {
  const [tab, setTab] = useState('overview')
  const [users, setUsers] = useState([])
  const [metrics, setMetrics] = useState({ users_count: 0, queries_count: 0, chunks_count: 0, pending_count: 0, active_announcement: null })
  const [activity, setActivity] = useState([])
  const [analytics, setAnalytics] = useState({ top_keywords: [], daily_volume: [], engagement_rate: 0, active_users: 0, total_users: 0 })
  const [announcement, setAnnouncement] = useState('')
  const [loading, setLoading] = useState(true)
  const [annMsg, setAnnMsg] = useState('')
  const activityTimer = useRef(null)
  const navigate = useNavigate()

  useEffect(() => {
    fetchOverview()
  }, [])

  useEffect(() => {
    if (tab === 'activity') {
      fetchActivity()
      activityTimer.current = setInterval(fetchActivity, 30000)
    } else {
      clearInterval(activityTimer.current)
    }
    if (tab === 'analytics') fetchAnalytics()
    return () => clearInterval(activityTimer.current)
  }, [tab])

  const fetchOverview = async () => {
    setLoading(true)
    try {
      const [usersRes, metricsRes] = await Promise.all([
        fetch('/api/admin/users', { headers: API_HEADERS() }),
        fetch('/api/admin/system', { headers: API_HEADERS() })
      ])
      setUsers(await usersRes.json())
      setMetrics(await metricsRes.json())
    } catch (e) { console.error(e) }
    finally { setLoading(false) }
  }

  const handleApproveUser = async (targetId) => {
    try {
      const res = await fetch(`/api/admin/users/${targetId}/approve`, {
        method: 'PUT',
        headers: API_HEADERS()
      })
      if (res.ok) fetchOverview()
      else alert('Onaylanamadı.')
    } catch (err) { alert('Sunucu hatası.') }
  }

  const fetchActivity = async () => {
    try {
      const res = await fetch('/api/admin/activity', { headers: API_HEADERS() })
      if (res.ok) setActivity(await res.json())
    } catch (e) { console.error(e) }
  }

  const fetchAnalytics = async () => {
    try {
      const res = await fetch('/api/admin/analytics', { headers: API_HEADERS() })
      if (res.ok) setAnalytics(await res.json())
    } catch (e) { console.error(e) }
  }

  const handleRoleChange = async (targetId, currentRole) => {
    const newRole = currentRole === 'admin' ? 'user' : 'admin'
    await fetch(`/api/admin/users/${targetId}/role`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json', ...API_HEADERS() },
      body: JSON.stringify({ role: newRole })
    })
    fetchOverview()
  }

  const handleDeleteUser = async (targetId) => {
    if (!window.confirm('Bu kullanıcıyı tamamen silmek istediğinize emin misiniz?')) return
    await fetch(`/api/admin/users/${targetId}`, { method: 'DELETE', headers: API_HEADERS() })
    fetchOverview()
  }

  const handlePostAnnouncement = async () => {
    if (!announcement.trim()) return
    setAnnMsg('')
    const res = await fetch('/api/admin/announcements', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...API_HEADERS() },
      body: JSON.stringify({ message: announcement })
    })
    if (res.ok) {
      setAnnMsg('Duyuru başarıyla yayınlandı.')
      setAnnouncement('')
      fetchOverview()
    }
  }

  const handleRemoveAnnouncement = async (id) => {
    await fetch(`/api/admin/announcements/${id}`, { method: 'DELETE', headers: API_HEADERS() })
    setAnnMsg('Duyuru kaldırıldı.')
    fetchOverview()
  }

  const timeAgo = (ts) => {
    const diff = (Date.now() - new Date(ts + 'Z').getTime()) / 1000
    if (diff < 60) return `${Math.floor(diff)}sn önce`
    if (diff < 3600) return `${Math.floor(diff / 60)}dk önce`
    if (diff < 86400) return `${Math.floor(diff / 3600)}sa önce`
    return `${Math.floor(diff / 86400)}g önce`
  }

  const maxKeyword = analytics.top_keywords.length > 0 ? analytics.top_keywords[0].count : 1

  const handleLogout = () => {
    localStorage.removeItem('token')
    onLogout()
    navigate('/login')
  }

  return (
    <div style={{ display: 'flex', height: '100vh' }}>
      {/* Sidebar */}
      <aside className="sidebar glass" style={{ width: '260px', borderRight: '1px solid var(--border)', padding: '2rem', display: 'flex', flexDirection: 'column' }}>
        <h2 className="text-gradient" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '2rem', fontSize: '1.2rem' }}>
          <Shield size={22} /> Admin Paneli
        </h2>
        <nav style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
          <Link to="/" style={{ textDecoration: 'none', color: 'var(--text-main)', display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.75rem', borderRadius: '8px', fontSize: '0.9rem' }} onMouseOver={e => e.currentTarget.style.background = '#f1f5f9'} onMouseOut={e => e.currentTarget.style.background = 'transparent'}>
            <ArrowLeft size={16} /> Sohbet Ekranı
          </Link>
          <Link to="/policies" style={{ textDecoration: 'none', color: 'var(--text-main)', display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.75rem', borderRadius: '8px', fontSize: '0.9rem' }} onMouseOver={e => e.currentTarget.style.background = '#f1f5f9'} onMouseOut={e => e.currentTarget.style.background = 'transparent'}>
            <BookOpen size={16} /> SUT Mevzuat Tarayıcısı
          </Link>
        </nav>
        <button onClick={handleLogout} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', background: 'transparent', color: '#ef4444', border: 'none', cursor: 'pointer', padding: '0.75rem', borderRadius: '8px', fontWeight: 600 }}>
          <LogOut size={16} /> Çıkış Yap
        </button>
      </aside>

      {/* Main */}
      <main style={{ flex: 1, display: 'flex', flexDirection: 'column', background: 'var(--bg)', overflow: 'hidden' }}>
        {/* Tab Bar */}
        <div style={{ padding: '1.5rem 2rem 0', borderBottom: '1px solid var(--border)', background: 'white', display: 'flex', gap: '0.5rem' }}>
          <TabButton active={tab === 'overview'} onClick={() => setTab('overview')} icon={<LayoutDashboard size={16} />} label="Genel Bakış" />
          <TabButton active={tab === 'activity'} onClick={() => setTab('activity')} icon={<Activity size={16} />} label="Canlı Aktivite" />
          <TabButton active={tab === 'analytics'} onClick={() => setTab('analytics')} icon={<TrendingUp size={16} />} label="Analitik" />
        </div>

        <div style={{ flex: 1, overflowY: 'auto', padding: '2rem' }}>
          {loading && tab === 'overview' ? (
            <div style={{ color: 'var(--text-muted)' }}>Yükleniyor...</div>
          ) : (
            <>
              {/* OVERVIEW TAB */}
              {tab === 'overview' && (
                <>
                  <h1 style={{ fontSize: '1.5rem', fontWeight: 700, marginBottom: '1.5rem' }}>Sistem Genel Bakış</h1>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1.5rem', marginBottom: '2rem' }}>
                    <MetricCard icon={<Users size={24} />} iconBg="#dcfce7" iconColor="#16a34a" value={metrics.users_count} label="Kayıtlı Kullanıcı" />
                    <MetricCard icon={<Activity size={24} />} iconBg="#e0f2fe" iconColor="#0284c7" value={metrics.queries_count} label="Toplam Sorgu" />
                    <MetricCard icon={<Database size={24} />} iconBg="#fef3c7" iconColor="#d97706" value={metrics.chunks_count} label="Veritabanı Belgesi" />
                    <MetricCard icon={<Shield size={24} />} iconBg="#fef2f2" iconColor="#ef4444" value={metrics.pending_count} label="Onay Bekleyen" />
                  </div>

                  {/* Pending Approvals Section */}
                  {users.filter(u => u.is_approved === 0).length > 0 && (
                    <div className="premium-card" style={{ padding: '1.5rem', marginBottom: '2rem', borderLeft: '4px solid #ef4444' }}>
                      <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem', fontSize: '1.1rem', fontWeight: 600, color: '#ef4444' }}>
                        <Shield size={20} /> Onay Bekleyen Kayıtlar
                      </h3>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                        {users.filter(u => u.is_approved === 0).map(u => (
                          <div key={u.id} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '0.75rem', background: '#fff5f5', borderRadius: '8px', border: '1px solid #fee2e2' }}>
                            <div>
                              <div style={{ fontWeight: 600 }}>{u.username}</div>
                              <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{u.email}</div>
                            </div>
                            <div style={{ display: 'flex', gap: '0.5rem' }}>
                              <button onClick={() => handleApproveUser(u.id)} className="btn-primary" style={{ padding: '0.4rem 1rem', fontSize: '0.85rem', background: '#16a34a' }}>Onayla</button>
                              <button onClick={() => handleDeleteUser(u.id)} style={{ padding: '0.4rem', borderRadius: '6px', border: '1px solid #ef4444', color: '#ef4444', background: 'transparent', cursor: 'pointer' }}><Trash2 size={16} /></button>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Announcement Manager */}
                  <div className="premium-card" style={{ padding: '1.5rem', marginBottom: '2rem' }}>
                    <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem', fontSize: '1rem', fontWeight: 600 }}>
                      <Megaphone size={18} color="var(--primary)" /> Sistem Duyurusu
                    </h3>
                    {metrics.active_announcement && (
                      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', padding: '0.75rem 1rem', background: '#fef3c7', borderRadius: '8px', marginBottom: '1rem', borderLeft: '3px solid #f59e0b' }}>
                        <CheckCircle size={16} color="#d97706" />
                        <span style={{ flex: 1, fontSize: '0.9rem' }}>{metrics.active_announcement.message}</span>
                        <button onClick={() => handleRemoveAnnouncement(metrics.active_announcement.id)} style={{ background: 'transparent', border: 'none', color: '#ef4444', cursor: 'pointer' }}>
                          <Trash2 size={16} />
                        </button>
                      </div>
                    )}
                    <div style={{ display: 'flex', gap: '0.75rem' }}>
                      <input
                        value={announcement}
                        onChange={e => setAnnouncement(e.target.value)}
                        placeholder="Yeni duyuru metni..."
                        style={{ flex: 1, padding: '0.75rem', borderRadius: '8px', border: '1px solid var(--border)', fontSize: '0.9rem' }}
                        onKeyDown={e => e.key === 'Enter' && handlePostAnnouncement()}
                      />
                      <button className="btn-primary" onClick={handlePostAnnouncement} style={{ padding: '0.75rem 1.25rem', whiteSpace: 'nowrap' }}>Yayınla</button>
                    </div>
                    {annMsg && <p style={{ fontSize: '0.8rem', color: '#10b981', marginTop: '0.5rem' }}>{annMsg}</p>}
                  </div>

                  {/* User Table */}
                  <h2 style={{ fontSize: '1rem', fontWeight: 600, marginBottom: '0.75rem' }}>Kullanıcı Yönetimi ({users.length})</h2>
                  <div className="premium-card" style={{ padding: 0, overflow: 'hidden' }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
                      <thead style={{ background: '#f8fafc', borderBottom: '1px solid var(--border)' }}>
                        <tr>
                          <th style={{ padding: '1rem 1.5rem' }}>Kullanıcı</th>
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
                              <span style={{ padding: '0.25rem 0.75rem', borderRadius: '20px', fontSize: '0.75rem', fontWeight: 700, background: u.role === 'admin' ? '#dcfce7' : '#f1f5f9', color: u.role === 'admin' ? '#166534' : 'var(--text-muted)', textTransform: 'uppercase' }}>
                                {u.role}
                              </span>
                              {u.is_approved === 0 && (
                                <span style={{ marginLeft: '0.5rem', padding: '0.2rem 0.6rem', background: '#fee2e2', color: '#ef4444', borderRadius: '12px', fontSize: '0.65rem', fontWeight: 700, textTransform: 'uppercase' }}>Onay Bekliyor</span>
                              )}
                            </td>
                            <td style={{ padding: '1rem 1.5rem', textAlign: 'right' }}>
                              {u.id !== user.id && (
                                <div style={{ display: 'flex', gap: '0.5rem', justifyContent: 'flex-end' }}>
                                  {u.is_approved === 0 && (
                                    <button onClick={() => handleApproveUser(u.id)} style={{ background: '#dcfce7', color: '#166534', border: 'none', padding: '0.5rem', borderRadius: '8px', cursor: 'pointer' }} title="Onayla">
                                      <CheckCircle size={16} />
                                    </button>
                                  )}
                                  <button onClick={() => handleRoleChange(u.id, u.role)} style={{ background: '#f1f5f9', border: 'none', padding: '0.5rem', borderRadius: '8px', cursor: 'pointer' }} title={u.role === 'admin' ? 'Kullanıcı Yap' : 'Admin Yap'}>
                                    {u.role === 'admin' ? <UserMinus size={16} /> : <UserPlus size={16} />}
                                  </button>
                                  <button onClick={() => handleDeleteUser(u.id)} style={{ background: '#fef2f2', color: '#ef4444', border: 'none', padding: '0.5rem', borderRadius: '8px', cursor: 'pointer' }} title="Sil">
                                    <Trash2 size={16} />
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

              {/* ACTIVITY TAB */}
              {tab === 'activity' && (
                <>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
                    <h1 style={{ fontSize: '1.5rem', fontWeight: 700 }}>Canlı Kullanıcı Aktivitesi</h1>
                    <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', background: '#f1f5f9', padding: '0.25rem 0.75rem', borderRadius: '20px' }}>
                      30 saniyede bir güncellenir
                    </span>
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                    {activity.length === 0 ? (
                      <p style={{ color: 'var(--text-muted)' }}>Henüz sorgu yapılmamış.</p>
                    ) : activity.map((a, i) => (
                      <div key={i} className="premium-card" style={{ padding: '1rem 1.5rem', display: 'flex', alignItems: 'center', gap: '1rem' }}>
                        <div style={{ width: '36px', height: '36px', borderRadius: '50%', background: a.role === 'admin' ? '#dcfce7' : '#e0f2fe', color: a.role === 'admin' ? '#16a34a' : '#0284c7', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 700, fontSize: '0.9rem', flexShrink: 0 }}>
                          {a.username[0].toUpperCase()}
                        </div>
                        <div style={{ flex: 1, minWidth: 0 }}>
                          <span style={{ fontWeight: 600 }}>{a.username}</span>
                          <span style={{ marginLeft: '0.5rem', fontSize: '0.75rem', padding: '0.1rem 0.5rem', borderRadius: '10px', background: a.role === 'admin' ? '#dcfce7' : '#f1f5f9', color: a.role === 'admin' ? '#16a34a' : 'var(--text-muted)' }}>{a.role}</span>
                          <p style={{ margin: '0.25rem 0 0', fontSize: '0.9rem', color: 'var(--text-muted)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                            {a.query}
                          </p>
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.25rem', fontSize: '0.75rem', color: 'var(--text-muted)', flexShrink: 0 }}>
                          <Clock size={12} /> {timeAgo(a.created_at)}
                        </div>
                      </div>
                    ))}
                  </div>
                </>
              )}

              {/* ANALYTICS TAB */}
              {tab === 'analytics' && (
                <>
                  <h1 style={{ fontSize: '1.5rem', fontWeight: 700, marginBottom: '1.5rem' }}>Analitik Özeti</h1>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', marginBottom: '1.5rem' }}>
                    <MetricCard icon={<Users size={24} />} iconBg="#f3e8ff" iconColor="#7c3aed" value={`${analytics.engagement_rate}%`} label="Kullanıcı Katılım Oranı" />
                    <MetricCard icon={<Activity size={24} />} iconBg="#fce7f3" iconColor="#db2777" value={`${analytics.active_users} / ${analytics.total_users}`} label="Aktif / Toplam Kullanıcı" />
                  </div>

                  {/* Keyword Bar Chart */}
                  <div className="premium-card" style={{ padding: '1.5rem', marginBottom: '1.5rem' }}>
                    <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.25rem', fontSize: '1rem', fontWeight: 600 }}>
                      <BarChart2 size={18} color="var(--primary)" /> En Çok Sorgulanan Konular (Top 10)
                    </h3>
                    {analytics.top_keywords.length === 0 ? (
                      <p style={{ color: 'var(--text-muted)' }}>Yeterli veri yok.</p>
                    ) : analytics.top_keywords.map((kw, i) => (
                      <div key={i} style={{ marginBottom: '0.75rem' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.3rem', fontSize: '0.85rem' }}>
                          <span style={{ fontWeight: 600 }}>{kw.keyword}</span>
                          <span style={{ color: 'var(--text-muted)' }}>{kw.count} sorgu</span>
                        </div>
                        <div style={{ background: '#f1f5f9', borderRadius: '100px', height: '8px', overflow: 'hidden' }}>
                          <div style={{ height: '100%', borderRadius: '100px', background: `hsl(${220 + i * 15}, 70%, 55%)`, width: `${(kw.count / maxKeyword) * 100}%`, transition: 'width 0.5s ease' }} />
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* Daily Volume */}
                  <div className="premium-card" style={{ padding: '1.5rem' }}>
                    <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.25rem', fontSize: '1rem', fontWeight: 600 }}>
                      <TrendingUp size={18} color="#10b981" /> Günlük Sorgu Hacmi (Son 7 Gün)
                    </h3>
                    {analytics.daily_volume.length === 0 ? (
                      <p style={{ color: 'var(--text-muted)' }}>Yeterli veri yok.</p>
                    ) : (
                      <div style={{ display: 'flex', alignItems: 'flex-end', gap: '0.5rem', height: '100px' }}>
                        {analytics.daily_volume.map((d, i) => {
                          const maxVol = Math.max(...analytics.daily_volume.map(x => x.count), 1)
                          return (
                            <div key={i} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.25rem' }}>
                              <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', fontWeight: 600 }}>{d.count}</span>
                              <div style={{ width: '100%', background: '#10b981', borderRadius: '4px 4px 0 0', height: `${(d.count / maxVol) * 70}px`, minHeight: '4px', transition: 'height 0.5s ease' }} />
                              <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>{d.day.slice(5)}</span>
                            </div>
                          )
                        })}
                      </div>
                    )}
                  </div>
                </>
              )}
            </>
          )}
        </div>
      </main>
    </div>
  )
}

export default AdminPanel
