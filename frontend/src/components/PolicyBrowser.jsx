import React, { useState, useEffect, useRef } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import { ArrowLeft, Search, BookOpen, ChevronRight, X, Filter, Share2, Loader, Shield, MessageSquare } from 'lucide-react'
import MarkdownBody from './MarkdownBody'

const NODE_COLORS = { RULE: '#3b82f6', DRUG: '#10b981', DIAGNOSIS: '#ef4444', SPECIALIST: '#f59e0b', CONDITION: '#8b5cf6', DOCUMENT: '#06b6d4', DEVICE: '#ec4899', DOSAGE: '#84cc16', AGE_LIMIT: '#f97316', EXCLUSION: '#6b7280' }
const NODE_TYPE_TR = { RULE: 'SUT Kuralı', DRUG: 'İlaç', DIAGNOSIS: 'Teşhis', SPECIALIST: 'Uzman', CONDITION: 'Koşul', DOCUMENT: 'Belge', DEVICE: 'Cihaz', DOSAGE: 'Doz', AGE_LIMIT: 'Yaş Sınırı', EXCLUSION: 'Dışlama' }

function escapeReg(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

function HighlightText({ text, term }) {
  if (!text) return null
  const t = (term || '').trim()
  if (!t || t.length < 2) return <>{text}</>
  try {
    const parts = String(text).split(new RegExp(`(${escapeReg(t)})`, 'gi'))
    return (
      <>
        {parts.map((p, i) =>
          p.toLowerCase() === t.toLowerCase() ? (
            <mark key={i} style={{ background: '#fef08a', padding: '0 1px', borderRadius: 2 }}>{p}</mark>
          ) : (
            <span key={i}>{p}</span>
          )
        )}
      </>
    )
  } catch {
    return <>{text}</>
  }
}

/** Best-effort RG / tarih / revizyon from ingest metadata */
function PolicyMetaSummary({ metadata }) {
  if (!metadata || typeof metadata !== 'object') return null
  const rg =
    metadata.resmi_gazete ||
    metadata.rg_no ||
    metadata.RG ||
    metadata['Resmi Gazete'] ||
    metadata.resmiGazete
  const tarih =
    metadata.son_degisiklik ||
    metadata.tarih ||
    metadata.date ||
    metadata.yayin_tarihi
  const rev = metadata.revision_count ?? metadata.degisiklik_sayisi ?? metadata.change_count
  if (!rg && !tarih && rev == null) return null
  return (
    <div style={{ marginTop: '0.75rem', padding: '0.65rem 0.85rem', borderRadius: '8px', background: 'var(--bg)', border: '1px solid var(--border)', fontSize: '0.78rem', color: 'var(--text-muted)' }}>
      {rg && (
        <div>
          <strong style={{ color: 'var(--text-main)' }}>Resmî Gazete / kaynak:</strong> {String(rg)}
        </div>
      )}
      {tarih && (
        <div style={{ marginTop: '0.35rem' }}>
          <strong style={{ color: 'var(--text-main)' }}>Tarih / güncelleme:</strong> {String(tarih)}
        </div>
      )}
      {rev != null && Number(rev) > 0 && (
        <div style={{ marginTop: '0.35rem' }}>
          Bu madde kayıtlarında <strong style={{ color: 'var(--text-main)' }}>{String(rev)}</strong> değişiklik izi bulunuyor (metadata).
        </div>
      )}
    </div>
  )
}

function KGPanel({ chunkTitle }) {
  const [nodes, setNodes] = useState(null)
  const [loading, setLoading] = useState(true)
  useEffect(() => {
    if (!chunkTitle) return
    setLoading(true)
    const q = chunkTitle.slice(0, 60)
    fetch(`/api/kg/nodes?q=${encodeURIComponent(q)}&limit=8`, {
      headers: { Authorization: `Bearer ${localStorage.getItem('token')}` },
    })
      .then(r => (r.ok ? r.json() : { nodes: [] }))
      .then(d => {
        setNodes(d.nodes || [])
      })
      .catch(() => setNodes([]))
      .finally(() => setLoading(false))
  }, [chunkTitle])

  if (loading)
    return (
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-muted)', fontSize: '0.8rem', marginTop: '0.75rem' }}>
        <Loader size={12} /> Bilgi Grafiği aranıyor...
      </div>
    )
  if (!nodes || nodes.length === 0) return null

  return (
    <div style={{ marginTop: '0.75rem', paddingTop: '0.75rem', borderTop: '1px solid var(--border)' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', marginBottom: '0.5rem', fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
        <Share2 size={12} /> Bilgi Grafiği İlişkili Düğümler
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem' }}>
        {nodes.map(n => (
          <span
            key={n.node_id}
            title={n.text_content || ''}
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.3rem',
              padding: '0.25rem 0.65rem',
              borderRadius: '20px',
              fontSize: '0.75rem',
              fontWeight: 600,
              background: (NODE_COLORS[n.type] || '#94a3b8') + '18',
              border: `1px solid ${(NODE_COLORS[n.type] || '#94a3b8')}44`,
              color: NODE_COLORS[n.type] || '#94a3b8',
            }}
          >
            <span style={{ width: 6, height: 6, borderRadius: '50%', background: NODE_COLORS[n.type] || '#94a3b8', flexShrink: 0, display: 'inline-block' }} />
            {n.label}
            <span style={{ opacity: 0.6, fontSize: '0.65rem' }}>{NODE_TYPE_TR[n.type] || n.type}</span>
          </span>
        ))}
      </div>
    </div>
  )
}

function PolicyBrowser({ user }) {
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const [query, setQuery] = useState('')
  const [section, setSection] = useState('')
  const [dateFrom, setDateFrom] = useState('')
  const [dateTo, setDateTo] = useState('')
  const [statusFilter, setStatusFilter] = useState('')
  /** phrase | and | or — backend `/api/policies?q_mode=` */
  const [qMode, setQMode] = useState('phrase')
  const [results, setResults] = useState([])
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(false)
  const [expanded, setExpanded] = useState(null)
  const [offset, setOffset] = useState(0)
  const debounceRef = useRef(null)
  const cardRefs = useRef({})
  const LIMIT = 20

  useEffect(() => {
    document.title = 'Mevzuat Tarayıcısı — SUT Asistanı'
  }, [])

  useEffect(() => {
    const chunkId = searchParams.get('chunkId')
    const highlight = searchParams.get('highlight')
    if (chunkId) {
      setExpanded(chunkId)
      searchDirectChunk(chunkId)
    } else if (highlight) {
      setQuery(highlight)
    }
  }, [])

  const searchDirectChunk = async cid => {
    setLoading(true)
    try {
      const res = await fetch(`/api/policies?chunk_id=${encodeURIComponent(cid)}&limit=5&offset=0`, {
        headers: { Authorization: `Bearer ${localStorage.getItem('token')}` },
      })
      if (res.ok) {
        const data = await res.json()
        setResults(data.results || [])
        setTotal(data.total || 0)
        setOffset(LIMIT)
      }
    } catch (e) {
      console.error(e)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    setOffset(0)
    clearTimeout(debounceRef.current)
    const cid = searchParams.get('chunkId')
    /* Tek madde chunk_id ile açıldıysa boş genel arama sonuçlarıyla üzerine yazmayı engelle */
    if (cid && !query.trim() && !section && !dateFrom && !dateTo && !statusFilter) {
      return () => clearTimeout(debounceRef.current)
    }
    debounceRef.current = setTimeout(() => search(0), 300)
    return () => clearTimeout(debounceRef.current)
  }, [query, section, dateFrom, dateTo, statusFilter, qMode, searchParams])

  useEffect(() => {
    const hl = searchParams.get('highlight')
    if (!hl || !results.length) return
    const match = results.find(r => r.title.includes(hl) || r.full_text.includes(hl))
    if (match) {
      setExpanded(match.id)
      setTimeout(() => {
        cardRefs.current[match.id]?.scrollIntoView({ behavior: 'smooth', block: 'center' })
      }, 100)
    }
  }, [results, searchParams])

  const search = async (newOffset = 0) => {
    setLoading(true)
    try {
      const params = new URLSearchParams({
        q: query,
        section,
        limit: LIMIT,
        offset: newOffset,
      })
      if (dateFrom) params.set('date_from', dateFrom)
      if (dateTo) params.set('date_to', dateTo)
      if (statusFilter) params.set('status_filter', statusFilter)
      if (qMode && qMode !== 'phrase') params.set('q_mode', qMode)
      const res = await fetch(`/api/policies?${params}`, {
        headers: { Authorization: `Bearer ${localStorage.getItem('token')}` },
      })
      if (res.ok) {
        const data = await res.json()
        if (newOffset === 0) {
          setResults(data.results)
        } else {
          setResults(prev => [...prev, ...data.results])
        }
        setTotal(data.total)
        setOffset(newOffset + LIMIT)
      }
    } catch (e) {
      console.error(e)
    } finally {
      setLoading(false)
    }
  }

  const askAbout = chunkId => {
    try {
      sessionStorage.setItem('prefillChat', `Bu maddeyi açıklayın: ${chunkId}`)
    } catch (_) {}
    navigate('/')
  }

  const sectionOptions = ['Genel Hükümler', 'Tanı ve Tedavi', 'İlaç', 'Son Hükümler']

  return (
    <div className="policy-layout" style={{ display: 'flex', minHeight: '100vh', background: 'var(--bg)' }}>
      <aside className="sidebar glass policy-sidebar" style={{ width: 'min(260px, 92vw)', borderRight: '1px solid var(--border)', padding: 'clamp(1rem, 3vw, 2rem)', display: 'flex', flexDirection: 'column' }}>
        <nav aria-label="Breadcrumb" style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>
          <Link to="/" style={{ color: 'var(--primary)', fontWeight: 600 }}>
            Ana sayfa
          </Link>{' '}
          / <span style={{ color: 'var(--text-main)', fontWeight: 600 }}>Mevzuat</span>
        </nav>
        <h2 className="text-gradient" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.5rem', fontSize: '1.2rem' }}>
          <BookOpen size={22} /> SUT Mevzuatı
        </h2>
        <nav style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
          <Link to="/" style={{ textDecoration: 'none', color: 'var(--text-main)', display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.75rem', borderRadius: '8px', fontSize: '0.9rem' }}>
            <ArrowLeft size={16} /> Sohbet
          </Link>
          {user?.role === 'admin' && (
            <Link to="/admin" style={{ textDecoration: 'none', color: 'var(--accent)', display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.75rem', borderRadius: '8px', fontSize: '0.9rem' }}>
              <Shield size={16} /> Admin Paneli
            </Link>
          )}
        </nav>

        <div style={{ borderTop: '1px solid var(--border)', paddingTop: '1.5rem' }}>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '1px', marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <Filter size={12} /> Bölüm
          </p>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem' }}>
            <button type="button" onClick={() => setSection('')} style={{ padding: '0.3rem 0.6rem', borderRadius: '6px', border: '1px solid var(--border)', fontSize: '0.75rem', cursor: 'pointer', background: section === '' ? 'var(--primary)' : 'var(--card-bg)', color: section === '' ? 'var(--card-bg)' : 'var(--text-muted)', fontWeight: 600 }}>
              Tümü
            </button>
            {sectionOptions.map(s => (
              <button type="button" key={s} onClick={() => setSection(section === s ? '' : s)} style={{ padding: '0.3rem 0.6rem', borderRadius: '6px', border: '1px solid var(--border)', fontSize: '0.75rem', cursor: 'pointer', background: section === s ? 'var(--accent)' : 'var(--card-bg)', color: section === s ? 'white' : 'var(--text-muted)', fontWeight: 600 }}>
                {s}
              </button>
            ))}
          </div>
          <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '1rem', marginBottom: '0.35rem' }}>Tarih / durum (metin eşleşmesi)</p>
          <input
            type="text"
            placeholder="Örn. 2023"
            value={dateFrom}
            onChange={e => setDateFrom(e.target.value)}
            style={{ width: '100%', marginBottom: '0.35rem', padding: '0.4rem', borderRadius: '6px', border: '1px solid var(--border)', fontSize: '0.78rem', boxSizing: 'border-box' }}
          />
          <input
            type="text"
            placeholder="RG / ek bilgi"
            value={dateTo}
            onChange={e => setDateTo(e.target.value)}
            style={{ width: '100%', marginBottom: '0.35rem', padding: '0.4rem', borderRadius: '6px', border: '1px solid var(--border)', fontSize: '0.78rem', boxSizing: 'border-box' }}
          />
          <input
            type="text"
            placeholder="Durum (aktif…)"
            value={statusFilter}
            onChange={e => setStatusFilter(e.target.value)}
            style={{ width: '100%', padding: '0.4rem', borderRadius: '6px', border: '1px solid var(--border)', fontSize: '0.78rem', boxSizing: 'border-box' }}
          />
        </div>
      </aside>

      <main style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', minWidth: 0 }}>
        <div style={{ padding: '1.5rem 2rem', borderBottom: '1px solid var(--border)', background: 'var(--card-bg)' }}>
          <h1 style={{ fontSize: 'clamp(1.1rem, 3vw, 1.5rem)', fontWeight: 700, marginBottom: '0.25rem' }}>SUT Mevzuat Tarayıcısı</h1>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginBottom: '1rem' }}>{total.toLocaleString('tr-TR')} kayıt indeksinde arama</p>
          <div style={{ position: 'relative' }}>
            <Search size={18} style={{ position: 'absolute', left: '1rem', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }} aria-hidden />
            <input
              value={query}
              onChange={e => setQuery(e.target.value)}
              placeholder="Madde veya konu arayın..."
              aria-label="Mevzuat araması"
              style={{ width: '100%', padding: '0.85rem 3rem', borderRadius: '12px', border: '1.5px solid var(--border)', fontSize: '0.95rem', outline: 'none', boxSizing: 'border-box', background: 'var(--bg)', color: 'var(--text-main)' }}
            />
            {query && (
              <button type="button" aria-label="Aramayı temizle" onClick={() => setQuery('')} style={{ position: 'absolute', right: '1rem', top: '50%', transform: 'translateY(-50%)', background: 'transparent', border: 'none', cursor: 'pointer', color: 'var(--text-muted)' }}>
                <X size={16} />
              </button>
            )}
          </div>
          <div style={{ marginTop: '0.75rem', display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '0.5rem' }}>
            <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontWeight: 600 }}>Arama mantığı</span>
            <select
              value={qMode}
              onChange={e => setQMode(e.target.value)}
              aria-label="Arama kelime mantığı"
              style={{ padding: '0.35rem 0.6rem', borderRadius: '8px', border: '1px solid var(--border)', fontSize: '0.8rem', background: 'var(--bg)', color: 'var(--text-main)' }}
            >
              <option value="phrase">Tam ifade (varsayılan)</option>
              <option value="and">Tüm kelimeler (AND)</option>
              <option value="or">Herhangi bir kelime (OR)</option>
            </select>
            <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', flex: '1 1 200px' }}>
              Virgül ile ayırarak gruplar: <code style={{ fontSize: '0.68rem' }}>ilaç, rapor</code> → (ilaç) VEYA (rapor); her grupta boşlukla AND.
            </span>
          </div>
        </div>

        <div style={{ flex: 1, overflowY: 'auto', padding: '1.5rem clamp(1rem, 3vw, 2rem)' }}>
          {loading && results.length === 0 ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              {[1, 2, 3, 4, 5].map(i => (
                <div key={i} className="loading-skeleton" style={{ height: '96px', borderRadius: '12px', width: '100%' }} />
              ))}
            </div>
          ) : results.length === 0 ? (
            <div style={{ textAlign: 'center', padding: '3rem', color: 'var(--text-muted)' }}>Sonuç bulunamadı.</div>
          ) : (
            <>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                {results.map(r => (
                  <article
                    key={r.id}
                    ref={el => {
                      cardRefs.current[r.id] = el
                    }}
                    className="premium-card policy-card"
                    style={{
                      padding: '1.25rem 1.5rem',
                      cursor: 'pointer',
                      borderLeft: expanded === r.id ? '3px solid var(--primary)' : '3px solid transparent',
                      transition: 'border-color 0.2s',
                    }}
                    onClick={() => setExpanded(expanded === r.id ? null : r.id)}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '1rem' }}>
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <p style={{ fontWeight: 600, fontSize: '0.9rem', color: 'var(--primary)', marginBottom: '0.4rem', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                          {r.title}
                        </p>
                        {expanded === r.id ? (
                          <div className="policy-detail-text" style={{ fontSize: '0.9rem', color: 'var(--text-main)', lineHeight: 1.65 }}>
                            <PolicyMetaSummary metadata={r.metadata} />
                            <MarkdownBody>{r.full_text}</MarkdownBody>
                            <button
                              type="button"
                              className="btn-primary"
                              style={{ marginTop: '1rem', display: 'inline-flex', alignItems: 'center', gap: '0.35rem', fontSize: '0.82rem' }}
                              onClick={e => {
                                e.stopPropagation()
                                askAbout(r.title)
                              }}
                            >
                              <MessageSquare size={14} /> Bu madde hakkında soru sor
                            </button>
                            <KGPanel chunkTitle={r.title} />
                          </div>
                        ) : (
                          <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)', lineHeight: 1.6 }}>
                            <HighlightText text={r.excerpt} term={query} />
                            {r.full_text.length > 300 ? '…' : ''}
                          </p>
                        )}
                      </div>
                      <ChevronRight size={18} color="var(--text-muted)" style={{ transform: expanded === r.id ? 'rotate(90deg)' : 'none', transition: 'transform 0.2s', flexShrink: 0, marginTop: '0.2rem' }} aria-hidden />
                    </div>
                  </article>
                ))}
              </div>

              {results.length < total && (
                <div style={{ textAlign: 'center', marginTop: '2rem' }}>
                  <button type="button" className="btn-secondary" onClick={() => search(offset)} disabled={loading} style={{ padding: '0.75rem 2rem' }}>
                    {loading ? 'Yükleniyor…' : 'Daha fazla'}
                  </button>
                </div>
              )}
            </>
          )}
        </div>
      </main>
      <style>{`
        .loading-skeleton { background: linear-gradient(90deg, rgba(0,0,0,0.06) 25%, rgba(0,0,0,0.12) 50%, rgba(0,0,0,0.06) 75%); background-size: 200% 100%; animation: pol-sh 1.4s infinite; }
        @keyframes pol-sh { from { background-position: 200% 0; } to { background-position: -200% 0; } }
        @media (max-width: 768px) {
          .policy-layout { flex-direction: column; }
          .policy-sidebar { width: 100% !important; border-right: none; border-bottom: 1px solid var(--border); }
        }
      `}</style>
    </div>
  )
}

export default PolicyBrowser
