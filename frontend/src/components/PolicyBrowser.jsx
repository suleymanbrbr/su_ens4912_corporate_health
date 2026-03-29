import React, { useState, useEffect, useRef } from 'react'
import { Link } from 'react-router-dom'
import { ArrowLeft, Search, BookOpen, ChevronRight, X, Filter } from 'lucide-react'

function PolicyBrowser({ user }) {
  const [query, setQuery] = useState('')
  const [section, setSection] = useState('')
  const [results, setResults] = useState([])
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(false)
  const [expanded, setExpanded] = useState(null)
  const [offset, setOffset] = useState(0)
  const debounceRef = useRef(null)
  const LIMIT = 20

  useEffect(() => {
    search(0)
  }, [])

  useEffect(() => {
    setOffset(0)
    clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(() => search(0), 300)
    return () => clearTimeout(debounceRef.current)
  }, [query, section])

  const search = async (newOffset = 0) => {
    setLoading(true)
    try {
      const params = new URLSearchParams({
        q: query,
        section: section,
        limit: LIMIT,
        offset: newOffset
      })
      const res = await fetch(`/api/policies?${params}`, {
        headers: { 'Authorization': `Bearer ${localStorage.getItem('token')}` }
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
    } catch (e) { console.error(e) }
    finally { setLoading(false) }
  }

  const sectionOptions = ['EK-1', 'EK-2', 'EK-3', 'EK-4', 'EK-5', 'EK-6', 'EK-7', 'EK-8', 'EK-9']

  return (
    <div style={{ display: 'flex', height: '100vh', background: 'var(--bg)' }}>
      {/* Sidebar */}
      <aside className="sidebar glass" style={{ width: '260px', borderRight: '1px solid var(--border)', padding: '2rem', display: 'flex', flexDirection: 'column' }}>
        <h2 className="text-gradient" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '2rem', fontSize: '1.2rem' }}>
          <BookOpen size={22} /> SUT Mevzuatı
        </h2>
        <nav style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
          <Link to="/" style={{ textDecoration: 'none', color: 'var(--text-main)', display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.75rem', borderRadius: '8px', fontSize: '0.9rem' }} onMouseOver={e => e.currentTarget.style.background = '#f1f5f9'} onMouseOut={e => e.currentTarget.style.background = 'transparent'}>
            <ArrowLeft size={16} /> Sohbet Ekranı
          </Link>
          <Link to="/admin" style={{ textDecoration: 'none', color: 'var(--text-main)', display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.75rem', borderRadius: '8px', fontSize: '0.9rem' }} onMouseOver={e => e.currentTarget.style.background = '#f1f5f9'} onMouseOut={e => e.currentTarget.style.background = 'transparent'}>
            <ArrowLeft size={16} /> Admin Paneli
          </Link>
        </nav>

        {/* Filter Panel */}
        <div style={{ borderTop: '1px solid var(--border)', paddingTop: '1.5rem' }}>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '1px', marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <Filter size={12} /> Ek Filtresi
          </p>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem' }}>
            <button
              onClick={() => setSection('')}
              style={{ padding: '0.3rem 0.6rem', borderRadius: '6px', border: '1px solid var(--border)', fontSize: '0.75rem', cursor: 'pointer', background: section === '' ? 'var(--primary)' : 'white', color: section === '' ? 'white' : 'var(--text-muted)', fontWeight: 600 }}
            >
              Tümü
            </button>
            {sectionOptions.map(s => (
              <button
                key={s}
                onClick={() => setSection(section === s ? '' : s)}
                style={{ padding: '0.3rem 0.6rem', borderRadius: '6px', border: '1px solid var(--border)', fontSize: '0.75rem', cursor: 'pointer', background: section === s ? 'var(--accent)' : 'white', color: section === s ? 'white' : 'var(--text-muted)', fontWeight: 600 }}
              >
                {s}
              </button>
            ))}
          </div>
        </div>
      </aside>

      {/* Main */}
      <main style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        {/* Search Header */}
        <div style={{ padding: '1.5rem 2rem', borderBottom: '1px solid var(--border)', background: 'white' }}>
          <h1 style={{ fontSize: '1.5rem', fontWeight: 700, marginBottom: '0.25rem' }}>SUT Mevzuat Tarayıcısı</h1>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginBottom: '1rem' }}>
            {total.toLocaleString('tr-TR')} madde arasında arama yapın
          </p>
          <div style={{ position: 'relative' }}>
            <Search size={18} style={{ position: 'absolute', left: '1rem', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }} />
            <input
              value={query}
              onChange={e => setQuery(e.target.value)}
              placeholder="Madde veya konu arayın... (örn: kanser, reçete, katılım payı)"
              style={{ width: '100%', padding: '0.85rem 3rem', borderRadius: '12px', border: '1.5px solid var(--border)', fontSize: '0.95rem', outline: 'none', boxSizing: 'border-box' }}
              onFocus={e => e.target.style.borderColor = 'var(--primary)'}
              onBlur={e => e.target.style.borderColor = 'var(--border)'}
            />
            {query && (
              <button onClick={() => setQuery('')} style={{ position: 'absolute', right: '1rem', top: '50%', transform: 'translateY(-50%)', background: 'transparent', border: 'none', cursor: 'pointer', color: 'var(--text-muted)' }}>
                <X size={16} />
              </button>
            )}
          </div>
        </div>

        {/* Results */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '1.5rem 2rem' }}>
          {loading && results.length === 0 ? (
            <div style={{ textAlign: 'center', padding: '3rem', color: 'var(--text-muted)' }}>Aranıyor...</div>
          ) : results.length === 0 ? (
            <div style={{ textAlign: 'center', padding: '3rem', color: 'var(--text-muted)' }}>Sonuç bulunamadı.</div>
          ) : (
            <>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                {results.map((r) => (
                  <div
                    key={r.id}
                    className="premium-card"
                    style={{ padding: '1.25rem 1.5rem', cursor: 'pointer', borderLeft: expanded === r.id ? '3px solid var(--primary)' : '3px solid transparent', transition: 'border-color 0.2s' }}
                    onClick={() => setExpanded(expanded === r.id ? null : r.id)}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '1rem' }}>
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <p style={{ fontWeight: 600, fontSize: '0.9rem', color: 'var(--primary)', marginBottom: '0.4rem', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                          {r.title}
                        </p>
                        <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)', lineHeight: 1.6 }}>
                          {expanded === r.id ? r.full_text : r.excerpt}
                          {!expanded && r.full_text.length > 300 ? '...' : ''}
                        </p>
                      </div>
                      <ChevronRight
                        size={18}
                        color="var(--text-muted)"
                        style={{ transform: expanded === r.id ? 'rotate(90deg)' : 'none', transition: 'transform 0.2s', flexShrink: 0, marginTop: '0.2rem' }}
                      />
                    </div>
                  </div>
                ))}
              </div>

              {results.length < total && (
                <div style={{ textAlign: 'center', marginTop: '2rem' }}>
                  <button
                    className="btn-secondary"
                    onClick={() => search(offset)}
                    disabled={loading}
                    style={{ padding: '0.75rem 2rem' }}
                  >
                    {loading ? 'Yükleniyor...' : 'Daha Fazla Göster'}
                  </button>
                </div>
              )}
            </>
          )}
        </div>
      </main>
    </div>
  )
}

export default PolicyBrowser
