import React, { useState, useEffect, useCallback, useMemo } from 'react'
import { List, Search, ChevronLeft, ChevronRight, RefreshCw } from 'lucide-react'
import { SkeletonLine } from './Skeleton'

const PAGE_SIZE = 50

const AUTH_HEADERS = () => ({ Authorization: `Bearer ${localStorage.getItem('token')}` })

/**
 * Normalize a single log row coming from /api/admin/audit-logs to the
 * shape the table expects. The backend has shipped two slightly different
 * field names over the project's life (action_type vs action,
 * entity_type vs target_type), so we accept both and fall back gracefully.
 */
function normalizeLog(raw) {
  if (!raw || typeof raw !== 'object') return null
  return {
    id: raw.log_id ?? raw.id ?? `${raw.created_at}-${Math.random()}`,
    timestamp: raw.created_at ?? raw.timestamp ?? null,
    user_id: raw.user_id ?? raw.user_name ?? raw.username ?? null,
    action: raw.action ?? raw.action_type ?? null,
    target_type: raw.target_type ?? raw.entity_type ?? null,
    target_id: raw.target_id ?? raw.entity_id ?? null,
    ip_address: raw.ip_address ?? raw.ip ?? null,
    details: raw.details ?? null,
  }
}

/**
 * Admin tab/section that lists recent audit log entries with simple
 * client-side filtering + paginated server fetches.
 *
 * The endpoint may not exist on every environment — if the request
 * returns a 404 we render "Henüz denetim kaydı yok" instead of crashing.
 */
export default function AuditLogViewer() {
  const [page, setPage] = useState(0)        // 0-indexed
  const [logs, setLogs] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [notFound, setNotFound] = useState(false)
  const [hasNext, setHasNext] = useState(false)

  const [filterAction, setFilterAction] = useState('')
  const [filterUser, setFilterUser] = useState('')

  const fetchLogs = useCallback(async () => {
    setLoading(true)
    setError(null)
    setNotFound(false)
    try {
      const offset = page * PAGE_SIZE
      const r = await fetch(`/api/admin/audit-logs?limit=${PAGE_SIZE}&offset=${offset}`, {
        headers: AUTH_HEADERS(),
      })
      if (r.status === 404) {
        setLogs([])
        setNotFound(true)
        setHasNext(false)
        return
      }
      if (r.status === 401) {
        try { localStorage.removeItem('token') } catch (_) {}
        if (typeof window !== 'undefined') window.location.assign('/login')
        return
      }
      if (!r.ok) {
        throw new Error(`HTTP ${r.status}`)
      }
      const data = await r.json()
      // Accept both `{ logs: [...] }` and a bare array payload.
      const rows = Array.isArray(data) ? data : (data.logs || data.items || [])
      const normalized = rows.map(normalizeLog).filter(Boolean)
      setLogs(normalized)
      // Heuristic: if the backend returned the full page, there's
      // likely more; otherwise we've hit the end. Backend may also
      // return a `has_more` flag, which we respect when present.
      const more = (typeof data.has_more === 'boolean')
        ? data.has_more
        : normalized.length === PAGE_SIZE
      setHasNext(more)
    } catch (e) {
      if (import.meta.env.DEV) console.error('[AuditLogViewer]', e)
      setError(e.message || 'Bilinmeyen hata')
      setLogs([])
      setHasNext(false)
    } finally {
      setLoading(false)
    }
  }, [page])

  useEffect(() => { fetchLogs() }, [fetchLogs])

  // Client-side filter for the current page.
  const visibleLogs = useMemo(() => {
    const aLower = filterAction.trim().toLowerCase()
    const uLower = filterUser.trim().toLowerCase()
    return logs.filter(l => {
      if (aLower && !(l.action || '').toLowerCase().includes(aLower)) return false
      if (uLower && !String(l.user_id || '').toLowerCase().includes(uLower)) return false
      return true
    })
  }, [logs, filterAction, filterUser])

  const actionOptions = useMemo(() => {
    const seen = new Set()
    logs.forEach(l => { if (l.action) seen.add(l.action) })
    return Array.from(seen).sort()
  }, [logs])

  return (
    <section aria-labelledby="audit-log-title">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem', flexWrap: 'wrap', gap: '0.75rem' }}>
        <h1 id="audit-log-title" style={{ fontSize: '1.5rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <List size={22} /> Denetim Kayıtları
        </h1>
        <button
          type="button"
          onClick={() => fetchLogs()}
          disabled={loading}
          className="btn-secondary"
          style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.8rem' }}
        >
          <RefreshCw size={14} className={loading ? 'spin' : ''} /> Yenile
        </button>
      </div>

      {/* Filters */}
      <div className="premium-card" style={{ padding: '1rem 1.25rem', marginBottom: '1rem', display: 'flex', gap: '0.75rem', flexWrap: 'wrap', alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', flex: '1 1 220px' }}>
          <Search size={14} color="var(--text-muted)" aria-hidden />
          <input
            type="search"
            placeholder="Kullanıcı ID ile ara…"
            value={filterUser}
            onChange={e => setFilterUser(e.target.value)}
            aria-label="Kullanıcı kimliği ile ara"
            style={{ flex: 1, padding: '0.45rem 0.6rem', borderRadius: '8px', border: '1px solid var(--border)', fontSize: '0.85rem', boxSizing: 'border-box' }}
          />
        </div>
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', fontSize: '0.85rem' }}>
          <span style={{ color: 'var(--text-muted)' }}>Aksiyon:</span>
          <select
            value={filterAction}
            onChange={e => setFilterAction(e.target.value)}
            aria-label="Aksiyona göre filtrele"
            style={{ padding: '0.4rem 0.6rem', borderRadius: '8px', border: '1px solid var(--border)', fontSize: '0.85rem', background: 'var(--card-bg)' }}
          >
            <option value="">Tümü</option>
            {actionOptions.map(a => (
              <option key={a} value={a}>{a}</option>
            ))}
          </select>
        </label>
      </div>

      {/* Table */}
      <div className="premium-card" style={{ padding: 0, overflow: 'hidden' }}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', minWidth: '720px' }}>
            <thead style={{ background: 'var(--bg-elev)', borderBottom: '1px solid var(--border)' }}>
              <tr>
                <th style={{ padding: '0.85rem 1.25rem', fontSize: '0.78rem', textTransform: 'uppercase', letterSpacing: '0.5px', color: 'var(--text-muted)' }}>Tarih</th>
                <th style={{ padding: '0.85rem 1.25rem', fontSize: '0.78rem', textTransform: 'uppercase', letterSpacing: '0.5px', color: 'var(--text-muted)' }}>Kullanıcı</th>
                <th style={{ padding: '0.85rem 1.25rem', fontSize: '0.78rem', textTransform: 'uppercase', letterSpacing: '0.5px', color: 'var(--text-muted)' }}>Aksiyon</th>
                <th style={{ padding: '0.85rem 1.25rem', fontSize: '0.78rem', textTransform: 'uppercase', letterSpacing: '0.5px', color: 'var(--text-muted)' }}>Hedef Türü</th>
                <th style={{ padding: '0.85rem 1.25rem', fontSize: '0.78rem', textTransform: 'uppercase', letterSpacing: '0.5px', color: 'var(--text-muted)' }}>Hedef ID</th>
                <th style={{ padding: '0.85rem 1.25rem', fontSize: '0.78rem', textTransform: 'uppercase', letterSpacing: '0.5px', color: 'var(--text-muted)' }}>IP</th>
              </tr>
            </thead>
            <tbody>
              {loading && (
                Array.from({ length: 6 }).map((_, i) => (
                  <tr key={`s-${i}`} style={{ borderBottom: '1px solid var(--border)' }}>
                    {Array.from({ length: 6 }).map((__, j) => (
                      <td key={j} style={{ padding: '0.85rem 1.25rem' }}>
                        <SkeletonLine width={`${50 + ((i + j) % 4) * 10}%`} height="0.75rem" />
                      </td>
                    ))}
                  </tr>
                ))
              )}
              {!loading && visibleLogs.map((log) => (
                <tr key={log.id} style={{ borderBottom: '1px solid var(--border)' }}>
                  <td style={{ padding: '0.85rem 1.25rem', fontSize: '0.82rem', color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>
                    {log.timestamp ? new Date(log.timestamp + (String(log.timestamp).endsWith('Z') ? '' : 'Z')).toLocaleString('tr-TR') : '—'}
                  </td>
                  <td style={{ padding: '0.85rem 1.25rem', fontSize: '0.85rem', fontWeight: 600 }}>
                    {log.user_id ?? <span style={{ color: 'var(--text-muted)' }}>Sistem</span>}
                  </td>
                  <td style={{ padding: '0.85rem 1.25rem' }}>
                    <span style={{ padding: '0.2rem 0.6rem', background: '#e0f2fe', color: '#0284c7', borderRadius: '6px', fontSize: '0.72rem', fontWeight: 700 }}>
                      {(log.action || '—').toString().toUpperCase()}
                    </span>
                  </td>
                  <td style={{ padding: '0.85rem 1.25rem', fontSize: '0.82rem', color: 'var(--primary)' }}>
                    {log.target_type || '—'}
                  </td>
                  <td style={{ padding: '0.85rem 1.25rem', fontSize: '0.82rem', color: 'var(--text-main)' }}>
                    {log.target_id ?? (log.details ? <code style={{ fontSize: '0.72rem' }}>{JSON.stringify(log.details).slice(0, 40)}</code> : '—')}
                  </td>
                  <td style={{ padding: '0.85rem 1.25rem', fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                    {log.ip_address || '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {!loading && !error && visibleLogs.length === 0 && (
          <p style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.9rem' }}>
            {notFound
              ? 'Henüz denetim kaydı yok.'
              : (filterAction || filterUser)
                ? 'Filtreyle eşleşen kayıt bulunamadı.'
                : 'Henüz denetim kaydı yok.'}
          </p>
        )}

        {!loading && error && (
          <p style={{ padding: '1.5rem', textAlign: 'center', color: '#b91c1c', fontSize: '0.85rem' }}>
            Kayıtlar yüklenirken hata oluştu: {error}
          </p>
        )}
      </div>

      {/* Pagination */}
      {!notFound && (
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '1rem' }}>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
            Sayfa {page + 1} · {visibleLogs.length} kayıt gösteriliyor
          </span>
          <div style={{ display: 'flex', gap: '0.5rem' }}>
            <button
              type="button"
              onClick={() => setPage(p => Math.max(0, p - 1))}
              disabled={loading || page === 0}
              className="btn-secondary"
              style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem', fontSize: '0.8rem', padding: '0.45rem 0.85rem' }}
            >
              <ChevronLeft size={14} /> Önceki
            </button>
            <button
              type="button"
              onClick={() => setPage(p => p + 1)}
              disabled={loading || !hasNext}
              className="btn-secondary"
              style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem', fontSize: '0.8rem', padding: '0.45rem 0.85rem' }}
            >
              Sonraki <ChevronRight size={14} />
            </button>
          </div>
        </div>
      )}
    </section>
  )
}
