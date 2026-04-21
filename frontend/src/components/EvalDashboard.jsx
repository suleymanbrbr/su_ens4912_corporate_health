import React, { useState, useEffect } from 'react'
import { BarChart2, Zap, Target, Clock, RefreshCw, CheckCircle, XCircle, FlaskConical } from 'lucide-react'

const API = (path, opts = {}) => fetch(path, {
  ...opts,
  headers: { 'Authorization': `Bearer ${localStorage.getItem('token')}`, ...(opts.headers || {}) }
})

const METRICS = ['hit_rate', 'mrr', 'ndcg', 'precision']
const METRIC_TR = { hit_rate: 'Hit Rate', mrr: 'MRR', ndcg: 'NDCG', precision: 'Precision' }
const K_VALUES = [1, 3, 5, 10]
const NEW_COLOR = '#3b82f6'
const OLD_COLOR = '#94a3b8'

function pct(v) { return (v * 100).toFixed(1) + '%' }

function MetricBar({ label, newVal, oldVal, k }) {
  const max = Math.max(newVal, oldVal, 0.01)
  const improvement = ((newVal - oldVal) / oldVal * 100).toFixed(1)
  const improved = newVal >= oldVal
  return (
    <div style={{ marginBottom: '1rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.3rem', fontSize: '0.82rem' }}>
        <span style={{ fontWeight: 600 }}>{label}@{k}</span>
        <span style={{ color: improved ? '#10b981' : '#ef4444', fontWeight: 700, fontSize: '0.78rem' }}>
          {improved ? '▲' : '▼'} {Math.abs(improvement)}%
        </span>
      </div>
      {/* New system */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.2rem' }}>
        <span style={{ width: 50, fontSize: '0.72rem', color: NEW_COLOR, fontWeight: 700, textAlign: 'right' }}>Yeni</span>
        <div style={{ flex: 1, background: '#f1f5f9', borderRadius: 6, height: 10, overflow: 'hidden' }}>
          <div style={{ width: `${(newVal / max) * 100}%`, height: '100%', background: NEW_COLOR, borderRadius: 6, transition: 'width 0.6s ease' }} />
        </div>
        <span style={{ width: 40, fontSize: '0.72rem', color: NEW_COLOR, fontWeight: 600 }}>{pct(newVal)}</span>
      </div>
      {/* Old system */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <span style={{ width: 50, fontSize: '0.72rem', color: OLD_COLOR, fontWeight: 700, textAlign: 'right' }}>Eski</span>
        <div style={{ flex: 1, background: '#f1f5f9', borderRadius: 6, height: 10, overflow: 'hidden' }}>
          <div style={{ width: `${(oldVal / max) * 100}%`, height: '100%', background: OLD_COLOR, borderRadius: 6, transition: 'width 0.6s ease' }} />
        </div>
        <span style={{ width: 40, fontSize: '0.72rem', color: OLD_COLOR, fontWeight: 600 }}>{pct(oldVal)}</span>
      </div>
    </div>
  )
}

export default function EvalDashboard() {
  const [evalData, setEvalData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [activeMetric, setActiveMetric] = useState('hit_rate')
  const [benchmark, setBenchmark] = useState(null)
  const [benchRunning, setBenchRunning] = useState(false)

  useEffect(() => {
    API('/api/admin/eval/results')
      .then(r => r.ok ? r.json() : Promise.reject('API error'))
      .then(d => { setEvalData(d); setLoading(false) })
      .catch(e => { setError(String(e)); setLoading(false) })
  }, [])

  const runBenchmark = async () => {
    setBenchRunning(true)
    setBenchmark(null)
    try {
      const r = await API('/api/admin/kg/benchmark', { method: 'POST' })
      if (r.ok) setBenchmark(await r.json())
      else setBenchmark({ error: 'API error' })
    } catch(e) { setBenchmark({ error: String(e) }) }
    setBenchRunning(false)
  }

  if (loading) return (
    <div style={{ padding: '3rem', textAlign: 'center', color: 'var(--text-muted)' }}>
      <RefreshCw size={28} style={{ opacity: 0.3, animation: 'spin 1s linear infinite', display: 'block', margin: '0 auto 0.75rem' }} />
      Değerlendirme sonuçları yükleniyor...
    </div>
  )
  if (error) return (
    <div style={{ padding: '2rem', color: '#ef4444' }}>Hata: {error}</div>
  )

  const n = evalData.new_system.metrics
  const o = evalData.old_system.metrics

  // Build spider/bar chart data for selected metric across k values
  const chartData = K_VALUES.map(k => ({
    k,
    new: n[`${activeMetric}@${k}`],
    old: o[`${activeMetric}@${k}`],
  }))
  const maxVal = Math.max(...chartData.flatMap(d => [d.new, d.old]), 0.01)

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
        <h1 style={{ fontSize: '1.5rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <BarChart2 size={22} color="var(--primary)" /> Değerlendirme Sonuçları
        </h1>
        <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
          {evalData.num_questions} soru • {new Date(evalData.evaluation_date).toLocaleDateString('tr-TR')}
        </div>
      </div>

      {/* System name cards */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1.5rem' }}>
        <div style={{ padding: '1rem', background: `${NEW_COLOR}10`, border: `1px solid ${NEW_COLOR}33`, borderRadius: '12px' }}>
          <div style={{ fontSize: '0.7rem', color: NEW_COLOR, fontWeight: 700, textTransform: 'uppercase', marginBottom: '0.35rem' }}>Yeni Sistem</div>
          <div style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--text-main)' }}>{evalData.new_system.name}</div>
        </div>
        <div style={{ padding: '1rem', background: '#f8fafc', border: '1px solid var(--border)', borderRadius: '12px' }}>
          <div style={{ fontSize: '0.7rem', color: OLD_COLOR, fontWeight: 700, textTransform: 'uppercase', marginBottom: '0.35rem' }}>Eski Sistem (Baseline)</div>
          <div style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--text-main)' }}>{evalData.old_system.name}</div>
        </div>
      </div>

      {/* Latency highlight */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem', marginBottom: '1.5rem' }}>
        {[
          { label: 'Hit Rate@1', val: n['hit_rate@1'], old: o['hit_rate@1'] },
          { label: 'MRR@3', val: n['mrr@3'], old: o['mrr@3'] },
          { label: 'NDCG@5', val: n['ndcg@5'], old: o['ndcg@5'] },
          { label: 'Ort. Gecikme', val: `${n.avg_latency_sec}s`, old: `${o.avg_latency_sec}s`, isLatency: true },
        ].map(m => {
          const improved = m.isLatency
            ? parseFloat(m.val) <= parseFloat(m.old)
            : parseFloat(m.val) >= parseFloat(m.old)
          return (
            <div key={m.label} style={{ padding: '1rem', background: 'var(--card-bg)', border: '1px solid var(--border)', borderRadius: '12px', textAlign: 'center' }}>
              <div style={{ fontSize: '1.4rem', fontWeight: 800, color: NEW_COLOR }}>{typeof m.val === 'number' ? pct(m.val) : m.val}</div>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginBottom: '0.25rem' }}>{m.label}</div>
              <div style={{ fontSize: '0.72rem', color: improved ? '#10b981' : '#ef4444', fontWeight: 600 }}>
                vs {typeof m.old === 'number' ? pct(m.old) : m.old} ({improved ? '↑ iyileşme' : '↓ gerileme'})
              </div>
            </div>
          )
        })}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', marginBottom: '1.5rem' }}>
        {/* Metric selector + bar chart */}
        <div className="premium-card" style={{ padding: '1.5rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
            <h3 style={{ margin: 0, fontSize: '0.95rem', fontWeight: 700 }}>Metrik Karşılaştırması</h3>
            <div style={{ display: 'flex', gap: '0.4rem' }}>
              {METRICS.map(m => (
                <button key={m} onClick={() => setActiveMetric(m)}
                  style={{ padding: '0.25rem 0.6rem', borderRadius: '20px', border: `1px solid ${activeMetric === m ? NEW_COLOR : 'var(--border)'}`, background: activeMetric === m ? NEW_COLOR : 'transparent', color: activeMetric === m ? 'white' : 'var(--text-muted)', fontSize: '0.72rem', cursor: 'pointer', fontWeight: 600 }}>
                  {METRIC_TR[m]}
                </button>
              ))}
            </div>
          </div>
          <div style={{ display: 'flex', alignItems: 'flex-end', gap: '1rem', height: 140, marginBottom: '0.5rem' }}>
            {chartData.map(d => (
              <div key={d.k} style={{ flex: 1, display: 'flex', alignItems: 'flex-end', gap: 4, height: '100%' }}>
                <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 3 }}>
                  <span style={{ fontSize: '0.65rem', color: NEW_COLOR, fontWeight: 700 }}>{pct(d.new)}</span>
                  <div style={{ width: '100%', height: `${(d.new / maxVal) * 110}px`, background: NEW_COLOR, borderRadius: '4px 4px 0 0', transition: 'height 0.5s ease', minHeight: 4 }} />
                </div>
                <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 3 }}>
                  <span style={{ fontSize: '0.65rem', color: OLD_COLOR, fontWeight: 700 }}>{pct(d.old)}</span>
                  <div style={{ width: '100%', height: `${(d.old / maxVal) * 110}px`, background: OLD_COLOR, borderRadius: '4px 4px 0 0', transition: 'height 0.5s ease', minHeight: 4 }} />
                </div>
              </div>
            ))}
          </div>
          <div style={{ display: 'flex', gap: '1rem' }}>
            {K_VALUES.map(k => (
              <div key={k} style={{ flex: 1, textAlign: 'center', fontSize: '0.72rem', color: 'var(--text-muted)' }}>k={k}</div>
            ))}
          </div>
          <div style={{ display: 'flex', gap: '1.5rem', marginTop: '1rem', justifyContent: 'center' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: '0.75rem' }}>
              <div style={{ width: 12, height: 12, borderRadius: 3, background: NEW_COLOR }} /> Yeni
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              <div style={{ width: 12, height: 12, borderRadius: 3, background: OLD_COLOR }} /> Eski
            </div>
          </div>
        </div>

        {/* All metrics at glance */}
        <div className="premium-card" style={{ padding: '1.5rem', overflowY: 'auto', maxHeight: 380 }}>
          <h3 style={{ margin: '0 0 1rem 0', fontSize: '0.95rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: 6 }}>
            <Target size={16} /> Tüm Metrikler
          </h3>
          {METRICS.map(metric =>
            K_VALUES.map(k => (
              <MetricBar
                key={`${metric}@${k}`}
                label={METRIC_TR[metric]}
                k={k}
                newVal={n[`${metric}@${k}`]}
                oldVal={o[`${metric}@${k}`]}
              />
            ))
          )}
          {/* Latency */}
          <div style={{ borderTop: '1px solid var(--border)', paddingTop: '0.75rem', marginTop: '0.5rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem', fontWeight: 600, marginBottom: '0.4rem' }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}><Clock size={12} /> Ortalama Gecikme</span>
              <span style={{ color: n.avg_latency_sec <= o.avg_latency_sec ? '#10b981' : '#f59e0b' }}>
                {n.avg_latency_sec}s (eski: {o.avg_latency_sec}s)
              </span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem', fontWeight: 600 }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}><Zap size={12} /> P95 Gecikme</span>
              <span style={{ color: 'var(--text-muted)' }}>{n.p95_latency_sec}s (eski: {o.p95_latency_sec}s)</span>
            </div>
          </div>
        </div>
      </div>

      {/* KG Benchmark */}
      <div className="premium-card" style={{ padding: '1.5rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
          <h3 style={{ margin: 0, fontSize: '0.95rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: 6 }}>
            <FlaskConical size={16} color="var(--primary)" /> KG Bilgi Grafiği Benchmark (10 Soru)
          </h3>
          <button onClick={runBenchmark} disabled={benchRunning}
            style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', padding: '0.5rem 1rem', background: 'var(--primary)', color: 'white', border: 'none', borderRadius: '8px', cursor: 'pointer', fontWeight: 600, fontSize: '0.82rem' }}>
            <RefreshCw size={13} style={{ animation: benchRunning ? 'spin 1s linear infinite' : 'none' }} />
            {benchRunning ? 'Çalışıyor...' : 'Benchmark Çalıştır'}
          </button>
        </div>

        {!benchmark && !benchRunning && (
          <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem', textAlign: 'center', padding: '1.5rem' }}>
            Butona basarak KG araçlarının 10 Türkçe SUT sorusunu bulup bulamadığını test edin.
          </div>
        )}
        {benchRunning && (
          <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem', textAlign: 'center', padding: '1.5rem' }}>
            <RefreshCw size={20} style={{ animation: 'spin 1s linear infinite', display: 'block', margin: '0 auto 0.5rem' }} />
            KG sorguları çalıştırılıyor...
          </div>
        )}
        {benchmark && !benchmark.error && (
          <>
            <div style={{ display: 'flex', gap: '1.5rem', marginBottom: '1rem', padding: '0.75rem 1rem', background: '#f8fafc', borderRadius: '10px', alignItems: 'center' }}>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: '1.8rem', fontWeight: 800, color: benchmark.hit_rate >= 0.7 ? '#10b981' : benchmark.hit_rate >= 0.5 ? '#f59e0b' : '#ef4444' }}>
                  {(benchmark.hit_rate * 100).toFixed(0)}%
                </div>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Hit Rate</div>
              </div>
              <div style={{ height: 40, width: 1, background: 'var(--border)' }} />
              <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                {benchmark.hits}/{benchmark.total} soruda doğru tür düğüm bulundu
              </div>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              {benchmark.results.map((r, i) => (
                <div key={i} style={{ display: 'flex', gap: '0.75rem', padding: '0.6rem 0.75rem', background: r.hit ? '#f0fdf4' : '#fef2f2', borderRadius: '8px', borderLeft: `3px solid ${r.hit ? '#10b981' : '#ef4444'}`, alignItems: 'flex-start' }}>
                  {r.hit
                    ? <CheckCircle size={15} color="#10b981" style={{ flexShrink: 0, marginTop: 2 }} />
                    : <XCircle size={15} color="#ef4444" style={{ flexShrink: 0, marginTop: 2 }} />
                  }
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: '0.82rem', fontWeight: 600, marginBottom: '0.2rem' }}>{r.question}</div>
                    <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                      Beklenen: {r.expected_types?.join(', ')} |{' '}
                      Bulunan: {r.found_types?.join(', ') || '-'} |{' '}
                      {r.found_nodes?.slice(0, 2).join(', ')}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </>
        )}
        {benchmark?.error && (
          <div style={{ color: '#ef4444', fontSize: '0.85rem' }}>Hata: {benchmark.error}</div>
        )}
      </div>

      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  )
}
