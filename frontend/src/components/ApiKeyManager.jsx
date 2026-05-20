import React, { useState, useEffect, useCallback } from 'react'
import toast from 'react-hot-toast'
import { Key, Eye, EyeOff, Trash2, Check, X, ExternalLink } from 'lucide-react'
import { apiGet, apiPost, apiDelete } from '../services/api'

const PROVIDERS = [
  { value: 'gemini',     label: 'Google Gemini',  helpUrl: 'https://aistudio.google.com/apikey',  needsBaseUrl: false },
  { value: 'openrouter', label: 'OpenRouter',     helpUrl: 'https://openrouter.ai/keys',          needsBaseUrl: false },
  { value: 'local',      label: 'Yerel (LM Studio vb.)', helpUrl: null,                            needsBaseUrl: true  },
]

const PROVIDER_LABEL = Object.fromEntries(PROVIDERS.map(p => [p.value, p.label]))
const DEFAULT_BASE_URL = 'http://localhost:1234/v1'

export default function ApiKeyManager() {
  const [keys, setKeys] = useState([])
  const [loadingList, setLoadingList] = useState(true)

  const [provider, setProvider] = useState('gemini')
  const [apiKey, setApiKey] = useState('')
  const [baseUrl, setBaseUrl] = useState(DEFAULT_BASE_URL)
  const [showKey, setShowKey] = useState(false)
  const [validationErr, setValidationErr] = useState('')

  const [testing, setTesting] = useState(false)
  const [testResult, setTestResult] = useState(null) // { ok: bool, msg: string }
  const [saving, setSaving] = useState(false)

  const refreshKeys = useCallback(async () => {
    setLoadingList(true)
    try {
      const data = await apiGet('/api/user/api-keys')
      setKeys(Array.isArray(data) ? data : [])
    } catch (err) {
      // Empty list on first load is OK; only show toast on real failures.
      if (!String(err.message).includes('404')) {
        toast.error('Anahtarlar yüklenemedi: ' + err.message)
      }
      setKeys([])
    } finally {
      setLoadingList(false)
    }
  }, [])

  useEffect(() => {
    refreshKeys()
  }, [refreshKeys])

  // Reset transient state when provider changes
  useEffect(() => {
    setTestResult(null)
    setValidationErr('')
    if (provider !== 'local') {
      // keep baseUrl in state but it won't be sent for non-local
    }
  }, [provider])

  const validate = () => {
    if (!apiKey.trim()) {
      setValidationErr('API anahtarı boş olamaz.')
      return false
    }
    if (apiKey.trim().length < 20) {
      setValidationErr('API anahtarı en az 20 karakter olmalı.')
      return false
    }
    if (provider === 'local' && !baseUrl.trim()) {
      setValidationErr('Yerel sağlayıcı için Base URL gerekli.')
      return false
    }
    setValidationErr('')
    return true
  }

  const buildPayload = () => {
    const payload = { provider, api_key: apiKey.trim() }
    if (provider === 'local') payload.base_url = baseUrl.trim()
    return payload
  }

  const handleTest = async () => {
    if (!validate()) return
    setTesting(true)
    setTestResult(null)
    try {
      const res = await apiPost('/api/user/api-keys/test', buildPayload())
      if (res && res.valid) {
        setTestResult({ ok: true, msg: 'Bağlantı başarılı.' })
        toast.success('Bağlantı doğrulandı.')
      } else {
        setTestResult({ ok: false, msg: (res && res.error) || 'Anahtar geçersiz.' })
        toast.error('Doğrulama başarısız.')
      }
    } catch (err) {
      setTestResult({ ok: false, msg: err.message })
      toast.error('Test edilemedi: ' + err.message)
    } finally {
      setTesting(false)
    }
  }

  const handleSave = async () => {
    if (!validate()) return
    setSaving(true)
    try {
      await apiPost('/api/user/api-keys', buildPayload())
      toast.success(`${PROVIDER_LABEL[provider]} anahtarı kaydedildi.`)
      setApiKey('')
      setShowKey(false)
      setTestResult(null)
      refreshKeys()
    } catch (err) {
      toast.error('Kaydedilemedi: ' + err.message)
    } finally {
      setSaving(false)
    }
  }

  const handleDelete = async (prov) => {
    if (!window.confirm(`${PROVIDER_LABEL[prov] || prov} anahtarını silmek istediğinize emin misiniz?`)) return
    try {
      await apiDelete(`/api/user/api-keys/${prov}`)
      toast.success('Anahtar silindi.')
      refreshKeys()
    } catch (err) {
      toast.error('Silinemedi: ' + err.message)
    }
  }

  const selectedProvider = PROVIDERS.find(p => p.value === provider) || PROVIDERS[0]

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {/* Saved keys list */}
      <div>
        <h4 style={{ fontSize: '0.95rem', fontWeight: 600, marginBottom: '0.75rem', color: 'var(--text-main)' }}>
          Kayıtlı Anahtarlar
        </h4>
        {loadingList ? (
          <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>Yükleniyor...</p>
        ) : keys.length === 0 ? (
          <div style={{
            padding: '1.5rem',
            background: 'rgba(59,130,246,0.06)',
            border: '1px dashed var(--border)',
            borderRadius: '10px',
            color: 'var(--text-muted)',
            fontSize: '0.9rem',
            textAlign: 'center',
          }}>
            Henüz anahtar yok. Sohbete başlamak için bir tane ekleyin.
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.65rem' }}>
            {keys.map(k => (
              <div
                key={k.id || k.provider}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  padding: '0.85rem 1rem',
                  background: 'var(--card-bg, #fff)',
                  border: '1px solid var(--border)',
                  borderRadius: '10px',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.85rem', minWidth: 0 }}>
                  <div style={{
                    width: 36, height: 36, borderRadius: '50%',
                    background: 'rgba(59,130,246,0.1)', color: 'var(--accent)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
                  }}>
                    <Key size={18} />
                  </div>
                  <div style={{ minWidth: 0 }}>
                    <div style={{ fontWeight: 600, fontSize: '0.95rem' }}>
                      {PROVIDER_LABEL[k.provider] || k.provider}
                    </div>
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontFamily: 'monospace' }}>
                      {k.key_hint ? `••••••••${k.key_hint}` : '••••••••'}
                      {k.is_active === false && (
                        <span style={{ marginLeft: '0.5rem', color: '#ef4444' }}>(pasif)</span>
                      )}
                    </div>
                  </div>
                </div>
                <button
                  type="button"
                  onClick={() => handleDelete(k.provider)}
                  title="Sil"
                  style={{
                    background: 'transparent',
                    color: '#ef4444',
                    border: '1px solid transparent',
                    padding: '0.4rem 0.6rem',
                    borderRadius: '8px',
                    cursor: 'pointer',
                    display: 'flex', alignItems: 'center', gap: '0.35rem',
                  }}
                  onMouseOver={e => { e.currentTarget.style.background = 'rgba(239,68,68,0.08)'; e.currentTarget.style.borderColor = 'rgba(239,68,68,0.25)' }}
                  onMouseOut={e => { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.borderColor = 'transparent' }}
                >
                  <Trash2 size={16} />
                  <span style={{ fontSize: '0.85rem', fontWeight: 600 }}>Sil</span>
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Add / update form */}
      <div style={{
        padding: '1.25rem',
        border: '1px solid var(--border)',
        borderRadius: '12px',
        background: 'var(--card-bg, #fff)',
      }}>
        <h4 style={{ fontSize: '0.95rem', fontWeight: 600, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Key size={16} color="var(--accent)" /> Anahtar Ekle / Güncelle
        </h4>

        <div style={{ display: 'grid', gap: '0.9rem' }}>
          <div>
            <label style={{ display: 'block', fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '0.4rem' }}>
              Sağlayıcı
            </label>
            <select
              value={provider}
              onChange={e => setProvider(e.target.value)}
              style={{
                width: '100%', padding: '0.7rem 0.85rem',
                borderRadius: '8px', border: '1px solid var(--border)',
                background: 'var(--card-bg, #fff)', color: 'var(--text-main)',
                fontSize: '0.95rem', fontFamily: 'inherit',
              }}
            >
              {PROVIDERS.map(p => (
                <option key={p.value} value={p.value}>{p.label}</option>
              ))}
            </select>
          </div>

          <div>
            <label style={{ display: 'block', fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '0.4rem' }}>
              API Anahtarı
            </label>
            <div style={{ position: 'relative' }}>
              <input
                type={showKey ? 'text' : 'password'}
                value={apiKey}
                onChange={e => { setApiKey(e.target.value); setTestResult(null); setValidationErr('') }}
                placeholder={provider === 'gemini' ? 'AIza...' : provider === 'openrouter' ? 'sk-or-v1-...' : 'lm-studio'}
                style={{ paddingRight: '2.75rem' }}
                autoComplete="off"
                spellCheck={false}
              />
              <button
                type="button"
                onClick={() => setShowKey(s => !s)}
                aria-label={showKey ? 'Anahtarı gizle' : 'Anahtarı göster'}
                style={{
                  position: 'absolute', right: '0.5rem', top: '50%', transform: 'translateY(-50%)',
                  background: 'transparent', border: 'none', padding: '0.4rem',
                  color: 'var(--text-muted)', cursor: 'pointer', display: 'flex', alignItems: 'center',
                }}
              >
                {showKey ? <EyeOff size={18} /> : <Eye size={18} />}
              </button>
            </div>
          </div>

          {selectedProvider.needsBaseUrl && (
            <div>
              <label style={{ display: 'block', fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '0.4rem' }}>
                Base URL
              </label>
              <input
                type="text"
                value={baseUrl}
                onChange={e => { setBaseUrl(e.target.value); setTestResult(null) }}
                placeholder={DEFAULT_BASE_URL}
                spellCheck={false}
              />
              <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.35rem' }}>
                Varsayılan LM Studio adresi: <code>{DEFAULT_BASE_URL}</code>
              </p>
            </div>
          )}

          {validationErr && (
            <div style={{ fontSize: '0.82rem', color: '#ef4444' }}>{validationErr}</div>
          )}

          {testResult && (
            <div style={{
              display: 'flex', alignItems: 'center', gap: '0.5rem',
              padding: '0.6rem 0.85rem', borderRadius: '8px',
              background: testResult.ok ? 'rgba(34,197,94,0.1)' : 'rgba(239,68,68,0.1)',
              color: testResult.ok ? '#15803d' : '#b91c1c',
              fontSize: '0.85rem', fontWeight: 600,
            }}>
              {testResult.ok ? <Check size={16} /> : <X size={16} />}
              <span>{testResult.msg}</span>
            </div>
          )}

          <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
            <button
              type="button"
              onClick={handleTest}
              disabled={testing || saving}
              className="btn-secondary"
              style={{
                background: 'transparent', color: 'var(--accent)',
                border: '1px solid var(--accent)', padding: '0.65rem 1.2rem',
                borderRadius: '8px', fontWeight: 600, cursor: testing ? 'not-allowed' : 'pointer',
                opacity: testing ? 0.6 : 1, display: 'flex', alignItems: 'center', gap: '0.4rem',
              }}
            >
              <Check size={16} /> {testing ? 'Test ediliyor...' : 'Bağlantıyı Test Et'}
            </button>
            <button
              type="button"
              onClick={handleSave}
              disabled={saving || testing}
              className="btn-primary"
              style={{ padding: '0.65rem 1.4rem', opacity: saving ? 0.6 : 1 }}
            >
              {saving ? 'Kaydediliyor...' : 'Kaydet'}
            </button>
          </div>
        </div>
      </div>

      {/* Info card */}
      <div style={{
        padding: '1.1rem 1.25rem',
        background: 'rgba(59,130,246,0.06)',
        border: '1px solid rgba(59,130,246,0.2)',
        borderRadius: '12px',
      }}>
        <h4 style={{ fontSize: '0.9rem', fontWeight: 700, marginBottom: '0.6rem', color: 'var(--text-main)' }}>
          Anahtarı nereden alırım?
        </h4>
        <ul style={{ margin: 0, paddingLeft: 0, listStyle: 'none', display: 'flex', flexDirection: 'column', gap: '0.4rem', fontSize: '0.85rem' }}>
          <li style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <span style={{ fontWeight: 600, minWidth: 110 }}>Gemini:</span>
            <a href="https://aistudio.google.com/apikey" target="_blank" rel="noreferrer"
              style={{ color: 'var(--accent)', textDecoration: 'none', display: 'inline-flex', alignItems: 'center', gap: '0.25rem' }}>
              aistudio.google.com/apikey <ExternalLink size={13} />
            </a>
          </li>
          <li style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <span style={{ fontWeight: 600, minWidth: 110 }}>OpenRouter:</span>
            <a href="https://openrouter.ai/keys" target="_blank" rel="noreferrer"
              style={{ color: 'var(--accent)', textDecoration: 'none', display: 'inline-flex', alignItems: 'center', gap: '0.25rem' }}>
              openrouter.ai/keys <ExternalLink size={13} />
            </a>
          </li>
          <li style={{ display: 'flex', alignItems: 'flex-start', gap: '0.5rem' }}>
            <span style={{ fontWeight: 600, minWidth: 110 }}>Yerel (LM Studio):</span>
            <span style={{ color: 'var(--text-muted)' }}>
              LM Studio'yu çalıştırın, "Local Server" sekmesinden sunucuyu başlatın
              (varsayılan: <code>{DEFAULT_BASE_URL}</code>).
            </span>
          </li>
        </ul>
      </div>
    </div>
  )
}
