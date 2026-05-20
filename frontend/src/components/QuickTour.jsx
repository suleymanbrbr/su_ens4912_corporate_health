import React, { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import FocusTrap from 'focus-trap-react'
import { Sparkles, Key, MessageSquare, ChevronLeft, ChevronRight, X } from 'lucide-react'

const STORAGE_KEY = 'mahiks_tour_seen'

/**
 * QuickTour — first-login onboarding wizard.
 *
 * Three slides:
 *   1) Welcome + what MAHIKS-TR does
 *   2) Add your LLM API key (link to /settings)
 *   3) Sample questions to get started
 *
 * Persistence: `localStorage[STORAGE_KEY] = '1'` after dismissal.
 *
 * Devtools reset: paste this in the console to re-trigger on next reload —
 *   localStorage.removeItem('mahiks_tour_seen'); location.reload();
 *
 * Co-existence with WelcomeModal: WelcomeModal lives inside ChatDashboard
 * with key `sut_onboarding_done`. To avoid two modals stacking on the very
 * first visit, QuickTour also dismisses itself if `sut_onboarding_done` is
 * already set (indicates the legacy onboarding already ran).
 */
export default function QuickTour({ user }) {
  const navigate = useNavigate()
  const [open, setOpen] = useState(false)
  const [step, setStep] = useState(0)

  useEffect(() => {
    if (!user) return
    try {
      const seen = localStorage.getItem(STORAGE_KEY)
      if (seen) return
      // If the legacy WelcomeModal flag is already set, treat this user as
      // already onboarded — silently mark the tour as seen too.
      const legacy = localStorage.getItem('sut_onboarding_done')
      if (legacy) {
        localStorage.setItem(STORAGE_KEY, '1')
        return
      }
      setOpen(true)
    } catch (_) { /* SSR / private mode — skip */ }
  }, [user])

  // ESC closes (FocusTrap intercepts ESC too, but we mark the tour seen
  // so it doesn't reopen).
  useEffect(() => {
    if (!open) return
    const onKey = (e) => { if (e.key === 'Escape') dismiss() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open])

  const dismiss = () => {
    try { localStorage.setItem(STORAGE_KEY, '1') } catch (_) {}
    setOpen(false)
  }

  if (!open) return null

  const TOTAL = 3
  const goNext = () => setStep(s => Math.min(s + 1, TOTAL - 1))
  const goBack = () => setStep(s => Math.max(s - 1, 0))

  const goToSettings = () => {
    dismiss()
    navigate('/settings#api-keys')
  }

  const fillExample = (q) => {
    try { sessionStorage.setItem('prefillChat', q) } catch (_) {}
    dismiss()
    navigate('/')
  }

  const Dots = () => (
    <div style={{ display: 'flex', gap: '0.4rem', justifyContent: 'center', marginBottom: '1rem' }}>
      {Array.from({ length: TOTAL }).map((_, i) => (
        <span
          key={i}
          aria-hidden="true"
          style={{
            width: i === step ? 22 : 8,
            height: 8,
            borderRadius: 999,
            background: i === step ? 'var(--accent)' : 'var(--border)',
            transition: 'all 0.25s ease',
          }}
        />
      ))}
    </div>
  )

  return (
    <FocusTrap focusTrapOptions={{ initialFocus: false, escapeDeactivates: false }}>
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="quicktour-title"
        data-testid="quick-tour"
        style={{
          position: 'fixed',
          inset: 0,
          zIndex: 4500,
          background: 'rgba(15,23,42,0.55)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '1rem',
        }}
        onClick={dismiss}
      >
        <div
          className="premium-card"
          style={{ maxWidth: '480px', width: '100%', padding: '1.75rem', position: 'relative' }}
          onClick={e => e.stopPropagation()}
        >
          <button
            type="button"
            aria-label="Turu kapat"
            onClick={dismiss}
            style={{
              position: 'absolute', top: '0.5rem', right: '0.5rem',
              background: 'transparent', border: 'none', cursor: 'pointer',
              color: 'var(--text-muted)', padding: '0.4rem', borderRadius: 6,
            }}
          >
            <X size={16} aria-hidden />
          </button>

          <Dots />

          {step === 0 && (
            <>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '0.75rem' }}>
                <Sparkles size={22} color="var(--accent)" />
                <h2 id="quicktour-title" style={{ margin: 0, fontSize: '1.25rem' }}>
                  MAHIKS-TR'ye Hoş Geldiniz
                </h2>
              </div>
              <p style={{ color: 'var(--text-main)', fontSize: '0.92rem', lineHeight: 1.7, marginBottom: '1.25rem' }}>
                MAHIKS-TR, resmi Sağlık Uygulama Tebliği (SUT) metinleri üzerinden sorularınıza
                kaynaklı yanıtlar üreten çok kullanıcılı bir asistandır. Her yanıt ilgili madde ve
                kaynak referanslarıyla birlikte sunulur.
              </p>
            </>
          )}

          {step === 1 && (
            <>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '0.75rem' }}>
                <Key size={22} color="var(--accent)" />
                <h2 id="quicktour-title" style={{ margin: 0, fontSize: '1.25rem' }}>
                  Önce API Anahtarınızı Ekleyin
                </h2>
              </div>
              <p style={{ color: 'var(--text-main)', fontSize: '0.92rem', lineHeight: 1.7, marginBottom: '0.85rem' }}>
                Sohbeti kullanabilmek için kendi LLM sağlayıcınızın (Gemini, OpenRouter, LM Studio)
                API anahtarını eklemelisiniz. Anahtarınız şifrelenerek saklanır.
              </p>
              <div style={{
                padding: '0.85rem 1rem',
                background: 'rgba(59,130,246,0.06)',
                border: '1px solid rgba(59,130,246,0.2)',
                borderRadius: '10px',
                fontSize: '0.85rem',
                color: 'var(--text-main)',
                marginBottom: '1.25rem',
              }}>
                Ayarlar &rarr; <strong>API Anahtarları</strong>
              </div>
              <button
                type="button"
                className="btn-primary"
                onClick={goToSettings}
                style={{ width: '100%', padding: '0.65rem', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.4rem', marginBottom: '0.5rem' }}
              >
                <Key size={16} /> Ayarlara Git
              </button>
            </>
          )}

          {step === 2 && (
            <>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '0.75rem' }}>
                <MessageSquare size={22} color="var(--accent)" />
                <h2 id="quicktour-title" style={{ margin: 0, fontSize: '1.25rem' }}>
                  Soru Sormaya Hazırsınız
                </h2>
              </div>
              <p style={{ color: 'var(--text-main)', fontSize: '0.92rem', lineHeight: 1.7, marginBottom: '1rem' }}>
                Aşağıdaki örnek soruları deneyebilir veya kendi sorunuzu yazabilirsiniz:
              </p>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', marginBottom: '1.25rem' }}>
                {['SGK diş implantı?', 'MS biyolojik ilaç?'].map(q => (
                  <button
                    key={q}
                    type="button"
                    className="btn-secondary"
                    onClick={() => fillExample(q)}
                    style={{ textAlign: 'left', padding: '0.6rem 0.85rem', fontSize: '0.88rem' }}
                  >
                    {q}
                  </button>
                ))}
              </div>
            </>
          )}

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '0.5rem' }}>
            <button
              type="button"
              onClick={goBack}
              disabled={step === 0}
              style={{
                background: 'transparent',
                border: 'none',
                color: step === 0 ? 'var(--border)' : 'var(--text-muted)',
                padding: '0.5rem 0.6rem',
                borderRadius: '8px',
                fontWeight: 600,
                cursor: step === 0 ? 'not-allowed' : 'pointer',
                display: 'flex', alignItems: 'center', gap: '0.25rem',
              }}
            >
              <ChevronLeft size={16} /> Geri
            </button>
            {step < TOTAL - 1 ? (
              <button
                type="button"
                onClick={goNext}
                className="btn-primary"
                style={{ padding: '0.55rem 1.2rem', display: 'flex', alignItems: 'center', gap: '0.35rem' }}
              >
                İleri <ChevronRight size={16} />
              </button>
            ) : (
              <button
                type="button"
                onClick={dismiss}
                className="btn-primary"
                style={{ padding: '0.55rem 1.5rem' }}
              >
                Başla
              </button>
            )}
          </div>
        </div>
      </div>
    </FocusTrap>
  )
}
