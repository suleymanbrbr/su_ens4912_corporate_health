import React, { useState } from 'react'
import FocusTrap from 'focus-trap-react'
import { useNavigate } from 'react-router-dom'
import { Sparkles, Users, Key, ChevronRight, ChevronLeft } from 'lucide-react'

/**
 * Three-step onboarding wizard shown once after login.
 * Completion tracked via localStorage `sut_onboarding_done` (set by parent).
 *
 * Steps:
 *   1. Welcome / project intro
 *   2. Persona selection hint
 *   3. API key setup prompt (CTA to /settings)
 */
export default function WelcomeModal({ open, username, onDismiss }) {
  const [step, setStep] = useState(0)
  const navigate = useNavigate()

  if (!open) return null

  const totalSteps = 3

  const goNext = () => setStep(s => Math.min(s + 1, totalSteps - 1))
  const goBack = () => setStep(s => Math.max(s - 1, 0))

  const handleGoToSettings = () => {
    onDismiss()
    // Anchor to the API keys card; falls back gracefully if no anchor support
    navigate('/settings#api-keys')
  }

  const handleSkip = () => {
    onDismiss()
  }

  const StepDots = () => (
    <div style={{ display: 'flex', gap: '0.4rem', justifyContent: 'center', marginBottom: '1rem' }}>
      {Array.from({ length: totalSteps }).map((_, i) => (
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
    <FocusTrap focusTrapOptions={{ initialFocus: false }}>
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="welcome-title"
        style={{
          position: 'fixed',
          inset: 0,
          zIndex: 4000,
          background: 'rgba(15,23,42,0.5)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '1rem',
        }}
        onClick={onDismiss}
      >
        <div
          className="premium-card"
          style={{ maxWidth: '460px', width: '100%', padding: '1.75rem' }}
          onClick={e => e.stopPropagation()}
        >
          <StepDots />

          {step === 0 && (
            <>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '0.6rem' }}>
                <Sparkles size={22} color="var(--accent)" />
                <h2 id="welcome-title" style={{ margin: 0, fontSize: '1.25rem' }}>
                  Hoş geldiniz{username ? `, ${username}` : ''}
                </h2>
              </div>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', lineHeight: 1.6, marginBottom: '1rem' }}>
                SUT Asistanı resmi Sağlık Uygulama Tebliği metinleri üzerinden sorularınıza yanıt üretir;
                her yanıtta kaynak maddelere gidebilirsiniz.
              </p>
              <ul style={{ margin: '0 0 1.25rem', paddingLeft: '1.25rem', color: 'var(--text-main)', fontSize: '0.88rem', lineHeight: 1.7 }}>
                <li>PDF rapor yükleyebilir, sohbeti dışa aktarabilirsiniz.</li>
                <li>Bilgi grafiği ile maddeler arası ilişkileri keşfedin.</li>
                <li>Yapay zeka hata yapabilir; kritik işlemlerde metni doğrulayın.</li>
              </ul>
            </>
          )}

          {step === 1 && (
            <>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '0.6rem' }}>
                <Users size={22} color="var(--accent)" />
                <h2 id="welcome-title" style={{ margin: 0, fontSize: '1.25rem' }}>
                  Personanızı seçin
                </h2>
              </div>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', lineHeight: 1.6, marginBottom: '1rem' }}>
                Sohbet ekranının üst kısmından kullanıcı tipinizi seçerek hızlı soruları ve yanıt
                tonunu kendinize uyarlayın.
              </p>
              <ul style={{ margin: '0 0 1.25rem', paddingLeft: '1.25rem', color: 'var(--text-main)', fontSize: '0.88rem', lineHeight: 1.7 }}>
                <li><strong>Hasta</strong> — günlük dilde, anlaşılır açıklamalar.</li>
                <li><strong>Hekim</strong> — klinik kriterler ve raporlama şartları.</li>
                <li><strong>Eczacı</strong> — reçete, muadil ve katılım payı odaklı.</li>
              </ul>
            </>
          )}

          {step === 2 && (
            <>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '0.6rem' }}>
                <Key size={22} color="var(--accent)" />
                <h2 id="welcome-title" style={{ margin: 0, fontSize: '1.25rem' }}>
                  LLM API anahtarınızı ekleyin
                </h2>
              </div>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', lineHeight: 1.6, marginBottom: '0.85rem' }}>
                Sohbet özelliğini kullanabilmek için kendi LLM sağlayıcınızın (Google Gemini,
                OpenRouter veya yerel LM Studio) API anahtarını eklemeniz gerekiyor. Anahtarınız
                şifrelenerek saklanır ve yalnızca sizin oturumunuzda kullanılır.
              </p>
              <div style={{
                padding: '0.85rem 1rem',
                background: 'rgba(59,130,246,0.06)',
                border: '1px solid rgba(59,130,246,0.2)',
                borderRadius: '10px',
                fontSize: '0.82rem',
                color: 'var(--text-main)',
                marginBottom: '1.25rem',
              }}>
                Ayarlar &rarr; <strong>LLM API Anahtarları</strong> bölümünden anahtarınızı ekleyebilirsiniz.
              </div>
              <div style={{ display: 'flex', gap: '0.65rem' }}>
                <button
                  type="button"
                  className="btn-primary"
                  onClick={handleGoToSettings}
                  style={{ flex: 1, padding: '0.65rem', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.4rem' }}
                >
                  <Key size={16} /> Anahtarı Ayarla
                </button>
                <button
                  type="button"
                  onClick={handleSkip}
                  style={{
                    flex: 1, padding: '0.65rem',
                    background: 'transparent', color: 'var(--text-muted)',
                    border: '1px solid var(--border)', borderRadius: '8px',
                    fontWeight: 600, cursor: 'pointer',
                  }}
                >
                  Şimdilik Atla
                </button>
              </div>
            </>
          )}

          {/* Navigation row (steps 0 & 1 only — step 2 has its own CTAs) */}
          {step < 2 && (
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
              <button
                type="button"
                onClick={goNext}
                className="btn-primary"
                style={{ padding: '0.55rem 1.2rem', display: 'flex', alignItems: 'center', gap: '0.35rem' }}
              >
                İleri <ChevronRight size={16} />
              </button>
            </div>
          )}
        </div>
      </div>
    </FocusTrap>
  )
}
