import React from 'react'
import FocusTrap from 'focus-trap-react'

/**
 * One-time onboarding after login (localStorage `sut_onboarding_done`).
 */
export default function WelcomeModal({ open, username, onDismiss }) {
  if (!open) return null

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
          style={{ maxWidth: '440px', width: '100%', padding: '1.5rem' }}
          onClick={e => e.stopPropagation()}
        >
          <h2 id="welcome-title" style={{ marginTop: 0, marginBottom: '0.5rem', fontSize: '1.25rem' }}>
            Hoş geldiniz{username ? `, ${username}` : ''}
          </h2>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', lineHeight: 1.6, marginBottom: '1rem' }}>
            SUT Asistanı resmi Sağlık Uygulama Tebliği metinleri üzerinden sorularınıza yanıt üretir; her yanıtta kaynak
            maddelere gidebilirsiniz.
          </p>
          <ul style={{ margin: '0 0 1rem', paddingLeft: '1.25rem', color: 'var(--text-main)', fontSize: '0.88rem', lineHeight: 1.7 }}>
            <li>Personanızı seçerek hızlı soruları ve tonu ayarlayın.</li>
            <li>PDF rapor yükleyebilir, sohbeti dışa aktarabilirsiniz.</li>
            <li>Yapay zeka hata yapabilir; kritik işlemlerde metni doğrulayın.</li>
          </ul>
          <button type="button" className="btn-primary" style={{ width: '100%', padding: '0.65rem' }} onClick={onDismiss}>
            Başlayalım
          </button>
        </div>
      </div>
    </FocusTrap>
  )
}
