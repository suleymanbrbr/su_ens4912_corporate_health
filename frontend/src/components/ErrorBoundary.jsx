import React from 'react'

/**
 * Top-level ErrorBoundary.
 *
 * Wraps the route tree so a crash in any lazy/eager component renders a
 * friendly Turkish recovery card instead of a blank screen. In DEV we
 * also surface the stack trace and pipe to console.error to make the
 * regression debuggable.
 *
 * Note: this is a classic class component because hooks cannot catch
 * render-phase errors. Avoid adding new dependencies — react-error-boundary
 * is intentionally not used here to keep the bundle lean.
 */
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props)
    this.state = { hasError: false, error: null, info: null }
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error }
  }

  componentDidCatch(error, info) {
    this.setState({ info })
    if (import.meta.env.DEV) {
      // Surface in dev only; production stays quiet so we don't leak
      // implementation details into the user's devtools.
      // eslint-disable-next-line no-console
      console.error('[ErrorBoundary]', error, info)
    }
  }

  handleReload = () => {
    try { window.location.reload() } catch (_) { /* noop */ }
  }

  render() {
    if (!this.state.hasError) return this.props.children

    const isDev = import.meta.env.DEV
    const err = this.state.error
    const stack = err && (err.stack || String(err))

    return (
      <div
        role="alert"
        data-testid="error-boundary"
        style={{
          minHeight: '100vh',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '1.5rem',
          background: 'var(--bg)',
          color: 'var(--text-main)',
        }}
      >
        <div
          className="premium-card"
          style={{
            maxWidth: '520px',
            width: '100%',
            padding: '2rem',
            textAlign: 'center',
          }}
        >
          <div
            aria-hidden
            style={{
              width: '56px',
              height: '56px',
              borderRadius: '16px',
              background: '#fef2f2',
              color: '#ef4444',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              margin: '0 auto 1rem',
              fontSize: '1.8rem',
              fontWeight: 700,
            }}
          >
            !
          </div>
          <h1 style={{ fontSize: '1.25rem', fontWeight: 700, marginBottom: '0.5rem' }}>
            Beklenmedik bir hata oluştu
          </h1>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.92rem', marginBottom: '1.25rem', lineHeight: 1.6 }}>
            Uygulamanın bu bölümü yüklenemedi. Sayfayı yenileyerek tekrar deneyebilirsiniz.
            Sorun devam ederse yönetici ile iletişime geçin.
          </p>
          <button
            type="button"
            className="btn-primary"
            onClick={this.handleReload}
            style={{ padding: '0.65rem 1.5rem' }}
          >
            Sayfayı yenile
          </button>

          {isDev && stack && (
            <details style={{ marginTop: '1.25rem', textAlign: 'left' }}>
              <summary style={{ cursor: 'pointer', fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 600 }}>
                Geliştirici ayrıntıları (DEV)
              </summary>
              <pre
                style={{
                  marginTop: '0.5rem',
                  padding: '0.75rem',
                  background: 'var(--bg-elev)',
                  borderRadius: '8px',
                  fontSize: '0.72rem',
                  color: '#b91c1c',
                  maxHeight: '240px',
                  overflow: 'auto',
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-word',
                }}
              >
                {stack}
              </pre>
            </details>
          )}
        </div>
      </div>
    )
  }
}

export default ErrorBoundary
