import React, { useState, useEffect } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { ArrowLeft, Settings as SettingsIcon, Moon, Sun, Monitor, Cpu, Sliders } from 'lucide-react'

function Settings({ user, onLogout }) {
  const [theme, setTheme] = useState('light')
  const [model, setModel] = useState('gemini')
  const [depth, setDepth] = useState(5)
  const navigate = useNavigate()

  useEffect(() => {
    setTheme(localStorage.getItem('theme') || 'light')
    setModel(localStorage.getItem('model') || 'gemini')
    setDepth(parseInt(localStorage.getItem('k_depth') || '5'))
  }, [])

  const handleSave = () => {
    localStorage.setItem('theme', theme)
    localStorage.setItem('model', model)
    localStorage.setItem('k_depth', depth.toString())
    
    // Quick hack to apply dark mode if we implemented it globally
    if(theme === 'dark') document.body.classList.add('dark-mode')
    else document.body.classList.remove('dark-mode')

    alert('Ayarlar başarıyla kaydedildi.')
  }

  return (
    <div style={{ display: 'flex', height: '100vh', background: 'var(--bg)' }}>
      <aside className="sidebar glass" style={{ width: '280px', borderRight: '1px solid var(--border)', padding: '2rem', display: 'flex', flexDirection: 'column' }}>
        <h2 className="text-gradient" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '2rem' }}>
          <SettingsIcon size={24} /> Ayarlar
        </h2>
        
        <nav style={{ flex: 1 }}>
          <Link to="/" style={{ textDecoration: 'none', color: 'var(--text-main)', display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.75rem', borderRadius: '8px' }} onMouseOver={e => e.currentTarget.style.background = '#f1f5f9'} onMouseOut={e => e.currentTarget.style.background = 'transparent'}>
            <ArrowLeft size={18} /> Sohbet Ekranı
          </Link>
        </nav>
      </aside>

      <main style={{ flex: 1, padding: '3rem', overflowY: 'auto' }}>
        <header style={{ marginBottom: '2rem' }}>
          <h1 style={{ fontSize: '2rem', fontWeight: 700 }}>Uygulama Ayarları</h1>
          <p style={{ color: 'var(--text-muted)' }}>Sistem davranışlarını ve arayüzü özelleştirin.</p>
        </header>

        <div style={{ display: 'grid', gap: '2rem', maxWidth: '800px' }}>
          
          <div className="premium-card" style={{ padding: '2rem' }}>
            <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.5rem', fontSize: '1.2rem' }}>
              <Monitor size={20} color="var(--primary)" /> Tema Tercihi
            </h3>
            <div style={{ display: 'flex', gap: '1rem' }}>
              <button onClick={() => setTheme('light')} className={`btn-${theme === 'light' ? 'primary' : 'secondary'}`} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flex: 1, justifyContent: 'center' }}>
                <Sun size={18} /> Aydınlık
              </button>
              <button onClick={() => setTheme('dark')} className={`btn-${theme === 'dark' ? 'primary' : 'secondary'}`} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flex: 1, justifyContent: 'center', opacity: 0.5 }}>
                <Moon size={18} /> Karanlık (Yakında)
              </button>
            </div>
          </div>

          <div className="premium-card" style={{ padding: '2rem' }}>
            <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.5rem', fontSize: '1.2rem' }}>
              <Cpu size={20} color="var(--accent)" /> Yapay Zeka Modeli
            </h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: '1rem', padding: '1rem', border: '1px solid var(--border)', borderRadius: '12px', cursor: 'pointer', background: model === 'gemini' ? '#f0fdf4' : 'white', borderColor: model === 'gemini' ? '#22c55e' : 'var(--border)' }}>
                <input type="radio" name="model" value="gemini" checked={model === 'gemini'} onChange={() => setModel('gemini')} />
                <div>
                  <div style={{ fontWeight: 600 }}>Google Gemini 1.5 Flash</div>
                  <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Varsayılan, hızlı ve güçlü akıl yürütme.</div>
                </div>
              </label>
              
              <label style={{ display: 'flex', alignItems: 'center', gap: '1rem', padding: '1rem', border: '1px solid var(--border)', borderRadius: '12px', cursor: 'not-allowed', background: '#f8fafc', opacity: 0.6 }}>
                <input type="radio" name="model" value="local" disabled />
                <div>
                  <div style={{ fontWeight: 600 }}>Yerel LLM (LMStudio)</div>
                  <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Kurumsal gizlilik için %100 yerel çalışma. (Kurulum Gerekir)</div>
                </div>
              </label>
            </div>
          </div>

          <div className="premium-card" style={{ padding: '2rem' }}>
            <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.5rem', fontSize: '1.2rem' }}>
              <Sliders size={20} color="#f59e0b" /> RAG Arama Derinliği (Chunks: {depth})
            </h3>
            <p style={{ fontSize: '0.9rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>
              Daha yüksek derinlik yapay zekaya daha fazla bağlam sağlar ancak yanıt süresini uzatabilir.
            </p>
            <input 
              type="range" 
              min="2" 
              max="10" 
              value={depth} 
              onChange={e => setDepth(parseInt(e.target.value))}
              style={{ width: '100%', cursor: 'pointer' }}
            />
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.5rem', fontWeight: 600 }}>
              <span>Daha Hızlı (2)</span>
              <span>Daha Detaylı (10)</span>
            </div>
          </div>

          <button className="btn-primary" onClick={handleSave} style={{ alignSelf: 'flex-start', padding: '0.75rem 2rem' }}>
            Ayarları Kaydet
          </button>
        </div>
      </main>
    </div>
  )
}

export default Settings
