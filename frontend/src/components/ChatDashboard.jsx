import React, { useState, useEffect, useRef } from 'react'
import { Link } from 'react-router-dom'
import { LogOut, Send, User, Bot, HelpCircle, ChevronRight, Book, Shield, Network, MessageSquare, Settings, Bookmark } from 'lucide-react'
import KnowledgeGraph from './KnowledgeGraph'

function ChatDashboard({ user, onLogout }) {
  const [activeTab, setActiveTab] = useState('chat')
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [currentAnalysis, setCurrentAnalysis] = useState('')
  const [sources, setSources] = useState([])
  const chatEndRef = useRef(null)

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages, currentAnalysis])

  const handleLogout = () => {
    localStorage.removeItem('token')
    onLogout()
  }

  const handleSaveResponse = async (query, responseText) => {
    try {
      const res = await fetch('/api/history/save', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify({ query, response: responseText })
      })
      if (res.ok) alert('Yanıt profilinize kaydedildi.')
      else alert('Kaydedilemedi.')
    } catch (err) {
      alert('Sunucu hatası.')
    }
  }

  const handleSendMessage = async (e) => {
    e.preventDefault()
    if (!input.trim() || loading) return

    const userMessage = input
    setInput('')
    setMessages(prev => [...prev, { role: 'user', content: userMessage }])
    setLoading(true)
    setCurrentAnalysis('')
    setSources([])

    try {
      const kDepth = parseInt(localStorage.getItem('k_depth') || '5')
      
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify({ message: userMessage, k: kDepth }),
      })

      if (!response.ok) throw new Error('Sunucu hatası.')

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let assistantMessage = ''

      while (true) {
        const { value, done } = await reader.read()
        if (done) break

        const chunk = decoder.decode(value)
        const lines = chunk.split('\n\n')

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = JSON.parse(line.replace('data: ', ''))

            if (data.status) {
              // Optionally handle status updates
            } else if (data.analysis_content) {
              setCurrentAnalysis(data.analysis_content)
            } else if (data.final_answer) {
              assistantMessage = data.final_answer
              // Update last message or add new one
              setMessages(prev => {
                const history = [...prev]
                const last = history[history.length - 1]
                if (last?.role === 'assistant') {
                  last.content = assistantMessage
                  last.analysis = currentAnalysis
                  return history
                } else {
                  return [...history, { role: 'assistant', content: assistantMessage, analysis: currentAnalysis }]
                }
              })
            } else if (data.source) {
              setSources(prev => [...prev, data.source])
            }
          }
        }
      }
    } catch (err) {
      setMessages(prev => [...prev, { role: 'assistant', content: 'Üzgünüm, bir hata oluştu: ' + err.message }])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="dashboard-layout" style={{ display: 'flex', height: '100vh', overflow: 'hidden' }}>
      {/* Sidebar */}
      <aside className="sidebar glass" style={{ width: '300px', borderRight: '1px solid var(--border)', display: 'flex', flexDirection: 'column' }}>
        <div style={{ padding: '2rem' }}>
          <h1 className="text-gradient" style={{ fontSize: '1.5rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Bot size={28} /> SUT Asistanı
          </h1>
        </div>
        
        <nav style={{ flex: 1, padding: '0 1rem' }}>
          {user?.role === 'admin' && (
            <div style={{ marginBottom: '2rem' }}>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '1rem', paddingLeft: '1rem', textTransform: 'uppercase', letterSpacing: '1px' }}>Yönetici</p>
              <Link to="/admin" style={{ textDecoration: 'none', width: '100%', textAlign: 'left', padding: '0.75rem 1rem', display: 'flex', alignItems: 'center', gap: '0.5rem', background: 'transparent', color: 'var(--accent)', fontSize: '0.9rem', fontWeight: 600, borderRadius: '8px' }} onMouseOver={e => e.currentTarget.style.background = '#f1f5f9'} onMouseOut={e => e.currentTarget.style.background = 'transparent'}>
                <Shield size={16} /> Admin Paneli
              </Link>
            </div>
          )}

          <div style={{ marginBottom: '2rem' }}>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '1rem', paddingLeft: '1rem', textTransform: 'uppercase', letterSpacing: '1px' }}>Görünümler</p>
            <button 
              onClick={() => setActiveTab('chat')}
              style={{ width: '100%', textAlign: 'left', padding: '0.75rem 1rem', display: 'flex', alignItems: 'center', gap: '0.5rem', background: activeTab === 'chat' ? '#e2e8f0' : 'transparent', color: 'var(--text-main)', fontSize: '0.9rem', borderRadius: '8px' }}
            >
              <MessageSquare size={16} color="var(--primary)" /> Sohbet
            </button>
            <button 
              onClick={() => setActiveTab('graph')}
              style={{ width: '100%', textAlign: 'left', padding: '0.75rem 1rem', display: 'flex', alignItems: 'center', gap: '0.5rem', background: activeTab === 'graph' ? '#e2e8f0' : 'transparent', color: 'var(--text-main)', fontSize: '0.9rem', borderRadius: '8px', marginTop: '0.5rem' }}
            >
              <Network size={16} color="#10b981" /> Bilgi Grafiği
            </button>
          </div>
          
          <div style={{ marginBottom: '2rem' }}>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '1rem', paddingLeft: '1rem', textTransform: 'uppercase', letterSpacing: '1px' }}>Hızlı Sorgular</p>
            {[
              "Kanser ilacı ödeme şartları",
              "Sürrenal yetmezlik raporu",
              "Fizik tedavi seans limitleri"
            ].map(q => (
              <button 
                key={q} 
                onClick={() => setInput(q)}
                style={{ width: '100%', textAlign: 'left', padding: '0.75rem 1rem', display: 'flex', alignItems: 'center', gap: '0.5rem', background: 'transparent', color: 'var(--text-main)', fontSize: '0.9rem' }}
                onMouseOver={e => e.currentTarget.style.background = '#f1f5f9'}
                onMouseOut={e => e.currentTarget.style.background = 'transparent'}
              >
                <HelpCircle size={16} color="var(--text-muted)" /> {q}
              </button>
            ))}
          </div>

          {sources.length > 0 && (
            <div>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '1rem', paddingLeft: '1rem', textTransform: 'uppercase', letterSpacing: '1px' }}>Kaynaklar</p>
              <div style={{ maxHeight: '200px', overflowY: 'auto' }}>
                {sources.map((s, i) => (
                  <div key={i} style={{ padding: '0.75rem 1rem', fontSize: '0.85rem', borderLeft: '2px solid var(--accent)', marginBottom: '0.5rem', background: '#f8fafc' }}>
                    <Book size={14} style={{ marginBottom: '4px' }} />
                    <div style={{ fontWeight: 600 }}>{s.title}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </nav>

        <div style={{ padding: '1.5rem', borderTop: '1px solid var(--border)' }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', marginBottom: '1rem' }}>
            <Link to="/profile" style={{ textDecoration: 'none', color: 'var(--text-main)', display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.5rem', borderRadius: '8px', fontSize: '0.9rem', fontWeight: 500 }} onMouseOver={e => e.currentTarget.style.background = '#f1f5f9'} onMouseOut={e => e.currentTarget.style.background = 'transparent'}>
              <User size={16} /> Profilim
            </Link>
            <Link to="/settings" style={{ textDecoration: 'none', color: 'var(--text-main)', display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.5rem', borderRadius: '8px', fontSize: '0.9rem', fontWeight: 500 }} onMouseOver={e => e.currentTarget.style.background = '#f1f5f9'} onMouseOut={e => e.currentTarget.style.background = 'transparent'}>
              <Settings size={16} /> Ayarlar
            </Link>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '0.5rem' }}>
            <div style={{ width: '40px', height: '40px', borderRadius: '50%', background: 'var(--accent)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'white', flexShrink: 0 }}>
              <User size={20} />
            </div>
            <div style={{ flex: 1, overflow: 'hidden' }}>
              <div style={{ fontWeight: 600, fontSize: '0.9rem', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{user.username}</div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{user.role.toUpperCase()}</div>
            </div>
            <button onClick={handleLogout} style={{ color: 'var(--text-muted)', background: 'transparent' }} title="Çıkış Yap">
              <LogOut size={20} />
            </button>
          </div>
        </div>
      </aside>

      {/* Main Chat Area */}
      <main style={{ flex: 1, display: 'flex', flexDirection: 'column', background: 'var(--bg)', position: 'relative' }}>
        <header className="glass" style={{ padding: '1rem 2rem', borderBottom: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', zIndex: 10 }}>
          <div>
            <span style={{ fontWeight: 600 }}>{activeTab === 'chat' ? 'SUT Mevzuat Sohbeti' : 'SUT Bilgi Grafiği'}</span>
          </div>
          <div style={{ fontSize: '0.8rem', color: '#10b981', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
            <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#10b981' }}></div>
            Sistem Çevrimiçi
          </div>
        </header>

        {activeTab === 'graph' ? (
          <div style={{ flex: 1, position: 'relative' }}>
            <KnowledgeGraph />
          </div>
        ) : (
          <>
            <div style={{ flex: 1, overflowY: 'auto', padding: '2rem 15%' }}>
              {messages.length === 0 && !currentAnalysis && (
                <div style={{ height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', textAlign: 'center' }}>
                  <div style={{ width: '80px', height: '80px', borderRadius: '24px', background: 'white', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--accent)', marginBottom: '1.5rem', boxShadow: '0 20px 25px -5px rgba(0,0,0,0.1)' }}>
                    <Bot size={48} />
                  </div>
                  <h2 style={{ fontSize: '2rem', fontWeight: 700, marginBottom: '0.5rem' }}>Merhaba, {user.username}</h2>
                  <p style={{ color: 'var(--text-muted)', maxWidth: '500px' }}>Bugün Sağlık Uygulama Tebliği (SUT) ile ilgili size nasıl yardımcı olabilirim? Merak ettiğiniz maddeyi veya durumu sorabilirsiniz.</p>
                </div>
              )}

              {messages.map((msg, i) => (
                <div key={i} style={{ marginBottom: '2.5rem', animation: 'fadeIn 0.5s ease' }}>
                  <div style={{ display: 'flex', gap: '1rem', justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start' }}>
                    {msg.role === 'assistant' && (
                      <div style={{ width: '36px', height: '36px', borderRadius: '10px', background: 'var(--primary)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'white', flexShrink: 0 }}>
                        <Bot size={18} />
                      </div>
                    )}
                    
                    <div style={{ maxWidth: '80%' }}>
                      <div className={msg.role === 'user' ? 'btn-primary' : 'premium-card'} style={{ padding: '1rem 1.25rem', borderRadius: msg.role === 'user' ? '20px 20px 4px 20px' : '4px 20px 20px 20px', fontSize: '0.95rem', boxShadow: msg.role === 'user' ? '0 4px 15px rgba(59,130,246,0.3)' : '0 2px 10px rgba(0,0,0,0.05)' }}>
                        {msg.content}
                      </div>
                      
                      {msg.analysis && (
                        <div style={{ marginTop: '0.75rem', padding: '0.75rem 1rem', background: '#f1f5f9', borderRadius: '12px', fontSize: '0.85rem', color: 'var(--text-muted)', borderLeft: '3px solid #cbd5e1' }}>
                          <p style={{ fontWeight: 600, color: 'var(--text-main)', marginBottom: '0.25rem', fontSize: '0.8rem' }}>Analiz:</p>
                          {msg.analysis}
                        </div>
                      )}
                      
                      {msg.role === 'assistant' && msg.content && !loading && (
                        <button 
                          onClick={() => {
                            // Find the most recent user query before this assistant message
                            let queryStr = 'Sorgu bulunamadı'
                            for(let j = i-1; j >= 0; j--) {
                              if (messages[j].role === 'user') { queryStr = messages[j].content; break; }
                            }
                            handleSaveResponse(queryStr, msg.content)
                          }}
                          style={{ marginTop: '0.5rem', marginLeft: '0.5rem', display: 'inline-flex', alignItems: 'center', gap: '0.25rem', fontSize: '0.75rem', color: 'var(--primary)', background: 'transparent', border: 'none', cursor: 'pointer', fontWeight: 600 }}
                        >
                          <Bookmark size={14} /> Yanıtı Profilime Kaydet
                        </button>
                      )}
                    </div>

                    {msg.role === 'user' && (
                      <div style={{ width: '36px', height: '36px', borderRadius: '10px', background: 'white', border: '1px solid var(--border)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--primary)', flexShrink: 0 }}>
                        <User size={18} />
                      </div>
                    )}
                  </div>
                </div>
              ))}

              {loading && !messages.find(m => m.role === 'assistant' && m.content) && (
                <div style={{ display: 'flex', gap: '1rem', marginBottom: '2rem' }}>
                  <div style={{ width: '36px', height: '36px', borderRadius: '10px', background: 'var(--primary)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'white' }}>
                    <Bot size={18} className="spin" />
                  </div>
                  <div style={{ flex: 1 }}>
                    {currentAnalysis && (
                      <div style={{ padding: '1rem', background: '#f1f5f9', borderRadius: '12px', fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>
                        <p style={{ fontWeight: 600, color: 'var(--text-main)', marginBottom: '0.5rem' }}>Analiz Ediliyor...</p>
                        {currentAnalysis}
                      </div>
                    )}
                    <div className="premium-card loading-skeleton" style={{ height: '60px', width: '200px' }}></div>
                  </div>
                </div>
              )}
              <div ref={chatEndRef} />
            </div>

            <div style={{ padding: '2rem 15%', background: 'white', borderTop: '1px solid var(--border)' }}>
              <form onSubmit={handleSendMessage} style={{ display: 'flex', gap: '1rem', position: 'relative' }}>
                <input 
                  type="text" 
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder="SUT mevzuatı hakkında soru sorun..."
                  style={{ paddingRight: '4rem', height: '56px', fontSize: '1rem', boxShadow: '0 10px 15px -3px rgba(0,0,0,0.05)' }}
                  disabled={loading}
                />
                <button 
                  type="submit" 
                  disabled={loading || !input.trim()}
                  style={{ position: 'absolute', right: '8px', top: '8px', bottom: '8px', width: '40px', borderRadius: '10px', display: 'flex', alignItems: 'center', justifyContent: 'center', background: input.trim() ? 'var(--accent)' : '#e2e8f0', color: 'white' }}
                >
                  <Send size={20} />
                </button>
              </form>
              <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textAlign: 'center', marginTop: '1rem' }}>
                Yapay zeka hatalar yapabilir. Önemli kararlar için resmi SUT metnini kontrol edin.
              </p>
            </div>
          </>
        )}
      </main>

      <style>{`
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .loading-skeleton { background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%); background-size: 200% 100%; animation: loading 1.5s infinite; }
        @keyframes loading { from { background-position: 200% 0; } to { background-position: -200% 0; } }
        .spin { animation: spin 2s linear infinite; }
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
      `}</style>
    </div>
  )
}

export default ChatDashboard
