import React, { useState, useEffect, useRef } from 'react'
import { Link } from 'react-router-dom'
import { LogOut, Send, User, Bot, HelpCircle, Clock, Book, Shield, Network, MessageSquare, Settings, Bookmark, X, Megaphone, ThumbsUp, ThumbsDown, BookOpen } from 'lucide-react'
import KnowledgeGraph from './KnowledgeGraph'

const AUTH_HEADER = () => ({ 'Authorization': `Bearer ${localStorage.getItem('token')}` })

function groupByDate(items) {
  const groups = {}
  const today = new Date().toDateString()
  const yesterday = new Date(Date.now() - 86400000).toDateString()
  items.forEach(item => {
    const d = new Date(item.created_at + 'Z').toDateString()
    const label = d === today ? 'Bugün' : d === yesterday ? 'Dün' : 'Bu Hafta'
    if (!groups[label]) groups[label] = []
    groups[label].push(item)
  })
  return groups
}

function ChatDashboard({ user, onLogout }) {
  const [activeTab, setActiveTab] = useState('chat')
  const [messages, setMessages] = useState([])
  const [activeConversationId, setActiveConversationId] = useState(null)
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [currentAnalysis, setCurrentAnalysis] = useState('')
  const [sources, setSources] = useState([])
  const [conversations, setConversations] = useState([])
  const [announcement, setAnnouncement] = useState(null)
  const [annDismissed, setAnnDismissed] = useState(false)
  const chatEndRef = useRef(null)

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, currentAnalysis])

  useEffect(() => {
    fetchHistory()
    fetchAnnouncement()
  }, [])

  const fetchHistory = async () => {
    try {
      const res = await fetch('/api/history', { headers: AUTH_HEADER() })
      if (res.ok) {
        const data = await res.json()
        setConversations(data.history.filter(h => h.response))
      }
    } catch (e) { console.error(e) }
  }

  const fetchAnnouncement = async () => {
    try {
      const res = await fetch('/api/announcements', { headers: AUTH_HEADER() })
      if (res.ok) {
        const data = await res.json()
        if (data.id) setAnnouncement(data)
      }
    } catch (e) { console.error(e) }
  }

  const handleLogout = () => {
    localStorage.removeItem('token')
    onLogout()
  }

  const renderMessageContent = (content) => {
    if (!content.includes('<KAYNAKLAR>')) {
      return content;
    }

    const mainText = content.split('<KAYNAKLAR>')[0];
    const sourcesMatch = content.match(/<KAYNAKLAR>([\s\S]*?)<\/KAYNAKLAR>/);
    
    let parsedSources = [];
    if (sourcesMatch && sourcesMatch[1]) {
      const sourceBlocks = sourcesMatch[1].split('</KAYNAK>').filter(s => s.trim().startsWith('<KAYNAK'));
      parsedSources = sourceBlocks.map(block => {
        const titleMatch = block.match(/baslik="(.*?)"/);
        const title = titleMatch ? titleMatch[1] : 'Kaynak';
        const textStart = block.indexOf('>') + 1;
        const text = block.substring(textStart).trim();
        return { title, text };
      });
    }

    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        <div>{mainText.trim()}</div>
        {parsedSources.length > 0 && (
          <div style={{ marginTop: '0.5rem' }}>
            <p style={{ fontWeight: 700, fontSize: '0.85rem', marginBottom: '0.75rem', color: 'var(--text-main)', borderTop: '1px solid rgba(0,0,0,0.1)', paddingTop: '1rem' }}>
              📚 Kullanılan Kaynaklar (Tam Metin):
            </p>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              {parsedSources.map((src, i) => (
                <details key={i} style={{ background: '#f8fafc', borderRadius: '8px', border: '1px solid var(--border)', overflow: 'hidden' }}>
                  <summary style={{ padding: '0.75rem 1rem', fontWeight: 600, fontSize: '0.8rem', cursor: 'pointer', outline: 'none', userSelect: 'none', color: 'var(--primary)', background: '#f1f5f9' }}>
                    {src.title}
                  </summary>
                  <div style={{ padding: '1rem', fontSize: '0.85rem', color: 'var(--text-main)', background: 'white', whiteSpace: 'pre-wrap', maxHeight: '350px', overflowY: 'auto' }}>
                    {src.text}
                  </div>
                </details>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  }

  const handleSaveResponse = async (query, responseText) => {
    try {
      const res = await fetch('/api/history/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...AUTH_HEADER() },
        body: JSON.stringify({ query, response: responseText })
      })
      if (res.ok) alert('Yanıt profilinize kaydedildi.')
      else alert('Kaydedilemedi.')
    } catch { alert('Sunucu hatası.') }
  }

  const handleFeedback = async (msgId, rating, isAccurate) => {
    if (!msgId) return
    try {
      const res = await fetch('/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...AUTH_HEADER() },
        body: JSON.stringify({ message_id: msgId, rating, is_accurate: isAccurate })
      })
      if (res.ok) {
        setMessages(prev => prev.map(m => m.id === msgId && m.role === 'assistant' ? { ...m, feedbackSubmitted: true } : m))
      }
    } catch (e) { console.error(e) }
  }

  const loadConversation = (convId) => {
    const msgs = conversations.filter(c => c.conversation_id === convId).reverse()
    const newMessages = []
    msgs.forEach(m => {
      newMessages.push({ role: 'user', content: m.query, id: m.id })
      if (m.response) newMessages.push({ role: 'assistant', content: m.response, id: m.id })
    })
    setMessages(newMessages)
    setActiveConversationId(convId)
    setSources([])
    setCurrentAnalysis('')
    setActiveTab('chat')
  }

  const handleNewChat = () => {
    setMessages([])
    setActiveConversationId(null)
    setSources([])
    setCurrentAnalysis('')
    setActiveTab('chat')
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

    let queryId = null
    let assistantMessage = ''

    try {
      const kDepth = parseInt(localStorage.getItem('k_depth') || '5')
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...AUTH_HEADER() },
        body: JSON.stringify({ message: userMessage, k: kDepth, conversation_id: activeConversationId }),
      })

      if (!response.ok) throw new Error('Sunucu hatası.')

      const reader = response.body.getReader()
      const decoder = new TextDecoder()

      while (true) {
        const { value, done } = await reader.read()
        if (done) break

        const chunk = decoder.decode(value)
        const lines = chunk.split('\n\n')

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          try {
            const data = JSON.parse(line.replace('data: ', ''))

            if (data.query_id) {
              queryId = data.query_id
              if (data.conversation_id) {
                setActiveConversationId(data.conversation_id)
              }
            } else if (data.status) {
              // Status updates — no-op
            } else if (data.analysis_content) {
              setCurrentAnalysis(data.analysis_content)
            } else if (data.final_answer) {
              assistantMessage = data.final_answer
              setMessages(prev => {
                const history = [...prev]
                const last = history[history.length - 1]
                if (last?.role === 'assistant') {
                  last.content = assistantMessage
                  last.analysis = currentAnalysis
                  last.id = queryId
                  return history
                }
                return [...history, { role: 'assistant', content: assistantMessage, analysis: currentAnalysis, id: queryId }]
              })
            }
          } catch { /* ignore malformed SSE lines */ }
        }
      }

      // Refresh conversation list to get latest DB changes
      if (queryId && assistantMessage) {
        fetchHistory()
      }
    } catch (err) {
      setMessages(prev => [...prev, { role: 'assistant', content: 'Üzgünüm, bir hata oluştu: ' + err.message }])
    } finally {
      setLoading(false)
    }
  }

  const groupedConvs = {}
  conversations.forEach(c => {
    if (!groupedConvs[c.conversation_id]) groupedConvs[c.conversation_id] = []
    groupedConvs[c.conversation_id].push(c)
  })
  const uniqueConvs = Object.values(groupedConvs).map(group => {
    const lastMsg = group[0]
    const firstMsg = group[group.length - 1]
    return { ...firstMsg, latest_date: lastMsg.created_at }
  }).sort((a,b) => new Date(b.latest_date) - new Date(a.latest_date))

  const mappedConvs = uniqueConvs.map(c => ({...c, created_at: c.latest_date}))
  const conversationGroups = groupByDate(mappedConvs)

  return (
    <div className="dashboard-layout" style={{ display: 'flex', height: '100vh', overflow: 'hidden', flexDirection: 'column' }}>

      {/* Announcement Banner */}
      {announcement && !annDismissed && (
        <div style={{ background: 'linear-gradient(135deg, #f59e0b, #ef4444)', color: 'white', padding: '0.65rem 2rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexShrink: 0, zIndex: 100 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', fontSize: '0.9rem', fontWeight: 500 }}>
            <Megaphone size={16} />
            <span>{announcement.message}</span>
          </div>
          <button onClick={() => setAnnDismissed(true)} style={{ background: 'rgba(255,255,255,0.2)', border: 'none', color: 'white', borderRadius: '6px', padding: '0.2rem 0.5rem', cursor: 'pointer' }}>
            <X size={14} />
          </button>
        </div>
      )}

      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        {/* Sidebar */}
        <aside className="sidebar glass" style={{ width: '300px', borderRight: '1px solid var(--border)', display: 'flex', flexDirection: 'column' }}>
          <div style={{ padding: '1.5rem 1.5rem 1rem' }}>
            <h1 className="text-gradient" style={{ fontSize: '1.3rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
              <Bot size={24} /> SUT Asistanı
            </h1>
            {/* View Tabs */}
            <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.75rem' }}>
              <button onClick={() => setActiveTab('chat')} style={{ flex: 1, padding: '0.5rem', borderRadius: '8px', border: 'none', background: activeTab === 'chat' ? 'var(--primary)' : '#f1f5f9', color: activeTab === 'chat' ? 'white' : 'var(--text-muted)', cursor: 'pointer', fontSize: '0.8rem', fontWeight: 600, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.3rem' }}>
                <MessageSquare size={14} /> Sohbet
              </button>
              <button onClick={() => setActiveTab('graph')} style={{ flex: 1, padding: '0.5rem', borderRadius: '8px', border: 'none', background: activeTab === 'graph' ? '#10b981' : '#f1f5f9', color: activeTab === 'graph' ? 'white' : 'var(--text-muted)', cursor: 'pointer', fontSize: '0.8rem', fontWeight: 600, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.3rem' }}>
                <Network size={14} /> Bilgi Grafiği
              </button>
            </div>
            <button onClick={handleNewChat} style={{ width: '100%', padding: '0.6rem', background: '#e2e8f0', color: 'var(--text-main)', border: 'none', borderRadius: '8px', cursor: 'pointer', fontWeight: 600, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem' }} onMouseOver={e => e.currentTarget.style.background = '#cbd5e1'} onMouseOut={e => e.currentTarget.style.background = '#e2e8f0'}>
              + Yeni Sohbet
            </button>
          </div>

          <nav style={{ flex: 1, overflowY: 'auto', padding: '0 0.75rem' }}>
            {/* Tools */}
            <div style={{ marginBottom: '1rem' }}>
              <p style={{ fontSize: '0.7rem', color: 'var(--text-muted)', paddingLeft: '0.5rem', textTransform: 'uppercase', letterSpacing: '1px', marginBottom: '0.5rem' }}>Araçlar</p>
              <Link to="/policies" style={{ textDecoration: 'none', color: 'var(--text-main)', display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.6rem 0.75rem', borderRadius: '8px', fontSize: '0.85rem', fontWeight: 600 }} onMouseOver={e => e.currentTarget.style.background = '#f1f5f9'} onMouseOut={e => e.currentTarget.style.background = 'transparent'}>
                <BookOpen size={14} /> SUT Mevzuat Tarayıcısı
              </Link>
            </div>

            {/* Admin links */}
            {user?.role === 'admin' && (
              <div style={{ marginBottom: '1rem' }}>
                <p style={{ fontSize: '0.7rem', color: 'var(--text-muted)', paddingLeft: '0.5rem', textTransform: 'uppercase', letterSpacing: '1px', marginBottom: '0.5rem' }}>Yönetici</p>
                <Link to="/admin" style={{ textDecoration: 'none', color: 'var(--accent)', display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.6rem 0.75rem', borderRadius: '8px', fontSize: '0.85rem', fontWeight: 600 }} onMouseOver={e => e.currentTarget.style.background = '#f1f5f9'} onMouseOut={e => e.currentTarget.style.background = 'transparent'}>
                  <Shield size={14} /> Admin Paneli
                </Link>
              </div>
            )}

            {/* Recent Conversations */}
            <div style={{ marginBottom: '1rem' }}>
              <p style={{ fontSize: '0.7rem', color: 'var(--text-muted)', paddingLeft: '0.5rem', textTransform: 'uppercase', letterSpacing: '1px', marginBottom: '0.5rem' }}>Son Konuşmalar</p>
              {conversations.length === 0 ? (
                <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', padding: '0.5rem', textAlign: 'center' }}>Henüz konuşma yok.</p>
              ) : (
                Object.entries(conversationGroups).map(([group, items]) => (
                  <div key={group}>
                    <p style={{ fontSize: '0.65rem', color: 'var(--text-muted)', padding: '0.25rem 0.75rem', textTransform: 'uppercase', letterSpacing: '0.5px', marginTop: '0.5rem' }}>{group}</p>
                    {items.slice(0, 5).map((conv, i) => (
                      <button
                        key={conv.conversation_id || i}
                        onClick={() => loadConversation(conv.conversation_id)}
                        style={{ width: '100%', textAlign: 'left', padding: '0.6rem 0.75rem', borderRadius: '8px', border: 'none', background: activeConversationId === conv.conversation_id ? '#e2e8f0' : 'transparent', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '0.5rem' }}
                        onMouseOver={e => e.currentTarget.style.background = activeConversationId === conv.conversation_id ? '#e2e8f0' : '#f1f5f9'}
                        onMouseOut={e => e.currentTarget.style.background = activeConversationId === conv.conversation_id ? '#e2e8f0' : 'transparent'}
                      >
                        <Clock size={12} color="var(--text-muted)" style={{ flexShrink: 0 }} />
                        <span style={{ fontSize: '0.82rem', color: 'var(--text-main)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                          {conv.query.slice(0, 38)}{conv.query.length > 38 ? '...' : ''}
                        </span>
                      </button>
                    ))}
                  </div>
                ))
              )}
              <Link to="/profile" style={{ display: 'block', textDecoration: 'none', color: 'var(--primary)', fontSize: '0.78rem', fontWeight: 600, padding: '0.5rem 0.75rem' }}>
                Tüm Geçmiş →
              </Link>
            </div>

            {/* Quick Queries */}
            <div style={{ marginBottom: '1rem' }}>
              <p style={{ fontSize: '0.7rem', color: 'var(--text-muted)', paddingLeft: '0.5rem', textTransform: 'uppercase', letterSpacing: '1px', marginBottom: '0.5rem' }}>Hızlı Sorgular</p>
              {["Kanser ilacı ödeme şartları", "Sürrenal yetmezlik raporu", "Fizik tedavi seans limitleri"].map(q => (
                <button key={q} onClick={() => setInput(q)} style={{ width: '100%', textAlign: 'left', padding: '0.6rem 0.75rem', borderRadius: '8px', border: 'none', background: 'transparent', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.82rem', color: 'var(--text-main)' }} onMouseOver={e => e.currentTarget.style.background = '#f1f5f9'} onMouseOut={e => e.currentTarget.style.background = 'transparent'}>
                  <HelpCircle size={12} color="var(--text-muted)" /> {q}
                </button>
              ))}
            </div>


          </nav>

          {/* User Footer */}
          <div style={{ padding: '1rem', borderTop: '1px solid var(--border)' }}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.15rem', marginBottom: '0.75rem' }}>
              <Link to="/profile" style={{ textDecoration: 'none', color: 'var(--text-main)', display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.4rem 0.5rem', borderRadius: '8px', fontSize: '0.85rem' }} onMouseOver={e => e.currentTarget.style.background = '#f1f5f9'} onMouseOut={e => e.currentTarget.style.background = 'transparent'}>
                <User size={14} /> Profilim
              </Link>
              <Link to="/settings" style={{ textDecoration: 'none', color: 'var(--text-main)', display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.4rem 0.5rem', borderRadius: '8px', fontSize: '0.85rem' }} onMouseOver={e => e.currentTarget.style.background = '#f1f5f9'} onMouseOut={e => e.currentTarget.style.background = 'transparent'}>
                <Settings size={14} /> Ayarlar
              </Link>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <div style={{ width: '36px', height: '36px', borderRadius: '50%', background: 'var(--accent)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'white', flexShrink: 0 }}>
                <User size={18} />
              </div>
              <div style={{ flex: 1, overflow: 'hidden' }}>
                <div style={{ fontWeight: 600, fontSize: '0.85rem', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{user.username}</div>
                <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>{user.role.toUpperCase()}</div>
              </div>
              <button onClick={handleLogout} style={{ color: 'var(--text-muted)', background: 'transparent', border: 'none', cursor: 'pointer' }} title="Çıkış Yap">
                <LogOut size={18} />
              </button>
            </div>
          </div>
        </aside>

        {/* Main Chat Area */}
        <main style={{ flex: 1, display: 'flex', flexDirection: 'column', background: 'var(--bg)', position: 'relative', overflow: 'hidden' }}>
          <header className="glass" style={{ padding: '1rem 2rem', borderBottom: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', zIndex: 10, flexShrink: 0 }}>
            <span style={{ fontWeight: 600 }}>{activeTab === 'chat' ? 'SUT Mevzuat Sohbeti' : 'SUT Bilgi Grafiği'}</span>
            <div style={{ fontSize: '0.8rem', color: '#10b981', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
              <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#10b981' }} />
              Sistem Çevrimiçi
            </div>
          </header>

          {activeTab === 'graph' ? (
            <div style={{ flex: 1, position: 'relative' }}>
              <KnowledgeGraph />
            </div>
          ) : (
            <>
              <div style={{ flex: 1, overflowY: 'auto', padding: '2rem 12%' }}>
                {messages.length === 0 && !currentAnalysis && (
                  <div style={{ height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', textAlign: 'center' }}>
                    <div style={{ width: '80px', height: '80px', borderRadius: '24px', background: 'white', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--accent)', marginBottom: '1.5rem', boxShadow: '0 20px 25px -5px rgba(0,0,0,0.1)' }}>
                      <Bot size={48} />
                    </div>
                    <h2 style={{ fontSize: '2rem', fontWeight: 700, marginBottom: '0.5rem' }}>Merhaba, {user.username}</h2>
                    <p style={{ color: 'var(--text-muted)', maxWidth: '500px' }}>Bugün Sağlık Uygulama Tebliği (SUT) ile ilgili size nasıl yardımcı olabilirim?</p>
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
                        <div className={msg.role === 'user' ? 'btn-primary' : 'premium-card'} style={{ padding: '1rem 1.25rem', borderRadius: msg.role === 'user' ? '20px 20px 4px 20px' : '4px 20px 20px 20px', fontSize: '0.95rem', whiteSpace: 'pre-wrap' }}>
                          {renderMessageContent(msg.content)}
                        </div>
                        {msg.analysis && (
                          <div style={{ marginTop: '0.75rem', padding: '0.75rem 1rem', background: '#f1f5f9', borderRadius: '12px', fontSize: '0.85rem', color: 'var(--text-muted)', borderLeft: '3px solid #cbd5e1' }}>
                            <p style={{ fontWeight: 600, color: 'var(--text-main)', marginBottom: '0.25rem', fontSize: '0.8rem' }}>Analiz:</p>
                            {msg.analysis}
                          </div>
                        )}
                        {msg.role === 'assistant' && msg.content && !loading && (
                          <div style={{ marginTop: '0.5rem', marginLeft: '0.5rem', display: 'flex', gap: '1rem', alignItems: 'center' }}>
                            <button
                              onClick={() => {
                                let queryStr = 'Sorgu bulunamadı'
                                for (let j = i - 1; j >= 0; j--) {
                                  if (messages[j].role === 'user') { queryStr = messages[j].content; break }
                                }
                                handleSaveResponse(queryStr, msg.content)
                              }}
                              style={{ display: 'inline-flex', alignItems: 'center', gap: '0.25rem', fontSize: '0.75rem', color: 'var(--primary)', background: 'transparent', border: 'none', cursor: 'pointer', fontWeight: 600 }}
                            >
                              <Bookmark size={14} /> Yanıtı Kaydet
                            </button>
                            
                            {msg.id && !msg.feedbackSubmitted && (
                              <div style={{ display: 'flex', gap: '0.5rem' }}>
                                <button onClick={() => handleFeedback(msg.id, 5, true)} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.25rem', fontSize: '0.75rem', color: '#10b981', background: 'transparent', border: 'none', cursor: 'pointer', fontWeight: 600 }}>
                                  <ThumbsUp size={14} /> Doğru
                                </button>
                                <button onClick={() => handleFeedback(msg.id, 1, false)} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.25rem', fontSize: '0.75rem', color: '#ef4444', background: 'transparent', border: 'none', cursor: 'pointer', fontWeight: 600 }}>
                                  <ThumbsDown size={14} /> Yanlış
                                </button>
                              </div>
                            )}
                            {msg.feedbackSubmitted && (
                              <span style={{ fontSize: '0.7rem', color: '#10b981' }}>✓ Geri bildirim alındı</span>
                            )}
                          </div>
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

                {loading && (
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
                      <div className="premium-card loading-skeleton" style={{ height: '60px', width: '200px' }} />
                    </div>
                  </div>
                )}
                <div ref={chatEndRef} />
              </div>

              <div style={{ padding: '1.5rem 12%', background: 'white', borderTop: '1px solid var(--border)', flexShrink: 0 }}>
                <form onSubmit={handleSendMessage} style={{ display: 'flex', gap: '1rem', position: 'relative' }}>
                  <input
                    type="text"
                    value={input}
                    onChange={e => setInput(e.target.value)}
                    placeholder="SUT mevzuatı hakkında soru sorun..."
                    style={{ paddingRight: '4rem', height: '56px', fontSize: '1rem', boxShadow: '0 10px 15px -3px rgba(0,0,0,0.05)' }}
                    disabled={loading}
                  />
                  <button type="submit" disabled={loading || !input.trim()} style={{ position: 'absolute', right: '8px', top: '8px', bottom: '8px', width: '40px', borderRadius: '10px', display: 'flex', alignItems: 'center', justifyContent: 'center', background: input.trim() ? 'var(--accent)' : '#e2e8f0', color: 'white', border: 'none', cursor: 'pointer' }}>
                    <Send size={20} />
                  </button>
                </form>
                <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textAlign: 'center', marginTop: '0.75rem' }}>
                  Yapay zeka hatalar yapabilir. Önemli kararlar için resmi SUT metnini kontrol edin.
                </p>
              </div>
            </>
          )}
        </main>
      </div>

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
