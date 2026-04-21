import React, { useState, useEffect, useRef } from 'react'
import { Link } from 'react-router-dom'
import { 
  MessageSquare, Send, BookOpen, Clock, Settings, LogOut, User, Search, 
  HelpCircle, Shield, Bot, Paperclip, Bookmark, ThumbsUp, ThumbsDown, 
  Trash2, Plus, Network, Share2, Info, GitBranch, Loader, FileText, Book, X, Megaphone
} from 'lucide-react';
import KnowledgeGraph from './KnowledgeGraph'
import ThemeToggle from './ThemeToggle'
import ResponseFeedback from './ResponseFeedback'

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

// ─── Agent Step Trace Component ────────────────────────────────────────────
const TOOL_COLORS = {
  search_sut_chunks:     { bg: '#eff6ff', border: '#3b82f6', text: '#1e40af' },
  search_sut_fulltext:   { bg: '#f0fdf4', border: '#22c55e', text: '#166534' },
  lookup_knowledge_graph:{ bg: '#fdf4ff', border: '#a855f7', text: '#7e22ce' },
  calculate:             { bg: '#fff7ed', border: '#f97316', text: '#9a3412' },
  finish:                { bg: '#f0fdf4', border: '#10b981', text: '#065f46' },
  error:                 { bg: '#fef2f2', border: '#ef4444', text: '#991b1b' },
}

function AgentTrace({ steps, live = false }) {
  const [open, setOpen] = useState(false)
  if (!steps || steps.length === 0) return null

  const colors = (tool) => TOOL_COLORS[tool] || { bg: '#f9fafb', border: '#9ca3af', text: '#374151' }

  return (
    <div style={{ marginTop: '0.75rem', borderTop: '1px solid rgba(0,0,0,0.07)', paddingTop: '0.5rem' }}>
      <button
        onClick={() => setOpen(o => !o)}
        style={{
          display: 'flex', alignItems: 'center', gap: '0.4rem',
          fontSize: '0.75rem', fontWeight: 600, color: '#6366f1',
          background: 'transparent', border: 'none', cursor: 'pointer', padding: '0.2rem 0',
        }}
      >
        {open ? '▾' : '▸'} Düşünce Süreci ({steps.length} adım){live ? ' ⏳' : ''}
      </button>

      {open && (
        <div style={{ marginTop: '0.5rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
          {steps.map((step, i) => {
            const c = colors(step.tool)
            const argsStr = step.args && Object.keys(step.args).length > 0
              ? Object.entries(step.args).map(([k, v]) => `${k}: ${String(v).slice(0,80)}`).join(' | ')
              : ''
            return (
              <div key={i} style={{
                borderRadius: '8px', border: `1px solid ${c.border}`,
                background: c.bg, padding: '0.6rem 0.75rem', fontSize: '0.78rem',
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', marginBottom: '0.35rem' }}>
                  <span style={{ fontWeight: 700, color: c.text }}>{step.icon} {step.tool}</span>
                  <span style={{ color: '#94a3b8', fontSize: '0.7rem' }}>Adım {step.iteration}</span>
                </div>
                {argsStr && <div style={{ color: '#64748b', marginBottom: '0.3rem', fontStyle: 'italic' }}>{argsStr}</div>}
                {step.result && step.tool !== 'finish' && (
                  <details style={{ marginTop: '0.25rem' }}>
                    <summary style={{ cursor: 'pointer', color: c.text, fontWeight: 600, listStyle: 'none', outline: 'none' }}>Sonucu görüntüle ▸</summary>
                    <div style={{
                      marginTop: '0.35rem', padding: '0.5rem', background: 'rgba(0,0,0,0.04)',
                      borderRadius: '6px', whiteSpace: 'pre-wrap', color: '#374151', maxHeight: '200px', overflowY: 'auto',
                    }}>{step.result}</div>
                  </details>
                )}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
// ────────────────────────────────────────────────────────────────────────────

function ChatDashboard({ user, onLogout }) {
  const [activeTab, setActiveTab] = useState('chat')
  const [messages, setMessages] = useState([])
  const [activeConversationId, setActiveConversationId] = useState(null)
  const [input, setInput] = useState('')
  const [selectedRole, setSelectedRole] = useState('PATIENT')
  const [loading, setLoading] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [currentAnalysis, setCurrentAnalysis] = useState('')
  const [liveAgentSteps, setLiveAgentSteps] = useState([])
  const [sources, setSources] = useState([])
  const [conversations, setConversations] = useState([])
  const [announcement, setAnnouncement] = useState(null)
  const [annDismissed, setAnnDismissed] = useState(false)
  const chatEndRef = useRef(null)
  const fileInputRef = useRef(null)

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
                <details key={i} style={{ background: 'var(--card-bg)', borderRadius: '8px', border: '1px solid var(--border)', overflow: 'hidden' }}>
                  <summary style={{ padding: '0.75rem 1rem', fontWeight: 600, fontSize: '0.8rem', cursor: 'pointer', outline: 'none', userSelect: 'none', color: 'var(--primary)', background: 'var(--bg)' }}>
                    {src.title}
                  </summary>
                  <div style={{ padding: '1rem', fontSize: '0.85rem', color: 'var(--text-main)', background: 'var(--card-bg)', whiteSpace: 'pre-wrap', maxHeight: '350px', overflowY: 'auto' }}>
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
    setLiveAgentSteps([])
    setSources([])

    let queryId = null
    let assistantMessage = ''
    let accumulatedSteps = []

    try {
      const kDepth = parseInt(localStorage.getItem('k_depth') || '5')
      const response = await fetch('/api/chat/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...AUTH_HEADER() },
        body: JSON.stringify({ 
          query: userMessage, 
          conversation_id: activeConversationId,
          role: selectedRole
        }),
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
              if (data.conversation_id) setActiveConversationId(data.conversation_id)

            } else if (data.status) {
              setCurrentAnalysis(data.status)

            } else if (data.agent_step) {
              // Live streaming of each tool call step
              accumulatedSteps = [...accumulatedSteps, data.agent_step]
              setLiveAgentSteps([...accumulatedSteps])

            } else if (data.agent_steps_complete) {
              // All steps done — attach them to the upcoming assistant message
              accumulatedSteps = data.agent_steps_complete

            } else if (data.final_answer) {
              assistantMessage = data.final_answer
              const finalSteps = accumulatedSteps
              setMessages(prev => {
                const history = [...prev]
                const last = history[history.length - 1]
                if (last?.role === 'assistant') {
                  last.content = assistantMessage
                  last.id = queryId
                  last.agentSteps = finalSteps
                  return [...history]
                }
                return [...history, {
                  role: 'assistant',
                  content: assistantMessage,
                  id: queryId,
                  agentSteps: finalSteps
                }]
              })
            }
          } catch { /* ignore malformed SSE lines */ }
        }
      }

      if (queryId && assistantMessage) fetchHistory()
    } catch (err) {
      setMessages(prev => [...prev, { role: 'assistant', content: 'Üzgünüm, bir hata oluştu: ' + err.message }])
    } finally {
      setLoading(false)
      setLiveAgentSteps([])
      setCurrentAnalysis('')
    }
  }

  const handleFileUpload = async (e) => {
    const file = e.target.files[0]
    if (!file) return
    if (!file.filename && !file.name.toLowerCase().endsWith('.pdf')) {
      alert('Sadece PDF dosyaları yüklenebilir.')
      return
    }

    setUploading(true)
    const formData = new FormData()
    formData.append('file', file)
    
    // Auto-generate conversation ID if none active
    const convId = activeConversationId || `conv_${Date.now()}`
    if (!activeConversationId) setActiveConversationId(convId)

    try {
      const res = await fetch(`/api/chat/upload?conversation_id=${convId}`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${localStorage.getItem('token')}` },
        body: formData
      })
      if (res.ok) {
        const data = await res.json()
        setMessages(prev => [...prev, { role: 'assistant', content: `📄 **${file.name}** başarıyla yüklendi ve analiz edildi. Artık bu döküman hakkında soru sorabilirsiniz.` }])
      } else {
        const err = await res.json()
        alert(`Yükleme hatası: ${err.detail || 'Bilinmeyen hata'}`)
      }
    } catch (err) {
      alert('Sunucuya bağlanılamadı.')
    } finally {
      setUploading(false)
      if (fileInputRef.current) fileInputRef.current.value = ''
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
              <button onClick={() => setActiveTab('chat')} style={{ flex: 1, padding: '0.5rem', borderRadius: '8px', border: 'none', background: activeTab === 'chat' ? 'var(--primary)' : 'var(--bg)', color: activeTab === 'chat' ? 'white' : 'var(--text-muted)', cursor: 'pointer', fontSize: '0.8rem', fontWeight: 600, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.3rem' }}>
                <MessageSquare size={14} /> Sohbet
              </button>
              <button onClick={() => setActiveTab('graph')} style={{ flex: 1, padding: '0.5rem', borderRadius: '8px', border: 'none', background: activeTab === 'graph' ? '#10b981' : 'var(--bg)', color: activeTab === 'graph' ? 'white' : 'var(--text-muted)', cursor: 'pointer', fontSize: '0.8rem', fontWeight: 600, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.3rem' }}>
                <Network size={14} /> Bilgi Grafiği
              </button>
            </div>
            <button onClick={handleNewChat} style={{ width: '100%', padding: '0.6rem', background: 'var(--bg)', color: 'var(--text-main)', border: '1px solid var(--border)', borderRadius: '8px', cursor: 'pointer', fontWeight: 600, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem' }}>
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
                <Link to="/admin" style={{ textDecoration: 'none', color: 'var(--accent)', display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.6rem 0.75rem', borderRadius: '8px', fontSize: '0.85rem', fontWeight: 600 }}>
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
                        style={{ width: '100%', textAlign: 'left', padding: '0.6rem 0.75rem', borderRadius: '8px', border: 'none', background: activeConversationId === conv.conversation_id ? 'var(--bg)' : 'transparent', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '0.5rem' }}
                      >
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', flex: 1, overflow: 'hidden' }}>
                          <Clock size={12} color="var(--text-muted)" style={{ flexShrink: 0 }} />
                          {conv.file_metadata && (
                            <FileText size={12} color="var(--primary)" style={{ flexShrink: 0 }} title={`Dosya: ${conv.file_metadata.filename || 'PDF'}`} />
                          )}
                          <span style={{ fontSize: '0.82rem', color: 'var(--text-main)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                            {conv.query.slice(0, 38)}{conv.query.length > 38 ? '...' : ''}
                          </span>
                        </div>
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
                <button key={q} onClick={() => setInput(q)} style={{ width: '100%', textAlign: 'left', padding: '0.6rem 0.75rem', borderRadius: '8px', border: 'none', background: 'transparent', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.82rem', color: 'var(--text-main)' }}>
                  <HelpCircle size={12} color="var(--text-muted)" /> {q}
                </button>
              ))}
            </div>


          </nav>

          {/* User Footer */}
          <div style={{ padding: '1rem', borderTop: '1px solid var(--border)' }}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.15rem', marginBottom: '0.75rem' }}>
              <Link to="/profile" style={{ textDecoration: 'none', color: 'var(--text-main)', display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.4rem 0.5rem', borderRadius: '8px', fontSize: '0.85rem' }}>
                <User size={14} /> Profilim
              </Link>
              <Link to="/settings" style={{ textDecoration: 'none', color: 'var(--text-main)', display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.4rem 0.5rem', borderRadius: '8px', fontSize: '0.85rem' }}>
                <Settings size={14} /> Ayarlar
              </Link>
            </div>
            <div style={{ marginBottom: '1rem' }}>
              <ThemeToggle />
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
          <header className="glass" style={{ padding: '0.75rem 2rem', borderBottom: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', zIndex: 10, flexShrink: 0 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem' }}>
              <span style={{ fontWeight: 600 }}>{activeTab === 'chat' ? 'SUT Mevzuat Sohbeti' : 'SUT Bilgi Grafiği'}</span>
              
              {activeTab === 'chat' && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', background: 'var(--bg)', padding: '0.2rem 0.5rem', borderRadius: '8px', border: '1px solid var(--border)' }}>
                  <span style={{ fontSize: '0.7rem', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Persona:</span>
                  <select 
                    value={selectedRole} 
                    onChange={(e) => setSelectedRole(e.target.value)}
                    style={{ background: 'transparent', border: 'none', fontSize: '0.8rem', fontWeight: 600, color: 'var(--primary)', cursor: 'pointer', outline: 'none' }}
                  >
                    <option value="PATIENT">Vatandaş (Sade)</option>
                    <option value="DOCTOR">Doktor (Teknik)</option>
                    <option value="ADMIN">SGK Yöneticisi</option>
                  </select>
                </div>
              )}
            </div>
            
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
                    <div style={{ width: '80px', height: '80px', borderRadius: '24px', background: 'var(--card-bg)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--accent)', marginBottom: '1.5rem', boxShadow: '0 20px 25px -5px rgba(0,0,0,0.1)' }}>
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
                        <div className={msg.role === 'user' ? 'btn-primary' : 'premium-card'} style={{ padding: '1rem 1.25rem', borderRadius: msg.role === 'user' ? '20px 20px 4px 20px' : '4px 20px 20px 20px', fontSize: '0.95rem' }}>
                          <div style={{ whiteSpace: 'pre-wrap' }}>
                            {renderMessageContent(msg.content)}
                          </div>

                          {/* Agent Trace — expandable per message */}
                          {msg.role === 'assistant' && msg.agentSteps && msg.agentSteps.length > 0 && (
                            <AgentTrace steps={msg.agentSteps} />
                          )}
                          
                          {msg.role === 'assistant' && (!loading && msg.content) && (
                            <div style={{ marginTop: '1rem', paddingTop: '0.75rem', borderTop: '1px solid rgba(0,0,0,0.06)', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                              {!loading && msg.content && (
                                <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
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
                                    <div style={{ display: 'flex', gap: '0.5rem', borderLeft: '1px solid var(--border)', paddingLeft: '0.75rem' }}>
                                      <button onClick={() => handleFeedback(msg.id, 5, true)} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.25rem', fontSize: '0.75rem', color: '#10b981', background: 'transparent', border: 'none', cursor: 'pointer', fontWeight: 600 }}>
                                        <ThumbsUp size={14} /> Doğru
                                      </button>
                                      <button onClick={() => handleFeedback(msg.id, 1, false)} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.25rem', fontSize: '0.75rem', color: '#ef4444', background: 'transparent', border: 'none', cursor: 'pointer', fontWeight: 600 }}>
                                        <ThumbsDown size={14} /> Yanlış
                                      </button>
                                    </div>
                                  )}
                                  {msg.feedbackSubmitted && (
                                    <span style={{ fontSize: '0.7rem', color: '#10b981', borderLeft: '1px solid var(--border)', paddingLeft: '0.75rem' }}>✓ Geri bildirim alındı</span>
                                  )}
                                </div>
                              )}
                            </div>
                          )}
                        </div>
                      </div>
                      {msg.role === 'user' && (
                        <div style={{ width: '36px', height: '36px', borderRadius: '10px', background: 'var(--card-bg)', border: '1px solid var(--border)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--primary)', flexShrink: 0 }}>
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
                        <div style={{ padding: '0.75rem 1rem', background: 'var(--card-bg)', borderRadius: '12px', fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                          <span style={{ animation: 'spin 1.5s linear infinite', display: 'inline-block' }}>⚙️</span>
                          <span style={{ fontWeight: 500 }}>{currentAnalysis}</span>
                        </div>
                      )}
                      {/* Live Agent Steps while loading */}
                      {liveAgentSteps.length > 0 && (
                        <AgentTrace steps={liveAgentSteps} live={true} />
                      )}
                      <div className="premium-card loading-skeleton" style={{ height: '50px', width: '160px', marginTop: '0.5rem' }} />
                    </div>
                  </div>
                )}
                <div ref={chatEndRef} />
              </div>

              <div style={{ padding: '1.5rem 12%', background: 'var(--card-bg)', borderTop: '1px solid var(--border)', flexShrink: 0 }}>
                <form onSubmit={handleSendMessage} style={{ display: 'flex', gap: '1rem', position: 'relative' }}>
                  <input
                    type="file"
                    ref={fileInputRef}
                    onChange={handleFileUpload}
                    style={{ display: 'none' }}
                    accept=".pdf"
                  />
                  <button 
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                    disabled={loading || uploading}
                    title="Döküman Yükle (PDF)"
                    style={{ width: '48px', height: '56px', borderRadius: '12px', background: 'var(--bg)', border: '1px solid var(--border)', color: 'var(--text-muted)', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
                  >
                    {uploading ? <div className="spin">⌛</div> : <Book size={20} />}
                  </button>
                  <input
                    type="text"
                    value={input}
                    onChange={e => setInput(e.target.value)}
                    placeholder="SUT mevzuatı hakkında soru sorun veya rapor yükleyin..."
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
