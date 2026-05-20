import React, { useState, useEffect, useRef, useCallback } from 'react'
import { Link, NavLink, useNavigate, useLocation } from 'react-router-dom'
import toast, { Toaster } from 'react-hot-toast'
import { 
  MessageSquare, Send, BookOpen, Clock, Settings, LogOut, User, Search, 
  HelpCircle, Shield, Bot, Bookmark, ThumbsUp, ThumbsDown, 
  Network, FileText, Book, X, Megaphone, Menu, Copy, Flag, Bell, Cpu, MoreHorizontal, Pencil, Star, Trash2
} from 'lucide-react'
import KnowledgeGraph from './KnowledgeGraph'
import MarkdownBody from './MarkdownBody'
import { extractSourceRefs } from '../utils/sourceRefs'
import { CommandPalette } from './CommandPalette'
import WelcomeModal from './WelcomeModal'
import FocusTrap from 'focus-trap-react'

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
  lookup_kg_entity:      { bg: '#fdf4ff', border: '#a855f7', text: '#7e22ce' },
  explore_kg_path:       { bg: '#fdf4ff', border: '#a855f7', text: '#7e22ce' },
  calculate:             { bg: '#fff7ed', border: '#f97316', text: '#9a3412' },
  finish:                { bg: '#f0fdf4', border: '#10b981', text: '#065f46' },
  error:                 { bg: '#fef2f2', border: '#ef4444', text: '#991b1b' },
  critic:                { bg: '#fffbeb', border: '#f59e0b', text: '#92400e' },
}

const QUICK_BY_ROLE = {
  PATIENT: [
    'İlaç ödenebilir mi?',
    'Rapor süresi ne kadar?',
    'Katılım payı ne kadar?',
    'Fizik tedavi limiti',
  ],
  DOCTOR: [
    'ICD-10 ile SUT eşleşmesi',
    'Uzman hekim raporu şartları',
    'Biyolojik ajan ön izin',
    'Yatarak tedavi ölçütleri',
  ],
  ECZACI: [
    'Muadil ilaç değişimi',
    'Reçete türleri ve süre',
    'Katılım payı istisnası',
    'Ek-4/A ilaç listesi',
  ],
  ADMIN: [
    'Fatura red kodları',
    'İlaç faturalama uyumu',
    'Sağlık hizmeti fiyatlandırma',
    'Kontrol raporu süreci',
  ],
  HASTANE_YONETICISI: [
    'SUT uyum denetimi',
    'Vaka bazında ödeme',
    'Yatak günü üst limiti',
    'Anlaşmalı kurum yükümlülükleri',
  ],
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
  const navigate = useNavigate()
  const location = useLocation()
  const [activeTab, setActiveTab] = useState('chat')
  const [messages, setMessages] = useState([])
  const [activeConversationId, setActiveConversationId] = useState(null)
  const [input, setInput] = useState('')
  const [selectedRole, setSelectedRole] = useState('PATIENT')
  const [loading, setLoading] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [currentAnalysis, setCurrentAnalysis] = useState('')
  const [liveAgentSteps, setLiveAgentSteps] = useState([])
  const [historyRows, setHistoryRows] = useState([])
  const [convSummaries, setConvSummaries] = useState([])
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [streamStarted, setStreamStarted] = useState(false)
  const [chatError, setChatError] = useState(null)
  const [modelLabel, setModelLabel] = useState('')
  const [paletteOpen, setPaletteOpen] = useState(false)
  const [reportOpen, setReportOpen] = useState(false)
  const [reportTargetId, setReportTargetId] = useState(null)
  const [reportCategory, setReportCategory] = useState('wrong_info')
  const [reportDetail, setReportDetail] = useState('')
  const [convSearch, setConvSearch] = useState('')
  const [announcement, setAnnouncement] = useState(null)
  const [annDismissed, setAnnDismissed] = useState(false)
  const [welcomeOpen, setWelcomeOpen] = useState(false)
  const chatEndRef = useRef(null)
  const fileInputRef = useRef(null)
  const pendingConvOpened = useRef(false)
  const chatAbortRef = useRef(null)

  // Abort any in-flight chat stream on unmount to prevent state updates on
  // unmounted component + dangling reader.
  useEffect(() => {
    return () => {
      try { chatAbortRef.current?.abort() } catch (_) {}
    }
  }, [])

  useEffect(() => {
    document.title = 'Sohbet — SUT Asistanı'
  }, [])

  useEffect(() => {
    try {
      if (!localStorage.getItem('sut_onboarding_done')) setWelcomeOpen(true)
    } catch (_) {}
  }, [])

  const dismissWelcome = () => {
    try {
      localStorage.setItem('sut_onboarding_done', '1')
    } catch (_) {}
    setWelcomeOpen(false)
  }

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, currentAnalysis])

  const refreshData = useCallback(async () => {
    try {
      const [hRes, cRes, cfgRes] = await Promise.all([
        fetch('/api/history', { headers: AUTH_HEADER() }),
        fetch('/api/conversations?limit=12', { headers: AUTH_HEADER() }),
        fetch('/api/config', { headers: AUTH_HEADER() }),
      ])
      // If session lapsed, bounce to login once instead of spamming network calls.
      if (hRes.status === 401 || cRes.status === 401 || cfgRes.status === 401) {
        try { localStorage.removeItem('token') } catch (_) {}
        navigate('/login')
        return
      }
      if (hRes.ok) {
        const data = await hRes.json()
        setHistoryRows((data.history || []).filter(x => x.response))
      }
      if (cRes.ok) {
        const data = await cRes.json()
        setConvSummaries(data.conversations || [])
      }
      if (cfgRes.ok) {
        const cfg = await cfgRes.json()
        setModelLabel(cfg.model_display_name || '')
      }
    } catch (e) {
      if (import.meta.env.DEV) console.error(e)
    }
  }, [navigate])

  useEffect(() => {
    refreshData()
    fetchAnnouncement()
  }, [refreshData])

  useEffect(() => {
    try {
      const p = sessionStorage.getItem('prefillChat')
      if (p) {
        sessionStorage.removeItem('prefillChat')
        setInput(p)
      }
    } catch (_) {}
  }, [])

  const fetchAnnouncement = async () => {
    try {
      const res = await fetch('/api/announcements', { headers: AUTH_HEADER() })
      if (res.ok) {
        const data = await res.json()
        if (data.id) setAnnouncement(data)
      }
    } catch (e) { if (import.meta.env.DEV) console.error(e) }
  }

  const handleLogout = () => {
    localStorage.removeItem('token')
    onLogout()
  }

  const renderMessageContent = (content) => {
    if (!content) return null
    const hasKaynak = content.includes('<KAYNAKLAR>')
    const mainText = hasKaynak ? content.split('<KAYNAKLAR>')[0] : content
    const sourcesMatch = hasKaynak ? content.match(/<KAYNAKLAR>([\s\S]*?)<\/KAYNAKLAR>/) : null

    let parsedSources = []
    if (sourcesMatch && sourcesMatch[1]) {
      const sourceBlocks = sourcesMatch[1].split('</KAYNAK>').filter(s => s.trim().startsWith('<KAYNAK'))
      parsedSources = sourceBlocks.map(block => {
        const titleMatch = block.match(/baslik="(.*?)"/)
        const title = titleMatch ? titleMatch[1] : 'Kaynak'
        const textStart = block.indexOf('>') + 1
        const text = block.substring(textStart).trim()
        return { title, text }
      })
    }

    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        <MarkdownBody>{mainText.trim()}</MarkdownBody>
        {parsedSources.length > 0 && (
          <div style={{ marginTop: '0.5rem' }}>
            <p style={{ fontWeight: 700, fontSize: '0.85rem', marginBottom: '0.75rem', color: 'var(--text-main)', borderTop: '1px solid rgba(0,0,0,0.1)', paddingTop: '1rem' }}>
              Kullanılan Kaynaklar (Tam Metin)
            </p>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              {parsedSources.map((src, i) => (
                <details key={i} style={{ background: 'var(--card-bg)', borderRadius: '8px', border: '1px solid var(--border)', overflow: 'hidden' }}>
                  <summary style={{ padding: '0.75rem 1rem', fontWeight: 600, fontSize: '0.8rem', cursor: 'pointer', outline: 'none', userSelect: 'none', color: 'var(--primary)', background: 'var(--bg)' }}>
                    {src.title}
                  </summary>
                  <div style={{ padding: '1rem', fontSize: '0.85rem', color: 'var(--text-main)', background: 'var(--card-bg)', whiteSpace: 'pre-wrap', maxHeight: '350px', overflowY: 'auto' }}>
                    <MarkdownBody>{src.text}</MarkdownBody>
                  </div>
                </details>
              ))}
            </div>
          </div>
        )}
      </div>
    )
  }

  const handleSaveResponse = async (query, responseText) => {
    try {
      const res = await fetch('/api/history/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...AUTH_HEADER() },
        body: JSON.stringify({ query, response: responseText })
      })
      if (res.ok) toast.success('Yanıt profilinize kaydedildi.')
      else toast.error('Kaydedilemedi.')
    } catch { toast.error('Sunucu hatası.') }
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
    } catch (e) { if (import.meta.env.DEV) console.error(e) }
  }

  const loadConversation = (convId) => {
    const msgs = historyRows.filter(c => c.conversation_id === convId).reverse()
    const newMessages = []
    msgs.forEach(m => {
      newMessages.push({ role: 'user', content: m.query, id: m.id })
      if (m.response) newMessages.push({ role: 'assistant', content: m.response, id: m.id })
    })
    setMessages(newMessages)
    setActiveConversationId(convId)
    setCurrentAnalysis('')
    setActiveTab('chat')
    setSidebarOpen(false)
  }

  const handleNewChat = () => {
    setMessages([])
    setActiveConversationId(null)
    setCurrentAnalysis('')
    setActiveTab('chat')
    setChatError(null)
  }

  useEffect(() => {
    if (pendingConvOpened.current) return
    let id = null
    try {
      id = sessionStorage.getItem('pendingConvId')
    } catch (_) {}
    if (!id || historyRows.length === 0) return
    pendingConvOpened.current = true
    try {
      sessionStorage.removeItem('pendingConvId')
    } catch (_) {}
    const msgs = historyRows.filter(r => r.conversation_id === id).reverse()
    const newMessages = []
    msgs.forEach(m => {
      newMessages.push({ role: 'user', content: m.query, id: m.id })
      if (m.response) newMessages.push({ role: 'assistant', content: m.response, id: m.id })
    })
    setMessages(newMessages)
    setActiveConversationId(id)
    setActiveTab('chat')
  }, [historyRows])

  useEffect(() => {
    const onKey = (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault()
        setPaletteOpen(true)
      }
      if ((e.ctrlKey || e.metaKey) && e.key === 'n') {
        e.preventDefault()
        handleNewChat()
      }
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter' && e.target?.tagName === 'INPUT') {
        e.preventDefault()
        document.querySelector('.chat-send-form')?.requestSubmit?.()
      }
      if (e.key === 'Escape') {
        setPaletteOpen(false)
        setReportOpen(false)
        setSidebarOpen(false)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const handleSendMessage = async (e) => {
    e.preventDefault()
    if (!input.trim() || loading) return

    const userMessage = input
    setInput('')
    setMessages(prev => [...prev, { role: 'user', content: userMessage }])
    setLoading(true)
    setStreamStarted(false)
    setChatError(null)
    setCurrentAnalysis('')
    setLiveAgentSteps([])

    let queryId = null
    let assistantMessage = ''
    let accumulatedSteps = []

    // Abort previous stream if any, then create a fresh controller.
    try { chatAbortRef.current?.abort() } catch (_) {}
    const controller = new AbortController()
    chatAbortRef.current = controller

    try {
      const kDepth = parseInt(localStorage.getItem('k_depth') || '5', 10)
      const response = await fetch('/api/chat/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...AUTH_HEADER() },
        body: JSON.stringify({
          query: userMessage,
          conversation_id: activeConversationId,
          role: selectedRole,
          k: kDepth,
        }),
        signal: controller.signal,
      })

      if (response.status === 401) {
        toast.error('Oturumunuz sona erdi — tekrar giriş yapın.')
        navigate('/login')
        return
      }
      if (response.status === 400) {
        // Backend signals missing LLM API key with a Turkish/English hint.
        const errBody = await response.json().catch(() => ({}))
        const detail = (errBody && (errBody.detail || errBody.message)) || ''
        const looksLikeMissingKey = /api[\s-]?key|anahtar|configure|yapıland(ı|i)r/i.test(String(detail))
        if (looksLikeMissingKey) {
          toast((t) => (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', maxWidth: 320 }}>
              <span style={{ fontWeight: 600 }}>
                ⚠️ Lütfen önce LLM API anahtarınızı ayarlayın.
              </span>
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                <button
                  onClick={() => { toast.dismiss(t.id); navigate('/settings#api-keys') }}
                  className="btn-primary"
                  style={{ padding: '0.4rem 0.8rem', fontSize: '0.85rem' }}
                >
                  Ayarlar'a Git
                </button>
                <button
                  onClick={() => toast.dismiss(t.id)}
                  style={{
                    padding: '0.4rem 0.8rem', fontSize: '0.85rem',
                    background: 'transparent', color: 'var(--text-muted)',
                    border: '1px solid var(--border)', borderRadius: 6, cursor: 'pointer',
                  }}
                >
                  Kapat
                </button>
              </div>
            </div>
          ), { duration: 10000, icon: '🔑' })
          setChatError(detail || 'LLM API anahtarı yapılandırılmamış.')
          return
        }
        throw new Error(detail || 'Sunucu hatası.')
      }
      if (!response.ok) {
        const errText = await response.text()
        throw new Error(errText || 'Sunucu hatası.')
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let sseBuffer = ''

      const applyAssistant = (text, steps, id, doneStream) => {
        setMessages(prev => {
          const history = [...prev]
          const last = history[history.length - 1]
          if (last?.role === 'assistant') {
            last.content = text
            last.id = id ?? last.id
            last.agentSteps = steps
            last.streaming = !doneStream
            return [...history]
          }
          return [...history, {
            role: 'assistant',
            content: text,
            id,
            agentSteps: steps,
            streaming: !doneStream,
          }]
        })
      }

      while (true) {
        const { value, done } = await reader.read()
        if (done) break

        sseBuffer += decoder.decode(value, { stream: true })
        const parts = sseBuffer.split('\n\n')
        sseBuffer = parts.pop() || ''

        for (const line of parts) {
          if (!line.startsWith('data: ')) continue
          try {
            const raw = line.slice(6).trim()
            if (!raw) continue
            const data = JSON.parse(raw)

            if (data.meta) {
              queryId = data.meta.query_id
              if (data.meta.conversation_id) setActiveConversationId(data.meta.conversation_id)
            } else if (data.error) {
              setChatError(data.error)
              toast.error(String(data.error))
            } else if (data.status) {
              setCurrentAnalysis(data.status)
            } else if (data.agent_step) {
              accumulatedSteps = [...accumulatedSteps, data.agent_step]
              setLiveAgentSteps([...accumulatedSteps])
            } else if (data.agent_steps_complete) {
              accumulatedSteps = data.agent_steps_complete
            } else if (data.answer_delta) {
              setStreamStarted(true)
              setLoading(false)
              assistantMessage += data.answer_delta
              applyAssistant(assistantMessage, accumulatedSteps, queryId, false)
            } else if (data.final_answer) {
              setStreamStarted(true)
              setLoading(false)
              assistantMessage = data.final_answer
              const finalSteps = accumulatedSteps
              applyAssistant(assistantMessage, finalSteps, queryId, true)
            }
          } catch {
            /* incomplete JSON */
          }
        }
      }

      if (queryId && assistantMessage) refreshData()
    } catch (err) {
      // Quietly ignore user/route-change aborts.
      if (err?.name === 'AbortError') return
      const msg = err?.message || 'Bilinmeyen hata'
      setChatError(msg)
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `Üzgünüm, bir sorun oluştu: ${msg}`,
      }])
      toast.error('Yanıt alınamadı')
    } finally {
      if (chatAbortRef.current === controller) chatAbortRef.current = null
      setLoading(false)
      setLiveAgentSteps([])
      setCurrentAnalysis('')
      setStreamStarted(false)
    }
  }

  const submitFeedbackReport = async () => {
    if (!reportTargetId) return
    try {
      const res = await fetch('/api/feedback/report', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...AUTH_HEADER() },
        body: JSON.stringify({
          message_id: reportTargetId,
          category: reportCategory,
          feedback_text: reportDetail || '',
        }),
      })
      if (res.ok) {
        toast.success('Bildiriminiz kaydedildi.')
        setReportOpen(false)
        setMessages(prev => prev.map(m => m.id === reportTargetId && m.role === 'assistant' ? { ...m, feedbackSubmitted: true } : m))
      } else toast.error('Kaydedilemedi.')
    } catch {
      toast.error('Sunucu hatası.')
    }
  }

  const exportChatText = () => {
    const lines = messages.map(m => `${m.role === 'user' ? 'Kullanıcı' : 'Asistan'}: ${m.content}`)
    navigator.clipboard.writeText(lines.join('\n\n')).then(() => toast.success('Panoya kopyalandı')).catch(() => toast.error('Kopyalanamadı'))
  }

  const printChat = () => {
    window.print()
  }

  const handleFileUpload = async (e) => {
    const file = e.target.files[0]
    if (!file) return

    // Strict client-side validation: filename + MIME + size
    const isPdf =
      file.name.toLowerCase().endsWith('.pdf') ||
      file.type === 'application/pdf'
    if (!isPdf) {
      toast.error('Sadece PDF dosyaları yüklenebilir.')
      if (fileInputRef.current) fileInputRef.current.value = ''
      return
    }
    const MAX_BYTES = 10 * 1024 * 1024 // 10 MB
    if (file.size > MAX_BYTES) {
      toast.error(`Dosya çok büyük (${(file.size / 1024 / 1024).toFixed(1)} MB). En fazla 10 MB.`)
      if (fileInputRef.current) fileInputRef.current.value = ''
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
      if (res.status === 401) {
        toast.error('Oturumunuz sona erdi — tekrar giriş yapın.')
        navigate('/login')
        return
      }
      if (res.ok) {
        setMessages(prev => [...prev, { role: 'assistant', content: `📄 **${file.name}** başarıyla yüklendi ve analiz edildi. Artık bu döküman hakkında soru sorabilirsiniz.` }])
        toast.success('PDF yüklendi.')
      } else {
        const err = await res.json().catch(() => ({}))
        toast.error(`Yükleme hatası: ${err.detail || 'Bilinmeyen hata'}`)
      }
    } catch (err) {
      toast.error('Sunucuya bağlanılamadı.')
    } finally {
      setUploading(false)
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }

  const filteredSummaries = convSummaries.filter(c =>
    !convSearch.trim() ||
    (c.title || '').toLowerCase().includes(convSearch.toLowerCase()) ||
    (c.conversation_id || '').includes(convSearch.trim())
  )
  const mappedForSidebar = filteredSummaries.map(c => ({
    ...c,
    query: c.title || 'Sohbet',
    created_at: c.updated_at,
    conversation_id: c.conversation_id,
  }))
  const conversationGroups = groupByDate(mappedForSidebar.length ? mappedForSidebar : [])

  const deleteConv = async (e, convId) => {
    e.stopPropagation()
    if (!window.confirm('Bu konuşmayı silmek istediğinize emin misiniz?')) return
    try {
      const res = await fetch(`/api/conversations/${encodeURIComponent(convId)}`, {
        method: 'DELETE',
        headers: AUTH_HEADER(),
      })
      if (res.ok) {
        toast.success('Konuşma silindi')
        if (activeConversationId === convId) handleNewChat()
        refreshData()
      } else toast.error('Silinemedi')
    } catch {
      toast.error('Sunucu hatası')
    }
  }

  const renameConv = async (e, convId, currentTitle) => {
    e.stopPropagation()
    const t = window.prompt('Yeni başlık', currentTitle || '')
    if (t === null) return
    try {
      const res = await fetch(`/api/conversations/${encodeURIComponent(convId)}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json', ...AUTH_HEADER() },
        body: JSON.stringify({ title: t }),
      })
      if (res.ok) {
        toast.success('Başlık güncellendi')
        refreshData()
      } else toast.error('Güncellenemedi')
    } catch {
      toast.error('Sunucu hatası')
    }
  }

  const toggleFavorite = async (e, convId, fav) => {
    e.stopPropagation()
    try {
      await fetch(`/api/conversations/${encodeURIComponent(convId)}/favorite`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json', ...AUTH_HEADER() },
        body: JSON.stringify({ favorited: !fav }),
      })
      refreshData()
    } catch {
      toast.error('Sunucu hatası')
    }
  }

  const quickList = QUICK_BY_ROLE[selectedRole] || QUICK_BY_ROLE.PATIENT

  return (
    <div className="dashboard-layout" style={{ display: 'flex', height: '100vh', overflow: 'hidden', flexDirection: 'column' }}>
      <Toaster position="bottom-right" toastOptions={{ duration: 3500 }} />
      <CommandPalette open={paletteOpen} onOpenChange={setPaletteOpen} user={user} />

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

      {sidebarOpen && (
        <div
          role="presentation"
          className="sidebar-overlay"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      <div style={{ display: 'flex', flex: 1, overflow: 'hidden', position: 'relative' }}>
        {/* Sidebar */}
        <aside
          className={`sidebar glass chat-sidebar ${sidebarOpen ? 'chat-sidebar-open' : ''}`}
          style={{ width: '300px', borderRight: '1px solid var(--border)', display: 'flex', flexDirection: 'column', zIndex: 50 }}
        >
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
              <NavLink
                to="/policies"
                style={({ isActive }) => ({
                  textDecoration: 'none',
                  color: 'var(--text-main)',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem',
                  padding: '0.6rem 0.75rem',
                  borderRadius: '8px',
                  fontSize: '0.85rem',
                  fontWeight: isActive ? 700 : 600,
                  borderLeft: isActive ? '3px solid var(--primary)' : '3px solid transparent',
                  background: isActive ? 'rgba(99,102,241,0.1)' : 'transparent',
                })}
              >
                <BookOpen size={14} /> SUT Mevzuat Tarayıcısı
              </NavLink>
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
              <input
                type="search"
                placeholder="Geçmişte ara…"
                value={convSearch}
                onChange={e => setConvSearch(e.target.value)}
                aria-label="Geçmişte ara"
                style={{ width: '100%', padding: '0.45rem 0.6rem', marginBottom: '0.5rem', borderRadius: '8px', border: '1px solid var(--border)', fontSize: '0.8rem', boxSizing: 'border-box' }}
              />
              {mappedForSidebar.length === 0 ? (
                <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', padding: '0.5rem', textAlign: 'center' }}>Henüz konuşma yok.</p>
              ) : (
                Object.entries(conversationGroups).map(([group, items]) => (
                  <div key={group}>
                    <p style={{ fontSize: '0.65rem', color: 'var(--text-muted)', padding: '0.25rem 0.75rem', textTransform: 'uppercase', letterSpacing: '0.5px', marginTop: '0.5rem' }}>{group}</p>
                    {items.slice(0, 10).map((conv, i) => (
                      <div
                        key={conv.conversation_id || i}
                        style={{
                          display: 'flex', alignItems: 'center', gap: '0.25rem',
                          borderRadius: '8px', background: activeConversationId === conv.conversation_id ? 'var(--bg)' : 'transparent',
                        }}
                      >
                        <button
                          type="button"
                          onClick={() => loadConversation(conv.conversation_id)}
                          style={{ flex: 1, textAlign: 'left', padding: '0.6rem 0.5rem', borderRadius: '8px', border: 'none', background: 'transparent', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '0.5rem', minWidth: 0 }}
                        >
                          <Clock size={12} color="var(--text-muted)" style={{ flexShrink: 0 }} />
                          <span style={{ fontSize: '0.82rem', color: 'var(--text-main)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                            {(conv.query || '').slice(0, 36)}{(conv.query || '').length > 36 ? '…' : ''}
                          </span>
                        </button>
                        <button type="button" aria-label="Yıldızla" onClick={e => toggleFavorite(e, conv.conversation_id, conv.favorited)} style={{ background: 'none', border: 'none', cursor: 'pointer', padding: '0.2rem', color: conv.favorited ? '#f59e0b' : 'var(--text-muted)' }}>
                          <Star size={14} fill={conv.favorited ? '#f59e0b' : 'none'} />
                        </button>
                        <button type="button" aria-label="Yeniden adlandır" onClick={e => renameConv(e, conv.conversation_id, conv.title)} style={{ background: 'none', border: 'none', cursor: 'pointer', padding: '0.2rem', color: 'var(--text-muted)' }}>
                          <Pencil size={14} />
                        </button>
                        <button type="button" aria-label="Sil" onClick={e => deleteConv(e, conv.conversation_id)} style={{ background: 'none', border: 'none', cursor: 'pointer', padding: '0.2rem', color: 'var(--text-muted)' }}>
                          <Trash2 size={14} />
                        </button>
                      </div>
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
              {quickList.map(q => (
                <button key={q} type="button" onClick={() => { setInput(q); setSidebarOpen(false) }} style={{ width: '100%', textAlign: 'left', padding: '0.6rem 0.75rem', borderRadius: '8px', border: 'none', background: 'transparent', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.82rem', color: 'var(--text-main)' }}>
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
        <main className="chat-main-area" style={{ flex: 1, display: 'flex', flexDirection: 'column', background: 'var(--bg)', position: 'relative', overflow: 'hidden', minWidth: 0 }}>
          <header className="glass chat-top-header no-print" style={{ padding: '0.75rem 1rem', borderBottom: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', zIndex: 10, flexShrink: 0, gap: '0.75rem', flexWrap: 'wrap' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
              <button type="button" className="mobile-only menu-btn" aria-label="Menüyü aç" onClick={() => setSidebarOpen(o => !o)} style={{ background: 'var(--bg)', border: '1px solid var(--border)', borderRadius: '8px', padding: '0.45rem', cursor: 'pointer' }}>
                <Menu size={20} />
              </button>
              <nav aria-label="Breadcrumb" style={{ fontSize: '0.78rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                <Link to="/" style={{ color: 'var(--primary)', fontWeight: 600 }}>Ana sayfa</Link>
                <span aria-hidden>/</span>
                <span style={{ color: 'var(--text-main)', fontWeight: 600 }}>{activeTab === 'chat' ? 'Sohbet' : 'Bilgi Grafiği'}</span>
              </nav>
              {activeTab === 'chat' && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', background: 'var(--bg)', padding: '0.2rem 0.5rem', borderRadius: '8px', border: '1px solid var(--border)' }}>
                  <span style={{ fontSize: '0.7rem', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Persona</span>
                  <select
                    value={selectedRole}
                    onChange={e => setSelectedRole(e.target.value)}
                    aria-label="Persona seçimi"
                    style={{ background: 'transparent', border: 'none', fontSize: '0.8rem', fontWeight: 600, color: 'var(--primary)', cursor: 'pointer', outline: 'none', maxWidth: '160px' }}
                  >
                    <option value="PATIENT">Vatandaş</option>
                    <option value="DOCTOR">Hekim</option>
                    <option value="ECZACI">Eczacı</option>
                    <option value="HASTANE_YONETICISI">Hastane yöneticisi</option>
                    <option value="ADMIN">SGK / Yönetici</option>
                  </select>
                </div>
              )}
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', flexWrap: 'wrap' }}>
              <button type="button" aria-label="Komut paleti" className="no-print" onClick={() => setPaletteOpen(true)} style={{ background: 'var(--bg)', border: '1px solid var(--border)', borderRadius: '8px', padding: '0.45rem', cursor: 'pointer', color: 'var(--text-muted)' }}>
                <Search size={18} />
              </button>
              {announcement && (
                <span title={announcement.message} style={{ position: 'relative', color: '#ef4444' }} aria-label="Duyuru">
                  <Bell size={18} />
                  <span style={{ position: 'absolute', top: -2, right: -2, width: 8, height: 8, borderRadius: '50%', background: '#ef4444' }} />
                </span>
              )}
              {modelLabel && (
                <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                  <Cpu size={14} /> {modelLabel}
                </span>
              )}
              <Link to="/settings" aria-label="Ayarlar" className="no-print" style={{ color: 'var(--text-muted)' }}>
                <Settings size={18} />
              </Link>
              <span style={{ fontSize: '0.75rem', color: '#10b981', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                <span style={{ width: 8, height: 8, borderRadius: '50%', background: '#10b981' }} aria-hidden /> Çevrimiçi
              </span>
            </div>
          </header>

          {activeTab === 'graph' ? (
            <div style={{ flex: 1, position: 'relative' }}>
              <KnowledgeGraph />
            </div>
          ) : (
            <>
              {chatError && (
                <div role="alert" style={{ padding: '0.65rem 1rem', background: '#fef2f2', color: '#991b1b', fontSize: '0.85rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexShrink: 0, gap: '1rem' }}>
                  <span style={{ flex: 1, minWidth: 0 }}>{chatError}</span>
                  <div style={{ display: 'flex', gap: '0.5rem', flexShrink: 0 }}>
                    {(() => {
                      const lastUser = [...messages].reverse().find(m => m.role === 'user')
                      if (!lastUser || loading) return null
                      return (
                        <button
                          type="button"
                          onClick={() => {
                            setChatError(null)
                            setInput(lastUser.content)
                            // Defer to give state a tick
                            setTimeout(() => {
                              document.querySelector('.chat-send-form')?.requestSubmit?.()
                            }, 0)
                          }}
                          style={{ background: 'transparent', border: '1px solid #b91c1c', color: '#991b1b', borderRadius: 6, padding: '0.25rem 0.6rem', cursor: 'pointer', fontSize: '0.78rem', fontWeight: 600 }}
                        >
                          Yeniden dene
                        </button>
                      )
                    })()}
                    <button type="button" aria-label="Hata mesajını kapat" onClick={() => setChatError(null)} style={{ background: 'transparent', border: 'none', cursor: 'pointer', fontSize: '1.1rem', lineHeight: 1, color: '#991b1b' }}>×</button>
                  </div>
                </div>
              )}
              {activeTab === 'chat' && messages.length > 0 && (
                <div className="no-print export-toolbar" style={{ padding: '0.35rem 1rem', borderBottom: '1px solid var(--border)', display: 'flex', gap: '0.5rem', justifyContent: 'flex-end', flexShrink: 0 }}>
                  <button type="button" className="btn-secondary" style={{ fontSize: '0.75rem', padding: '0.35rem 0.65rem' }} onClick={exportChatText}>Metni kopyala</button>
                  <button type="button" className="btn-secondary" style={{ fontSize: '0.75rem', padding: '0.35rem 0.65rem' }} onClick={printChat}>Yazdır / PDF</button>
                </div>
              )}
              <div className="chat-scroll print-messages" style={{ flex: 1, overflowY: 'auto', padding: '2rem clamp(1rem, 5vw, 12%)' }} aria-live="polite">
                {messages.length === 0 && !currentAnalysis && (
                  <div style={{ height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', textAlign: 'center', padding: '1rem' }}>
                    <div style={{ width: '80px', height: '80px', borderRadius: '24px', background: 'var(--card-bg)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--accent)', marginBottom: '1.5rem', boxShadow: '0 20px 25px -5px rgba(0,0,0,0.1)' }}>
                      <Bot size={48} aria-hidden />
                    </div>
                    <h2 style={{ fontSize: 'clamp(1.25rem, 4vw, 2rem)', fontWeight: 700, marginBottom: '0.5rem' }}>Merhaba, {user.username}</h2>
                    <p style={{ color: 'var(--text-muted)', maxWidth: '520px', marginBottom: '1rem' }}>
                      SUT mevzuatında soru sorun; her yanıtta kaynak referanslarına gidebilirsiniz. Personanızı seçerek hızlı soruları değiştirin.
                    </p>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', justifyContent: 'center', maxWidth: '560px', marginBottom: '1.5rem' }}>
                      {quickList.slice(0, 4).map(q => (
                        <button key={q} type="button" className="btn-secondary" style={{ padding: '0.55rem 0.9rem', borderRadius: '10px', fontSize: '0.85rem', cursor: 'pointer' }} onClick={() => setInput(q)}>
                          {q}
                        </button>
                      ))}
                    </div>
                    <ul style={{ textAlign: 'left', color: 'var(--text-muted)', fontSize: '0.85rem', maxWidth: '420px', lineHeight: 1.7 }}>
                      <li>PDF rapor yükleyebilirsiniz</li>
                      <li>Her yanıt kaynak ve madde referanslarıyla ilişkilidir</li>
                      <li>Yapay zeka hata yapabilir; kritik işlemlerde resmi metni doğrulayın</li>
                    </ul>
                  </div>
                )}

                {messages.map((msg, i) => {
                  const isAssistant = msg.role === 'assistant'
                  // Compute once per message render instead of twice.
                  const refs = isAssistant && msg.content
                    ? extractSourceRefs(msg.content.replace(/<KAYNAKLAR>[\s\S]*$/i, ''))
                    : []
                  return (
                  <div key={msg.id || `m-${i}`} style={{ marginBottom: '2.5rem', animation: 'fadeIn 0.5s ease' }}>
                    <div style={{ display: 'flex', gap: '1rem', justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start' }}>
                      {isAssistant && (
                        <div style={{ width: '36px', height: '36px', borderRadius: '10px', background: 'var(--primary)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'white', flexShrink: 0 }} aria-hidden>
                          <Bot size={18} />
                        </div>
                      )}
                      <div style={{ maxWidth: 'min(80%, 720px)' }}>
                        <div className={msg.role === 'user' ? 'btn-primary' : 'premium-card'} style={{ padding: '1rem 1.25rem', borderRadius: msg.role === 'user' ? '20px 20px 4px 20px' : '4px 20px 20px 20px', fontSize: '0.95rem' }}>
                          {msg.role === 'user' ? (
                            <div style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</div>
                          ) : (
                            renderMessageContent(msg.content)
                          )}

                          {isAssistant && refs.length > 0 && (
                            <div style={{ marginTop: '0.75rem', paddingTop: '0.75rem', borderTop: '1px dashed var(--border)' }}>
                              <p style={{ fontSize: '0.72rem', fontWeight: 700, color: 'var(--text-muted)', marginBottom: '0.5rem' }}>Kaynaklar</p>
                              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem' }}>
                                {refs.map(ref => (
                                  <Link
                                    key={ref.label}
                                    to={`/policies?highlight=${encodeURIComponent(ref.hrefQuery)}`}
                                    style={{
                                      fontSize: '0.75rem', fontWeight: 600, padding: '0.25rem 0.65rem', borderRadius: '999px',
                                      background: 'rgba(99,102,241,0.12)', color: 'var(--primary)', textDecoration: 'none', border: '1px solid rgba(99,102,241,0.25)',
                                    }}
                                  >
                                    {ref.label}
                                  </Link>
                                ))}
                              </div>
                            </div>
                          )}

                          {/* Agent Trace — expandable per message */}
                          {msg.role === 'assistant' && msg.agentSteps && msg.agentSteps.length > 0 && (
                            <AgentTrace steps={msg.agentSteps} />
                          )}
                          
                          {msg.role === 'assistant' && !msg.streaming && msg.content && (
                            <div style={{ marginTop: '1rem', paddingTop: '0.75rem', borderTop: '1px solid rgba(0,0,0,0.06)', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.6rem', alignItems: 'center' }}>
                                  <button type="button" aria-label="Metni kopyala" className="msg-action-btn" onClick={() => { navigator.clipboard.writeText(msg.content); toast.success('Kopyalandı') }} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.25rem', fontSize: '0.75rem', color: 'var(--primary)', background: 'transparent', border: 'none', cursor: 'pointer', fontWeight: 600 }}>
                                    <Copy size={14} /> Kopyala
                                  </button>
                                  <button type="button" className="msg-action-btn" onClick={() => { const q = messages.slice(0, i + 1).map(x => `${x.role}: ${x.content}`).join('\n\n'); navigator.clipboard.writeText(q); toast.success('Sohbet özeti kopyalandı') }} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.25rem', fontSize: '0.75rem', color: 'var(--primary)', background: 'transparent', border: 'none', cursor: 'pointer', fontWeight: 600 }}>
                                    Paylaş
                                  </button>
                                  <button type="button" aria-label="Hata bildir" className="msg-action-btn" disabled={!msg.id} onClick={() => { setReportTargetId(msg.id); setReportDetail(''); setReportCategory('wrong_info'); setReportOpen(true) }} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.25rem', fontSize: '0.75rem', color: '#f59e0b', background: 'transparent', border: 'none', cursor: msg.id ? 'pointer' : 'not-allowed', fontWeight: 600 }}>
                                    <Flag size={14} /> Hata bildir
                                  </button>
                                  <button type="button" className="msg-action-btn" onClick={() => {
                                    let queryStr = 'Sorgu bulunamadı'
                                    for (let j = i - 1; j >= 0; j--) {
                                      if (messages[j].role === 'user') { queryStr = messages[j].content; break }
                                    }
                                    handleSaveResponse(queryStr, msg.content)
                                  }} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.25rem', fontSize: '0.75rem', color: 'var(--primary)', background: 'transparent', border: 'none', cursor: 'pointer', fontWeight: 600 }}>
                                    <Bookmark size={14} /> Kaydet
                                  </button>

                                  {msg.id && !msg.feedbackSubmitted && (
                                    <span style={{ display: 'inline-flex', gap: '0.5rem', borderLeft: '1px solid var(--border)', paddingLeft: '0.75rem' }}>
                                      <button type="button" aria-label="Doğru yanıt" onClick={() => handleFeedback(msg.id, 5, true)} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.25rem', fontSize: '0.75rem', color: '#10b981', background: 'transparent', border: 'none', cursor: 'pointer', fontWeight: 600 }}>
                                        <ThumbsUp size={14} /> İyi
                                      </button>
                                      <button type="button" aria-label="Yetersiz yanıt" onClick={() => handleFeedback(msg.id, 1, false)} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.25rem', fontSize: '0.75rem', color: '#ef4444', background: 'transparent', border: 'none', cursor: 'pointer', fontWeight: 600 }}>
                                        <ThumbsDown size={14} /> Zayıf
                                      </button>
                                    </span>
                                  )}
                                  {msg.feedbackSubmitted && (
                                    <span style={{ fontSize: '0.7rem', color: '#10b981', borderLeft: '1px solid var(--border)', paddingLeft: '0.75rem' }}>Geri bildirim kaydedildi</span>
                                  )}
                              </div>
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
                  )
                })}

                {loading && !streamStarted && (
                  <div style={{ display: 'flex', gap: '1rem', marginBottom: '2rem' }} aria-busy="true" aria-live="polite">
                    <div style={{ width: '36px', height: '36px', borderRadius: '10px', background: 'var(--primary)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'white' }}>
                      <Bot size={18} className="spin" aria-hidden />
                    </div>
                    <div style={{ flex: 1 }}>
                      {currentAnalysis && (
                        <div style={{ padding: '0.75rem 1rem', background: 'var(--card-bg)', borderRadius: '12px', fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                          <span style={{ animation: 'spin 1.5s linear infinite', display: 'inline-block' }} aria-hidden>⚙️</span>
                          <span style={{ fontWeight: 500 }}>{currentAnalysis}</span>
                        </div>
                      )}
                      {liveAgentSteps.length > 0 && (
                        <AgentTrace steps={liveAgentSteps} live={true} />
                      )}
                      <div className="premium-card typing-bubble" style={{ padding: '0.75rem 1rem', marginTop: '0.5rem', fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                        <span className="typing-dots" aria-hidden><span /><span /><span /></span>
                        {' '}Yanıt hazırlanıyor…
                      </div>
                      {!currentAnalysis && liveAgentSteps.length === 0 && (
                        <div className="loading-skeleton" style={{ height: '48px', width: '100%', maxWidth: '220px', marginTop: '0.5rem', borderRadius: '12px' }} />
                      )}
                    </div>
                  </div>
                )}
                <div ref={chatEndRef} />
              </div>

              <div className="chat-input-bar no-print" style={{ padding: '1.5rem clamp(1rem, 5vw, 12%)', paddingBottom: 'max(1.5rem, env(safe-area-inset-bottom))', background: 'var(--card-bg)', borderTop: '1px solid var(--border)', flexShrink: 0 }}>
                <form className="chat-send-form" onSubmit={handleSendMessage} style={{ display: 'flex', gap: '1rem', position: 'relative' }}>
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
                    aria-label="Sohbet mesajı"
                    style={{ paddingRight: '4rem', height: '56px', fontSize: '1rem', boxShadow: '0 10px 15px -3px rgba(0,0,0,0.05)' }}
                    disabled={loading}
                  />
                  <button type="submit" aria-label="Gönder" disabled={loading || !input.trim()} style={{ position: 'absolute', right: '8px', top: '8px', bottom: '8px', width: '40px', borderRadius: '10px', display: 'flex', alignItems: 'center', justifyContent: 'center', background: input.trim() ? 'var(--accent)' : 'var(--border)', color: 'white', border: 'none', cursor: input.trim() ? 'pointer' : 'not-allowed' }}>
                    <Send size={20} aria-hidden />
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

      <nav className="mobile-tab-bar no-print" aria-label="Mobil navigasyon">
        <Link
          to="/"
          aria-current={location.pathname === '/' ? 'page' : undefined}
          className={location.pathname === '/' ? 'active' : ''}
          onClick={() => setActiveTab('chat')}
          style={{ flex: 1, textAlign: 'center', padding: '0.65rem', textDecoration: 'none', color: location.pathname === '/' ? 'var(--primary)' : 'var(--text-muted)', fontSize: '0.72rem', fontWeight: 600 }}
        >Sohbet</Link>
        <Link
          to="/policies"
          aria-current={location.pathname === '/policies' ? 'page' : undefined}
          className={location.pathname === '/policies' ? 'active' : ''}
          style={{ flex: 1, textAlign: 'center', padding: '0.65rem', textDecoration: 'none', color: location.pathname === '/policies' ? 'var(--primary)' : 'var(--text-muted)', fontSize: '0.72rem', fontWeight: 600 }}
        >Tarayıcı</Link>
        <Link
          to="/profile"
          aria-current={location.pathname === '/profile' ? 'page' : undefined}
          className={location.pathname === '/profile' ? 'active' : ''}
          style={{ flex: 1, textAlign: 'center', padding: '0.65rem', textDecoration: 'none', color: location.pathname === '/profile' ? 'var(--primary)' : 'var(--text-muted)', fontSize: '0.72rem', fontWeight: 600 }}
        >Geçmiş</Link>
        <Link
          to="/settings"
          aria-current={location.pathname === '/settings' ? 'page' : undefined}
          className={location.pathname === '/settings' ? 'active' : ''}
          style={{ flex: 1, textAlign: 'center', padding: '0.65rem', textDecoration: 'none', color: location.pathname === '/settings' ? 'var(--primary)' : 'var(--text-muted)', fontSize: '0.72rem', fontWeight: 600 }}
        >Ayarlar</Link>
      </nav>

      <WelcomeModal open={welcomeOpen} username={user?.username} onDismiss={dismissWelcome} />

      {reportOpen && (
        <FocusTrap focusTrapOptions={{ clickOutsideDeactivates: true, escapeDeactivates: false }}>
          <div role="dialog" aria-modal="true" aria-labelledby="report-title" style={{ position: 'fixed', inset: 0, zIndex: 3000, background: 'rgba(15,23,42,0.45)', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '1rem' }} onClick={() => setReportOpen(false)}>
            <div className="premium-card" style={{ maxWidth: '420px', width: '100%', padding: '1.25rem' }} onClick={e => e.stopPropagation()}>
              <h3 id="report-title" style={{ marginTop: 0 }}>Hata bildir</h3>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Değerlendirme sürecine iletilir.</p>
              <label htmlFor="report-cat" style={{ fontSize: '0.8rem', fontWeight: 600 }}>Kategori</label>
              <select id="report-cat" value={reportCategory} onChange={e => setReportCategory(e.target.value)} style={{ width: '100%', marginTop: '0.35rem', marginBottom: '0.75rem', padding: '0.5rem', borderRadius: '8px', border: '1px solid var(--border)' }}>
                <option value="wrong_info">Yanlış bilgi</option>
                <option value="missing_source">Eksik kaynak</option>
                <option value="other">Diğer</option>
              </select>
              <label htmlFor="report-txt" style={{ fontSize: '0.8rem', fontWeight: 600 }}>Açıklama</label>
              <textarea id="report-txt" value={reportDetail} onChange={e => setReportDetail(e.target.value)} rows={3} style={{ width: '100%', marginTop: '0.35rem', padding: '0.5rem', borderRadius: '8px', border: '1px solid var(--border)', resize: 'vertical' }} />
              <div style={{ display: 'flex', gap: '0.5rem', marginTop: '1rem', justifyContent: 'flex-end' }}>
                <button type="button" className="btn-secondary" onClick={() => setReportOpen(false)}>İptal</button>
                <button type="button" className="btn-primary" onClick={() => submitFeedbackReport()}>Gönder</button>
              </div>
            </div>
          </div>
        </FocusTrap>
      )}

      <style>{`
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .loading-skeleton { background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%); background-size: 200% 100%; animation: loading 1.5s infinite; }
        @keyframes loading { from { background-position: 200% 0; } to { background-position: -200% 0; } }
        .spin { animation: spin 2s linear infinite; }
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        .typing-dots span {
          display: inline-block; width: 6px; height: 6px; margin: 0 2px; border-radius: 50%;
          background: var(--text-muted); animation: bounce 1.2s infinite ease-in-out both;
        }
        .typing-dots span:nth-child(1) { animation-delay: 0s; }
        .typing-dots span:nth-child(2) { animation-delay: 0.15s; }
        .typing-dots span:nth-child(3) { animation-delay: 0.3s; }
        @keyframes bounce { 0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; } 40% { transform: scale(1); opacity: 1; } }
        .md-body p { margin: 0.5em 0; }
        .md-body ul, .md-body ol { margin: 0.5em 0; padding-left: 1.25rem; }
        .md-body code { background: rgba(0,0,0,0.06); padding: 0.1em 0.35em; border-radius: 4px; font-size: 0.9em; }
        .msg-action-btn { opacity: 0.88; }
        .msg-action-btn:hover { opacity: 1; }
        .sidebar-overlay { display: none; }
        .mobile-tab-bar { display: none; }
        .mobile-only.menu-btn { display: none !important; }
        @media (max-width: 768px) {
          .sidebar-overlay { display: block; position: fixed; inset: 0; z-index: 40; background: rgba(15,23,42,0.35); }
          .mobile-only.menu-btn { display: inline-flex !important; align-items: center; justify-content: center; }
          .chat-sidebar {
            position: fixed; left: 0; top: 0; bottom: 0; height: 100vh; max-width: min(300px, 88vw);
            transform: translateX(-102%); transition: transform 0.2s ease; z-index: 50; box-shadow: 4px 0 24px rgba(0,0,0,0.12);
          }
          .chat-sidebar-open { transform: translateX(0); }
          .chat-main-area { min-height: 0; }
          .mobile-tab-bar {
            display: flex; position: fixed; left: 0; right: 0; bottom: 0; z-index: 30;
            background: var(--card-bg); border-top: 1px solid var(--border);
            padding-bottom: env(safe-area-inset-bottom);
          }
          .mobile-tab-bar a.active { color: var(--primary) !important; }
          .dashboard-layout { padding-bottom: 56px; }
          .export-toolbar { display: none; }
        }
        @media print {
          .no-print { display: none !important; }
          .print-messages { padding: 1rem !important; }
        }
      `}</style>
    </div>
  )
}

export default ChatDashboard
