import React, { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import FocusTrap from 'focus-trap-react'
import { Command } from 'cmdk'
import { MessageSquare, BookOpen, Settings, User, Shield } from 'lucide-react'

const AUTH = () => ({ Authorization: `Bearer ${localStorage.getItem('token')}` })

export function CommandPalette({ open, onOpenChange, user }) {
  const navigate = useNavigate()
  const [convs, setConvs] = useState([])

  useEffect(() => {
    if (!open) return
    fetch('/api/conversations?limit=15', { headers: AUTH() })
      .then(r => (r.ok ? r.json() : { conversations: [] }))
      .then(d => setConvs(d.conversations || []))
      .catch(() => setConvs([]))
  }, [open])

  if (!open) return null

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label="Komut paleti"
      style={{
        position: 'fixed', inset: 0, zIndex: 2000,
        background: 'rgba(15,23,42,0.45)', display: 'flex', alignItems: 'flex-start', justifyContent: 'center', paddingTop: '12vh',
      }}
      onClick={() => onOpenChange(false)}
    >
      <FocusTrap focusTrapOptions={{ clickOutsideDeactivates: true, escapeDeactivates: false }}>
        <div onClick={e => e.stopPropagation()} style={{ width: 'min(520px, 94vw)' }}>
        <Command
          className="cmd-root"
          style={{
            background: 'var(--card-bg)', borderRadius: '12px', border: '1px solid var(--border)',
            boxShadow: '0 25px 50px -12px rgba(0,0,0,0.25)', overflow: 'hidden',
          }}
          label="Komut paleti"
        >
          <Command.Input
            placeholder="Sayfa veya sohbet ara…"
            style={{
              width: '100%', padding: '1rem 1.25rem', fontSize: '1rem', border: 'none', borderBottom: '1px solid var(--border)',
              outline: 'none', background: 'var(--card-bg)', color: 'var(--text-main)', boxSizing: 'border-box',
            }}
          />
          <Command.List style={{ maxHeight: 'min(60vh, 400px)', overflowY: 'auto', padding: '0.5rem' }}>
            <Command.Empty style={{ padding: '1rem', fontSize: '0.85rem', color: 'var(--text-muted)' }}>Sonuç yok.</Command.Empty>
            <Command.Group heading="Sayfalar" style={{ fontSize: '0.65rem', textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-muted)', padding: '0.5rem 0.75rem' }}>
              <Command.Item
                onSelect={() => { navigate('/'); onOpenChange(false) }}
                style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.65rem 0.75rem', borderRadius: '8px', cursor: 'pointer' }}
              >
                <MessageSquare size={16} /> Sohbet
              </Command.Item>
              <Command.Item
                onSelect={() => { navigate('/policies'); onOpenChange(false) }}
                style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.65rem 0.75rem', borderRadius: '8px', cursor: 'pointer' }}
              >
                <BookOpen size={16} /> Mevzuat Tarayıcısı
              </Command.Item>
              <Command.Item
                onSelect={() => { navigate('/settings'); onOpenChange(false) }}
                style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.65rem 0.75rem', borderRadius: '8px', cursor: 'pointer' }}
              >
                <Settings size={16} /> Ayarlar
              </Command.Item>
              <Command.Item
                onSelect={() => { navigate('/profile'); onOpenChange(false) }}
                style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.65rem 0.75rem', borderRadius: '8px', cursor: 'pointer' }}
              >
                <User size={16} /> Profil
              </Command.Item>
              {user?.role === 'admin' && (
                <Command.Item
                  onSelect={() => { navigate('/admin'); onOpenChange(false) }}
                  style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.65rem 0.75rem', borderRadius: '8px', cursor: 'pointer' }}
                >
                  <Shield size={16} /> Admin Paneli
                </Command.Item>
              )}
            </Command.Group>
            {convs.length > 0 && (
              <Command.Group heading="Son konuşmalar" style={{ fontSize: '0.65rem', textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-muted)', padding: '0.5rem 0.75rem' }}>
                {convs.map(c => (
                  <Command.Item
                    key={c.conversation_id}
                    value={c.title || c.conversation_id}
                    onSelect={() => {
                      try {
                        sessionStorage.setItem('pendingConvId', c.conversation_id)
                      } catch (_) {}
                      navigate('/')
                      onOpenChange(false)
                    }}
                    style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.65rem 0.75rem', borderRadius: '8px', cursor: 'pointer' }}
                  >
                    <MessageSquare size={16} /> {(c.title || 'Sohbet').slice(0, 48)}
                  </Command.Item>
                ))}
              </Command.Group>
            )}
          </Command.List>
          <div style={{ padding: '0.5rem 1rem', fontSize: '0.72rem', color: 'var(--text-muted)', borderTop: '1px solid var(--border)' }}>
            Esc kapat · Ctrl+K
          </div>
        </Command>
        </div>
      </FocusTrap>
    </div>
  )
}
