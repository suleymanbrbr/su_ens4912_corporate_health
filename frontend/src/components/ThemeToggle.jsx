import React, { useEffect, useState } from 'react'
import { Sun, Moon } from 'lucide-react'

function ThemeToggle() {
  const [theme, setTheme] = useState('light')

  useEffect(() => {
    // Check initial preference from localStorage or system OS
    const saved = localStorage.getItem('theme')
    if (saved) {
      setTheme(saved)
      document.documentElement.setAttribute('data-theme', saved)
    } else {
      const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches
      if (prefersDark) {
        setTheme('dark')
        document.documentElement.setAttribute('data-theme', 'dark')
      }
    }
  }, [])

  const toggleTheme = () => {
    const newTheme = theme === 'light' ? 'dark' : 'light'
    setTheme(newTheme)
    localStorage.setItem('theme', newTheme)
    document.documentElement.setAttribute('data-theme', newTheme)
  }

  return (
    <button 
      onClick={toggleTheme}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: '0.5rem',
        padding: '0.75rem',
        background: 'transparent',
        border: 'none',
        color: 'var(--text-muted)',
        cursor: 'pointer',
        fontSize: '0.9rem',
        borderRadius: '8px',
        width: '100%',
        transition: 'all 0.2s',
      }}
      onMouseOver={e => {
        e.currentTarget.style.background = 'var(--hover-bg, rgba(0,0,0,0.05))'
        e.currentTarget.style.color = 'var(--text-main)'
      }}
      onMouseOut={e => {
        e.currentTarget.style.background = 'transparent'
        e.currentTarget.style.color = 'var(--text-muted)'
      }}
    >
      {theme === 'light' ? <Moon size={18} /> : <Sun size={18} />}
      {theme === 'light' ? 'Karanlık Tema' : 'Aydınlık Tema'}
    </button>
  )
}

export default ThemeToggle
