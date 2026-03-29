import React, { useState } from 'react'
import { ThumbsUp, ThumbsDown } from 'lucide-react'

function ResponseFeedback({ messageId }) {
  const [voted, setVoted] = useState(null)

  const handleVote = async (rating) => {
    if (voted !== null) return
    setVoted(rating)
    try {
      await fetch('/api/feedback', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify({
          message_id: messageId,
          rating: rating,
          feedback_text: '',
          is_accurate: rating > 0
        })
      })
    } catch (e) {
      // ignore
    }
  }

  return (
    <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.5rem' }}>
      <button
        onClick={() => handleVote(1)}
        title="Yararlı"
        style={{
          background: voted === 1 ? '#dcfce7' : 'transparent',
          border: '1px solid ' + (voted === 1 ? '#22c55e' : 'var(--border)'),
          borderRadius: '6px',
          padding: '0.25rem 0.5rem',
          cursor: voted !== null ? 'default' : 'pointer',
          color: voted === 1 ? '#16a34a' : 'var(--text-muted)',
          display: 'flex',
          alignItems: 'center',
          gap: '0.25rem',
          fontSize: '0.75rem'
        }}
      >
        <ThumbsUp size={12} />
      </button>
      <button
        onClick={() => handleVote(-1)}
        title="Yararsız"
        style={{
          background: voted === -1 ? '#fef2f2' : 'transparent',
          border: '1px solid ' + (voted === -1 ? '#ef4444' : 'var(--border)'),
          borderRadius: '6px',
          padding: '0.25rem 0.5rem',
          cursor: voted !== null ? 'default' : 'pointer',
          color: voted === -1 ? '#dc2626' : 'var(--text-muted)',
          display: 'flex',
          alignItems: 'center',
          gap: '0.25rem',
          fontSize: '0.75rem'
        }}
      >
        <ThumbsDown size={12} />
      </button>
    </div>
  )
}

export default ResponseFeedback
