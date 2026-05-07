import React from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { stripLegacyMarkup } from '../utils/textNormalize'

export default function MarkdownBody({ children, className = '' }) {
  const raw = typeof children === 'string' ? stripLegacyMarkup(children) : ''
  return (
    <div className={`md-body ${className}`}>
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{raw}</ReactMarkdown>
    </div>
  )
}
