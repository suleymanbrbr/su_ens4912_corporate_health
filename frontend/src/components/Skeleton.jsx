import React from 'react'

/**
 * Skeleton loader primitives.
 *
 * All variants share the `.skeleton-shimmer` class (defined in index.css)
 * which animates a horizontal gradient using CSS vars so light + dark
 * modes inherit the right tones without per-component overrides.
 *
 * Keep these intentionally tiny — they should be drop-in replacements
 * for "Yükleniyor…" text spans.
 */

const baseStyle = {
  display: 'block',
  borderRadius: '6px',
  background: 'var(--bg-elev, var(--border))',
}

export function SkeletonLine({ width = '100%', height = '0.85rem', style }) {
  return (
    <span
      aria-hidden="true"
      className="skeleton-shimmer"
      style={{
        ...baseStyle,
        width,
        height,
        margin: '0.35rem 0',
        ...style,
      }}
    />
  )
}

export function SkeletonAvatar({ size = 40, style }) {
  return (
    <span
      aria-hidden="true"
      className="skeleton-shimmer"
      style={{
        ...baseStyle,
        width: size,
        height: size,
        borderRadius: '50%',
        flexShrink: 0,
        ...style,
      }}
    />
  )
}

export function SkeletonCard({ rows = 2, style }) {
  return (
    <div
      aria-hidden="true"
      className="premium-card"
      style={{
        padding: '0.85rem 1rem',
        display: 'flex',
        flexDirection: 'column',
        gap: '0.5rem',
        ...style,
      }}
    >
      <SkeletonLine width="70%" height="0.95rem" />
      {Array.from({ length: Math.max(0, rows - 1) }).map((_, i) => (
        <SkeletonLine key={i} width={`${50 + (i % 3) * 15}%`} height="0.75rem" />
      ))}
    </div>
  )
}

export default { SkeletonLine, SkeletonAvatar, SkeletonCard }
