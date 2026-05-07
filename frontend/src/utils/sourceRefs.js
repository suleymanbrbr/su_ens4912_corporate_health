/** Extract madde / ek refs for policy chips */
export function extractSourceRefs(text) {
  if (!text || typeof text !== 'string') return []
  const seen = new Set()
  const out = []
  const madde = /\bMadde\s+([0-9]+(?:\.[0-9]+)*(?:-[a-zğüşıöç])?)\b/gi
  let m
  while ((m = madde.exec(text)) !== null) {
    const label = `Madde ${m[1]}`
    if (!seen.has(label)) {
      seen.add(label)
      out.push({ label, hrefQuery: `Madde ${m[1]}` })
    }
  }
  const ek = /\bEk[-\s]?([0-9]+)\b/gi
  while ((m = ek.exec(text)) !== null) {
    const label = `Ek-${m[1]}`
    if (!seen.has(label)) {
      seen.add(label)
      out.push({ label, hrefQuery: `Ek ${m[1]}` })
    }
  }
  return out.slice(0, 12)
}
