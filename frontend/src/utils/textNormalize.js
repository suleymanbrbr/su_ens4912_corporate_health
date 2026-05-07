/** Remove Pandoc / LaTeX-style noise from SUT text before markdown render */
export function stripLegacyMarkup(text) {
  if (!text || typeof text !== 'string') return ''
  return text
    .replace(/\{\s*\.underline\s*\}/gi, '')
    .replace(/\[\s*([^\]]+)\s*\]\{\s*\.underline\s*\}/g, '$1')
    .replace(/\\?\(([0-9]+)\\?\)/g, '($1)')
}
