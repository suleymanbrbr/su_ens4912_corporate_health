// Centralized fetch wrapper. New code should prefer the helpers in this file;
// existing components still using raw fetch can import `authHeader` and
// `handleAuthFailure` to behave consistently.
const BASE = import.meta.env.VITE_API_BASE_URL || ''

export function getToken() {
  try { return localStorage.getItem('token') } catch { return null }
}

/** Bearer header object — safe to spread into request headers. */
export function authHeader() {
  const t = getToken()
  return t ? { Authorization: `Bearer ${t}` } : {}
}

/**
 * Backwards-compatible alias — some files import { authHeaders }.
 */
export const authHeaders = authHeader

/**
 * Handle a 401 response uniformly: clear the stale token and bounce to /login.
 * Returns true when the response was a 401 (caller should bail out).
 */
export function handleAuthFailure(response) {
  if (response && response.status === 401) {
    try { localStorage.removeItem('token') } catch (_) {}
    // Hard navigate so any in-memory state is discarded.
    if (typeof window !== 'undefined' && window.location.pathname !== '/login') {
      window.location.assign('/login')
    }
    return true
  }
  return false
}

async function parseErrorBody(r) {
  try {
    const ct = r.headers.get('content-type') || ''
    if (ct.includes('application/json')) {
      const body = await r.json()
      return body?.detail || body?.message || `${r.status}`
    }
    const text = await r.text()
    return text || `${r.status}`
  } catch {
    return `${r.status}`
  }
}

export async function apiGet(path, { signal } = {}) {
  const r = await fetch(`${BASE}${path}`, { headers: { ...authHeader() }, signal })
  if (handleAuthFailure(r)) throw new Error('401')
  if (!r.ok) throw new Error(await parseErrorBody(r))
  return r.json()
}

export async function apiPost(path, body, { signal } = {}) {
  const r = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeader() },
    body: JSON.stringify(body),
    signal,
  })
  if (handleAuthFailure(r)) throw new Error('401')
  if (!r.ok) throw new Error(await parseErrorBody(r))
  return r.json()
}

export async function apiPut(path, body, { signal } = {}) {
  const r = await fetch(`${BASE}${path}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json', ...authHeader() },
    body: JSON.stringify(body),
    signal,
  })
  if (handleAuthFailure(r)) throw new Error('401')
  if (!r.ok) throw new Error(await parseErrorBody(r))
  return r.json().catch(() => ({}))
}

export async function apiPatch(path, body, { signal } = {}) {
  const r = await fetch(`${BASE}${path}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json', ...authHeader() },
    body: JSON.stringify(body),
    signal,
  })
  if (handleAuthFailure(r)) throw new Error('401')
  if (!r.ok) throw new Error(await parseErrorBody(r))
  return r.json().catch(() => ({}))
}

export async function apiDelete(path, { signal } = {}) {
  const r = await fetch(`${BASE}${path}`, { method: 'DELETE', headers: authHeader(), signal })
  if (handleAuthFailure(r)) throw new Error('401')
  if (!r.ok && r.status !== 204) throw new Error(await parseErrorBody(r))
  return true
}
