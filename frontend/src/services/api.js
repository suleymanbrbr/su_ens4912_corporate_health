// Centralized fetch wrapper. New components should use this; existing
// components keep their inline fetch calls to avoid wider refactors.
const BASE = import.meta.env.VITE_API_BASE_URL || '';

function authHeaders() {
  const t = localStorage.getItem('token');
  return t ? { Authorization: `Bearer ${t}` } : {};
}

export async function apiGet(path) {
  const r = await fetch(`${BASE}${path}`, { headers: { ...authHeaders() } });
  if (!r.ok) {
    const body = await r.json().catch(() => ({}));
    throw new Error(body.detail || `${r.status}`);
  }
  return r.json();
}

export async function apiPost(path, body) {
  const r = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    const errBody = await r.json().catch(() => ({}));
    throw new Error(errBody.detail || `${r.status}`);
  }
  return r.json();
}

export async function apiDelete(path) {
  const r = await fetch(`${BASE}${path}`, { method: 'DELETE', headers: authHeaders() });
  if (!r.ok && r.status !== 204) {
    throw new Error(`${r.status}`);
  }
  return true;
}
