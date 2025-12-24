const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

async function handleResponse(res: Response) {
  if (res.status === 401) {
    // Force logout / redirect to login in dev
    try {
      localStorage.removeItem('token');
      localStorage.removeItem('user');
    } catch (e) {}
    if (typeof window !== 'undefined') window.location.href = '/auth/login';
    throw new Error('Unauthorized');
  }

  const contentType = res.headers.get('content-type') || '';
  if (contentType.includes('application/json')) return res.json();
  return res.text();
}

function getToken() {
  try {
    const token = localStorage.getItem('token');
    return token && token !== 'null' ? token : null;
  } catch (e) {
    return null;
  }
}

export async function apiGet(path: string) {
  const token = getToken();
  const headers: Record<string, string> = { 'Accept': 'application/json' };
  if (token) headers['Authorization'] = `Bearer ${token}`;

  const res = await fetch(`${API_URL}${path}`, { headers, credentials: 'include' });
  return handleResponse(res);
}

export async function apiPost(path: string, body?: any, isForm = false) {
  const token = getToken();
  const headers: Record<string, string> = {};
  let payload: any = body;
  if (!isForm) {
    headers['Content-Type'] = 'application/json';
    payload = body ? JSON.stringify(body) : undefined;
  }
  if (token) headers['Authorization'] = `Bearer ${token}`;

  const res = await fetch(`${API_URL}${path}`, { method: 'POST', headers, body: payload, credentials: 'include' });
  return handleResponse(res);
}

export async function apiPut(path: string, body?: any) {
  const token = getToken();
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (token) headers['Authorization'] = `Bearer ${token}`;
  const res = await fetch(`${API_URL}${path}`, { method: 'PUT', headers, body: JSON.stringify(body), credentials: 'include' });
  return handleResponse(res);
}

export default {
  get: apiGet,
  post: apiPost,
  put: apiPut,
};
