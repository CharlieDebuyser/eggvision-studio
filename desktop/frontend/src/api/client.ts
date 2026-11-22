/**
 * Client HTTP minimal pour l'API EggVision (même origine, préfixe /api).
 *
 *   const info = await api.get<ProjectInfo>('/project');
 *   await api.post<ProjectInfo>('/project/open', { path });
 *   await api.get<Job[]>('/jobs', undefined, { query: { status: 'running' } });
 *
 * Les chemins sont relatifs à /api (le préfixe est ajouté s'il manque).
 * Toute réponse non-2xx lève une ApiError { status, detail } ; detail est le message
 * français renvoyé par le backend (ou un message générique).
 */

export const API_BASE = '/api';

export class ApiError extends Error {
  readonly status: number;
  readonly detail: string;
  constructor(status: number, detail: string) {
    super(detail);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
  }
}

export type QueryValue = string | number | boolean | null | undefined;

export interface RequestOptions {
  signal?: AbortSignal;
  /** Paramètres d'URL ; les valeurs null/undefined/'' sont ignorées. */
  query?: Record<string, QueryValue>;
  headers?: Record<string, string>;
}

/** Construit l'URL complète d'un endpoint (utile aussi pour <img>, <a download>…). */
export function apiUrl(path: string, query?: Record<string, QueryValue>): string {
  let p = path.startsWith('/') ? path : `/${path}`;
  if (p.startsWith('/api/')) p = p.slice(4);
  let url = `${API_BASE}${p}`;
  if (query) {
    const qs = new URLSearchParams();
    for (const [k, v] of Object.entries(query)) {
      if (v === undefined || v === null || v === '') continue;
      qs.append(k, String(v));
    }
    const s = qs.toString();
    if (s) url += (url.includes('?') ? '&' : '?') + s;
  }
  return url;
}

function extractDetail(body: unknown, status: number, statusText: string): string {
  if (body && typeof body === 'object' && 'detail' in body) {
    const d = (body as { detail: unknown }).detail;
    if (typeof d === 'string') return d;
    // Erreurs de validation FastAPI : [{loc, msg, type}]
    if (Array.isArray(d)) {
      const msgs = d
        .map((e) => (e && typeof e === 'object' && 'msg' in e ? String((e as { msg: unknown }).msg) : ''))
        .filter(Boolean);
      if (msgs.length) return msgs.join(' ; ');
    }
  }
  if (typeof body === 'string' && body.trim()) return body.trim().slice(0, 300);
  if (status === 0) return 'Le serveur local ne répond pas.';
  return `Erreur ${status}${statusText ? ` (${statusText})` : ''}`;
}

async function request<T>(method: string, path: string, body?: unknown, opts: RequestOptions = {}): Promise<T> {
  const headers: Record<string, string> = { Accept: 'application/json', ...opts.headers };
  let payload: BodyInit | undefined;
  if (body !== undefined) {
    if (body instanceof FormData || body instanceof Blob) {
      payload = body;
    } else {
      headers['Content-Type'] = 'application/json';
      payload = JSON.stringify(body);
    }
  }

  let res: Response;
  try {
    res = await fetch(apiUrl(path, opts.query), { method, headers, body: payload, signal: opts.signal });
  } catch (err) {
    if (err instanceof DOMException && err.name === 'AbortError') throw err;
    throw new ApiError(0, 'Impossible de joindre le serveur local. Vérifiez que l’application est démarrée.');
  }

  const ctype = res.headers.get('content-type') ?? '';
  let data: unknown = null;
  if (res.status !== 204) {
    const text = await res.text();
    if (text) {
      if (ctype.includes('json')) {
        try {
          data = JSON.parse(text);
        } catch {
          data = text;
        }
      } else {
        data = text;
      }
    }
  }

  if (!res.ok) throw new ApiError(res.status, extractDetail(data, res.status, res.statusText));
  return data as T;
}

export const api = {
  get: <T>(path: string, _body?: undefined, opts?: RequestOptions) => request<T>('GET', path, undefined, opts),
  post: <T>(path: string, body?: unknown, opts?: RequestOptions) => request<T>('POST', path, body, opts),
  put: <T>(path: string, body?: unknown, opts?: RequestOptions) => request<T>('PUT', path, body, opts),
  patch: <T>(path: string, body?: unknown, opts?: RequestOptions) => request<T>('PATCH', path, body, opts),
  del: <T>(path: string, body?: unknown, opts?: RequestOptions) => request<T>('DELETE', path, body, opts),
};

/** URLs d'images servies par le backend (JPEG, utilisables directement dans <img src>). */
export const mediaUrl = {
  frame: (lotId: string | number, index: number, w?: number) =>
    apiUrl(`/media/frame/${encodeURIComponent(String(lotId))}/${index}`, { w }),
  crop: (trackId: string | number, view: number) =>
    apiUrl(`/media/crop/${encodeURIComponent(String(trackId))}/${view}`),
};

/** Message lisible pour n'importe quelle erreur (ApiError ou autre). */
export function errorMessage(err: unknown): string {
  if (err instanceof ApiError) return err.detail;
  if (err instanceof Error) return err.message;
  return String(err ?? 'Erreur inconnue');
}
