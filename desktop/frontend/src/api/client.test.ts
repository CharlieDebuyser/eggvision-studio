import { afterEach, describe, expect, it, vi } from 'vitest';
import { api, apiUrl, ApiError, mediaUrl, errorMessage } from './client';

function mockFetch(status: number, body: unknown, contentType = 'application/json') {
  const fn = vi.fn(async () => new Response(typeof body === 'string' ? body : JSON.stringify(body), { status, headers: { 'content-type': contentType } }));
  vi.stubGlobal('fetch', fn);
  return fn;
}

afterEach(() => vi.unstubAllGlobals());

describe('apiUrl', () => {
  it('ajoute le préfixe /api et ignore les paramètres vides', () => {
    expect(apiUrl('/jobs', { kind: 'demo', status: undefined, limit: 10, q: '' })).toBe('/api/jobs?kind=demo&limit=10');
    expect(apiUrl('/api/project')).toBe('/api/project');
    expect(apiUrl('project')).toBe('/api/project');
  });
});

describe('mediaUrl', () => {
  it('construit les URL des images', () => {
    expect(mediaUrl.frame(3, 12)).toBe('/api/media/frame/3/12');
    expect(mediaUrl.frame('lot a', 0, 320)).toBe('/api/media/frame/lot%20a/0?w=320');
    expect(mediaUrl.crop('t-1', 2)).toBe('/api/media/crop/t-1/2');
  });
});

describe('api', () => {
  it('envoie du JSON et renvoie la réponse décodée', async () => {
    const f = mockFetch(200, { open: true, name: 'p' });
    const res = await api.post<{ open: boolean }>('/project/open', { path: '/x' });
    expect(res).toEqual({ open: true, name: 'p' });
    const [url, init] = f.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe('/api/project/open');
    expect(init.method).toBe('POST');
    expect(init.body).toBe(JSON.stringify({ path: '/x' }));
    expect((init.headers as Record<string, string>)['Content-Type']).toBe('application/json');
  });

  it('transmet les paramètres de requête', async () => {
    const f = mockFetch(200, []);
    await api.get('/jobs', undefined, { query: { status: 'running' } });
    expect((f.mock.calls[0] as unknown as [string])[0]).toBe('/api/jobs?status=running');
  });

  it('lève ApiError avec le detail du backend', async () => {
    mockFetch(409, { detail: 'Aucun projet ouvert.' });
    const err = (await api.get('/project/config').catch((e) => e)) as ApiError;
    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(409);
    expect(err.detail).toBe('Aucun projet ouvert.');
    expect(errorMessage(err)).toBe('Aucun projet ouvert.');
  });

  it('agrège les erreurs de validation FastAPI', async () => {
    mockFetch(422, { detail: [{ msg: 'champ requis' }, { msg: 'nom invalide' }] });
    const err = (await api.post('/project/create', {}).catch((e) => e)) as ApiError;
    expect(err.detail).toBe('champ requis ; nom invalide');
  });

  it('signale un serveur injoignable (status 0)', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => { throw new TypeError('Failed to fetch'); }));
    const err = (await api.get('/health').catch((e) => e)) as ApiError;
    expect(err.status).toBe(0);
    expect(err.detail).toMatch(/serveur local/);
  });

  it('gère une réponse non JSON en erreur', async () => {
    mockFetch(500, 'Internal Server Error', 'text/plain');
    const err = (await api.get('/x').catch((e) => e)) as ApiError;
    expect(err.status).toBe(500);
    expect(err.detail).toBe('Internal Server Error');
  });
});
