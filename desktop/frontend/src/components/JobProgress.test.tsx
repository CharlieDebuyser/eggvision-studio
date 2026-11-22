import { afterEach, describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import { ToastProvider } from '@/components/ui';
import { JobProgress } from './JobProgress';

const baseJob = {
  id: 'j1', kind: 'demo', title: 'Projet de démonstration', params: {}, error: null,
  created_at: '2026-11-20T10:00:00Z', started_at: '2026-11-20T10:00:01Z',
};

afterEach(() => vi.unstubAllGlobals());

function wrap(ui: React.ReactNode) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <ToastProvider>
        <MemoryRouter>{ui}</MemoryRouter>
      </ToastProvider>
    </QueryClientProvider>,
  );
}

describe('JobProgress', () => {
  it('affiche la progression puis appelle onDone une seule fois', async () => {
    const responses = [
      { ...baseJob, status: 'running', progress: 0.4, message: 'Rendu des images', result: null, ended_at: null },
      { ...baseJob, status: 'done', progress: 1, message: 'Terminé', result: { path: '/p' }, ended_at: '2026-11-20T10:00:09Z' },
    ];
    let i = 0;
    vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify(responses[Math.min(i++, 1)]), { headers: { 'content-type': 'application/json' } })));
    const onDone = vi.fn();
    wrap(<JobProgress jobId="j1" onDone={onDone} />);

    expect(await screen.findByText('Rendu des images')).toBeInTheDocument();
    expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '40');
    expect(screen.getByRole('button', { name: /Annuler/ })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /Voir le journal/ })).toHaveAttribute('href', '/taches?job=j1');

    await waitFor(() => expect(onDone).toHaveBeenCalledTimes(1), { timeout: 3000 });
    expect(onDone.mock.calls[0][0].result).toEqual({ path: '/p' });
    expect(screen.queryByRole('button', { name: /Annuler/ })).not.toBeInTheDocument();
  });
});
