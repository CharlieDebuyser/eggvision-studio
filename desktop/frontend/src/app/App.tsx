import { Suspense, useState, type ReactNode } from 'react';
import { HashRouter, Navigate, Outlet, Route, Routes, useLocation } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useProject } from '@/api/hooks';
import { ApiError } from '@/api/client';
import { ErrorState, LoadingState, Page, ToastProvider } from '@/components/ui';
import { Sidebar } from './Sidebar';
import { MODULES } from './modules';
import { ErrorBoundary } from './ErrorBoundary';
import ProjetPage from '@/pages/projet';
import TachesPage from '@/pages/taches';
import DiagnosticPage from '@/pages/diagnostic';

export function createQueryClient() {
  return new QueryClient({
    defaultOptions: {
      queries: {
        retry: (count, err) => !(err instanceof ApiError && err.status >= 400 && err.status < 500) && count < 2,
        refetchOnWindowFocus: false,
        staleTime: 2_000,
      },
      mutations: { retry: false },
    },
  });
}

/** Garde : redirige vers « / » tant qu'aucun projet n'est ouvert. */
export function RequireProject({ children }: { children: ReactNode }) {
  const project = useProject();
  const location = useLocation();
  if (project.isLoading) return <LoadingState label="Chargement du projet…" />;
  if (project.isError && !project.data)
    return (
      <Page>
        <ErrorState title="Impossible de lire l’état du projet" error={project.error} onRetry={() => project.refetch()} />
      </Page>
    );
  if (!project.data?.open) return <Navigate to="/" replace state={{ from: location.pathname }} />;
  return <>{children}</>;
}

function Shell() {
  const location = useLocation();
  return (
    <div className="app">
      <Sidebar />
      <main className="app__main" id="contenu">
        {/* La clé réinitialise la frontière d'erreur à chaque changement de page. */}
        <ErrorBoundary key={location.pathname.split('/')[1]}>
          <Suspense fallback={<LoadingState label="Chargement du module…" />}>
            <Outlet />
          </Suspense>
        </ErrorBoundary>
      </main>
    </div>
  );
}

export function AppRoutes() {
  return (
    <Routes>
      <Route element={<Shell />}>
        <Route index element={<ProjetPage />} />
        {MODULES.map((m) => (
          <Route
            key={m.path}
            path={`${m.path}/*`}
            element={
              <RequireProject>
                <m.component />
              </RequireProject>
            }
          />
        ))}
        <Route path="taches" element={<TachesPage />} />
        <Route path="diagnostic" element={<DiagnosticPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}

export function App({ queryClient }: { queryClient?: QueryClient }) {
  const [client] = useState(() => queryClient ?? createQueryClient());
  return (
    <QueryClientProvider client={client}>
      <ToastProvider>
        <HashRouter>
          <AppRoutes />
        </HashRouter>
      </ToastProvider>
    </QueryClientProvider>
  );
}
