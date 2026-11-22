/**
 * Hooks React Query partagés. Les modules peuvent écrire les leurs sur le même modèle :
 *
 *   const q = useQuery({ queryKey: ['lots'], queryFn: ({ signal }) => api.get<Lot[]>('/lots', undefined, { signal }) });
 *
 * Clés de cache : voir `queryKeys`. Après une mutation qui change l'état du projet,
 * invalider `queryKeys.project` pour rafraîchir les compteurs.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { useMutation, useQuery, useQueryClient, type UseQueryOptions } from '@tanstack/react-query';
import { api, errorMessage } from './client';
import type {
  CreateDemoBody,
  CreateProjectBody,
  Diagnostics,
  FsHome,
  FsListing,
  Job,
  JobFilter,
  JobLogChunk,
  JobRef,
  JobStatus,
  MkdirBody,
  ProjectConfig,
  ProjectInfo,
  RecentProject,
} from './types';

export const queryKeys = {
  health: ['health'] as const,
  project: ['project'] as const,
  recent: ['project', 'recent'] as const,
  config: ['project', 'config'] as const,
  fsHome: ['fs', 'home'] as const,
  fsList: (path: string) => ['fs', 'list', path] as const,
  jobs: (filter?: JobFilter) => ['jobs', 'list', filter ?? {}] as const,
  jobsAll: ['jobs'] as const,
  job: (id: string) => ['jobs', 'one', id] as const,
  diagnostics: ['diagnostics'] as const,
  diagnosticsLog: (lines: number) => ['diagnostics', 'log', lines] as const,
};

export const isJobActive = (s: JobStatus | undefined) => s === 'queued' || s === 'running';

// ---------------------------------------------------------------- Projet

/** Projet courant (`data.open` indique si un projet est ouvert). */
export function useProject() {
  return useQuery({
    queryKey: queryKeys.project,
    queryFn: ({ signal }) => api.get<ProjectInfo>('/project', undefined, { signal }),
    staleTime: 5_000,
  });
}

export function useRecentProjects() {
  return useQuery({
    queryKey: queryKeys.recent,
    queryFn: ({ signal }) => api.get<RecentProject[]>('/project/recent', undefined, { signal }),
  });
}

function useProjectMutation<B>(fn: (body: B) => Promise<ProjectInfo>) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: fn,
    onSuccess: (info) => {
      qc.setQueryData(queryKeys.project, info);
      // Tout ce qui dépend du projet doit être relu.
      void qc.invalidateQueries({ predicate: (q) => q.queryKey[0] !== 'project' || q.queryKey.length > 1 });
    },
  });
}

export const useCreateProject = () =>
  useProjectMutation((b: CreateProjectBody) => api.post<ProjectInfo>('/project/create', b));
export const useOpenProject = () =>
  useProjectMutation((b: { path: string }) => api.post<ProjectInfo>('/project/open', b));
export const useCloseProject = () => useProjectMutation((_: void) => api.post<ProjectInfo>('/project/close'));

export function useCreateDemo() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (b: CreateDemoBody) => api.post<JobRef>('/project/demo', b),
    onSuccess: () => void qc.invalidateQueries({ queryKey: queryKeys.jobsAll }),
  });
}

export function useProjectConfig() {
  return useQuery({
    queryKey: queryKeys.config,
    queryFn: ({ signal }) => api.get<ProjectConfig>('/project/config', undefined, { signal }),
  });
}

/** Fusion partielle de config.yaml ; renvoie la config complète. */
export function useUpdateProjectConfig() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (patch: ProjectConfig) => api.put<ProjectConfig>('/project/config', patch),
    onSuccess: (cfg) => qc.setQueryData(queryKeys.config, cfg),
  });
}

// ---------------------------------------------------------------- Système de fichiers

export function useFsHome() {
  return useQuery({
    queryKey: queryKeys.fsHome,
    queryFn: ({ signal }) => api.get<FsHome>('/fs/home', undefined, { signal }),
    staleTime: Infinity,
  });
}

/** Liste les sous-dossiers de `path` ('' = dossier personnel). */
export function useFsList(path: string, enabled = true) {
  return useQuery({
    queryKey: queryKeys.fsList(path),
    queryFn: ({ signal }) => api.get<FsListing>('/fs/list', undefined, { signal, query: { path } }),
    enabled,
    staleTime: 2_000,
    placeholderData: (prev) => prev,
  });
}

/** Crée un dossier (POST /api/fs/mkdir) et rafraîchit la liste du parent. */
export function useMkdir() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (b: MkdirBody) => api.post<{ path: string }>('/fs/mkdir', b),
    onSuccess: (_r, b) => void qc.invalidateQueries({ queryKey: queryKeys.fsList(b.parent_dir) }),
  });
}

// ---------------------------------------------------------------- Tâches

/**
 * Liste des tâches (plus récentes d'abord). Rafraîchie toutes les 1,5 s tant qu'une
 * tâche est active, sinon toutes les 5 s (pour voir apparaître les nouvelles).
 */
export function useJobs(filter?: JobFilter, options?: Pick<UseQueryOptions<Job[]>, 'enabled'>) {
  return useQuery({
    queryKey: queryKeys.jobs(filter),
    queryFn: ({ signal }) =>
      api.get<Job[]>('/jobs', undefined, {
        signal,
        query: { kind: filter?.kind, status: filter?.status, limit: filter?.limit ?? 100 },
      }),
    refetchInterval: (q) => (q.state.data?.some((j) => isJobActive(j.status)) ? 1_500 : 5_000),
    ...options,
  });
}

/**
 * Une tâche. Interroge toutes les secondes tant qu'elle est `queued`/`running`,
 * puis s'arrête. `id` null/undefined → requête désactivée.
 */
export function useJob<P = Record<string, unknown>, R = Record<string, unknown>>(id: string | null | undefined) {
  return useQuery({
    queryKey: queryKeys.job(id ?? ''),
    queryFn: ({ signal }) => api.get<Job<P, R>>(`/jobs/${encodeURIComponent(id!)}`, undefined, { signal }),
    enabled: !!id,
    refetchInterval: (q) => (q.state.error ? 3_000 : !q.state.data || isJobActive(q.state.data.status) ? 1_000 : false),
  });
}

export function useCancelJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => api.post<Job>(`/jobs/${encodeURIComponent(id)}/cancel`),
    onSuccess: (job) => {
      qc.setQueryData(queryKeys.job(job.id), job);
      void qc.invalidateQueries({ queryKey: queryKeys.jobsAll });
    },
  });
}

export interface JobLogState {
  text: string;
  loading: boolean;
  error: string | null;
  /** Fin du fichier atteinte lors de la dernière lecture. */
  eof: boolean;
  /** Relit le journal depuis le début. */
  reload: () => void;
}

const MAX_LOG_CHARS = 400_000;

/**
 * Journal d'une tâche, lu de façon incrémentale (offset). Suit le fichier toutes les
 * secondes tant que la tâche est active (ou `follow` forcé), puis fait une dernière lecture.
 */
export function useJobLog(id: string | null | undefined, opts: { follow?: boolean; intervalMs?: number } = {}): JobLogState {
  const job = useJob(id);
  const active = opts.follow ?? isJobActive(job.data?.status);
  const interval = opts.intervalMs ?? 1_000;
  const [state, setState] = useState({ text: '', offset: 0, eof: false, loading: !!id, error: null as string | null });
  const [epoch, setEpoch] = useState(0);
  const offsetRef = useRef(0);

  useEffect(() => {
    offsetRef.current = 0;
    setState({ text: '', offset: 0, eof: false, loading: !!id, error: null });
  }, [id, epoch]);

  useEffect(() => {
    if (!id) return;
    const ctrl = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    let stopped = false;

    const tick = async () => {
      try {
        // Lit tous les morceaux disponibles jusqu'à eof.
        for (let guard = 0; guard < 50; guard++) {
          const chunk = await api.get<JobLogChunk>(`/jobs/${encodeURIComponent(id)}/log`, undefined, {
            signal: ctrl.signal,
            query: { offset: offsetRef.current },
          });
          offsetRef.current = chunk.next_offset;
          setState((s) => {
            let text = s.text + chunk.text;
            if (text.length > MAX_LOG_CHARS) text = '… (début tronqué)\n' + text.slice(-MAX_LOG_CHARS);
            return { text, offset: chunk.next_offset, eof: chunk.eof, loading: false, error: null };
          });
          if (chunk.eof || !chunk.text) break;
        }
      } catch (err) {
        if (ctrl.signal.aborted) return;
        setState((s) => ({ ...s, loading: false, error: errorMessage(err) }));
      }
      if (!stopped && active) timer = setTimeout(tick, interval);
    };
    void tick();
    return () => {
      stopped = true;
      ctrl.abort();
      if (timer) clearTimeout(timer);
    };
  }, [id, active, interval, epoch]);

  const reload = useCallback(() => setEpoch((e) => e + 1), []);
  return { text: state.text, loading: state.loading, error: state.error, eof: state.eof, reload };
}

// ---------------------------------------------------------------- Diagnostic

export function useDiagnostics() {
  return useQuery({
    queryKey: queryKeys.diagnostics,
    queryFn: ({ signal }) => api.get<Diagnostics>('/diagnostics', undefined, { signal }),
  });
}

export function useDiagnosticsLog(lines = 300) {
  return useQuery({
    queryKey: queryKeys.diagnosticsLog(lines),
    queryFn: ({ signal }) => api.get<{ text: string }>('/diagnostics/log', undefined, { signal, query: { lines } }),
  });
}
