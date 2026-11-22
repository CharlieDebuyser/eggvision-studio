/** Appels API et hooks React Query du module Simulation. */
import { useEffect, useState } from 'react';
import { keepPreviousData, useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { api, apiUrl } from '@/api/client';
import { queryKeys } from '@/api/hooks';
import type { Job } from '@/api/types';
import type {
  CompareResult,
  Defaults,
  EggDetail,
  EligibleModel,
  SimDetail,
  SimList,
  SimParams,
  Timing,
  TimelineChunk,
  VideoOptions,
} from './types';

export const simKeys = {
  all: ['simulation'] as const,
  list: (page: number, pageSize: number) => ['simulation', 'list', page, pageSize] as const,
  one: (id: string) => ['simulation', 'one', id] as const,
  models: ['simulation', 'models'] as const,
  defaults: ['simulation', 'defaults'] as const,
  timing: (p: unknown) => ['simulation', 'timing', p] as const,
  egg: (id: string, c: number, l: number) => ['simulation', 'egg', id, c, l] as const,
  compare: (ids: string[]) => ['simulation', 'compare', ids] as const,
};

export const simUrls = {
  journal: (id: string) => apiUrl(`/simulation/${encodeURIComponent(id)}/journal.csv`),
  video: (id: string, download = false) => apiUrl(`/simulation/${encodeURIComponent(id)}/video`, { download: download ? 1 : undefined }),
  media: (url: string) => apiUrl(url),
};

export function fetchTimeline(id: string, start: number, count: number, signal?: AbortSignal) {
  return api.get<TimelineChunk>(`/simulation/${encodeURIComponent(id)}/timeline`, undefined, { query: { start, count }, signal });
}

export function useSimDefaults() {
  return useQuery({
    queryKey: simKeys.defaults,
    queryFn: ({ signal }) => api.get<Defaults>('/simulation/defaults', undefined, { signal }),
    staleTime: Infinity,
  });
}

export function useSimModels() {
  return useQuery({
    queryKey: simKeys.models,
    queryFn: ({ signal }) => api.get<EligibleModel[]>('/simulation/models', undefined, { signal }),
  });
}

export function useSimList(page: number, pageSize: number) {
  return useQuery({
    queryKey: simKeys.list(page, pageSize),
    queryFn: ({ signal }) => api.get<SimList>('/simulation', undefined, { signal, query: { limit: pageSize, offset: page * pageSize } }),
    placeholderData: keepPreviousData,
    refetchInterval: (q) => (q.state.data?.items.some((s) => s.status === 'running' || s.status === 'created') ? 2000 : false),
  });
}

export function useSim(id: string | undefined) {
  return useQuery({
    queryKey: simKeys.one(id ?? ''),
    queryFn: ({ signal }) => api.get<SimDetail>(`/simulation/${encodeURIComponent(id!)}`, undefined, { signal }),
    enabled: !!id,
    refetchInterval: (q) => {
      const s = q.state.data;
      const videoActive = s?.video_job && (s.video_job.status === 'queued' || s.video_job.status === 'running');
      return s && (s.status === 'running' || s.status === 'created' || videoActive) ? 1500 : false;
    },
  });
}

/** Valeur « retardée » : évite un appel à chaque frappe. */
export function useDebounced<T>(value: T, ms = 250): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setV(value), ms);
    return () => clearTimeout(t);
  }, [value, ms]);
  return v;
}

/** Grandeurs temporelles dérivées (budget…) calculées par le backend, en direct. */
export function useTiming(params: Partial<SimParams> | null) {
  const debounced = useDebounced(params, 200);
  return useQuery({
    queryKey: simKeys.timing(debounced),
    queryFn: ({ signal }) => {
      const { model_id: _ignored, ...rest } = debounced ?? {};
      return api.post<Timing>('/simulation/timing', rest, { signal });
    },
    enabled: !!debounced,
    placeholderData: keepPreviousData,
    retry: false,
  });
}

export function useEgg(id: string, column: number | null, lane: number | null) {
  return useQuery({
    queryKey: simKeys.egg(id, column ?? -1, lane ?? -1),
    queryFn: ({ signal }) => api.get<EggDetail>(`/simulation/${encodeURIComponent(id)}/egg/${column}/${lane}`, undefined, { signal }),
    enabled: column !== null && lane !== null,
    staleTime: Infinity,
  });
}

export function useCompare(ids: string[]) {
  return useQuery({
    queryKey: simKeys.compare(ids),
    queryFn: ({ signal }) =>
      api.get<CompareResult>(`/simulation/compare?${ids.map((i) => `ids=${encodeURIComponent(i)}`).join('&')}`, undefined, { signal }),
    enabled: ids.length > 0,
  });
}

interface Created {
  simulation: SimDetail;
  job: Job;
}

export function useCreateSim() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: Partial<SimParams> & { name?: string }) => api.post<Created>('/simulation', body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: simKeys.all });
      qc.invalidateQueries({ queryKey: queryKeys.project });
    },
  });
}

export function useDuplicateSim() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, ...body }: { id: string; name?: string; model_id?: string; threshold?: number | null; variant?: string | null }) =>
      api.post<Created>(`/simulation/${encodeURIComponent(id)}/duplicate`, body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: simKeys.all });
      qc.invalidateQueries({ queryKey: queryKeys.project });
    },
  });
}

export function useDeleteSim() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => api.del<{ deleted: string }>(`/simulation/${encodeURIComponent(id)}`),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: simKeys.all });
      qc.invalidateQueries({ queryKey: queryKeys.project });
    },
  });
}

export function useStartVideo() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, ...opts }: { id: string } & Partial<VideoOptions>) =>
      api.post<{ job: Job }>(`/simulation/${encodeURIComponent(id)}/video`, opts),
    onSuccess: (_d, v) => qc.invalidateQueries({ queryKey: simKeys.one(v.id) }),
  });
}
