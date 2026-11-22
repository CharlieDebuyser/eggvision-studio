/** Appels API et hooks React Query du module Entraînement (docs/api/training.md). */
import { useEffect, useState } from 'react';
import { keepPreviousData, useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { api, apiUrl } from '@/api/client';
import { queryKeys } from '@/api/hooks';
import type {
  CamMeta,
  CompareResult,
  ErrorsResponse,
  HistoryResponse,
  ModelDetail,
  ModelSummary,
  Objective,
  Options,
  Paged,
  Report,
  Split,
  SplitDetail,
  SplitParams,
  SplitPreview,
  Started,
  Subset,
  Suggestion,
  TrainParams,
} from './types';

const enc = encodeURIComponent;

/** Valeur « retardée » : évite un appel à chaque frappe. */
export function useDebounced<T>(value: T, ms = 250): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setV(value), ms);
    return () => clearTimeout(t);
  }, [value, ms]);
  return v;
}

export const trainKeys = {
  all: ['training'] as const,
  options: ['training', 'options'] as const,
  splits: ['training', 'splits'] as const,
  split: (id: number) => ['training', 'splits', id] as const,
  models: ['training', 'models'] as const,
  modelList: (page: number, size: number) => ['training', 'models', 'list', page, size] as const,
  model: (id: string) => ['training', 'models', 'one', id] as const,
  history: (id: string) => ['training', 'models', 'history', id] as const,
  report: (id: string) => ['training', 'models', 'report', id] as const,
  errors: (id: string, p: unknown) => ['training', 'models', 'errors', id, p] as const,
  cam: (id: string, track: number) => ['training', 'models', 'cam', id, track] as const,
  compare: (ids: string[], p: unknown) => ['training', 'compare', ids, p] as const,
};

export const isModelActive = (m?: { status: string; job_active?: boolean }) =>
  !!m && (m.job_active || m.status === 'training' || m.status === 'created');

export function useOptions() {
  return useQuery({
    queryKey: trainKeys.options,
    queryFn: ({ signal }) => api.get<Options>('/training/options', undefined, { signal }),
    staleTime: 30_000,
  });
}

// ------------------------------------------------------------------ découpages
export function useSplits() {
  return useQuery({
    queryKey: trainKeys.splits,
    queryFn: ({ signal }) => api.get<Paged<Split>>('/training/splits', undefined, { signal, query: { limit: 200 } }),
  });
}

export function useSplit(id: number | null | undefined) {
  return useQuery({
    queryKey: trainKeys.split(id ?? -1),
    queryFn: ({ signal }) => api.get<SplitDetail>(`/training/splits/${id}`, undefined, { signal }),
    enabled: id != null && id >= 0,
  });
}

export function useSplitPreview(params: SplitParams | null) {
  return useQuery({
    queryKey: ['training', 'split-preview', params],
    queryFn: ({ signal }) => api.post<SplitPreview>('/training/splits/preview', params, { signal }),
    enabled: !!params,
    placeholderData: keepPreviousData,
    retry: false,
  });
}

export function useCreateSplit() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: SplitParams & { name: string }) => api.post<Split>('/training/splits', body),
    onSuccess: () => qc.invalidateQueries({ queryKey: trainKeys.splits }),
  });
}

export function usePatchAssignments() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, changes, move_group }: { id: number; changes: Record<string, Subset | null>; move_group: boolean }) =>
      api.patch<Split>(`/training/splits/${id}/assignments`, { changes, move_group }),
    onSuccess: () => qc.invalidateQueries({ queryKey: trainKeys.splits }),
  });
}

export function useDeleteSplit() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: number) => api.del<{ deleted: number }>(`/training/splits/${id}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: trainKeys.splits }),
  });
}

// ------------------------------------------------------------------ modèles
export function useModels(page = 0, size = 50, poll = true) {
  return useQuery({
    queryKey: trainKeys.modelList(page, size),
    queryFn: ({ signal }) => api.get<Paged<ModelSummary>>('/training/models', undefined, { signal, query: { limit: size, offset: page * size } }),
    placeholderData: keepPreviousData,
    refetchInterval: (q) => (poll && q.state.data?.items.some(isModelActive) ? 2000 : false),
  });
}

export function useModel(id: string | undefined) {
  return useQuery({
    queryKey: trainKeys.model(id ?? ''),
    queryFn: ({ signal }) => api.get<ModelDetail>(`/training/models/${enc(id!)}`, undefined, { signal }),
    enabled: !!id,
    refetchInterval: (q) => (isModelActive(q.state.data) ? 2000 : false),
  });
}

export function useHistory(id: string, live: boolean) {
  return useQuery({
    queryKey: trainKeys.history(id),
    queryFn: ({ signal }) => api.get<HistoryResponse>(`/training/models/${enc(id)}/history`, undefined, { signal }),
    refetchInterval: live ? 1500 : false,
  });
}

export function useReport(id: string, enabled = true) {
  return useQuery({
    queryKey: trainKeys.report(id),
    queryFn: ({ signal }) => api.get<Report>(`/training/models/${enc(id)}/report`, undefined, { signal }),
    enabled,
    retry: false,
  });
}

function invalidateModel(qc: ReturnType<typeof useQueryClient>, id: string) {
  qc.invalidateQueries({ queryKey: trainKeys.models });
  qc.invalidateQueries({ queryKey: trainKeys.report(id) });
  qc.invalidateQueries({ queryKey: ['training', 'models', 'errors', id] });
  qc.invalidateQueries({ queryKey: ['export'] });
}

export function useStartTraining() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: { split_id: number; preset: string | null; hparams: Partial<TrainParams>; name?: string }) =>
      api.post<Started>('/training/train', body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: trainKeys.all });
      qc.invalidateQueries({ queryKey: queryKeys.project });
    },
  });
}

export function useStartAnomaly() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: { split_id: number; params: Record<string, unknown>; name?: string }) => api.post<Started>('/training/anomaly', body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: trainKeys.all });
      qc.invalidateQueries({ queryKey: queryKeys.project });
    },
  });
}

export function useResume() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, ...body }: { id: string; mode: 'continue' | 'extend'; epochs?: number | null; hparams?: Partial<TrainParams> }) =>
      api.post<Started>(`/training/models/${enc(id)}/resume`, body),
    onSuccess: () => qc.invalidateQueries({ queryKey: trainKeys.all }),
  });
}

export function useCancelTraining() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => api.post<{ job_id: string }>(`/training/models/${enc(id)}/cancel`),
    onSuccess: (_d, id) => qc.invalidateQueries({ queryKey: trainKeys.model(id) }),
  });
}

export function useRenameModel() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, name }: { id: string; name: string }) => api.patch<ModelSummary>(`/training/models/${enc(id)}`, { name }),
    onSuccess: () => qc.invalidateQueries({ queryKey: trainKeys.models }),
  });
}

export function useDeleteModel() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => api.del<{ deleted: string }>(`/training/models/${enc(id)}`),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: trainKeys.all });
      qc.invalidateQueries({ queryKey: queryKeys.project });
    },
  });
}

// ------------------------------------------------------------------ rapport
export function useSuggest(id: string, body: { objective: Objective; target: number; subset: 'val' | 'test' } | null) {
  return useQuery({
    queryKey: ['training', 'models', 'suggest', id, body],
    queryFn: ({ signal }) => api.post<Suggestion>(`/training/models/${enc(id)}/threshold/suggest`, body, { signal }),
    enabled: !!body,
    retry: false,
  });
}

export function useSetThreshold() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, threshold }: { id: string; threshold: number }) =>
      api.put<{ threshold: number; export_outdated: boolean }>(`/training/models/${enc(id)}/threshold`, { threshold }),
    onSuccess: (_d, v) => invalidateModel(qc, v.id),
  });
}

export function useRefreshReport() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => api.post<Report>(`/training/models/${enc(id)}/report/refresh`),
    onSuccess: (_d, id) => invalidateModel(qc, id),
  });
}

export function useErrors(id: string, p: { kind: 'fn' | 'fp'; subset: Subset; threshold: number | null; limit: number; offset: number }) {
  return useQuery({
    queryKey: trainKeys.errors(id, p),
    queryFn: ({ signal }) => api.get<ErrorsResponse>(`/training/models/${enc(id)}/errors`, undefined, { signal, query: { ...p } }),
    placeholderData: keepPreviousData,
  });
}

export function useCam(id: string, trackId: number | null, enabled: boolean) {
  return useQuery({
    queryKey: trainKeys.cam(id, trackId ?? -1),
    queryFn: ({ signal }) => api.get<CamMeta>(`/training/models/${enc(id)}/gradcam/${trackId}`, undefined, { signal }),
    enabled: enabled && trackId != null,
    staleTime: Infinity,
  });
}

export const camUrl = (url: string) => apiUrl(url);

export function useCompare(ids: string[], p: { objective: Objective; target: number; subset: 'val' | 'test' }) {
  return useQuery({
    queryKey: trainKeys.compare(ids, p),
    queryFn: ({ signal }) =>
      api.get<CompareResult>(`/training/compare?${ids.map((i) => `ids=${enc(i)}`).join('&')}`, undefined, { signal, query: { ...p } }),
    enabled: ids.length > 0,
  });
}
