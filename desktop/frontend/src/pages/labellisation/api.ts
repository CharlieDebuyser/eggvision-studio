/**
 * Contrat et appels de l'API de labellisation (/api/labeling). Référence : docs/api/labeling.md.
 */
import { useCallback } from 'react';
import { useQuery, useQueryClient, type QueryClient } from '@tanstack/react-query';
import { api, type QueryValue } from '@/api/client';

// ---------------------------------------------------------------- Types
export type Verdict = 'viable' | 'non_viable' | 'incertain' | 'exclu';
export type Reason = 'contredit' | 'hesite';

export interface LabelState {
  verdict: Verdict | null;
  defect: string | null;
  reviewed: boolean;
  source?: string | null;
  reviewer?: string;
  note?: string;
  updated_at?: string | null;
}

export interface ModelRef {
  id: string;
  name: string;
  threshold: number;
  created_at: string | null;
}

export interface TrackRow {
  id: number;
  lot_id: string;
  lot_type: string;
  lane: number;
  start_frame: number;
  start_ts: number;
  n_views: number;
  quality: string;
  flags: string[];
  prelabel: { verdict: Verdict | null; defect: string | null };
  verdict: Verdict | null;
  defect: string | null;
  reviewed: boolean;
  source: string | null;
  reviewer: string;
  updated_at: string | null;
  note: string;
  score?: number | null;
  priority?: number | null;
  reason?: Reason | null;
}

export interface TrackPage {
  items: TrackRow[];
  next_cursor: string | null;
  total: number | null;
  sort: string;
  model: ModelRef | null;
  message?: string | null;
}

export interface ViewInfo {
  index: number;
  frame_index: number;
  sharpness: number;
  partial: boolean;
  defect_visible: boolean;
  url: string;
}

export interface HistoryEntry {
  id: number;
  track_id: number;
  lot_id: string;
  lane: number;
  user: string;
  at: string;
  action: string;
  action_label: string;
  before: Record<string, unknown>;
  after: Record<string, unknown>;
}

export interface TrackDetail {
  id: number;
  lot: { id: string; name: string; type: string; farm: string; session_date: string } | null;
  lane: number;
  start_frame: number;
  end_frame: number;
  start_ts: number;
  end_ts: number;
  n_detections: number;
  n_views: number;
  quality: string;
  flags: string[];
  mean_sharpness: number;
  prelabel: { verdict: Verdict | null; defect: string | null };
  label: LabelState;
  views: ViewInfo[];
  prediction: {
    model: ModelRef;
    score: number;
    view_scores: number[];
    subset: string;
    predicted: Verdict | null;
    priority: number;
    reason: Reason | null;
  } | null;
  synthetic_truth: Record<string, unknown> | null;
  history: HistoryEntry[];
}

export interface LabelWriteResult {
  label: LabelState & { track_id: number };
  history_id: number;
  changed: boolean;
}

export interface UndoResult {
  undone: number;
  conflicts: number;
  track_ids: number[];
  action: string | null;
  labels: (LabelState & { track_id: number })[];
  message: string;
}

export interface BulkResult {
  updated: number;
  skipped: number[];
  batch: string;
}

export interface LotInfo {
  id: string;
  name: string;
  type: string;
  farm: string;
  session_date: string;
  track_count: number;
}

export interface ModelInfo extends ModelRef {
  status: string;
  n_predictions: number;
}

export interface LabelingMeta {
  user: string;
  min_examples: number;
  default_margin: number;
  max_limit: number;
  max_bulk: number;
  actions: { key: string; label: string }[];
}

export interface CountPair {
  total: number;
  reviewed: number;
}

export interface Alert {
  level: 'warning' | 'info' | 'error';
  code: string;
  message: string;
  count?: number;
  class?: string;
  lots?: string[];
}

export interface Dashboard {
  totals: {
    tracks: number;
    labeled: number;
    reviewed: number;
    unreviewed: number;
    progress: number;
    usable: number;
    usable_reviewed: number;
    doubtful: number;
    doubtful_unreviewed: number;
    defect_untyped: number;
  };
  verdicts: ({ verdict: Verdict | null; label: string } & CountPair)[];
  defects: ({ defect: string | null; label: string } & CountPair)[];
  lots: {
    lot_id: string;
    name: string;
    type: string;
    farm: string;
    session_date: string;
    status: string;
    total: number;
    reviewed: number;
    progress: number;
    doubtful: number;
    verdicts: Record<string, number>;
    defects: Record<string, number>;
  }[];
  lots_total: number;
  sessions: { farm: string; session_date: string; lots: number; total: number; reviewed: number; progress: number }[];
  alerts: Alert[];
  min_examples: number;
  active: ({ model: ModelRef } & { contradictions: number; hesitations: number; unreviewed: number; total: number }) | null;
}

export interface HistoryPage {
  items: HistoryEntry[];
  next_before_id: number | null;
  total: number | null;
}

// ---------------------------------------------------------------- Filtres
export type ReviewStatus = '' | 'a_relire' | 'relues';

export interface Filters {
  lots: string[];
  verdicts: string[]; // + 'aucun'
  defects: string[]; // + 'aucun'
  status: ReviewStatus;
  quality: '' | 'ok' | 'douteuse';
  modelId: string; // '' = modèle par défaut
  scoreMin: number | null;
  scoreMax: number | null;
}

export const EMPTY_FILTERS: Filters = {
  lots: [],
  verdicts: [],
  defects: [],
  status: '',
  quality: '',
  modelId: '',
  scoreMin: null,
  scoreMax: null,
};

export function activeFilterCount(f: Filters): number {
  return (
    (f.lots.length ? 1 : 0) +
    (f.verdicts.length ? 1 : 0) +
    (f.defects.length ? 1 : 0) +
    (f.status ? 1 : 0) +
    (f.quality ? 1 : 0) +
    (f.scoreMin !== null || f.scoreMax !== null ? 1 : 0)
  );
}

export function filterQuery(f: Filters): Record<string, QueryValue> {
  return {
    lot: f.lots.join(','),
    verdict: f.verdicts.join(','),
    defect: f.defects.join(','),
    reviewed: f.status === 'a_relire' ? false : f.status === 'relues' ? true : undefined,
    quality: f.quality,
    model_id: f.modelId,
    score_min: f.scoreMin,
    score_max: f.scoreMax,
  };
}

// ---------------------------------------------------------------- Appels
export const lk = {
  all: ['labeling'] as const,
  meta: ['labeling', 'meta'] as const,
  lots: ['labeling', 'lots'] as const,
  models: ['labeling', 'models'] as const,
  track: (id: number) => ['labeling', 'track', id] as const,
  dashboard: ['labeling', 'dashboard'] as const,
  history: (q: Record<string, QueryValue>) => ['labeling', 'history', q] as const,
  users: ['labeling', 'history-users'] as const,
};

export const labelingApi = {
  tracks: (query: Record<string, QueryValue>, signal?: AbortSignal) => api.get<TrackPage>('/labeling/tracks', undefined, { query, signal }),
  active: (query: Record<string, QueryValue>, signal?: AbortSignal) => api.get<TrackPage>('/labeling/active', undefined, { query, signal }),
  track: (id: number, modelId?: string, signal?: AbortSignal) =>
    api.get<TrackDetail>(`/labeling/tracks/${id}`, undefined, { query: { model_id: modelId }, signal }),
  setLabel: (id: number, body: { verdict?: Verdict | null; defect?: string | null; reviewed?: boolean; note?: string }) =>
    api.put<LabelWriteResult>(`/labeling/tracks/${id}/label`, body),
  confirm: (id: number) => api.post<LabelWriteResult>(`/labeling/tracks/${id}/confirm`),
  reopen: (id: number) => api.post<LabelWriteResult>(`/labeling/tracks/${id}/reopen`),
  markView: (id: number, view: number, visible?: boolean) =>
    api.post<{ defect_visible: boolean }>(`/labeling/tracks/${id}/views/${view}/defect-visible`, visible === undefined ? {} : { visible }),
  bulk: (body: { track_ids: number[]; verdict?: Verdict | null; defect?: string | null; confirm?: boolean }) =>
    api.post<BulkResult>('/labeling/bulk', body),
  undo: () => api.post<UndoResult>('/labeling/undo'),
  history: (query: Record<string, QueryValue>, signal?: AbortSignal) => api.get<HistoryPage>('/labeling/history', undefined, { query, signal }),
};

export function useLabelingMeta() {
  return useQuery({ queryKey: lk.meta, queryFn: ({ signal }) => api.get<LabelingMeta>('/labeling/meta', undefined, { signal }) });
}

export function useLots() {
  return useQuery({
    queryKey: lk.lots,
    queryFn: ({ signal }) => api.get<{ items: LotInfo[]; total: number }>('/labeling/lots', undefined, { signal, query: { limit: 2000 } }),
    staleTime: 60_000,
  });
}

export function useModels() {
  return useQuery({
    queryKey: lk.models,
    queryFn: ({ signal }) => api.get<{ items: ModelInfo[]; default: ModelRef | null }>('/labeling/models', undefined, { signal }),
    staleTime: 30_000,
  });
}

export function useDashboard() {
  return useQuery({ queryKey: lk.dashboard, queryFn: ({ signal }) => api.get<Dashboard>('/labeling/dashboard', undefined, { signal }) });
}

export function useTrackDetail(id: number | null, modelId: string) {
  return useQuery({
    queryKey: [...lk.track(id ?? -1), modelId] as const,
    queryFn: ({ signal }) => labelingApi.track(id as number, modelId || undefined, signal),
    enabled: id !== null,
    staleTime: 30_000,
  });
}

export function prefetchTrack(qc: QueryClient, id: number, modelId: string) {
  return qc.prefetchQuery({
    queryKey: [...lk.track(id), modelId] as const,
    queryFn: ({ signal }) => labelingApi.track(id, modelId || undefined, signal),
    staleTime: 30_000,
  });
}

/** Met à jour le label d'une piste dans le cache du détail (toutes variantes de modèle). */
export function patchCachedLabel(qc: QueryClient, id: number, label: Partial<LabelState>) {
  qc.setQueriesData<TrackDetail>({ queryKey: lk.track(id) }, (old) => (old ? { ...old, label: { ...old.label, ...label } } : old));
}

export function useInvalidateLabeling() {
  const qc = useQueryClient();
  return useCallback(() => {
    void qc.invalidateQueries({ queryKey: lk.dashboard });
    void qc.invalidateQueries({ queryKey: ['labeling', 'history'] });
    void qc.invalidateQueries({ queryKey: ['project'], exact: true });
  }, [qc]);
}

/** Précharge des images (cache navigateur). */
const preloaded = new Set<string>();
export function preloadImages(urls: string[]) {
  for (const u of urls) {
    if (preloaded.has(u)) continue;
    preloaded.add(u);
    const img = new Image();
    img.decoding = 'async';
    img.src = u;
  }
  if (preloaded.size > 4000) preloaded.clear();
}
