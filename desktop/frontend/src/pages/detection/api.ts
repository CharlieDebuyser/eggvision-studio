/** Appels API et hooks React Query du module « Détection et suivi » (préfixe /api/detection). */
import { useEffect, useState } from 'react';
import { keepPreviousData, useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { api } from '@/api/client';
import { queryKeys } from '@/api/hooks';
import type { Job } from '@/api/types';
import type {
  Annotation,
  AnnotationList,
  Box,
  DetectorInfo,
  DetLotList,
  FrameList,
  LotStats,
  ParamsResponse,
  ParamValues,
  PreviewResponse,
  Scene,
  SceneResponse,
  TrackDetail,
  TrackList,
} from './types';

const enc = encodeURIComponent;

export const detKeys = {
  all: ['detection'] as const,
  scene: (lotId?: string) => ['detection', 'scene', lotId ?? ''] as const,
  params: ['detection', 'params'] as const,
  lots: (page: number, size: number, q: string) => ['detection', 'lots', page, size, q] as const,
  lotsAll: ['detection', 'lots'] as const,
  frames: (lotId: string, offset: number, limit: number) => ['detection', 'frames', lotId, offset, limit] as const,
  preview: (body: unknown) => ['detection', 'preview', body] as const,
  tracks: (lotId: string, filter: unknown) => ['detection', 'tracks', lotId, filter] as const,
  track: (id: number) => ['detection', 'track', id] as const,
  stats: (lotId: string) => ['detection', 'stats', lotId] as const,
  frameDets: (lotId: string, frame: number) => ['detection', 'frameDets', lotId, frame] as const,
  detector: ['detection', 'detector'] as const,
  annotations: (page: number) => ['detection', 'annotations', page] as const,
  annotation: (id: string) => ['detection', 'annotation', id] as const,
};

/** Valeur « retardée » : suit `value` après `delay` ms sans changement (aperçu en direct). */
export function useDebounced<T>(value: T, delay = 250): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setV(value), delay);
    return () => clearTimeout(t);
  }, [value, delay]);
  return v;
}

// ---------------------------------------------------------------- Scène
export function useScene(lotId?: string) {
  return useQuery({
    queryKey: detKeys.scene(lotId),
    queryFn: ({ signal }) => api.get<SceneResponse>('/detection/scene', undefined, { signal, query: { lot_id: lotId } }),
  });
}

export function useSaveScene() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (scene: Scene) => api.put<{ scene: Scene }>('/detection/scene', scene),
    onSuccess: () => void qc.invalidateQueries({ queryKey: ['detection', 'scene'] }),
  });
}

export const proposeLanes = (body: {
  lot_id: string;
  mode: 'regular' | 'detections';
  n_lanes: number;
  direction: string;
  mm_per_px?: number;
  frame_index?: number;
}) => api.post<{ lanes: Scene['lanes']; n_points: number; mode: string }>('/detection/scene/propose-lanes', body);

export const rulerScale = (p1: [number, number], p2: [number, number], length_mm: number) =>
  api.post<{ mm_per_px: number; distance_px: number }>('/detection/scene/ruler', { p1, p2, length_mm });

// ---------------------------------------------------------------- Lots et images
export function useDetLots(page: number, size: number, q = '', poll = false) {
  return useQuery({
    queryKey: detKeys.lots(page, size, q),
    queryFn: ({ signal }) =>
      api.get<DetLotList>('/detection/lots', undefined, { signal, query: { limit: size, offset: page * size, q } }),
    placeholderData: keepPreviousData,
    refetchInterval: poll ? 2000 : false,
  });
}

export function useFrames(lotId: string | null, offset = 0, limit = 1) {
  return useQuery({
    queryKey: detKeys.frames(lotId ?? '', offset, limit),
    queryFn: ({ signal }) =>
      api.get<FrameList>(`/detection/lots/${enc(lotId!)}/frames`, undefined, { signal, query: { offset, limit } }),
    enabled: !!lotId,
  });
}

// ---------------------------------------------------------------- Paramètres et aperçu
export function useParams() {
  return useQuery({
    queryKey: detKeys.params,
    queryFn: ({ signal }) => api.get<ParamsResponse>('/detection/params', undefined, { signal }),
  });
}

export function useSaveParams() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: { detection?: ParamValues; tracking?: ParamValues }) => api.put<ParamsResponse>('/detection/params', body),
    onSuccess: (data) => qc.setQueryData(detKeys.params, data),
  });
}

export function useResetParams() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (section: 'detection' | 'tracking' | 'all') => api.post<ParamsResponse>('/detection/params/reset', { section }),
    onSuccess: (data) => qc.setQueryData(detKeys.params, data),
  });
}

export interface PreviewBody {
  lot_id: string;
  frame_index: number;
  params?: ParamValues;
  scene?: Scene;
  mask?: boolean;
  mask_width?: number;
}

export function usePreview(body: PreviewBody | null) {
  return useQuery({
    queryKey: detKeys.preview(body),
    queryFn: ({ signal }) => api.post<PreviewResponse>('/detection/preview', body, { signal }),
    enabled: !!body,
    placeholderData: keepPreviousData,
    staleTime: 60_000,
    retry: false,
  });
}

// ---------------------------------------------------------------- Traitement
export const runCheck = (lot_ids: string[]) =>
  api.post<{ needs_force: boolean; lots: { lot_id: string; reviewed: number }[] }>('/detection/run/check', { lot_ids });

export function useRun() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: { lot_ids: string[]; force?: boolean }) => api.post<Job>('/detection/run', body),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: detKeys.lotsAll });
      void qc.invalidateQueries({ queryKey: queryKeys.jobsAll });
    },
  });
}

// ---------------------------------------------------------------- Pistes
export interface TrackFilter {
  quality?: 'ok' | 'douteuse' | '';
  flag?: string;
  lane?: number | null;
  order?: 'start' | 'quality' | 'views' | 'sharpness';
  page: number;
  size: number;
}

export function useTracks(lotId: string | null, f: TrackFilter) {
  return useQuery({
    queryKey: detKeys.tracks(lotId ?? '', f),
    queryFn: ({ signal }) =>
      api.get<TrackList>(`/detection/lots/${enc(lotId!)}/tracks`, undefined, {
        signal,
        query: { quality: f.quality, flag: f.flag, lane: f.lane ?? undefined, order: f.order, limit: f.size, offset: f.page * f.size },
      }),
    enabled: !!lotId,
    placeholderData: keepPreviousData,
  });
}

export function useTrack(id: number | null) {
  return useQuery({
    queryKey: detKeys.track(id ?? 0),
    queryFn: ({ signal }) => api.get<TrackDetail>(`/detection/tracks/${id}`, undefined, { signal }),
    enabled: id !== null,
  });
}

export function useLotStats(lotId: string | null) {
  return useQuery({
    queryKey: detKeys.stats(lotId ?? ''),
    queryFn: ({ signal }) => api.get<LotStats>(`/detection/lots/${enc(lotId!)}/stats`, undefined, { signal }),
    enabled: !!lotId,
  });
}

// ---------------------------------------------------------------- Détecteur entraînable
export function useDetector() {
  return useQuery({
    queryKey: detKeys.detector,
    queryFn: ({ signal }) => api.get<DetectorInfo>('/detection/detector', undefined, { signal }),
  });
}

export function useAnnotations(page: number, size: number) {
  return useQuery({
    queryKey: detKeys.annotations(page),
    queryFn: ({ signal }) =>
      api.get<AnnotationList>('/detection/detector/annotations', undefined, { signal, query: { limit: size, offset: page * size } }),
    placeholderData: keepPreviousData,
  });
}

export function useAnnotation(id: string | null) {
  return useQuery({
    queryKey: detKeys.annotation(id ?? ''),
    queryFn: ({ signal }) => api.get<Annotation>(`/detection/detector/annotations/${enc(id!)}`, undefined, { signal }),
    enabled: !!id,
  });
}

function useDetectorMutation<B, R>(fn: (b: B) => Promise<R>) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: fn,
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: detKeys.detector });
      void qc.invalidateQueries({ queryKey: ['detection', 'annotations'] });
      void qc.invalidateQueries({ queryKey: ['detection', 'annotation'] });
      void qc.invalidateQueries({ queryKey: detKeys.params });
    },
  });
}

export const useAddAnnotations = () =>
  useDetectorMutation((b: { lot_id: string; n: number }) => api.post<{ added: number }>('/detection/detector/annotations', b));
export const useSaveBoxes = () =>
  useDetectorMutation((b: { id: string; boxes: Box[] }) => api.put<Annotation>(`/detection/detector/annotations/${enc(b.id)}`, { boxes: b.boxes }));
export const useDeleteAnnotation = () =>
  useDetectorMutation((id: string) => api.del<{ deleted: string }>(`/detection/detector/annotations/${enc(id)}`));
export const useTrainDetector = () =>
  useDetectorMutation((b: { epochs: number; input_width: number; name: string; activate: boolean }) => api.post<Job>('/detection/detector/train', b));
export const useActivateDetector = () =>
  useDetectorMutation((b: { detector_id?: string | null; use: boolean }) => api.post<DetectorInfo>('/detection/detector/activate', b));
export const useDeleteDetector = () => useDetectorMutation((id: string) => api.del<{ deleted: string }>(`/detection/detector/${enc(id)}`));
