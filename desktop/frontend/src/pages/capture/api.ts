/** Appels API et hooks React Query du module Prise de vue (docs/api/capture.md). */
import { keepPreviousData, useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { api, apiUrl } from '@/api/client';
import { queryKeys } from '@/api/hooks';
import type { Job } from '@/api/types';
import type {
  AgentLot,
  Analysis,
  CameraSettings,
  CaptureOptions,
  CaptureStatus,
  DiscoveredAgent,
  LiveState,
  LotForm,
  Overlays,
  Preset,
  ProjectLot,
  SourceKind,
  UsbDevice,
} from './types';

export const capKeys = {
  all: ['capture'] as const,
  status: ['capture', 'status'] as const,
  settings: ['capture', 'settings'] as const,
  live: ['capture', 'live'] as const,
  presets: ['capture', 'presets'] as const,
  overlays: ['capture', 'overlays'] as const,
  options: ['capture', 'options'] as const,
  analysis: (detections: boolean) => ['capture', 'analysis', detections] as const,
  lots: (page: number, size: number) => ['capture', 'lots', page, size] as const,
  lotsAll: ['capture', 'lots'] as const,
  agentLots: ['capture', 'agent-lots'] as const,
  usb: ['capture', 'usb'] as const,
  syncJobs: ['capture', 'sync-jobs'] as const,
};

export const previewUrl = {
  mjpeg: (width: number, nonce: number) => apiUrl('/capture/preview.mjpeg', { fps: 12, width, quality: 75, n: nonce }),
  jpeg: (width: number, nonce: number) => apiUrl('/capture/preview.jpg', { width, quality: 80, n: nonce }),
};

export function useCaptureStatus() {
  return useQuery({
    queryKey: capKeys.status,
    queryFn: ({ signal }) => api.get<CaptureStatus>('/capture/status', undefined, { signal }),
    refetchInterval: 2000,
  });
}

export function useLive(enabled: boolean) {
  return useQuery({
    queryKey: capKeys.live,
    queryFn: ({ signal }) => api.get<LiveState>('/capture/lots/current', undefined, { signal }),
    refetchInterval: enabled ? 1000 : 5000,
  });
}

export function useSettings(enabled: boolean) {
  return useQuery({
    queryKey: capKeys.settings,
    queryFn: ({ signal }) => api.get<CameraSettings>('/capture/settings', undefined, { signal }),
    enabled,
    retry: false,
  });
}

export function useAnalysis(enabled: boolean, detections: boolean) {
  return useQuery({
    queryKey: capKeys.analysis(detections),
    queryFn: ({ signal }) => api.get<Analysis>('/capture/preview/analysis', undefined, { signal, query: { detections, image: detections, bins: 64, width: 640 } }),
    enabled,
    refetchInterval: enabled ? 350 : false,
    retry: false,
    placeholderData: keepPreviousData,
  });
}

export function usePresets() {
  return useQuery({ queryKey: capKeys.presets, queryFn: ({ signal }) => api.get<Preset[]>('/capture/presets', undefined, { signal }) });
}

export function useOverlays() {
  return useQuery({ queryKey: capKeys.overlays, queryFn: ({ signal }) => api.get<Overlays>('/capture/overlays', undefined, { signal }) });
}

export function useOptions() {
  return useQuery({ queryKey: capKeys.options, queryFn: ({ signal }) => api.get<CaptureOptions>('/capture/options', undefined, { signal }) });
}

export function useUsbDevices(enabled: boolean) {
  return useQuery({
    queryKey: capKeys.usb,
    queryFn: ({ signal }) => api.get<UsbDevice[]>('/capture/usb/devices', undefined, { signal }),
    enabled,
    staleTime: 30_000,
  });
}

export function useProjectLots(page: number, size: number) {
  return useQuery({
    queryKey: capKeys.lots(page, size),
    queryFn: ({ signal }) =>
      api.get<{ total: number; items: ProjectLot[] }>('/capture/lots', undefined, { signal, query: { limit: size, offset: page * size } }),
    placeholderData: keepPreviousData,
    refetchInterval: (q) => (q.state.data?.items.some((l) => l.status === 'en_cours' || l.sync_status === 'syncing' || l.sync_status === 'pending') ? 2000 : 10000),
  });
}

export function useAgentLots(enabled: boolean) {
  return useQuery({
    queryKey: capKeys.agentLots,
    queryFn: ({ signal }) => api.get<{ agent_url: string; total: number; unsynced: number; items: AgentLot[] }>('/capture/agent/lots', undefined, { signal }),
    enabled,
    retry: false,
    refetchInterval: enabled ? 3000 : false,
  });
}

export function useSyncJobs() {
  return useQuery({
    queryKey: capKeys.syncJobs,
    queryFn: ({ signal }) => api.get<Job[]>('/capture/sync', undefined, { signal, query: { limit: 20 } }),
    refetchInterval: (q) => (q.state.data?.some((j) => j.status === 'queued' || j.status === 'running') ? 1500 : 5000),
  });
}

export const discoverAgents = (timeout = 2) =>
  api.get<{ agents: DiscoveredAgent[] }>('/capture/discover', undefined, { query: { timeout } });

/** Invalide tout l'état du module (après connexion, lot, synchro…). */
export function useInvalidateCapture() {
  const qc = useQueryClient();
  return () => {
    qc.invalidateQueries({ queryKey: capKeys.all });
    qc.invalidateQueries({ queryKey: queryKeys.project });
  };
}

export function useConnect() {
  const invalidate = useInvalidateCapture();
  return useMutation({
    mutationFn: (body: { kind: SourceKind; url?: string; device?: number | string; scale?: number; folder?: string }) =>
      api.post<CaptureStatus>('/capture/connect', body),
    onSettled: invalidate,
  });
}

export function useDisconnect() {
  const invalidate = useInvalidateCapture();
  return useMutation({ mutationFn: () => api.post<CaptureStatus>('/capture/disconnect'), onSettled: invalidate });
}

export function useSaveSettings() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (changes: Partial<CameraSettings>) => api.put<CameraSettings>('/capture/settings', changes),
    onSuccess: (data) => {
      qc.setQueryData(capKeys.settings, data);
      qc.invalidateQueries({ queryKey: capKeys.status });
    },
  });
}

export function useAwb() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (duration_s: number) =>
      api.post<{ red_gain: number; blue_gain: number; settings: CameraSettings }>('/capture/awb', { duration_s }),
    onSuccess: (data) => qc.setQueryData(capKeys.settings, data.settings),
  });
}

export function useSavePreset() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ name, settings }: { name: string; settings?: CameraSettings }) =>
      api.put<Preset>(`/capture/presets/${encodeURIComponent(name)}`, settings ? { settings } : undefined),
    onSettled: () => qc.invalidateQueries({ queryKey: capKeys.presets }),
  });
}

export function useDeletePreset() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (name: string) => api.del(`/capture/presets/${encodeURIComponent(name)}`),
    onSettled: () => qc.invalidateQueries({ queryKey: capKeys.presets }),
  });
}

export function useApplyPreset() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (name: string) => api.post<CameraSettings>(`/capture/presets/${encodeURIComponent(name)}/apply`),
    onSuccess: (data) => qc.setQueryData(capKeys.settings, data),
  });
}

export function useSaveOverlays() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (o: Overlays) => api.put<Overlays>('/capture/overlays', o),
    onSuccess: (data) => qc.setQueryData(capKeys.overlays, data),
  });
}

export function useSaveOptions() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (o: CaptureOptions) => api.put<CaptureOptions>('/capture/options', o),
    onSuccess: (data) => qc.setQueryData(capKeys.options, data),
  });
}

export function useStartLot() {
  const invalidate = useInvalidateCapture();
  return useMutation({ mutationFn: (form: LotForm) => api.post<LiveState>('/capture/lots', form), onSettled: invalidate });
}

export function useStopLot() {
  const invalidate = useInvalidateCapture();
  return useMutation({ mutationFn: () => api.post<LiveState>('/capture/lots/stop'), onSettled: invalidate });
}

export function useSyncLot() {
  const invalidate = useInvalidateCapture();
  return useMutation({ mutationFn: (lotId: string) => api.post<Job>(`/capture/lots/${encodeURIComponent(lotId)}/sync`), onSettled: invalidate });
}

export function useDeleteAgentLot() {
  const invalidate = useInvalidateCapture();
  return useMutation({
    mutationFn: ({ lotId, force }: { lotId: string; force?: boolean }) =>
      api.del(`/capture/agent/lots/${encodeURIComponent(lotId)}`, undefined, { query: { force: force ? true : undefined } }),
    onSettled: invalidate,
  });
}
