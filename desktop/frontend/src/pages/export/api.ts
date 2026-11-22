/** Appels API et hooks du module Export (docs/api/export.md). */
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { api, apiUrl } from '@/api/client';
import type { BenchHistory, CaptureStatus, ExportStatus } from './types';

const enc = encodeURIComponent;

export const exportKeys = {
  all: ['export'] as const,
  status: (id: string) => ['export', 'status', id] as const,
  bench: (id: string) => ['export', 'bench', id] as const,
  agent: ['export', 'agent'] as const,
};

export const downloadUrl = (id: string) => apiUrl(`/export/models/${enc(id)}/download`);

export function useExportStatus(id: string | undefined) {
  return useQuery({
    queryKey: exportKeys.status(id ?? ''),
    queryFn: ({ signal }) => api.get<ExportStatus>(`/export/models/${enc(id!)}`, undefined, { signal }),
    enabled: !!id,
    refetchInterval: (q) => (q.state.data?.active_jobs.length ? 1500 : false),
  });
}

export function useBenchHistory(id: string, live: boolean) {
  return useQuery({
    queryKey: exportKeys.bench(id),
    queryFn: ({ signal }) => api.get<BenchHistory>(`/export/models/${enc(id)}/bench`, undefined, { signal }),
    refetchInterval: live ? 2000 : 10_000,
  });
}

/** État de la source de prise de vue (agent Pi connecté ?) — API du module Prise de vue. */
export function useAgentStatus() {
  return useQuery({
    queryKey: exportKeys.agent,
    queryFn: ({ signal }) => api.get<CaptureStatus>('/capture/status', undefined, { signal }),
    refetchInterval: 5000,
    retry: false,
  });
}

export function useStartExport() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, ...body }: { id: string; variants: ('fp32' | 'int8')[]; n_reference: number; n_calibration_eggs: number; n_parity_eggs: number }) =>
      api.post<{ job_id: string }>(`/export/models/${enc(id)}`, body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: exportKeys.all });
      qc.invalidateQueries({ queryKey: ['training', 'models'] });
    },
  });
}

export function useStartBench() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, ...body }: { id: string; target: 'local' | 'pi'; variant: 'fp32' | 'int8'; n_columns: number; threads: number | null }) =>
      api.post<{ job_id: string }>(`/export/models/${enc(id)}/bench`, body),
    onSuccess: (_d, v) => {
      qc.invalidateQueries({ queryKey: exportKeys.status(v.id) });
      qc.invalidateQueries({ queryKey: exportKeys.bench(v.id) });
    },
  });
}
