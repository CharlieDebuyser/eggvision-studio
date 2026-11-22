/** Types du module Export (contrat : docs/api/export.md). */
import type { Job } from '@/api/types';

export interface VerifyReport {
  n_eggs: number;
  max_abs_diff: number;
  max_abs_diff_views: number;
  tolerance: number;
  ok: boolean;
  scores?: number[];
}

export interface DecisionMetrics {
  auc_roc: number | null;
  recall: number | null;
  false_reject_rate: number | null;
  tp: number;
  fn: number;
  fp: number;
  tn: number;
}

export interface Int8Info {
  format?: string;
  activations?: string;
  weights?: string;
  calibration?: string;
  n_calibration_eggs?: number;
  excluded_nodes?: string[];
  size_bytes?: number;
  vs_fp32?: { max_abs_diff: number; mean_abs_diff: number; max_abs_diff_views: number };
  test?: { n: number; fp32: DecisionMetrics; int8: DecisionMetrics; max_abs_diff: number };
  alert?: string;
  error?: string;
}

export interface ParityFp32 {
  tolerance: number;
  max_abs_diff: number;
  ok: boolean;
  n_eggs: number;
  by_n_views: Record<string, { n_eggs: number; max_abs_diff: number; max_abs_diff_views: number }>;
}

export interface BenchResult {
  n_columns: number;
  n_lanes: number;
  n_views: number;
  input_size: number;
  variant: string | null;
  engine: string;
  threads: number | null;
  latency_ms: { mean: number; median: number; p95: number; p99: number; min: number; max: number };
  preprocess_ms_mean: number;
  inference_ms_mean: number;
  columns_per_second: number | null;
  parity: VerifyReport | null;
  device: Record<string, unknown>;
  onnxruntime: string;
}

export interface BenchEntry {
  at: string;
  target: 'local' | 'pi';
  variant: 'fp32' | 'int8';
  n_columns: number;
  threads: number | null;
  agent_url: string | null;
  export_version: string | null;
  result: BenchResult;
}

export interface ExportInfo {
  package_dir?: string;
  zip?: string;
  zip_size_bytes?: number;
  variants?: ('fp32' | 'int8')[];
  files?: Record<string, { file: string; size_bytes: number }>;
  parity?: { fp32: ParityFp32; int8: Int8Info | null; package_verify: { fp32: VerifyReport | null; int8: VerifyReport | null } };
  threshold?: number;
  version?: string;
  export_count?: number;
  opset?: number;
  n_views?: number;
  input_size?: number;
  exported_at?: string;
  duration_s?: number;
  warnings?: string[];
  bench?: BenchEntry[];
}

export interface ExportStatus {
  model_id: string;
  exportable: boolean;
  exported: boolean;
  outdated: boolean;
  current_threshold: number;
  export: ExportInfo;
  active_jobs: Job[];
}

export interface BenchHistory {
  agent_url: string | null;
  items: BenchEntry[];
}

export interface CaptureStatus {
  connected: boolean;
  kind: string | null;
  agent_url: string | null;
  last_agent_url?: string | null;
  capturing?: boolean;
  source?: Record<string, unknown> | null;
}
