/** Types du module Simulation (contrat : docs/api/simulation.md). */
import type { Job } from '@/api/types';

export type Outcome = 'ejected_ok' | 'missed' | 'missed_late' | 'false_reject' | 'accepted_ok';
export type SimStatus = 'created' | 'running' | 'done' | 'failed' | 'cancelled';
export type Variant = 'fp32' | 'int8';

export interface SimParams {
  model_id: string;
  variant: Variant | null;
  threshold: number | null;
  n_lanes: number;
  period_s: number;
  pitch_mm: number;
  speed_mm_s?: number | null;
  camera_zone_mm: number;
  camera_to_trap_mm: number;
  trap_actuation_ms: number;
  n_columns: number;
  defect_rate: number;
  seed: number | null;
  empty_lane_rate: number;
  injected_latency_ms: number;
  latency_jitter_ms: number;
  latency_scale: number;
  missing_view_rate: number;
  inference_threads: number | null;
  workers: number;
}

export interface Timing {
  speed_mm_s: number;
  pitch_mm: number;
  period_s: number;
  columns_per_min: number;
  eggs_per_hour: number;
  camera_time_s: number;
  budget_s: number;
  budget_ms: number;
  effective_budget_ms: number;
  trap_x_mm: number;
  columns_in_flight: number;
  virtual_duration_s: number;
  warnings: string[];
}

export interface Defaults {
  params: Omit<SimParams, 'model_id'>;
  timing: Timing;
  outcomes: { key: Outcome; label: string; color: string }[];
}

export interface TestSetSummary {
  split_id: number;
  n_test_lots: number;
  n_viable: number;
  n_non_viable: number;
  n_total: number;
  by_defect: Record<string, number>;
  excluded_no_verdict: number;
  excluded_no_views: number;
}

export interface EligibleModel {
  id: string;
  name: string;
  status: string;
  created_at: string | null;
  threshold: number;
  n_views: number;
  variants: string[];
  split_id: number | null;
  test_set: TestSetSummary | null;
  eligible: boolean;
  reason: string | null;
}

export interface SimSummary {
  id: string;
  name: string;
  model_id: string | null;
  model_name: string | null;
  status: SimStatus;
  created_at: string | null;
  job_id: string | null;
  n_columns: number | null;
  defect_rate: number | null;
  seed: number | null;
  threshold: number | null;
  variant: string | null;
  recall: number | null;
  false_reject_rate: number | null;
  late_rate: number | null;
  has_video: boolean;
  error: string | null;
}

export interface SimList {
  items: SimSummary[];
  total: number;
  limit: number;
  offset: number;
}

export interface LatencyStats {
  mean: number | null;
  median: number | null;
  p95: number | null;
  p99: number | null;
  max: number | null;
  min: number | null;
}

export interface DefectBreakdown {
  label: string;
  n: number;
  ejected_ok: number;
  missed: number;
  missed_late: number;
  recall: number | null;
}

export interface VideoInfo {
  columns: [number, number];
  speed: number;
  fps: number;
  width: number;
  height: number;
  n_frames: number;
  duration_s: number;
  encoder: string;
  size_bytes: number;
  path: string;
  rendered_at: string;
}

export interface SimResults {
  n_columns: number;
  threshold: number;
  variant: string | null;
  counts: {
    eggs: number;
    non_viable: number;
    viable: number;
    ejected_ok: number;
    missed: number;
    missed_late: number;
    missed_total: number;
    false_reject: number;
    accepted_ok: number;
    late_decisions: number;
    missed_traps: number;
    no_decision: number;
    bin_reject: number;
    bin_viable: number;
  };
  rates: {
    recall: number | null;
    miss_rate: number | null;
    miss_late_rate: number | null;
    false_reject_rate: number | null;
    late_rate: number | null;
    model_recall: number | null;
    model_false_reject_rate: number | null;
    residual_defect_rate: number | null;
    reject_bin_purity: number | null;
    accuracy: number | null;
  };
  by_defect: Record<string, DefectBreakdown>;
  viable: { label: string; n: number; false_reject: number; accepted_ok: number; false_reject_rate: number | null };
  latency: {
    budget_ms: number;
    effective_budget_ms: number;
    trap_actuation_ms: number;
    measured_ms: LatencyStats;
    compute_ms: LatencyStats;
    wait_ms: LatencyStats;
    response_ms: LatencyStats;
    min_margin_ms: number | null;
    late_columns: number;
    late_column_rate: number | null;
    histogram: { edges: number[]; counts: number[]; budget_ms: number; budget_in_range?: boolean };
  };
  throughput: {
    required_columns_per_s: number;
    capacity_columns_per_s: number | null;
    capacity_ratio: number | null;
    measured_columns_per_s: number | null;
    wall_s: number;
    wall_columns_per_s: number | null;
    virtual_duration_s: number;
  };
  composition: {
    n_eggs: number;
    n_slots: number;
    n_empty_slots: number;
    n_non_viable: number;
    n_viable: number;
    defect_rate_actual: number;
    n_distinct_eggs: number;
    reuse_rate: number;
    with_replacement: boolean;
    n_views_model: number;
    n_views_missing: number;
    n_eggs_without_views: number;
    pool: TestSetSummary;
  };
  model?: { id: string; name: string };
  video?: VideoInfo;
  video_job_id?: string;
  error?: string;
  finished_at?: string;
}

export interface SimDetail extends SimSummary {
  params: SimParams;
  results: Partial<SimResults>;
  timing: Timing | null;
  job: Job | null;
  video_job: Job | null;
}

export interface Geometry {
  n_lanes: number;
  period_s: number;
  pitch_mm: number;
  speed_mm_s: number;
  camera_zone_mm: number;
  camera_to_trap_mm: number;
  trap_x_mm: number;
  trap_actuation_ms: number;
  camera_time_s: number;
  budget_s: number;
  effective_budget_s: number;
}

export interface TimelineEgg {
  track_id: number;
  truth: 'viable' | 'non_viable';
  defect: string | null;
  n_views: number;
  score: number | null;
  decision: 0 | 1 | null;
  late: boolean;
  ejected: boolean;
  outcome: Outcome;
}

export interface TimelineColumn {
  k: number;
  t_enter: number;
  t_acq: number;
  t_decision: number;
  t_deadline: number;
  t_trap: number;
  measured_ms: number;
  injected_ms: number;
  compute_ms: number;
  wait_ms: number;
  response_ms: number;
  margin_ms: number;
  decided: boolean;
  late: boolean;
  /** Cumuls jusqu'à cette colonne incluse : [bac rebut, dont à tort, bac viables, dont non viables]. */
  cum: [number, number, number, number];
  lanes: (TimelineEgg | null)[];
}

export interface TimelineChunk {
  geometry: Geometry;
  threshold: number;
  total: number;
  start: number;
  columns: TimelineColumn[];
}

export interface EggDetail {
  column: number;
  lane: number;
  track_id: number;
  lot_id: string | null;
  lot_name: string | null;
  truth: 'viable' | 'non_viable';
  defect: string | null;
  current_label: { verdict: string | null; defect: string | null } | null;
  score: number | null;
  threshold: number;
  decision: 0 | 1 | null;
  late: boolean;
  ejected: boolean;
  outcome: Outcome;
  outcome_label: string;
  n_views: number;
  n_views_expected: number | null;
  views: { index: number; url: string; presented: boolean; score: number | null }[];
  timing: Record<
    't_enter' | 't_acq' | 't_decision' | 't_deadline' | 't_trap' | 'measured_ms' | 'injected_ms' | 'compute_ms' | 'wait_ms' | 'response_ms' | 'margin_ms',
    number
  >;
  labeling_url: string;
}

export interface CompareResult {
  simulations: { id: string; name: string; model_id: string | null; model_name: string | null; status: SimStatus; created_at: string | null }[];
  params: { key: string; values: unknown[]; differs: boolean }[];
  metrics: { key: string; label: string; format: 'int' | 'pct' | 'ms' | 'ratio'; better: '' | 'max' | 'min'; values: (number | null)[]; best: boolean[] | null }[];
  by_defect: { defect: string; label: string; values: ({ n: number; recall: number | null; ejected_ok: number } | null)[] }[];
}

export interface VideoOptions {
  start_column: number;
  end_column: number | null;
  speed: number | null;
  max_duration_s: number;
  fps: number;
  width: number;
  height: number;
}
