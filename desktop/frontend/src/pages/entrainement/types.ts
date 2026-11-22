/** Types du module Entraînement (contrat : docs/api/training.md). */
import type { Job } from '@/api/types';

export type Subset = 'train' | 'val' | 'test';
export type ModelStatus = 'created' | 'training' | 'trained' | 'failed' | 'cancelled';
export type ModelKind = 'classifier' | 'anomaly';
export type Objective = 'recall' | 'false_reject';

export interface PretrainedStatus {
  available: boolean;
  file: string;
  size_bytes?: number;
  source?: string;
  sha256?: string;
  fetched_at?: string;
}

export interface Architecture {
  id: string;
  label: string;
  description: string;
  params_m: number;
  int8_friendly: boolean;
  pretrained: PretrainedStatus;
}

export interface TrainParams {
  arch: string;
  aggregation: 'max' | 'attention';
  pretrained: boolean;
  input_size: number;
  n_views: number;
  epochs: number;
  batch_size: number;
  lr: number;
  backbone_lr_mult: number;
  weight_decay: number;
  warmup_epochs: number;
  min_lr_ratio: number;
  freeze_epochs: number;
  dropout: number;
  imbalance: 'pos_weight' | 'sampler' | 'none';
  augment: 'none' | 'light' | 'standard';
  monitor: 'val_auc' | 'val_ap' | 'val_loss';
  patience: number;
  min_delta: number;
  seed: number;
  device: 'auto' | 'cuda' | 'mps' | 'cpu';
  amp: boolean;
  num_workers: number;
  cache_mb: number;
  reviewed_only: boolean;
  grad_clip: number;
  preset?: string | null;
}

export interface AnomalyParams {
  arch: string;
  input_size: number;
  n_views: number;
  patches_per_view: number;
  bank_size: number;
  coreset: 'greedy' | 'random';
  projection_dim: number;
  max_train_eggs: number;
  batch_size: number;
  seed: number;
  device: string;
  reviewed_only: boolean;
  cache_mb: number;
}

export interface Preset {
  id: 'rapide' | 'standard' | 'int8';
  label: string;
  description: string;
  hparams: TrainParams;
}

export interface LotRow {
  id: string;
  name: string;
  type: string;
  farm: string;
  session_date: string;
  n_eggs: number;
  viable: number;
  non_viable: number;
  defects: Record<string, number>;
  subset?: Subset | null;
}

export interface Options {
  architectures: Architecture[];
  presets: Preset[];
  defaults: TrainParams;
  anomaly_defaults: AnomalyParams;
  choices: Record<'aggregation' | 'imbalance' | 'monitor' | 'augment' | 'device', string[]>;
  device: { kind: string; name: string };
  split: {
    farms: { farm: string; n_eggs: number }[];
    dates: { date: string; n_eggs: number }[];
    lots: LotRow[];
    defaults: SplitParams;
  };
}

export interface SplitParams {
  group_by: 'session' | 'lot';
  ratios: [number, number, number];
  holdout_farms: string[];
  holdout_dates: string[];
  seed: number;
  reviewed_only: boolean;
  exclude_lots: string[];
}

export interface SubsetCounts {
  n_lots: number;
  n_eggs: number;
  viable: number;
  non_viable: number;
  defects: Record<string, number>;
  fraction: number;
}

export interface SplitSummary {
  subsets: Record<Subset, SubsetCounts>;
  total_eggs: number;
  total_lots: number;
  classes: string[];
  leakage_check: {
    unit: 'lot' | 'session';
    lots_in_one_subset: boolean;
    shared_sessions: string[];
    ok: boolean;
    holdout_respected?: boolean;
  };
  warnings: string[];
}

export interface SplitPreview {
  assignments: Record<string, Subset>;
  summary: SplitSummary;
  lots: LotRow[];
}

export interface Split {
  id: number;
  name: string;
  version: number;
  created_at: string | null;
  strategy: Partial<SplitParams> & { manual_changes?: number; derived_from?: number };
  summary: SplitSummary;
  used_by: string[];
  locked: boolean;
  assignments?: Record<string, Subset>;
  new_version?: boolean;
}

export interface SplitDetail extends Split {
  current_summary: SplitSummary;
  lots: LotRow[];
  unassigned_lots: LotRow[];
}

export interface Paged<T> {
  total: number;
  items: T[];
}

export interface Epoch {
  epoch: number;
  train_loss: number;
  lr: number;
  lr_backbone: number;
  backbone_frozen: boolean;
  val_loss?: number;
  val_auc?: number | null;
  val_ap?: number | null;
  val_recall?: number | null;
  val_false_reject_rate?: number | null;
  duration_s: number;
  images_per_s: number;
  best?: boolean;
}

export interface SubsetMetrics {
  n: number;
  n_pos: number;
  n_neg: number;
  auc_roc: number | null;
  average_precision: number | null;
  recall: number | null;
  false_reject_rate: number | null;
  precision: number | null;
  tp: number;
  fn: number;
  fp: number;
  tn: number;
  recall_by_defect?: Record<string, number | null>;
}

export interface MetricsSummary {
  threshold?: number;
  generated_at?: string;
  train?: SubsetMetrics;
  val?: SubsetMetrics;
  test?: SubsetMetrics;
  test_set?: { n: number; fingerprint: string };
}

export interface ModelSummary {
  id: string;
  name: string;
  kind: ModelKind;
  arch: string;
  status: ModelStatus;
  number: number | null;
  created_at: string | null;
  split_id: number | null;
  parent_id: string | null;
  threshold: number;
  n_views: number;
  input_size: number;
  metrics: MetricsSummary;
  epochs_done: number;
  best_epoch: number | null;
  device: { kind?: string; name?: string; used?: string; amp?: boolean } | null;
  duration_s: number | null;
  warnings: string[];
  pretrained: { loaded?: boolean; warning?: string | null; n_tensors?: number; init_from?: string } | null;
  job_id: string | null;
  job_active: boolean;
  resumable: boolean;
  exported: boolean;
  export_outdated: boolean;
}

export interface DataSummary {
  split_id?: number;
  split_name?: string;
  split_version?: number;
  group_by?: string;
  reviewed_only?: boolean;
  counts?: Record<string, { n: number; viable: number; non_viable: number; defects: Record<string, number> }>;
  lots?: Record<string, string[]>;
  n_eggs?: number;
  labels_fingerprint?: string;
  bank_eggs?: number;
}

export interface ModelDetail extends ModelSummary {
  hparams: Partial<TrainParams> & Partial<AnomalyParams> & Record<string, unknown>;
  data_summary: DataSummary;
  history: Epoch[];
  export: Record<string, unknown>;
  extra: Record<string, unknown>;
}

export interface HistoryResponse {
  model_id: string;
  status: ModelStatus;
  epochs: number | null;
  history: Epoch[];
  job: Job | null;
}

export interface Confusion {
  threshold: number;
  tp: number;
  fn: number;
  fp: number;
  tn: number;
  n: number;
  n_pos: number;
  n_neg: number;
  recall: number | null;
  false_reject_rate: number | null;
  precision: number | null;
  specificity: number | null;
  accuracy: number | null;
  objective_met?: boolean | null;
}

export interface ReportBlock {
  n: number;
  n_pos: number;
  n_neg: number;
  auc_roc: number | null;
  average_precision: number | null;
  at_threshold: Confusion;
  roc?: { fpr: number[]; tpr: number[]; thresholds: (number | null)[] };
  pr?: { recall: number[]; precision: number[]; thresholds: (number | null)[] };
  histogram?: { edges: number[]; viable: number[]; non_viable: number[] };
  recall_by_defect?: Record<string, { n: number; detected: number; recall: number | null }>;
}

export interface Suggestion {
  objective: Objective;
  target: number;
  computed_on: 'val' | 'test';
  threshold: number;
  val?: Confusion;
  test?: Confusion;
  error?: string;
}

export interface Report {
  format: number;
  model_id: string;
  generated_at: string;
  threshold: number;
  subsets: Partial<Record<Subset, ReportBlock>>;
  test_set?: { n: number; fingerprint: string };
  threshold_suggestions: Suggestion[];
  training?: { epochs_done: number; best_epoch: number | null; stopped_early: boolean; monitor: string; duration_s: number; warnings: string[] };
  anomaly?: { bank_size: number; bank_eggs: number; calibration: Record<string, unknown>; duration_s: number; warnings: string[] };
}

export interface ErrorItem {
  track_id: number;
  lot_id: string;
  lot_name: string;
  lane: number;
  subset: Subset;
  score: number;
  severity: number;
  verdict: string;
  defect: string | null;
  reviewed: boolean;
  n_views: number;
  views: { index: number; score: number }[];
}

export interface ErrorsResponse {
  kind: 'fn' | 'fp';
  subset: Subset;
  threshold: number;
  total: number;
  limit: number;
  offset: number;
  items: ErrorItem[];
}

export interface CamMeta {
  method: 'grad_cam' | 'distance_patch';
  track_id: number;
  views: { index: number; score: number; peak: number; url: string }[];
}

export interface CompareModel {
  id: string;
  name: string;
  kind: ModelKind;
  arch: string;
  split_id: number | null;
  number: number | null;
  threshold: number;
  status: ModelStatus;
  test_set?: { n: number; n_pos: number; fingerprint: string };
  auc_roc?: number | null;
  average_precision?: number | null;
  own_threshold?: Confusion;
  recall_by_defect?: Record<string, { n: number; detected: number; recall: number | null }>;
  common_objective?: { threshold?: number; computed_on?: string; test?: Confusion; error?: string };
  error?: string;
}

export interface CompareResult {
  objective: { objective: Objective; target: number; computed_on: string };
  same_test_set: boolean;
  models: CompareModel[];
  warnings: string[];
}

export interface Started {
  model_id: string;
  job_id: string;
}
