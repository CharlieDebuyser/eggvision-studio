/** Types du module « Détection et suivi » (contrat : docs/api/detection.md). */

export type Direction = 'up' | 'down' | 'left' | 'right';
export type Point = [number, number];

export interface Lane {
  index: number;
  polygon: Point[];
}

export interface Scene {
  image_width: number;
  image_height: number;
  direction: Direction;
  mm_per_px: number;
  lanes: Lane[];
  reference?: { lot_id: string; frame_index: number } | null;
}

export interface SceneResponse {
  scene: Scene | null;
  configured: boolean;
  directions: Record<Direction, string>;
  lot_scene?: Scene;
}

export interface Detection {
  cx: number;
  cy: number;
  major: number;
  minor: number;
  angle: number;
  lane: number;
  partial: boolean;
  sharpness: number;
  score: number;
  id?: number;
  track_id?: number | null;
  frame_index?: number;
}

export type ParamValue = number | string | boolean | null;
export type ParamValues = Record<string, ParamValue>;

export interface ParamField {
  name: string;
  label: string;
  description: string;
  unit: string;
  group: string;
  advanced: boolean;
  step: number | null;
  default: ParamValue;
  type: 'number' | 'integer' | 'boolean' | 'enum' | 'string';
  min?: number;
  max?: number;
  options?: string[];
  nullable?: boolean;
}

export interface ParamsResponse {
  detection: ParamValues;
  tracking: ParamValues;
  defaults: { detection: ParamValues; tracking: ParamValues };
  schema: { detection: ParamField[]; tracking: ParamField[] };
}

export interface PreviewResponse {
  lot_id: string;
  frame_index: number;
  width: number;
  height: number;
  detections: Detection[];
  count: number;
  complete: number;
  elapsed_ms: number;
  threshold: number | null;
  scene: Scene;
  mask_png?: string;
}

export type DetectionStatus = 'none' | 'running' | 'done' | 'error';

export interface DetLot {
  id: string;
  name: string;
  type: string;
  farm: string;
  session_date: string;
  status: string;
  frame_count: number;
  width: number;
  height: number;
  detection_status: DetectionStatus;
  track_count: number;
  tracks_ok: number;
  tracks_doubtful: number;
  usable: number;
  usable_rate: number | null;
  reviewed: number;
  method: string | null;
  ms_per_frame: number | null;
  speed_mm_s: number | null;
  error: string | null;
  job_id: string | null;
}

export interface DetLotList {
  total: number;
  items: DetLot[];
  min_views: number;
}

export interface FrameList {
  total: number;
  items: { index: number; seq: number; t: number }[];
}

export interface TrackRow {
  id: number;
  lot_id: string;
  lane: number;
  start_frame: number;
  end_frame: number;
  start_ts: number;
  end_ts: number;
  n_detections: number;
  n_views: number;
  quality: 'ok' | 'douteuse';
  flags: string[];
  mean_sharpness: number;
  partial_ratio: number;
  speed_px_s: number;
  mean_major_px: number;
  mean_minor_px: number;
  entered: boolean | null;
  exited: boolean | null;
}

export interface TrackList {
  total: number;
  items: TrackRow[];
  flag_labels: Record<string, string>;
}

export interface TrackDetail extends TrackRow {
  extra: Record<string, unknown>;
  flag_labels: Record<string, string>;
  views: { index: number; frame_index: number; sharpness: number; partial: boolean; url: string }[];
  detections: Detection[];
  label: { verdict: string | null; defect: string | null; source: string; reviewed: boolean } | null;
}

export interface RunStats {
  tracks: number;
  ok: number;
  doubtful: number;
  usable: number;
  frames: number;
  missing_files: number;
  detections: number;
  speed_mm_s: number | null;
  ms_per_frame: number;
  detect_ms_per_frame: number;
  read_ms_per_frame: number;
  elapsed_s: number;
}

export interface LotStats {
  lot_id: string;
  detection_status: DetectionStatus;
  min_views: number;
  tracks: number;
  ok: number;
  doubtful: number;
  usable: number;
  usable_rate: number | null;
  complete_tracks: number;
  usable_rate_complete: number | null;
  flags: Record<string, number>;
  flag_labels: Record<string, string>;
  lanes: Record<string, { tracks: number; usable: number }>;
  views_histogram: Record<string, number>;
  run: RunStats | null;
  method: string | null;
  finished_at: string | null;
}

export interface Box {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface AnnotationSummary {
  id: string;
  lot_id: string;
  frame_index: number;
  width: number;
  height: number;
  source: 'auto' | 'corrige';
  updated_at: string;
  n_boxes: number;
}

export interface Annotation extends Omit<AnnotationSummary, 'n_boxes'> {
  boxes: Box[];
}

export interface AnnotationList {
  total: number;
  items: AnnotationSummary[];
  n_boxes: number;
  n_corrected: number;
}

export interface DetectorMeta {
  id: string;
  name: string;
  arch: string;
  created_at: string;
  input_width: number;
  input_height: number;
  epochs: number;
  n_images: number;
  n_boxes: number;
  train_seconds: number;
  device: string;
  metrics: { recall: number; precision: number; subset: string; tp: number; fp: number; fn: number };
}

export interface DetectorInfo {
  detectors: DetectorMeta[];
  active: string | null;
  method: 'classic' | 'nn';
  annotations: { total: number; n_boxes: number; n_corrected: number };
}
