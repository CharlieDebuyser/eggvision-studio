/** Types du module Prise de vue — contrat : docs/api/capture.md. */

export type SourceKind = 'pi' | 'usb' | 'simulated';

export interface CameraSettings {
  exposure_us: number;
  analogue_gain: number;
  red_gain: number;
  blue_gain: number;
  fps: number;
  /** [x, y, largeur, hauteur] en pixels capteur ; null = plein capteur. */
  roi: [number, number, number, number] | null;
}

export interface CaptureStats {
  lot_id: string;
  status: string;
  running: boolean;
  active?: boolean;
  stop_reason: string | null;
  error: string | null;
  elapsed_s: number;
  duration_s: number;
  frames_received: number;
  frames_written: number;
  frames_queued: number;
  frames_dropped: number;
  frames_dropped_queue: number;
  frames_dropped_upstream: number;
  write_errors: number;
  drop_ratio: number;
  measured_fps: number;
  instant_fps: number;
  bytes_written: number;
  queue_size: number;
  queue_fill: number;
  queue_max_fill: number;
  disk_free_bytes: number | null;
  min_free_bytes: number;
  name?: string;
  type?: string;
}

export interface SourceStatus {
  kind: SourceKind;
  label: string;
  connected: boolean;
  ready?: boolean;
  error?: string | null;
  url?: string;
  version?: string | null;
  hostname?: string | null;
  camera?: {
    backend?: string;
    model?: string;
    sensor_width?: number;
    sensor_height?: number;
    output_width?: number;
    output_height?: number;
    settings?: CameraSettings;
    controls?: Record<string, boolean>;
    awb_running?: boolean;
  };
  actual_fps?: number;
  device?: { model: string; cpu_count: number; temperature_c: number | null } | null;
  storage?: { free_bytes?: number; total_bytes?: number; min_free_bytes?: number; path?: string } | null;
  clock_offset_s?: number;
  capturing?: boolean;
  capture?: CaptureStats | null;
}

export interface LastLot {
  lot_id: string;
  status: string | null;
  stop_reason: string | null;
  stats: { frames_written?: number; frames_dropped?: number; duration_s?: number; measured_fps?: number; bytes_written?: number } | null;
  kind: SourceKind;
  sync_job_id?: string;
}

export interface CurrentLot {
  lot_id: string;
  kind: SourceKind;
  name: string;
  type: string;
  agent_url?: string;
}

export interface CaptureStatus {
  connected: boolean;
  kind: SourceKind | null;
  source: SourceStatus | null;
  capturing: boolean;
  locked: boolean;
  current_lot: CurrentLot | null;
  last_lot: LastLot | null;
  agent_url: string | null;
  last_agent_url: string | null;
}

export interface LiveState {
  active: boolean;
  lot: CurrentLot | null;
  stats: CaptureStats | null;
  error: string | null;
  locked: boolean;
  last: LastLot | null;
  result?: { lot_id: string; manifest: unknown };
}

export interface DiscoveredAgent {
  name: string;
  hostname: string;
  addresses: string[];
  port: number;
  version: string | null;
  api: string | null;
  url: string;
}

export interface UsbDevice {
  device: number;
  label: string;
  width: number;
  height: number;
}

export interface Preset {
  name: string;
  settings: CameraSettings;
  updated_at: string | null;
}

export interface Detection {
  cx: number;
  cy: number;
  major: number;
  minor: number;
  angle: number;
  lane: number | null;
  partial: boolean | null;
}

export interface Analysis {
  width: number;
  height: number;
  sharpness: number;
  histogram: { bins: number; b: number[]; g: number[]; r: number[]; luma: number[] };
  saturated: { any: number; b: number; g: number; r: number };
  saturation_warning: boolean;
  mean: { b: number; g: number; r: number };
  dark_ratio: number;
  detections?: Detection[] | null;
  detections_error?: string | null;
  detections_frame?: { width: number; height: number };
  /** Image analysée (data URL JPEG), demandée avec les détections. */
  image?: string;
}

export interface LaneLine {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface Overlays {
  lane_lines: LaneLine[];
  grid: { enabled: boolean; cols: number; rows: number };
  show_lanes: boolean;
  show_detections: boolean;
}

export interface CaptureOptions {
  min_free_gb: number;
  auto_sync: boolean;
}

export interface ProjectLot {
  id: string;
  name: string;
  type: string;
  farm: string;
  operator: string;
  status: string;
  source_kind: string;
  sync_status: string;
  frame_count: number;
  dropped_frames: number;
  duration_s: number;
  fps: number;
  width: number;
  height: number;
  created_at: string | null;
  stop_reason: string | null;
}

export interface AgentLot {
  lot_id: string;
  name?: string;
  type?: string;
  farm?: string;
  operator?: string;
  status: string;
  created_at?: string;
  stats?: { frames_written: number; frames_dropped: number; duration_s: number; measured_fps: number; bytes_written: number };
  capturing?: boolean;
  in_project: boolean;
  sync_status: string | null;
  synced: boolean;
  sync_job_id: string | null;
}

export interface LotForm {
  name: string;
  type: string;
  farm: string;
  operator: string;
  comment: string;
}
