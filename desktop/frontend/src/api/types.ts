/**
 * Types du contrat d'API backend (FastAPI, préfixe /api).
 * Toute erreur HTTP renvoie `{ detail: string }` (message en français) → voir ApiError.
 */

export type ISODate = string;

// ---------------------------------------------------------------- Santé
export interface Health {
  status: 'ok';
  version: string;
}

// ---------------------------------------------------------------- Projet
export interface ProjectStats {
  lots: number;
  frames: number;
  tracks: number;
  tracks_reviewed: number;
  models: number;
  simulations: number;
}

export interface ProjectInfo {
  open: boolean;
  path?: string;
  name?: string;
  created_at?: ISODate;
  stats?: ProjectStats;
}

export interface RecentProject {
  path: string;
  name: string;
  last_opened: ISODate;
  exists: boolean;
}

export interface CreateProjectBody {
  parent_dir: string;
  name: string;
}

export type DemoSize = 'mini' | 'standard';

export interface CreateDemoBody {
  parent_dir: string;
  name: string;
  size: DemoSize;
}

export interface JobRef {
  job_id: string;
}

/** Contenu libre de config.yaml. */
export type ProjectConfig = Record<string, unknown>;

// ---------------------------------------------------------------- Système de fichiers
export interface FsEntry {
  name: string;
  path: string;
  is_dir: boolean;
  is_project: boolean;
}

export interface FsListing {
  path: string;
  parent: string | null;
  entries: FsEntry[];
}

export interface MkdirBody {
  parent_dir: string;
  name: string;
}

export interface FsHome {
  home: string;
  documents: string;
}

// ---------------------------------------------------------------- Tâches (jobs)
export type JobStatus = 'queued' | 'running' | 'done' | 'failed' | 'cancelled';

export const JOB_KINDS = [
  'demo',
  'sync',
  'capture',
  'detection',
  'detector_training',
  'training',
  'evaluation',
  'export',
  'bench',
  'simulation',
  'video',
] as const;
/** Types connus ; le backend peut en ajouter (d'où `string`). */
export type JobKind = (typeof JOB_KINDS)[number] | (string & {});

export interface Job<P = Record<string, unknown>, R = Record<string, unknown>> {
  id: string;
  kind: JobKind;
  title: string;
  status: JobStatus;
  /** 0..1 */
  progress: number;
  message: string;
  params: P;
  result: R | null;
  error: string | null;
  created_at: ISODate;
  started_at: ISODate | null;
  ended_at: ISODate | null;
}

export interface JobFilter {
  kind?: string;
  status?: JobStatus;
  limit?: number;
}

export interface JobLogChunk {
  text: string;
  next_offset: number;
  eof: boolean;
}

// ---------------------------------------------------------------- Diagnostic
export type DeviceKind = 'cuda' | 'mps' | 'cpu';

export interface AgentStatus {
  connected: boolean;
  url?: string;
  version?: string;
  detail?: string;
}

export interface Diagnostics {
  app_version: string;
  python: string;
  platform: string;
  machine: string;
  packages: Record<string, string | null>;
  device: { kind: DeviceKind; name: string };
  agent: AgentStatus | null;
  project: { path: string; name: string } | null;
  log_file: string;
  data_dir: string;
}

export interface LogText {
  text: string;
}
