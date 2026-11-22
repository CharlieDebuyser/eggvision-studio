/**
 * Faux backend pour le développement (VITE_MOCK=1 npm run dev, ou npm run dev:mock).
 * Middleware Vite : répond à /api/* avec un état en mémoire, conforme au contrat d'API.
 * Il n'est PAS inclus dans le bundle de production (utilisé seulement par vite.config.ts).
 *
 * Endpoints d'un module : créer `mock/<module>.ts` exportant
 *   export default function register(route: RegisterRoute) { route('GET', '/api/lots', () => [...]); }
 * puis l'importer et l'ajouter à MODULE_MOCKS ci-dessous (une ligne chacun).
 */
import type { IncomingMessage, ServerResponse } from 'node:http';
import type { Plugin } from 'vite';

// ---------------------------------------------------------------- Types locaux (contrat)
type JobStatus = 'queued' | 'running' | 'done' | 'failed' | 'cancelled';
interface MockJob {
  id: string;
  kind: string;
  title: string;
  status: JobStatus;
  progress: number;
  message: string;
  params: Record<string, unknown>;
  result: Record<string, unknown> | null;
  error: string | null;
  created_at: string;
  started_at: string | null;
  ended_at: string | null;
  // interne
  _start: number;
  _duration: number; // ms ; Infinity = tourne indéfiniment
  _failAt?: number; // 0..1
  _log: string;
  _steps: number;
  _resultOnDone?: Record<string, unknown>;
  _onDone?: () => void;
}

interface Ctx {
  req: IncomingMessage;
  res: ServerResponse;
  url: URL;
  params: Record<string, string>;
  body: unknown;
}

type Handler = (ctx: Ctx) => unknown | Promise<unknown>;

/** À lever dans un handler pour renvoyer { detail } avec le code HTTP voulu. */
export class HttpError extends Error {
  constructor(
    public status: number,
    public detail: string,
  ) {
    super(detail);
  }
}

// ---------------------------------------------------------------- Système de fichiers simulé
const HOME = '/Users/eleveur';
const DOCS = `${HOME}/Documents`;
const dirs = new Map<string, { project: boolean }>();
function addDir(path: string, project = false) {
  dirs.set(path, { project });
  const parent = path.slice(0, path.lastIndexOf('/')) || '/';
  if (parent !== path && !dirs.has(parent)) addDir(parent);
}
[
  `${HOME}/Bureau`,
  `${HOME}/Téléchargements`,
  `${DOCS}/Comptabilité`,
  `${DOCS}/EggVision`,
  `${DOCS}/Photos banc`,
  `${DOCS}/Photos banc/2026-10`,
  `${DOCS}/Photos banc/2026-11`,
].forEach((d) => addDir(d));
addDir(`${DOCS}/EggVision/elevage-nord`, true);
addDir(`${DOCS}/EggVision/essai-convoyeur-v2`, true);
addDir(`${HOME}/Bureau/ancien-projet`, true);

// ---------------------------------------------------------------- État
const now = () => new Date().toISOString();
const iso = (msAgo: number) => new Date(Date.now() - msAgo).toISOString();

interface ProjectState {
  path: string;
  name: string;
  created_at: string;
  stats: { lots: number; frames: number; tracks: number; tracks_reviewed: number; models: number; simulations: number };
}
const projectData = new Map<string, ProjectState>([
  [
    `${DOCS}/EggVision/elevage-nord`,
    {
      path: `${DOCS}/EggVision/elevage-nord`,
      name: 'elevage-nord',
      created_at: iso(86400e3 * 21),
      stats: { lots: 14, frames: 18342, tracks: 2875, tracks_reviewed: 1620, models: 3, simulations: 0 },
    },
  ],
  [
    `${DOCS}/EggVision/essai-convoyeur-v2`,
    {
      path: `${DOCS}/EggVision/essai-convoyeur-v2`,
      name: 'essai-convoyeur-v2',
      created_at: iso(86400e3 * 3),
      stats: { lots: 2, frames: 1210, tracks: 0, tracks_reviewed: 0, models: 0, simulations: 0 },
    },
  ],
]);
let current: ProjectState | null = null;
let recents: { path: string; name: string; last_opened: string }[] = [
  { path: `${DOCS}/EggVision/elevage-nord`, name: 'elevage-nord', last_opened: iso(3600e3 * 5) },
  { path: `${DOCS}/EggVision/essai-convoyeur-v2`, name: 'essai-convoyeur-v2', last_opened: iso(86400e3 * 2) },
  { path: `${HOME}/Projets/supprime`, name: 'supprime', last_opened: iso(86400e3 * 40) },
];
let config: Record<string, unknown> = {
  camera: { fps: 30, exposure_us: 4000, width: 1456, height: 1088 },
  convoyeur: { vitesse_mm_s: 120, trappes: 3 },
  detection: { modele: 'yolo-oeuf-n', seuil: 0.45 },
};

const jobs: MockJob[] = [];
let jobSeq = 1;

function projectInfo() {
  if (!current) return { open: false };
  return { open: true, path: current.path, name: current.name, created_at: current.created_at, stats: current.stats };
}

function touchRecent(p: ProjectState) {
  recents = [{ path: p.path, name: p.name, last_opened: now() }, ...recents.filter((r) => r.path !== p.path)].slice(0, 10);
}

// ---------------------------------------------------------------- Jobs simulés
const LOG_PHRASES: Record<string, string[]> = {
  demo: ['Génération des lots synthétiques', 'Rendu des images', 'Détection des œufs', 'Construction des pistes', 'Écriture de la base'],
  training: ['Époque terminée', 'Validation', 'Sauvegarde du point de contrôle', 'Ajustement du taux d’apprentissage'],
  detection: ['Lot analysé', 'Pistes fusionnées', 'Images traitées'],
  export: ['Conversion ONNX', 'Quantification INT8', 'Vérification de la sortie'],
  default: ['Étape terminée', 'Traitement en cours', 'Lecture des données'],
};

function createJob(opts: {
  kind: string;
  title: string;
  duration: number;
  params?: Record<string, unknown>;
  failAt?: number;
  agoMs?: number;
  result?: Record<string, unknown>;
  onDone?: () => void;
  steps?: number;
}): MockJob {
  const start = Date.now() - (opts.agoMs ?? 0);
  const j: MockJob = {
    id: `job-${String(jobSeq++).padStart(4, '0')}`,
    kind: opts.kind,
    title: opts.title,
    status: 'queued',
    progress: 0,
    message: '',
    params: opts.params ?? {},
    result: null,
    error: null,
    created_at: new Date(start - 400).toISOString(),
    started_at: null,
    ended_at: null,
    _start: start,
    _duration: opts.duration,
    _failAt: opts.failAt,
    _log: `[${new Date(start).toISOString()}] INFO  Tâche ${opts.kind} créée\n`,
    _steps: 0,
    _resultOnDone: opts.result,
    _onDone: opts.onDone,
  };
  jobs.unshift(j);
  return j;
}

function advance(j: MockJob) {
  if (j.status === 'done' || j.status === 'failed' || j.status === 'cancelled') return;
  const elapsed = Date.now() - j._start;
  if (elapsed < 600) return; // reste « queued » un court instant
  if (j.status === 'queued') {
    j.status = 'running';
    j.started_at = new Date(j._start + 600).toISOString();
    j._log += `[${j.started_at}] INFO  Démarrage\n`;
  }
  const raw = Number.isFinite(j._duration) ? (elapsed - 600) / j._duration : ((elapsed / 1000) % 120) / 120;
  const progress = Math.max(0, Math.min(1, raw));
  const phrases = LOG_PHRASES[j.kind] ?? LOG_PHRASES.default;
  const targetSteps = Math.floor(progress * 40);
  while (j._steps < targetSteps) {
    j._steps++;
    const t = new Date(j._start + 600 + (j._steps / 40) * (Number.isFinite(j._duration) ? j._duration : 120000)).toISOString();
    const phrase = phrases[j._steps % phrases.length];
    j._log += `[${t}] INFO  ${phrase} (${j._steps}/40)\n`;
    if (j._steps % 9 === 0) j._log += `[${t}] WARN  Image floue ignorée : lot_03/frame_${String(j._steps * 37).padStart(5, '0')}.jpg\n`;
  }
  j.progress = progress;
  j.message = `${phrases[j._steps % phrases.length]}… (${j._steps}/40)`;

  if (j._failAt !== undefined && progress >= j._failAt) {
    j.status = 'failed';
    j.progress = j._failAt;
    j.error = 'Mémoire GPU insuffisante : réduisez la taille de lot (batch_size) ou la résolution.';
    j.ended_at = now();
    j._log += `[${j.ended_at}] ERROR RuntimeError: CUDA out of memory. Tried to allocate 512.00 MiB\nTraceback (most recent call last):\n  File "eggvision/train.py", line 214, in run\n    loss.backward()\n`;
    return;
  }
  if (progress >= 1) {
    j.status = 'done';
    j.progress = 1;
    j.message = 'Terminé';
    j.ended_at = new Date(j._start + 600 + j._duration).toISOString();
    j.result = j._resultOnDone ?? { ok: true };
    j._log += `[${j.ended_at}] INFO  Terminé avec succès\n`;
    j._onDone?.();
  }
}

// Tâches de départ, pour peupler la page Tâches.
createJob({ kind: 'export', title: 'Export ONNX — modele-v3', duration: 42_000, agoMs: 86400e3, params: { model: 'modele-v3', quantize: 'int8' }, result: { file: 'exports/modele-v3.onnx', size: 9_830_123 } });
createJob({ kind: 'training', title: 'Entraînement — classifieur défauts v4', duration: 300_000, failAt: 0.37, agoMs: 7200e3, params: { epochs: 40, batch_size: 64, lr: 0.001 } });
createJob({ kind: 'detection', title: 'Détection — lot 2026-11-18 matin', duration: 61_000, agoMs: 3600e3, params: { lot_id: 12 }, result: { tracks: 412 } });
createJob({ kind: 'bench', title: 'Banc de mesure Pi — latence', duration: 20_000, agoMs: 1800e3 }).status = 'cancelled';
createJob({ kind: 'training', title: 'Entraînement — classifieur défauts v5', duration: Infinity, agoMs: 25_000, params: { epochs: 60, batch_size: 32, lr: 0.0005 } });
// La tâche annulée doit avoir des dates cohérentes.
for (const j of jobs) if (j.status === 'cancelled') {
  j.started_at = new Date(j._start + 600).toISOString();
  j.ended_at = new Date(j._start + 9000).toISOString();
  j.progress = 0.42;
  j.message = 'Annulée par l’utilisateur';
}

// ---------------------------------------------------------------- Routes
const routes: { method: string; pattern: RegExp; keys: string[]; handler: Handler }[] = [];

/** Ajoute une route simulée : registerMockRoute('GET', '/api/lots/:id', ({ params }) => ...). */
export function registerMockRoute(method: string, path: string, handler: Handler) {
  const keys: string[] = [];
  const pattern = new RegExp('^' + path.replace(/:(\w+)/g, (_, k: string) => (keys.push(k), '([^/]+)')) + '/?$');
  routes.push({ method, pattern, keys, handler });
}
const route = registerMockRoute;

function requireProject() {
  if (!current) throw new HttpError(409, 'Aucun projet ouvert.');
  return current;
}

route('GET', '/api/health', () => ({ status: 'ok', version: '0.1.0-mock' }));

route('GET', '/api/project', () => projectInfo());
route('GET', '/api/project/recent', () => recents.map((r) => ({ ...r, exists: dirs.has(r.path) })));
route('POST', '/api/project/create', ({ body }) => {
  const { parent_dir, name } = (body ?? {}) as { parent_dir?: string; name?: string };
  if (!parent_dir || !name) throw new HttpError(422, 'Le nom et le dossier parent sont obligatoires.');
  if (!dirs.has(parent_dir)) throw new HttpError(404, `Dossier parent introuvable : ${parent_dir}`);
  const path = `${parent_dir}/${name}`;
  if (dirs.has(path)) throw new HttpError(409, `Le dossier « ${name} » existe déjà dans ${parent_dir}.`);
  addDir(path, true);
  const p: ProjectState = { path, name, created_at: now(), stats: { lots: 0, frames: 0, tracks: 0, tracks_reviewed: 0, models: 0, simulations: 0 } };
  projectData.set(path, p);
  current = p;
  touchRecent(p);
  return projectInfo();
});
route('POST', '/api/project/open', ({ body }) => {
  const { path } = (body ?? {}) as { path?: string };
  const p = path ? projectData.get(path) : undefined;
  if (!path || !dirs.has(path)) throw new HttpError(404, `Dossier introuvable : ${path ?? ''}`);
  if (!p) throw new HttpError(400, 'Ce dossier n’est pas un projet EggVision (config.yaml absent).');
  current = p;
  touchRecent(p);
  return projectInfo();
});
route('POST', '/api/project/close', () => {
  current = null;
  return projectInfo();
});
route('POST', '/api/project/demo', ({ body }) => {
  const { parent_dir, name, size } = (body ?? {}) as { parent_dir?: string; name?: string; size?: string };
  if (!parent_dir || !name) throw new HttpError(422, 'Le nom et le dossier parent sont obligatoires.');
  const path = `${parent_dir}/${name}`;
  if (dirs.has(path)) throw new HttpError(409, `Le dossier « ${name} » existe déjà.`);
  const big = size === 'standard';
  const j = createJob({
    kind: 'demo',
    title: `Projet de démonstration — ${name}`,
    duration: big ? 15_000 : 7_000,
    params: { parent_dir, name, size },
    result: { path },
    onDone: () => {
      addDir(path, true);
      projectData.set(path, {
        path,
        name,
        created_at: now(),
        stats: big
          ? { lots: 8, frames: 9600, tracks: 1400, tracks_reviewed: 1400, models: 1, simulations: 1 }
          : { lots: 3, frames: 1800, tracks: 260, tracks_reviewed: 120, models: 0, simulations: 0 },
      });
    },
  });
  return { job_id: j.id };
});
route('GET', '/api/project/config', () => {
  requireProject();
  return config;
});
route('PUT', '/api/project/config', ({ body }) => {
  requireProject();
  const merge = (a: Record<string, unknown>, b: Record<string, unknown>): Record<string, unknown> => {
    const out = { ...a };
    for (const [k, v] of Object.entries(b)) {
      out[k] = v && typeof v === 'object' && !Array.isArray(v) && a[k] && typeof a[k] === 'object' ? merge(a[k] as Record<string, unknown>, v as Record<string, unknown>) : v;
    }
    return out;
  };
  config = merge(config, (body ?? {}) as Record<string, unknown>);
  return config;
});

route('GET', '/api/fs/home', () => ({ home: HOME, documents: DOCS }));
route('POST', '/api/fs/mkdir', ({ body }) => {
  const { parent_dir, name } = (body ?? {}) as { parent_dir?: string; name?: string };
  if (!parent_dir || !name) throw new HttpError(422, 'Le nom et le dossier parent sont obligatoires.');
  if (!dirs.has(parent_dir)) throw new HttpError(404, `Dossier parent introuvable : ${parent_dir}`);
  const path = `${parent_dir}/${name}`;
  if (dirs.has(path)) throw new HttpError(409, `Le dossier « ${name} » existe déjà.`);
  addDir(path);
  return { path };
});
route('GET', '/api/fs/list', ({ url }) => {
  const path = url.searchParams.get('path') || HOME;
  if (!dirs.has(path)) throw new HttpError(404, `Dossier introuvable : ${path}`);
  const prefix = path === '/' ? '/' : `${path}/`;
  const entries = [...dirs.entries()]
    .filter(([p]) => p.startsWith(prefix) && p !== path && !p.slice(prefix.length).includes('/'))
    .map(([p, v]) => ({ name: p.slice(prefix.length), path: p, is_dir: true, is_project: v.project }))
    .sort((a, b) => a.name.localeCompare(b.name, 'fr'));
  const parent = path === '/' ? null : path.slice(0, path.lastIndexOf('/')) || '/';
  return { path, parent, entries };
});

const publicJob = (j: MockJob) => {
  advance(j);
  const { _start, _duration, _failAt, _log, _steps, _resultOnDone, _onDone, ...pub } = j;
  void [_start, _duration, _failAt, _log, _steps, _resultOnDone, _onDone];
  return pub;
};
const findJob = (id: string) => {
  const j = jobs.find((x) => x.id === id);
  if (!j) throw new HttpError(404, `Tâche introuvable : ${id}`);
  return j;
};

route('GET', '/api/jobs', ({ url }) => {
  const kind = url.searchParams.get('kind');
  const status = url.searchParams.get('status');
  const limit = Number(url.searchParams.get('limit') ?? 100);
  return jobs
    .map(publicJob)
    .filter((j) => (!kind || j.kind === kind) && (!status || j.status === status))
    .slice(0, limit);
});
route('GET', '/api/jobs/:id', ({ params }) => publicJob(findJob(params.id)));
route('POST', '/api/jobs/:id/cancel', ({ params }) => {
  const j = findJob(params.id);
  advance(j);
  if (j.status === 'queued' || j.status === 'running') {
    j.status = 'cancelled';
    j.ended_at = now();
    j.message = 'Annulée par l’utilisateur';
    j._log += `[${j.ended_at}] WARN  Annulation demandée par l’utilisateur\n`;
  }
  return publicJob(j);
});
route('GET', '/api/jobs/:id/log', ({ params, url }) => {
  const j = findJob(params.id);
  advance(j);
  const offset = Number(url.searchParams.get('offset') ?? 0);
  const CHUNK = 4096;
  const text = j._log.slice(offset, offset + CHUNK);
  return { text, next_offset: offset + text.length, eof: offset + text.length >= j._log.length };
});

route('GET', '/api/diagnostics', () => ({
  app_version: '0.1.0-mock',
  python: '3.12.7',
  platform: 'macOS-15.1-arm64',
  machine: 'arm64',
  packages: { torch: '2.5.1', torchvision: '0.20.1', ultralytics: '8.3.40', opencv: '4.10.0', onnx: '1.17.0', onnxruntime: '1.20.1', fastapi: '0.115.5', pywebview: '5.3.2', tflite_runtime: null },
  device: { kind: 'mps', name: 'Apple M3 Pro' },
  agent: { connected: false, url: 'http://raspberrypi.local:8770', detail: 'Délai de connexion dépassé (3 s).' },
  project: current ? { path: current.path, name: current.name } : null,
  log_file: `${HOME}/Library/Logs/EggVision/eggvision.log`,
  data_dir: `${HOME}/Library/Application Support/EggVision`,
}));
route('GET', '/api/diagnostics/log', ({ url }) => {
  const lines = Number(url.searchParams.get('lines') ?? 300);
  const out: string[] = [];
  const levels = ['INFO ', 'INFO ', 'INFO ', 'DEBUG', 'WARN '];
  const msgs = ['GET /api/project 200', 'GET /api/jobs 200', 'Connexion à l’agent Pi : délai dépassé', 'Cache des miniatures : 312 entrées', 'Projet ouvert'];
  for (let i = Math.min(lines, 120); i > 0; i--) {
    out.push(`${iso(i * 13_000)} ${levels[i % levels.length]} eggvision.server  ${msgs[i % msgs.length]}`);
  }
  return { text: out.join('\n') + '\n' };
});

// Médias : SVG généré (le vrai backend renvoie du JPEG ; <img> accepte les deux).
function eggSvg(seed: number, w: number, h: number, label: string) {
  const hue = 30 + (seed * 7) % 20;
  const cx = w / 2 + ((seed * 13) % 40) - 20;
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">
<rect width="100%" height="100%" fill="#2b2a27"/>
<ellipse cx="${cx}" cy="${h / 2}" rx="${w * 0.18}" ry="${h * 0.3}" fill="hsl(${hue} 45% 72%)" stroke="hsl(${hue} 30% 50%)" stroke-width="2"/>
<text x="8" y="${h - 10}" font-family="monospace" font-size="12" fill="#cfcac0">${label}</text></svg>`;
}
route('GET', '/api/media/frame/:lot/:index', ({ params, url, res }) => {
  const w = Number(url.searchParams.get('w') ?? 640);
  const h = Math.round((w * 3) / 4);
  res.setHeader('Content-Type', 'image/svg+xml');
  return { __raw: eggSvg(Number(params.index), w, h, `lot ${params.lot} · image ${params.index}`) };
});
route('GET', '/api/media/crop/:track/:view', ({ params, res }) => {
  res.setHeader('Content-Type', 'image/svg+xml');
  return { __raw: eggSvg(Number(params.view) + params.track.length, 160, 160, `piste ${params.track} · vue ${params.view}`) };
});

// ---------------------------------------------------------------- Mocks des modules
export type RegisterRoute = typeof registerMockRoute;
/** Ajouter ici les fonctions `register` des fichiers mock/<module>.ts. */
const MODULE_MOCKS: Array<(route: RegisterRoute) => void> = [];
for (const register of MODULE_MOCKS) register(registerMockRoute);

// ---------------------------------------------------------------- Plugin Vite
async function readBody(req: IncomingMessage): Promise<unknown> {
  if (req.method === 'GET' || req.method === 'HEAD') return undefined;
  const chunks: Buffer[] = [];
  for await (const c of req) chunks.push(c as Buffer);
  const raw = Buffer.concat(chunks).toString('utf8');
  if (!raw) return undefined;
  try {
    return JSON.parse(raw);
  } catch {
    throw new HttpError(400, 'Corps JSON invalide.');
  }
}

export function mockApiPlugin(): Plugin {
  return {
    name: 'eggvision-mock-api',
    configureServer(server) {
      server.config.logger.info('\n  [mock] API simulée active sur /api (VITE_MOCK=1)\n');
      server.middlewares.use(async (req, res, next) => {
        if (!req.url?.startsWith('/api/') && req.url !== '/api') return next();
        const url = new URL(req.url, 'http://localhost');
        const send = (status: number, payload: unknown) => {
          res.statusCode = status;
          if (payload && typeof payload === 'object' && '__raw' in payload) {
            res.end((payload as { __raw: string }).__raw);
            return;
          }
          res.setHeader('Content-Type', 'application/json; charset=utf-8');
          res.end(JSON.stringify(payload));
        };
        // Petite latence réaliste.
        await new Promise((r) => setTimeout(r, 80 + Math.random() * 120));
        try {
          for (const r of routes) {
            if (r.method !== req.method) continue;
            const m = r.pattern.exec(url.pathname);
            if (!m) continue;
            const params = Object.fromEntries(r.keys.map((k, i) => [k, decodeURIComponent(m[i + 1])]));
            const body = await readBody(req);
            const out = await r.handler({ req, res, url, params, body });
            return send(200, out);
          }
          send(404, { detail: `Route simulée inconnue : ${req.method} ${url.pathname}` });
        } catch (e) {
          if (e instanceof HttpError) send(e.status, { detail: e.detail });
          else send(500, { detail: `Erreur interne du mock : ${(e as Error).message}` });
        }
      });
    },
  };
}
