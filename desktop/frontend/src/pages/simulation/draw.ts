/**
 * Dessin de l'animation 2D (vue de dessus) sur <canvas> — mêmes éléments que la vidéo MP4
 * rendue côté serveur (desktop/src/eggvision/simulation/video.py).
 *
 * Repère : x en mm le long du défilement (0 = entrée de la zone caméra), une ligne par couloir.
 * Pour la colonne k à l'instant t : x = (t − t_enter(k)) · v.
 */
import type { Geometry, Outcome, TimelineColumn } from './types';

/** Couleurs : jetons CSS (thème clair / sombre), lus au moment du dessin. */
export const OUTCOME_META: Record<Outcome, { label: string; color: string }> = {
  accepted_ok: { label: 'Accepté à raison', color: '--success' },
  ejected_ok: { label: 'Éjecté à raison', color: '--info' },
  missed: { label: 'Manqué', color: '--danger' },
  missed_late: { label: 'Manqué (retard)', color: '--chart-5' },
  false_reject: { label: 'Éjecté à tort', color: '--warning' },
};
export const OUTCOME_ORDER: Outcome[] = ['accepted_ok', 'ejected_ok', 'missed', 'missed_late', 'false_reject'];
export const PENDING_COLOR = '--text-subtle';
export const LATE_COLOR = '--accent';

export interface Palette {
  outcome: Record<Outcome, string>;
  pending: string;
  late: string;
  table: string;
  roller: string;
  border: string;
  camera: string;
  cameraSoft: string;
  trapClosed: string;
  trapOpen: string;
  text: string;
  muted: string;
  surface: string;
  viableBin: string;
  rejectBin: string;
  selected: string;
  /** Police calculée de l'élément (les variables CSS ne sont pas résolues par le canvas). */
  font: string;
}

export function readPalette(el: Element): Palette {
  const cs = getComputedStyle(el);
  const v = (name: string) => cs.getPropertyValue(name).trim() || '#888';
  const outcome = Object.fromEntries(OUTCOME_ORDER.map((o) => [o, v(OUTCOME_META[o].color)])) as Record<Outcome, string>;
  return {
    outcome,
    pending: v(PENDING_COLOR),
    late: v(LATE_COLOR),
    table: v('--surface-2'),
    roller: v('--border'),
    border: v('--border-strong'),
    camera: v('--info'),
    cameraSoft: v('--info-soft'),
    trapClosed: v('--surface-3'),
    trapOpen: v('--text'),
    text: v('--text'),
    muted: v('--text-muted'),
    surface: v('--surface'),
    viableBin: v('--success'),
    rejectBin: v('--danger'),
    selected: v('--text'),
    font: cs.fontFamily || 'system-ui, sans-serif',
  };
}

export interface Layout {
  w: number;
  h: number;
  xMin: number;
  xEnd: number;
  scale: number;
  left: number;
  top: number;
  laneH: number;
  rx: number;
  ry: number;
  tableBottom: number;
  binX: number;
  binW: number;
  rejectBox: [number, number, number, number];
}

export function computeLayout(g: Geometry, w: number, h: number): Layout {
  const labelW = 22;
  const margin = 12;
  const binW = Math.max(70, Math.round(w * 0.1));
  const top = 26;
  const bottom = Math.max(64, Math.round(h * 0.2));
  const xMin = -1.5 * g.pitch_mm;
  const xEnd = g.trap_x_mm + 1.5 * g.pitch_mm;
  const usable = w - labelW - margin * 3 - binW;
  const scale = usable / (xEnd - xMin);
  const avail = h - top - bottom;
  const laneH = Math.max(8, Math.min(avail / g.n_lanes, g.pitch_mm * scale * 1.1));
  const tableTop = top + Math.max(0, (avail - laneH * g.n_lanes) / 2);
  const rx = Math.max(2, Math.min(0.33 * g.pitch_mm, 30) * scale);
  const ry = Math.max(2, Math.min(rx * 0.75, laneH * 0.34));
  const left = margin + labelW;
  const tableBottom = tableTop + laneH * g.n_lanes;
  const binX = left + (xEnd - xMin) * scale + margin;
  const trapPx = left + (g.trap_x_mm - xMin) * scale;
  const rbW = Math.max(90, binW * 1.4);
  const rejectBox: [number, number, number, number] = [trapPx - rbW / 2, tableBottom + 12, rbW, Math.max(36, bottom - 24)];
  return { w, h, xMin, xEnd, scale, left, top: tableTop, laneH, rx, ry, tableBottom, binX, binW, rejectBox };
}

export interface Hit {
  column: number;
  lane: number;
  x: number;
  y: number;
  rx: number;
  ry: number;
}

export interface SceneInput {
  g: Geometry;
  total: number;
  t: number;
  get: (k: number) => TimelineColumn | undefined;
  palette: Palette;
  selected: { column: number; lane: number } | null;
  /** Effectifs finaux des bacs (pour le niveau de remplissage). */
  finals: { reject: number; viable: number };
  /** Derniers compteurs connus (si la colonne courante n'est pas encore chargée). */
  lastCounts: { current: [number, number, number, number] };
}

/** Plage de colonnes visibles à l'instant t. */
export function visibleRange(g: Geometry, L: Layout, t: number, total: number): [number, number] {
  const lo = Math.max(0, Math.floor((t - L.xEnd / g.speed_mm_s) / g.period_s) - 1);
  const hi = Math.min(total - 1, Math.ceil((t - L.xMin / g.speed_mm_s) / g.period_s) + 1);
  return [lo, hi];
}

export function drawScene(ctx: CanvasRenderingContext2D, L: Layout, s: SceneInput): Hit[] {
  const { g, t, palette: P } = s;
  const px = (x: number) => L.left + (x - L.xMin) * L.scale;
  const laneY = (lane: number) => L.top + (lane + 0.5) * L.laneH;
  const hits: Hit[] = [];

  ctx.clearRect(0, 0, L.w, L.h);
  ctx.font = `11px ${P.font}`;
  ctx.textBaseline = 'middle';

  // Table et rouleaux qui défilent.
  const x0 = px(L.xMin);
  const x1 = px(L.xEnd);
  ctx.fillStyle = P.table;
  ctx.fillRect(x0, L.top, x1 - x0, L.tableBottom - L.top);
  ctx.strokeStyle = P.roller;
  ctx.lineWidth = 1;
  const spacing = g.pitch_mm / 2;
  const offset = (((t * g.speed_mm_s) % spacing) + spacing) % spacing;
  ctx.beginPath();
  for (let x = L.xMin + offset; x < L.xEnd; x += spacing) {
    const p = Math.round(px(x)) + 0.5;
    ctx.moveTo(p, L.top + 1);
    ctx.lineTo(p, L.tableBottom - 1);
  }
  for (let lane = 1; lane < g.n_lanes; lane++) {
    const y = Math.round(L.top + lane * L.laneH) + 0.5;
    ctx.moveTo(x0, y);
    ctx.lineTo(x1, y);
  }
  ctx.stroke();
  ctx.strokeStyle = P.border;
  ctx.strokeRect(x0 + 0.5, L.top + 0.5, x1 - x0 - 1, L.tableBottom - L.top - 1);

  // Numéros de couloir.
  ctx.fillStyle = P.muted;
  ctx.textAlign = 'right';
  for (let lane = 0; lane < g.n_lanes; lane++) ctx.fillText(String(lane + 1), x0 - 6, laneY(lane));

  // Zone caméra.
  const cx0 = px(0);
  const cx1 = px(g.camera_zone_mm);
  ctx.globalAlpha = 0.35;
  ctx.fillStyle = P.cameraSoft;
  ctx.fillRect(cx0, L.top - 4, cx1 - cx0, L.tableBottom - L.top + 8);
  ctx.globalAlpha = 1;
  ctx.strokeStyle = P.camera;
  ctx.lineWidth = 2;
  ctx.strokeRect(cx0, L.top - 4, cx1 - cx0, L.tableBottom - L.top + 8);
  ctx.fillStyle = P.camera;
  ctx.textAlign = 'left';
  ctx.fillText('Caméra', cx0 + 2, L.top - 13);

  const [kLo, kHi] = visibleRange(g, L, t, s.total);

  // Trappes.
  const trapOpen = new Array<boolean>(g.n_lanes).fill(false);
  const halfP = 0.5 * g.period_s;
  for (let k = kLo; k <= kHi; k++) {
    const c = s.get(k);
    if (!c || !c.decided || c.late || t < c.t_deadline || t > c.t_trap + halfP) continue;
    c.lanes.forEach((e, lane) => {
      if (e && e.ejected) trapOpen[lane] = true;
    });
  }
  const tx = px(g.trap_x_mm);
  const tw = Math.max(4, g.pitch_mm * 0.28 * L.scale);
  for (let lane = 0; lane < g.n_lanes; lane++) {
    const y = laneY(lane);
    const th = L.laneH * 0.42;
    ctx.fillStyle = trapOpen[lane] ? P.trapOpen : P.trapClosed;
    ctx.fillRect(tx - tw, y - th, 2 * tw, 2 * th);
    if (trapOpen[lane]) {
      ctx.strokeStyle = P.late;
      ctx.lineWidth = 1.5;
      ctx.strokeRect(tx - tw, y - th, 2 * tw, 2 * th);
    }
  }
  ctx.fillStyle = P.text;
  ctx.textAlign = 'center';
  ctx.fillText('Trappes', tx, L.top - 13);

  // Œufs (limités à la table).
  ctx.save();
  ctx.beginPath();
  ctx.rect(x0, L.top - 8, x1 - x0, L.tableBottom - L.top + 16);
  ctx.clip();
  const fall = 0.25 * g.period_s;
  for (let k = kLo; k <= kHi; k++) {
    const c = s.get(k);
    if (!c) continue;
    const xmm = (t - c.t_enter) * g.speed_mm_s;
    if (xmm < L.xMin) continue;
    c.lanes.forEach((e, lane) => {
      if (!e) return;
      let rx = L.rx;
      let ry = L.ry;
      let xx = xmm;
      if (e.ejected && t > c.t_trap) {
        const dt = t - c.t_trap;
        if (dt > fall) return;
        const f = 1 - dt / fall;
        rx *= f;
        ry *= f;
        xx = g.trap_x_mm;
      } else if (xmm > L.xEnd) {
        return;
      }
      const decided = c.decided && t >= c.t_decision && e.decision !== null;
      let color = decided ? P.outcome[e.outcome] : P.pending;
      if (e.decision === null && xmm >= g.camera_zone_mm) color = P.outcome[e.outcome];
      const cx = px(xx);
      const cy = laneY(lane);
      ctx.beginPath();
      ctx.ellipse(cx, cy, Math.max(1, rx), Math.max(1, ry), 0, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      if (decided && e.late) {
        ctx.lineWidth = 2.5;
        ctx.strokeStyle = P.late;
        ctx.beginPath();
        ctx.ellipse(cx, cy, rx + 2.5, ry + 2.5, 0, 0, Math.PI * 2);
        ctx.stroke();
      }
      if (s.selected && s.selected.column === k && s.selected.lane === lane) {
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 3]);
        ctx.strokeStyle = P.selected;
        ctx.beginPath();
        ctx.ellipse(cx, cy, rx + 5, ry + 5, 0, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);
      }
      hits.push({ column: k, lane, x: cx, y: cy, rx: Math.max(rx, 6), ry: Math.max(ry, 6) });
    });
  }

  ctx.restore();

  // Bacs (compteurs = cumuls de la dernière colonne arrivée).
  const tTrap0 = g.camera_time_s + g.budget_s;
  const kTrap = Math.min(s.total - 1, Math.floor((t - tTrap0) / g.period_s + 1e-9));
  const kEnd = Math.min(s.total - 1, Math.floor((t - L.xEnd / g.speed_mm_s) / g.period_s + 1e-9));
  const counts = s.lastCounts.current;
  if (kTrap < 0) {
    counts[0] = 0;
    counts[1] = 0;
  } else {
    const c = s.get(kTrap);
    if (c) {
      counts[0] = c.cum[0];
      counts[1] = c.cum[1];
    }
  }
  if (kEnd < 0) {
    counts[2] = 0;
    counts[3] = 0;
  } else {
    const c = s.get(kEnd);
    if (c) {
      counts[2] = c.cum[2];
      counts[3] = c.cum[3];
    }
  }
  drawBin(ctx, P, [L.binX, L.top, L.binW, L.tableBottom - L.top], P.viableBin, 'Bac viables', counts[2], `dont ${counts[3]} défaut(s)`, s.finals.viable, 'top');
  drawBin(ctx, P, L.rejectBox, P.rejectBin, 'Bac rebut', counts[0], `dont ${counts[1]} à tort`, s.finals.reject, 'left');
  return hits;
}

function drawBin(
  ctx: CanvasRenderingContext2D,
  P: Palette,
  [x, y, w, h]: [number, number, number, number],
  color: string,
  title: string,
  count: number,
  sub: string,
  final: number,
  titlePos: 'top' | 'left',
) {
  ctx.fillStyle = P.surface;
  ctx.fillRect(x, y, w, h);
  const level = final > 0 ? Math.min(1, count / final) : 0;
  ctx.globalAlpha = 0.18;
  ctx.fillStyle = color;
  ctx.fillRect(x, y + h * (1 - level), w, h * level);
  ctx.globalAlpha = 1;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.strokeRect(x + 1, y + 1, w - 2, h - 2);
  ctx.fillStyle = color;
  if (titlePos === 'top') {
    ctx.textAlign = 'center';
    ctx.fillText(title, x + w / 2, y - 13);
  } else {
    ctx.textAlign = 'right';
    ctx.fillText(title, x - 8, y + h / 2);
  }
  ctx.textAlign = 'center';
  fitFont(ctx, String(count), w - 8, 18, `600 {}px ${P.font}`);
  ctx.fillText(String(count), x + w / 2, y + h / 2 - (h > 50 ? 8 : 6));
  fitFont(ctx, sub, w - 8, 11, `{}px ${P.font}`);
  ctx.fillStyle = P.muted;
  ctx.fillText(sub, x + w / 2, y + h / 2 + (h > 50 ? 12 : 9));
  ctx.font = `11px ${P.font}`;
}

/** Choisit la plus grande taille de police (≤ max) pour que le texte tienne dans la largeur. */
function fitFont(ctx: CanvasRenderingContext2D, text: string, width: number, max: number, pattern: string) {
  let size = max;
  ctx.font = pattern.replace('{}', String(size));
  while (size > 8 && ctx.measureText(text).width > width) {
    size -= 1;
    ctx.font = pattern.replace('{}', String(size));
  }
}

export function hitTest(hits: Hit[], x: number, y: number): Hit | null {
  for (const h of hits) {
    const dx = (x - h.x) / h.rx;
    const dy = (y - h.y) / h.ry;
    if (dx * dx + dy * dy <= 1) return h;
  }
  return null;
}
