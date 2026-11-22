/**
 * Aperçu en direct (flux MJPEG, repli sur des instantanés) surmonté d'un calque SVG en pixels
 * de l'image : lignes de couloirs déplaçables, grille, détections, tracé de la ROI.
 */
import { useEffect, useRef, useState, type PointerEvent as ReactPointerEvent } from 'react';
import { Crop, Grid3x3, Minus, Move, Plus, RotateCcw, ScanSearch, Spline } from 'lucide-react';
import { Button, ButtonGroup, IconButton, NumberInput, Tooltip, useToast } from '@/components/ui';
import { formatNumber } from '@/lib/format';
import { previewUrl, useSaveOverlays } from './api';
import type { Analysis, CameraSettings, LaneLine, Overlays } from './types';
import s from './capture.module.css';

type Mode = 'view' | 'lanes' | 'roi';
type Drag =
  | { kind: 'end'; line: number; end: 1 | 2 }
  | { kind: 'line'; line: number; startX: number; orig: LaneLine }
  | { kind: 'roi'; x0: number; y0: number; x1: number; y1: number };

const laneColor = (i: number) => `var(--chart-${(i % 6) + 1})`;
const clamp01 = (v: number) => Math.min(1, Math.max(0, v));
const even = (v: number) => Math.max(0, Math.round(v / 2) * 2);

/**
 * Image d'un flux MJPEG. Chrome ne ferme pas la connexion d'un flux multipart quand l'élément est
 * retiré du DOM : on vide `src` au démontage, sinon les connexions s'accumulent jusqu'à bloquer
 * toutes les requêtes vers le serveur (6 connexions par hôte).
 */
function StreamImg({ src, onLoad, onError }: { src: string; onLoad: () => void; onError: () => void }) {
  const ref = useRef<HTMLImageElement>(null);
  const alive = useRef(true);
  useEffect(() => {
    const img = ref.current;
    alive.current = true;
    // React StrictMode (dev) démonte puis remonte : on rétablit la source vidée au démontage.
    if (img && img.getAttribute('src') !== src) img.src = src;
    return () => {
      alive.current = false; // l'erreur provoquée par le vidage de src ne doit rien déclencher
      if (img) img.src = 'data:,';
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps — montage / démontage uniquement
  return (
    <img
      ref={ref}
      className={s.stageImg}
      src={src}
      alt="Aperçu de la caméra"
      draggable={false}
      onLoad={() => alive.current && onLoad()}
      onError={() => alive.current && onError()}
    />
  );
}

export function evenLines(n: number): LaneLine[] {
  return Array.from({ length: n + 1 }, (_, i) => ({ x1: i / n, y1: 0, x2: i / n, y2: 1 }));
}

export interface LivePreviewProps {
  sourceKey: string;
  width: number;
  height: number;
  sensor: { width: number; height: number };
  overlays: Overlays;
  settings: CameraSettings | undefined;
  analysis: Analysis | undefined;
  /** Réglages verrouillés (capture en cours) : tracé de ROI désactivé. */
  locked: boolean;
  onRoi: (roi: [number, number, number, number]) => void;
}

export function LivePreview({ sourceKey, width, height, sensor, overlays, settings, analysis, locked, onRoi }: LivePreviewProps) {
  const toast = useToast();
  const saveOverlays = useSaveOverlays();
  const [mode, setMode] = useState<Mode>('view');
  const [lines, setLines] = useState<LaneLine[]>(overlays.lane_lines);
  const [drag, setDrag] = useState<Drag | null>(null);
  const [snapshot, setSnapshot] = useState(false);
  const [nonce, setNonce] = useState(() => Date.now());
  const [loaded, setLoaded] = useState(false);
  const svgRef = useRef<SVGSVGElement>(null);

  // Synchronise les lignes locales quand la configuration change (hors glisser-déposer).
  useEffect(() => {
    if (!drag) setLines(overlays.lane_lines);
  }, [overlays.lane_lines]); // eslint-disable-line react-hooks/exhaustive-deps

  // Nouvelle source → nouveau flux (pas au premier rendu : le flux vient d'être ouvert).
  const prevKey = useRef(sourceKey);
  useEffect(() => {
    if (prevKey.current === sourceKey) return;
    prevKey.current = sourceKey;
    setSnapshot(false);
    setLoaded(false);
    setNonce(Date.now());
  }, [sourceKey]);

  // Repli : instantanés JPEG rafraîchis (si le flux MJPEG échoue).
  useEffect(() => {
    if (!snapshot) return;
    const t = setInterval(() => setNonce(Date.now()), 250);
    // nouvel essai du flux MJPEG au bout de 15 s (coupure passagère)
    const retry = setTimeout(() => setSnapshot(false), 15_000);
    return () => {
      clearInterval(t);
      clearTimeout(retry);
    };
  }, [snapshot]);

  useEffect(() => {
    if (locked && mode === 'roi') setMode('view');
  }, [locked, mode]);

  const W = width || 1;
  const H = height || 1;
  // Dernier état des repères (plusieurs bascules rapides ne s'écrasent pas entre elles).
  const latest = useRef(overlays);
  useEffect(() => {
    latest.current = overlays;
  }, [overlays]);
  const save = (next: Partial<Overlays>) => {
    const merged = { ...latest.current, lane_lines: lines, ...next };
    latest.current = merged;
    saveOverlays.mutate(merged, { onError: (e) => toast.error('Repères non enregistrés', String(e)) });
  };

  const toImage = (e: ReactPointerEvent<SVGElement>): [number, number] => {
    const r = svgRef.current!.getBoundingClientRect();
    return [((e.clientX - r.left) / r.width) * W, ((e.clientY - r.top) / r.height) * H];
  };

  const onDown = (e: ReactPointerEvent<SVGSVGElement>) => {
    if (mode !== 'roi') return;
    const [x, y] = toImage(e);
    svgRef.current?.setPointerCapture(e.pointerId);
    setDrag({ kind: 'roi', x0: x, y0: y, x1: x, y1: y });
  };
  const onMove = (e: ReactPointerEvent<SVGSVGElement>) => {
    if (!drag) return;
    const [x, y] = toImage(e);
    if (drag.kind === 'roi') setDrag({ ...drag, x1: x, y1: y });
    else if (drag.kind === 'end') {
      setLines((ls) => ls.map((l, i) => (i !== drag.line ? l : drag.end === 1 ? { ...l, x1: clamp01(x / W), y1: clamp01(y / H) } : { ...l, x2: clamp01(x / W), y2: clamp01(y / H) })));
    } else {
      const dx = (x - drag.startX) / W;
      const o = drag.orig;
      const d = Math.min(1 - Math.max(o.x1, o.x2), Math.max(-Math.min(o.x1, o.x2), dx));
      setLines((ls) => ls.map((l, i) => (i !== drag.line ? l : { ...o, x1: o.x1 + d, x2: o.x2 + d })));
    }
  };
  const onUp = () => {
    if (!drag) return;
    if (drag.kind === 'roi') {
      const x0 = Math.max(0, Math.min(drag.x0, drag.x1));
      const y0 = Math.max(0, Math.min(drag.y0, drag.y1));
      const w = Math.min(W, Math.max(drag.x0, drag.x1)) - x0;
      const h = Math.min(H, Math.max(drag.y0, drag.y1)) - y0;
      if (w > 8 && h > 8) {
        // L'aperçu montre la ROI courante : on compose avec elle pour obtenir des pixels capteur.
        const base = settings?.roi ?? [0, 0, sensor.width, sensor.height];
        const sx = base[2] / W;
        const sy = base[3] / H;
        onRoi([even(base[0] + x0 * sx), even(base[1] + y0 * sy), Math.max(16, even(w * sx)), Math.max(16, even(h * sy))]);
        setMode('view');
      }
    } else {
      save({ lane_lines: lines });
    }
    setDrag(null);
  };

  const startEnd = (line: number, end: 1 | 2) => (e: ReactPointerEvent<SVGElement>) => {
    e.stopPropagation();
    svgRef.current?.setPointerCapture(e.pointerId);
    setDrag({ kind: 'end', line, end });
  };
  const startLine = (line: number) => (e: ReactPointerEvent<SVGElement>) => {
    e.stopPropagation();
    svgRef.current?.setPointerCapture(e.pointerId);
    setDrag({ kind: 'line', line, startX: toImage(e)[0], orig: lines[line] });
  };

  const nLanes = Math.max(1, lines.length - 1);
  const showLanes = overlays.show_lanes || mode === 'lanes';
  const dets = overlays.show_detections ? analysis?.detections : null;
  // Détections : on affiche l'image analysée elle-même (sinon décalage avec le flux, la scène défile).
  const syncedImage = overlays.show_detections && analysis?.image ? analysis.image : null;
  const df = analysis?.detections_frame;
  const ds = df && df.width ? W / df.width : 1;
  const stroke = Math.max(1, W / 500);

  return (
    <div className={s.preview}>
      <div className={s.previewToolbar}>
        <ButtonGroup>
          <Tooltip content="Afficher les lignes de couloirs">
            <Button size="sm" variant={overlays.show_lanes ? 'secondary' : 'ghost'} icon={Spline} onClick={() => save({ show_lanes: !overlays.show_lanes })}>
              Couloirs
            </Button>
          </Tooltip>
          <Tooltip content="Afficher la grille">
            <Button size="sm" variant={overlays.grid.enabled ? 'secondary' : 'ghost'} icon={Grid3x3} onClick={() => save({ grid: { ...overlays.grid, enabled: !overlays.grid.enabled } })}>
              Grille
            </Button>
          </Tooltip>
          <Tooltip content="Détections d’œufs en direct (module de détection)">
            <Button size="sm" variant={overlays.show_detections ? 'secondary' : 'ghost'} icon={ScanSearch} onClick={() => save({ show_detections: !overlays.show_detections })}>
              Détections
            </Button>
          </Tooltip>
        </ButtonGroup>
        {overlays.grid.enabled && (
          <span className={s.gridInputs}>
            <NumberInput size="sm" value={overlays.grid.cols} min={1} max={64} step={1} stepper={false} aria-label="Colonnes de la grille" onChange={(v) => v && save({ grid: { ...overlays.grid, cols: v } })} />
            <span className="u-muted u-small">×</span>
            <NumberInput size="sm" value={overlays.grid.rows} min={1} max={64} step={1} stepper={false} aria-label="Lignes de la grille" onChange={(v) => v && save({ grid: { ...overlays.grid, rows: v } })} />
          </span>
        )}
        <span className="u-spacer" />
        <ButtonGroup>
          <Button size="sm" variant={mode === 'lanes' ? 'accent' : 'ghost'} icon={Move} onClick={() => setMode(mode === 'lanes' ? 'view' : 'lanes')}>
            {mode === 'lanes' ? 'Terminer' : 'Ajuster les couloirs'}
          </Button>
          <Tooltip content={locked ? 'Réglages verrouillés pendant la capture' : 'Tracer la zone de recadrage (ROI) sur l’aperçu'}>
            <Button size="sm" variant={mode === 'roi' ? 'accent' : 'ghost'} icon={Crop} disabled={locked} onClick={() => setMode(mode === 'roi' ? 'view' : 'roi')}>
              {mode === 'roi' ? 'Annuler le tracé' : 'Tracer la ROI'}
            </Button>
          </Tooltip>
        </ButtonGroup>
      </div>
      {mode === 'lanes' && (
        <div className={s.laneEditBar}>
          <span className="u-small">
            {nLanes} couloir{nLanes > 1 ? 's' : ''} — faire glisser une ligne ou ses extrémités.
          </span>
          <IconButton size="sm" icon={Minus} label="Retirer un couloir" disabled={nLanes <= 1} onClick={() => { const l = evenLines(nLanes - 1); setLines(l); save({ lane_lines: l }); }} />
          <IconButton size="sm" icon={Plus} label="Ajouter un couloir" disabled={nLanes >= 32} onClick={() => { const l = evenLines(nLanes + 1); setLines(l); save({ lane_lines: l }); }} />
          <Button size="sm" variant="ghost" icon={RotateCcw} onClick={() => { const l = evenLines(nLanes); setLines(l); save({ lane_lines: l }); }}>
            Répartir régulièrement
          </Button>
        </div>
      )}
      {mode === 'roi' && <div className={s.laneEditBar}><span className="u-small">Tracer un rectangle sur l’aperçu : il devient la ROI (à appliquer dans les réglages).</span></div>}
      <div className={s.stage} style={{ aspectRatio: `${W} / ${H}` }}>
        {syncedImage ? null : (
          <StreamImg
            key={snapshot ? 'snap' : `mjpeg-${nonce}`}
            src={snapshot ? previewUrl.jpeg(960, nonce) : previewUrl.mjpeg(960, nonce)}
            onLoad={() => setLoaded(true)}
            onError={() => {
              if (!snapshot) setSnapshot(true);
            }}
          />
        )}
        {syncedImage && <img className={s.stageImg} src={syncedImage} alt="Image analysée (détections)" draggable={false} />}
        {!loaded && !analysis && <div className={s.stageWait}>Attente des images…</div>}
        <svg
          ref={svgRef}
          className={s.stageSvg}
          data-mode={mode}
          viewBox={`0 0 ${W} ${H}`}
          preserveAspectRatio="none"
          onPointerDown={onDown}
          onPointerMove={onMove}
          onPointerUp={onUp}
          onPointerCancel={onUp}
        >
          {overlays.grid.enabled && (
            <g className={s.gridLines} strokeWidth={stroke * 0.8}>
              {Array.from({ length: overlays.grid.cols - 1 }, (_, i) => (
                <line key={`c${i}`} x1={((i + 1) * W) / overlays.grid.cols} y1={0} x2={((i + 1) * W) / overlays.grid.cols} y2={H} />
              ))}
              {Array.from({ length: overlays.grid.rows - 1 }, (_, i) => (
                <line key={`r${i}`} x1={0} y1={((i + 1) * H) / overlays.grid.rows} x2={W} y2={((i + 1) * H) / overlays.grid.rows} />
              ))}
            </g>
          )}
          {showLanes &&
            lines.map((l, i) => (
              <g key={i}>
                <line x1={l.x1 * W} y1={l.y1 * H} x2={l.x2 * W} y2={l.y2 * H} className={s.laneLine} strokeWidth={stroke * 2} />
                {mode === 'lanes' && (
                  <>
                    <line x1={l.x1 * W} y1={l.y1 * H} x2={l.x2 * W} y2={l.y2 * H} className={s.laneHit} strokeWidth={stroke * 14} onPointerDown={startLine(i)} />
                    <circle cx={l.x1 * W} cy={l.y1 * H} r={stroke * 7} className={s.laneHandle} onPointerDown={startEnd(i, 1)} />
                    <circle cx={l.x2 * W} cy={l.y2 * H} r={stroke * 7} className={s.laneHandle} onPointerDown={startEnd(i, 2)} />
                  </>
                )}
                {i < lines.length - 1 && (
                  <text x={((l.x1 + l.x2 + lines[i + 1].x1 + lines[i + 1].x2) / 4) * W} y={H * 0.04 + stroke * 8} className={s.laneLabel} fontSize={stroke * 12}>
                    {i + 1}
                  </text>
                )}
              </g>
            ))}
          {dets?.map((d, i) => (
            <ellipse
              key={i}
              cx={d.cx * ds}
              cy={d.cy * ds}
              rx={(d.major * ds) / 2}
              ry={(d.minor * ds) / 2}
              transform={`rotate(${d.angle} ${d.cx * ds} ${d.cy * ds})`}
              className={s.detection}
              stroke={d.lane != null && d.lane >= 0 ? laneColor(d.lane) : 'var(--text-muted)'}
              strokeDasharray={d.partial ? `${stroke * 4} ${stroke * 3}` : undefined}
              strokeWidth={stroke * 2}
            />
          ))}
          {drag?.kind === 'roi' && (
            <rect
              x={Math.min(drag.x0, drag.x1)}
              y={Math.min(drag.y0, drag.y1)}
              width={Math.abs(drag.x1 - drag.x0)}
              height={Math.abs(drag.y1 - drag.y0)}
              className={s.roiRect}
              strokeWidth={stroke * 2}
            />
          )}
        </svg>
      </div>
      <div className={s.previewFoot}>
        <span className="u-xs u-muted">
          {W}×{H} px{settings?.roi ? ` · ROI ${settings.roi.join(', ')}` : ' · plein capteur'}
          {snapshot ? ' · mode instantanés' : ''}
          {syncedImage ? ' · image analysée (~3 im/s, synchronisée avec les détections)' : ''}
        </span>
        {overlays.show_detections && (
          <span className="u-xs u-muted">
            {analysis?.detections_error
              ? analysis.detections_error
              : dets
                ? `${dets.length} œuf${dets.length > 1 ? 's' : ''} détecté${dets.length > 1 ? 's' : ''}`
                : 'Détections : en attente…'}
          </span>
        )}
        {snapshot && (
          <Button size="sm" variant="ghost" onClick={() => { setSnapshot(false); setNonce(Date.now()); }}>
            Réessayer le flux
          </Button>
        )}
        <span className="u-xs u-muted">{analysis ? `moyenne R ${formatNumber(analysis.mean.r, 0)} · V ${formatNumber(analysis.mean.g, 0)} · B ${formatNumber(analysis.mean.b, 0)}` : ''}</span>
      </div>
    </div>
  );
}
