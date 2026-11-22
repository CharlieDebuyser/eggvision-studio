/** Lecteur : animation 2D (vue de dessus) d'une simulation terminée, contrôles et détail d'un œuf. */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ExternalLink, Pause, Play, SkipBack, SkipForward, StepBack, StepForward, X } from 'lucide-react';
import { useHotkeys } from '@/lib/hotkeys';
import { cx } from '@/lib/cx';
import { defectLabel } from '@/lib/domain';
import { formatNumber } from '@/lib/format';
import { Badge, Button, ButtonGroup, ErrorState, IconButton, LoadingState, Panel, SegmentedControl, Tooltip, VerdictBadge } from '@/components/ui';
import { simUrls, useEgg } from './api';
import { computeLayout, drawScene, hitTest, LATE_COLOR, OUTCOME_META, OUTCOME_ORDER, PENDING_COLOR, readPalette, visibleRange, type Hit } from './draw';
import type { SimDetail } from './types';
import { useTimeline } from './useTimeline';
import s from './simulation.module.css';

const SPEEDS = ['0.25', '0.5', '1', '2', '5', '10'] as const;
type SpeedKey = (typeof SPEEDS)[number];

export function Player({ sim }: { sim: SimDetail }) {
  const tl = useTimeline(sim.id);
  const g = tl.geometry;
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const tRef = useRef(0);
  const hitsRef = useRef<Hit[]>([]);
  const countsRef = useRef<[number, number, number, number]>([0, 0, 0, 0]);
  const [size, setSize] = useState({ w: 900, h: 380 });
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState<SpeedKey>('1');
  const [tUi, setTUi] = useState(0);
  const [selected, setSelected] = useState<{ column: number; lane: number } | null>(null);
  const [themeTick, setThemeTick] = useState(0);
  const dirty = useRef(true);

  const counts = sim.results.counts;
  const finals = useMemo(() => ({ reject: counts?.bin_reject ?? 0, viable: counts?.bin_viable ?? 0 }), [counts]);

  // Bornes temporelles de la lecture.
  const tMin = g ? -1.5 * g.pitch_mm / g.speed_mm_s : 0;
  const tMax = g ? (tl.total - 1) * g.period_s + (g.trap_x_mm + 1.5 * g.pitch_mm) / g.speed_mm_s + 0.3 * g.period_s : 1;

  useEffect(() => {
    if (g) {
      tRef.current = tMin;
      setTUi(tMin);
      dirty.current = true;
    }
  }, [g, tMin]);

  // Taille du canvas = taille du conteneur.
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => {
      const w = Math.max(480, Math.floor(entry.contentRect.width));
      setSize({ w, h: Math.round(Math.min(560, Math.max(300, w * 0.42))) });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Changement de thème clair / sombre : relire la palette.
  useEffect(() => {
    const mo = new MutationObserver(() => setThemeTick((n) => n + 1));
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    return () => mo.disconnect();
  }, []);

  const layout = useMemo(() => (g ? computeLayout(g, size.w, size.h) : null), [g, size]);
  const palette = useMemo(() => (canvasRef.current ? readPalette(canvasRef.current) : null), [themeTick, g, size]); // eslint-disable-line react-hooks/exhaustive-deps

  const selectedRef = useRef(selected);
  selectedRef.current = selected;
  useEffect(() => {
    dirty.current = true;
  }, [selected, palette, layout, tl.loadedChunks]);

  // Boucle d'animation.
  const playingRef = useRef(playing);
  playingRef.current = playing;
  const speedRef = useRef(Number(speed));
  speedRef.current = Number(speed);
  useEffect(() => {
    if (!g || !layout || !palette) return;
    const canvas = canvasRef.current!;
    const dpr = window.devicePixelRatio || 1;
    // Affecter width/height efface le canvas : ne le faire que si la taille change réellement,
    // sinon une pause (qui relance cet effet) laisserait un canvas vide jusqu'à la prochaine image.
    const cw = Math.round(layout.w * dpr);
    const ch = Math.round(layout.h * dpr);
    if (canvas.width !== cw) canvas.width = cw;
    if (canvas.height !== ch) canvas.height = ch;
    const ctx = canvas.getContext('2d')!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    dirty.current = true; // redimensionner le canvas l'efface : toujours redessiner, même en pause
    let raf = 0;
    let last = performance.now();
    let lastUi = 0;
    const frame = (now: number) => {
      const dt = Math.min(0.1, (now - last) / 1000);
      last = now;
      if (playingRef.current) {
        tRef.current = Math.min(tMax, tRef.current + dt * speedRef.current);
        dirty.current = true;
        if (tRef.current >= tMax) setPlaying(false);
      }
      if (dirty.current) {
        dirty.current = false;
        const [lo, hi] = visibleRange(g, layout, tRef.current, tl.total);
        tl.ensure(lo, hi + 40);
        hitsRef.current = drawScene(ctx, layout, {
          g,
          total: tl.total,
          t: tRef.current,
          get: tl.get,
          palette,
          selected: selectedRef.current,
          finals,
          lastCounts: countsRef,
        });
        if (now - lastUi > 100 || !playingRef.current) {
          lastUi = now;
          setTUi(tRef.current);
        }
      }
      raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(raf);
  }, [g, layout, palette, tl, tMax, finals]);

  const seek = useCallback(
    (t: number) => {
      tRef.current = Math.max(tMin, Math.min(tMax, t));
      dirty.current = true;
      setTUi(tRef.current);
    },
    [tMin, tMax],
  );
  const step = useCallback(
    (dir: 1 | -1) => {
      if (!g) return;
      setPlaying(false);
      // Pas à pas : avance / recule d'une colonne (une période), calé sur la grille des colonnes.
      const k = Math.round(tRef.current / g.period_s) + dir;
      seek(k * g.period_s);
    },
    [g, seek],
  );
  const togglePlay = useCallback(() => {
    if (tRef.current >= tMax - 1e-6) seek(tMin);
    setPlaying((p) => !p);
  }, [seek, tMax, tMin]);

  useHotkeys({ Space: togglePlay, ArrowRight: () => step(1), ArrowLeft: () => step(-1), Escape: () => setSelected(null) }, { enabled: !!g });

  const onCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const hit = hitTest(hitsRef.current, e.clientX - r.left, e.clientY - r.top);
    if (hit) {
      setPlaying(false);
      setSelected({ column: hit.column, lane: hit.lane });
    }
  };
  const onCanvasMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    e.currentTarget.style.cursor = hitTest(hitsRef.current, e.clientX - r.left, e.clientY - r.top) ? 'pointer' : 'default';
  };

  if (tl.error) return <ErrorState error={tl.error} title="Chronologie indisponible" />;

  const currentCol = g ? Math.max(0, Math.min(tl.total - 1, Math.floor(tUi / g.period_s))) : 0;
  const currentLate = tl.get(currentCol)?.late ?? false;

  return (
    <div className={cx(s.playerGrid, selected && s.playerGridWithPanel)}>
      <div className={s.playerMain}>
        <Panel flush>
          <div className={s.canvasWrap} ref={wrapRef}>
            {!g && <LoadingState label="Chargement de la chronologie…" />}
            <canvas
              ref={canvasRef}
              className={s.canvas}
              style={{ width: size.w, height: g ? size.h : 0 }}
              onClick={onCanvasClick}
              onMouseMove={onCanvasMove}
              role="img"
              aria-label="Animation de la table de tri vue de dessus ; cliquer sur un œuf pour afficher son détail"
            />
          </div>
          <div className={s.controls}>
            <ButtonGroup>
              <IconButton icon={SkipBack} label="Début" onClick={() => seek(tMin)} disabled={!g} />
              <IconButton icon={StepBack} label="Colonne précédente (←)" onClick={() => step(-1)} disabled={!g} />
              <IconButton icon={playing ? Pause : Play} label={playing ? 'Pause (Espace)' : 'Lecture (Espace)'} variant="primary" onClick={togglePlay} disabled={!g} />
              <IconButton icon={StepForward} label="Colonne suivante (→)" onClick={() => step(1)} disabled={!g} />
              <IconButton icon={SkipForward} label="Fin" onClick={() => seek(tMax)} disabled={!g} />
            </ButtonGroup>
            <input
              type="range"
              className={s.timebar}
              min={tMin}
              max={tMax}
              step={0.01}
              value={tUi}
              onChange={(e) => seek(Number(e.target.value))}
              aria-label="Barre de temps"
              disabled={!g}
            />
            <SegmentedControl<SpeedKey>
              size="sm"
              aria-label="Vitesse de lecture"
              value={speed}
              onChange={setSpeed}
              options={SPEEDS.map((v) => ({ value: v, label: `×${v.replace('.', ',')}` }))}
            />
          </div>
          <div className={s.clockRow}>
            <span className="u-mono">t = {formatNumber(tUi, 2, 2)} s</span>
            <span>
              Colonne <strong className="u-num">{formatNumber(currentCol + 1)}</strong> / {formatNumber(tl.total)}
            </span>
            {g && (
              <span className="u-muted">
                Budget {formatNumber(g.budget_s * 1000)} ms · vitesse {formatNumber(g.speed_mm_s, 0)} mm/s · cadence {formatNumber(g.period_s, 2)} s
              </span>
            )}
            {currentLate && <Badge tone="warning">Colonne en retard</Badge>}
            <span className="u-spacer" />
            <span className="u-subtle u-xs">Espace : lecture/pause · ← → : colonne par colonne · clic : détail d'un œuf</span>
          </div>
          <Legend />
        </Panel>
      </div>
      {selected && <EggPanel simId={sim.id} column={selected.column} lane={selected.lane} onClose={() => setSelected(null)} />}
    </div>
  );
}

function Legend() {
  return (
    <ul className={s.legend} aria-label="Légende">
      <li>
        <span className={s.swatch} style={{ background: `var(${PENDING_COLOR})` }} />
        Décision en attente
      </li>
      {OUTCOME_ORDER.map((o) => (
        <li key={o}>
          <span className={s.swatch} style={{ background: `var(${OUTCOME_META[o].color})` }} />
          {OUTCOME_META[o].label}
        </li>
      ))}
      <li>
        <span className={cx(s.swatch, s.swatchRing)} style={{ borderColor: `var(${LATE_COLOR})` }} />
        Décision en retard (trappe ratée si éjection)
      </li>
    </ul>
  );
}

function EggPanel({ simId, column, lane, onClose }: { simId: string; column: number; lane: number; onClose: () => void }) {
  const egg = useEgg(simId, column, lane);
  const navigate = useNavigate();
  const d = egg.data;
  return (
    <Panel
      className={s.eggPanel}
      title={`Œuf — colonne ${column + 1}, couloir ${lane + 1}`}
      actions={<IconButton icon={X} label="Fermer (Échap)" size="sm" onClick={onClose} />}
    >
      {egg.isLoading && <LoadingState compact />}
      {egg.error && <ErrorState error={egg.error} compact />}
      {d && (
        <div className="u-stack-4">
          <div className={s.eggHead}>
            <span className={s.outcomeChip} style={{ background: `var(${OUTCOME_META[d.outcome].color})` }} />
            <strong>{d.outcome_label}</strong>
            {d.late && <Badge tone="warning">Décision en retard</Badge>}
          </div>
          <dl className={s.kv}>
            <dt>Vérité terrain</dt>
            <dd>
              <VerdictBadge verdict={d.truth} /> {d.defect && <span className="u-muted">{defectLabel(d.defect)}</span>}
            </dd>
            <dt>Score / seuil</dt>
            <dd className="u-num">
              {d.score === null ? '—' : formatNumber(d.score, 3)} / {formatNumber(d.threshold, 3)}
            </dd>
            <dt>Décision</dt>
            <dd>{d.decision === null ? 'aucune (pas de vue)' : d.decision === 1 ? 'éjecter' : 'laisser passer'}</dd>
            <dt>Vues présentées</dt>
            <dd className="u-num">
              {d.n_views} / {d.n_views_expected ?? d.views.length}
            </dd>
            <dt>Latence de calcul</dt>
            <dd className="u-num">{formatNumber(d.timing.compute_ms, 1)} ms</dd>
            <dt>Attente (file)</dt>
            <dd className="u-num">{formatNumber(d.timing.wait_ms, 1)} ms</dd>
            <dt>Marge</dt>
            <dd className={cx('u-num', d.timing.margin_ms < 0 && s.negative)}>{formatNumber(d.timing.margin_ms, 1)} ms</dd>
            <dt>Piste</dt>
            <dd>
              <span className="u-mono">#{d.track_id}</span> <span className="u-muted u-small">{d.lot_name ?? d.lot_id}</span>
            </dd>
          </dl>
          {d.current_label && d.current_label.verdict !== d.truth && (
            <Badge tone="info">Label modifié depuis : {d.current_label.verdict ?? 'aucun'}</Badge>
          )}
          <div className={s.views}>
            {d.views.map((v) => (
              <Tooltip key={v.index} content={v.presented ? `Vue ${v.index + 1} — score ${formatNumber(v.score, 3)}` : `Vue ${v.index + 1} — non présentée`}>
                <figure className={cx(s.view, !v.presented && s.viewMissing)}>
                  <img src={simUrls.media(v.url)} alt={`Vue ${v.index + 1}`} loading="lazy" />
                  <figcaption className="u-num">{v.presented ? formatNumber(v.score, 2) : '—'}</figcaption>
                </figure>
              </Tooltip>
            ))}
          </div>
          <Button size="sm" iconRight={ExternalLink} block onClick={() => navigate(d.labeling_url)}>
            Ouvrir dans la labellisation
          </Button>
        </div>
      )}
    </Panel>
  );
}
