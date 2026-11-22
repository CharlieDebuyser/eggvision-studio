import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type CSSProperties } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { useQueryClient } from '@tanstack/react-query';
import { AlertTriangle, CheckCheck, Flag, ListRestart, RotateCcw } from 'lucide-react';
import { mediaUrl, errorMessage } from '@/api/client';
import { Badge, Button, DefectBadge, EmptyState, ErrorState, Kbd, LoadingState, ProgressBar, Tooltip, VerdictBadge, useToast } from '@/components/ui';
import { cx } from '@/lib/cx';
import { DEFECTS, VERDICT_BY_KEY, defectLabel, isVerdictKey, lotTypeLabel, verdictLabel } from '@/lib/domain';
import { formatNumber, formatPercent } from '@/lib/format';
import { useHotkeys } from '@/lib/hotkeys';
import {
  filterQuery,
  labelingApi,
  lk,
  patchCachedLabel,
  prefetchTrack,
  preloadImages,
  useDashboard,
  useInvalidateLabeling,
  useLots,
  useTrackDetail,
  type ModelRef,
  type TrackDetail,
  type TrackRow,
  type Verdict,
} from './api';
import { FilterBar } from './FilterBar';
import { useLabelingState } from './state';
import s from './labellisation.module.css';

const PAGE = 100;
const PREFETCH_DETAILS = 3;
const PRELOAD_TRACKS = 6;
const PACE_WINDOW_MS = 5 * 60_000;

type Patch = { verdict?: Verdict | null; defect?: string | null };

/** Mêmes règles de cohérence que le backend (mise à jour optimiste). */
function resolveLocal(cur: { verdict: Verdict | null; defect: string | null }, patch: Patch): { verdict: Verdict | null; defect: string | null } {
  if (patch.defect) return { verdict: 'non_viable', defect: patch.defect };
  if (patch.verdict !== undefined) {
    if (patch.verdict === 'non_viable') return { verdict: 'non_viable', defect: cur.verdict === 'non_viable' ? cur.defect : null };
    return { verdict: patch.verdict, defect: null };
  }
  return cur;
}

function rowFromDetail(d: TrackDetail): TrackRow {
  return {
    id: d.id,
    lot_id: d.lot?.id ?? '',
    lot_type: d.lot?.type ?? '',
    lane: d.lane,
    start_frame: d.start_frame,
    start_ts: d.start_ts,
    n_views: d.n_views,
    quality: d.quality,
    flags: d.flags,
    prelabel: d.prelabel,
    verdict: d.label.verdict,
    defect: d.label.defect,
    reviewed: d.label.reviewed,
    source: d.label.source ?? null,
    reviewer: d.label.reviewer ?? '',
    updated_at: d.label.updated_at ?? null,
    note: d.label.note ?? '',
    score: d.prediction?.score ?? null,
    priority: d.prediction?.priority ?? null,
    reason: d.prediction?.reason ?? null,
  };
}

/** Disposition de la mosaïque : nombre de colonnes maximisant la taille des vues dans la place disponible. */
function useMosaicLayout(n: number) {
  const [box, boxRef] = useState<HTMLDivElement | null>(null);
  const [layout, setLayout] = useState({ cols: 4, cell: 160, height: 520 });
  useLayoutEffect(() => {
    if (!box) return;
    const compute = () => {
      const gap = 6;
      const w = box.clientWidth;
      const top = box.getBoundingClientRect().top;
      const h = Math.max(240, window.innerHeight - top - 20);
      let best = { cols: 4, cell: 0 };
      const count = Math.max(1, n);
      for (let cols = 1; cols <= count; cols++) {
        const rows = Math.ceil(count / cols);
        const cell = Math.floor(Math.min((w - gap * (cols - 1)) / cols, (h - gap * (rows - 1)) / rows));
        if (cell > best.cell) best = { cols, cell };
      }
      setLayout({ cols: best.cols, cell: Math.max(60, Math.min(best.cell, 448)), height: h });
    };
    compute();
    const ro = new ResizeObserver(compute);
    ro.observe(box);
    window.addEventListener('resize', compute);
    return () => {
      ro.disconnect();
      window.removeEventListener('resize', compute);
    };
  }, [n, box]);
  return { boxRef, ...layout };
}

function VerdictPill({ verdict, defect, flash }: { verdict: Verdict | null; defect: string | null; flash?: number }) {
  const color = verdict && isVerdictKey(verdict) ? VERDICT_BY_KEY[verdict].color : undefined;
  return (
    <span key={flash} className={cx(s.current, flash ? s.flash : undefined)} style={{ ['--vcolor' as string]: color } as CSSProperties}>
      <span className={s.currentDot} aria-hidden />
      {verdict ? verdictLabel(verdict) : 'Sans label'}
      {defect && <span className="u-muted">· {defectLabel(defect)}</span>}
    </span>
  );
}

function PreLabel({ verdict, defect }: { verdict: Verdict | null; defect: string | null }) {
  if (!verdict) return <span className="u-subtle">aucun</span>;
  return (
    <span className="u-row" style={{ gap: 4 }}>
      <VerdictBadge verdict={verdict} />
      {defect && <DefectBadge defect={defect} />}
    </span>
  );
}

const KEY_ROWS: { keys: string[]; label: string; title?: string; action: string; color?: string }[] = [
  { keys: ['V'], label: 'Viable', action: 'viable', color: 'var(--verdict-viable)' },
  { keys: ['N'], label: 'Non viable', action: 'non_viable', color: 'var(--verdict-non-viable)' },
  { keys: ['I'], label: 'Incertain', action: 'incertain', color: 'var(--verdict-incertain)' },
  { keys: ['X'], label: 'Exclure', title: 'Piste inexploitable (exclue de l’apprentissage)', action: 'exclu', color: 'var(--verdict-exclu)' },
  ...DEFECTS.map((d) => ({ keys: [d.hotkey], label: d.label, title: `Non viable : ${d.label.toLowerCase()}`, action: `defect:${d.key}` })),
];

export function Review() {
  const { filters, activeQueue, setActiveQueue } = useLabelingState();
  const params = useParams();
  const anchorId = params.trackId ? Number(params.trackId) : null;
  const navigate = useNavigate();
  const qc = useQueryClient();
  const toast = useToast();
  const lots = useLots();
  const dashboard = useDashboard();
  const invalidate = useInvalidateLabeling();

  const [items, setItems] = useState<TrackRow[]>([]);
  const [cursor, setCursor] = useState<string | null>(null);
  const [total, setTotal] = useState<number | null>(null);
  const [model, setModel] = useState<ModelRef | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [pos, setPos] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<unknown>(null);
  const [overrides, setOverrides] = useState<Record<number, Partial<TrackRow>>>({});
  const [flash, setFlash] = useState(0);
  const [reloadKey, setReloadKey] = useState(0);
  const [, setTick] = useState(0);

  const gen = useRef(0);
  const loadingMore = useRef(false);
  const chain = useRef<Promise<unknown>>(Promise.resolve());
  const paceTimes = useRef<number[]>([]);
  const sessionCount = useRef(0);
  const writesSinceRefresh = useRef(0);

  const baseQuery = useMemo(() => filterQuery(filters), [filters]);
  const fetchPage = useCallback(
    (extra: Record<string, string | number | boolean | null | undefined>, signal?: AbortSignal) =>
      activeQueue ? labelingApi.active({ ...baseQuery, ...extra }, signal) : labelingApi.tracks({ ...baseQuery, ...extra }, signal),
    [activeQueue, baseQuery],
  );

  // -- chargement initial (filtres, file, lien profond) --------------------
  useEffect(() => {
    const g = ++gen.current;
    const ctrl = new AbortController();
    setLoading(true);
    setError(null);
    setOverrides({});
    loadingMore.current = false;
    (async () => {
      try {
        const [page, detail] = await Promise.all([
          fetchPage({ limit: PAGE, anchor: anchorId ?? undefined, with_total: true }, ctrl.signal),
          anchorId !== null ? labelingApi.track(anchorId, filters.modelId || undefined, ctrl.signal) : Promise.resolve(null),
        ]);
        if (g !== gen.current) return;
        let list = page.items;
        if (detail && list[0]?.id !== detail.id) list = [rowFromDetail(detail), ...list.filter((r) => r.id !== detail.id)];
        setItems(list);
        setCursor(page.next_cursor);
        setTotal(page.total);
        setModel(page.model);
        setMessage(page.message ?? null);
        setPos(0);
      } catch (e) {
        if (g !== gen.current || (e instanceof DOMException && e.name === 'AbortError')) return;
        setError(e);
      } finally {
        if (g === gen.current) setLoading(false);
      }
    })();
    return () => ctrl.abort();
  }, [fetchPage, anchorId, filters.modelId, reloadKey]);

  // -- pages suivantes -------------------------------------------------------
  useEffect(() => {
    if (loading || !cursor || loadingMore.current || pos < items.length - 25) return;
    const g = gen.current;
    loadingMore.current = true;
    fetchPage({ limit: PAGE, cursor, with_total: false })
      .then((page) => {
        if (g !== gen.current) return;
        setItems((prev) => {
          const seen = new Set(prev.map((r) => r.id));
          return [...prev, ...page.items.filter((r) => !seen.has(r.id))];
        });
        setCursor(page.next_cursor);
      })
      .catch((e) => g === gen.current && toast.error(e))
      .finally(() => {
        if (g === gen.current) loadingMore.current = false;
      });
  }, [pos, items.length, cursor, loading, fetchPage, toast]);

  const row: TrackRow | null = items[pos] ? { ...items[pos], ...overrides[items[pos].id] } : null;
  const detail = useTrackDetail(row?.id ?? null, filters.modelId);

  // -- préchargement (détails + images des pistes suivantes) ----------------
  useEffect(() => {
    for (let k = 1; k <= PRELOAD_TRACKS; k++) {
      const r = items[pos + k];
      if (!r) break;
      preloadImages(Array.from({ length: r.n_views }, (_, i) => mediaUrl.crop(r.id, i)));
      if (k <= PREFETCH_DETAILS) void prefetchTrack(qc, r.id, filters.modelId);
    }
    const prev = items[pos - 1];
    if (prev) void prefetchTrack(qc, prev.id, filters.modelId);
  }, [pos, items, qc, filters.modelId]);

  // -- écritures (sérialisées : l'ordre des requêtes suit l'ordre des frappes) ---
  const enqueue = useCallback(<T,>(fn: () => Promise<T>): Promise<T> => {
    const p = chain.current.then(fn, fn);
    chain.current = p.catch(() => undefined);
    return p;
  }, []);

  const afterWrite = useCallback(() => {
    writesSinceRefresh.current += 1;
    if (writesSinceRefresh.current >= 25) {
      writesSinceRefresh.current = 0;
      invalidate();
    }
  }, [invalidate]);

  const recordPace = () => {
    const now = Date.now();
    paceTimes.current = [...paceTimes.current.filter((t) => now - t < PACE_WINDOW_MS), now];
    sessionCount.current += 1;
  };

  const advance = useCallback(() => setPos((p) => Math.min(p + 1, items.length)), [items.length]);

  const writeLabel = useCallback(
    (kind: 'label' | 'confirm', patch: Patch = {}) => {
      if (!row) return;
      const id = row.id;
      const before: Partial<TrackRow> = { verdict: row.verdict, defect: row.defect, reviewed: row.reviewed, source: row.source };
      if (kind === 'confirm' && !row.verdict) {
        toast.warning('Rien à confirmer', 'Cette piste n’a pas encore de label : utilisez V, N, 1–6, I ou X.');
        return;
      }
      const next = kind === 'confirm' ? { verdict: row.verdict, defect: row.defect } : resolveLocal(row, patch);
      const optimistic = { ...next, reviewed: true, source: 'humain' };
      setOverrides((o) => ({ ...o, [id]: { ...o[id], ...optimistic } }));
      patchCachedLabel(qc, id, optimistic);
      setFlash((f) => f + 1);
      recordPace();
      advance();
      enqueue(() => (kind === 'confirm' ? labelingApi.confirm(id) : labelingApi.setLabel(id, patch)))
        .then((res) => {
          setOverrides((o) => ({ ...o, [id]: { ...o[id], ...res.label } }));
          patchCachedLabel(qc, id, res.label);
          afterWrite();
        })
        .catch((e) => {
          setOverrides((o) => ({ ...o, [id]: { ...o[id], ...before } }));
          patchCachedLabel(qc, id, before as Partial<TrackDetail['label']>);
          toast.error('Label non enregistré', `Piste ${id} : ${errorMessage(e)}`);
        });
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [row, qc, toast, advance, enqueue, afterWrite],
  );

  const reopen = useCallback(() => {
    if (!row) return;
    const id = row.id;
    setOverrides((o) => ({ ...o, [id]: { ...o[id], reviewed: false } }));
    patchCachedLabel(qc, id, { reviewed: false });
    enqueue(() => labelingApi.reopen(id))
      .then(() => afterWrite())
      .catch((e) => toast.error(e));
  }, [row, qc, enqueue, afterWrite, toast]);

  const undo = useCallback(() => {
    enqueue(() => labelingApi.undo())
      .then((res) => {
        if (!res.undone) {
          toast.warning(res.message);
          return;
        }
        setOverrides((o) => {
          const n = { ...o };
          for (const l of res.labels) n[l.track_id] = { ...n[l.track_id], verdict: l.verdict, defect: l.defect, reviewed: l.reviewed, source: l.source ?? null };
          return n;
        });
        for (const l of res.labels) patchCachedLabel(qc, l.track_id, l);
        if (res.action === 'marquage_vue') for (const id of res.track_ids) void qc.invalidateQueries({ queryKey: lk.track(id) });
        const idx = items.findIndex((r) => r.id === res.track_ids[0]);
        if (idx >= 0 && res.track_ids.length === 1) setPos(idx);
        sessionCount.current = Math.max(0, sessionCount.current - (res.action === 'marquage_vue' ? 0 : res.undone));
        toast.info(res.message);
        afterWrite();
      })
      .catch((e) => toast.error(e));
  }, [enqueue, qc, items, toast, afterWrite]);

  const toggleView = useCallback(
    (viewIndex: number) => {
      if (!row) return;
      const id = row.id;
      const current = detail.data?.views.find((v) => v.index === viewIndex)?.defect_visible ?? false;
      const setVisible = (visible: boolean) =>
        qc.setQueriesData<TrackDetail>({ queryKey: lk.track(id) }, (old) =>
          old ? { ...old, views: old.views.map((v) => (v.index === viewIndex ? { ...v, defect_visible: visible } : v)) } : old,
        );
      setVisible(!current);
      enqueue(() => labelingApi.markView(id, viewIndex, !current))
        .then(() => afterWrite())
        .catch((e) => {
          setVisible(current);
          toast.error(e);
        });
    },
    [row, detail.data, qc, enqueue, afterWrite, toast],
  );

  const go = useCallback((delta: number) => setPos((p) => Math.max(0, Math.min(items.length - (cursor ? 1 : 0), p + delta))), [items.length, cursor]);

  // -- raccourcis clavier ----------------------------------------------------
  const once = (fn: () => void) => (e: KeyboardEvent) => {
    if (!e.repeat) fn();
  };
  const hotkeys: Record<string, (e: KeyboardEvent) => void> = {
    v: once(() => writeLabel('label', { verdict: 'viable' })),
    n: once(() => writeLabel('label', { verdict: 'non_viable' })),
    i: once(() => writeLabel('label', { verdict: 'incertain' })),
    x: once(() => writeLabel('label', { verdict: 'exclu' })),
    Enter: once(() => writeLabel('confirm')),
    Space: once(() => writeLabel('confirm')),
    ArrowRight: () => go(1),
    ArrowLeft: () => go(-1),
    z: once(undo),
    'mod+z': once(undo),
    r: once(reopen),
  };
  for (const d of DEFECTS) hotkeys[d.hotkey] = once(() => writeLabel('label', { defect: d.key }));
  useHotkeys(hotkeys, { enabled: !loading && !error });

  // rafraîchit le tableau de bord en quittant l'écran
  useEffect(() => () => invalidate(), [invalidate]);

  // cadence : recalcul périodique
  useEffect(() => {
    const t = window.setInterval(() => setTick((x) => x + 1), 5000);
    return () => window.clearInterval(t);
  }, []);

  const lotById = useMemo(() => new Map((lots.data?.items ?? []).map((l) => [l.id, l])), [lots.data]);
  const views = useMemo(() => {
    if (!row) return [];
    const byIndex = new Map((detail.data?.id === row.id ? detail.data.views : []).map((v) => [v.index, v]));
    return Array.from({ length: row.n_views }, (_, i) => ({ index: i, info: byIndex.get(i) }));
  }, [row, detail.data]);
  const layout = useMosaicLayout(row?.n_views ?? 12);

  const now = Date.now();
  const recent = paceTimes.current.filter((t) => now - t < PACE_WINDOW_MS);
  // cadence : œufs/min sur les 5 dernières minutes (au moins 3 saisies, durée minimale 20 s pour éviter les pics)
  const pace = recent.length >= 3 ? ((recent.length - 1) / Math.max(20_000, recent[recent.length - 1] - recent[0])) * 60_000 : null;
  const totalShown = total ?? items.length;
  const dash = dashboard.data?.totals;

  // -------------------------------------------------------------------------- rendu
  if (loading && items.length === 0) {
    return (
      <>
        <FilterBar />
        <LoadingState label="Chargement des pistes…" />
      </>
    );
  }
  if (error) {
    return (
      <>
        <FilterBar />
        <ErrorState error={error} title="Impossible de charger les pistes" onRetry={() => setReloadKey((k) => k + 1)} />
      </>
    );
  }

  const prediction = detail.data?.id === row?.id ? detail.data?.prediction : null;
  const score = row?.score ?? prediction?.score ?? null;
  const threshold = model?.threshold ?? prediction?.model.threshold ?? null;
  const reason = row?.reason ?? prediction?.reason ?? null;
  const lot = row ? lotById.get(row.lot_id) : undefined;
  const atEnd = pos >= items.length;

  return (
    <>
      <FilterBar />
      {items.length === 0 ? (
        <EmptyState
          icon={activeQueue ? Flag : CheckCheck}
          title={activeQueue ? 'File d’apprentissage actif vide' : 'Aucune piste ne correspond aux filtres'}
          description={message ?? (activeQueue ? 'Le modèle ne contredit aucun label et n’hésite sur aucune piste.' : 'Modifiez les filtres ou relancez la détection sur de nouveaux lots.')}
          actions={activeQueue ? <Button onClick={() => setActiveQueue(false)}>Revenir à la liste normale</Button> : undefined}
        />
      ) : (
        <div className={s.review}>
          <div className={s.stage}>
            {row && !atEnd ? (
              <>
                <div className={s.band} aria-live="polite">
                  <VerdictPill verdict={row.verdict} defect={row.defect} flash={flash} />
                  {row.reviewed ? (
                    <Badge tone="success" icon={CheckCheck}>
                      Relue{row.reviewer ? ` · ${row.reviewer}` : ''}
                    </Badge>
                  ) : (
                    <Badge tone="warning">À relire</Badge>
                  )}
                  <span className={s.bandItem}>
                    <span className={s.bandLabel}>Pré-label</span>
                    <PreLabel verdict={row.prelabel.verdict} defect={row.prelabel.defect} />
                  </span>
                  <span className={s.bandItem} title={row.lot_id}>
                    <span className={s.bandLabel}>Lot</span>
                    <span className={s.bandLot}>{lot?.name ?? row.lot_id}</span>
                    <Badge variant="outline">{lotTypeLabel(row.lot_type)}</Badge>
                  </span>
                  <span className={s.bandItem}>
                    <span className={s.bandLabel}>Couloir</span>
                    <span className={s.bandStrong}>{row.lane + 1}</span>
                  </span>
                  {row.quality !== 'ok' && (
                    <Badge tone="warning" icon={AlertTriangle} title={row.flags.join(', ')}>
                      Piste douteuse{row.flags.length ? ` (${row.flags.join(', ')})` : ''}
                    </Badge>
                  )}
                  {score !== null && score !== undefined && (
                    <span className={s.bandItem}>
                      <span className={s.bandLabel}>Modèle</span>
                      <Tooltip content={`Score de défaut ${formatNumber(score, 3)} · seuil ${formatNumber(threshold, 2)}${model ? ` · ${model.name}` : ''}`}>
                        <span className="u-num u-strong">{formatNumber(score, 2)}</span>
                      </Tooltip>
                      {threshold !== null && (
                        <span className="u-small u-muted">→ {score >= threshold ? 'non viable' : 'viable'}</span>
                      )}
                    </span>
                  )}
                  {reason && (
                    <Badge tone={reason === 'contredit' ? 'danger' : 'info'} className={s.reasonBadge}>
                      {reason === 'contredit' ? 'Contredit le label' : 'Hésite'}
                    </Badge>
                  )}
                </div>
                <div className={s.mosaicBox} ref={layout.boxRef}>
                  <div className={s.mosaic} style={{ gridTemplateColumns: `repeat(${layout.cols}, ${layout.cell}px)` }}>
                    {views.map(({ index, info }) => {
                      const marked = !!info?.defect_visible;
                      const vs = prediction?.view_scores?.[index];
                      return (
                        <button
                          key={`${row.id}-${index}`}
                          type="button"
                          className={cx(s.view, marked && s.viewMarked)}
                          style={{ width: layout.cell, height: layout.cell }}
                          onMouseDown={(e) => e.preventDefault()}
                          onClick={() => toggleView(index)}
                          title={marked ? 'Défaut visible sur cette vue (cliquer pour retirer)' : 'Cliquer pour marquer « défaut visible »'}
                          aria-pressed={marked}
                          aria-label={`Vue ${index + 1}${marked ? ', défaut visible' : ''}`}
                        >
                          <img src={mediaUrl.crop(row.id, index)} alt="" draggable={false} decoding="async" />
                          <span className={s.viewIdx}>{index + 1}</span>
                          {vs !== undefined && <span className={cx(s.viewIdx, s.viewScore)}>{formatNumber(vs, 2)}</span>}
                          {info?.partial && <span className={cx(s.viewMark, s.viewPartial)}>tronquée</span>}
                          {marked && (
                            <span className={s.viewMark}>
                              <Flag size={10} aria-hidden /> défaut
                            </span>
                          )}
                        </button>
                      );
                    })}
                  </div>
                </div>
              </>
            ) : (
              <div className={s.endCard}>
                <EmptyState
                  icon={CheckCheck}
                  title="Fin de la liste"
                  description={`${formatNumber(sessionCount.current)} œuf(s) relu(s) pendant cette session.`}
                  actions={
                    <>
                      <Button icon={ListRestart} onClick={() => setReloadKey((k) => k + 1)}>
                        Recharger la liste
                      </Button>
                      <Button variant="ghost" onClick={() => go(-1)}>
                        Revenir à la dernière piste
                      </Button>
                      <Button variant="ghost" onClick={() => navigate('/labellisation/tableau-de-bord')}>
                        Tableau de bord
                      </Button>
                    </>
                  }
                />
              </div>
            )}
          </div>

          <aside className={s.side} style={{ maxHeight: layout.height + 60, overflow: 'auto' }}>
            <div className={s.sideCard}>
              <div className={s.counter}>
                <span className={s.counterMain}>
                  {formatNumber(Math.min(pos + 1, totalShown))} <span className="u-muted">/ {formatNumber(totalShown)}</span>
                </span>
                <span className={s.counterSub}>
                  {activeQueue ? 'file active' : 'pistes filtrées'}
                  {row && !atEnd && <span className="u-mono u-subtle"> · #{row.id}</span>}
                </span>
              </div>
              <ProgressBar value={totalShown ? Math.min(1, (pos + (atEnd ? 0 : 1)) / totalShown) : 0} size="xs" />
              <div className={s.paceRow}>
                <span>
                  Session : <span className={s.paceValue}>{formatNumber(sessionCount.current)}</span>
                </span>
                <span>
                  <span className={s.paceValue}>{pace === null ? '—' : formatNumber(pace, 1)}</span> œufs/min
                </span>
              </div>
              {dash && (
                <div className={s.paceRow}>
                  <span>Projet relu</span>
                  <span className={s.paceValue}>{formatPercent(dash.progress, 1)}</span>
                </div>
              )}
            </div>

            <div className={s.sideCard}>
              <div className={s.sideTitle}>Raccourcis — chaque touche enregistre et avance</div>
              <div className={s.keyList}>
                <div className={s.keyGrid}>
                {KEY_ROWS.map((k) => {
                  const active =
                    row &&
                    (k.action.startsWith('defect:') ? row.defect === k.action.slice(7) : row.verdict === k.action && (k.action !== 'non_viable' || !row.defect));
                  return (
                    <button
                      key={k.action}
                      type="button"
                      className={cx(s.keyRow, active && s.keyRowActive)}
                      title={k.title}
                      disabled={!row || atEnd}
                      onMouseDown={(e) => e.preventDefault()}
                      onClick={() =>
                        k.action.startsWith('defect:')
                          ? writeLabel('label', { defect: k.action.slice(7) })
                          : writeLabel('label', { verdict: k.action as Verdict })
                      }
                    >
                      <Kbd keys={k.keys} />
                      <span>
                        {k.color && <span className={s.swatch} style={{ background: k.color }} />}
                        {k.label}
                      </span>
                    </button>
                  );
                })}
                </div>
                <div className={s.keySep} />
                <button type="button" className={s.keyRow} disabled={!row?.verdict || atEnd} onClick={() => writeLabel('confirm')}>
                  <Kbd keys={['↵']} />
                  <span>Confirmer le label affiché (ou Espace)</span>
                </button>
                <div className={s.keyRow}>
                  <Kbd keys={['←', '→']} />
                  <span>Piste précédente / suivante</span>
                </div>
                <button type="button" className={s.keyRow} onClick={undo}>
                  <Kbd keys={['Z']} />
                  <span>
                    <RotateCcw size={12} aria-hidden style={{ verticalAlign: -1 }} /> Annuler la dernière action
                  </span>
                </button>
                <button type="button" className={s.keyRow} disabled={!row?.reviewed || atEnd} onClick={reopen}>
                  <Kbd keys={['R']} />
                  <span>Remettre à relire</span>
                </button>
                <div className={s.keyRow}>
                  <span className="u-xs u-muted">Clic</span>
                  <span className="u-small u-muted">sur une vue : défaut visible</span>
                </div>
              </div>
            </div>
          </aside>
        </div>
      )}
    </>
  );
}
