import { useCallback, useEffect, useMemo, useRef, useState, type MouseEvent } from 'react';
import { useNavigate } from 'react-router-dom';
import { keepPreviousData, useQuery, useQueryClient } from '@tanstack/react-query';
import { Ban, CheckCheck, ChevronLeft, ChevronRight, Maximize2, ThumbsUp } from 'lucide-react';
import { mediaUrl } from '@/api/client';
import { Button, Checkbox, EmptyState, ErrorState, Kbd, LoadingState, Select, Tooltip, useToast } from '@/components/ui';
import { cx } from '@/lib/cx';
import { VERDICT_BY_KEY, defectLabel, isVerdictKey, verdictLabel } from '@/lib/domain';
import { formatNumber } from '@/lib/format';
import { useHotkeys } from '@/lib/hotkeys';
import { filterQuery, labelingApi, lk, useInvalidateLabeling, type TrackRow, type Verdict } from './api';
import { FilterBar } from './FilterBar';
import { useLabelingState } from './state';
import s from './labellisation.module.css';

const PAGE_SIZES = [48, 96, 192] as const;

function Tile({
  row,
  selected,
  onClick,
  onOpen,
}: {
  row: TrackRow;
  selected: boolean;
  onClick: (e: MouseEvent) => void;
  onOpen: () => void;
}) {
  const rep = Math.floor(row.n_views / 2);
  const [view, setView] = useState(rep);
  const onMove = (e: MouseEvent<HTMLDivElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const i = Math.min(row.n_views - 1, Math.max(0, Math.floor(((e.clientX - r.left) / r.width) * row.n_views)));
    if (i !== view) setView(i);
  };
  const color = row.verdict && isVerdictKey(row.verdict) ? VERDICT_BY_KEY[row.verdict].color : 'var(--text-subtle)';
  return (
    <div
      role="checkbox"
      aria-checked={selected}
      tabIndex={-1}
      className={cx(s.tile, selected && s.tileSelected)}
      onClick={onClick}
      onDoubleClick={onOpen}
      title="Clic : sélectionner · Maj+clic : plage · double-clic : ouvrir en relecture"
    >
      <div className={s.tileImg} onMouseMove={onMove} onMouseLeave={() => setView(rep)}>
        <img src={mediaUrl.crop(row.id, view)} alt={`Piste ${row.id}, vue ${view + 1}`} draggable={false} loading="lazy" decoding="async" />
        <span className={s.tileCheck} onClick={(e) => e.stopPropagation()}>
          <Checkbox checked={selected} onChange={() => onClick({ shiftKey: false } as MouseEvent)} aria-label={`Sélectionner la piste ${row.id}`} />
        </span>
        <button
          type="button"
          className={s.tileOpen}
          onClick={(e) => {
            e.stopPropagation();
            onOpen();
          }}
          aria-label={`Ouvrir la piste ${row.id} en relecture`}
        >
          <Maximize2 size={12} />
        </button>
        <div className={s.tileScrub} aria-hidden>
          {Array.from({ length: row.n_views }, (_, i) => (
            <span key={i} className={cx(i === view && s.on)} />
          ))}
        </div>
      </div>
      <div className={s.tileMeta}>
        <span className={s.tileVerdict} style={{ background: color }} aria-hidden />
        <span className="u-truncate">{row.verdict ? verdictLabel(row.verdict) : 'Sans label'}</span>
        {row.defect && <span className="u-muted u-truncate">· {defectLabel(row.defect)}</span>}
        {row.reviewed && <CheckCheck size={12} className={s.tileDone} aria-label="relue" />}
        {row.score !== undefined && row.score !== null && <span className="u-num u-muted">{formatNumber(row.score, 2)}</span>}
        <span className={s.tileId}>#{row.id}</span>
      </div>
    </div>
  );
}

export function Grid() {
  const { filters } = useLabelingState();
  const navigate = useNavigate();
  const toast = useToast();
  const qc = useQueryClient();
  const invalidate = useInvalidateLabeling();
  const [pageSize, setPageSize] = useState<number>(96);
  const [cursors, setCursors] = useState<(string | null)[]>([null]);
  const [pageIdx, setPageIdx] = useState(0);
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [busy, setBusy] = useState(false);
  const lastIndex = useRef<number | null>(null);

  const baseQuery = useMemo(() => filterQuery(filters), [filters]);
  // Filtres ou taille de page modifiés : retour à la première page.
  useEffect(() => {
    setCursors([null]);
    setPageIdx(0);
    setSelected(new Set());
    lastIndex.current = null;
  }, [baseQuery, pageSize]);

  const cursor = cursors[pageIdx] ?? null;
  const page = useQuery({
    queryKey: ['labeling', 'grid', baseQuery, pageSize, cursor] as const,
    queryFn: ({ signal }) => labelingApi.tracks({ ...baseQuery, limit: pageSize, cursor, with_total: true }, signal),
    placeholderData: keepPreviousData,
  });
  const rows = useMemo(() => page.data?.items ?? [], [page.data]);
  const total = page.data?.total ?? null;

  useEffect(() => {
    const next = page.data?.next_cursor;
    if (next && cursors.length === pageIdx + 1) setCursors((c) => [...c, next]);
  }, [page.data, pageIdx, cursors.length]);

  const onTileClick = useCallback(
    (index: number, e: MouseEvent) => {
      const id = rows[index].id;
      const anchor = lastIndex.current; // lu avant la mise à jour (l'updater s'exécute plus tard)
      const range = e.shiftKey && anchor !== null;
      setSelected((prev) => {
        const next = new Set(prev);
        if (range) {
          const [a, b] = [anchor, index].sort((x, y) => x - y);
          for (let i = a; i <= b; i++) next.add(rows[i].id);
        } else if (next.has(id)) next.delete(id);
        else next.add(id);
        return next;
      });
      lastIndex.current = index;
    },
    [rows],
  );

  const ids = useMemo(() => rows.filter((r) => selected.has(r.id)).map((r) => r.id), [rows, selected]);
  const allSelected = rows.length > 0 && ids.length === rows.length;

  const undo = useCallback(async () => {
    try {
      const res = await labelingApi.undo();
      if (res.undone) toast.info(res.message);
      else toast.warning(res.message);
      void qc.invalidateQueries({ queryKey: ['labeling', 'grid'] });
      invalidate();
    } catch (e) {
      toast.error(e);
    }
  }, [qc, toast, invalidate]);

  const runBulk = useCallback(
    async (body: { verdict?: Verdict; confirm?: boolean }, done: string) => {
      if (!ids.length || busy) return;
      setBusy(true);
      try {
        const res = await labelingApi.bulk({ track_ids: ids, ...body });
        const skipped = res.skipped.length ? ` · ${formatNumber(res.skipped.length)} sans label ignorée(s)` : '';
        toast.show({
          title: `${formatNumber(res.updated)} piste(s) ${done}`,
          description: skipped ? skipped.slice(3) : undefined,
          tone: 'success',
          action: { label: 'Annuler', onClick: () => void undo() },
        });
        setSelected(new Set());
        lastIndex.current = null;
        for (const id of ids) void qc.invalidateQueries({ queryKey: lk.track(id) });
        await qc.invalidateQueries({ queryKey: ['labeling', 'grid'] });
        invalidate();
      } catch (e) {
        toast.error(e);
      } finally {
        setBusy(false);
      }
    },
    [ids, busy, toast, undo, qc, invalidate],
  );

  const selectAll = () => setSelected(allSelected ? new Set() : new Set(rows.map((r) => r.id)));
  const open = (id: number) => navigate(`/labellisation/piste/${id}`);

  useHotkeys({
    'mod+a': () => setSelected(new Set(rows.map((r) => r.id))),
    Escape: () => setSelected(new Set()),
    v: () => void runBulk({ verdict: 'viable' }, 'validée(s) comme viables'),
    Enter: () => void runBulk({ confirm: true }, 'confirmée(s)'),
    x: () => void runBulk({ verdict: 'exclu' }, 'exclue(s)'),
    z: () => void undo(),
    'mod+z': () => void undo(),
    ArrowRight: () => cursors[pageIdx + 1] && setPageIdx((p) => p + 1),
    ArrowLeft: () => pageIdx > 0 && setPageIdx((p) => p - 1),
  });

  const from = pageIdx * pageSize;
  return (
    <>
      <FilterBar showQueueToggle={false} />
      <div className={s.gridToolbar}>
        <Checkbox
          checked={allSelected}
          indeterminate={ids.length > 0 && !allSelected}
          onChange={selectAll}
          label={ids.length ? `${formatNumber(ids.length)} sélectionnée(s)` : 'Tout sélectionner'}
        />
        <Tooltip content={<span>Raccourci <Kbd>V</Kbd></span>}>
          <Button variant="primary" size="sm" icon={ThumbsUp} disabled={!ids.length} loading={busy} onClick={() => void runBulk({ verdict: 'viable' }, 'validée(s) comme viables')}>
            Valider {ids.length ? formatNumber(ids.length) : ''} comme viables
          </Button>
        </Tooltip>
        <Tooltip content={<span>Raccourci <Kbd>↵</Kbd> — marque relues sans changer leur label</span>}>
          <Button size="sm" icon={CheckCheck} disabled={!ids.length || busy} onClick={() => void runBulk({ confirm: true }, 'confirmée(s)')}>
            Confirmer le pré-label
          </Button>
        </Tooltip>
        <Tooltip content={<span>Raccourci <Kbd>X</Kbd> — pistes inexploitables</span>}>
          <Button size="sm" variant="ghost" icon={Ban} disabled={!ids.length || busy} onClick={() => void runBulk({ verdict: 'exclu' }, 'exclue(s)')}>
            Exclure
          </Button>
        </Tooltip>
        <span className="u-spacer" />
        <span className="u-small u-muted">
          {total === null ? '' : total === 0 ? 'Aucune piste' : `${formatNumber(from + 1)}–${formatNumber(from + rows.length)} sur ${formatNumber(total)}`}
        </span>
        <Select
          size="sm"
          aria-label="Pistes par page"
          value={String(pageSize)}
          onChange={(v) => setPageSize(Number(v))}
          options={PAGE_SIZES.map((n) => ({ value: String(n), label: `${n} par page` }))}
        />
        <Button size="sm" variant="secondary" icon={ChevronLeft} disabled={pageIdx === 0} onClick={() => setPageIdx((p) => p - 1)} aria-label="Page précédente" />
        <Button size="sm" variant="secondary" icon={ChevronRight} disabled={!cursors[pageIdx + 1]} onClick={() => setPageIdx((p) => p + 1)} aria-label="Page suivante" />
      </div>

      {page.isLoading ? (
        <LoadingState label="Chargement des pistes…" />
      ) : page.isError ? (
        <ErrorState error={page.error} onRetry={() => page.refetch()} />
      ) : rows.length === 0 ? (
        <EmptyState icon={CheckCheck} title="Aucune piste" description="Aucune piste ne correspond aux filtres. Astuce : filtrez sur les lots viables « à relire » pour valider en masse les œufs évidents." />
      ) : (
        <div className={s.tiles} style={{ opacity: page.isPlaceholderData ? 0.6 : 1 }}>
          {rows.map((r, i) => (
            <Tile key={r.id} row={r} selected={selected.has(r.id)} onClick={(e) => onTileClick(i, e)} onOpen={() => open(r.id)} />
          ))}
        </div>
      )}
      {rows.length > 0 && (
        <div className={s.gridFoot}>
          <Button size="sm" variant="secondary" icon={ChevronLeft} disabled={pageIdx === 0} onClick={() => setPageIdx((p) => p - 1)}>
            Précédente
          </Button>
          <span className="u-small u-muted">Page {pageIdx + 1}</span>
          <Button size="sm" variant="secondary" iconRight={ChevronRight} disabled={!cursors[pageIdx + 1]} onClick={() => setPageIdx((p) => p + 1)}>
            Suivante
          </Button>
        </div>
      )}
    </>
  );
}
