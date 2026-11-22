import { useMemo, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useInfiniteQuery, useQuery } from '@tanstack/react-query';
import { ArrowRight, Eye, EyeOff, X } from 'lucide-react';
import { api } from '@/api/client';
import { Badge, Button, DataTable, DefectBadge, NumberInput, Panel, Select, VerdictBadge, type Column } from '@/components/ui';
import { formatDateTime, formatNumber, formatRelative } from '@/lib/format';
import { labelingApi, lk, useLabelingMeta, type HistoryEntry, type Verdict } from './api';
import s from './labellisation.module.css';

const PAGE = 100;

function LabelSnap({ snap }: { snap: Record<string, unknown> }) {
  if (!snap || Object.keys(snap).length === 0) return <span className="u-subtle">—</span>;
  if ('view_index' in snap) {
    const on = !!snap.defect_visible;
    return (
      <Badge tone={on ? 'danger' : 'neutral'} variant="outline" icon={on ? Eye : EyeOff}>
        Vue {Number(snap.view_index) + 1} : {on ? 'défaut visible' : 'non marquée'}
      </Badge>
    );
  }
  const verdict = (snap.verdict as Verdict | null | undefined) ?? null;
  const defect = (snap.defect as string | null | undefined) ?? null;
  return (
    <span className={s.change}>
      {verdict ? <VerdictBadge verdict={verdict} /> : <span className="u-subtle">sans label</span>}
      {defect && <DefectBadge defect={defect} />}
      {'reviewed' in snap && <span className="u-xs u-muted">{snap.reviewed ? 'relue' : 'à relire'}</span>}
    </span>
  );
}

export function History() {
  const navigate = useNavigate();
  const meta = useLabelingMeta();
  const [user, setUser] = useState('');
  const [action, setAction] = useState('');
  const [trackId, setTrackId] = useState<number | null>(null);

  const users = useQuery({
    queryKey: lk.users,
    queryFn: ({ signal }) => api.get<string[]>('/labeling/history/users', undefined, { signal }),
  });
  const query = useMemo(() => ({ user, action, track_id: trackId ?? undefined, limit: PAGE }), [user, action, trackId]);
  const hist = useInfiniteQuery({
    queryKey: lk.history(query),
    initialPageParam: null as number | null,
    queryFn: ({ pageParam, signal }) => labelingApi.history({ ...query, before_id: pageParam ?? undefined, with_total: pageParam === null }, signal),
    getNextPageParam: (last) => last.next_before_id,
  });
  const rows = useMemo(() => hist.data?.pages.flatMap((p) => p.items) ?? [], [hist.data]);
  const total = hist.data?.pages[0]?.total ?? null;

  const columns: Column<HistoryEntry>[] = [
    {
      key: 'at',
      header: 'Quand',
      width: 170,
      render: (r) => (
        <span title={formatRelative(r.at)} className="u-num">
          {formatDateTime(r.at)}
        </span>
      ),
    },
    { key: 'user', header: 'Qui', width: 130, truncate: true, render: (r) => (r.user === 'auto' ? <span className="u-muted">automatique</span> : r.user) },
    { key: 'action', header: 'Action', width: 170, render: (r) => <Badge tone={r.action === 'annulation' ? 'warning' : r.action === 'prelabel' ? 'neutral' : 'info'}>{r.action_label}</Badge> },
    {
      key: 'track',
      header: 'Piste',
      width: 200,
      render: (r) => (
        <span className="u-col" style={{ gap: 0 }}>
          <Link to={`/labellisation/piste/${r.track_id}`} onClick={(e) => e.stopPropagation()} className="u-mono">
            #{r.track_id}
          </Link>
          <span className="u-xs u-subtle u-truncate" title={r.lot_id}>
            {r.lot_id} · couloir {r.lane + 1}
          </span>
        </span>
      ),
    },
    {
      key: 'change',
      header: 'Modification',
      render: (r) => (
        <span className={s.change}>
          <LabelSnap snap={r.before} />
          <ArrowRight size={13} className={s.changeArrow} aria-hidden />
          <LabelSnap snap={r.after} />
        </span>
      ),
    },
  ];

  const filtered = user || action || trackId !== null;
  return (
    <>
      <div className={s.histFilters}>
        <Select
          size="sm"
          aria-label="Auteur"
          value={user}
          onChange={setUser}
          options={[{ value: '', label: 'Tous les auteurs' }, ...(users.data ?? []).map((u) => ({ value: u, label: u === 'auto' ? 'automatique' : u }))]}
        />
        <Select
          size="sm"
          aria-label="Action"
          value={action}
          onChange={setAction}
          options={[{ value: '', label: 'Toutes les actions' }, ...(meta.data?.actions ?? []).map((a) => ({ value: a.key, label: a.label }))]}
        />
        <NumberInput size="sm" aria-label="Piste" placeholder="n° de piste" value={trackId} onChange={setTrackId} allowEmpty stepper={false} min={1} />
        {filtered && (
          <Button size="sm" variant="ghost" icon={X} onClick={() => { setUser(''); setAction(''); setTrackId(null); }}>
            Effacer les filtres
          </Button>
        )}
        <span className="u-spacer" />
        {total !== null && <span className="u-small u-muted">{formatNumber(total)} entrée(s)</span>}
      </div>
      <Panel flush>
        <DataTable
          columns={columns}
          rows={rows}
          rowKey={(r) => r.id}
          loading={hist.isLoading}
          error={hist.error}
          onRetry={() => hist.refetch()}
          onRowClick={(r) => navigate(`/labellisation/piste/${r.track_id}`)}
          empty="Aucune modification enregistrée."
          dense
          stickyHeader
        />
      </Panel>
      {hist.hasNextPage && (
        <div className={s.histMore}>
          <Button variant="secondary" loading={hist.isFetchingNextPage} onClick={() => hist.fetchNextPage()}>
            Afficher {PAGE} entrées de plus
          </Button>
        </div>
      )}
    </>
  );
}
