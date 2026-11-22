import { useNavigate } from 'react-router-dom';
import { AlertTriangle, CheckCheck, Database, Flag, Info, RotateCw, Route as RouteIcon, ScanEye } from 'lucide-react';
import { Badge, Button, Callout, DataTable, ErrorState, IconButton, LoadingState, Panel, ProgressBar, Stat, type Column } from '@/components/ui';
import { VERDICT_BY_KEY, isVerdictKey, lotTypeLabel } from '@/lib/domain';
import { formatDate, formatNumber, formatPercent } from '@/lib/format';
import { EMPTY_FILTERS, useDashboard, type CountPair, type Dashboard as DashboardData } from './api';
import { useLabelingState } from './state';
import s from './labellisation.module.css';

type LotRow = DashboardData['lots'][number];
type SessionRow = DashboardData['sessions'][number];

function Bars({ rows, max, minExamples }: { rows: { key: string; label: string; color: string; counts: CountPair; checkMin: boolean }[]; max: number; minExamples: number }) {
  const scale = Math.max(1, max, minExamples);
  return (
    <div>
      {rows.map((r) => {
        const low = r.checkMin && r.counts.total < minExamples;
        return (
          <div key={r.key} className={s.barRow}>
            <span className="u-truncate">{r.label}</span>
            <div className={s.barTrack} style={{ ['--bar' as string]: r.color }} title={`${formatNumber(r.counts.reviewed)} relues sur ${formatNumber(r.counts.total)}`}>
              <div className={s.barTotal} style={{ width: `${(r.counts.total / scale) * 100}%` }} />
              <div className={s.barReviewed} style={{ width: `${(r.counts.reviewed / scale) * 100}%` }} />
              {r.checkMin && <div className={s.barMin} style={{ left: `${(minExamples / scale) * 100}%` }} title={`Seuil d’alerte : ${minExamples} exemples`} />}
            </div>
            <span className={`${s.barNums} ${low ? s.barLow : ''}`}>
              {formatNumber(r.counts.total)} <span className="u-subtle">· {formatNumber(r.counts.reviewed)} relues</span>
            </span>
          </div>
        );
      })}
    </div>
  );
}

export function Dashboard() {
  const dash = useDashboard();
  const navigate = useNavigate();
  const { setFilters, filters, setActiveQueue } = useLabelingState();

  if (dash.isLoading) return <LoadingState label="Calcul du tableau de bord…" />;
  if (dash.isError || !dash.data) return <ErrorState error={dash.error} onRetry={() => dash.refetch()} />;
  const d = dash.data;
  const t = d.totals;

  const openLot = (lotId: string) => {
    setActiveQueue(false);
    setFilters({ ...EMPTY_FILTERS, modelId: filters.modelId, lots: [lotId], status: 'a_relire' });
    navigate('/labellisation');
  };

  const verdictRows = d.verdicts.map((v) => ({
    key: v.verdict ?? 'aucun',
    label: v.label,
    color: v.verdict && isVerdictKey(v.verdict) ? VERDICT_BY_KEY[v.verdict].color : 'var(--chart-6)',
    counts: v,
    checkMin: v.verdict === 'viable' || v.verdict === 'non_viable',
  }));
  const defectRows = d.defects
    .filter((x) => x.defect !== null || x.total > 0)
    .map((x, i) => ({ key: x.defect ?? 'aucun', label: x.label, color: x.defect ? `var(--chart-${(i % 5) + 1})` : 'var(--warning)', counts: x, checkMin: x.defect !== null }));

  const lotColumns: Column<LotRow>[] = [
    { key: 'name', header: 'Lot', truncate: true, sortValue: (r) => r.name, render: (r) => <span title={r.lot_id}>{r.name}</span> },
    { key: 'type', header: 'Type', width: 150, sortValue: (r) => r.type, render: (r) => <Badge variant="outline">{lotTypeLabel(r.type)}</Badge> },
    { key: 'farm', header: 'Élevage', width: 130, truncate: true, sortValue: (r) => r.farm },
    { key: 'day', header: 'Journée', width: 95, sortValue: (r) => r.session_date, render: (r) => (r.session_date ? formatDate(r.session_date) : '—') },
    { key: 'total', header: 'Pistes', width: 70, align: 'right', sortValue: (r) => r.total, render: (r) => <span className="u-num">{formatNumber(r.total)}</span> },
    {
      key: 'progress',
      header: 'Relues',
      width: 150,
      sortValue: (r) => r.progress,
      render: (r) => (
        <span className={s.progressCell}>
          <ProgressBar value={r.progress} size="xs" tone={r.progress >= 1 ? 'success' : 'accent'} />
          <span className="u-num u-small">{formatPercent(r.progress, 0)}</span>
        </span>
      ),
    },
    { key: 'doubtful', header: 'Douteuses', width: 85, align: 'right', sortValue: (r) => r.doubtful, render: (r) => (r.doubtful ? <span className="u-num">{formatNumber(r.doubtful)}</span> : <span className="u-subtle">—</span>) },
    {
      key: 'open',
      header: '',
      width: 76,
      align: 'right',
      render: (r) =>
        r.reviewed < r.total ? (
          <Button size="sm" variant="ghost" onClick={(e) => { e.stopPropagation(); openLot(r.lot_id); }}>
            Relire
          </Button>
        ) : (
          <CheckCheck size={14} className="tone-fg--success" aria-label="Lot entièrement relu" />
        ),
    },
  ];

  const sessionColumns: Column<SessionRow>[] = [
    { key: 'farm', header: 'Élevage', truncate: true, sortValue: (r) => r.farm, render: (r) => r.farm || <span className="u-subtle">non renseigné</span> },
    { key: 'day', header: 'Journée', width: 110, sortValue: (r) => r.session_date, render: (r) => (r.session_date ? formatDate(r.session_date) : '—') },
    { key: 'lots', header: 'Lots', width: 60, align: 'right', render: (r) => formatNumber(r.lots) },
    { key: 'total', header: 'Pistes', width: 80, align: 'right', render: (r) => formatNumber(r.total) },
    {
      key: 'progress',
      header: 'Relues',
      width: 170,
      sortValue: (r) => r.progress,
      render: (r) => (
        <span className={s.progressCell}>
          <ProgressBar value={r.progress} size="xs" tone={r.progress >= 1 ? 'success' : 'accent'} />
          <span className="u-num u-small">{formatPercent(r.progress, 0)}</span>
        </span>
      ),
    },
  ];

  return (
    <>
      <div className={s.dashStats}>
        <Stat label="Pistes (œufs)" value={formatNumber(t.tracks)} icon={RouteIcon} />
        <Stat
          label="Relues"
          value={formatPercent(t.progress, 1)}
          hint={`${formatNumber(t.reviewed)} relues · ${formatNumber(t.unreviewed)} à relire`}
          icon={CheckCheck}
          onClick={() => {
            setFilters({ ...filters, status: 'a_relire' });
            navigate('/labellisation');
          }}
        />
        <Stat label="Utilisables" value={formatNumber(t.usable)} hint={`viables + non viables · ${formatNumber(t.usable_reviewed)} relues`} icon={Database} />
        <Stat
          label="Pistes douteuses"
          value={formatNumber(t.doubtful)}
          hint={t.doubtful_unreviewed ? `${formatNumber(t.doubtful_unreviewed)} à relire` : 'toutes relues'}
          icon={AlertTriangle}
          tone={t.doubtful_unreviewed ? 'warning' : undefined}
          onClick={() => {
            setFilters({ ...EMPTY_FILTERS, modelId: filters.modelId, quality: 'douteuse' });
            navigate('/labellisation');
          }}
        />
        <Stat
          label="Apprentissage actif"
          value={d.active ? formatNumber(d.active.total) : '—'}
          hint={d.active ? `${formatNumber(d.active.contradictions)} contradictions · ${formatNumber(d.active.hesitations)} hésitations` : 'aucun modèle entraîné'}
          icon={ScanEye}
          onClick={
            d.active
              ? () => {
                  setActiveQueue(true);
                  navigate('/labellisation');
                }
              : undefined
          }
        />
      </div>

      <div style={{ marginBottom: 'var(--sp-4)' }}>
        <ProgressBar value={t.progress} showValue label="Avancement de la relecture" size="md" />
      </div>

      {d.alerts.length > 0 && (
        <div className={s.alerts}>
          {d.alerts.map((a, i) => (
            <Callout key={`${a.code}-${i}`} tone={a.level === 'warning' ? 'warning' : a.level === 'error' ? 'danger' : 'info'} icon={a.level === 'warning' ? AlertTriangle : a.code === 'contradictions_modele' ? Flag : Info}>
              {a.message}
            </Callout>
          ))}
        </div>
      )}

      <div className={s.dashGrid}>
        <Panel title="Par label" description={`Classes d’apprentissage : alerte sous ${d.min_examples} exemples (trait orange).`}>
          <Bars rows={verdictRows} max={Math.max(...verdictRows.map((r) => r.counts.total))} minExamples={d.min_examples} />
          <div className={s.legend}>
            <span>
              <span className={s.legendSwatch} style={{ background: 'var(--text-muted)' }} />
              relues
            </span>
            <span>
              <span className={s.legendSwatch} style={{ background: 'color-mix(in srgb, var(--text-muted) 35%, transparent)' }} />
              total (pré-labels compris)
            </span>
            <span>
              <span className={s.legendSwatch} style={{ background: 'var(--warning)', width: 3 }} />
              seuil d’alerte
            </span>
          </div>
        </Panel>
        <Panel title="Par type de défaut" description="Œufs non viables, par type de défaut.">
          <Bars rows={defectRows} max={Math.max(...defectRows.map((r) => r.counts.total))} minExamples={d.min_examples} />
        </Panel>
      </div>

      <Panel
        title="Par lot"
        description={d.lots_total > d.lots.length ? `${formatNumber(d.lots.length)} premiers lots sur ${formatNumber(d.lots_total)}.` : undefined}
        actions={<IconButton icon={RotateCw} label="Recalculer" variant="ghost" loading={dash.isFetching} onClick={() => dash.refetch()} />}
        flush
      >
        <DataTable columns={lotColumns} rows={d.lots} rowKey={(r) => r.lot_id} onRowClick={(r) => openLot(r.lot_id)} dense stickyHeader minWidth={820} />
      </Panel>
      <div style={{ height: 'var(--sp-4)' }} />
      <Panel title="Par élevage et journée" flush>
        <DataTable columns={sessionColumns} rows={d.sessions} rowKey={(r) => `${r.farm}|${r.session_date}`} dense />
      </Panel>
    </>
  );
}
