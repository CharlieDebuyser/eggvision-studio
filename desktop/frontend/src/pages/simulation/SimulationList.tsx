/** Liste des simulations (paginée) avec sélection pour la comparaison. */
import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Columns3, FlaskConical, Plus } from 'lucide-react';
import { formatDateTime, formatNumber, formatPercent } from '@/lib/format';
import { Badge, Button, Checkbox, DataTable, Pagination, Panel, type Column } from '@/components/ui';
import { useSimList } from './api';
import { SimStatusBadge } from './SimStatusBadge';
import type { SimSummary } from './types';
import s from './simulation.module.css';

const PAGE = 20;

export function SimulationList() {
  const navigate = useNavigate();
  const [page, setPage] = useState(0);
  const [picked, setPicked] = useState<string[]>([]);
  const list = useSimList(page, PAGE);
  const toggle = (id: string) => setPicked((p) => (p.includes(id) ? p.filter((x) => x !== id) : [...p, id].slice(-12)));

  const columns: Column<SimSummary>[] = [
    {
      key: 'pick',
      header: <span className="u-sr-only">Comparer</span>,
      width: 36,
      render: (r) => (
        <span onClick={(e) => e.stopPropagation()}>
          <Checkbox checked={picked.includes(r.id)} onChange={() => toggle(r.id)} disabled={r.status !== 'done'} aria-label={`Comparer ${r.name}`} />
        </span>
      ),
    },
    {
      key: 'name',
      header: 'Simulation',
      truncate: true,
      render: (r) => (
        <div className={s.nameCell}>
          <span className="u-strong u-truncate">{r.name}</span>
          <span className="u-subtle u-xs">{formatDateTime(r.created_at)}</span>
        </div>
      ),
    },
    { key: 'model', header: 'Modèle', truncate: true, render: (r) => r.model_name ?? r.model_id ?? '—' },
    {
      key: 'cfg',
      header: 'Colonnes · défauts · seuil',
      render: (r) => (
        <span className="u-num">
          {formatNumber(r.n_columns)} · {formatPercent(r.defect_rate, 0)} · {r.threshold === null ? 'modèle' : formatNumber(r.threshold, 3)}
        </span>
      ),
    },
    { key: 'status', header: 'État', width: 110, render: (r) => <SimStatusBadge status={r.status} /> },
    { key: 'recall', header: 'Éjectés', align: 'right', width: 90, render: (r) => formatPercent(r.recall), sortValue: (r) => r.recall ?? -1 },
    {
      key: 'fr',
      header: 'Éjectés à tort',
      align: 'right',
      width: 110,
      render: (r) => formatPercent(r.false_reject_rate, 2),
      sortValue: (r) => r.false_reject_rate ?? -1,
    },
    {
      key: 'late',
      header: 'Retards',
      align: 'right',
      width: 90,
      render: (r) => (r.late_rate ? <Badge tone="warning">{formatPercent(r.late_rate, 2)}</Badge> : formatPercent(r.late_rate, 0)),
      sortValue: (r) => r.late_rate ?? -1,
    },
  ];

  const items = list.data?.items ?? [];
  return (
    <Panel
      flush
      title="Simulations"
      description="Cocher des simulations terminées pour les comparer côte à côte."
      actions={
        <>
          <Button icon={Columns3} disabled={picked.length < 2} onClick={() => navigate(`/simulation/comparer?ids=${picked.map(encodeURIComponent).join(',')}`)}>
            Comparer{picked.length ? ` (${picked.length})` : ''}
          </Button>
          <Button icon={Plus} variant="primary" onClick={() => navigate('/simulation/nouvelle')}>
            Nouvelle simulation
          </Button>
        </>
      }
    >
      <DataTable
        columns={columns}
        rows={items}
        rowKey={(r) => r.id}
        onRowClick={(r) => navigate(`/simulation/${encodeURIComponent(r.id)}`)}
        loading={list.isLoading}
        error={list.error}
        onRetry={() => list.refetch()}
        minWidth={900}
        empty={
          <div className={s.emptyList}>
            <FlaskConical size={22} aria-hidden />
            <p>Aucune simulation. Rejouez le jeu test d'un modèle exporté pour mesurer le tri réel, retards compris.</p>
            <Button variant="primary" icon={Plus} onClick={() => navigate('/simulation/nouvelle')}>
              Nouvelle simulation
            </Button>
          </div>
        }
      />
      {(list.data?.total ?? 0) > PAGE && (
        <div className={s.pagination}>
          <Pagination page={page} pageSize={PAGE} total={list.data?.total ?? 0} onPageChange={setPage} />
        </div>
      )}
    </Panel>
  );
}
