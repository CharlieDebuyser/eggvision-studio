/** Liste versionnée des modèles : statut, données, métriques clés ; sélection pour comparer. */
import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { GitCompare, Plus } from 'lucide-react';
import { Badge, Button, DataTable, EmptyState, Pagination, Panel, ProgressBar, type Column } from '@/components/ui';
import { useJob } from '@/api/hooks';
import { formatDateTime, formatDuration } from '@/lib/format';
import { useModels, useOptions } from './api';
import { ModelStatusBadge, archLabel, num3, pct } from './common';
import type { ModelSummary } from './types';
import s from './entrainement.module.css';

const PAGE = 25;

export function ModelList() {
  const navigate = useNavigate();
  const [page, setPage] = useState(0);
  const [selected, setSelected] = useState<string[]>([]);
  const models = useModels(page, PAGE);
  const options = useOptions();
  const archs = options.data?.architectures;
  const toggle = (id: string) => setSelected((old) => (old.includes(id) ? old.filter((x) => x !== id) : [...old, id].slice(-6)));

  const columns: Column<ModelSummary>[] = [
    {
      key: 'sel',
      header: '',
      width: 36,
      render: (m) =>
        m.status === 'trained' ? (
          <input
            type="checkbox"
            checked={selected.includes(m.id)}
            onClick={(e) => e.stopPropagation()}
            onChange={() => toggle(m.id)}
            aria-label={`Sélectionner ${m.name} pour la comparaison`}
          />
        ) : null,
    },
    {
      key: 'name',
      header: 'Modèle',
      width: '28%',
      render: (m) => (
        <div className={s.nameCell}>
          <span className="u-strong u-truncate">{m.name}</span>
          <span className="u-xs u-muted">
            {m.kind === 'anomaly' ? 'Anomalies · ' : ''}
            {archLabel(archs, m.arch)} · {m.input_size} px · {m.n_views} vues
            {m.parent_id ? ' · prolongation' : ''}
          </span>
        </div>
      ),
      sortValue: (m) => m.number ?? 0,
    },
    { key: 'status', header: 'Statut', render: (m) => <StatusCell m={m} />, sortValue: (m) => m.status },
    { key: 'split', header: 'Découpage', render: (m) => (m.split_id != null ? `n° ${m.split_id}` : '—') },
    { key: 'auc', header: 'AUC test', align: 'right', render: (m) => num3(m.metrics.test?.auc_roc), sortValue: (m) => m.metrics.test?.auc_roc ?? -1 },
    { key: 'recall', header: 'Rappel', align: 'right', render: (m) => pct(m.metrics.test?.recall), sortValue: (m) => m.metrics.test?.recall ?? -1 },
    {
      key: 'fr',
      header: 'Faux rejets',
      align: 'right',
      render: (m) => pct(m.metrics.test?.false_reject_rate),
      sortValue: (m) => m.metrics.test?.false_reject_rate ?? 2,
    },
    { key: 'thr', header: 'Seuil', align: 'right', render: (m) => num3(m.threshold) },
    {
      key: 'export',
      header: 'Export',
      render: (m) => (m.exported ? m.export_outdated ? <Badge tone="warning">obsolète</Badge> : <Badge tone="success">exporté</Badge> : <span className="u-subtle">—</span>),
    },
    { key: 'date', header: 'Créé', render: (m) => formatDateTime(m.created_at), sortValue: (m) => m.created_at ?? '' },
  ];

  return (
    <div className="u-stack-4">
      <div className={s.toolbar}>
        <Button variant="primary" icon={Plus} onClick={() => navigate('/entrainement/nouveau')}>
          Nouvel entraînement
        </Button>
        <Button icon={GitCompare} disabled={selected.length < 2} onClick={() => navigate(`/entrainement/comparer?ids=${selected.map(encodeURIComponent).join(',')}`)}>
          Comparer ({selected.length})
        </Button>
        {selected.length > 0 && (
          <Button variant="ghost" size="sm" onClick={() => setSelected([])}>
            Effacer la sélection
          </Button>
        )}
      </div>
      <Panel flush>
        <DataTable
          columns={columns}
          rows={models.data?.items}
          rowKey={(m) => m.id}
          loading={models.isLoading}
          error={models.error}
          onRetry={() => models.refetch()}
          onRowClick={(m) => navigate(`/entrainement/modeles/${encodeURIComponent(m.id)}`)}
          minWidth={980}
          empty={
            <EmptyState
              title="Aucun modèle"
              description="Créez un découpage, puis lancez un entraînement « rapide » pour vérifier que les données apprennent."
              actions={
                <Button variant="primary" icon={Plus} onClick={() => navigate('/entrainement/nouveau')}>
                  Nouvel entraînement
                </Button>
              }
            />
          }
        />
        {(models.data?.total ?? 0) > PAGE && (
          <div className={s.pagination}>
            <Pagination page={page} pageSize={PAGE} total={models.data!.total} onPageChange={setPage} />
          </div>
        )}
      </Panel>
    </div>
  );
}

function StatusCell({ m }: { m: ModelSummary }) {
  const job = useJob(m.job_active ? m.job_id : null);
  if (m.job_active && job.data) {
    return (
      <div className={s.nameCell} style={{ minWidth: 140 }}>
        <ModelStatusBadge status={m.status === 'created' ? 'created' : 'training'} />
        <ProgressBar value={job.data.progress} size="xs" />
      </div>
    );
  }
  return (
    <div className={s.nameCell}>
      <ModelStatusBadge status={m.status} />
      {m.status === 'trained' && m.duration_s != null && <span className="u-xs u-muted">{formatDuration(m.duration_s)}</span>}
    </div>
  );
}
