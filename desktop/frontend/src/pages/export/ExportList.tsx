/** Liste des modèles exportables (classifieurs entraînés) avec l'état de leur paquet. */
import { useNavigate } from 'react-router-dom';
import { Badge, DataTable, EmptyState, Button, Panel, type Column } from '@/components/ui';
import { formatDateTime, formatNumber, formatPercent } from '@/lib/format';
import { useModels, useOptions } from '@/pages/entrainement/api';
import { Int8Badge, archLabel } from '@/pages/entrainement/common';
import type { ModelSummary } from '@/pages/entrainement/types';
import s from './export.module.css';

export function ExportList() {
  const navigate = useNavigate();
  const models = useModels(0, 200);
  const options = useOptions();
  const archs = options.data?.architectures;
  const rows = models.data?.items.filter((m) => m.kind === 'classifier' && m.status === 'trained');
  const columns: Column<ModelSummary>[] = [
    {
      key: 'name',
      header: 'Modèle',
      width: '28%',
      render: (m) => (
        <div className={s.nameCell}>
          <span className="u-strong u-truncate">{m.name}</span>
          <span className="u-xs u-muted">
            {archLabel(archs, m.arch)} · {m.input_size} px · {m.n_views} vues · n° {m.number ?? '—'}
          </span>
        </div>
      ),
      sortValue: (m) => m.number ?? 0,
    },
    {
      key: 'int8',
      header: 'Quantification',
      render: (m) => {
        const a = archs?.find((x) => x.id === m.arch);
        return a ? <Int8Badge friendly={a.int8_friendly} /> : null;
      },
    },
    { key: 'auc', header: 'AUC test', align: 'right', render: (m) => formatNumber(m.metrics.test?.auc_roc, 3), sortValue: (m) => m.metrics.test?.auc_roc ?? -1 },
    { key: 'rec', header: 'Rappel', align: 'right', render: (m) => formatPercent(m.metrics.test?.recall), sortValue: (m) => m.metrics.test?.recall ?? -1 },
    { key: 'fr', header: 'Faux rejets', align: 'right', render: (m) => formatPercent(m.metrics.test?.false_reject_rate, 2) },
    { key: 'thr', header: 'Seuil', align: 'right', render: (m) => formatNumber(m.threshold, 3) },
    {
      key: 'exp',
      header: 'Paquet',
      render: (m) =>
        !m.exported ? (
          <span className="u-subtle">non exporté</span>
        ) : m.export_outdated ? (
          <Badge tone="warning">obsolète (seuil modifié)</Badge>
        ) : (
          <Badge tone="success">à jour</Badge>
        ),
      sortValue: (m) => (m.exported ? (m.export_outdated ? 1 : 2) : 0),
    },
    { key: 'date', header: 'Créé', render: (m) => formatDateTime(m.created_at), sortValue: (m) => m.created_at ?? '' },
  ];
  return (
    <Panel flush>
      <DataTable
        columns={columns}
        rows={rows}
        rowKey={(m) => m.id}
        loading={models.isLoading}
        error={models.error}
        onRetry={() => models.refetch()}
        onRowClick={(m) => navigate(`/export/${encodeURIComponent(m.id)}`)}
        minWidth={900}
        empty={
          <EmptyState
            title="Aucun modèle exportable"
            description="Seuls les classifieurs entraînés s'exportent (le modèle de référence par anomalies n'est pas exportable)."
            actions={<Button onClick={() => navigate('/entrainement/nouveau')}>Entraîner un modèle</Button>}
          />
        }
      />
    </Panel>
  );
}
