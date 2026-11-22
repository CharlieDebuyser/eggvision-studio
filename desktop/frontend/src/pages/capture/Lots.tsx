/** Onglet « Lots et synchronisation » : lots du projet, lots restés sur l'agent, synchronisations. */
import { useState } from 'react';
import { CloudDownload, RefreshCw, Trash2 } from 'lucide-react';
import { errorMessage } from '@/api/client';
import { JobProgress } from '@/components/JobProgress';
import {
  Badge,
  Button,
  ConfirmDialog,
  DataTable,
  EmptyState,
  IconButton,
  Pagination,
  Panel,
  Tooltip,
  useToast,
  type Column,
} from '@/components/ui';
import { lotTypeLabel } from '@/lib/domain';
import { formatBytes, formatDateTime, formatDuration, formatNumber, formatPercent } from '@/lib/format';
import { useAgentLots, useCaptureStatus, useDeleteAgentLot, useInvalidateCapture, useProjectLots, useSyncJobs, useSyncLot } from './api';
import { KIND_LABEL } from './Connection';
import type { AgentLot, ProjectLot, SourceKind } from './types';
import s from './capture.module.css';

const PAGE = 25;

const SYNC_META: Record<string, { label: string; tone: 'neutral' | 'success' | 'warning' | 'danger' | 'info' }> = {
  done: { label: 'Synchronisé', tone: 'success' },
  pending: { label: 'À synchroniser', tone: 'warning' },
  syncing: { label: 'En cours', tone: 'info' },
  error: { label: 'Erreur', tone: 'danger' },
  none: { label: '—', tone: 'neutral' },
};

const STATUS_META: Record<string, { label: string; tone: 'neutral' | 'success' | 'warning' | 'danger' | 'info' }> = {
  en_cours: { label: 'En cours', tone: 'info' },
  termine: { label: 'Terminé', tone: 'success' },
  interrompu: { label: 'Interrompu', tone: 'warning' },
};

export function SyncBadge({ status }: { status: string | null | undefined }) {
  const m = SYNC_META[status ?? 'none'] ?? { label: status ?? '—', tone: 'neutral' as const };
  return <Badge tone={m.tone} dot>{m.label}</Badge>;
}

function LotStatusBadge({ status }: { status: string }) {
  const m = STATUS_META[status] ?? { label: status, tone: 'neutral' as const };
  return <Badge tone={m.tone}>{m.label}</Badge>;
}

const dropText = (written: number, dropped: number) => {
  const total = written + dropped;
  return `${formatNumber(dropped)}${total ? ` (${formatPercent(dropped / total, 2)})` : ''}`;
};

export function LotsView() {
  const status = useCaptureStatus();
  const isPi = status.data?.kind === 'pi';
  return (
    <div className="u-stack-6">
      <SyncJobs />
      {isPi ? (
        <AgentLots />
      ) : (
        <Panel title="Lots restés sur l’agent Pi">
          <EmptyState compact title="Aucun agent Pi connecté" description="Connecter l’agent (onglet Acquisition) pour lister et synchroniser les lots de son SSD." />
        </Panel>
      )}
      <ProjectLots canSync={isPi} />
    </div>
  );
}

// ---------------------------------------------------------------- Synchronisations actives
function SyncJobs() {
  const jobs = useSyncJobs();
  const invalidate = useInvalidateCapture();
  const active = (jobs.data ?? []).filter((j) => j.status === 'queued' || j.status === 'running');
  if (!active.length) return null;
  return (
    <Panel title="Synchronisations en cours" description="Copie vérifiée (SHA-256) des lots de l’agent vers le projet ; reprise automatique après coupure.">
      <div className="u-stack-4">
        {active.map((j) => (
          <JobProgress key={j.id} jobId={j.id} onDone={invalidate} />
        ))}
      </div>
    </Panel>
  );
}

// ---------------------------------------------------------------- Lots de l'agent
function AgentLots() {
  const toast = useToast();
  const lots = useAgentLots(true);
  const sync = useSyncLot();
  const del = useDeleteAgentLot();
  const [toDelete, setToDelete] = useState<AgentLot | null>(null);
  const items = lots.data?.items ?? [];

  const doSync = async (lotId: string) => {
    try {
      await sync.mutateAsync(lotId);
      toast.info('Synchronisation lancée', lotId);
    } catch (e) {
      toast.error('Synchronisation impossible', errorMessage(e));
    }
  };
  const syncAll = async () => {
    for (const l of items.filter((x) => !x.synced && !x.capturing && !x.sync_job_id)) await doSync(l.lot_id);
  };

  const columns: Column<AgentLot>[] = [
    {
      key: 'name',
      header: 'Lot',
      render: (l) => (
        <div className={s.nameCell}>
          <span className="u-strong u-truncate">{l.name ?? l.lot_id}</span>
          <span className="u-mono u-xs u-muted u-truncate">{l.lot_id}</span>
        </div>
      ),
      sortValue: (l) => l.name ?? l.lot_id,
    },
    { key: 'type', header: 'Type', render: (l) => (l.type ? lotTypeLabel(l.type) : '—') },
    { key: 'created', header: 'Créé', render: (l) => formatDateTime(l.created_at ?? null), sortValue: (l) => l.created_at ?? '' },
    { key: 'status', header: 'Statut', render: (l) => (l.capturing ? <Badge tone="info">Capture en cours</Badge> : <LotStatusBadge status={l.status} />) },
    { key: 'frames', header: 'Images', align: 'right', render: (l) => formatNumber(l.stats?.frames_written ?? 0), sortValue: (l) => l.stats?.frames_written ?? 0 },
    { key: 'drop', header: 'Perdues', align: 'right', render: (l) => dropText(l.stats?.frames_written ?? 0, l.stats?.frames_dropped ?? 0) },
    { key: 'size', header: 'Volume', align: 'right', render: (l) => formatBytes(l.stats?.bytes_written ?? 0), sortValue: (l) => l.stats?.bytes_written ?? 0 },
    { key: 'sync', header: 'Projet', render: (l) => (l.sync_job_id ? <SyncBadge status="syncing" /> : l.in_project ? <SyncBadge status={l.sync_status} /> : <Badge tone="warning" dot>Absent du projet</Badge>) },
    {
      key: 'actions',
      header: '',
      align: 'right',
      render: (l) => (
        <span className={s.rowActions}>
          {!l.synced && !l.capturing && (
            <Button size="sm" icon={CloudDownload} disabled={!!l.sync_job_id} onClick={() => doSync(l.lot_id)}>
              Synchroniser
            </Button>
          )}
          <Tooltip content={l.synced ? 'Libérer l’espace sur le SSD du Pi' : 'Lot non synchronisé : suppression forcée (perte définitive)'}>
            <IconButton size="sm" variant="ghost" icon={Trash2} label="Supprimer de l’agent" disabled={l.capturing} onClick={() => setToDelete(l)} />
          </Tooltip>
        </span>
      ),
    },
  ];

  return (
    <Panel
      title="Lots restés sur l’agent Pi"
      description={lots.data ? `${lots.data.agent_url} — ${lots.data.unsynced} lot(s) non synchronisé(s)` : undefined}
      actions={
        <span className="u-row">
          <IconButton size="sm" variant="ghost" icon={RefreshCw} label="Actualiser" onClick={() => lots.refetch()} loading={lots.isFetching} />
          <Button size="sm" icon={CloudDownload} disabled={!lots.data?.unsynced} onClick={syncAll}>
            Tout synchroniser
          </Button>
        </span>
      }
      flush
    >
      <DataTable
        columns={columns}
        rows={items}
        rowKey={(l) => l.lot_id}
        loading={lots.isLoading}
        error={lots.error}
        empty={<EmptyState compact title="Aucun lot sur l’agent" />}
        dense
        minWidth={900}
      />
      <ConfirmDialog
        open={!!toDelete}
        title="Supprimer le lot de l’agent ?"
        message={
          toDelete?.synced
            ? `« ${toDelete?.name ?? toDelete?.lot_id} » est synchronisé dans le projet : seule la copie du SSD du Pi sera supprimée.`
            : `« ${toDelete?.name ?? toDelete?.lot_id} » n’est PAS synchronisé dans le projet : ses images seront définitivement perdues.`
        }
        confirmLabel={toDelete?.synced ? 'Supprimer' : 'Supprimer définitivement'}
        variant="danger"
        onCancel={() => setToDelete(null)}
        onConfirm={async () => {
          if (!toDelete) return;
          try {
            await del.mutateAsync({ lotId: toDelete.lot_id, force: !toDelete.synced });
            toast.info('Lot supprimé de l’agent');
          } catch (e) {
            toast.error(e);
          } finally {
            setToDelete(null);
          }
        }}
      />
    </Panel>
  );
}

// ---------------------------------------------------------------- Lots du projet
function ProjectLots({ canSync }: { canSync: boolean }) {
  const toast = useToast();
  const [page, setPage] = useState(0);
  const lots = useProjectLots(page, PAGE);
  const sync = useSyncLot();
  const columns: Column<ProjectLot>[] = [
    {
      key: 'name',
      header: 'Lot',
      render: (l) => (
        <div className={s.nameCell}>
          <span className="u-strong u-truncate">{l.name}</span>
          <span className="u-mono u-xs u-muted u-truncate">{l.id}</span>
        </div>
      ),
    },
    { key: 'type', header: 'Type', render: (l) => lotTypeLabel(l.type) },
    { key: 'farm', header: 'Élevage', render: (l) => l.farm || '—', truncate: true },
    { key: 'created', header: 'Date', render: (l) => formatDateTime(l.created_at) },
    { key: 'source', header: 'Source', render: (l) => KIND_LABEL[l.source_kind as SourceKind] ?? (l.source_kind || '—') },
    {
      key: 'status',
      header: 'Statut',
      render: (l) => (
        <span className="u-row">
          <LotStatusBadge status={l.status} />
          {l.stop_reason === 'disk_low' && (
            <Tooltip content="Arrêt automatique : espace disque sous le seuil">
              <Badge tone="warning">disque</Badge>
            </Tooltip>
          )}
        </span>
      ),
    },
    { key: 'frames', header: 'Images', align: 'right', render: (l) => formatNumber(l.frame_count) },
    { key: 'drop', header: 'Perdues', align: 'right', render: (l) => dropText(l.frame_count, l.dropped_frames) },
    { key: 'duration', header: 'Durée', align: 'right', render: (l) => formatDuration(l.duration_s) },
    {
      key: 'sync',
      header: 'Synchro',
      render: (l) =>
        l.source_kind === 'pi' ? (
          <span className="u-row">
            <SyncBadge status={l.sync_status} />
            {canSync && l.sync_status !== 'done' && l.sync_status !== 'syncing' && l.status !== 'en_cours' && (
              <IconButton
                size="sm"
                variant="ghost"
                icon={CloudDownload}
                label="Synchroniser maintenant"
                onClick={() => sync.mutate(l.id, { onError: (e) => toast.error('Synchronisation impossible', errorMessage(e)) })}
              />
            )}
          </span>
        ) : (
          <span className="u-xs u-muted">local</span>
        ),
    },
  ];
  return (
    <Panel title="Lots du projet" description="Lots capturés ou synchronisés dans le projet, du plus récent au plus ancien." flush>
      <DataTable
        columns={columns}
        rows={lots.data?.items}
        rowKey={(l) => l.id}
        loading={lots.isLoading}
        error={lots.error}
        empty={<EmptyState compact title="Aucun lot" description="Démarrer une capture depuis l’onglet Acquisition." />}
        dense
        minWidth={1000}
      />
      {(lots.data?.total ?? 0) > PAGE && (
        <div className={s.pagination}>
          <Pagination page={page} pageSize={PAGE} total={lots.data!.total} onPageChange={setPage} />
        </div>
      )}
    </Panel>
  );
}
