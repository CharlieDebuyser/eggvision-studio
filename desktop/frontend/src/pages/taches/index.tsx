import { useMemo, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { ListChecks, RotateCw, X } from 'lucide-react';
import { useCancelJob, useJobs, isJobActive } from '@/api/hooks';
import { JOB_KINDS, type Job, type JobStatus } from '@/api/types';
import { jobKindLabel } from '@/lib/domain';
import { durationBetween, formatDateTime, formatDuration, formatRelative } from '@/lib/format';
import {
  ConfirmDialog,
  DataTable,
  EmptyState,
  IconButton,
  Page,
  PageHeader,
  Panel,
  ProgressBar,
  SegmentedControl,
  Select,
  StatusBadge,
  useToast,
  type Column,
} from '@/components/ui';
import { JobDetail } from './JobDetail';
import s from './taches.module.css';

type StatusFilter = 'all' | 'active' | 'done' | 'failed' | 'cancelled';

const STATUS_GROUPS: Record<StatusFilter, (st: JobStatus) => boolean> = {
  all: () => true,
  active: (st) => isJobActive(st),
  done: (st) => st === 'done',
  failed: (st) => st === 'failed',
  cancelled: (st) => st === 'cancelled',
};

export default function TachesPage() {
  const [params, setParams] = useSearchParams();
  const selectedId = params.get('job');
  const [status, setStatus] = useState<StatusFilter>('all');
  const [kind, setKind] = useState('');
  const jobs = useJobs({ kind: kind || undefined, limit: 200 });
  const cancel = useCancelJob();
  const toast = useToast();
  const [toCancel, setToCancel] = useState<Job | null>(null);

  const select = (id: string | null) => {
    const next = new URLSearchParams(params);
    if (id) next.set('job', id);
    else next.delete('job');
    setParams(next, { replace: true });
  };

  const counts = useMemo(() => {
    const c: Record<StatusFilter, number> = { all: 0, active: 0, done: 0, failed: 0, cancelled: 0 };
    for (const j of jobs.data ?? []) {
      (Object.keys(STATUS_GROUPS) as StatusFilter[]).forEach((k) => {
        if (STATUS_GROUPS[k](j.status)) c[k]++;
      });
    }
    return c;
  }, [jobs.data]);

  const rows = useMemo(() => jobs.data?.filter((j) => STATUS_GROUPS[status](j.status)), [jobs.data, status]);

  const kindOptions = useMemo(() => {
    const set = new Set<string>(JOB_KINDS);
    jobs.data?.forEach((j) => set.add(j.kind));
    return [...set].map((k) => ({ value: k, label: jobKindLabel(k) }));
  }, [jobs.data]);

  const doCancel = async (job: Job) => {
    try {
      await cancel.mutateAsync(job.id);
      toast.info('Annulation demandée', job.title);
    } catch (e) {
      toast.error(e);
    } finally {
      setToCancel(null);
    }
  };

  const columns: Column<Job>[] = [
    {
      key: 'title',
      header: 'Tâche',
      sortValue: (j) => j.title,
      render: (j) => (
        <div className={s.titleCell}>
          <span className="u-truncate u-strong">{j.title}</span>
          <span className="u-xs u-subtle u-truncate">{jobKindLabel(j.kind)}</span>
        </div>
      ),
    },
    { key: 'status', header: 'Statut', width: 104, sortValue: (j) => j.status, render: (j) => <StatusBadge status={j.status} /> },
    {
      key: 'progress',
      header: 'Progression',
      width: 124,
      sortValue: (j) => j.progress,
      render: (j) =>
        j.status === 'done' ? (
          <span className="u-xs u-subtle">—</span>
        ) : (
          <ProgressBar value={j.status === 'queued' ? null : j.progress} tone={j.status === 'failed' ? 'danger' : j.status === 'cancelled' ? 'neutral' : 'accent'} showValue size="xs" />
        ),
    },
    {
      key: 'created',
      header: 'Créée',
      width: 104,
      sortValue: (j) => Date.parse(j.created_at),
      render: (j) => (
        <span className="u-small u-muted" title={formatDateTime(j.created_at)}>
          {formatRelative(j.created_at)}
        </span>
      ),
    },
    {
      key: 'duration',
      header: 'Durée',
      width: 78,
      align: 'right',
      sortValue: (j) => durationBetween(j.started_at, j.ended_at),
      render: (j) => <span className="u-num u-small">{formatDuration(durationBetween(j.started_at, j.ended_at))}</span>,
    },
    {
      key: 'actions',
      header: <span className="u-sr-only">Actions</span>,
      width: 40,
      align: 'right',
      render: (j) =>
        isJobActive(j.status) ? (
          <IconButton
            icon={X}
            label="Annuler la tâche"
            size="sm"
            onClick={(e) => {
              e.stopPropagation();
              setToCancel(j);
            }}
          />
        ) : null,
    },
  ];

  return (
    <Page wide>
      <PageHeader
        title="Tâches"
        subtitle="Traitements de fond : progression en direct, journal et annulation."
        actions={<IconButton icon={RotateCw} label="Rafraîchir" variant="secondary" loading={jobs.isFetching && !jobs.isLoading} onClick={() => jobs.refetch()} />}
      />

      <div className={s.layout}>
        <Panel flush className={s.listPanel}>
          <div className={s.toolbar}>
            <SegmentedControl<StatusFilter>
              size="sm"
              aria-label="Filtrer par statut"
              value={status}
              onChange={setStatus}
              options={[
                { value: 'all', label: `Toutes · ${counts.all}` },
                { value: 'active', label: `En cours · ${counts.active}` },
                { value: 'done', label: `Terminées · ${counts.done}` },
                { value: 'failed', label: `Échecs · ${counts.failed}` },
                { value: 'cancelled', label: `Annulées · ${counts.cancelled}` },
              ]}
            />
            <span className="u-spacer" />
            <Select size="sm" aria-label="Filtrer par type" value={kind} onChange={setKind} placeholder="Tous les types" options={kindOptions} className={s.kindSelect} />
          </div>
          <DataTable
            columns={columns}
            rows={rows}
            rowKey={(j) => j.id}
            loading={jobs.isLoading}
            error={jobs.error}
            onRetry={() => jobs.refetch()}
            onRowClick={(j) => select(j.id)}
            selectedKey={selectedId}
            stickyHeader
            minWidth={560}
            className={s.table}
            empty={
              <EmptyState
                icon={ListChecks}
                title={status === 'all' && !kind ? 'Aucune tâche' : 'Aucune tâche ne correspond aux filtres'}
                description={status === 'all' && !kind ? 'Les traitements lancés depuis les modules apparaîtront ici.' : undefined}
                compact
              />
            }
          />
        </Panel>

        <div className={s.detail}>
          {selectedId ? (
            <JobDetail key={selectedId} id={selectedId} onClose={() => select(null)} />
          ) : (
            <Panel>
              <EmptyState icon={ListChecks} title="Aucune tâche sélectionnée" description="Sélectionnez une tâche pour voir son détail et son journal." compact />
            </Panel>
          )}
        </div>
      </div>

      <ConfirmDialog
        open={!!toCancel}
        title="Annuler la tâche ?"
        message={toCancel ? `« ${toCancel.title} » sera interrompue.` : undefined}
        confirmLabel="Annuler la tâche"
        cancelLabel="Continuer"
        variant="danger"
        onConfirm={() => (toCancel ? doCancel(toCancel) : undefined)}
        onCancel={() => setToCancel(null)}
      />
    </Page>
  );
}
