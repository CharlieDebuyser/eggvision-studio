import { useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { ScrollText, X } from 'lucide-react';
import { useCancelJob, useJob, isJobActive } from '@/api/hooks';
import type { Job } from '@/api/types';
import { errorMessage } from '@/api/client';
import { cx } from '@/lib/cx';
import { durationBetween, formatDuration } from '@/lib/format';
import { ProgressBar, StatusBadge, Button, useToast, ConfirmDialog, Callout } from '@/components/ui';

export interface JobProgressProps<R = Record<string, unknown>> {
  jobId: string | null | undefined;
  /** Appelé une fois quand la tâche se termine (done, failed ou cancelled). */
  onDone?: (job: Job<Record<string, unknown>, R>) => void;
  /** Masquer le titre de la tâche. */
  hideTitle?: boolean;
  /** Masquer le bouton Annuler. */
  hideCancel?: boolean;
  /** Demander confirmation avant d'annuler (défaut true). */
  confirmCancel?: boolean;
  compact?: boolean;
  className?: string;
}

/**
 * Suivi d'une tâche de fond : barre, message, statut, durée, Annuler, lien vers le journal.
 *   <JobProgress jobId={id} onDone={(job) => job.status === 'done' && open(job.result!.path)} />
 */
export function JobProgress<R = Record<string, unknown>>({
  jobId,
  onDone,
  hideTitle,
  hideCancel,
  confirmCancel = true,
  compact,
  className,
}: JobProgressProps<R>) {
  const { data: job, error } = useJob<Record<string, unknown>, R>(jobId);
  const cancel = useCancelJob();
  const toast = useToast();
  const [asking, setAsking] = useState(false);
  const notified = useRef<string | null>(null);
  const onDoneRef = useRef(onDone);
  onDoneRef.current = onDone;
  const [, tick] = useState(0);

  const active = isJobActive(job?.status);

  useEffect(() => {
    if (!job || active || notified.current === job.id) return;
    notified.current = job.id;
    onDoneRef.current?.(job);
  }, [job, active]);

  // Rafraîchit la durée affichée chaque seconde pendant l'exécution.
  useEffect(() => {
    if (!active) return;
    const t = setInterval(() => tick((n) => n + 1), 1000);
    return () => clearInterval(t);
  }, [active]);

  if (!jobId) return null;
  if (error && !job) return <Callout tone="danger" title="Tâche introuvable">{errorMessage(error)}</Callout>;

  const doCancel = async () => {
    try {
      await cancel.mutateAsync(jobId);
      toast.info('Annulation demandée');
    } catch (e) {
      toast.error(e);
    } finally {
      setAsking(false);
    }
  };

  const tone = job?.status === 'failed' ? 'danger' : job?.status === 'done' ? 'success' : job?.status === 'cancelled' ? 'neutral' : 'accent';
  const elapsed = durationBetween(job?.started_at, job?.ended_at);

  return (
    <div className={cx('job-progress', compact && 'job-progress--compact', className)}>
      <div className="job-progress__head">
        {!hideTitle && <span className="job-progress__title u-truncate">{job?.title ?? 'Tâche'}</span>}
        {job && <StatusBadge status={job.status} />}
        <span className="u-spacer" />
        {elapsed !== null && <span className="u-xs u-subtle u-num">{formatDuration(elapsed)}</span>}
      </div>
      <ProgressBar value={job ? (job.status === 'queued' ? null : job.progress) : null} tone={tone} showValue size={compact ? 'xs' : 'sm'} />
      <div className="job-progress__foot">
        <span className={cx('job-progress__msg u-truncate', job?.status === 'failed' && 'tone-fg--danger')} title={job?.error ?? job?.message}>
          {job?.status === 'failed' ? (job.error ?? 'Échec') : job?.message || (job?.status === 'queued' ? 'En attente de démarrage…' : '')}
        </span>
        <Link to={`/taches?job=${encodeURIComponent(jobId)}`} className="job-progress__link">
          <ScrollText size={13} aria-hidden /> Voir le journal
        </Link>
        {!hideCancel && active && (
          <Button size="sm" variant="ghost" icon={X} loading={cancel.isPending} onClick={() => (confirmCancel ? setAsking(true) : void doCancel())}>
            Annuler
          </Button>
        )}
      </div>
      <ConfirmDialog
        open={asking}
        title="Annuler la tâche ?"
        message={`« ${job?.title ?? 'Tâche'} » sera interrompue. Le travail déjà effectué peut être perdu.`}
        confirmLabel="Annuler la tâche"
        cancelLabel="Continuer"
        variant="danger"
        onConfirm={doCancel}
        onCancel={() => setAsking(false)}
      />
    </div>
  );
}
