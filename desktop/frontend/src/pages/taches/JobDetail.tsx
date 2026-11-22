import { useState } from 'react';
import { Copy, RotateCw, X } from 'lucide-react';
import { useJob, useJobLog } from '@/api/hooks';
import { jobKindLabel } from '@/lib/domain';
import { durationBetween, formatDateTime, formatDuration } from '@/lib/format';
import { Callout, ErrorState, IconButton, KeyValue, LoadingState, Panel, Tabs, Toggle, useToast } from '@/components/ui';
import { JobProgress } from '@/components/JobProgress';
import { LogView } from '@/components/LogView';
import s from './taches.module.css';

type DetailTab = 'log' | 'params' | 'result';

function Json({ value }: { value: unknown }) {
  const text = value == null ? '' : JSON.stringify(value, null, 2);
  return <pre className="log-view" style={{ maxHeight: 360 }}>{text && text !== '{}' ? text : '—'}</pre>;
}

export function JobDetail({ id, onClose }: { id: string; onClose: () => void }) {
  const job = useJob(id);
  const [tab, setTab] = useState<DetailTab>('log');
  const [follow, setFollow] = useState(true);
  const log = useJobLog(id);
  const toast = useToast();

  if (job.isLoading) return <Panel><LoadingState compact /></Panel>;
  if (job.isError || !job.data)
    return (
      <Panel title="Tâche" actions={<IconButton icon={X} label="Fermer" size="sm" onClick={onClose} />}>
        <ErrorState error={job.error} onRetry={() => job.refetch()} compact />
      </Panel>
    );
  const j = job.data;

  const copyLog = async () => {
    try {
      await navigator.clipboard.writeText(log.text);
      toast.success('Journal copié');
    } catch {
      toast.error('Copie impossible', 'Le presse-papiers n’est pas accessible.');
    }
  };

  return (
    <Panel
      title={<span className="u-truncate">{j.title}</span>}
      description={jobKindLabel(j.kind)}
      actions={<IconButton icon={X} label="Fermer le détail" size="sm" onClick={onClose} />}
      className={s.detailPanel}
    >
      <div className="u-stack-4">
        <JobProgress jobId={id} hideTitle />
        {j.status === 'failed' && j.error && (
          <Callout tone="danger" title="Erreur">
            <span className="u-mono u-break">{j.error}</span>
          </Callout>
        )}
        <KeyValue
          items={[
            { label: 'Identifiant', value: j.id, mono: true },
            { label: 'Créée', value: formatDateTime(j.created_at) },
            { label: 'Démarrée', value: formatDateTime(j.started_at) },
            { label: 'Terminée', value: formatDateTime(j.ended_at) },
            { label: 'Durée', value: formatDuration(durationBetween(j.started_at, j.ended_at)) },
          ]}
        />
        <div>
          <div className={s.detailTabs}>
            <Tabs<DetailTab>
              aria-label="Détail de la tâche"
              value={tab}
              onChange={setTab}
              items={[
                { key: 'log', label: 'Journal' },
                { key: 'params', label: 'Paramètres' },
                { key: 'result', label: 'Résultat', disabled: !j.result },
              ]}
            />
            {tab === 'log' && (
              <div className={s.logTools}>
                <Toggle size="sm" checked={follow} onChange={setFollow} label={<span className="u-xs">Suivre</span>} />
                <IconButton icon={RotateCw} label="Relire le journal" size="sm" onClick={log.reload} />
                <IconButton icon={Copy} label="Copier le journal" size="sm" onClick={copyLog} disabled={!log.text} />
              </div>
            )}
          </div>
          {tab === 'log' &&
            (log.error ? (
              <ErrorState error={log.error} onRetry={log.reload} compact />
            ) : log.loading ? (
              <LoadingState compact label="Lecture du journal…" />
            ) : (
              <LogView text={log.text} follow={follow} height={380} className={s.log} />
            ))}
          {tab === 'params' && <Json value={j.params} />}
          {tab === 'result' && <Json value={j.result} />}
        </div>
      </div>
    </Panel>
  );
}
