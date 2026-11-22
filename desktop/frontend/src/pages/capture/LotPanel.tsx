/** Lot : formulaire, Démarrer / Arrêter, statistiques en direct, fin de lot et synchronisation. */
import { useEffect, useState } from 'react';
import { AlertTriangle, CircleStop, Play } from 'lucide-react';
import { errorMessage } from '@/api/client';
import { JobProgress } from '@/components/JobProgress';
import { Badge, Button, Callout, Field, NumberInput, Panel, Select, TextArea, TextInput, Toggle, useToast } from '@/components/ui';
import { LOT_TYPES, lotTypeLabel } from '@/lib/domain';
import { formatBytes, formatDuration, formatNumber, formatPercent } from '@/lib/format';
import { useInvalidateCapture, useOptions, useSaveOptions, useStartLot, useStopLot } from './api';
import type { CaptureStatus, LiveState, LotForm } from './types';
import s from './capture.module.css';

const MEMO_KEY = 'eggvision.capture.lotForm';
const MAX_DROP = 0.001; // critère : moins de 0,1 % d'images perdues

const STOP_REASONS: Record<string, string> = {
  user: 'arrêt demandé',
  disk_low: 'arrêt automatique : espace disque sous le seuil',
  write_error: 'arrêt sur erreur d’écriture',
  shutdown: 'arrêt du service',
  agent: 'arrêt côté agent',
};

function loadMemo(): Partial<LotForm> {
  try {
    return JSON.parse(localStorage.getItem(MEMO_KEY) || '{}') as Partial<LotForm>;
  } catch {
    return {};
  }
}

export function LotPanel({ status, live }: { status: CaptureStatus; live: LiveState | undefined }) {
  const active = !!live?.active;
  return active && live ? <LiveLot live={live} /> : <NewLot status={status} last={live?.last ?? status.last_lot} />;
}

// ---------------------------------------------------------------- Nouveau lot
function NewLot({ status, last }: { status: CaptureStatus; last: LiveState['last'] }) {
  const toast = useToast();
  const start = useStartLot();
  const invalidate = useInvalidateCapture();
  const memo = loadMemo();
  const [form, setForm] = useState<LotForm>({
    name: '',
    type: memo.type ?? 'viable',
    farm: memo.farm ?? '',
    operator: memo.operator ?? '',
    comment: '',
  });
  const set = <K extends keyof LotForm>(k: K, v: LotForm[K]) => setForm((f) => ({ ...f, [k]: v }));
  const ready = !!status.source?.ready;
  const canStart = status.connected && ready && form.name.trim().length > 0;

  const submit = async () => {
    try {
      const res = await start.mutateAsync({ ...form, name: form.name.trim() });
      localStorage.setItem(MEMO_KEY, JSON.stringify({ type: form.type, farm: form.farm, operator: form.operator }));
      toast.success('Capture démarrée', res.lot?.lot_id);
      setForm((f) => ({ ...f, name: '', comment: '' }));
    } catch (e) {
      toast.error('Démarrage impossible', errorMessage(e));
    }
  };

  return (
    <Panel title="Nouveau lot" description="Le lot démarre aussitôt ; les réglages caméra sont figés dans ses métadonnées.">
      <div className="u-stack-4">
        <Field label="Nom du lot" required>
          <TextInput value={form.name} onChange={(e) => set('name', e.target.value)} maxLength={200} placeholder="ex. Viables bâtiment 2 — matin" onKeyDown={(e) => e.key === 'Enter' && canStart && submit()} />
        </Field>
        <Field label="Type">
          <Select value={form.type} onChange={(v) => set('type', v)} options={LOT_TYPES.map((t) => ({ value: t.value, label: t.label }))} />
        </Field>
        <div className={s.fields2}>
          <Field label="Élevage">
            <TextInput value={form.farm} onChange={(e) => set('farm', e.target.value)} maxLength={200} />
          </Field>
          <Field label="Opérateur" hint="Vide = utilisateur du projet">
            <TextInput value={form.operator} onChange={(e) => set('operator', e.target.value)} maxLength={200} />
          </Field>
        </div>
        <Field label="Commentaire">
          <TextArea value={form.comment} onChange={(e) => set('comment', e.target.value)} rows={2} maxLength={5000} />
        </Field>
        <Button variant="primary" size="lg" icon={Play} block disabled={!canStart} loading={start.isPending} onClick={submit}>
          Démarrer la capture
        </Button>
        {!status.connected && <p className="u-xs u-muted">Connecter une source pour démarrer un lot.</p>}
        {status.connected && !ready && <p className="u-xs u-muted">Caméra en préparation…</p>}
        <AutoStopOptions />
        {last && <LastLot last={last} onSyncDone={invalidate} />}
      </div>
    </Panel>
  );
}

function AutoStopOptions({ disabled }: { disabled?: boolean }) {
  const toast = useToast();
  const options = useOptions();
  const save = useSaveOptions();
  if (!options.data) return null;
  const o = options.data;
  const update = (next: Partial<typeof o>) => save.mutate({ ...o, ...next }, { onError: (e) => toast.error(e) });
  return (
    <div className={s.optionsBox}>
      <Field label="Arrêt automatique sous" hint="Espace libre minimal sur le disque de destination (SSD du Pi ou disque du projet).">
        <NumberInput value={o.min_free_gb} onChange={(v) => v != null && update({ min_free_gb: v })} min={0} max={10000} step={0.5} unit="Go" disabled={disabled} />
      </Field>
      <Toggle checked={o.auto_sync} onChange={(v) => update({ auto_sync: v })} label="Synchroniser automatiquement" description="Lots de l’agent Pi : copie vérifiée vers le projet à la fin du lot." disabled={disabled} />
    </div>
  );
}

function LastLot({ last, onSyncDone }: { last: NonNullable<LiveState['last']>; onSyncDone: () => void }) {
  const st = last.stats;
  const dropRatio = st && st.frames_written != null ? (st.frames_dropped ?? 0) / Math.max(1, (st.frames_written ?? 0) + (st.frames_dropped ?? 0)) : 0;
  return (
    <div className={s.lastLot}>
      <div className="u-row">
        <span className="u-small u-strong">Dernier lot</span>
        <span className="u-spacer" />
        <Badge tone={last.status === 'termine' ? 'success' : last.status === 'interrompu' ? 'warning' : 'neutral'}>{last.status ?? '—'}</Badge>
      </div>
      <span className="u-mono u-xs u-truncate">{last.lot_id}</span>
      {st && (
        <span className="u-xs u-muted">
          {formatNumber(st.frames_written ?? 0)} images · {formatNumber(st.frames_dropped ?? 0)} perdues ({formatPercent(dropRatio, 3)}) ·{' '}
          {formatDuration(st.duration_s ?? 0)} · {formatNumber(st.measured_fps ?? 0, 2)} im/s
        </span>
      )}
      {last.stop_reason && last.stop_reason !== 'user' && (
        <Callout tone="warning">{STOP_REASONS[last.stop_reason] ?? last.stop_reason}</Callout>
      )}
      {last.sync_job_id && (
        <div className="u-stack-4">
          <span className="u-xs u-muted">Synchronisation vers le projet</span>
          <JobProgress jobId={last.sync_job_id} compact hideTitle onDone={onSyncDone} />
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------- Lot en cours
function LiveLot({ live }: { live: LiveState }) {
  const toast = useToast();
  const stop = useStopLot();
  const st = live.stats;
  const lot = live.lot!;
  const [tick, setTick] = useState(0);
  useEffect(() => {
    const t = setInterval(() => setTick((n) => n + 1), 1000);
    return () => clearInterval(t);
  }, []);
  void tick;

  const dropRatio = st?.drop_ratio ?? 0;
  const free = st?.disk_free_bytes ?? null;
  const minFree = st?.min_free_bytes ?? 0;
  const diskTone = free == null ? 'neutral' : free < minFree * 1.5 ? 'danger' : free < minFree * 3 ? 'warning' : 'success';
  const onStop = async () => {
    try {
      const res = await stop.mutateAsync();
      const m = res.last;
      toast.success('Lot terminé', m ? `${formatNumber(m.stats?.frames_written ?? 0)} images` : undefined);
    } catch (e) {
      toast.error('Arrêt impossible', errorMessage(e));
    }
  };

  return (
    <Panel
      title={
        <span className="u-row">
          <span className={s.recDot} aria-hidden /> Capture en cours
        </span>
      }
      description={`${lot.name} — ${lotTypeLabel(lot.type)}`}
    >
      <div className="u-stack-4">
        <Button variant="danger" size="lg" icon={CircleStop} block loading={stop.isPending} onClick={onStop}>
          Arrêter le lot
        </Button>
        {live.error && <Callout tone="warning" title="Agent momentanément injoignable">{live.error} — la capture continue sur le Pi.</Callout>}
        <div className={s.liveGrid}>
          <LiveStat label="Durée" value={formatDuration(st?.elapsed_s ?? 0)} />
          <LiveStat label="Images" value={formatNumber(st?.frames_written ?? 0)} hint={st?.frames_queued ? `${st.frames_queued} en file` : undefined} />
          <LiveStat
            label="Cadence réelle"
            value={`${formatNumber(st?.instant_fps ?? 0, 1)} im/s`}
            hint={`moyenne ${formatNumber(st?.measured_fps ?? 0, 2)}`}
          />
          <LiveStat
            label="Images perdues"
            value={formatNumber(st?.frames_dropped ?? 0)}
            hint={formatPercent(dropRatio, 3)}
            tone={dropRatio > MAX_DROP ? 'danger' : (st?.frames_dropped ?? 0) > 0 ? 'warning' : 'success'}
          />
          <LiveStat label="Espace libre" value={formatBytes(free)} hint={`arrêt sous ${formatNumber(minFree / 1e9, 1)} Go`} tone={diskTone} />
          <LiveStat label="Volume écrit" value={formatBytes(st?.bytes_written ?? 0)} hint={`file ${formatPercent(st?.queue_fill ?? 0, 0)} (max ${formatPercent(st?.queue_max_fill ?? 0, 0)})`} />
        </div>
        {dropRatio > MAX_DROP && (
          <Callout tone="danger" icon={AlertTriangle} title="Pertes au-delà de 0,1 %">
            Le disque ou le processeur ne suit pas la cadence. Vérifier le SSD, la température du Pi ou baisser la cadence.
          </Callout>
        )}
        {(st?.write_errors ?? 0) > 0 && <Callout tone="danger">Erreurs d’écriture : {st?.error}</Callout>}
        <p className="u-xs u-muted u-mono u-truncate">{lot.lot_id}</p>
      </div>
    </Panel>
  );
}

function LiveStat({ label, value, hint, tone }: { label: string; value: string; hint?: string; tone?: string }) {
  return (
    <div className={s.liveStat} data-tone={tone}>
      <span className="u-xs u-muted">{label}</span>
      <span className={s.liveValue}>{value}</span>
      {hint && <span className="u-xs u-muted">{hint}</span>}
    </div>
  );
}
