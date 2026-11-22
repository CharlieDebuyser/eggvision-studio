/**
 * Traitement par lot : tableau des lots (statut, images, pistes, % exploitables), lancement sur un
 * ou plusieurs lots, progression, relance avec confirmation si des labels ont été relus.
 */
import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useQueryClient } from '@tanstack/react-query';
import { AlertTriangle, Play, RotateCw, ScanSearch, Search } from 'lucide-react';
import { JobProgress } from '@/components/JobProgress';
import { queryKeys, useJob } from '@/api/hooks';
import {
  Badge,
  Button,
  Callout,
  Checkbox,
  ConfirmDialog,
  DataTable,
  EmptyState,
  Pagination,
  Panel,
  TextInput,
  Tooltip,
  useToast,
  type Column,
} from '@/components/ui';
import { formatNumber, formatPercent } from '@/lib/format';
import { lotTypeLabel } from '@/lib/domain';
import { detKeys, runCheck, useDebounced, useDetLots, useRun } from './api';
import type { DetLot, DetectionStatus, RunStats } from './types';
import s from './detection.module.css';

const PAGE = 25;

const STATUS: Record<DetectionStatus, { label: string; tone: 'neutral' | 'info' | 'success' | 'danger' }> = {
  none: { label: 'À traiter', tone: 'neutral' },
  running: { label: 'En cours', tone: 'info' },
  done: { label: 'Traité', tone: 'success' },
  error: { label: 'Erreur', tone: 'danger' },
};

export function Processing() {
  const navigate = useNavigate();
  const toast = useToast();
  const qc = useQueryClient();
  const [page, setPage] = useState(0);
  const [q, setQ] = useState('');
  const query = useDebounced(q, 300);
  const [picked, setPicked] = useState<string[]>([]);
  const [jobId, setJobId] = useState<string | null>(null);
  const [confirm, setConfirm] = useState<{ lots: { lot_id: string; reviewed: number }[]; ids: string[] } | null>(null);
  const lots = useDetLots(page, PAGE, query, true);
  const run = useRun();
  const items = lots.data?.items ?? [];
  const minViews = lots.data?.min_views ?? 8;

  // Reprend le suivi d'un traitement déjà en cours (retour sur l'écran).
  useEffect(() => {
    if (jobId) return;
    const active = items.find((l) => l.job_id);
    if (active?.job_id) setJobId(active.job_id);
  }, [items, jobId]);

  const toggle = (id: string) => setPicked((p) => (p.includes(id) ? p.filter((x) => x !== id) : [...p, id]));
  const selectable = items.filter((l) => l.frame_count > 0 && !l.job_id);
  const allPicked = selectable.length > 0 && selectable.every((l) => picked.includes(l.id));

  const launch = async (ids: string[], force = false) => {
    try {
      if (!force) {
        const check = await runCheck(ids);
        if (check.needs_force) {
          setConfirm({ lots: check.lots, ids });
          return;
        }
      }
      const job = await run.mutateAsync({ lot_ids: ids, force });
      setJobId(job.id);
      setPicked([]);
      toast.info('Traitement lancé', `${formatNumber(ids.length)} lot(s) : détection, suivi et extraction des vues.`);
    } catch (err) {
      toast.error(err);
    }
  };

  const columns: Column<DetLot>[] = [
    {
      key: 'pick',
      header: (
        <Checkbox
          checked={allPicked}
          indeterminate={!allPicked && picked.length > 0}
          onChange={(v) => setPicked(v ? selectable.map((l) => l.id) : [])}
          aria-label="Tout sélectionner"
        />
      ),
      width: 40,
      render: (r) => (
        <span onClick={(e) => e.stopPropagation()}>
          <Checkbox checked={picked.includes(r.id)} onChange={() => toggle(r.id)} disabled={r.frame_count === 0 || !!r.job_id} aria-label={`Sélectionner ${r.name}`} />
        </span>
      ),
    },
    {
      key: 'name',
      header: 'Lot',
      truncate: true,
      render: (r) => (
        <div className={s.nameCell}>
          <span className="u-strong u-truncate">{r.name}</span>
          <span className="u-subtle u-xs u-truncate">
            {lotTypeLabel(r.type)} · {r.farm || '—'} · {r.session_date || '—'}
          </span>
        </div>
      ),
      sortValue: (r) => r.name,
    },
    {
      key: 'status',
      header: 'Détection',
      width: 130,
      render: (r) => {
        const st = STATUS[r.job_id ? 'running' : r.detection_status] ?? STATUS.none;
        const badge = (
          <Badge tone={st.tone} dot>
            {st.label}
          </Badge>
        );
        return r.error ? <Tooltip content={r.error}>{badge}</Tooltip> : badge;
      },
    },
    { key: 'frames', header: 'Images', align: 'right', width: 90, render: (r) => formatNumber(r.frame_count), sortValue: (r) => r.frame_count },
    {
      key: 'tracks',
      header: 'Pistes',
      align: 'right',
      width: 120,
      render: (r) =>
        r.detection_status === 'done' || r.track_count ? (
          <span className="u-num">
            {formatNumber(r.track_count)}
            {r.tracks_doubtful > 0 && <span className="u-subtle u-xs"> · {formatNumber(r.tracks_doubtful)} douteuses</span>}
          </span>
        ) : (
          '—'
        ),
      sortValue: (r) => r.track_count,
    },
    {
      key: 'usable',
      header: `Exploitables (≥ ${minViews} vues)`,
      align: 'right',
      width: 170,
      render: (r) =>
        r.usable_rate === null ? '—' : (
          <span className="u-num">
            {formatNumber(r.usable)} · <span className="u-strong">{formatPercent(r.usable_rate, 0)}</span>
          </span>
        ),
      sortValue: (r) => r.usable_rate ?? -1,
    },
    {
      key: 'reviewed',
      header: 'Relues',
      align: 'right',
      width: 80,
      render: (r) => (r.reviewed ? <Badge tone="warning">{formatNumber(r.reviewed)}</Badge> : <span className="u-subtle">0</span>),
    },
    {
      key: 'speed',
      header: 'ms/image',
      align: 'right',
      width: 90,
      render: (r) => (r.ms_per_frame === null ? '—' : formatNumber(r.ms_per_frame, 1)),
    },
    {
      key: 'actions',
      header: <span className="u-sr-only">Actions</span>,
      width: 110,
      align: 'right',
      render: (r) => (
        <span onClick={(e) => e.stopPropagation()}>
          <Button
            size="sm"
            variant="ghost"
            icon={r.detection_status === 'done' ? RotateCw : Play}
            disabled={r.frame_count === 0 || !!r.job_id || run.isPending}
            onClick={() => void launch([r.id])}
          >
            {r.detection_status === 'done' ? 'Relancer' : 'Lancer'}
          </Button>
        </span>
      ),
    },
  ];

  return (
    <div className="u-stack-4">
      {jobId && (
        <Panel title="Traitement en cours">
          <JobProgress
            jobId={jobId}
            onDone={(job) => {
              void qc.invalidateQueries({ queryKey: detKeys.lotsAll });
              void qc.invalidateQueries({ queryKey: queryKeys.project });
              void qc.invalidateQueries({ queryKey: ['detection', 'stats'] });
              void qc.invalidateQueries({ queryKey: ['detection', 'tracks'] });
              if (job.status === 'done') toast.success('Traitement terminé', job.title);
            }}
          />
          <JobLotResults jobId={jobId} />
        </Panel>
      )}

      <Panel
        flush
        title="Lots"
        description="Sélectionnez un ou plusieurs lots puis lancez la détection : les pistes et les vues recadrées remplacent celles d’un traitement précédent."
        actions={
          <div className="u-row">
            <TextInput
              size="sm"
              leading={<Search size={14} aria-hidden />}
              placeholder="Rechercher un lot…"
              value={q}
              onChange={(e) => {
                setQ(e.target.value);
                setPage(0);
              }}
              aria-label="Rechercher un lot"
            />
            <Button variant="primary" icon={Play} disabled={!picked.length} loading={run.isPending} onClick={() => void launch(picked)}>
              Traiter {picked.length ? `${formatNumber(picked.length)} lot(s)` : 'la sélection'}
            </Button>
          </div>
        }
      >
        <DataTable
          columns={columns}
          rows={items}
          rowKey={(r) => r.id}
          loading={lots.isLoading}
          error={lots.error}
          onRetry={() => void lots.refetch()}
          onRowClick={(r) => r.track_count > 0 && navigate(`/detection/pistes?lot=${encodeURIComponent(r.id)}`)}
          empty={<EmptyState icon={ScanSearch} title="Aucun lot" description="Les lots enregistrés en prise de vue apparaîtront ici." compact />}
        />
        {(lots.data?.total ?? 0) > PAGE && (
          <div className={s.pagination}>
            <Pagination page={page} pageSize={PAGE} total={lots.data?.total ?? 0} onPageChange={setPage} />
          </div>
        )}
      </Panel>

      <ConfirmDialog
        open={!!confirm}
        variant="danger"
        title="Remplacer des pistes déjà relues ?"
        confirmLabel="Relancer quand même"
        message={
          confirm && (
            <div className="u-stack-4">
              <Callout tone="warning" icon={AlertTriangle}>
                Relancer la détection supprime les pistes actuelles de ces lots, <strong>y compris les labels relus par un humain</strong>.
              </Callout>
              <ul className={s.plainList}>
                {confirm.lots.map((l) => (
                  <li key={l.lot_id}>
                    <span className="u-mono">{l.lot_id}</span> : {formatNumber(l.reviewed)} piste(s) relue(s)
                  </li>
                ))}
              </ul>
            </div>
          )
        }
        onCancel={() => setConfirm(null)}
        onConfirm={async () => {
          const ids = confirm!.ids;
          setConfirm(null);
          await launch(ids, true);
        }}
      />
    </div>
  );
}

type LotResult = { status: 'done' | 'refused' | 'error'; error?: string } & Partial<RunStats>;

/** Résultats par lot publiés au fil du traitement (lots déjà terminés). */
function JobLotResults({ jobId }: { jobId: string }) {
  const { data: job } = useJob<Record<string, unknown>, { lots?: Record<string, LotResult> }>(jobId);
  const entries = Object.entries(job?.result?.lots ?? {});
  if (!entries.length) return null;
  return (
    <ul className={s.resultList}>
      {entries.map(([lotId, r]) => (
        <li key={lotId}>
          <span className="u-mono u-truncate">{lotId}</span>
          {r.status === 'done' ? (
            <span className="u-num u-small">
              {formatNumber(r.tracks)} pistes · {formatNumber(r.usable)} exploitables · {formatNumber(r.ms_per_frame, 1)} ms/image ·{' '}
              {r.speed_mm_s ? `${formatNumber(r.speed_mm_s, 0)} mm/s` : 'vitesse inconnue'}
            </span>
          ) : (
            <Badge tone="danger">{r.error ?? 'Erreur'}</Badge>
          )}
        </li>
      ))}
    </ul>
  );
}

export default Processing;
