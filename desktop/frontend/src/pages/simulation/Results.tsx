/** Résultats chiffrés d'une simulation : chiffres clés, ventilation par défaut, latences vs budget, exports. */
import { useMemo, useState, type ReactNode } from 'react';
import { Download, FileSpreadsheet, Film } from 'lucide-react';
import { Bar, BarChart, CartesianGrid, ReferenceLine, ResponsiveContainer, Tooltip as RTooltip, XAxis, YAxis } from 'recharts';
import { JobProgress } from '@/components/JobProgress';
import { cssVar } from '@/lib/theme';
import { formatBytes, formatDuration, formatNumber, formatPercent } from '@/lib/format';
import { Button, Callout, DataTable, Field, NumberInput, Panel, Select, Stat, useToast, type Column } from '@/components/ui';
import { simUrls, useStartVideo } from './api';
import type { DefectBreakdown, SimDetail, SimResults } from './types';
import s from './simulation.module.css';

export function Results({ sim }: { sim: SimDetail }) {
  const r = sim.results as SimResults;
  if (!r?.counts) return <Callout tone="info">Résultats indisponibles.</Callout>;
  const c = r.counts;
  const lat = r.latency;
  const comp = r.composition;
  return (
    <div className="u-stack-6">
      <div className={s.statsGrid}>
        <Stat label="Non viables éjectés" value={formatPercent(r.rates.recall)} hint={`${formatNumber(c.ejected_ok)} / ${formatNumber(c.non_viable)}`} tone="success" />
        <Stat
          label="Non viables manqués"
          value={formatNumber(c.missed_total)}
          hint={`dont ${formatNumber(c.missed_late)} par retard · ${formatPercent(r.rates.miss_rate)}`}
          tone="danger"
        />
        <Stat label="Viables éjectés à tort" value={formatNumber(c.false_reject)} hint={`${formatPercent(r.rates.false_reject_rate, 2)} des ${formatNumber(c.viable)} viables`} tone="warning" />
        <Stat
          label="Décisions en retard"
          value={formatNumber(c.late_decisions)}
          hint={`${formatNumber(c.missed_traps)} trappe(s) ratée(s) · ${formatPercent(r.rates.late_rate, 2)}`}
          tone={c.late_decisions ? 'danger' : 'success'}
        />
        <Stat label="Défauts résiduels" value={formatPercent(r.rates.residual_defect_rate, 2)} hint={`dans le bac viables (${formatNumber(c.bin_viable)} œufs)`} />
        <Stat label="Rappel du modèle seul" value={formatPercent(r.rates.model_recall)} hint={`seuil ${formatNumber(r.threshold, 3)} · ${r.variant ?? ''}`} />
      </div>

      {comp?.with_replacement && (
        <Callout tone="info" title="Jeu test réutilisé (tirage avec remise)">
          {formatNumber(comp.n_distinct_eggs)} œufs distincts pour {formatNumber(comp.n_eggs)} œufs simulés (taux de réutilisation {formatPercent(comp.reuse_rate)}).
          Les résultats restent valables mais leur précision statistique est celle du jeu test.
        </Callout>
      )}

      <div className={s.twoCols}>
        <Panel title="Ventilation par type de défaut" description="Non viables du jeu test, par défaut labellisé." flush>
          <DefectTable r={r} />
        </Panel>
        <Panel title="Latences et budget de temps" description="Délai entre la fin d'acquisition et la décision (attente comprise).">
          <LatencyHistogram r={r} />
          <dl className={s.kvGrid}>
            <dt>Budget (caméra → trappes)</dt>
            <dd className="u-num">{formatNumber(lat.budget_ms)} ms</dd>
            <dt>Budget effectif (− ouverture {formatNumber(lat.trap_actuation_ms)} ms)</dt>
            <dd className="u-num">{formatNumber(lat.effective_budget_ms)} ms</dd>
            <dt>Latence de calcul moy. / médiane / p95 / p99 / max</dt>
            <dd className="u-num">
              {[lat.compute_ms.mean, lat.compute_ms.median, lat.compute_ms.p95, lat.compute_ms.p99, lat.compute_ms.max].map((v) => formatNumber(v, 1)).join(' / ')} ms
            </dd>
            <dt>Attente en file max</dt>
            <dd className="u-num">{formatNumber(lat.wait_ms.max, 1)} ms</dd>
            <dt>Marge minimale</dt>
            <dd className="u-num">{formatNumber(lat.min_margin_ms, 1)} ms</dd>
            <dt>Capacité / cadence requise</dt>
            <dd className="u-num">
              {r.throughput.capacity_ratio === null ? '—' : `×${formatNumber(r.throughput.capacity_ratio, 1)}`}
              <span className="u-muted"> ({formatNumber(r.throughput.capacity_columns_per_s, 1)} col/s pour {formatNumber(r.throughput.required_columns_per_s, 1)} requises)</span>
            </dd>
            <dt>Durée réelle du calcul</dt>
            <dd className="u-num">
              {formatDuration(r.throughput.wall_s)} <span className="u-muted">(durée simulée {formatDuration(r.throughput.virtual_duration_s)})</span>
            </dd>
          </dl>
        </Panel>
      </div>

      <Exports sim={sim} />
    </div>
  );
}

type DefectRow = DefectBreakdown & { key: string };

function DefectTable({ r }: { r: SimResults }) {
  const rows: DefectRow[] = Object.entries(r.by_defect).map(([key, v]) => ({ key, ...v }));
  const columns: Column<DefectRow>[] = [
    { key: 'label', header: 'Défaut', render: (d) => d.label, sortValue: (d) => d.label },
    { key: 'n', header: 'Œufs', align: 'right', render: (d) => formatNumber(d.n), sortValue: (d) => d.n },
    { key: 'ok', header: 'Éjectés', align: 'right', render: (d) => formatNumber(d.ejected_ok), sortValue: (d) => d.ejected_ok },
    { key: 'missed', header: 'Manqués', align: 'right', render: (d) => formatNumber(d.missed), sortValue: (d) => d.missed },
    { key: 'late', header: 'Retard', align: 'right', render: (d) => formatNumber(d.missed_late), sortValue: (d) => d.missed_late },
    { key: 'recall', header: 'Rappel', align: 'right', render: (d) => formatPercent(d.recall), sortValue: (d) => d.recall ?? -1 },
  ];
  return (
    <>
      <DataTable columns={columns} rows={rows} rowKey={(d) => d.key} dense empty="Aucun œuf non viable dans cette simulation." minWidth={420} />
      <div className={s.tableFoot}>
        Viables : {formatNumber(r.viable.n)} · éjectés à tort {formatNumber(r.viable.false_reject)} ({formatPercent(r.viable.false_reject_rate, 2)})
      </div>
    </>
  );
}

function LatencyHistogram({ r }: { r: SimResults }) {
  const h = r.latency.histogram;
  const data = useMemo(
    () => h.counts.map((n, i) => ({ x: (h.edges[i] + h.edges[i + 1]) / 2, n, late: h.edges[i] >= h.budget_ms })),
    [h],
  );
  if (!data.length) return <p className="u-muted">Aucune décision.</p>;
  const width = h.edges[1] - h.edges[0];
  return (
    <div className={s.chart}>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={data} margin={{ top: 16, right: 12, bottom: 4, left: 0 }} barCategoryGap={1}>
          <CartesianGrid stroke={cssVar('--chart-grid')} vertical={false} />
          <XAxis dataKey="x" tickFormatter={(v: number) => formatNumber(v, 0)} stroke={cssVar('--chart-axis')} fontSize={11} unit=" ms" />
          <YAxis stroke={cssVar('--chart-axis')} fontSize={11} allowDecimals={false} width={40} />
          <RTooltip
            formatter={(v) => [`${formatNumber(Number(v))} colonne(s)`, 'Effectif']}
            labelFormatter={(x) => `${formatNumber(Number(x) - width / 2, 0)}–${formatNumber(Number(x) + width / 2, 0)} ms`}
          />
          <Bar dataKey="n" fill={cssVar('--chart-2')} isAnimationActive={false} />
          {h.budget_in_range !== false && (
            <ReferenceLine
              x={data.reduce((best, d) => (Math.abs(d.x - h.budget_ms) < Math.abs(best - h.budget_ms) ? d.x : best), data[0].x)}
              stroke={cssVar('--danger')}
              strokeDasharray="4 3"
              label={{ value: `budget ${formatNumber(h.budget_ms)} ms`, position: 'insideTopRight', fill: cssVar('--danger'), fontSize: 11 }}
            />
          )}
        </BarChart>
      </ResponsiveContainer>
      {h.budget_in_range === false && (
        <p className="u-muted u-small">
          Budget effectif de {formatNumber(h.budget_ms)} ms hors échelle : délai maximal observé {formatNumber(r.latency.response_ms.max, 1)} ms
          (marge ×{formatNumber(h.budget_ms / Math.max(1e-9, r.latency.response_ms.max ?? 1), 0)}).
        </p>
      )}
    </div>
  );
}

const RESOLUTIONS = [
  { value: '1280x720', label: '1280 × 720 (HD)' },
  { value: '1920x1080', label: '1920 × 1080 (Full HD)' },
  { value: '960x540', label: '960 × 540' },
  { value: '640x360', label: '640 × 360' },
];

function Exports({ sim }: { sim: SimDetail }) {
  const toast = useToast();
  const start = useStartVideo();
  const r = sim.results as SimResults;
  const [res, setRes] = useState('1280x720');
  const [fps, setFps] = useState<number | null>(25);
  const [maxDur, setMaxDur] = useState<number | null>(60);
  const [speed, setSpeed] = useState<number | null>(null);
  const [first, setFirst] = useState<number | null>(1);
  const [last, setLast] = useState<number | null>(r.n_columns);
  const videoJob = sim.video_job;
  const videoActive = videoJob && (videoJob.status === 'queued' || videoJob.status === 'running');

  const launch = async () => {
    const [w, h] = res.split('x').map(Number);
    try {
      await start.mutateAsync({
        id: sim.id,
        width: w,
        height: h,
        fps: fps ?? 25,
        max_duration_s: maxDur ?? 60,
        speed: speed,
        start_column: Math.max(0, (first ?? 1) - 1),
        end_column: last ? last - 1 : null,
      });
      toast.info('Rendu de la vidéo lancé');
    } catch (e) {
      toast.error(e);
    }
  };

  return (
    <div className={s.twoCols}>
      <Panel title="Journal des décisions" description="Une ligne par œuf (CSV, séparateur « ; », tableur Excel / LibreOffice).">
        <DownloadLink href={simUrls.journal(sim.id)} icon={<FileSpreadsheet size={15} aria-hidden />}>
          Exporter le journal CSV
        </DownloadLink>
        <p className="u-muted u-small">
          {formatNumber(r.counts.eggs)} lignes : vérité, score, décision, latence, retard et issue de chaque œuf.
        </p>
      </Panel>
      <Panel title="Vidéo MP4 de l'animation" description="Rendue côté serveur (H.264). Vitesse automatique : la vidéo dure au plus la durée visée.">
        <div className={s.videoForm}>
          <Field label="Résolution">
            <Select value={res} onChange={setRes} options={RESOLUTIONS} />
          </Field>
          <Field label="Images / s">
            <NumberInput value={fps} onChange={setFps} min={5} max={60} step={1} />
          </Field>
          <Field label="Durée visée" hint="si vitesse automatique">
            <NumberInput value={maxDur} onChange={setMaxDur} min={2} max={3600} step={10} unit="s" />
          </Field>
          <Field label="Vitesse" hint="vide = automatique">
            <NumberInput value={speed} onChange={setSpeed} min={0.1} max={1000} step={0.5} allowEmpty placeholder="auto" unit="×" />
          </Field>
          <Field label="Colonnes de">
            <NumberInput value={first} onChange={setFirst} min={1} max={r.n_columns} step={1} />
          </Field>
          <Field label="à">
            <NumberInput value={last} onChange={setLast} min={1} max={r.n_columns} step={1} />
          </Field>
        </div>
        <div className="u-row">
          <Button icon={Film} variant="primary" onClick={launch} loading={start.isPending} disabled={!!videoActive}>
            {r.video ? 'Générer à nouveau' : 'Générer la vidéo'}
          </Button>
          {r.video && !videoActive && (
            <DownloadLink href={simUrls.video(sim.id, true)} icon={<Download size={15} aria-hidden />}>
              Télécharger ({formatBytes(r.video.size_bytes)})
            </DownloadLink>
          )}
        </div>
        {videoJob && videoActive && <JobProgress jobId={videoJob.id} compact hideTitle />}
        {videoJob?.status === 'failed' && <Callout tone="danger" title="Échec du rendu">{videoJob.error}</Callout>}
        {r.video && !videoActive && (
          <div className="u-stack-4">
            <p className="u-muted u-small">
              {formatNumber(r.video.n_frames)} images · {formatDuration(r.video.duration_s)} · {r.video.width} × {r.video.height} · lecture ×
              {formatNumber(r.video.speed, 1)} · colonnes {r.video.columns[0] + 1} à {r.video.columns[1] + 1}
            </p>
            <video className={s.video} src={`${simUrls.video(sim.id)}?v=${encodeURIComponent(r.video.rendered_at)}`} controls preload="metadata" />
          </div>
        )}
      </Panel>
    </div>
  );
}

/** Lien de téléchargement présenté comme un bouton secondaire (classes du socle). */
function DownloadLink({ href, icon, children }: { href: string; icon: ReactNode; children: ReactNode }) {
  return (
    <a href={href} download className="btn btn--secondary btn--md">
      {icon}
      <span className="btn__label">{children}</span>
    </a>
  );
}
