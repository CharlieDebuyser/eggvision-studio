/** Banc de mesure : latence par colonne sur le Raspberry Pi (agent) ou sur ce PC, historisée et comparée à la cadence. */
import { useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { Cpu, Gauge, Laptop } from 'lucide-react';
import { Bar, BarChart, CartesianGrid, Legend, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { JobProgress } from '@/components/JobProgress';
import { Badge, Button, Callout, DataTable, Field, NumberInput, Panel, SegmentedControl, Stat, useToast, type Column } from '@/components/ui';
import { formatDateTime, formatNumber } from '@/lib/format';
import { cssVar } from '@/lib/theme';
import { useAgentStatus, useBenchHistory, useExportStatus, useStartBench } from './api';
import type { BenchEntry } from './types';
import s from './export.module.css';

/** Cadence nominale de la table : une colonne de 6 œufs toutes les 0,5 s. */
const CADENCE_MS = 500;

function verdict(p99: number) {
  const ratio = CADENCE_MS / p99;
  if (ratio >= 1.5) return { tone: 'success' as const, label: `Tient la cadence (marge ×${formatNumber(ratio, 1)})` };
  if (ratio >= 1) return { tone: 'warning' as const, label: `Juste (marge ×${formatNumber(ratio, 2)})` };
  return { tone: 'danger' as const, label: `Trop lent (×${formatNumber(1 / ratio, 1)} la cadence)` };
}

const temperature = (e: BenchEntry) => {
  const t = e.result.device?.temperature_c;
  return typeof t === 'number' ? `${formatNumber(t, 1)} °C` : '—';
};
const deviceName = (e: BenchEntry) => {
  const d = e.result.device ?? {};
  if (d.model) return String(d.model);
  if (d.system) return `${String(d.system)} ${String(d.machine ?? '')}`.trim();
  return e.target === 'pi' ? 'Raspberry Pi' : 'ce PC';
};

export function BenchPanel({ modelId, variants, nViews }: { modelId: string; variants: ('fp32' | 'int8')[]; nViews: number }) {
  const toast = useToast();
  const status = useExportStatus(modelId);
  const [startedJob, setStartedJob] = useState<string | null>(null);
  const activeJob = status.data?.active_jobs.find((j) => j.kind === 'bench')?.id ?? null;
  const benchJob = startedJob ?? activeJob;
  const history = useBenchHistory(modelId, !!benchJob);
  const agent = useAgentStatus();
  const start = useStartBench();
  const [variant, setVariant] = useState<'fp32' | 'int8'>(variants.includes('int8') ? 'int8' : 'fp32');
  const [nColumns, setNColumns] = useState<number | null>(50);
  const [threads, setThreads] = useState<number | null>(null);
  const agentUrl = (agent.data?.kind === 'pi' && agent.data.connected ? agent.data.agent_url : null) ?? history.data?.agent_url ?? null;
  const items = history.data?.items ?? [];
  const last = items[0];

  const run = async (target: 'local' | 'pi') => {
    try {
      const out = await start.mutateAsync({ id: modelId, target, variant, n_columns: nColumns ?? 50, threads });
      setStartedJob(out.job_id);
      toast.info(target === 'pi' ? 'Paquet envoyé à l’agent : mesure en cours' : 'Mesure en cours sur ce PC');
    } catch (e) {
      toast.error(e);
    }
  };

  return (
    <Panel
      title="Banc de mesure"
      description={`Latence par colonne (6 œufs × ${nViews} vues, prétraitement compris) avec ONNX Runtime sur CPU, comparée à la cadence de ${CADENCE_MS} ms par colonne.`}
    >
      <div className="u-stack-6">
        <div className="u-stack-4">
          <div className={s.toolbar}>
            {agentUrl ? (
              <Badge tone="success" dot>
                Agent Pi connecté — {agentUrl}
              </Badge>
            ) : (
              <Badge tone="warning" dot>
                Aucun agent Pi connecté
              </Badge>
            )}
            {!agentUrl && (
              <Link to="/capture" className="u-small">
                Se connecter au Pi (Prise de vue)
              </Link>
            )}
          </div>
          <div className={s.fields3}>
            <Field label="Variante">
              <SegmentedControl
                value={variant}
                onChange={setVariant}
                options={[
                  { value: 'fp32', label: 'FP32' },
                  { value: 'int8', label: 'INT8', disabled: !variants.includes('int8') },
                ]}
              />
            </Field>
            <Field label="Colonnes mesurées">
              <NumberInput value={nColumns} onChange={setNColumns} min={1} max={10000} step={10} />
            </Field>
            <Field label="Fils ONNX Runtime" hint="vide = défaut">
              <NumberInput value={threads} onChange={setThreads} min={1} max={64} allowEmpty placeholder="auto" />
            </Field>
          </div>
          <div className="u-row">
            <Button variant="primary" icon={Cpu} onClick={() => run('pi')} disabled={!agentUrl || !!benchJob} loading={start.isPending && start.variables?.target === 'pi'}>
              Tester sur le Pi
            </Button>
            <Button icon={Laptop} onClick={() => run('local')} disabled={!!benchJob} loading={start.isPending && start.variables?.target === 'local'}>
              Tester sur ce PC
            </Button>
          </div>
          {benchJob && (
            <JobProgress
              jobId={benchJob}
              compact
              onDone={(job) => {
                setStartedJob(null);
                history.refetch();
                status.refetch();
                if (job.status === 'failed') toast.error('Banc de mesure en échec', job.error ?? undefined);
              }}
            />
          )}
        </div>

        {last && <LastResult e={last} />}
        {items.length > 0 && <HistoryChart items={items} />}
        {items.length > 0 && <HistoryTable items={items} />}
        <Callout tone="info" title="Moteurs d'inférence">
          ONNX Runtime (CPU) est disponible. Les moteurs Hailo (accélérateur AI Kit) et TFLite sont prévus par l'interface du moteur d'inférence mais pas encore
          disponibles.
        </Callout>
      </div>
    </Panel>
  );
}

function LastResult({ e }: { e: BenchEntry }) {
  const r = e.result;
  const v = verdict(r.latency_ms.p99);
  const parity = r.parity;
  const total = r.preprocess_ms_mean + r.inference_ms_mean;
  return (
    <div className="u-stack-4">
      <div className={s.toolbar}>
        <Gauge size={16} aria-hidden />
        <span className="u-strong">
          Dernière mesure — {e.target === 'pi' ? 'Raspberry Pi' : 'ce PC'} ({deviceName(e)}), {e.variant.toUpperCase()}
        </span>
        <span className="u-small u-muted">{formatDateTime(e.at)}</span>
        <span className="u-spacer" />
        <Badge tone={v.tone}>{v.label}</Badge>
      </div>
      <div className={s.statsGrid}>
        <Stat label="Moyenne / colonne" value={`${formatNumber(r.latency_ms.mean, 1)} ms`} hint={`${formatNumber(r.columns_per_second, 1)} colonnes/s`} />
        <Stat label="p95" value={`${formatNumber(r.latency_ms.p95, 1)} ms`} hint={`médiane ${formatNumber(r.latency_ms.median, 1)} ms`} />
        <Stat label="p99" value={`${formatNumber(r.latency_ms.p99, 1)} ms`} hint={`max ${formatNumber(r.latency_ms.max, 1)} ms`} tone={v.tone} />
        <Stat
          label={e.target === 'pi' ? 'Parité sur le Pi' : 'Parité sur ce PC'}
          value={parity ? formatNumber(parity.max_abs_diff, 6) : '—'}
          hint={parity ? (parity.ok ? 'conforme à PyTorch' : e.variant === 'int8' ? 'INT8 : écart informatif' : 'hors tolérance') : undefined}
          tone={parity?.ok ? 'success' : e.variant === 'int8' ? 'warning' : 'danger'}
        />
      </div>
      <div>
        <div className="u-xs u-muted" style={{ marginBottom: 4 }}>
          Décomposition moyenne : prétraitement {formatNumber(r.preprocess_ms_mean, 1)} ms · inférence {formatNumber(r.inference_ms_mean, 1)} ms — budget {CADENCE_MS} ms
          {e.target === 'pi' ? ` · température ${temperature(e)}` : ''} · {r.threads ? `${r.threads} fils` : 'fils par défaut'} · ONNX Runtime {r.onnxruntime}
        </div>
        <div className={s.budgetBar} title="Part du budget de 500 ms par colonne">
          <div className={s.budgetFill} style={{ width: `${Math.min(100, (total / CADENCE_MS) * 100)}%`, background: cssVar('--chart-2') }} />
          <div className={s.budgetFill} style={{ width: `${Math.min(100, (r.preprocess_ms_mean / CADENCE_MS) * 100)}%`, background: cssVar('--chart-1') }} />
        </div>
      </div>
    </div>
  );
}

function HistoryChart({ items }: { items: BenchEntry[] }) {
  const data = useMemo(
    () =>
      [...items].reverse().map((e, i) => ({
        name: `${i + 1}. ${e.target === 'pi' ? 'Pi' : 'PC'} ${e.variant.toUpperCase()}`,
        pre: e.result.preprocess_ms_mean,
        inf: e.result.inference_ms_mean,
        p99: e.result.latency_ms.p99,
      })),
    [items],
  );
  const axis = cssVar('--chart-axis');
  const max = Math.max(CADENCE_MS * 1.1, ...data.map((d) => d.p99));
  return (
    <div className={s.chart}>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={data} margin={{ top: 12, right: 12, bottom: 4, left: 0 }}>
          <CartesianGrid stroke={cssVar('--chart-grid')} vertical={false} />
          <XAxis dataKey="name" stroke={axis} fontSize={11} />
          <YAxis stroke={axis} fontSize={11} width={48} unit=" ms" domain={[0, Math.ceil(max / 50) * 50]} />
          <Tooltip formatter={(v, n) => [`${formatNumber(Number(v), 1)} ms`, n === 'pre' ? 'Prétraitement' : n === 'inf' ? 'Inférence' : 'p99']} />
          <Legend wrapperStyle={{ fontSize: 12 }} formatter={(n) => (n === 'pre' ? 'Prétraitement (moy.)' : 'Inférence (moy.)')} />
          <Bar dataKey="pre" stackId="a" fill={cssVar('--chart-1')} isAnimationActive={false} />
          <Bar dataKey="inf" stackId="a" fill={cssVar('--chart-2')} isAnimationActive={false} />
          <ReferenceLine y={CADENCE_MS} stroke={cssVar('--danger')} strokeDasharray="4 3" label={{ value: 'cadence 500 ms', position: 'insideTopRight', fontSize: 11, fill: cssVar('--danger') }} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function HistoryTable({ items }: { items: BenchEntry[] }) {
  const columns: Column<BenchEntry>[] = [
    { key: 'at', header: 'Date', render: (e) => formatDateTime(e.at), sortValue: (e) => e.at },
    { key: 'target', header: 'Cible', render: (e) => (e.target === 'pi' ? <Badge tone="info">Pi</Badge> : <Badge>PC</Badge>), sortValue: (e) => e.target },
    { key: 'dev', header: 'Matériel', render: (e) => <span className="u-truncate">{deviceName(e)}</span>, truncate: true },
    { key: 'v', header: 'Variante', render: (e) => e.variant.toUpperCase(), sortValue: (e) => e.variant },
    { key: 'th', header: 'Fils', align: 'right', render: (e) => e.result.threads ?? 'déf.' },
    { key: 'n', header: 'Colonnes', align: 'right', render: (e) => formatNumber(e.result.n_columns) },
    { key: 'mean', header: 'Moy.', align: 'right', render: (e) => `${formatNumber(e.result.latency_ms.mean, 1)} ms`, sortValue: (e) => e.result.latency_ms.mean },
    { key: 'p95', header: 'p95', align: 'right', render: (e) => `${formatNumber(e.result.latency_ms.p95, 1)} ms`, sortValue: (e) => e.result.latency_ms.p95 },
    { key: 'p99', header: 'p99', align: 'right', render: (e) => `${formatNumber(e.result.latency_ms.p99, 1)} ms`, sortValue: (e) => e.result.latency_ms.p99 },
    { key: 'pre', header: 'Prétrait. / inf.', align: 'right', render: (e) => `${formatNumber(e.result.preprocess_ms_mean, 0)} / ${formatNumber(e.result.inference_ms_mean, 0)} ms` },
    {
      key: 'par',
      header: 'Parité',
      align: 'right',
      render: (e) => (e.result.parity ? <span className={e.result.parity.ok ? s.ok : e.variant === 'int8' ? s.warn : s.ko}>{formatNumber(e.result.parity.max_abs_diff, 6)}</span> : '—'),
    },
    { key: 'temp', header: 'Temp.', align: 'right', render: temperature },
    {
      key: 'verdict',
      header: 'Cadence 0,5 s',
      render: (e) => {
        const v = verdict(e.result.latency_ms.p99);
        return <Badge tone={v.tone}>{v.label}</Badge>;
      },
    },
  ];
  return <DataTable columns={columns} rows={items} rowKey={(e) => e.at + e.target + e.variant} dense minWidth={1100} />;
}
