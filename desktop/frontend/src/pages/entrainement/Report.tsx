/** Rapport d'évaluation : chiffres clés, matrice de confusion, ROC / PR, rappel par défaut, choix du seuil. */
import { useEffect, useMemo, useState } from 'react';
import { keepPreviousData, useQuery } from '@tanstack/react-query';
import { Check, RefreshCw, Target } from 'lucide-react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceDot,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { api } from '@/api/client';
import {
  Button,
  Callout,
  DataTable,
  ErrorState,
  Field,
  LoadingState,
  NumberInput,
  Panel,
  SegmentedControl,
  Slider,
  Stat,
  useToast,
  type Column,
} from '@/components/ui';
import { defectLabel } from '@/lib/domain';
import { formatDateTime, formatNumber } from '@/lib/format';
import { cssVar } from '@/lib/theme';
import { useDebounced, useRefreshReport, useReport, useSetThreshold, useSuggest } from './api';
import { num3, pct } from './common';
import type { Confusion, ModelDetail, Objective, ReportBlock } from './types';
import s from './entrainement.module.css';

type Evaluated = { threshold: number; val?: Confusion & { recall_by_defect: Record<string, { n: number; detected: number; recall: number | null }> }; test?: Evaluated['val'] };

function useEvaluate(id: string, threshold: number) {
  const t = useDebounced(threshold, 120);
  return useQuery({
    queryKey: ['training', 'models', 'evaluate', id, t],
    queryFn: ({ signal }) => api.get<Evaluated>(`/training/models/${encodeURIComponent(id)}/threshold/evaluate`, undefined, { signal, query: { threshold: t } }),
    placeholderData: keepPreviousData,
  });
}

export function ReportView({ model }: { model: ModelDetail }) {
  const toast = useToast();
  const report = useReport(model.id);
  const refresh = useRefreshReport();
  const setThr = useSetThreshold();
  const [thr, setThrLocal] = useState(model.threshold);
  useEffect(() => setThrLocal(model.threshold), [model.threshold]);
  const ev = useEvaluate(model.id, thr);

  if (report.isLoading) return <LoadingState />;
  if (report.error || !report.data) return <ErrorState error={report.error} onRetry={() => report.refetch()} />;
  const r = report.data;
  const test = r.subsets.test;
  const val = r.subsets.val;
  const at = ev.data?.test ?? test?.at_threshold;
  const changed = Math.abs(thr - model.threshold) > 1e-12;

  const apply = async (value: number) => {
    try {
      const out = await setThr.mutateAsync({ id: model.id, threshold: value });
      toast.success(`Seuil ${formatNumber(value, 4)} appliqué`, out.export_outdated ? 'Le paquet exporté utilise un autre seuil : ré-exporter.' : undefined);
    } catch (e) {
      toast.error(e);
    }
  };

  return (
    <div className="u-stack-6">
      <div className={s.toolbar}>
        <span className="u-muted u-small">
          Rapport du {formatDateTime(r.generated_at)} · décision : éjecter si score ≥ seuil · seuil du modèle <span className="u-num u-strong">{formatNumber(model.threshold, 4)}</span>
        </span>
        <span className="u-spacer" />
        <Button
          icon={RefreshCw}
          size="sm"
          loading={refresh.isPending}
          onClick={() =>
            refresh.mutate(model.id, {
              onSuccess: () => toast.success('Rapport recalculé avec les labels actuels'),
              onError: (e) => toast.error(e),
            })
          }
        >
          Recalculer (après corrections de labels)
        </Button>
      </div>
      {model.export_outdated && (
        <Callout tone="warning" title="Export obsolète">
          Le seuil a changé depuis le dernier export : le paquet de déploiement utilise encore l'ancien seuil.
        </Callout>
      )}
      {!test ? (
        <Callout tone="warning">Jeu de test vide : pas d'évaluation finale.</Callout>
      ) : (
        <>
          <div className={s.statsGrid}>
            <Stat label="Rappel non viables (test)" value={pct(at?.recall)} hint={`${at?.tp ?? '—'} / ${at?.n_pos ?? '—'} éjectés`} tone="success" />
            <Stat label="Faux rejets (test)" value={pct(at?.false_reject_rate, 2)} hint={`${at?.fp ?? '—'} viables sur ${at?.n_neg ?? '—'}`} tone="warning" />
            <Stat label="AUC ROC" value={num3(test.auc_roc)} hint={val ? `validation ${num3(val.auc_roc)}` : undefined} />
            <Stat label="Précision moyenne" value={num3(test.average_precision)} hint="aire sous la courbe PR" />
            <Stat label="Œufs de test" value={formatNumber(test.n)} hint={`${test.n_pos} non viables · ${test.n_neg} viables`} />
          </div>

          <Panel title="Seuil de décision" description="Faites varier le seuil : toutes les métriques de cette page sont recalculées ; « Appliquer » l'enregistre dans le modèle (et dans le prochain export).">
            <div className="u-stack-4">
              <div className={s.thresholdBar}>
                <Slider value={thr} onChange={setThrLocal} min={0} max={1} step={0.001} format={(v) => formatNumber(v, 3)} />
                <div className="u-row">
                  <NumberInput value={thr} onChange={(v) => v != null && setThrLocal(Math.min(1, Math.max(0, v)))} min={0} max={1} step={0.01} decimals={6} size="sm" />
                  <Button variant="primary" icon={Check} disabled={!changed} loading={setThr.isPending} onClick={() => apply(thr)}>
                    Appliquer
                  </Button>
                  {changed && (
                    <Button variant="ghost" size="sm" onClick={() => setThrLocal(model.threshold)}>
                      Revenir à {formatNumber(model.threshold, 3)}
                    </Button>
                  )}
                </div>
              </div>
              <ObjectivePicker modelId={model.id} onExplore={setThrLocal} onApply={apply} applying={setThr.isPending} />
            </div>
          </Panel>

          <div className={s.twoCols}>
            <Panel title="Matrice de confusion (test)" description={`Au seuil ${formatNumber(thr, 3)}.`}>
              {at && <ConfusionMatrix c={at} />}
              {ev.data?.val && (
                <p className="u-xs u-muted">
                  Validation au même seuil : rappel {pct(ev.data.val.recall)} · faux rejets {pct(ev.data.val.false_reject_rate, 2)}
                </p>
              )}
            </Panel>
            <Panel title="Rappel par type de défaut (test)" flush>
              <DefectRecall data={ev.data?.test?.recall_by_defect ?? test.recall_by_defect ?? {}} />
            </Panel>
          </div>

          <div className={s.twoCols}>
            <Panel title="Courbe ROC (test)" description="Taux de rappel en fonction du taux de faux rejets ; point = seuil courant.">
              <RocChart block={test} point={at ? { x: at.false_reject_rate ?? 0, y: at.recall ?? 0 } : null} />
            </Panel>
            <Panel title="Courbe précision / rappel (test)" description="Point = seuil courant.">
              <PrChart block={test} point={at ? { x: at.recall ?? 0, y: at.precision ?? 1 } : null} />
            </Panel>
          </div>
          {test.histogram && (
            <Panel title="Distribution des scores (test)" description="Les œufs à droite du seuil sont éjectés.">
              <Histogram block={test} threshold={thr} />
            </Panel>
          )}
        </>
      )}
      {r.training && (
        <p className="u-xs u-muted">
          Entraînement : {r.training.epochs_done} époques, meilleure {r.training.best_epoch ?? '—'} ({r.training.monitor}){r.training.stopped_early ? ', arrêt anticipé' : ''}.
        </p>
      )}
    </div>
  );
}

function ObjectivePicker({ modelId, onExplore, onApply, applying }: { modelId: string; onExplore: (t: number) => void; onApply: (t: number) => void; applying: boolean }) {
  const [objective, setObjective] = useState<Objective>('recall');
  const [target, setTarget] = useState<number | null>(95);
  const [subset, setSubset] = useState<'val' | 'test'>('val');
  const body = useMemo(() => (target == null ? null : { objective, target: target / 100, subset }), [objective, target, subset]);
  const debounced = useDebounced(body, 250);
  const sug = useSuggest(modelId, debounced);
  const sd = sug.data;
  return (
    <div className="u-stack-4">
      <div className={s.objectiveRow}>
        <Field label="Objectif">
          <SegmentedControl
            value={objective}
            onChange={(v) => {
              setObjective(v);
              setTarget(v === 'recall' ? 95 : 2);
            }}
            options={[
              { value: 'recall', label: 'Rappel non viables ≥' },
              { value: 'false_reject', label: 'Faux rejets ≤' },
            ]}
          />
        </Field>
        <Field label="Valeur">
          <NumberInput value={target} onChange={setTarget} min={0} max={100} step={objective === 'recall' ? 1 : 0.5} decimals={2} unit="%" />
        </Field>
        <Field label="Calculé sur">
          <SegmentedControl
            value={subset}
            onChange={setSubset}
            options={[
              { value: 'val', label: 'Validation', title: 'Recommandé : le test reste une mesure indépendante' },
              { value: 'test', label: 'Test' },
            ]}
          />
        </Field>
      </div>
      {sug.error ? (
        <Callout tone="danger">{(sug.error as { detail?: string }).detail ?? 'Objectif impossible à évaluer.'}</Callout>
      ) : sd ? (
        <div className="u-stack-4">
          <div className={s.suggestion}>
            <span className={s.head}>Seuil proposé</span>
            <span className={s.head}>Validation</span>
            <span className={s.head}>Test</span>
            <span className="u-num u-strong" style={{ fontSize: 'var(--fs-xl)' }}>
              {formatNumber(sd.threshold, 4)}
            </span>
            <SugCell c={sd.val} />
            <SugCell c={sd.test} />
          </div>
          <div className="u-row">
            <Button icon={Target} size="sm" onClick={() => onExplore(sd.threshold)}>
              Explorer ce seuil
            </Button>
            <Button variant="primary" size="sm" icon={Check} loading={applying} onClick={() => onApply(sd.threshold)}>
              Appliquer ce seuil
            </Button>
          </div>
        </div>
      ) : (
        <LoadingState />
      )}
    </div>
  );
}

function SugCell({ c }: { c?: Confusion }) {
  if (!c) return <span className="u-muted">—</span>;
  return (
    <span className="u-small">
      rappel <span className="u-num u-strong">{pct(c.recall)}</span> · faux rejets <span className="u-num u-strong">{pct(c.false_reject_rate, 2)}</span>{' '}
      {c.objective_met === true ? <span className={s.ok}>✓ objectif atteint</span> : c.objective_met === false ? <span className={s.ko}>✗ non atteint</span> : null}
    </span>
  );
}

export function ConfusionMatrix({ c }: { c: Confusion }) {
  return (
    <table className={s.confusion}>
      <thead>
        <tr>
          <th />
          <th>Décision : viable</th>
          <th>Décision : éjecté</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th scope="row">Réel : viable ({c.n_neg})</th>
          <td className={s.cellGood}>
            <div className={s.cellValue}>{formatNumber(c.tn)}</div>
            <div className="u-xs u-muted">conservés</div>
          </td>
          <td className={c.fp ? s.cellBad : undefined}>
            <div className={s.cellValue}>{formatNumber(c.fp)}</div>
            <div className="u-xs u-muted">faux rejets</div>
          </td>
        </tr>
        <tr>
          <th scope="row">Réel : non viable ({c.n_pos})</th>
          <td className={c.fn ? s.cellBad : undefined}>
            <div className={s.cellValue}>{formatNumber(c.fn)}</div>
            <div className="u-xs u-muted">manqués</div>
          </td>
          <td className={s.cellGood}>
            <div className={s.cellValue}>{formatNumber(c.tp)}</div>
            <div className="u-xs u-muted">éjectés</div>
          </td>
        </tr>
      </tbody>
    </table>
  );
}

type DefRow = { key: string; n: number; detected: number; recall: number | null };

function DefectRecall({ data }: { data: Record<string, { n: number; detected: number; recall: number | null }> }) {
  const rows: DefRow[] = Object.entries(data).map(([key, v]) => ({ key, ...v }));
  const columns: Column<DefRow>[] = [
    { key: 'd', header: 'Défaut', render: (r) => (r.key === 'non_precise' ? 'Non précisé' : defectLabel(r.key)), sortValue: (r) => r.key },
    { key: 'n', header: 'Œufs', align: 'right', render: (r) => formatNumber(r.n), sortValue: (r) => r.n },
    { key: 'det', header: 'Éjectés', align: 'right', render: (r) => formatNumber(r.detected), sortValue: (r) => r.detected },
    {
      key: 'r',
      header: 'Rappel',
      align: 'right',
      render: (r) => <span className={r.recall != null && r.recall < 0.9 ? s.ko : undefined}>{pct(r.recall)}</span>,
      sortValue: (r) => r.recall ?? -1,
    },
  ];
  return <DataTable columns={columns} rows={rows} rowKey={(r) => r.key} dense minWidth={360} empty="Aucun œuf non viable dans le test." />;
}

function RocChart({ block, point }: { block: ReportBlock; point: { x: number; y: number } | null }) {
  const data = useMemo(() => (block.roc ? block.roc.fpr.map((x, i) => ({ x, y: block.roc!.tpr[i] })) : []), [block]);
  return <Curve data={data} point={point} xLabel="faux rejets" yLabel="rappel" diagonal />;
}

function PrChart({ block, point }: { block: ReportBlock; point: { x: number; y: number } | null }) {
  const data = useMemo(() => (block.pr ? block.pr.recall.map((x, i) => ({ x, y: block.pr!.precision[i] })) : []), [block]);
  return <Curve data={data} point={point} xLabel="rappel" yLabel="précision" />;
}

function Curve({ data, point, xLabel, yLabel, diagonal }: { data: { x: number; y: number }[]; point: { x: number; y: number } | null; xLabel: string; yLabel: string; diagonal?: boolean }) {
  const axis = cssVar('--chart-axis');
  return (
    <div className={s.chart}>
      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={data} margin={{ top: 8, right: 16, bottom: 16, left: 0 }}>
          <CartesianGrid stroke={cssVar('--chart-grid')} />
          <XAxis dataKey="x" type="number" domain={[0, 1]} stroke={axis} fontSize={11} tickFormatter={(v: number) => formatNumber(v, 1)} label={{ value: xLabel, position: 'insideBottom', offset: -8, fontSize: 11, fill: axis }} />
          <YAxis dataKey="y" type="number" domain={[0, 1]} stroke={axis} fontSize={11} width={40} tickFormatter={(v: number) => formatNumber(v, 1)} label={{ value: yLabel, angle: -90, position: 'insideLeft', fontSize: 11, fill: axis }} />
          <Tooltip formatter={(v) => formatNumber(Number(v), 3)} labelFormatter={(x) => `${xLabel} ${formatNumber(Number(x), 3)}`} />
          {diagonal && <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke={axis} strokeDasharray="3 3" />}
          <Line dataKey="y" type="stepAfter" stroke={cssVar('--chart-2')} dot={false} strokeWidth={2} isAnimationActive={false} name={yLabel} />
          {point && <ReferenceDot x={point.x} y={point.y} r={6} fill={cssVar('--chart-1')} stroke={cssVar('--surface')} strokeWidth={2} />}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function Histogram({ block, threshold }: { block: ReportBlock; threshold: number }) {
  const h = block.histogram!;
  const data = h.viable.map((v, i) => ({ x: (h.edges[i] + h.edges[i + 1]) / 2, viable: v, non_viable: h.non_viable[i] }));
  const axis = cssVar('--chart-axis');
  const nearest = data.reduce((b, d) => (Math.abs(d.x - threshold) < Math.abs(b - threshold) ? d.x : b), data[0]?.x ?? 0);
  return (
    <div className={s.chart}>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={data} margin={{ top: 16, right: 12, bottom: 4, left: 0 }} barCategoryGap={1}>
          <CartesianGrid stroke={cssVar('--chart-grid')} vertical={false} />
          <XAxis dataKey="x" stroke={axis} fontSize={11} tickFormatter={(v: number) => formatNumber(v, 2)} />
          <YAxis stroke={axis} fontSize={11} width={36} allowDecimals={false} />
          <Tooltip labelFormatter={(x) => `score ≈ ${formatNumber(Number(x), 2)}`} />
          <Legend wrapperStyle={{ fontSize: 12 }} formatter={(n) => (n === 'viable' ? 'Viables' : 'Non viables')} />
          <Bar dataKey="viable" stackId="a" fill={cssVar('--verdict-viable')} isAnimationActive={false} />
          <Bar dataKey="non_viable" stackId="a" fill={cssVar('--verdict-non-viable')} isAnimationActive={false} />
          <ReferenceLine x={nearest} stroke={cssVar('--chart-1')} strokeWidth={2} label={{ value: `seuil ${formatNumber(threshold, 3)}`, position: 'insideTopRight', fontSize: 11, fill: axis }} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

