/** Comparaison de modèles côte à côte : seuil propre de chaque modèle et objectif commun. */
import { useState, type ReactNode } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { Callout, ErrorState, Field, LoadingState, NumberInput, PageHeader, Panel, SegmentedControl } from '@/components/ui';
import { defectLabel } from '@/lib/domain';
import { formatNumber } from '@/lib/format';
import { useCompare, useDebounced, useOptions } from './api';
import { archLabel, num3, pct } from './common';
import type { CompareModel, Objective } from './types';
import s from './entrainement.module.css';

type Row = { label: ReactNode; get: (m: CompareModel) => number | null | undefined; fmt: (v: number | null | undefined) => ReactNode; better?: 'high' | 'low' };

export function ComparePage() {
  const [params] = useSearchParams();
  const ids = (params.get('ids') ?? '').split(',').map(decodeURIComponent).filter(Boolean);
  const [objective, setObjective] = useState<Objective>('recall');
  const [target, setTarget] = useState<number | null>(95);
  const [subset, setSubset] = useState<'val' | 'test'>('val');
  const p = useDebounced({ objective, target: (target ?? 0) / 100, subset }, 250);
  const cmp = useCompare(ids, p);
  const options = useOptions();

  const header = <PageHeader breadcrumbs={[{ label: 'Entraînement', to: '/entrainement' }, { label: 'Comparaison' }]} title="Comparaison de modèles" />;
  if (ids.length < 2)
    return (
      <>
        {header}
        <Callout tone="info">Sélectionnez au moins deux modèles entraînés dans la liste.</Callout>
      </>
    );

  const models = cmp.data?.models ?? [];
  const defects = Array.from(new Set(models.flatMap((m) => Object.keys(m.recall_by_defect ?? {})))).sort();
  const sections: { title: string; rows: Row[] }[] = [
    {
      title: 'Jeu de test',
      rows: [
        { label: 'Œufs de test', get: (m) => m.test_set?.n, fmt: (v) => formatNumber(v) },
        { label: 'dont non viables', get: (m) => m.test_set?.n_pos, fmt: (v) => formatNumber(v) },
        { label: 'AUC ROC', get: (m) => m.auc_roc, fmt: num3, better: 'high' },
        { label: 'Précision moyenne', get: (m) => m.average_precision, fmt: num3, better: 'high' },
      ],
    },
    {
      title: 'Au seuil propre de chaque modèle',
      rows: [
        { label: 'Seuil', get: (m) => m.threshold, fmt: (v) => formatNumber(v, 3) },
        { label: 'Rappel non viables', get: (m) => m.own_threshold?.recall, fmt: (v) => pct(v), better: 'high' },
        { label: 'Faux rejets', get: (m) => m.own_threshold?.false_reject_rate, fmt: (v) => pct(v, 2), better: 'low' },
        ...defects.map<Row>((d) => ({
          label: <span className={s.indent}>rappel {d === 'non_precise' ? 'non précisé' : defectLabel(d)}</span>,
          get: (m) => m.recall_by_defect?.[d]?.recall,
          fmt: (v) => pct(v),
          better: 'high',
        })),
      ],
    },
    {
      title: `Objectif commun : ${objective === 'recall' ? 'rappel ≥' : 'faux rejets ≤'} ${formatNumber(target, 1)} % (seuil calculé sur ${subset === 'val' ? 'la validation' : 'le test'})`,
      rows: [
        { label: 'Seuil', get: (m) => m.common_objective?.threshold, fmt: (v) => formatNumber(v, 3) },
        { label: 'Rappel non viables (test)', get: (m) => m.common_objective?.test?.recall, fmt: (v) => pct(v), better: 'high' },
        { label: 'Faux rejets (test)', get: (m) => m.common_objective?.test?.false_reject_rate, fmt: (v) => pct(v, 2), better: 'low' },
      ],
    },
  ];

  return (
    <>
      {header}
      <div className="u-stack-4">
        <Panel>
          <div className={s.objectiveRow}>
            <Field label="Objectif commun">
              <SegmentedControl
                value={objective}
                onChange={(v) => {
                  setObjective(v);
                  setTarget(v === 'recall' ? 95 : 2);
                }}
                options={[
                  { value: 'recall', label: 'Rappel ≥' },
                  { value: 'false_reject', label: 'Faux rejets ≤' },
                ]}
              />
            </Field>
            <Field label="Valeur">
              <NumberInput value={target} onChange={setTarget} min={0} max={100} step={1} decimals={2} unit="%" />
            </Field>
            <Field label="Seuil calculé sur">
              <SegmentedControl
                value={subset}
                onChange={setSubset}
                options={[
                  { value: 'val', label: 'Validation' },
                  { value: 'test', label: 'Test' },
                ]}
              />
            </Field>
          </div>
        </Panel>
        {cmp.data?.warnings.map((w) => (
          <Callout key={w} tone="warning">
            {w}
          </Callout>
        ))}
        {cmp.error ? (
          <ErrorState error={cmp.error} onRetry={() => cmp.refetch()} />
        ) : !cmp.data ? (
          <LoadingState />
        ) : (
          <Panel flush>
            <div className={s.compareScroll}>
              <table className={s.compareTable}>
                <thead>
                  <tr>
                    <th />
                    {models.map((m) => (
                      <th key={m.id}>
                        <Link to={`/entrainement/modeles/${encodeURIComponent(m.id)}`}>{m.name}</Link>
                        <div className="u-xs u-muted">
                          {m.kind === 'anomaly' ? 'anomalies · ' : ''}
                          {archLabel(options.data?.architectures, m.arch)} · découpage n° {m.split_id ?? '—'}
                        </div>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {sections.map((sec) => (
                    <SectionRows key={sec.title} title={sec.title} rows={sec.rows} models={models} />
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>
        )}
      </div>
    </>
  );
}

function SectionRows({ title, rows, models }: { title: string; rows: Row[]; models: CompareModel[] }) {
  return (
    <>
      <tr className={s.sectionRow}>
        <th colSpan={models.length + 1}>{title}</th>
      </tr>
      {rows.map((r, i) => {
        const vals = models.map((m) => r.get(m));
        const nums = vals.filter((v): v is number => typeof v === 'number');
        const best = r.better && nums.length > 1 && new Set(nums).size > 1 ? (r.better === 'high' ? Math.max(...nums) : Math.min(...nums)) : null;
        return (
          <tr key={i}>
            <th scope="row">{r.label}</th>
            {models.map((m, j) => (
              <td key={m.id} className={best != null && vals[j] === best ? s.best : undefined}>
                {m.error ? <span className="u-muted">{j === 0 && i === 0 ? m.error : '—'}</span> : r.fmt(vals[j])}
              </td>
            ))}
          </tr>
        );
      })}
    </>
  );
}
