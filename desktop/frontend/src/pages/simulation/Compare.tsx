/** Comparaison côte à côte de plusieurs simulations (paramètres qui diffèrent, métriques, par défaut). */
import { useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { cx } from '@/lib/cx';
import { formatNumber, formatPercent } from '@/lib/format';
import { EmptyState, ErrorState, LoadingState, PageHeader, Panel, Toggle } from '@/components/ui';
import { useCompare } from './api';
import { paramLabel, paramValue } from './SimulationDetail';
import type { CompareResult } from './types';
import s from './simulation.module.css';

function fmt(format: CompareResult['metrics'][number]['format'], v: number | null): string {
  if (v === null || v === undefined) return '—';
  switch (format) {
    case 'pct':
      return formatPercent(v, 2);
    case 'ms':
      return `${formatNumber(v, 1)} ms`;
    case 'ratio':
      return `×${formatNumber(v, 2)}`;
    default:
      return formatNumber(v);
  }
}

export function Compare() {
  const [params] = useSearchParams();
  const ids = (params.get('ids') ?? '').split(',').filter(Boolean);
  const cmp = useCompare(ids);
  const [allParams, setAllParams] = useState(false);

  const header = <PageHeader breadcrumbs={[{ label: 'Simulations', to: '/simulation' }, { label: 'Comparaison' }]} title="Comparaison de simulations" />;
  if (ids.length < 2)
    return (
      <>
        {header}
        <EmptyState title="Sélectionner au moins deux simulations" description="Cocher les simulations à comparer dans la liste." />
      </>
    );
  if (cmp.isLoading)
    return (
      <>
        {header}
        <LoadingState />
      </>
    );
  if (cmp.error || !cmp.data)
    return (
      <>
        {header}
        <ErrorState error={cmp.error} onRetry={() => cmp.refetch()} />
      </>
    );
  const d = cmp.data;
  const rows = d.params.filter((p) => allParams || p.differs);
  return (
    <>
      {header}
      <div className="u-stack-6">
        <Panel
          flush
          title="Paramètres"
          description={rows.length ? undefined : 'Paramètres identiques.'}
          actions={<Toggle checked={allParams} onChange={setAllParams} label="Tous les paramètres" size="sm" />}
        >
          <div className={s.compareScroll}>
            <table className={s.compareTable}>
              <thead>
                <tr>
                  <th />
                  {d.simulations.map((sim) => (
                    <th key={sim.id}>
                      <Link to={`/simulation/${encodeURIComponent(sim.id)}/resultats`}>{sim.name}</Link>
                      <div className="u-subtle u-xs">{sim.model_name ?? sim.model_id}</div>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((p) => (
                  <tr key={p.key} className={cx(p.differs && s.differs)}>
                    <th scope="row">{paramLabel(p.key)}</th>
                    {p.values.map((v, i) => (
                      <td key={i} className="u-num">
                        {paramValue(p.key, v)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
        <Panel flush title="Métriques" description="En gras : la meilleure valeur quand elle se distingue.">
          <div className={s.compareScroll}>
            <table className={s.compareTable}>
              <tbody>
                {d.metrics.map((m) => (
                  <tr key={m.key}>
                    <th scope="row">{m.label}</th>
                    {m.values.map((v, i) => (
                      <td key={i} className={cx('u-num', m.best?.[i] && s.best)}>
                        {fmt(m.format, v)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
        <Panel flush title="Rappel par type de défaut" description="Part des non viables de ce défaut correctement éjectés (effectif entre parenthèses).">
          <div className={s.compareScroll}>
            <table className={s.compareTable}>
              <tbody>
                {d.by_defect.map((row) => (
                  <tr key={row.defect}>
                    <th scope="row">{row.label}</th>
                    {row.values.map((v, i) => (
                      <td key={i} className="u-num">
                        {v ? `${formatPercent(v.recall)} (${v.ejected_ok}/${v.n})` : '—'}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
      </div>
    </>
  );
}
