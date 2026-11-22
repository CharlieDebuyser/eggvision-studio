/** Courbes d'apprentissage (perte, métriques de validation) — mises à jour en direct pendant l'entraînement. */
import { CartesianGrid, Legend, Line, LineChart, ReferenceArea, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { cssVar } from '@/lib/theme';
import { formatNumber } from '@/lib/format';
import type { Epoch } from './types';
import s from './entrainement.module.css';

const NAMES: Record<string, string> = {
  train_loss: 'Perte entraînement',
  val_loss: 'Perte validation',
  val_auc: 'AUC ROC (val.)',
  val_ap: 'Précision moyenne (val.)',
  val_recall: 'Rappel @ 0,5 (val.)',
  val_false_reject_rate: 'Faux rejets @ 0,5 (val.)',
};

export function Curves({ history, bestEpoch, totalEpochs }: { history: Epoch[]; bestEpoch: number | null; totalEpochs?: number | null }) {
  if (history.length === 0) return <p className="u-muted u-small">Les courbes apparaissent à la fin de la première époque.</p>;
  const frozenEnd = history.filter((h) => h.backbone_frozen).reduce((m, h) => Math.max(m, h.epoch), 0);
  const axis = cssVar('--chart-axis');
  const grid = cssVar('--chart-grid');
  const xDomain: [number, number] = [1, Math.max(totalEpochs ?? 0, history.length, 2)];
  const common = (
    <>
      <CartesianGrid stroke={grid} vertical={false} />
      <XAxis dataKey="epoch" type="number" domain={xDomain} allowDecimals={false} stroke={axis} fontSize={11} label={{ value: 'époque', position: 'insideBottomRight', offset: -2, fontSize: 11, fill: axis }} />
      <Tooltip
        formatter={(v, n) => [formatNumber(Number(v), 4), NAMES[String(n)] ?? n]}
        labelFormatter={(e) => `Époque ${e}`}
      />
      <Legend formatter={(n) => NAMES[String(n)] ?? n} wrapperStyle={{ fontSize: 12 }} />
      {frozenEnd > 0 && (
        <ReferenceArea x1={1} x2={frozenEnd + 0.5} fill={cssVar('--surface-3')} fillOpacity={0.6} label={{ value: 'gelé', fontSize: 10, fill: axis, position: 'insideBottomLeft' }} />
      )}
      {bestEpoch != null && (
        <ReferenceLine x={bestEpoch} stroke={cssVar('--success')} strokeDasharray="4 3" label={{ value: `meilleure (${bestEpoch})`, fontSize: 10, fill: cssVar('--success'), position: 'insideTopRight' }} />
      )}
    </>
  );
  return (
    <div className={s.twoCols}>
      <div className={s.chart}>
        <h4 className={s.chartTitle}>Perte</h4>
        <ResponsiveContainer width="100%" height={240}>
          <LineChart data={history} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
            {common}
            <YAxis stroke={axis} fontSize={11} width={48} tickFormatter={(v: number) => formatNumber(v, 2)} />
            <Line dataKey="train_loss" stroke={cssVar('--chart-2')} dot={false} strokeWidth={2} isAnimationActive={false} />
            <Line dataKey="val_loss" stroke={cssVar('--chart-1')} dot={false} strokeWidth={2} isAnimationActive={false} connectNulls />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className={s.chart}>
        <h4 className={s.chartTitle}>Validation</h4>
        <ResponsiveContainer width="100%" height={240}>
          <LineChart data={history} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
            {common}
            <YAxis stroke={axis} fontSize={11} width={48} domain={[0, 1]} tickFormatter={(v: number) => formatNumber(v, 1)} />
            <Line dataKey="val_auc" stroke={cssVar('--chart-3')} dot={false} strokeWidth={2} isAnimationActive={false} connectNulls />
            <Line dataKey="val_ap" stroke={cssVar('--chart-5')} dot={false} strokeWidth={1.5} isAnimationActive={false} connectNulls />
            <Line dataKey="val_recall" stroke={cssVar('--chart-2')} dot={false} strokeWidth={1.5} strokeDasharray="5 3" isAnimationActive={false} connectNulls />
            <Line dataKey="val_false_reject_rate" stroke={cssVar('--chart-4')} dot={false} strokeWidth={1.5} strokeDasharray="5 3" isAnimationActive={false} connectNulls />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
