/** Petits composants partagés du module Entraînement. */
import type { ReactNode } from 'react';
import { AlertTriangle, Cpu, ShieldCheck } from 'lucide-react';
import { Badge, Callout, Tooltip } from '@/components/ui';
import { defectLabel } from '@/lib/domain';
import { formatNumber, formatPercent } from '@/lib/format';
import type { Architecture, ModelStatus, Subset, SplitSummary } from './types';
import s from './entrainement.module.css';

export const SUBSET_LABEL: Record<Subset, string> = { train: 'Entraînement', val: 'Validation', test: 'Test' };
export const SUBSETS: Subset[] = ['train', 'val', 'test'];

const STATUS_META: Record<ModelStatus, { label: string; tone: 'neutral' | 'info' | 'success' | 'danger' | 'warning' }> = {
  created: { label: 'En attente', tone: 'neutral' },
  training: { label: 'Entraînement…', tone: 'info' },
  trained: { label: 'Entraîné', tone: 'success' },
  failed: { label: 'Échec', tone: 'danger' },
  cancelled: { label: 'Interrompu', tone: 'warning' },
};

export function ModelStatusBadge({ status }: { status: ModelStatus }) {
  const m = STATUS_META[status] ?? { label: status, tone: 'neutral' as const };
  return (
    <Badge tone={m.tone} dot>
      {m.label}
    </Badge>
  );
}

export function SubsetBadge({ subset }: { subset: Subset | null | undefined }) {
  if (!subset) return <span className="u-subtle">—</span>;
  const tone = subset === 'train' ? 'info' : subset === 'val' ? 'accent' : 'success';
  return <Badge tone={tone}>{SUBSET_LABEL[subset]}</Badge>;
}

export function archLabel(archs: Architecture[] | undefined, id: string): string {
  return archs?.find((a) => a.id === id)?.label ?? id;
}

/** Indicateur « adapté à l'INT8 ». */
export function Int8Badge({ friendly }: { friendly: boolean }) {
  return friendly ? (
    <Tooltip content="Conçu pour la quantification : la variante INT8 conserve ses performances.">
      <span>
        <Badge tone="success" icon={ShieldCheck}>
          INT8 OK
        </Badge>
      </span>
    </Tooltip>
  ) : (
    <Tooltip content="Se dégrade en INT8 (quantification post-entraînement) : déployer la variante FP32.">
      <span>
        <Badge tone="warning" icon={Cpu}>
          FP32 conseillé
        </Badge>
      </span>
    </Tooltip>
  );
}

export const pct = (v: number | null | undefined, d = 1) => formatPercent(v, d);
export const num3 = (v: number | null | undefined) => formatNumber(v, 3);

/** Effectifs par sous-ensemble × verdict × défaut. */
export function SummaryTable({ summary }: { summary: SplitSummary }) {
  const defects = summary.classes.filter((c) => c !== 'viable');
  const rows: { key: string; label: ReactNode; get: (sub: Subset) => number; strong?: boolean }[] = [
    { key: 'lots', label: 'Lots', get: (sub) => summary.subsets[sub].n_lots },
    { key: 'eggs', label: 'Œufs utilisables', get: (sub) => summary.subsets[sub].n_eggs, strong: true },
    { key: 'viable', label: 'Viables', get: (sub) => summary.subsets[sub].viable },
    { key: 'nv', label: 'Non viables', get: (sub) => summary.subsets[sub].non_viable },
    ...defects.map((d) => ({ key: d, label: <span className={s.indent}>dont {defectLabel(d)}</span>, get: (sub: Subset) => summary.subsets[sub].defects[d] ?? 0 })),
  ];
  return (
    <table className={s.matrix}>
      <thead>
        <tr>
          <th />
          {SUBSETS.map((sub) => (
            <th key={sub}>
              {SUBSET_LABEL[sub]}
              <div className="u-xs u-muted">{formatPercent(summary.subsets[sub].fraction, 0)}</div>
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.key} className={r.strong ? s.strongRow : undefined}>
            <th scope="row">{r.label}</th>
            {SUBSETS.map((sub) => {
              const v = r.get(sub);
              const missing = v === 0 && r.key !== 'lots';
              return (
                <td key={sub} className={missing ? s.missing : undefined}>
                  {formatNumber(v)}
                </td>
              );
            })}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function SplitWarnings({ summary, holdout = false }: { summary: SplitSummary; holdout?: boolean }) {
  const lc = summary.leakage_check;
  return (
    <div className="u-stack-4">
      {lc.ok ? (
        <Callout tone="success" icon={ShieldCheck} title="Aucune fuite possible">
          Chaque lot est dans un seul sous-ensemble
          {lc.unit === 'session' ? ' et chaque session (journée) aussi' : ''}
          {holdout && lc.holdout_respected ? ' ; élevages et journées réservés entièrement en test' : ''}.
          {lc.unit === 'lot' && lc.shared_sessions.length > 0 && (
            <span className="u-muted"> Des journées sont partagées entre sous-ensembles ({lc.shared_sessions.join(', ')}) : regrouper par session pour l'éviter.</span>
          )}
        </Callout>
      ) : (
        <Callout tone="danger" icon={AlertTriangle} title="Fuite possible entre sous-ensembles">
          {lc.shared_sessions.length > 0 && <>Sessions partagées : {lc.shared_sessions.join(', ')}. </>}
          {lc.holdout_respected === false && 'Des lots réservés au test sont placés ailleurs.'}
        </Callout>
      )}
      {summary.warnings.length > 0 && (
        <Callout tone="warning" title={`${summary.warnings.length} avertissement(s)`}>
          <ul className={s.warnList}>
            {summary.warnings.map((w) => (
              <li key={w}>{w}</li>
            ))}
          </ul>
        </Callout>
      )}
    </div>
  );
}
