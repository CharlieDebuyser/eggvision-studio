import type { ReactNode } from 'react';
import type { LucideIcon } from 'lucide-react';
import { cx } from '@/lib/cx';
import { JOB_STATUS_META, VERDICT_BY_KEY, defectLabel, isVerdictKey, type Tone } from '@/lib/domain';
import type { JobStatus } from '@/api/types';

export interface BadgeProps {
  tone?: Tone;
  /** 'soft' (fond teinté, défaut), 'solid', 'outline'. */
  variant?: 'soft' | 'solid' | 'outline';
  icon?: LucideIcon;
  /** Pastille colorée à gauche. */
  dot?: boolean;
  children: ReactNode;
  title?: string;
  className?: string;
}

export function Badge({ tone = 'neutral', variant = 'soft', icon: Icon, dot, children, title, className }: BadgeProps) {
  return (
    <span className={cx('badge', `badge--${variant}`, `badge--${tone}`, className)} title={title}>
      {dot && <span className="badge__dot" aria-hidden />}
      {Icon && <Icon size={12} aria-hidden strokeWidth={2.2} />}
      {children}
    </span>
  );
}

/** Statut de tâche (En attente, En cours, Terminée, Échec, Annulée). */
export function StatusBadge({ status, className }: { status: JobStatus; className?: string }) {
  const meta = JOB_STATUS_META[status] ?? { label: status, tone: 'neutral' as Tone };
  return (
    <Badge tone={meta.tone} dot className={cx(status === 'running' && 'badge--pulse', className)}>
      {meta.label}
    </Badge>
  );
}

/** Verdict (Viable, Non viable, Incertain, Exclu) avec sa couleur sémantique. */
export function VerdictBadge({ verdict, className }: { verdict: string; className?: string }) {
  if (!isVerdictKey(verdict)) return <Badge className={className}>{verdict}</Badge>;
  const v = VERDICT_BY_KEY[verdict];
  return (
    <span className={cx('badge', 'badge--soft', 'badge--verdict', className)} style={{ ['--badge-color' as string]: v.color }}>
      <span className="badge__dot" aria-hidden />
      {v.label}
    </span>
  );
}

/** Étiquette de défaut (Souillure, Fêlure…). */
export function DefectBadge({ defect, className }: { defect: string; className?: string }) {
  return (
    <Badge tone="neutral" variant="outline" className={className}>
      {defectLabel(defect)}
    </Badge>
  );
}
