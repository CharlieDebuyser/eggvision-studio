import type { ReactNode } from 'react';
import { AlertTriangle, Inbox, RotateCw, type LucideIcon } from 'lucide-react';
import { cx } from '@/lib/cx';
import { errorMessage } from '@/api/client';
import { formatPercent } from '@/lib/format';
import type { Tone } from '@/lib/domain';

// ---------------------------------------------------------------- Spinner
export function Spinner({ size = 16, className, label }: { size?: number; className?: string; label?: string }) {
  return (
    <span
      className={cx('spinner', className)}
      style={{ width: size, height: size }}
      role={label ? 'status' : undefined}
      aria-label={label}
      aria-hidden={label ? undefined : true}
    />
  );
}

/** Bloc de chargement centré (pour une page ou une carte). */
export function LoadingState({ label = 'Chargement…', compact }: { label?: string; compact?: boolean }) {
  return (
    <div className={cx('state', compact && 'state--compact')} role="status" aria-live="polite">
      <Spinner size={compact ? 16 : 22} />
      <span className="state__desc">{label}</span>
    </div>
  );
}

/** Rectangle gris animé pour réserver la place d'un contenu en chargement. */
export function Skeleton({ width, height = 14, className }: { width?: number | string; height?: number | string; className?: string }) {
  return <span className={cx('skeleton', className)} style={{ width, height }} aria-hidden />;
}

// ---------------------------------------------------------------- ProgressBar
export interface ProgressBarProps {
  /** 0..1 ; null/undefined = indéterminé (animation). */
  value?: number | null;
  tone?: Tone;
  size?: 'xs' | 'sm' | 'md';
  /** Affiche le pourcentage à droite. */
  showValue?: boolean;
  label?: string;
  className?: string;
}

export function ProgressBar({ value, tone = 'accent', size = 'sm', showValue, label, className }: ProgressBarProps) {
  const indeterminate = value === null || value === undefined || Number.isNaN(value);
  const v = indeterminate ? 0 : Math.max(0, Math.min(1, value));
  return (
    <div className={cx('progress', `progress--${size}`, className)}>
      <div
        className={cx('progress__track', indeterminate && 'is-indeterminate')}
        role="progressbar"
        aria-label={label ?? 'Progression'}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={indeterminate ? undefined : Math.round(v * 100)}
      >
        <div className={cx('progress__fill', `tone-bg--${tone}`)} style={indeterminate ? undefined : { width: `${v * 100}%` }} />
      </div>
      {showValue && <span className="progress__value u-num">{indeterminate ? '…' : formatPercent(v, 0)}</span>}
    </div>
  );
}

// ---------------------------------------------------------------- EmptyState / ErrorState
export interface EmptyStateProps {
  icon?: LucideIcon;
  title: ReactNode;
  description?: ReactNode;
  /** Boutons d'action. */
  actions?: ReactNode;
  compact?: boolean;
  className?: string;
}

export function EmptyState({ icon: Icon = Inbox, title, description, actions, compact, className }: EmptyStateProps) {
  return (
    <div className={cx('state', compact && 'state--compact', className)}>
      <span className="state__icon" aria-hidden>
        <Icon size={compact ? 18 : 22} strokeWidth={1.75} />
      </span>
      <div className="state__title">{title}</div>
      {description && <div className="state__desc">{description}</div>}
      {actions && <div className="state__actions">{actions}</div>}
    </div>
  );
}

export interface ErrorStateProps {
  /** Erreur brute (ApiError, Error…) ou message. */
  error?: unknown;
  title?: ReactNode;
  onRetry?: () => void;
  compact?: boolean;
  className?: string;
}

export function ErrorState({ error, title = 'Une erreur est survenue', onRetry, compact, className }: ErrorStateProps) {
  return (
    <div className={cx('state', 'state--error', compact && 'state--compact', className)} role="alert">
      <span className="state__icon" aria-hidden>
        <AlertTriangle size={compact ? 18 : 22} strokeWidth={1.75} />
      </span>
      <div className="state__title">{title}</div>
      {error != null && <div className="state__desc">{typeof error === 'string' ? error : errorMessage(error)}</div>}
      {onRetry && (
        <div className="state__actions">
          <button type="button" className="btn btn--secondary btn--sm" onClick={onRetry}>
            <RotateCw size={14} aria-hidden />
            <span className="btn__label">Réessayer</span>
          </button>
        </div>
      )}
    </div>
  );
}

/** Bandeau d'information inline (dans une carte ou un formulaire). */
export function Callout({
  tone = 'info',
  icon: Icon,
  title,
  children,
  className,
}: {
  tone?: Tone;
  icon?: LucideIcon;
  title?: ReactNode;
  children?: ReactNode;
  className?: string;
}) {
  return (
    <div className={cx('callout', `callout--${tone}`, className)} role={tone === 'danger' ? 'alert' : undefined}>
      {Icon && <Icon size={16} className="callout__icon" aria-hidden />}
      <div className="callout__body">
        {title && <div className="callout__title">{title}</div>}
        {children && <div className="callout__text">{children}</div>}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- Stat
export interface StatProps {
  label: ReactNode;
  value: ReactNode;
  /** Texte secondaire sous la valeur (ex. « dont 120 relues »). */
  hint?: ReactNode;
  icon?: LucideIcon;
  tone?: Tone;
  onClick?: () => void;
  className?: string;
}

/** Tuile chiffre clé. */
export function Stat({ label, value, hint, icon: Icon, tone, onClick, className }: StatProps) {
  const Tag = onClick ? 'button' : 'div';
  return (
    <Tag type={onClick ? 'button' : undefined} className={cx('stat', onClick && 'stat--clickable', className)} onClick={onClick}>
      <div className="stat__label">
        {Icon && <Icon size={14} aria-hidden className={tone ? `tone-fg--${tone}` : undefined} />}
        <span>{label}</span>
      </div>
      <div className="stat__value u-num">{value}</div>
      {hint && <div className="stat__hint">{hint}</div>}
    </Tag>
  );
}

// ---------------------------------------------------------------- Kbd
/** Touche clavier : <Kbd>1</Kbd> ou <Kbd keys={['⌘', 'S']} />. */
export function Kbd({ children, keys, className }: { children?: ReactNode; keys?: string[]; className?: string }) {
  if (keys) {
    return (
      <span className={cx('kbd-group', className)}>
        {keys.map((k, i) => (
          <kbd key={i} className="kbd">
            {k}
          </kbd>
        ))}
      </span>
    );
  }
  return <kbd className={cx('kbd', className)}>{children}</kbd>;
}
