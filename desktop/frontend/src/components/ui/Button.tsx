import { forwardRef, type ButtonHTMLAttributes, type ReactNode } from 'react';
import type { LucideIcon } from 'lucide-react';
import { cx } from '@/lib/cx';
import { Spinner } from './Feedback';
import { Tooltip } from './Tooltip';

export type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'danger' | 'accent';
export type ButtonSize = 'sm' | 'md' | 'lg';

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  /** Affiche un spinner et désactive le bouton. */
  loading?: boolean;
  /** Icône lucide affichée à gauche (ou à droite avec iconRight). */
  icon?: LucideIcon;
  iconRight?: LucideIcon;
  /** Occupe toute la largeur. */
  block?: boolean;
}

const ICON_SIZE: Record<ButtonSize, number> = { sm: 14, md: 15, lg: 17 };

/** Bouton standard. `type="button"` par défaut (passer type="submit" dans un formulaire). */
export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { variant = 'secondary', size = 'md', loading, icon: Icon, iconRight: IconRight, block, className, children, disabled, type, ...rest },
  ref,
) {
  return (
    <button
      ref={ref}
      type={type ?? 'button'}
      className={cx('btn', `btn--${variant}`, `btn--${size}`, block && 'btn--block', loading && 'is-loading', className)}
      disabled={disabled || loading}
      aria-busy={loading || undefined}
      {...rest}
    >
      {loading ? <Spinner size={ICON_SIZE[size]} /> : Icon ? <Icon size={ICON_SIZE[size]} aria-hidden strokeWidth={2} /> : null}
      {children != null && <span className="btn__label">{children}</span>}
      {IconRight && !loading && <IconRight size={ICON_SIZE[size]} aria-hidden strokeWidth={2} />}
    </button>
  );
});

export interface IconButtonProps extends Omit<ButtonHTMLAttributes<HTMLButtonElement>, 'children'> {
  icon: LucideIcon;
  /** Libellé accessible, aussi affiché en infobulle. Obligatoire. */
  label: string;
  variant?: ButtonVariant;
  size?: ButtonSize;
  loading?: boolean;
  /** Désactive l'infobulle (le libellé reste accessible). */
  noTooltip?: boolean;
  active?: boolean;
}

/** Bouton carré à icône seule ; `label` sert d'aria-label et d'infobulle. */
export const IconButton = forwardRef<HTMLButtonElement, IconButtonProps>(function IconButton(
  { icon: Icon, label, variant = 'ghost', size = 'md', loading, noTooltip, active, className, disabled, type, ...rest },
  ref,
) {
  const btn = (
    <button
      ref={ref}
      type={type ?? 'button'}
      aria-label={label}
      aria-pressed={active}
      className={cx('btn', 'btn--icon', `btn--${variant}`, `btn--${size}`, active && 'is-active', className)}
      disabled={disabled || loading}
      {...rest}
    >
      {loading ? <Spinner size={ICON_SIZE[size]} /> : <Icon size={ICON_SIZE[size]} aria-hidden strokeWidth={2} />}
    </button>
  );
  return noTooltip ? btn : <Tooltip content={label}>{btn}</Tooltip>;
});

/** Groupe de boutons accolés. */
export function ButtonGroup({ children, className }: { children: ReactNode; className?: string }) {
  return <div className={cx('btn-group', className)}>{children}</div>;
}
