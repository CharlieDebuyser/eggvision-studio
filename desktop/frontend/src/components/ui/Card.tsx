import type { HTMLAttributes, ReactNode } from 'react';
import { cx } from '@/lib/cx';

export interface CardProps extends HTMLAttributes<HTMLDivElement> {
  /** Supprime le padding interne (pour tableau, image pleine largeur…). */
  flush?: boolean;
  /** Variante en retrait (fond surface-2, sans ombre). */
  sunken?: boolean;
}

/** Conteneur de base : fond surface, bordure, rayon. */
export function Card({ flush, sunken, className, ...rest }: CardProps) {
  return <div className={cx('card', flush && 'card--flush', sunken && 'card--sunken', className)} {...rest} />;
}

export interface PanelProps extends Omit<HTMLAttributes<HTMLElement>, 'title'> {
  title?: ReactNode;
  /** Texte d'aide sous le titre. */
  description?: ReactNode;
  /** Actions à droite de l'en-tête (boutons, filtres). */
  actions?: ReactNode;
  /** Pied de panneau (boutons de validation…). */
  footer?: ReactNode;
  /** Corps sans padding. */
  flush?: boolean;
  bodyClassName?: string;
}

/** Carte avec en-tête (titre, description, actions), corps et pied optionnels. */
export function Panel({ title, description, actions, footer, flush, className, bodyClassName, children, ...rest }: PanelProps) {
  return (
    <section className={cx('card', 'panel', className)} {...rest}>
      {(title || actions) && (
        <header className="panel__header">
          <div className="panel__heading">
            {title && <h2 className="panel__title">{title}</h2>}
            {description && <p className="panel__desc">{description}</p>}
          </div>
          {actions && <div className="panel__actions">{actions}</div>}
        </header>
      )}
      <div className={cx('panel__body', flush && 'panel__body--flush', bodyClassName)}>{children}</div>
      {footer && <footer className="panel__footer">{footer}</footer>}
    </section>
  );
}

/** Liste clé/valeur compacte (fiches d'information). */
export function KeyValue({ items, className }: { items: Array<{ label: ReactNode; value: ReactNode; mono?: boolean }>; className?: string }) {
  return (
    <dl className={cx('kv', className)}>
      {items.map((it, i) => (
        <div className="kv__row" key={i}>
          <dt className="kv__label">{it.label}</dt>
          <dd className={cx('kv__value', it.mono && 'u-mono u-break')}>{it.value}</dd>
        </div>
      ))}
    </dl>
  );
}
