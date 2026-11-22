import { useRef, type KeyboardEvent, type ReactNode } from 'react';
import { NavLink } from 'react-router-dom';
import { cx } from '@/lib/cx';

export interface TabItem<K extends string = string> {
  key: K;
  label: ReactNode;
  /** Compteur affiché à droite du libellé. */
  count?: number;
  disabled?: boolean;
}

export interface TabsProps<K extends string> {
  items: TabItem<K>[];
  value: K;
  onChange: (key: K) => void;
  className?: string;
  'aria-label'?: string;
}

/** Onglets contrôlés (état local). Flèches gauche/droite pour naviguer. */
export function Tabs<K extends string>({ items, value, onChange, className, ...rest }: TabsProps<K>) {
  const refs = useRef<(HTMLButtonElement | null)[]>([]);
  const onKey = (e: KeyboardEvent, idx: number) => {
    if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
    e.preventDefault();
    const dir = e.key === 'ArrowRight' ? 1 : -1;
    for (let i = 1; i <= items.length; i++) {
      const j = (idx + dir * i + items.length) % items.length;
      if (!items[j].disabled) {
        refs.current[j]?.focus();
        onChange(items[j].key);
        break;
      }
    }
  };
  return (
    <div className={cx('tabs', className)} role="tablist" aria-label={rest['aria-label']}>
      {items.map((t, i) => {
        const selected = t.key === value;
        return (
          <button
            key={t.key}
            ref={(el) => {
              refs.current[i] = el;
            }}
            type="button"
            role="tab"
            aria-selected={selected}
            tabIndex={selected ? 0 : -1}
            disabled={t.disabled}
            className={cx('tabs__tab', selected && 'is-active')}
            onClick={() => onChange(t.key)}
            onKeyDown={(e) => onKey(e, i)}
          >
            {t.label}
            {t.count !== undefined && <span className="tabs__count u-num">{t.count}</span>}
          </button>
        );
      })}
    </div>
  );
}

export interface RouteTabItem {
  /** Chemin relatif ou absolu passé à NavLink. */
  to: string;
  label: ReactNode;
  count?: number;
  /** Correspondance exacte (défaut true pour l'onglet index ''). */
  end?: boolean;
}

/**
 * Onglets liés aux sous-routes d'un module :
 *   <RouteTabs items={[{ to: '', label: 'Lots', end: true }, { to: 'reglages', label: 'Réglages' }]} />
 * Les chemins relatifs sont résolus par rapport à la route du module.
 */
export function RouteTabs({ items, className }: { items: RouteTabItem[]; className?: string }) {
  return (
    <nav className={cx('tabs', className)}>
      {items.map((t) => (
        <NavLink key={t.to} to={t.to} end={t.end ?? t.to === ''} className={({ isActive }) => cx('tabs__tab', isActive && 'is-active')}>
          {t.label}
          {t.count !== undefined && <span className="tabs__count u-num">{t.count}</span>}
        </NavLink>
      ))}
    </nav>
  );
}
