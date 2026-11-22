import { useEffect, type ReactNode } from 'react';
import { Link } from 'react-router-dom';
import { ChevronRight } from 'lucide-react';
import { cx } from '@/lib/cx';

export interface Crumb {
  label: ReactNode;
  to?: string;
}

export interface PageHeaderProps {
  title: ReactNode;
  subtitle?: ReactNode;
  /** Boutons à droite. */
  actions?: ReactNode;
  /** Fil d'Ariane au-dessus du titre (sous-pages de module). */
  breadcrumbs?: Crumb[];
  /** Petit libellé au-dessus du titre (ex. « Étape 2 »). */
  eyebrow?: ReactNode;
  /** Contenu sous le titre (onglets, filtres). */
  children?: ReactNode;
  /** Titre de la fenêtre ; défaut = title s'il est une chaîne. */
  documentTitle?: string;
  className?: string;
}

/** En-tête de page : à placer en premier enfant de chaque page. */
export function PageHeader({ title, subtitle, actions, breadcrumbs, eyebrow, children, documentTitle, className }: PageHeaderProps) {
  const docTitle = documentTitle ?? (typeof title === 'string' ? title : undefined);
  useEffect(() => {
    if (docTitle) document.title = `${docTitle} · EggVision Studio`;
  }, [docTitle]);

  return (
    <header className={cx('page-header', className)}>
      {breadcrumbs && breadcrumbs.length > 0 && (
        <nav className="page-header__crumbs" aria-label="Fil d’Ariane">
          {breadcrumbs.map((c, i) => (
            <span key={i} className="page-header__crumb">
              {c.to ? <Link to={c.to}>{c.label}</Link> : <span>{c.label}</span>}
              {i < breadcrumbs.length - 1 && <ChevronRight size={12} aria-hidden />}
            </span>
          ))}
        </nav>
      )}
      <div className="page-header__row">
        <div className="page-header__text">
          {eyebrow && <div className="page-header__eyebrow">{eyebrow}</div>}
          <h1 className="page-header__title">{title}</h1>
          {subtitle && <p className="page-header__subtitle">{subtitle}</p>}
        </div>
        {actions && <div className="page-header__actions">{actions}</div>}
      </div>
      {children && <div className="page-header__extra">{children}</div>}
    </header>
  );
}

/** Conteneur standard d'une page (marges, largeur max). */
export function Page({ children, className, wide }: { children: ReactNode; className?: string; wide?: boolean }) {
  return <div className={cx('page', wide && 'page--wide', className)}>{children}</div>;
}
