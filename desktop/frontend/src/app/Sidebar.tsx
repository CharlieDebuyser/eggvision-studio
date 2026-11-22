import { NavLink, useNavigate } from 'react-router-dom';
import { Egg, FolderOpen, ListChecks, Moon, Stethoscope, Sun } from 'lucide-react';
import { useJobs, useProject, isJobActive } from '@/api/hooks';
import { cx } from '@/lib/cx';
import { useTheme } from '@/lib/theme';
import { formatPercent } from '@/lib/format';
import { ProgressBar, Tooltip } from '@/components/ui';
import { MODULES } from './modules';

function ActiveJobs() {
  const jobs = useJobs({ limit: 50 });
  const active = (jobs.data ?? []).filter((j) => isJobActive(j.status));
  const running = active.filter((j) => j.status === 'running');
  const mean = running.length ? running.reduce((s, j) => s + j.progress, 0) / running.length : null;
  const count = active.length;

  return (
    <NavLink to="/taches" className={({ isActive }) => cx('nav-item', isActive && 'is-active')}>
      <ListChecks size={16} aria-hidden className="nav-item__icon" />
      <span className="nav-item__label">Tâches</span>
      {count > 0 && (
        <span className="nav-item__count" aria-label={`${count} tâche(s) en cours`}>
          {count}
        </span>
      )}
      {count > 0 && (
        <span className="nav-item__progress" title={mean !== null ? `Progression moyenne ${formatPercent(mean, 0)}` : 'En attente'}>
          <ProgressBar value={mean} size="xs" label="Progression des tâches en cours" />
        </span>
      )}
    </NavLink>
  );
}

export function Sidebar() {
  const project = useProject();
  const { theme, toggle } = useTheme();
  const navigate = useNavigate();
  const isOpen = !!project.data?.open;

  return (
    <aside className="sidebar">
      <div className="sidebar__brand">
        <span className="sidebar__logo" aria-hidden>
          <Egg size={16} strokeWidth={2.2} />
        </span>
        <span className="sidebar__name">
          EggVision <span className="sidebar__name-sub">Studio</span>
        </span>
      </div>

      <nav className="sidebar__nav" aria-label="Navigation principale">
        <NavLink to="/" end className={({ isActive }) => cx('nav-item', isActive && 'is-active')}>
          <FolderOpen size={16} aria-hidden className="nav-item__icon" />
          <span className="nav-item__label">Projet</span>
        </NavLink>

        <div className="sidebar__section">Parcours</div>
        {MODULES.map((m) =>
          isOpen ? (
            <NavLink key={m.path} to={`/${m.path}`} className={({ isActive }) => cx('nav-item', isActive && 'is-active')}>
              <m.icon size={16} aria-hidden className="nav-item__icon" />
              <span className="nav-item__label">{m.label}</span>
              <span className="nav-item__step">{m.step}</span>
            </NavLink>
          ) : (
            <Tooltip key={m.path} content="Ouvrez un projet pour accéder à ce module" side="right">
              <span className="nav-item is-disabled" aria-disabled="true" tabIndex={0}>
                <m.icon size={16} aria-hidden className="nav-item__icon" />
                <span className="nav-item__label">{m.label}</span>
                <span className="nav-item__step">{m.step}</span>
              </span>
            </Tooltip>
          ),
        )}
      </nav>

      <div className="sidebar__bottom">
        <ActiveJobs />
        <NavLink to="/diagnostic" className={({ isActive }) => cx('nav-item', isActive && 'is-active')}>
          <Stethoscope size={16} aria-hidden className="nav-item__icon" />
          <span className="nav-item__label">Diagnostic</span>
        </NavLink>
        <button type="button" className="nav-item" onClick={toggle} aria-label={theme === 'dark' ? 'Passer au thème clair' : 'Passer au thème sombre'}>
          {theme === 'dark' ? <Sun size={16} aria-hidden className="nav-item__icon" /> : <Moon size={16} aria-hidden className="nav-item__icon" />}
          <span className="nav-item__label">{theme === 'dark' ? 'Thème clair' : 'Thème sombre'}</span>
        </button>

        <button type="button" className={cx('sidebar__project', !isOpen && 'is-empty')} onClick={() => navigate('/')} title={project.data?.path}>
          <span className={cx('sidebar__project-dot', isOpen && 'is-open')} aria-hidden />
          <span className="sidebar__project-text">
            <span className="sidebar__project-label">{isOpen ? 'Projet ouvert' : 'Aucun projet'}</span>
            <span className="sidebar__project-name u-truncate">
              {project.isLoading ? 'Chargement…' : project.isError ? 'Serveur injoignable' : isOpen ? project.data?.name : 'Créer ou ouvrir'}
            </span>
          </span>
        </button>
      </div>
    </aside>
  );
}
