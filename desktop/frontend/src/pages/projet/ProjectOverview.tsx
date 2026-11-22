import { Link, useNavigate } from 'react-router-dom';
import { ArrowRight, BrainCircuit, Camera, CheckCheck, FolderX, Images, Route as RouteIcon, RotateCw, Split } from 'lucide-react';
import { useCloseProject, useJobs, useProject } from '@/api/hooks';
import type { ProjectInfo, ProjectStats } from '@/api/types';
import { MODULES } from '@/app/modules';
import { cx } from '@/lib/cx';
import { formatDateTime, formatNumber, formatPercent, formatRelative } from '@/lib/format';
import { jobKindLabel } from '@/lib/domain';
import { Badge, Button, EmptyState, IconButton, Page, PageHeader, Panel, Stat, StatusBadge, useToast } from '@/components/ui';
import s from './projet.module.css';

/** Indication d'avancement affichée sur chaque étape du parcours. */
function stepHint(path: string, st: ProjectStats): string {
  switch (path) {
    case 'capture':
      return st.lots ? `${formatNumber(st.lots)} lot(s), ${formatNumber(st.frames)} images` : 'Aucun lot';
    case 'detection':
      return st.tracks ? `${formatNumber(st.tracks)} pistes` : 'Aucune piste';
    case 'labellisation':
      return st.tracks ? `${formatNumber(st.tracks_reviewed)} / ${formatNumber(st.tracks)} relues` : 'Rien à relire';
    case 'entrainement':
      return st.models ? `${formatNumber(st.models)} modèle(s)` : 'Aucun modèle';
    case 'export':
      return st.models ? 'Modèle disponible' : 'Nécessite un modèle';
    case 'simulation':
      return st.simulations ? `${formatNumber(st.simulations)} simulation(s)` : 'Aucune simulation';
    default:
      return '';
  }
}

/** Première étape qui reste à faire (export : jamais suggéré, son état n'est pas connu). */
function suggestedStep(st: ProjectStats): string | null {
  if (!st.lots) return 'capture';
  if (!st.tracks) return 'detection';
  if (!st.tracks_reviewed) return 'labellisation';
  if (!st.models) return 'entrainement';
  if (!st.simulations) return 'simulation';
  return null;
}

const EMPTY_STATS: ProjectStats = { lots: 0, frames: 0, tracks: 0, tracks_reviewed: 0, models: 0, simulations: 0 };

export function ProjectOverview({ info }: { info: ProjectInfo }) {
  const navigate = useNavigate();
  const toast = useToast();
  const project = useProject();
  const close = useCloseProject();
  const jobs = useJobs({ limit: 6 });
  const st = info.stats ?? EMPTY_STATS;
  const next = suggestedStep(st);

  const onClose = async () => {
    try {
      await close.mutateAsync();
      toast.info('Projet fermé');
    } catch (e) {
      toast.error(e);
    }
  };

  return (
    <Page>
      <PageHeader
        eyebrow="Projet"
        title={info.name ?? 'Projet'}
        subtitle={
          <span className={s.subtitle}>
            <span className="u-mono u-break">{info.path}</span>
            {info.created_at && <span className="u-subtle">· créé le {formatDateTime(info.created_at)}</span>}
          </span>
        }
        actions={
          <>
            <IconButton icon={RotateCw} label="Rafraîchir les compteurs" variant="secondary" loading={project.isFetching} onClick={() => project.refetch()} />
            <Button icon={FolderX} onClick={onClose} loading={close.isPending}>
              Fermer le projet
            </Button>
          </>
        }
      />

      <div className={s.stats}>
        <Stat label="Lots" value={formatNumber(st.lots)} icon={Camera} onClick={() => navigate('/capture')} />
        <Stat label="Images" value={formatNumber(st.frames)} icon={Images} />
        <Stat label="Pistes" value={formatNumber(st.tracks)} icon={RouteIcon} onClick={() => navigate('/detection')} />
        <Stat
          label="Pistes relues"
          value={formatNumber(st.tracks_reviewed)}
          hint={st.tracks ? `${formatPercent(st.tracks_reviewed / st.tracks, 0)} des pistes` : undefined}
          icon={CheckCheck}
          onClick={() => navigate('/labellisation')}
        />
        <Stat label="Modèles" value={formatNumber(st.models)} icon={BrainCircuit} onClick={() => navigate('/entrainement')} />
        <Stat label="Simulations" value={formatNumber(st.simulations)} icon={Split} onClick={() => navigate('/simulation')} />
      </div>

      <section className={s.section} aria-labelledby="parcours-title">
        <div className={s.sectionHead}>
          <h2 id="parcours-title" className={s.sectionTitle}>
            Parcours
          </h2>
          <span className="u-small u-muted">De la prise de vue à la simulation du tri, en six étapes.</span>
        </div>
        <ol className={s.steps}>
          {MODULES.map((m) => {
            const isNext = m.path === next;
            return (
              <li key={m.path}>
                <Link to={`/${m.path}`} className={cx(s.step, isNext && s.stepNext)}>
                  <span className={s.stepTop}>
                    <span className={s.stepNum}>{m.step}</span>
                    <m.icon size={17} aria-hidden className={s.stepIcon} />
                    {isNext && (
                      <Badge tone="accent" className={s.stepBadge}>
                        Étape suggérée
                      </Badge>
                    )}
                  </span>
                  <span className={s.stepLabel}>{m.label}</span>
                  <span className={s.stepDesc}>{m.description}</span>
                  <span className={s.stepFoot}>
                    <span className="u-xs u-muted">{stepHint(m.path, st)}</span>
                    <ArrowRight size={14} aria-hidden className={s.stepArrow} />
                  </span>
                </Link>
              </li>
            );
          })}
        </ol>
      </section>

      <Panel
        title="Tâches récentes"
        flush
        actions={
          <Button size="sm" variant="ghost" iconRight={ArrowRight} onClick={() => navigate('/taches')}>
            Toutes les tâches
          </Button>
        }
      >
        {jobs.data && jobs.data.length > 0 ? (
          <ul className={s.jobList}>
            {jobs.data.map((j) => (
              <li key={j.id}>
                <Link to={`/taches?job=${encodeURIComponent(j.id)}`} className={s.jobItem}>
                  <span className="u-truncate u-grow">{j.title}</span>
                  <span className="u-xs u-subtle">{jobKindLabel(j.kind)}</span>
                  <span className={cx('u-xs u-subtle u-num', s.jobWhen)}>{formatRelative(j.created_at)}</span>
                  <StatusBadge status={j.status} />
                </Link>
              </li>
            ))}
          </ul>
        ) : (
          <EmptyState title="Aucune tâche pour l’instant" description="Les traitements longs (détection, entraînement, export…) apparaîtront ici." compact />
        )}
      </Panel>
    </Page>
  );
}
