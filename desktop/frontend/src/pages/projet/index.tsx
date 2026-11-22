import { useProject } from '@/api/hooks';
import { ErrorState, LoadingState, Page, PageHeader } from '@/components/ui';
import { ProjectOverview } from './ProjectOverview';
import { ProjectWelcome } from './ProjectWelcome';

/** Accueil : création/ouverture de projet, ou résumé du projet ouvert. */
export default function ProjetPage() {
  const project = useProject();

  if (project.isLoading) return <LoadingState label="Connexion au serveur local…" />;
  if (project.isError || !project.data)
    return (
      <Page>
        <PageHeader title="Projet" />
        <ErrorState title="Le serveur local ne répond pas" error={project.error} onRetry={() => project.refetch()} />
      </Page>
    );

  return project.data.open ? <ProjectOverview info={project.data} /> : <ProjectWelcome />;
}
