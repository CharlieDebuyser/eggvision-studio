import { Construction } from 'lucide-react';
import { MODULES } from '@/app/modules';
import { Card, EmptyState, Page, PageHeader } from '@/components/ui';

/** Page provisoire d'un module pas encore intégré (à remplacer par la vraie page). */
export function ModulePlaceholder({ path }: { path: string }) {
  const m = MODULES.find((x) => x.path === path);
  return (
    <Page>
      <PageHeader eyebrow={m ? `Étape ${m.step}` : undefined} title={m?.label ?? path} subtitle={m?.description} />
      <Card>
        <EmptyState
          icon={Construction}
          title="Module en cours d’intégration"
          description="Cette section sera disponible dans une prochaine version d’EggVision Studio."
        />
      </Card>
    </Page>
  );
}
