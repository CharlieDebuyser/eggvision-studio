import { useEffect, useState } from 'react';
import { Route, Routes } from 'react-router-dom';
import { useQueryClient } from '@tanstack/react-query';
import { UserRound } from 'lucide-react';
import { useUpdateProjectConfig } from '@/api/hooks';
import { Page, PageHeader, RouteTabs, TextInput, Tooltip, useToast } from '@/components/ui';
import { lk, useLabelingMeta } from './api';
import { Dashboard } from './Dashboard';
import { Grid } from './Grid';
import { History } from './History';
import { Review } from './Review';
import { LabelingStateProvider } from './state';
import s from './labellisation.module.css';

/** Nom du relecteur (config.yaml → user.name), enregistré dans l'historique de chaque modification. */
function ReviewerField() {
  const meta = useLabelingMeta();
  const update = useUpdateProjectConfig();
  const qc = useQueryClient();
  const toast = useToast();
  const saved = meta.data?.user ?? '';
  const [draft, setDraft] = useState(saved);
  useEffect(() => setDraft(saved), [saved]);

  const commit = async () => {
    const name = draft.trim();
    if (name === saved) return;
    try {
      await update.mutateAsync({ user: { name } });
      await qc.invalidateQueries({ queryKey: lk.meta });
      toast.success('Relecteur enregistré', name ? `Les modifications seront signées « ${name} ».` : undefined);
    } catch (e) {
      toast.error(e);
    }
  };

  return (
    <Tooltip content="Nom enregistré dans l’historique de chaque modification">
      <div className={s.reviewer}>
        <TextInput
          size="sm"
          className={s.reviewerInput}
          aria-label="Relecteur"
          placeholder="Votre nom"
          leading={<UserRound size={14} />}
          value={draft}
          invalid={!meta.isLoading && !saved && !draft.trim()}
          onChange={(e) => setDraft(e.target.value)}
          onBlur={() => void commit()}
          onKeyDown={(e) => {
            if (e.key === 'Enter') (e.target as HTMLInputElement).blur();
            if (e.key === 'Escape') {
              setDraft(saved);
              (e.target as HTMLInputElement).blur();
            }
          }}
        />
      </div>
    </Tooltip>
  );
}

/** Module « labellisation » — monté sur /labellisation/*. */
export default function LabellisationPage() {
  return (
    <LabelingStateProvider>
      <Page wide>
        <PageHeader
          className={s.header}
          eyebrow="Étape 3"
          title="Labellisation"
          actions={<ReviewerField />}
        >
          <RouteTabs
            items={[
              { to: '/labellisation', label: 'Relecture', end: true },
              { to: '/labellisation/grille', label: 'Grille' },
              { to: '/labellisation/tableau-de-bord', label: 'Tableau de bord' },
              { to: '/labellisation/historique', label: 'Historique' },
            ]}
          />
        </PageHeader>
        <Routes>
          <Route index element={<Review />} />
          <Route path="piste/:trackId" element={<Review />} />
          <Route path="grille" element={<Grid />} />
          <Route path="tableau-de-bord" element={<Dashboard />} />
          <Route path="historique" element={<History />} />
          <Route path="*" element={<Review />} />
        </Routes>
      </Page>
    </LabelingStateProvider>
  );
}
