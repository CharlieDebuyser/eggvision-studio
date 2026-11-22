/**
 * Module « Entraînement et évaluation » — monté sur /entrainement/*.
 *
 *   /entrainement                         modèles (liste versionnée, sélection pour comparer)
 *   /entrainement/decoupages              découpages ; /nouveau (création assistée) ; /:splitId (détail, ajustement)
 *   /entrainement/nouveau?split=…         nouvel entraînement (classifieur ou anomalies)
 *   /entrainement/modeles/:modelId        suivi (courbes en direct) ; /rapport ; /erreurs
 *   /entrainement/comparer?ids=a,b        comparaison
 *
 * API : docs/api/training.md.
 */
import { Route, Routes } from 'react-router-dom';
import { Page, PageHeader, RouteTabs } from '@/components/ui';
import { ComparePage } from './Compare';
import { ModelDetailPage } from './ModelDetail';
import { ModelList } from './ModelList';
import { NewTraining } from './NewTraining';
import { SplitDetailPage, SplitEditor, SplitList } from './Splits';

const TABS = [
  { to: '/entrainement', label: 'Modèles', end: true },
  { to: '/entrainement/decoupages', label: 'Découpages' },
  { to: '/entrainement/nouveau', label: 'Nouvel entraînement' },
];

function Header({ subtitle }: { subtitle: string }) {
  return (
    <PageHeader eyebrow="Étape 4" title="Entraînement et évaluation" subtitle={subtitle}>
      <RouteTabs items={TABS} />
    </PageHeader>
  );
}

export default function EntrainementPage() {
  return (
    <Page wide>
      <Routes>
        <Route
          index
          element={
            <>
              <Header subtitle="Modèles entraînés, versionnés avec leurs données, leur découpage, leurs hyperparamètres et leurs métriques." />
              <ModelList />
            </>
          }
        />
        <Route
          path="decoupages"
          element={
            <>
              <Header subtitle="Séparation entraînement / validation / test par lot ou par session — jamais par image." />
              <SplitList />
            </>
          }
        />
        <Route
          path="decoupages/nouveau"
          element={
            <>
              <PageHeader
                breadcrumbs={[{ label: 'Entraînement', to: '/entrainement' }, { label: 'Découpages', to: '/entrainement/decoupages' }, { label: 'Nouveau' }]}
                title="Nouveau découpage"
                subtitle="Aperçu en direct : rien n'est enregistré avant validation."
              />
              <SplitEditor />
            </>
          }
        />
        <Route path="decoupages/:splitId" element={<SplitDetailPage />} />
        <Route
          path="nouveau"
          element={
            <>
              <Header subtitle="Choisir les données, le modèle et les hyperparamètres, puis suivre les courbes en direct." />
              <NewTraining />
            </>
          }
        />
        <Route path="modeles/:modelId/*" element={<ModelDetailPage />} />
        <Route path="comparer" element={<ComparePage />} />
      </Routes>
    </Page>
  );
}
