/**
 * Module « Simulation du tri avec trappes » — monté sur /simulation/*.
 *
 *   /simulation                  liste des simulations (+ sélection pour comparer)
 *   /simulation/nouvelle         formulaire de paramètres et lancement
 *   /simulation/comparer?ids=…   comparaison côte à côte
 *   /simulation/:simId           lecteur (animation 2D) ; /resultats ; /parametres
 *
 * API : docs/api/simulation.md.
 */
import { Route, Routes } from 'react-router-dom';
import { Page, PageHeader } from '@/components/ui';
import { Compare } from './Compare';
import { NewSimulation } from './NewSimulation';
import { SimulationDetail } from './SimulationDetail';
import { SimulationList } from './SimulationList';

export default function SimulationPage() {
  return (
    <Page wide>
      <Routes>
        <Route
          index
          element={
            <>
              <PageHeader
                eyebrow="Étape 6"
                title="Simulation du tri"
                subtitle="Rejouer le jeu test avec un modèle exporté, comme sur la machine complète : colonnes, caméra, décisions, trappes."
              />
              <SimulationList />
            </>
          }
        />
        <Route
          path="nouvelle"
          element={
            <>
              <PageHeader breadcrumbs={[{ label: 'Simulations', to: '/simulation' }, { label: 'Nouvelle' }]} title="Nouvelle simulation" />
              <NewSimulation />
            </>
          }
        />
        <Route path="comparer" element={<Compare />} />
        <Route path=":simId/*" element={<SimulationDetail />} />
      </Routes>
    </Page>
  );
}
