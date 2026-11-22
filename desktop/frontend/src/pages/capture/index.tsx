/**
 * Module « Prise de vue » — monté sur /capture/*.
 *
 *   /capture        acquisition : source, aperçu en direct + repères, réglages, lot en cours
 *   /capture/lots   lots du projet, lots restés sur l'agent Pi, synchronisations
 *
 * API : docs/api/capture.md.
 */
import { Route, Routes } from 'react-router-dom';
import { Page, PageHeader, RouteTabs } from '@/components/ui';
import { useAgentLots, useCaptureStatus } from './api';
import { Acquisition } from './Acquisition';
import { LotsView } from './Lots';

export default function CapturePage() {
  const status = useCaptureStatus();
  const isPi = status.data?.kind === 'pi';
  const agentLots = useAgentLots(isPi);
  const unsynced = isPi ? agentLots.data?.unsynced : undefined;
  return (
    <Page wide>
      <PageHeader
        eyebrow="Étape 1"
        title="Prise de vue"
        subtitle="Piloter la caméra (agent Raspberry Pi, caméra USB ou simulation), régler l'image et enregistrer des lots."
      >
        <RouteTabs
          items={[
            { to: '/capture', label: 'Acquisition', end: true },
            { to: '/capture/lots', label: 'Lots et synchronisation', count: unsynced || undefined },
          ]}
        />
      </PageHeader>
      <Routes>
        <Route index element={<Acquisition />} />
        <Route path="lots" element={<LotsView />} />
      </Routes>
    </Page>
  );
}
