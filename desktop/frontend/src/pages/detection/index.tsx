/**
 * Module « Détection et suivi » — monté sur /detection/*.
 *
 *   /detection               Traitement : lots, lancement, progression
 *   /detection/calibration   Calibration de la scène (couloirs, sens, échelle)
 *   /detection/reglages      Réglages avec aperçu en direct
 *   /detection/pistes        Parcours des pistes d'un lot (?lot=<id>)
 *   /detection/detecteur     Détecteur entraînable (option)
 *
 * API : docs/api/detection.md.
 */
import { Route, Routes } from 'react-router-dom';
import { Page, PageHeader, RouteTabs } from '@/components/ui';
import { Calibration } from './Calibration';
import { Detector } from './Detector';
import { Processing } from './Processing';
import { Settings } from './Settings';
import { Tracks } from './Tracks';

export default function DetectionPage() {
  return (
    <Page wide>
      <PageHeader
        eyebrow="Étape 2"
        title="Détection et suivi"
        subtitle="Détecter les œufs sur la table à rouleaux, les suivre d’image en image et extraire leurs vues recadrées."
      >
        <RouteTabs
          items={[
            { to: '/detection', label: 'Traitement', end: true },
            { to: '/detection/calibration', label: 'Calibration' },
            { to: '/detection/reglages', label: 'Réglages' },
            { to: '/detection/pistes', label: 'Pistes' },
            { to: '/detection/detecteur', label: 'Détecteur entraînable' },
          ]}
        />
      </PageHeader>
      <Routes>
        <Route index element={<Processing />} />
        <Route path="calibration" element={<Calibration />} />
        <Route path="reglages" element={<Settings />} />
        <Route path="pistes" element={<Tracks />} />
        <Route path="detecteur" element={<Detector />} />
      </Routes>
    </Page>
  );
}
