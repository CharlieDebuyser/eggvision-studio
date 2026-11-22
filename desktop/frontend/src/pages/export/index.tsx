/**
 * Module « Export vers Raspberry Pi » — monté sur /export/*.
 *
 *   /export            modèles exportables et état de leur paquet
 *   /export/:modelId   export (FP32 / INT8, parité, paquet, téléchargement) et banc de mesure (Pi / PC)
 *
 * API : docs/api/export.md.
 */
import { Route, Routes } from 'react-router-dom';
import { Page, PageHeader } from '@/components/ui';
import { ExportDetail } from './ExportDetail';
import { ExportList } from './ExportList';

export default function ExportPage() {
  return (
    <Page wide>
      <Routes>
        <Route
          index
          element={
            <>
              <PageHeader
                eyebrow="Étape 5"
                title="Export vers Raspberry Pi"
                subtitle="Modèle ONNX (FP32 et INT8) vérifié contre PyTorch, paquet de déploiement autonome et mesure de latence sur le Pi."
              />
              <ExportList />
            </>
          }
        />
        <Route path=":modelId" element={<ExportDetail />} />
      </Routes>
    </Page>
  );
}
