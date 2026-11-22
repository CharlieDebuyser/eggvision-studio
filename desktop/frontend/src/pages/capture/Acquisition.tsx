/** Onglet « Acquisition » : source, aperçu en direct, qualité d'image, réglages et lot. */
import { useState } from 'react';
import { CameraOff } from 'lucide-react';
import { useProject } from '@/api/hooks';
import { EmptyState, ErrorState, LoadingState, Panel } from '@/components/ui';
import { useAnalysis, useCaptureStatus, useLive, useOverlays, useSettings } from './api';
import { ConnectionPanel } from './Connection';
import { ImageQuality } from './ImageQuality';
import { LivePreview } from './LivePreview';
import { LotPanel } from './LotPanel';
import { SettingsPanel } from './SettingsPanel';
import s from './capture.module.css';

export function Acquisition() {
  const status = useCaptureStatus();
  const project = useProject();
  const st = status.data;
  const connected = !!st?.connected && !!st.source;
  const live = useLive(!!st?.capturing);
  const settings = useSettings(connected);
  const overlays = useOverlays();
  const showDet = !!overlays.data?.show_detections;
  const analysis = useAnalysis(connected && !!st?.source?.ready, showDet);
  const [pendingRoi, setPendingRoi] = useState<[number, number, number, number] | null>(null);

  if (status.isLoading) return <LoadingState />;
  if (status.error || !st) return <ErrorState error={status.error} onRetry={() => status.refetch()} />;

  const src = st.source;
  const cam = src?.camera;
  const sensor = { width: cam?.sensor_width || 1456, height: cam?.sensor_height || 1088 };
  const width = cam?.output_width || sensor.width;
  const height = cam?.output_height || sensor.height;
  const locked = st.locked || !!live.data?.active;
  const sourceKey = `${st.kind}|${src?.url ?? src?.label ?? ''}`;

  return (
    <div className="u-stack-4">
      <ConnectionPanel status={st} />
      <div className={s.mainGrid}>
        <div className={s.mainCol}>
          {connected && overlays.data ? (
            <Panel flush>
              <LivePreview
                sourceKey={sourceKey}
                width={width}
                height={height}
                sensor={sensor}
                overlays={overlays.data}
                settings={settings.data}
                analysis={analysis.data}
                locked={locked}
                onRoi={setPendingRoi}
              />
            </Panel>
          ) : (
            <Panel>
              <EmptyState
                icon={CameraOff}
                title={connected ? 'Chargement de l’aperçu…' : 'Aucune source connectée'}
                description="Connecter l’agent Raspberry Pi, une caméra USB ou la simulation pour afficher l’aperçu en direct."
              />
            </Panel>
          )}
          {connected && <ImageQuality analysis={analysis.data} error={analysis.error} sourceKey={sourceKey} />}
        </div>
        <div className={s.sideCol}>
          <LotPanel status={st} live={live.data} />
          {connected && (
            <SettingsPanel
              settings={settings.data}
              locked={locked}
              sensor={sensor}
              pendingRoi={pendingRoi}
              onRoiConsumed={() => setPendingRoi(null)}
              hasProject={!!project.data?.open}
            />
          )}
        </div>
      </div>
    </div>
  );
}
