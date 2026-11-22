/**
 * Réglages de la détection et du suivi, avec aperçu en direct sur une image d'un lot.
 * Les curseurs sont construits à partir du schéma des paramètres renvoyé par le backend.
 */
import { useEffect, useMemo, useState } from 'react';
import { Eye, EyeOff, Layers, RotateCcw, Save, SlidersHorizontal } from 'lucide-react';
import {
  Badge,
  Button,
  Callout,
  ConfirmDialog,
  EmptyState,
  ErrorState,
  Field,
  IconButton,
  LoadingState,
  NumberInput,
  Panel,
  SegmentedControl,
  Slider,
  Tabs,
  Toggle,
  useToast,
} from '@/components/ui';
import { formatNumber } from '@/lib/format';
import { useDebounced, useDetLots, useParams, usePreview, useSaveParams } from './api';
import { DetectionsOverlay, FrameStage, LanesOverlay, LotFramePicker } from './FrameStage';
import type { ParamField, ParamValue, ParamValues } from './types';
import s from './detection.module.css';

type Section = 'detection' | 'tracking';

const ENUM_LABELS: Record<string, string> = {
  classic: 'Classique',
  nn: 'Détecteur entraînable',
  fixe: 'Fixe',
  auto: 'Automatique',
};

export function Settings() {
  const toast = useToast();
  const params = useParams();
  const save = useSaveParams();
  const lots = useDetLots(0, 200);
  const lotItems = lots.data?.items ?? [];

  const [section, setSection] = useState<Section>('detection');
  const [draft, setDraft] = useState<{ detection: ParamValues; tracking: ParamValues } | null>(null);
  const [dirty, setDirty] = useState(false);
  const [advanced, setAdvanced] = useState(false);
  const [showMask, setShowMask] = useState(false);
  const [showLanes, setShowLanes] = useState(true);
  const [askDefaults, setAskDefaults] = useState(false);
  const [lotId, setLotId] = useState<string | null>(null);
  const [frame, setFrame] = useState(0);

  useEffect(() => {
    if (params.data && !dirty) setDraft({ detection: params.data.detection, tracking: params.data.tracking });
  }, [params.data, dirty]);

  useEffect(() => {
    if (lotId || !lotItems.length) return;
    const first = lotItems.find((l) => l.frame_count > 0);
    if (first) {
      setLotId(first.id);
      setFrame(Math.floor(first.frame_count / 2));
    }
  }, [lotItems, lotId]);

  const lot = lotItems.find((l) => l.id === lotId) ?? null;
  const detParams = useDebounced(draft?.detection ?? null, 200);
  const preview = usePreview(lotId && detParams ? { lot_id: lotId, frame_index: frame, params: detParams, mask: showMask, mask_width: 640 } : null);

  const groups = useMemo(() => {
    const fields = params.data?.schema[section] ?? [];
    const out = new Map<string, ParamField[]>();
    for (const f of fields) {
      if (f.name === 'nn_model') continue; // choisi dans l'onglet « Détecteur entraînable »
      if (f.advanced && !advanced) continue;
      const g = f.group || 'Général';
      out.set(g, [...(out.get(g) ?? []), f]);
    }
    return [...out.entries()];
  }, [params.data, section, advanced]);

  if (params.isLoading || lots.isLoading) return <LoadingState />;
  if (params.error) return <ErrorState error={params.error} onRetry={() => void params.refetch()} />;
  if (!draft || !params.data) return <LoadingState />;

  const setValue = (name: string, v: ParamValue) => {
    setDraft((d) => (d ? { ...d, [section]: { ...d[section], [name]: v } } : d));
    setDirty(true);
  };

  const onSave = async () => {
    try {
      await save.mutateAsync(draft);
      setDirty(false);
      toast.success('Réglages enregistrés', 'Ils s’appliqueront aux prochains traitements de lots.');
    } catch (err) {
      toast.error(err);
    }
  };

  const changed = (name: string) => draft[section][name] !== params.data!.defaults[section][name];
  const p = preview.data;

  return (
    <div className={s.splitLayout}>
      <div className={s.mainCol}>
        <Panel
          flush
          title="Aperçu en direct"
          description="Les ellipses suivent les réglages en cours, avant même leur enregistrement. Pointillés : œuf tronqué par le bord."
          actions={
            <div className="u-row">
              <IconButton icon={Layers} label={showMask ? 'Masquer la segmentation' : 'Afficher la segmentation'} size="sm" active={showMask} onClick={() => setShowMask((v) => !v)} />
              <IconButton icon={showLanes ? Eye : EyeOff} label={showLanes ? 'Masquer les couloirs' : 'Afficher les couloirs'} size="sm" active={showLanes} onClick={() => setShowLanes((v) => !v)} />
            </div>
          }
        >
          <div className={s.stagePad}>
            {!lot ? (
              <EmptyState icon={SlidersHorizontal} title="Aucun lot avec des images" description="Enregistrez un lot pour régler la détection." compact />
            ) : (
              <FrameStage
                lotId={lot.id}
                frameIndex={frame}
                width={p?.width ?? lot.width}
                height={p?.height ?? lot.height}
                thumb={1200}
                overlayImage={showMask ? p?.mask_png : null}
              >
                {showLanes && p && <LanesOverlay lanes={p.scene.lanes} width={p.width} dim />}
                {p && <DetectionsOverlay detections={p.detections} />}
              </FrameStage>
            )}
          </div>
          <div className={s.previewStats}>
            {preview.error ? (
              <Callout tone="danger" title="Aperçu impossible">
                {(preview.error as Error).message}
              </Callout>
            ) : p ? (
              <>
                <Badge tone="accent">{formatNumber(p.count)} œufs</Badge>
                <Badge tone="neutral">{formatNumber(p.complete)} complets</Badge>
                <Badge tone={p.detections.some((d) => d.lane < 0) ? 'warning' : 'neutral'}>
                  {formatNumber(p.detections.filter((d) => d.lane < 0).length)} hors couloir
                </Badge>
                <span className="u-small u-muted u-num">
                  {formatNumber(p.elapsed_ms, 1)} ms · seuil {formatNumber(p.threshold, 0)}
                </span>
                {preview.isFetching && <span className="u-xs u-subtle">mise à jour…</span>}
              </>
            ) : (
              <span className="u-small u-muted">Calcul de l’aperçu…</span>
            )}
          </div>
          <div className={s.stageFooter}>
            <LotFramePicker
              lots={lotItems}
              lotId={lotId}
              onLot={(id) => {
                const l = lotItems.find((x) => x.id === id);
                setLotId(id);
                setFrame(l ? Math.floor(l.frame_count / 2) : 0);
              }}
              frame={frame}
              onFrame={setFrame}
              frameCount={lot?.frame_count ?? 1}
            />
          </div>
        </Panel>
      </div>

      <div className={s.sideCol}>
        <Panel
          flush
          title="Paramètres"
          actions={dirty ? <Badge tone="warning" dot>Non enregistrés</Badge> : <Badge tone="success" dot>Enregistrés</Badge>}
          footer={
            <div className="u-row">
              <Button variant="ghost" icon={RotateCcw} onClick={() => setAskDefaults(true)}>
                Valeurs par défaut
              </Button>
              <span className="u-spacer" />
              <Button variant="ghost" disabled={!dirty} onClick={() => setDirty(false)}>
                Annuler
              </Button>
              <Button variant="primary" icon={Save} onClick={onSave} loading={save.isPending} disabled={!dirty}>
                Enregistrer
              </Button>
            </div>
          }
        >
          <div className={s.paramsHeader}>
            <Tabs<Section>
              value={section}
              onChange={setSection}
              items={[
                { key: 'detection', label: 'Détection' },
                { key: 'tracking', label: 'Suivi et vues' },
              ]}
            />
            <Toggle size="sm" checked={advanced} onChange={setAdvanced} label="Réglages avancés" />
          </div>
          <div className={s.paramsBody}>
            {section === 'tracking' && (
              <p className="u-small u-muted">Ces réglages agissent sur le traitement des lots (pas sur l’aperçu d’une image).</p>
            )}
            {groups.map(([group, fields]) => (
              <fieldset key={group} className={s.paramGroup}>
                <legend>{group}</legend>
                {fields.map((f) => (
                  <ParamInput key={f.name} field={f} value={draft[section][f.name]} onChange={(v) => setValue(f.name, v)} changed={changed(f.name)} />
                ))}
              </fieldset>
            ))}
          </div>
        </Panel>
      </div>

      <ConfirmDialog
        open={askDefaults}
        title="Revenir aux valeurs par défaut ?"
        message="Tous les paramètres de détection et de suivi reprennent leur valeur par défaut. Rien n’est enregistré tant que vous ne cliquez pas sur « Enregistrer »."
        confirmLabel="Valeurs par défaut"
        onCancel={() => setAskDefaults(false)}
        onConfirm={() => {
          const d = params.data!.defaults;
          setDraft({ detection: { ...d.detection, nn_model: draft.detection.nn_model }, tracking: { ...d.tracking } });
          setDirty(true);
          setAskDefaults(false);
        }}
      />
    </div>
  );
}

function ParamInput({ field: f, value, onChange, changed }: { field: ParamField; value: ParamValue; onChange: (v: ParamValue) => void; changed: boolean }) {
  const label = (
    <span className="u-row">
      {f.label}
      {changed && <span className={s.changedDot} title="Différent de la valeur par défaut" />}
    </span>
  );
  if (f.type === 'boolean') {
    return <Toggle checked={!!value} onChange={onChange} label={f.label} description={f.description !== f.label ? f.description : undefined} size="sm" />;
  }
  if (f.type === 'enum') {
    return (
      <Field label={label} hint={f.description !== f.label ? f.description : undefined}>
        <SegmentedControl<string>
          size="sm"
          value={String(value)}
          onChange={onChange}
          options={(f.options ?? []).map((o) => ({ value: o, label: ENUM_LABELS[o] ?? o }))}
        />
      </Field>
    );
  }
  const hint = f.description !== f.label ? f.description : undefined;
  const num = value === null || value === undefined ? null : Number(value);
  const step = f.step ?? (f.type === 'integer' ? 1 : 0.1);
  const slider = f.min !== undefined && f.max !== undefined && (f.max - f.min) / step <= 2000 && !f.nullable;
  return (
    <Field label={label} hint={hint}>
      {slider ? (
        <Slider value={num ?? f.min!} onChange={onChange} min={f.min} max={f.max} step={step} unit={f.unit} aria-label={f.label} />
      ) : (
        <NumberInput value={num} onChange={onChange} min={f.min} max={f.max} step={step} unit={f.unit} allowEmpty={f.nullable} placeholder={f.nullable ? 'automatique' : undefined} size="sm" />
      )}
    </Field>
  );
}

export default Settings;
