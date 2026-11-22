/** Réglages manuels de la caméra, calibration de la balance des blancs, préréglages nommés. */
import { useEffect, useMemo, useState } from 'react';
import { Lock, Pipette, Save, Trash2, Upload } from 'lucide-react';
import { errorMessage } from '@/api/client';
import {
  Button,
  Callout,
  ConfirmDialog,
  Field,
  IconButton,
  Modal,
  NumberInput,
  Panel,
  Select,
  TextInput,
  useToast,
} from '@/components/ui';
import { formatNumber } from '@/lib/format';
import { useApplyPreset, useAwb, useDeletePreset, usePresets, useSaveSettings, useSavePreset } from './api';
import type { CameraSettings } from './types';
import s from './capture.module.css';

const same = (a: CameraSettings | null, b: CameraSettings | undefined) => !!a && !!b && JSON.stringify(a) === JSON.stringify(b);

export interface SettingsPanelProps {
  settings: CameraSettings | undefined;
  locked: boolean;
  sensor: { width: number; height: number };
  /** ROI tracée sur l'aperçu, à reporter dans le formulaire. */
  pendingRoi: [number, number, number, number] | null;
  onRoiConsumed: () => void;
  hasProject: boolean;
}

export function SettingsPanel({ settings, locked, sensor, pendingRoi, onRoiConsumed, hasProject }: SettingsPanelProps) {
  const toast = useToast();
  const save = useSaveSettings();
  const [draft, setDraft] = useState<CameraSettings | null>(settings ?? null);
  const [awbOpen, setAwbOpen] = useState(false);

  useEffect(() => {
    if (settings) setDraft(settings);
  }, [settings]);

  useEffect(() => {
    if (pendingRoi && draft) {
      setDraft({ ...draft, roi: pendingRoi });
      onRoiConsumed();
    }
  }, [pendingRoi]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!draft) return <Panel title="Réglages caméra"><p className="u-muted u-small">Chargement des réglages…</p></Panel>;

  const set = <K extends keyof CameraSettings>(k: K, v: CameraSettings[K]) => setDraft((d) => (d ? { ...d, [k]: v } : d));
  const setRoi = (i: number, v: number | null) => {
    const roi = [...(draft.roi ?? [0, 0, sensor.width, sensor.height])] as [number, number, number, number];
    roi[i] = v ?? 0;
    set('roi', roi);
  };
  const framePeriod = Math.floor(1e6 / (draft.fps || 1));
  const dirty = !same(draft, settings);
  const tooLong = draft.exposure_us > framePeriod;

  const apply = async () => {
    try {
      await save.mutateAsync(draft);
      toast.success('Réglages appliqués');
    } catch (e) {
      toast.error('Réglages refusés', errorMessage(e));
    }
  };

  return (
    <Panel
      title="Réglages caméra"
      description="Tous les automatismes (exposition, balance des blancs, gain) sont coupés pendant la capture."
      footer={
        <div className={s.settingsFoot}>
          <Button size="sm" variant="ghost" disabled={!dirty || locked} onClick={() => setDraft(settings ?? null)}>
            Annuler
          </Button>
          <Button size="sm" variant="primary" disabled={!dirty || locked || tooLong} loading={save.isPending} onClick={apply}>
            Appliquer
          </Button>
        </div>
      }
    >
      {locked && (
        <Callout tone="warning" icon={Lock} title="Réglages verrouillés">
          Une capture est en cours : les réglages sont figés et enregistrés dans les métadonnées du lot. Ils redeviennent modifiables à l’arrêt du lot.
        </Callout>
      )}
      <fieldset className={s.fieldset} disabled={locked}>
        <div className={s.fields2}>
          <Field label="Exposition" error={tooLong ? `Max. ${formatNumber(framePeriod)} µs à ${formatNumber(draft.fps, 1)} im/s` : undefined}>
            <NumberInput value={draft.exposure_us} onChange={(v) => v != null && set('exposure_us', Math.round(v))} min={20} max={1_000_000} step={50} unit="µs" disabled={locked} />
          </Field>
          <Field label="Cadence">
            <NumberInput value={draft.fps} onChange={(v) => v != null && set('fps', v)} min={1} max={60} step={1} decimals={1} unit="im/s" disabled={locked} />
          </Field>
          <Field label="Gain analogique">
            <NumberInput value={draft.analogue_gain} onChange={(v) => v != null && set('analogue_gain', v)} min={1} max={16} step={0.1} unit="×" disabled={locked} />
          </Field>
          <span />
          <Field label="Gain rouge">
            <NumberInput value={draft.red_gain} onChange={(v) => v != null && set('red_gain', v)} min={0.1} max={8} step={0.01} disabled={locked} />
          </Field>
          <Field label="Gain bleu">
            <NumberInput value={draft.blue_gain} onChange={(v) => v != null && set('blue_gain', v)} min={0.1} max={8} step={0.01} disabled={locked} />
          </Field>
        </div>
        <Button icon={Pipette} size="sm" block disabled={locked} onClick={() => setAwbOpen(true)}>
          Calibrer la balance des blancs
        </Button>
        <div className={s.roiBlock}>
          <div className={s.roiHead}>
            <span className="u-small u-strong">Zone de recadrage (ROI)</span>
            <Button size="sm" variant="ghost" disabled={locked || draft.roi === null} onClick={() => set('roi', null)}>
              Plein capteur
            </Button>
          </div>
          {draft.roi ? (
            <div className={s.fields4}>
              {(['x', 'y', 'largeur', 'hauteur'] as const).map((lbl, i) => (
                <Field key={lbl} label={lbl}>
                  <NumberInput
                    size="sm"
                    value={draft.roi![i]}
                    onChange={(v) => setRoi(i, v)}
                    min={i < 2 ? 0 : 16}
                    max={i % 2 === 0 ? sensor.width : sensor.height}
                    step={2}
                    stepper={false}
                    disabled={locked}
                  />
                </Field>
              ))}
            </div>
          ) : (
            <p className="u-xs u-muted">
              Plein capteur ({sensor.width}×{sensor.height}).{' '}
              <button type="button" className={s.linkBtn} disabled={locked} onClick={() => set('roi', [0, 0, sensor.width, sensor.height])}>
                Saisir une ROI
              </button>{' '}
              ou « Tracer la ROI » sur l’aperçu.
            </p>
          )}
        </div>
        {dirty && !locked && <p className="u-xs u-muted">Modifications non appliquées.</p>}
      </fieldset>
      {hasProject && <Presets settings={settings} draft={draft} locked={locked} />}
      <AwbDialog open={awbOpen} onClose={() => setAwbOpen(false)} />
    </Panel>
  );
}

// ---------------------------------------------------------------- Balance des blancs
function AwbDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const toast = useToast();
  const awb = useAwb();
  const run = async () => {
    try {
      const r = await awb.mutateAsync(2);
      toast.success('Balance des blancs figée', `Gain rouge ${formatNumber(r.red_gain, 2)} · gain bleu ${formatNumber(r.blue_gain, 2)}`);
      onClose();
    } catch (e) {
      toast.error('Calibration impossible', errorMessage(e));
    }
  };
  return (
    <Modal
      open={open}
      onClose={() => !awb.isPending && onClose()}
      dismissible={!awb.isPending}
      title="Calibrer la balance des blancs"
      size="sm"
      footer={
        <>
          <Button variant="ghost" onClick={onClose} disabled={awb.isPending}>
            Annuler
          </Button>
          <Button variant="primary" icon={Pipette} onClick={run} loading={awb.isPending}>
            {awb.isPending ? 'Calibration (≈ 2 s)…' : 'Lancer'}
          </Button>
        </>
      }
    >
      <ol className={s.steps}>
        <li>Placer une carte grise neutre sous la caméra, bien éclairée, sur toute la zone utile.</li>
        <li>Lancer : la balance automatique tourne environ 2 secondes.</li>
        <li>Les gains rouge et bleu obtenus sont figés (automatisme recoupé).</li>
      </ol>
    </Modal>
  );
}

// ---------------------------------------------------------------- Préréglages
function Presets({ settings, draft, locked }: { settings: CameraSettings | undefined; draft: CameraSettings; locked: boolean }) {
  const toast = useToast();
  const presets = usePresets();
  const savePreset = useSavePreset();
  const del = useDeletePreset();
  const apply = useApplyPreset();
  const [selected, setSelected] = useState('');
  const [name, setName] = useState('');
  const [confirm, setConfirm] = useState(false);
  const list = useMemo(() => presets.data ?? [], [presets.data]);

  useEffect(() => {
    if (selected && !list.some((p) => p.name === selected)) setSelected('');
  }, [list, selected]);

  const onSave = async () => {
    const n = name.trim();
    if (!n) return;
    try {
      await savePreset.mutateAsync({ name: n, settings: draft });
      toast.success(`Préréglage « ${n} » enregistré`);
      setSelected(n);
      setName('');
    } catch (e) {
      toast.error(e);
    }
  };
  const onLoad = async () => {
    try {
      await apply.mutateAsync(selected);
      toast.success(`Préréglage « ${selected} » appliqué`);
    } catch (e) {
      toast.error('Préréglage non appliqué', errorMessage(e));
    }
  };
  const current = list.find((p) => p.name === selected);

  return (
    <div className={s.presets}>
      <span className="u-small u-strong">Préréglages</span>
      <div className={s.presetRow}>
        <Select
          size="sm"
          value={selected}
          onChange={setSelected}
          placeholder={list.length ? 'Choisir un préréglage…' : 'Aucun préréglage'}
          options={list.map((p) => ({ value: p.name, label: p.name }))}
          aria-label="Préréglage"
          className="u-grow"
        />
        <Button size="sm" icon={Upload} disabled={!selected || locked} loading={apply.isPending} onClick={onLoad}>
          Charger
        </Button>
        <IconButton size="sm" icon={Trash2} label="Supprimer le préréglage" variant="ghost" disabled={!selected} onClick={() => setConfirm(true)} />
      </div>
      {current && (
        <p className="u-xs u-muted u-mono">
          {current.settings.exposure_us} µs · gain {current.settings.analogue_gain} · R {current.settings.red_gain} · B {current.settings.blue_gain} ·{' '}
          {current.settings.fps} im/s{current.settings.roi ? ` · ROI ${current.settings.roi.join(',')}` : ''}
        </p>
      )}
      <div className={s.presetRow}>
        <TextInput
          size="sm"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="Nom du nouveau préréglage"
          maxLength={100}
          onKeyDown={(e) => e.key === 'Enter' && onSave()}
          aria-label="Nom du préréglage"
          className="u-grow"
        />
        <Button size="sm" icon={Save} disabled={!name.trim()} loading={savePreset.isPending} onClick={onSave}>
          Enregistrer
        </Button>
      </div>
      {!same(draft, settings) && <p className="u-xs u-muted">Le préréglage enregistre les valeurs du formulaire.</p>}
      <ConfirmDialog
        open={confirm}
        title="Supprimer le préréglage ?"
        message={`« ${selected} » sera supprimé du projet.`}
        confirmLabel="Supprimer"
        variant="danger"
        onCancel={() => setConfirm(false)}
        onConfirm={async () => {
          try {
            await del.mutateAsync(selected);
            toast.info('Préréglage supprimé');
          } catch (e) {
            toast.error(e);
          } finally {
            setConfirm(false);
          }
        }}
      />
    </div>
  );
}
