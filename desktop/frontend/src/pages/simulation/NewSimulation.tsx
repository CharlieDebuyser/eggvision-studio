/** Formulaire « Nouvelle simulation » : paramètres regroupés, budget de temps calculé en direct, lancement. */
import { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Play, RotateCcw } from 'lucide-react';
import { defectLabel } from '@/lib/domain';
import { formatDuration, formatNumber } from '@/lib/format';
import {
  Button,
  Callout,
  EmptyState,
  ErrorState,
  Field,
  LoadingState,
  NumberInput,
  Panel,
  SegmentedControl,
  Select,
  TextInput,
  useToast,
} from '@/components/ui';
import { useCreateSim, useSimDefaults, useSimModels, useTiming } from './api';
import type { SimParams, Variant } from './types';
import s from './simulation.module.css';

type Form = Omit<SimParams, 'model_id'> & { model_id: string };
type SpeedMode = 'pitch' | 'speed';

export function NewSimulation() {
  const defaults = useSimDefaults();
  const models = useSimModels();
  if (defaults.isLoading || models.isLoading) return <LoadingState />;
  if (defaults.error || models.error) return <ErrorState error={defaults.error ?? models.error} onRetry={() => (defaults.refetch(), models.refetch())} />;
  const eligible = (models.data ?? []).filter((m) => m.eligible);
  if (!eligible.length)
    return (
      <EmptyState
        title="Aucun modèle exporté utilisable"
        description={
          (models.data ?? []).length
            ? (models.data ?? []).map((m) => `${m.name} : ${m.reason}`).join(' · ')
            : "La simulation rejoue le jeu test avec un modèle exporté (ONNX). Entraînez puis exportez un modèle d'abord."
        }
      />
    );
  return <SimForm defaults={defaults.data!.params} models={models.data!} />;
}

function SimForm({ defaults, models }: { defaults: Omit<SimParams, 'model_id'>; models: NonNullable<ReturnType<typeof useSimModels>['data']> }) {
  const navigate = useNavigate();
  const toast = useToast();
  const create = useCreateSim();
  const eligible = models.filter((m) => m.eligible);
  const initial: Form = { ...defaults, model_id: eligible[0].id };
  const [f, setF] = useState<Form>(initial);
  const [name, setName] = useState('');
  const [speedMode, setSpeedMode] = useState<SpeedMode>('pitch');
  const [speedInput, setSpeedInput] = useState<number | null>(defaults.pitch_mm / defaults.period_s);
  const set = <K extends keyof Form>(k: K, v: Form[K]) => setF((prev) => ({ ...prev, [k]: v }));
  const model = eligible.find((m) => m.id === f.model_id) ?? eligible[0];

  // Variante disponible dans le paquet choisi.
  useEffect(() => {
    if (f.variant && !model.variants.includes(f.variant)) set('variant', null);
  }, [model, f.variant]);

  const payload = useMemo(() => {
    const p: Partial<SimParams> = { ...f };
    if (speedMode === 'speed' && speedInput) p.speed_mm_s = speedInput;
    return p;
  }, [f, speedMode, speedInput]);
  const timing = useTiming(payload);
  const t = timing.data;

  const launch = async () => {
    try {
      const res = await create.mutateAsync({ ...payload, name: name.trim() || undefined });
      toast.success('Simulation lancée', res.simulation.name);
      navigate(`/simulation/${encodeURIComponent(res.simulation.id)}`);
    } catch (e) {
      toast.error(e);
    }
  };

  const pct = (v: number) => Math.round(v * 1000) / 10;
  const variantOptions: { value: Variant | 'auto'; label: string; disabled?: boolean }[] = [
    { value: 'auto', label: 'Défaut du paquet' },
    { value: 'fp32', label: 'FP32', disabled: !model.variants.includes('fp32') },
    { value: 'int8', label: 'INT8', disabled: !model.variants.includes('int8') },
  ];

  return (
    <div className={s.formGrid}>
      <div className="u-stack-4">
        <Panel title="Modèle">
          <div className={s.fields3}>
            <Field label="Modèle exporté" required>
              <Select
                value={f.model_id}
                onChange={(v) => set('model_id', v)}
                options={models.map((m) => ({ value: m.id, label: m.eligible ? m.name : `${m.name} (${m.reason})`, disabled: !m.eligible }))}
              />
            </Field>
            <Field label="Variante">
              <Select<Variant | 'auto'> value={f.variant ?? 'auto'} onChange={(v) => set('variant', v === 'auto' ? null : v)} options={variantOptions} />
            </Field>
            <Field label="Seuil de décision" hint={`vide = seuil du modèle (${formatNumber(model.threshold, 3)})`}>
              <NumberInput value={f.threshold} onChange={(v) => set('threshold', v)} min={0} max={1} step={0.01} decimals={3} allowEmpty placeholder={formatNumber(model.threshold, 3)} />
            </Field>
          </div>
          {model.test_set && (
            <p className="u-muted u-small">
              Jeu test : {formatNumber(model.test_set.n_viable)} viables, {formatNumber(model.test_set.n_non_viable)} non viables
              {Object.keys(model.test_set.by_defect).length > 0 &&
                ` (${Object.entries(model.test_set.by_defect)
                  .map(([d, n]) => `${defectLabel(d)} ${n}`)
                  .join(', ')})`}{' '}
              — {formatNumber(model.test_set.n_test_lots)} lot(s). Seules ces pistes sont tirées, jamais celles d'entraînement ou de validation.
            </p>
          )}
        </Panel>

        <Panel title="Table virtuelle">
          <div className={s.fields3}>
            <Field label="Couloirs">
              <NumberInput value={f.n_lanes} onChange={(v) => set('n_lanes', v ?? 6)} min={1} max={16} step={1} />
            </Field>
            <Field label="Cadence" hint="une colonne toutes les…">
              <NumberInput value={f.period_s} onChange={(v) => set('period_s', v ?? 0.5)} min={0.02} max={60} step={0.05} decimals={2} unit="s" />
            </Field>
            <Field label="Saisie" hint="pas entre colonnes ou vitesse">
              <SegmentedControl<SpeedMode>
                value={speedMode}
                onChange={setSpeedMode}
                options={[
                  { value: 'pitch', label: 'Pas' },
                  { value: 'speed', label: 'Vitesse' },
                ]}
              />
            </Field>
            {speedMode === 'pitch' ? (
              <Field label="Pas entre colonnes">
                <NumberInput value={f.pitch_mm} onChange={(v) => set('pitch_mm', v ?? 76)} min={2} max={2000} step={1} unit="mm" />
              </Field>
            ) : (
              <Field label="Vitesse de défilement" hint={t ? `pas résultant : ${formatNumber(t.pitch_mm, 1)} mm` : undefined}>
                <NumberInput value={speedInput} onChange={setSpeedInput} min={1} max={10000} step={1} unit="mm/s" />
              </Field>
            )}
            <Field label="Zone caméra" hint="longueur où les vues sont présentées">
              <NumberInput value={f.camera_zone_mm} onChange={(v) => set('camera_zone_mm', v ?? 152)} min={1} max={5000} step={1} unit="mm" />
            </Field>
            <Field label="Distance caméra → trappes">
              <NumberInput value={f.camera_to_trap_mm} onChange={(v) => set('camera_to_trap_mm', v ?? 600)} min={0} max={20000} step={10} unit="mm" />
            </Field>
            <Field label="Temps d'ouverture des trappes" hint="à anticiper avant l'arrivée">
              <NumberInput value={f.trap_actuation_ms} onChange={(v) => set('trap_actuation_ms', v ?? 40)} min={0} max={10000} step={5} unit="ms" />
            </Field>
          </div>
        </Panel>

        <Panel title="Constitution des colonnes">
          <div className={s.fields3}>
            <Field label="Nombre de colonnes">
              <NumberInput value={f.n_columns} onChange={(v) => set('n_columns', v ?? 500)} min={1} max={50000} step={100} />
            </Field>
            <Field label="Proportion de défauts" hint="5 % = production, 30 % = test de charge">
              <NumberInput value={pct(f.defect_rate)} onChange={(v) => set('defect_rate', (v ?? 5) / 100)} min={0} max={100} step={1} decimals={1} unit="%" />
            </Field>
            <Field label="Graine aléatoire" hint="vide = tirée au hasard (enregistrée)">
              <NumberInput value={f.seed} onChange={(v) => set('seed', v)} min={0} max={2147483647} step={1} allowEmpty placeholder="aléatoire" />
            </Field>
            <Field label="Couloirs vides" hint="colonnes incomplètes">
              <NumberInput value={pct(f.empty_lane_rate)} onChange={(v) => set('empty_lane_rate', (v ?? 0) / 100)} min={0} max={95} step={1} decimals={1} unit="%" />
            </Field>
          </div>
        </Panel>

        <Panel title="Robustesse" description="Éprouver le système : latence artificielle, vues manquantes, cible plus lente.">
          <div className={s.fields3}>
            <Field label="Latence injectée">
              <NumberInput value={f.injected_latency_ms} onChange={(v) => set('injected_latency_ms', v ?? 0)} min={0} max={60000} step={10} unit="ms" />
            </Field>
            <Field label="Gigue" hint="ajout aléatoire 0…gigue">
              <NumberInput value={f.latency_jitter_ms} onChange={(v) => set('latency_jitter_ms', v ?? 0)} min={0} max={60000} step={10} unit="ms" />
            </Field>
            <Field label="Facteur de latence" hint="ex. ×4 pour émuler le Raspberry Pi">
              <NumberInput value={f.latency_scale} onChange={(v) => set('latency_scale', v ?? 1)} min={0.01} max={1000} step={0.5} decimals={2} unit="×" />
            </Field>
            <Field label="Vues manquantes">
              <NumberInput value={pct(f.missing_view_rate)} onChange={(v) => set('missing_view_rate', (v ?? 0) / 100)} min={0} max={100} step={1} decimals={1} unit="%" />
            </Field>
            <Field label="Unités d'inférence" hint="décisions calculées en parallèle">
              <NumberInput value={f.workers} onChange={(v) => set('workers', v ?? 1)} min={1} max={16} step={1} />
            </Field>
            <Field label="Fils ONNX Runtime" hint="vide = automatique">
              <NumberInput value={f.inference_threads} onChange={(v) => set('inference_threads', v)} min={1} max={64} step={1} allowEmpty placeholder="auto" />
            </Field>
          </div>
        </Panel>
      </div>

      <div className={s.formSide}>
        <Panel title="Budget de temps" description="Calculé en direct à partir de la géométrie.">
          {timing.error && <Callout tone="danger">Paramètres invalides : vérifier les valeurs saisies.</Callout>}
          {t && (
            <div className="u-stack-4">
              <div className={s.budget}>
                <span className={s.budgetValue}>{formatNumber(t.budget_ms)} ms</span>
                <span className="u-muted u-small">
                  distance caméra → trappes ÷ vitesse ; {formatNumber(t.effective_budget_ms)} ms une fois l'ouverture des trappes anticipée
                </span>
              </div>
              <dl className={s.kv}>
                <dt>Vitesse</dt>
                <dd className="u-num">{formatNumber(t.speed_mm_s, 1)} mm/s</dd>
                <dt>Pas</dt>
                <dd className="u-num">{formatNumber(t.pitch_mm, 1)} mm</dd>
                <dt>Passage sous la caméra</dt>
                <dd className="u-num">{formatNumber(t.camera_time_s * 1000)} ms</dd>
                <dt>Colonnes en vol</dt>
                <dd className="u-num">{t.columns_in_flight}</dd>
                <dt>Débit</dt>
                <dd className="u-num">{formatNumber(t.eggs_per_hour)} œufs/h</dd>
                <dt>Durée simulée</dt>
                <dd className="u-num">{formatDuration(t.virtual_duration_s)}</dd>
              </dl>
              {t.warnings.map((w) => (
                <Callout key={w} tone="warning">
                  {w}
                </Callout>
              ))}
            </div>
          )}
        </Panel>
        <Panel>
          <div className="u-stack-4">
            <Field label="Nom" hint="facultatif">
              <TextInput value={name} onChange={(e) => setName(e.target.value)} placeholder={`${model.name} — date`} maxLength={200} />
            </Field>
            <Button variant="primary" icon={Play} size="lg" block onClick={launch} loading={create.isPending} disabled={!!timing.error}>
              Lancer la simulation
            </Button>
            <Button variant="ghost" icon={RotateCcw} size="sm" onClick={() => (setF(initial), setSpeedMode('pitch'))}>
              Valeurs par défaut
            </Button>
          </div>
        </Panel>
      </div>
    </div>
  );
}
