/** Détail d'une simulation : suivi de la tâche, lecteur, résultats, paramètres ; duplication et suppression. */
import { useState } from 'react';
import { Route, Routes, useNavigate, useParams } from 'react-router-dom';
import { Copy, Trash2 } from 'lucide-react';
import { JobProgress } from '@/components/JobProgress';
import { useQueryClient } from '@tanstack/react-query';
import { formatDateTime, formatNumber, formatPercent } from '@/lib/format';
import {
  Button,
  Callout,
  ConfirmDialog,
  ErrorState,
  Field,
  KeyValue,
  LoadingState,
  Modal,
  NumberInput,
  PageHeader,
  Panel,
  RouteTabs,
  SegmentedControl,
  Select,
  TextInput,
  useToast,
} from '@/components/ui';
import { simKeys, useDeleteSim, useDuplicateSim, useSim, useSimModels } from './api';
import { Player } from './Player';
import { Results } from './Results';
import { SimStatusBadge } from './SimStatusBadge';
import type { SimDetail } from './types';

export function SimulationDetail() {
  const { simId = '' } = useParams();
  const sim = useSim(simId);
  const qc = useQueryClient();
  const navigate = useNavigate();
  const toast = useToast();
  const del = useDeleteSim();
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [dupOpen, setDupOpen] = useState(false);

  if (sim.isLoading) return <LoadingState />;
  if (sim.error || !sim.data) return <ErrorState error={sim.error} title="Simulation introuvable" onRetry={() => sim.refetch()} />;
  const d = sim.data;
  const base = `/simulation/${encodeURIComponent(d.id)}`;
  const running = d.status === 'running' || d.status === 'created';

  return (
    <>
      <PageHeader
        breadcrumbs={[{ label: 'Simulations', to: '/simulation' }, { label: d.name }]}
        title={d.name}
        subtitle={
          <span>
            <SimStatusBadge status={d.status} /> {d.model_name ?? d.model_id} · {formatNumber(d.n_columns)} colonnes · {formatPercent(d.defect_rate, 0)} de défauts · graine{' '}
            <span className="u-mono">{d.seed}</span> · {formatDateTime(d.created_at)}
          </span>
        }
        actions={
          <>
            <Button icon={Copy} onClick={() => setDupOpen(true)} disabled={running}>
              Relancer avec…
            </Button>
            <Button icon={Trash2} variant="danger" onClick={() => setConfirmDelete(true)} disabled={running}>
              Supprimer
            </Button>
          </>
        }
      >
        {d.status === 'done' && (
          <RouteTabs
            items={[
              { to: base, label: 'Lecteur', end: true },
              { to: `${base}/resultats`, label: 'Résultats' },
              { to: `${base}/parametres`, label: 'Paramètres' },
            ]}
          />
        )}
      </PageHeader>

      {running && (
        <Panel title="Simulation en cours" description="Inférence réelle sur chaque colonne ; l'horloge de la table est virtuelle.">
          <JobProgress jobId={d.job_id} onDone={() => qc.invalidateQueries({ queryKey: simKeys.all })} />
        </Panel>
      )}
      {d.status === 'failed' && (
        <Callout tone="danger" title="La simulation a échoué">
          {d.results.error ?? d.job?.error ?? 'Erreur inconnue (voir le journal de la tâche).'}
        </Callout>
      )}
      {d.status === 'cancelled' && <Callout tone="warning" title="Simulation annulée">Aucun résultat n'a été enregistré.</Callout>}

      {d.status === 'done' && (
        <Routes>
          <Route index element={<Player sim={d} />} />
          <Route path="resultats" element={<Results sim={d} />} />
          <Route path="parametres" element={<ParamsView sim={d} />} />
        </Routes>
      )}
      {d.status !== 'done' && !running && <ParamsView sim={d} />}

      <ConfirmDialog
        open={confirmDelete}
        title="Supprimer la simulation ?"
        message="Le journal, la chronologie et la vidéo éventuelle seront supprimés définitivement."
        confirmLabel="Supprimer"
        variant="danger"
        onCancel={() => setConfirmDelete(false)}
        onConfirm={async () => {
          try {
            await del.mutateAsync(d.id);
            toast.info('Simulation supprimée');
            navigate('/simulation');
          } catch (e) {
            toast.error(e);
          } finally {
            setConfirmDelete(false);
          }
        }}
      />
      {dupOpen && <DuplicateModal sim={d} onClose={() => setDupOpen(false)} />}
    </>
  );
}

const PARAM_LABELS: Record<string, [string, (v: unknown) => string]> = {
  model_id: ['Modèle', (v) => String(v)],
  variant: ['Variante', (v) => (v ? String(v).toUpperCase() : 'défaut du paquet')],
  threshold: ['Seuil', (v) => (v === null || v === undefined ? 'celui du modèle' : formatNumber(Number(v), 3))],
  n_lanes: ['Couloirs', (v) => String(v)],
  period_s: ['Cadence', (v) => `${formatNumber(Number(v), 2)} s / colonne`],
  pitch_mm: ['Pas entre colonnes', (v) => `${formatNumber(Number(v), 1)} mm`],
  camera_zone_mm: ['Zone caméra', (v) => `${formatNumber(Number(v))} mm`],
  camera_to_trap_mm: ['Distance caméra → trappes', (v) => `${formatNumber(Number(v))} mm`],
  trap_actuation_ms: ["Temps d'ouverture des trappes", (v) => `${formatNumber(Number(v))} ms`],
  n_columns: ['Colonnes', (v) => formatNumber(Number(v))],
  defect_rate: ['Proportion de défauts', (v) => formatPercent(Number(v), 1)],
  seed: ['Graine', (v) => String(v)],
  empty_lane_rate: ['Couloirs vides', (v) => formatPercent(Number(v), 1)],
  injected_latency_ms: ['Latence injectée', (v) => `${formatNumber(Number(v))} ms`],
  latency_jitter_ms: ['Gigue', (v) => `${formatNumber(Number(v))} ms`],
  latency_scale: ['Facteur de latence', (v) => `×${formatNumber(Number(v), 2)}`],
  missing_view_rate: ['Vues manquantes', (v) => formatPercent(Number(v), 1)],
  inference_threads: ['Fils ONNX Runtime', (v) => (v ? String(v) : 'automatique')],
  workers: ["Unités d'inférence", (v) => String(v)],
};

export function paramLabel(key: string): string {
  return PARAM_LABELS[key]?.[0] ?? key;
}
export function paramValue(key: string, v: unknown): string {
  const f = PARAM_LABELS[key]?.[1];
  return f ? f(v) : v === null || v === undefined ? '—' : String(v);
}

function ParamsView({ sim }: { sim: SimDetail }) {
  const t = sim.timing;
  const comp = sim.results.composition;
  return (
    <div className="u-grid-2">
      <Panel title="Paramètres">
        <KeyValue items={Object.keys(sim.params).filter((k) => k in PARAM_LABELS).map((k) => ({ label: paramLabel(k), value: paramValue(k, (sim.params as unknown as Record<string, unknown>)[k]) }))} />
      </Panel>
      <div className="u-stack-4">
        {t && (
          <Panel title="Grandeurs dérivées">
            <KeyValue
              items={[
                { label: 'Vitesse', value: `${formatNumber(t.speed_mm_s, 1)} mm/s` },
                { label: 'Budget de temps', value: `${formatNumber(t.budget_ms)} ms (effectif ${formatNumber(t.effective_budget_ms)} ms)` },
                { label: 'Passage sous la caméra', value: `${formatNumber(t.camera_time_s * 1000)} ms` },
                { label: 'Colonnes en vol', value: String(t.columns_in_flight) },
                { label: 'Débit', value: `${formatNumber(t.eggs_per_hour)} œufs/h` },
              ]}
            />
          </Panel>
        )}
        {comp && (
          <Panel title="Composition">
            <KeyValue
              items={[
                { label: 'Œufs simulés', value: `${formatNumber(comp.n_eggs)} (${formatNumber(comp.n_non_viable)} non viables, ${formatPercent(comp.defect_rate_actual, 2)})` },
                { label: 'Couloirs vides', value: formatNumber(comp.n_empty_slots) },
                { label: 'Œufs distincts du jeu test', value: `${formatNumber(comp.n_distinct_eggs)} (réutilisation ${formatPercent(comp.reuse_rate)})` },
                { label: 'Vues manquantes', value: `${formatNumber(comp.n_views_missing)} · ${formatNumber(comp.n_eggs_without_views)} œuf(s) sans vue` },
                { label: 'Jeu test', value: `${formatNumber(comp.pool.n_viable)} viables, ${formatNumber(comp.pool.n_non_viable)} non viables, ${comp.pool.n_test_lots} lot(s)` },
              ]}
            />
          </Panel>
        )}
      </div>
    </div>
  );
}

function DuplicateModal({ sim, onClose }: { sim: SimDetail; onClose: () => void }) {
  const models = useSimModels();
  const dup = useDuplicateSim();
  const navigate = useNavigate();
  const toast = useToast();
  const [modelId, setModelId] = useState(sim.model_id ?? '');
  const [useModelThreshold, setUseModelThreshold] = useState<'model' | 'custom'>(sim.params.threshold === null ? 'model' : 'custom');
  const [threshold, setThreshold] = useState<number | null>(sim.params.threshold ?? sim.results.threshold ?? 0.5);
  const [name, setName] = useState('');
  const eligible = (models.data ?? []).filter((m) => m.eligible);

  const submit = async () => {
    const body: Parameters<typeof dup.mutateAsync>[0] = { id: sim.id };
    if (modelId && modelId !== sim.model_id) body.model_id = modelId;
    const newThreshold = useModelThreshold === 'model' ? null : threshold;
    if (newThreshold !== sim.params.threshold || body.model_id) body.threshold = newThreshold;
    if (name.trim()) body.name = name.trim();
    try {
      const res = await dup.mutateAsync(body);
      toast.success('Simulation relancée', res.simulation.name);
      onClose();
      navigate(`/simulation/${encodeURIComponent(res.simulation.id)}`);
    } catch (e) {
      toast.error(e);
    }
  };

  return (
    <Modal
      open
      onClose={onClose}
      title="Relancer avec un autre modèle ou seuil"
      description="Mêmes paramètres et même graine : avec le même découpage, les colonnes sont identiques et la comparaison ne porte que sur le modèle ou le seuil."
      footer={
        <>
          <Button variant="ghost" onClick={onClose}>
            Annuler
          </Button>
          <Button variant="primary" onClick={submit} loading={dup.isPending}>
            Lancer
          </Button>
        </>
      }
    >
      <div className="u-stack-4">
        <Field label="Modèle">
          <Select value={modelId} onChange={setModelId} options={eligible.map((m) => ({ value: m.id, label: m.name }))} />
        </Field>
        <Field label="Seuil">
          <SegmentedControl<'model' | 'custom'>
            value={useModelThreshold}
            onChange={setUseModelThreshold}
            options={[
              { value: 'model', label: 'Seuil du modèle' },
              { value: 'custom', label: 'Seuil personnalisé' },
            ]}
          />
        </Field>
        {useModelThreshold === 'custom' && (
          <Field label="Valeur du seuil">
            <NumberInput value={threshold} onChange={setThreshold} min={0} max={1} step={0.01} decimals={3} />
          </Field>
        )}
        <Field label="Nom" hint="facultatif">
          <TextInput value={name} onChange={(e) => setName(e.target.value)} placeholder="nom généré automatiquement" />
        </Field>
      </div>
    </Modal>
  );
}
