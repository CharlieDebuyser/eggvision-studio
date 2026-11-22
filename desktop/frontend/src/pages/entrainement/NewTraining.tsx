/** Lancement d'un entraînement (classifieur multi-vues) ou d'un modèle de référence par anomalies. */
import { useEffect, useMemo, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { ChevronDown, ChevronRight, Play, TriangleAlert } from 'lucide-react';
import {
  Badge,
  Button,
  Callout,
  ErrorState,
  Field,
  LoadingState,
  NumberInput,
  Panel,
  SegmentedControl,
  Select,
  TextInput,
  Toggle,
  useToast,
} from '@/components/ui';
import { formatBytes, formatNumber } from '@/lib/format';
import { useOptions, useSplits, useStartAnomaly, useStartTraining } from './api';
import { Int8Badge, SUBSET_LABEL, SUBSETS, SplitWarnings, SummaryTable } from './common';
import type { AnomalyParams, Architecture, Preset, TrainParams } from './types';
import s from './entrainement.module.css';

type Mode = 'classifier' | 'anomaly';

const LABELS: Record<string, string> = {
  pos_weight: 'Pondération de la perte',
  sampler: 'Échantillonnage équilibré',
  none: 'Aucune',
  light: 'Légères',
  standard: 'Standard',
  max: 'Maximum des vues',
  attention: 'Pooling appris (attention)',
  val_auc: 'AUC ROC (validation)',
  val_ap: 'Précision moyenne (validation)',
  val_loss: 'Perte (validation)',
  auto: 'Automatique',
  cuda: 'CUDA (NVIDIA)',
  mps: 'MPS (Apple)',
  cpu: 'CPU',
};
const opts = (values: string[]) => values.map((v) => ({ value: v, label: LABELS[v] ?? v }));

export function NewTraining() {
  const navigate = useNavigate();
  const toast = useToast();
  const [params] = useSearchParams();
  const options = useOptions();
  const splits = useSplits();
  const startTrain = useStartTraining();
  const startAnomaly = useStartAnomaly();
  const [mode, setMode] = useState<Mode>('classifier');
  const [splitId, setSplitId] = useState<string>(params.get('split') ?? '');
  const [presetId, setPresetId] = useState<Preset['id']>('standard');
  const [hp, setHp] = useState<TrainParams | null>(null);
  const [edited, setEdited] = useState(false);
  const [an, setAn] = useState<AnomalyParams | null>(null);
  const [advanced, setAdvanced] = useState(false);
  const [name, setName] = useState('');

  useEffect(() => {
    if (options.data && !hp) {
      setHp(options.data.presets.find((p) => p.id === 'standard')?.hparams ?? options.data.defaults);
      setAn(options.data.anomaly_defaults);
    }
  }, [options.data, hp]);
  useEffect(() => {
    if (!splitId && splits.data?.items.length) setSplitId(String(splits.data.items[0].id));
  }, [splits.data, splitId]);

  const split = useMemo(() => splits.data?.items.find((x) => String(x.id) === splitId), [splits.data, splitId]);

  if (options.isLoading || splits.isLoading || !hp || !an) return <LoadingState />;
  if (options.error) return <ErrorState error={options.error} onRetry={() => options.refetch()} />;
  const opt = options.data!;

  const choosePreset = (p: Preset) => {
    setPresetId(p.id);
    setHp({ ...p.hparams });
    setEdited(false);
  };
  const set = <K extends keyof TrainParams>(k: K, v: TrainParams[K]) => {
    setHp((old) => (old ? { ...old, [k]: v } : old));
    setEdited(true);
  };
  const setA = <K extends keyof AnomalyParams>(k: K, v: AnomalyParams[K]) => setAn((old) => (old ? { ...old, [k]: v } : old));
  const arch = opt.architectures.find((a) => a.id === (mode === 'classifier' ? hp.arch : an.arch));
  const noSplit = !splits.data?.items.length;

  const launch = async () => {
    if (!split) return;
    try {
      const out =
        mode === 'classifier'
          ? await startTrain.mutateAsync({ split_id: split.id, preset: presetId, hparams: hp, name: name.trim() || undefined })
          : await startAnomaly.mutateAsync({ split_id: split.id, params: { ...an }, name: name.trim() || undefined });
      toast.success('Entraînement lancé');
      navigate(`/entrainement/modeles/${encodeURIComponent(out.model_id)}`);
    } catch (e) {
      toast.error(e);
    }
  };

  return (
    <div className={s.formGrid}>
      <div className="u-stack-4">
        <Panel title="Type de modèle">
          <SegmentedControl
            value={mode}
            onChange={setMode}
            options={[
              { value: 'classifier', label: 'Classifieur multi-vues' },
              { value: 'anomaly', label: 'Référence par anomalies (PatchCore)', title: 'Appris uniquement sur les œufs viables ; non exportable vers le Pi' },
            ]}
          />
          {mode === 'anomaly' && (
            <p className="u-small u-muted">
              Modèle de référence : banque de caractéristiques des œufs viables d'entraînement, score = distance au plus proche voisin. Aucun
              apprentissage des défauts ; sert de point de comparaison. Il n'est pas exportable vers le Raspberry Pi.
            </p>
          )}
        </Panel>

        {mode === 'classifier' && (
          <Panel title="Préréglage" description="Point de départ ; chaque hyperparamètre reste modifiable ci-dessous.">
            <div className={s.presets}>
              {opt.presets.map((p) => (
                <button type="button" key={p.id} className={`${s.archCard} ${presetId === p.id ? s.archCardOn : ''}`} onClick={() => choosePreset(p)}>
                  <span className={s.archHead}>
                    <span className="u-strong">{p.label}</span>
                    {presetId === p.id && edited && <Badge tone="accent">modifié</Badge>}
                  </span>
                  <span className="u-xs u-muted">{p.description}</span>
                </button>
              ))}
            </div>
          </Panel>
        )}

        <Panel title="Architecture" description="Poids pré-entraînés ImageNet embarqués dans l'application (aucun accès Internet).">
          <div className={s.archList}>
            {opt.architectures.map((a) => (
              <ArchCard
                key={a.id}
                arch={a}
                selected={(mode === 'classifier' ? hp.arch : an.arch) === a.id}
                showInt8={mode === 'classifier'}
                onClick={() => (mode === 'classifier' ? set('arch', a.id) : setA('arch', a.id))}
              />
            ))}
          </div>
        </Panel>

        {mode === 'classifier' ? (
          <Panel title="Hyperparamètres principaux">
            <div className="u-stack-4">
              <div className={s.fields4}>
                <Field label="Taille d'entrée">
                  <NumberInput value={hp.input_size} onChange={(v) => set('input_size', v ?? 224)} min={32} max={512} step={16} unit="px" />
                </Field>
                <Field label="Vues par œuf">
                  <NumberInput value={hp.n_views} onChange={(v) => set('n_views', v ?? 12)} min={1} max={64} />
                </Field>
                <Field label="Époques max">
                  <NumberInput value={hp.epochs} onChange={(v) => set('epochs', v ?? 10)} min={1} max={1000} />
                </Field>
                <Field label="Œufs par lot" hint={`${formatNumber(hp.batch_size * hp.n_views)} images`}>
                  <NumberInput value={hp.batch_size} onChange={(v) => set('batch_size', v ?? 8)} min={1} max={512} />
                </Field>
                <Field label="Taux d'apprentissage">
                  <NumberInput value={hp.lr} onChange={(v) => set('lr', v ?? 5e-4)} min={0.000001} max={0.5} step={0.0001} decimals={6} />
                </Field>
                <Field label="Agrégation des vues">
                  <Select value={hp.aggregation} onChange={(v) => set('aggregation', v)} options={opts(opt.choices.aggregation) as { value: TrainParams['aggregation']; label: string }[]} />
                </Field>
                <Field label="Déséquilibre des classes">
                  <Select value={hp.imbalance} onChange={(v) => set('imbalance', v)} options={opts(opt.choices.imbalance) as { value: TrainParams['imbalance']; label: string }[]} />
                </Field>
                <Field label="Augmentations" hint="rotations, retournements, couleur, flou de bouger">
                  <Select value={hp.augment} onChange={(v) => set('augment', v)} options={opts(opt.choices.augment) as { value: TrainParams['augment']; label: string }[]} />
                </Field>
              </div>
              <button type="button" className="btn btn--ghost btn--sm" onClick={() => setAdvanced((a) => !a)} aria-expanded={advanced}>
                {advanced ? <ChevronDown size={14} aria-hidden /> : <ChevronRight size={14} aria-hidden />}
                <span className="btn__label">Réglages avancés</span>
              </button>
              {advanced && (
                <div className={s.fields4}>
                  <Field label="Époques backbone gelé" hint="seule la tête apprend">
                    <NumberInput value={hp.freeze_epochs} onChange={(v) => set('freeze_epochs', v ?? 0)} min={0} max={100} />
                  </Field>
                  <Field label="Arrêt anticipé : patience" hint="époques sans amélioration">
                    <NumberInput value={hp.patience} onChange={(v) => set('patience', v ?? 5)} min={1} max={200} />
                  </Field>
                  <Field label="Métrique surveillée">
                    <Select value={hp.monitor} onChange={(v) => set('monitor', v)} options={opts(opt.choices.monitor) as { value: TrainParams['monitor']; label: string }[]} />
                  </Field>
                  <Field label="Échauffement" hint="époques">
                    <NumberInput value={hp.warmup_epochs} onChange={(v) => set('warmup_epochs', v ?? 0)} min={0} max={20} step={0.5} decimals={1} />
                  </Field>
                  <Field label="Facteur lr backbone">
                    <NumberInput value={hp.backbone_lr_mult} onChange={(v) => set('backbone_lr_mult', v ?? 0.3)} min={0.01} max={1} step={0.05} decimals={2} />
                  </Field>
                  <Field label="Weight decay">
                    <NumberInput value={hp.weight_decay} onChange={(v) => set('weight_decay', v ?? 0)} min={0} max={1} step={0.0001} decimals={5} />
                  </Field>
                  <Field label="Dropout">
                    <NumberInput value={hp.dropout} onChange={(v) => set('dropout', v ?? 0)} min={0} max={0.9} step={0.05} decimals={2} />
                  </Field>
                  <Field label="Graine">
                    <NumberInput value={hp.seed} onChange={(v) => set('seed', v ?? 0)} min={0} max={1_000_000} />
                  </Field>
                  <Field label="Appareil" hint={`détecté : ${opt.device.kind.toUpperCase()}`}>
                    <Select value={hp.device} onChange={(v) => set('device', v)} options={opts(opt.choices.device) as { value: TrainParams['device']; label: string }[]} />
                  </Field>
                  <Field label="Processus de chargement" hint="0 = dans le processus principal">
                    <NumberInput value={hp.num_workers} onChange={(v) => set('num_workers', v ?? 0)} min={0} max={32} />
                  </Field>
                  <Field label="Cache des vues">
                    <NumberInput value={hp.cache_mb} onChange={(v) => set('cache_mb', v ?? 0)} min={0} max={65536} step={128} unit="Mo" />
                  </Field>
                  <div className="u-stack-4">
                    <Toggle checked={hp.pretrained} onChange={(v) => set('pretrained', v)} label="Poids pré-entraînés" />
                    <Toggle checked={hp.reviewed_only} onChange={(v) => set('reviewed_only', v)} label="Labels relus uniquement" />
                    <Toggle checked={hp.amp} onChange={(v) => set('amp', v)} label="Précision mixte (CUDA)" />
                  </div>
                </div>
              )}
            </div>
          </Panel>
        ) : (
          <Panel title="Paramètres du modèle d'anomalies">
            <div className={s.fields4}>
              <Field label="Taille d'entrée">
                <NumberInput value={an.input_size} onChange={(v) => setA('input_size', v ?? 224)} min={32} max={512} step={16} unit="px" />
              </Field>
              <Field label="Vues par œuf">
                <NumberInput value={an.n_views} onChange={(v) => setA('n_views', v ?? 12)} min={1} max={64} />
              </Field>
              <Field label="Banque mémoire" hint="patches retenus">
                <NumberInput value={an.bank_size} onChange={(v) => setA('bank_size', v ?? 4000)} min={10} max={100000} step={500} />
              </Field>
              <Field label="Patches par vue">
                <NumberInput value={an.patches_per_view} onChange={(v) => setA('patches_per_view', v ?? 48)} min={1} max={4096} />
              </Field>
              <Field label="Sous-échantillonnage">
                <Select
                  value={an.coreset}
                  onChange={(v) => setA('coreset', v)}
                  options={[
                    { value: 'greedy', label: 'Coreset glouton' },
                    { value: 'random', label: 'Aléatoire' },
                  ]}
                />
              </Field>
              <Field label="Œufs viables max">
                <NumberInput value={an.max_train_eggs} onChange={(v) => setA('max_train_eggs', v ?? 400)} min={1} max={100000} />
              </Field>
              <Field label="Graine">
                <NumberInput value={an.seed} onChange={(v) => setA('seed', v ?? 0)} min={0} max={1_000_000} />
              </Field>
              <Toggle checked={an.reviewed_only} onChange={(v) => setA('reviewed_only', v)} label="Labels relus uniquement" />
            </div>
          </Panel>
        )}
      </div>

      <div className={s.formSide}>
        <Panel title="Données">
          {noSplit ? (
            <Callout tone="warning" title="Aucun découpage">
              Créez d'abord un découpage entraînement / validation / test.
              <div className="u-row" style={{ marginTop: 8 }}>
                <Button size="sm" onClick={() => navigate('/entrainement/decoupages/nouveau')}>
                  Créer un découpage
                </Button>
              </div>
            </Callout>
          ) : (
            <div className="u-stack-4">
              <Field label="Découpage">
                <Select
                  value={splitId}
                  onChange={setSplitId}
                  options={(splits.data?.items ?? []).map((x) => ({ value: String(x.id), label: `${x.name} v${x.version}${x.locked ? ' (figé)' : ''}` }))}
                />
              </Field>
              {split && (
                <>
                  <SummaryTable summary={split.summary} />
                  {(!split.summary.leakage_check.ok || split.summary.warnings.length > 0) && <SplitWarnings summary={split.summary} />}
                </>
              )}
            </div>
          )}
        </Panel>
        <Panel>
          <div className="u-stack-4">
            {arch && !arch.pretrained.available && (
              <Callout tone="warning" icon={TriangleAlert} title="Poids pré-entraînés absents">
                {arch.label} démarrera en initialisation aléatoire (résultats nettement moins bons).
              </Callout>
            )}
            <Field label="Nom" hint="Facultatif : un nom est proposé automatiquement.">
              <TextInput value={name} onChange={(e) => setName(e.target.value)} placeholder="ex. Small 224 px — relecture J3" />
            </Field>
            <p className="u-xs u-muted">
              Entraînement : {SUBSETS.map((sub) => `${SUBSET_LABEL[sub]} ${formatNumber(split?.summary.subsets[sub].n_eggs)}`).join(' · ')} œufs ; appareil{' '}
              {opt.device.kind.toUpperCase()} ({opt.device.name}).
            </p>
            <Button
              variant="primary"
              icon={Play}
              block
              size="lg"
              onClick={launch}
              loading={startTrain.isPending || startAnomaly.isPending}
              disabled={!split}
            >
              {mode === 'classifier' ? "Lancer l'entraînement" : 'Construire le modèle de référence'}
            </Button>
          </div>
        </Panel>
      </div>
    </div>
  );
}

function ArchCard({ arch, selected, onClick, showInt8 }: { arch: Architecture; selected: boolean; onClick: () => void; showInt8: boolean }) {
  return (
    <button type="button" className={`${s.archCard} ${selected ? s.archCardOn : ''}`} onClick={onClick} aria-pressed={selected}>
      <span className={s.archHead}>
        <span className="u-strong">{arch.label}</span>
        <span className="u-xs u-muted">≈ {formatNumber(arch.params_m, 1)} M paramètres</span>
        <span className="u-spacer" />
        {showInt8 && <Int8Badge friendly={arch.int8_friendly} />}
        {arch.pretrained.available ? (
          <Badge tone="success" dot>
            Poids ImageNet {formatBytes(arch.pretrained.size_bytes)}
          </Badge>
        ) : (
          <Badge tone="danger" dot>
            Poids absents
          </Badge>
        )}
      </span>
      <span className="u-xs u-muted">{arch.description}</span>
    </button>
  );
}
