/** Page d'un modèle : suivi (courbes en direct, reprise), rapport d'évaluation, explorateur d'erreurs. */
import { useState } from 'react';
import { Route, Routes, useNavigate, useParams } from 'react-router-dom';
import { Pencil, RotateCcw, Square, Trash2, Upload } from 'lucide-react';
import { JobProgress } from '@/components/JobProgress';
import {
  Badge,
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
  TextInput,
  useToast,
} from '@/components/ui';
import { defectLabel } from '@/lib/domain';
import { formatDateTime, formatDuration, formatNumber } from '@/lib/format';
import { isModelActive, useCancelTraining, useDeleteModel, useHistory, useModel, useOptions, useRenameModel, useResume } from './api';
import { Int8Badge, ModelStatusBadge, SUBSET_LABEL, archLabel } from './common';
import { Curves } from './Curves';
import { ErrorExplorer } from './Errors';
import { ReportView } from './Report';
import type { ModelDetail as Model, Subset } from './types';
import s from './entrainement.module.css';

export function ModelDetailPage() {
  const { modelId = '' } = useParams();
  const model = useModel(modelId);
  const options = useOptions();
  const navigate = useNavigate();
  const toast = useToast();
  const del = useDeleteModel();
  const cancel = useCancelTraining();
  const [renaming, setRenaming] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [resuming, setResuming] = useState(false);

  if (model.isLoading) return <LoadingState />;
  if (model.error || !model.data) return <ErrorState error={model.error} onRetry={() => model.refetch()} />;
  const m = model.data;
  const arch = options.data?.architectures.find((a) => a.id === m.arch);
  const base = `/entrainement/modeles/${encodeURIComponent(m.id)}`;
  const active = isModelActive(m);
  const evaluated = m.status === 'trained';

  return (
    <>
      <PageHeader
        breadcrumbs={[{ label: 'Entraînement', to: '/entrainement' }, { label: m.name }]}
        title={m.name}
        subtitle={
          <span className="u-row">
            <ModelStatusBadge status={m.status} />
            <span>
              {m.kind === 'anomaly' ? 'Référence par anomalies · ' : ''}
              {archLabel(options.data?.architectures, m.arch)} · {m.input_size} px · {m.n_views} vues · version n° {m.number ?? '—'}
            </span>
            {arch && m.kind === 'classifier' && <Int8Badge friendly={arch.int8_friendly} />}
          </span>
        }
        actions={
          <>
            {active && (
              <Button icon={Square} variant="danger" loading={cancel.isPending} onClick={() => cancel.mutate(m.id, { onError: (e) => toast.error(e) })}>
                Arrêter
              </Button>
            )}
            {!active && m.kind === 'classifier' && (m.resumable || m.status === 'trained') && (
              <Button icon={RotateCcw} onClick={() => setResuming(true)}>
                {m.resumable ? 'Reprendre' : 'Prolonger'}
              </Button>
            )}
            {evaluated && m.kind === 'classifier' && (
              <Button icon={Upload} variant="primary" onClick={() => navigate(`/export/${encodeURIComponent(m.id)}`)}>
                Exporter
              </Button>
            )}
            <Button icon={Pencil} variant="ghost" onClick={() => setRenaming(true)}>
              Renommer
            </Button>
            {!active && (
              <Button icon={Trash2} variant="ghost" onClick={() => setDeleting(true)}>
                Supprimer
              </Button>
            )}
          </>
        }
      >
        <RouteTabs
          items={[
            { to: base, label: 'Suivi', end: true },
            { to: `${base}/rapport`, label: 'Rapport' },
            { to: `${base}/erreurs`, label: 'Erreurs' },
          ]}
        />
      </PageHeader>
      <Routes>
        <Route index element={<Overview m={m} />} />
        <Route path="rapport" element={evaluated ? <ReportView model={m} /> : <NotEvaluated m={m} />} />
        <Route path="erreurs" element={evaluated ? <ErrorExplorer model={m} /> : <NotEvaluated m={m} />} />
      </Routes>
      <RenameModal open={renaming} onClose={() => setRenaming(false)} model={m} />
      <ResumeModal open={resuming} onClose={() => setResuming(false)} model={m} />
      <ConfirmDialog
        open={deleting}
        title="Supprimer ce modèle ?"
        message="Les poids, le rapport, les prédictions et le paquet exporté seront supprimés définitivement."
        confirmLabel="Supprimer"
        variant="danger"
        onCancel={() => setDeleting(false)}
        onConfirm={async () => {
          try {
            await del.mutateAsync(m.id);
            toast.success('Modèle supprimé');
            navigate('/entrainement');
          } catch (e) {
            toast.error(e);
          }
        }}
      />
    </>
  );
}

function NotEvaluated({ m }: { m: Model }) {
  return (
    <Callout tone="info" title="Pas encore évalué">
      {m.status === 'training' || m.status === 'created'
        ? "Le rapport sera disponible à la fin de l'entraînement (évaluation automatique sur le test)."
        : "Ce modèle n'a pas terminé son entraînement : le reprendre pour obtenir le rapport."}
    </Callout>
  );
}

function Overview({ m }: { m: Model }) {
  const active = isModelActive(m);
  const hist = useHistory(m.id, active);
  const history = hist.data?.history ?? m.history;
  const hp = m.hparams;
  const ds = m.data_summary;
  const lastJob = hist.data?.job;
  const stoppedEarly = m.extra?.stopped_early === true;
  return (
    <div className="u-stack-6">
      {active && m.job_id && (
        <Panel title="Entraînement en cours">
          <JobProgress jobId={m.job_id} hideTitle />
        </Panel>
      )}
      {m.warnings.length > 0 && (
        <Callout tone="warning" title="Avertissements">
          <ul className={s.warnList}>
            {m.warnings.map((w) => (
              <li key={w}>{w}</li>
            ))}
          </ul>
        </Callout>
      )}
      {m.kind === 'classifier' && (
        <Panel
          title="Courbes d'apprentissage"
          description={
            active
              ? `En direct — époque ${history.length} / ${hp.epochs ?? '?'}`
              : `${history.length} époque(s)${stoppedEarly ? ` — arrêt anticipé (pas d'amélioration de « ${hp.monitor} » pendant ${hp.patience} époques)` : ''}`
          }
          actions={
            m.best_epoch != null ? (
              <Badge tone="success">Meilleure époque : {m.best_epoch}</Badge>
            ) : lastJob ? (
              <Badge tone="info">Démarrage…</Badge>
            ) : undefined
          }
        >
          <Curves history={history} bestEpoch={m.best_epoch} totalEpochs={hp.epochs as number | undefined} />
          {history.length > 0 && (
            <p className="u-xs u-muted">
              Dernière époque : {formatDuration(history[history.length - 1].duration_s)} · {formatNumber(history[history.length - 1].images_per_s)} images/s ·
              lr tête {formatNumber(history[history.length - 1].lr, 6)}
            </p>
          )}
        </Panel>
      )}
      {m.status === 'trained' && m.metrics.test && (
        <Panel title="Résultat sur le jeu de test" description={`Seuil ${formatNumber(m.threshold, 3)} — détails dans l'onglet « Rapport ».`}>
          <KeyValue
            items={[
              { label: 'AUC ROC', value: formatNumber(m.metrics.test.auc_roc, 3) },
              { label: 'Rappel non viables', value: `${formatNumber((m.metrics.test.recall ?? 0) * 100, 1)} %` },
              { label: 'Faux rejets', value: `${formatNumber((m.metrics.test.false_reject_rate ?? 0) * 100, 1)} %` },
              { label: 'Œufs (non viables / viables)', value: `${m.metrics.test.n} (${m.metrics.test.n_pos} / ${m.metrics.test.n_neg})` },
            ]}
          />
        </Panel>
      )}
      <div className={s.twoCols}>
        <Panel title="Données utilisées">
          <KeyValue
            items={[
              { label: 'Découpage', value: ds.split_name ? `${ds.split_name} v${ds.split_version} (n° ${ds.split_id})` : `n° ${m.split_id ?? '—'}` },
              { label: 'Regroupement', value: ds.group_by === 'lot' ? 'par lot' : ds.group_by ? 'par session' : '—' },
              { label: 'Labels', value: ds.reviewed_only ? 'relus uniquement' : 'tous les labels viable / non viable' },
              ...(['train', 'val', 'test'] as Subset[]).map((sub) => {
                const c = ds.counts?.[sub];
                return {
                  label: SUBSET_LABEL[sub],
                  value: c ? `${c.n} œufs (${c.non_viable} non viables · ${ds.lots?.[sub]?.length ?? '?'} lots)` : '—',
                };
              }),
              ...(ds.bank_eggs ? [{ label: 'Banque mémoire', value: `${ds.bank_eggs} œufs viables` }] : []),
              { label: 'Empreinte des labels', value: ds.labels_fingerprint ?? '—', mono: true },
            ]}
          />
          {ds.counts?.test && Object.keys(ds.counts.test.defects).length > 0 && (
            <p className="u-xs u-muted">
              Test par défaut : {Object.entries(ds.counts.test.defects).map(([k, v]) => `${defectLabel(k)} ${v}`).join(' · ')}
            </p>
          )}
        </Panel>
        <Panel title="Hyperparamètres et exécution">
          <KeyValue
            items={[
              { label: 'Préréglage', value: String(hp.preset ?? 'personnalisé') },
              ...Object.entries(hp)
                .filter(([k]) => !['preset'].includes(k))
                .map(([k, v]) => ({ label: k, value: String(v), mono: true })),
              { label: 'Appareil', value: m.device ? `${(m.device.used ?? m.device.kind ?? '').toUpperCase()} — ${m.device.name ?? ''}${m.device.amp ? ' (précision mixte)' : ''}` : '—' },
              { label: 'Poids initiaux', value: m.pretrained?.init_from ? `modèle ${m.pretrained.init_from}` : m.pretrained?.loaded ? 'ImageNet (local)' : 'aléatoires' },
              { label: 'Durée', value: formatDuration(m.duration_s) },
              { label: 'Créé le', value: formatDateTime(m.created_at) },
              ...(m.parent_id ? [{ label: 'Prolonge', value: m.parent_id, mono: true }] : []),
            ]}
          />
        </Panel>
      </div>
    </div>
  );
}

function RenameModal({ open, onClose, model }: { open: boolean; onClose: () => void; model: Model }) {
  const [name, setName] = useState(model.name);
  const rename = useRenameModel();
  const toast = useToast();
  return (
    <Modal
      open={open}
      onClose={onClose}
      title="Renommer le modèle"
      size="sm"
      footer={
        <>
          <Button variant="ghost" onClick={onClose}>
            Annuler
          </Button>
          <Button
            variant="primary"
            loading={rename.isPending}
            disabled={!name.trim()}
            onClick={async () => {
              try {
                await rename.mutateAsync({ id: model.id, name: name.trim() });
                onClose();
              } catch (e) {
                toast.error(e);
              }
            }}
          >
            Renommer
          </Button>
        </>
      }
    >
      <Field label="Nom">
        <TextInput value={name} onChange={(e) => setName(e.target.value)} autoFocus />
      </Field>
    </Modal>
  );
}

function ResumeModal({ open, onClose, model }: { open: boolean; onClose: () => void; model: Model }) {
  const navigate = useNavigate();
  const toast = useToast();
  const resume = useResume();
  const total = Number(model.hparams.epochs ?? 10);
  const [mode, setMode] = useState<'continue' | 'extend'>(model.resumable ? 'continue' : 'extend');
  const [epochs, setEpochs] = useState<number | null>(model.resumable ? total : Math.max(1, Math.floor(total / 2)));
  const [lr, setLr] = useState<number | null>(null);
  return (
    <Modal
      open={open}
      onClose={onClose}
      title="Reprendre l'entraînement"
      size="md"
      footer={
        <>
          <Button variant="ghost" onClick={onClose}>
            Annuler
          </Button>
          <Button
            variant="primary"
            icon={RotateCcw}
            loading={resume.isPending}
            onClick={async () => {
              try {
                const out = await resume.mutateAsync({ id: model.id, mode, epochs, hparams: lr ? { lr } : {} });
                onClose();
                toast.success(mode === 'continue' ? 'Reprise lancée' : 'Prolongation lancée (nouveau modèle)');
                navigate(`/entrainement/modeles/${encodeURIComponent(out.model_id)}`);
              } catch (e) {
                toast.error(e);
              }
            }}
          >
            Lancer
          </Button>
        </>
      }
    >
      <div className="u-stack-4">
        <SegmentedControl
          value={mode}
          onChange={(v) => {
            setMode(v);
            setEpochs(v === 'continue' ? total : Math.max(1, Math.floor(total / 2)));
          }}
          options={[
            { value: 'continue', label: 'Continuer', disabled: !model.resumable },
            { value: 'extend', label: 'Prolonger', disabled: model.status !== 'trained' && !model.resumable },
          ]}
        />
        <p className="u-small u-muted">
          {mode === 'continue'
            ? `Reprise exacte depuis la dernière époque terminée (${model.epochs_done}) sur ce même modèle.`
            : 'Crée un nouveau modèle initialisé avec les meilleurs poids de celui-ci (même découpage), sans gel ni échauffement.'}
        </p>
        <div className={s.fields3}>
          <Field label={mode === 'continue' ? 'Nombre total d’époques' : 'Époques supplémentaires'}>
            <NumberInput value={epochs} onChange={setEpochs} min={mode === 'continue' ? Math.max(1, model.epochs_done + 1) : 1} max={1000} />
          </Field>
          <Field label="Taux d'apprentissage" hint="vide = inchangé">
            <NumberInput value={lr} onChange={setLr} min={0.000001} max={0.5} step={0.0001} decimals={6} allowEmpty placeholder={String(model.hparams.lr ?? '')} />
          </Field>
        </div>
      </div>
    </Modal>
  );
}
