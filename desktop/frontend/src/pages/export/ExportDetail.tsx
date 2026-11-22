/** Export d'un modèle : lancement, parité PyTorch ↔ ONNX, FP32 / INT8, contenu du paquet, téléchargement, banc de mesure. */
import { useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { AlertTriangle, CheckCircle2, Download, PackageOpen, XCircle } from 'lucide-react';
import { JobProgress } from '@/components/JobProgress';
import {
  Badge,
  Button,
  Callout,
  Checkbox,
  ErrorState,
  Field,
  KeyValue,
  LoadingState,
  NumberInput,
  PageHeader,
  Panel,
  Stat,
  useToast,
} from '@/components/ui';
import { formatBytes, formatDateTime, formatDuration, formatNumber, formatPercent } from '@/lib/format';
import { useModel, useOptions } from '@/pages/entrainement/api';
import { Int8Badge, archLabel } from '@/pages/entrainement/common';
import { downloadUrl, useExportStatus, useStartExport } from './api';
import { BenchPanel } from './Bench';
import type { ExportInfo, Int8Info, ParityFp32, VerifyReport } from './types';
import s from './export.module.css';

const sci = (v: number | null | undefined) => (v == null ? '—' : v === 0 ? '0' : v.toExponential(1).replace('.', ','));

export function ExportDetail() {
  const { modelId = '' } = useParams();
  const toast = useToast();
  const model = useModel(modelId);
  const options = useOptions();
  const status = useExportStatus(modelId);
  const start = useStartExport();
  const [int8, setInt8] = useState(true);
  const [nRef, setNRef] = useState<number | null>(6);
  const [nCal, setNCal] = useState<number | null>(32);
  const [nPar, setNPar] = useState<number | null>(16);

  if (model.isLoading || status.isLoading) return <LoadingState />;
  if (model.error || !model.data) return <ErrorState error={model.error} onRetry={() => model.refetch()} />;
  if (status.error || !status.data) return <ErrorState error={status.error} onRetry={() => status.refetch()} />;
  const m = model.data;
  const st = status.data;
  const e = st.export;
  const arch = options.data?.architectures.find((a) => a.id === m.arch);
  const exportJob = st.active_jobs.find((j) => j.kind === 'export');
  const canExport = m.kind === 'classifier' && m.status === 'trained';

  const launch = async () => {
    try {
      await start.mutateAsync({
        id: m.id,
        variants: int8 ? ['fp32', 'int8'] : ['fp32'],
        n_reference: nRef ?? 6,
        n_calibration_eggs: nCal ?? 32,
        n_parity_eggs: nPar ?? 16,
      });
      toast.info('Export lancé');
    } catch (err) {
      toast.error(err);
    }
  };

  return (
    <>
      <PageHeader
        breadcrumbs={[{ label: 'Export', to: '/export' }, { label: m.name }]}
        title={`Export — ${m.name}`}
        subtitle={
          <span className="u-row">
            <span>
              {archLabel(options.data?.architectures, m.arch)} · {m.input_size} px · {m.n_views} vues · seuil {formatNumber(m.threshold, 4)}
            </span>
            {arch && <Int8Badge friendly={arch.int8_friendly} />}
            <Link to={`/entrainement/modeles/${encodeURIComponent(m.id)}/rapport`} className="u-small">
              rapport du modèle
            </Link>
          </span>
        }
        actions={
          st.exported ? (
            <a href={downloadUrl(m.id)} download className="btn btn--primary btn--md">
              <Download size={15} aria-hidden />
              <span className="btn__label">Télécharger le paquet ({formatBytes(e.zip_size_bytes)})</span>
            </a>
          ) : undefined
        }
      />
      <div className={s.mainGrid}>
        <div className="u-stack-6">
          {!canExport && (
            <Callout tone="warning">
              {m.kind === 'anomaly' ? "L'export ONNX du modèle de référence par anomalies n'est pas pris en charge." : 'Le modèle doit être entraîné avant l’export.'}
            </Callout>
          )}
          {st.outdated && (
            <Callout tone="warning" icon={AlertTriangle} title="Export obsolète">
              Le seuil du modèle ({formatNumber(st.current_threshold, 4)}) a changé depuis l'export (paquet : {formatNumber(e.threshold, 4)}). Relancer l'export pour
              mettre le paquet à jour.
            </Callout>
          )}
          {exportJob && (
            <Panel title="Export en cours">
              <JobProgress jobId={exportJob.id} hideTitle onDone={() => status.refetch()} />
            </Panel>
          )}
          {!st.exported && !exportJob && canExport && (
            <Callout tone="info" icon={PackageOpen} title="Pas encore exporté">
              L'export produit le modèle ONNX FP32 (et INT8), vérifie la parité avec PyTorch sur des œufs réels du projet et assemble le paquet de déploiement.
            </Callout>
          )}
          {st.exported && <ExportResult e={e} int8Friendly={arch?.int8_friendly} />}
          {st.exported && <BenchPanel modelId={m.id} variants={e.variants ?? ['fp32']} nViews={e.n_views ?? m.n_views} />}
        </div>

        <div className={s.side}>
          <Panel title={st.exported ? 'Ré-exporter' : 'Exporter'} description="ONNX opset 17, entrée « views » [B, V, 3, S, S], sorties « egg_scores » et « view_scores ».">
            <div className="u-stack-4">
              <Checkbox checked disabled label="FP32 (référence, parité vérifiée)" onChange={() => undefined} />
              <Checkbox
                checked={int8}
                onChange={setInt8}
                label="INT8 (quantification statique)"
                description={
                  arch && !arch.int8_friendly ? 'Architecture peu adaptée : l’INT8 risque de dégrader les scores (alerte automatique).' : 'Calibrée sur des vues d’entraînement.'
                }
              />
              <div className={s.fields3}>
                <Field label="Œufs de référence">
                  <NumberInput value={nRef} onChange={setNRef} min={1} max={50} />
                </Field>
                <Field label="Calibration INT8">
                  <NumberInput value={nCal} onChange={setNCal} min={1} max={1000} />
                </Field>
                <Field label="Contrôle parité">
                  <NumberInput value={nPar} onChange={setNPar} min={1} max={500} />
                </Field>
              </div>
              <Button variant="primary" block onClick={launch} loading={start.isPending} disabled={!canExport || !!exportJob}>
                {st.exported ? 'Relancer l’export' : 'Exporter le modèle'}
              </Button>
            </div>
          </Panel>
          {st.exported && <PackageContent e={e} />}
        </div>
      </div>
    </>
  );
}

function ExportResult({ e, int8Friendly }: { e: ExportInfo; int8Friendly?: boolean }) {
  const p = e.parity;
  return (
    <>
      <div className={s.statsGrid}>
        <Stat label="Version du paquet" value={e.version ?? '—'} hint={formatDateTime(e.exported_at)} />
        <Stat label="Parité FP32" value={sci(p?.fp32.max_abs_diff)} hint={`tolérance ${sci(p?.fp32.tolerance)}`} tone={p?.fp32.ok ? 'success' : 'danger'} icon={p?.fp32.ok ? CheckCircle2 : XCircle} />
        <Stat label="Seuil embarqué" value={formatNumber(e.threshold, 4)} hint="éjecter si score ≥ seuil" />
        <Stat label="Durée de l'export" value={formatDuration(e.duration_s)} hint={`zip ${formatBytes(e.zip_size_bytes)}`} />
      </div>
      {(e.warnings ?? []).filter((w) => !w.startsWith('INT8 dégrade')).map((w) => (
        <Callout key={w} tone="warning">
          {w}
        </Callout>
      ))}
      <div className={s.twoCols}>
        {p && <ParityPanel fp32={p.fp32} verify={p.package_verify.fp32} />}
        <Int8Panel info={p?.int8 ?? null} verify={p?.package_verify.int8 ?? null} int8Friendly={int8Friendly} />
      </div>
    </>
  );
}

function ParityPanel({ fp32, verify }: { fp32: ParityFp32; verify: VerifyReport | null }) {
  return (
    <Panel title="Parité PyTorch ↔ ONNX (FP32)" description="Mêmes œufs réels du projet, mêmes vues, prétraitement partagé.">
      <div className="u-stack-4">
        <div className={s.verdict}>
          {fp32.ok ? <CheckCircle2 size={28} className={s.ok} aria-hidden /> : <XCircle size={28} className={s.ko} aria-hidden />}
          <div>
            <div className={s.verdictValue}>{sci(fp32.max_abs_diff)}</div>
            <div className="u-small u-muted">
              écart absolu max · tolérance {sci(fp32.tolerance)} · {fp32.ok ? 'conforme' : 'NON conforme'}
            </div>
          </div>
        </div>
        <table className={s.table}>
          <thead>
            <tr>
              <th scope="col" style={{ textAlign: 'left' }}>
                Vues par œuf (axe dynamique)
              </th>
              <th>Œufs</th>
              <th>Écart score œuf</th>
              <th>Écart scores vues</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(fp32.by_n_views).map(([v, r]) => (
              <tr key={v}>
                <th scope="row">V = {v}</th>
                <td>{r.n_eggs}</td>
                <td>{sci(r.max_abs_diff)}</td>
                <td>{sci(r.max_abs_diff_views)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {verify && (
          <p className="u-small">
            <span className={verify.ok ? s.ok : s.ko}>{verify.ok ? '✓' : '✗'}</span> Paquet vérifié par <code>python infer.py --verify</code> (processus séparé) :{' '}
            {verify.n_eggs} œufs de référence, écart {sci(verify.max_abs_diff)} avec les scores PyTorch.
          </p>
        )}
      </div>
    </Panel>
  );
}

function Int8Panel({ info, verify, int8Friendly }: { info: Int8Info | null; verify: VerifyReport | null; int8Friendly?: boolean }) {
  if (!info)
    return (
      <Panel title="Variante INT8">
        <p className="u-muted u-small">Non demandée lors de cet export.</p>
      </Panel>
    );
  if (info.error)
    return (
      <Panel title="Variante INT8">
        <Callout tone="warning" title="Quantification impossible">
          {info.error} — le paquet ne contient que la variante FP32.
        </Callout>
      </Panel>
    );
  const t = info.test;
  return (
    <Panel title="FP32 ↔ INT8" description={`${info.format} · poids ${info.weights} · activations ${info.activations} · calibration ${info.calibration} sur ${info.n_calibration_eggs} œufs`}>
      <div className="u-stack-4">
        {info.alert ? (
          <Callout tone="danger" icon={AlertTriangle} title="INT8 dégrade les performances">
            {info.alert}
            {int8Friendly === false && ' Cette architecture se quantifie mal : choisir EfficientNet-Lite0 (préréglage « Déploiement INT8 ») pour un déploiement INT8.'}
          </Callout>
        ) : (
          <Callout tone="success" icon={CheckCircle2}>
            INT8 fidèle à FP32 sur le jeu de test (écart moyen {sci(info.vs_fp32?.mean_abs_diff)}).
          </Callout>
        )}
        {t && (
          <table className={s.table}>
            <thead>
              <tr>
                <th scope="col" style={{ textAlign: 'left' }}>
                  Test ({t.n} œufs)
                </th>
                <th>FP32</th>
                <th>INT8</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <th scope="row">AUC ROC</th>
                <td>{formatNumber(t.fp32.auc_roc, 3)}</td>
                <td className={diffClass(t.fp32.auc_roc, t.int8.auc_roc)}>{formatNumber(t.int8.auc_roc, 3)}</td>
              </tr>
              <tr>
                <th scope="row">Rappel non viables</th>
                <td>{formatPercent(t.fp32.recall)}</td>
                <td className={diffClass(t.fp32.recall, t.int8.recall)}>{formatPercent(t.int8.recall)}</td>
              </tr>
              <tr>
                <th scope="row">Faux rejets</th>
                <td>{formatPercent(t.fp32.false_reject_rate, 2)}</td>
                <td className={diffClass(t.int8.false_reject_rate, t.fp32.false_reject_rate)}>{formatPercent(t.int8.false_reject_rate, 2)}</td>
              </tr>
              <tr>
                <th scope="row">Écart de score max (test)</th>
                <td>—</td>
                <td>{formatNumber(t.max_abs_diff, 4)}</td>
              </tr>
            </tbody>
          </table>
        )}
        <KeyValue
          items={[
            { label: 'Écart moyen / max (œufs de parité)', value: `${formatNumber(info.vs_fp32?.mean_abs_diff, 4)} / ${formatNumber(info.vs_fp32?.max_abs_diff, 4)}` },
            { label: 'infer.py --verify --variant int8', value: verify ? `écart ${formatNumber(verify.max_abs_diff, 4)} (informatif)` : '—' },
            { label: 'Taille du modèle INT8', value: formatBytes(info.size_bytes) },
            { label: 'Nœuds laissés en flottant', value: `${info.excluded_nodes?.length ?? 0} (tête de classification et agrégation)` },
          ]}
        />
        <p className="u-xs u-muted">Variante par défaut du paquet : FP32. L'INT8 est à choisir explicitement sur le Pi.</p>
      </div>
    </Panel>
  );
}

function diffClass(ref: number | null | undefined, v: number | null | undefined) {
  if (ref == null || v == null) return undefined;
  return ref - v > 0.02 ? s.ko : undefined;
}

function PackageContent({ e }: { e: ExportInfo }) {
  const files: [string, string, number | undefined][] = [
    ['inference_config.json', 'prétraitement, vues, seuil, couloirs, échelle, métriques', undefined],
    ...Object.entries(e.files ?? {}).map(([k, f]) => [f.file, `modèle ${k.toUpperCase()}`, f.size_bytes] as [string, string, number]),
    ['preprocess.py', 'prétraitement (copie exacte du code partagé)', undefined],
    ['infer.py', 'moteur autonome : --column, --bench, --verify', undefined],
    ['reference/', 'œufs de référence (PNG) + scores PyTorch attendus', undefined],
    ['README.txt', 'mode d’emploi sur le Pi', undefined],
  ];
  return (
    <Panel title="Contenu du paquet" description={e.zip?.split('/').pop()}>
      <ul className={s.files}>
        {files.map(([f, d, size]) => (
          <li key={f}>
            <code>{f}</code>
            <span className="u-muted u-xs">{d}</span>
            <span className="u-xs u-num">{size ? formatBytes(size) : ''}</span>
          </li>
        ))}
      </ul>
      <p className="u-xs u-muted">
        Opset {e.opset} · {e.n_views} vues · {e.input_size} px · export n° {e.export_count}. Le Pi n'a besoin que de numpy, opencv-python-headless et onnxruntime.
      </p>
      <div className="u-row">
        <Badge tone="success">ONNX Runtime (CPU)</Badge>
        <Badge tone="neutral">Hailo — prévu</Badge>
        <Badge tone="neutral">TFLite — prévu</Badge>
      </div>
    </Panel>
  );
}
