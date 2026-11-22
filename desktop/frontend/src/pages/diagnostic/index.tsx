import { useMemo } from 'react';
import { Copy, Cpu, Gauge, PlugZap, RotateCw, Unplug, Zap } from 'lucide-react';
import { useDiagnostics, useDiagnosticsLog } from '@/api/hooks';
import type { Diagnostics } from '@/api/types';
import {
  Badge,
  Button,
  Callout,
  DataTable,
  ErrorState,
  IconButton,
  KeyValue,
  LoadingState,
  Page,
  PageHeader,
  Panel,
  useToast,
  type Column,
} from '@/components/ui';
import { LogView } from '@/components/LogView';
import s from './diagnostic.module.css';

const DEVICE_META: Record<Diagnostics['device']['kind'], { label: string; tone: 'success' | 'info' | 'neutral'; hint: string }> = {
  cuda: { label: 'CUDA', tone: 'success', hint: 'Accélération GPU NVIDIA disponible.' },
  mps: { label: 'MPS', tone: 'info', hint: 'Accélération Apple Silicon (Metal) disponible.' },
  cpu: { label: 'CPU', tone: 'neutral', hint: 'Aucun accélérateur détecté : les entraînements seront plus lents.' },
};

function DeviceCard({ d }: { d: Diagnostics['device'] }) {
  const meta = DEVICE_META[d.kind] ?? DEVICE_META.cpu;
  const Icon = d.kind === 'cpu' ? Cpu : Zap;
  return (
    <Panel title="Accélérateur de calcul">
      <div className={s.hero}>
        <span className={s.heroIcon} data-tone={meta.tone}>
          <Icon size={20} aria-hidden />
        </span>
        <div className="u-grow">
          <div className="u-row">
            <span className={s.heroTitle}>{d.name || meta.label}</span>
            <Badge tone={meta.tone}>{meta.label}</Badge>
          </div>
          <div className="u-small u-muted">{meta.hint}</div>
        </div>
      </div>
    </Panel>
  );
}

function AgentCard({ a }: { a: Diagnostics['agent'] }) {
  return (
    <Panel title="Agent Raspberry Pi">
      <div className={s.hero}>
        <span className={s.heroIcon} data-tone={a?.connected ? 'success' : 'neutral'}>
          {a?.connected ? <PlugZap size={20} aria-hidden /> : <Unplug size={20} aria-hidden />}
        </span>
        <div className="u-grow">
          <div className="u-row">
            <span className={s.heroTitle}>{a == null ? 'Non configuré' : a.connected ? 'Connecté' : 'Injoignable'}</span>
            {a?.version && <Badge tone="neutral">v{a.version}</Badge>}
          </div>
          <div className="u-small u-muted u-mono u-break">{a?.url ?? 'Aucune adresse d’agent renseignée.'}</div>
          {a?.detail && <div className="u-xs u-subtle">{a.detail}</div>}
        </div>
      </div>
    </Panel>
  );
}

function buildReport(d: Diagnostics): string {
  const lines = [
    `EggVision Studio ${d.app_version}`,
    `Python ${d.python} — ${d.platform} (${d.machine})`,
    `Accélérateur : ${d.device.kind.toUpperCase()} — ${d.device.name}`,
    `Agent Pi : ${d.agent ? (d.agent.connected ? `connecté ${d.agent.url ?? ''} ${d.agent.version ?? ''}` : `injoignable ${d.agent.url ?? ''} ${d.agent.detail ?? ''}`) : 'non configuré'}`,
    `Projet : ${d.project ? `${d.project.name} (${d.project.path})` : 'aucun'}`,
    `Données : ${d.data_dir}`,
    `Journal : ${d.log_file}`,
    'Paquets :',
    ...Object.entries(d.packages).map(([k, v]) => `  ${k} ${v ?? '(absent)'}`),
  ];
  return lines.join('\n');
}

export default function DiagnosticPage() {
  const diag = useDiagnostics();
  const log = useDiagnosticsLog(300);
  const toast = useToast();

  const packages = useMemo(
    () => Object.entries(diag.data?.packages ?? {}).map(([name, version]) => ({ name, version })),
    [diag.data],
  );

  const copy = async (text: string, what: string) => {
    try {
      await navigator.clipboard.writeText(text);
      toast.success(`${what} copié`);
    } catch {
      toast.error('Copie impossible', 'Le presse-papiers n’est pas accessible.');
    }
  };

  const pkgColumns: Column<{ name: string; version: string | null }>[] = [
    { key: 'name', header: 'Paquet', sortValue: (p) => p.name, render: (p) => <span className="u-mono">{p.name}</span> },
    {
      key: 'version',
      header: 'Version',
      align: 'right',
      render: (p) => (p.version ? <span className="u-mono u-num">{p.version}</span> : <Badge tone="warning">Absent</Badge>),
    },
  ];

  const d = diag.data;

  return (
    <Page>
      <PageHeader
        title="Diagnostic"
        subtitle="État de l’installation, du matériel et de la connexion au Raspberry Pi."
        actions={
          <>
            <Button icon={Copy} onClick={() => d && copy(buildReport(d), 'Rapport')} disabled={!d}>
              Copier le rapport
            </Button>
            <IconButton icon={RotateCw} label="Rafraîchir" variant="secondary" loading={diag.isFetching} onClick={() => diag.refetch()} />
          </>
        }
      />

      {diag.isLoading ? (
        <LoadingState />
      ) : diag.isError || !d ? (
        <ErrorState title="Diagnostic indisponible" error={diag.error} onRetry={() => diag.refetch()} />
      ) : (
        <div className="u-stack-4">
          <div className="u-grid-2">
            <DeviceCard d={d.device} />
            <AgentCard a={d.agent} />
          </div>
          <div className={s.grid}>
            <Panel title="Application">
              <KeyValue
                items={[
                  { label: 'Version', value: d.app_version, mono: true },
                  { label: 'Python', value: d.python, mono: true },
                  { label: 'Système', value: d.platform },
                  { label: 'Architecture', value: d.machine, mono: true },
                  { label: 'Dossier de données', value: d.data_dir, mono: true },
                  { label: 'Fichier journal', value: d.log_file, mono: true },
                ]}
              />
            </Panel>
            <Panel title="Projet">
              {d.project ? (
                <KeyValue
                  items={[
                    { label: 'Nom', value: d.project.name },
                    { label: 'Chemin', value: d.project.path, mono: true },
                  ]}
                />
              ) : (
                <Callout tone="neutral" icon={Gauge}>
                  Aucun projet ouvert.
                </Callout>
              )}
            </Panel>
            <Panel title="Paquets" flush className={s.packages}>
              <DataTable columns={pkgColumns} rows={packages} rowKey={(p) => p.name} dense stickyHeader defaultSort={{ key: 'name', dir: 'asc' }} />
            </Panel>
          </div>
        </div>
      )}

      <Panel
        className={s.logPanel}
        title="Journal de l’application"
        description={d ? <span className="u-mono">{d.log_file}</span> : '300 dernières lignes'}
        flush
        actions={
          <>
            <Button size="sm" icon={Copy} onClick={() => copy(log.data?.text ?? '', 'Journal')} disabled={!log.data?.text}>
              Copier
            </Button>
            <Button size="sm" icon={RotateCw} onClick={() => log.refetch()} loading={log.isFetching}>
              Rafraîchir
            </Button>
          </>
        }
      >
        {log.isLoading ? (
          <LoadingState compact />
        ) : log.isError ? (
          <ErrorState error={log.error} onRetry={() => log.refetch()} compact />
        ) : (
          <LogView text={log.data?.text ?? ''} height={420} />
        )}
      </Panel>
    </Page>
  );
}
