/** Connexion à une source (agent Pi, caméra USB, simulation) et indicateurs d'état en continu. */
import { useEffect, useState } from 'react';
import { Camera, Cpu, HardDrive, Plug, Radar, RefreshCw, Thermometer, Unplug, Wifi } from 'lucide-react';
import { errorMessage } from '@/api/client';
import {
  Badge,
  Button,
  Callout,
  Field,
  IconButton,
  NumberInput,
  Panel,
  SegmentedControl,
  Select,
  TextInput,
  Tooltip,
  useToast,
} from '@/components/ui';
import { formatBytes, formatNumber } from '@/lib/format';
import { discoverAgents, useConnect, useDisconnect, useUsbDevices } from './api';
import type { CaptureStatus, DiscoveredAgent, SourceKind } from './types';
import s from './capture.module.css';

const KIND_OPTIONS: { value: SourceKind; label: string; title: string }[] = [
  { value: 'pi', label: 'Agent Raspberry Pi', title: 'Caméra Global Shutter pilotée par l’agent Pi' },
  { value: 'usb', label: 'Caméra USB', title: 'Caméra UVC / USB3 branchée sur ce PC' },
  { value: 'simulated', label: 'Simulation', title: 'Images synthétiques ou dossier d’images' },
];

export const KIND_LABEL: Record<SourceKind, string> = { pi: 'Agent Pi', usb: 'Caméra USB', simulated: 'Simulation' };

export function ConnectionPanel({ status }: { status: CaptureStatus }) {
  return status.connected && status.source ? <ConnectedBar status={status} /> : <ConnectForm status={status} />;
}

// ---------------------------------------------------------------- Formulaire de connexion
function ConnectForm({ status }: { status: CaptureStatus }) {
  const toast = useToast();
  const connect = useConnect();
  const [kind, setKind] = useState<SourceKind>('pi');
  const [address, setAddress] = useState('');
  const [agents, setAgents] = useState<DiscoveredAgent[] | null>(null);
  const [discovering, setDiscovering] = useState(false);
  const [device, setDevice] = useState<number | null>(0);
  const [scale, setScale] = useState<'0.25' | '0.5' | '1'>('0.5');
  const [folder, setFolder] = useState('');
  const usb = useUsbDevices(kind === 'usb');

  useEffect(() => {
    if (!address && status.last_agent_url) setAddress(status.last_agent_url);
  }, [status.last_agent_url]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const first = usb.data?.[0];
    if (first && !usb.data?.some((d) => d.device === device)) setDevice(first.device);
  }, [usb.data]); // eslint-disable-line react-hooks/exhaustive-deps

  const discover = async () => {
    setDiscovering(true);
    try {
      const res = await discoverAgents(2);
      setAgents(res.agents);
      if (res.agents.length === 1) setAddress(res.agents[0].url);
    } catch (e) {
      toast.error(e);
    } finally {
      setDiscovering(false);
    }
  };

  const submit = async () => {
    try {
      if (kind === 'pi') {
        if (!address.trim()) return toast.warning('Adresse manquante', 'Saisir l’adresse IP de l’agent ou lancer la découverte.');
        await connect.mutateAsync({ kind, url: address.trim() });
      } else if (kind === 'usb') {
        await connect.mutateAsync({ kind, device: device ?? 0 });
      } else {
        await connect.mutateAsync({ kind, scale: Number(scale), folder: folder.trim() || undefined });
      }
      toast.success('Source connectée');
    } catch (e) {
      toast.error('Connexion impossible', errorMessage(e));
    }
  };

  return (
    <Panel
      title="Source de prise de vue"
      description="Une seule source active à la fois. Les lots de l’agent Pi sont synchronisés vers le projet à la fin de chaque lot."
    >
      <div className={s.connectGrid}>
        <SegmentedControl value={kind} onChange={setKind} options={KIND_OPTIONS} />
        {kind === 'pi' && (
          <div className={s.connectRow}>
            <Field label="Adresse de l’agent" hint="IP ou nom d’hôte ; port 8770 par défaut (point d’accès Wi-Fi : 10.42.0.1)." className="u-grow">
              <TextInput
                value={address}
                onChange={(e) => setAddress(e.target.value)}
                placeholder="192.168.4.1 ou http://eggpi.local:8770"
                onKeyDown={(e) => e.key === 'Enter' && submit()}
                leading={<Wifi size={14} />}
              />
            </Field>
            <Button icon={Radar} onClick={discover} loading={discovering}>
              Découvrir
            </Button>
          </div>
        )}
        {kind === 'pi' && agents && (
          <div className={s.agentList} role="list" aria-label="Agents découverts">
            {agents.length === 0 ? (
              <span className="u-muted u-small">
                Aucun agent annoncé sur le réseau (multicast filtré ?). Saisir l’adresse IP à la main.
              </span>
            ) : (
              agents.map((a) => (
                <button
                  type="button"
                  role="listitem"
                  key={a.url}
                  className={s.agentChip}
                  data-active={a.url === address || undefined}
                  onClick={() => setAddress(a.url)}
                >
                  <Cpu size={14} />
                  <span className="u-strong">{a.hostname || a.name}</span>
                  <span className="u-mono u-xs u-muted">{a.url.replace('http://', '')}</span>
                  {a.version && <Badge tone="neutral">v{a.version}</Badge>}
                </button>
              ))
            )}
          </div>
        )}
        {kind === 'usb' && (
          <div className={s.connectRow}>
            <Field label="Caméra" className="u-grow" hint={usb.isFetching ? 'Recherche des caméras…' : usb.error ? errorMessage(usb.error) : undefined}>
              {usb.data && usb.data.length > 0 ? (
                <Select
                  value={String(device ?? 0)}
                  onChange={(v) => setDevice(Number(v))}
                  options={usb.data.map((d) => ({ value: String(d.device), label: `${d.label} — ${d.width}×${d.height}` }))}
                />
              ) : (
                <NumberInput value={device} onChange={setDevice} min={0} max={15} step={1} aria-label="Indice de la caméra" />
              )}
            </Field>
            <IconButton icon={RefreshCw} label="Rechercher les caméras" onClick={() => usb.refetch()} loading={usb.isFetching} />
          </div>
        )}
        {kind === 'usb' && (
          <Callout tone="info">
            Réglages manuels selon le pilote : l’exposition n’est pas réglable sous macOS ; les gains rouge/bleu et la ROI sont appliqués en logiciel.
          </Callout>
        )}
        {kind === 'simulated' && (
          <div className={s.connectRow}>
            <Field label="Résolution">
              <SegmentedControl
                value={scale}
                onChange={setScale}
                options={[
                  { value: '0.25', label: '¼ (368×272)' },
                  { value: '0.5', label: '½ (728×544)' },
                  { value: '1', label: 'Pleine (1456×1088)' },
                ]}
              />
            </Field>
            <Field label="Dossier d’images (facultatif)" hint="Relu en boucle ; vide = images synthétiques." className="u-grow">
              <TextInput value={folder} onChange={(e) => setFolder(e.target.value)} placeholder="/chemin/vers/images ou un lot" />
            </Field>
          </div>
        )}
        <div className={s.connectActions}>
          <Button variant="primary" icon={Plug} onClick={submit} loading={connect.isPending}>
            Connecter
          </Button>
        </div>
      </div>
    </Panel>
  );
}

// ---------------------------------------------------------------- Barre d'état (connecté)
function ConnectedBar({ status }: { status: CaptureStatus }) {
  const toast = useToast();
  const disconnect = useDisconnect();
  const src = status.source!;
  const fps = src.actual_fps ?? 0;
  const target = src.camera?.settings?.fps ?? 0;
  const temp = src.device?.temperature_c;
  const free = src.storage?.free_bytes;
  const localCapture = status.capturing && status.kind !== 'pi';
  const offset = src.clock_offset_s ?? 0;
  const fpsTone = !fps ? 'danger' : target && Math.abs(fps - target) / target > 0.03 ? 'warning' : 'success';

  return (
    <div className={s.statusBar}>
      <div className={s.statusMain}>
        <span className={s.statusDot} data-tone={src.connected === false ? 'danger' : src.ready ? 'success' : 'warning'} aria-hidden />
        <div className="u-col">
          <span className="u-strong u-truncate">{src.label}</span>
          <span className="u-xs u-muted">
            {KIND_LABEL[src.kind]}
            {src.version ? ` · agent v${src.version}` : ''}
            {src.camera?.model ? ` · ${src.camera.model}` : ''}
            {src.camera?.output_width ? ` · ${src.camera.output_width}×${src.camera.output_height}` : ''}
          </span>
        </div>
      </div>
      <div className={s.indicators}>
        <Indicator icon={Camera} label="Cadence réelle" value={`${formatNumber(fps, 1)} im/s`} tone={fpsTone} />
        {src.kind === 'pi' && (
          <Indicator
            icon={Thermometer}
            label="Température du Pi"
            value={temp == null ? '—' : `${formatNumber(temp, 1)} °C`}
            tone={temp == null ? 'neutral' : temp >= 80 ? 'danger' : temp >= 70 ? 'warning' : 'success'}
          />
        )}
        <Indicator
          icon={HardDrive}
          label={src.kind === 'pi' ? 'Espace libre sur le SSD du Pi' : 'Espace libre (dossier du projet)'}
          value={free == null ? '—' : formatBytes(free)}
          tone={free == null ? 'neutral' : free < 5e9 ? 'danger' : free < 20e9 ? 'warning' : 'success'}
        />
        {src.kind === 'pi' && Math.abs(offset) > 5 && (
          <Tooltip content="Horloge du Pi décalée par rapport au PC (pas de NTP ?). Les dates des lots sont fournies par le PC.">
            <Badge tone="warning">Horloge {offset > 0 ? '+' : ''}{formatNumber(offset, 0)} s</Badge>
          </Tooltip>
        )}
        {src.error && <Badge tone="danger">{src.error}</Badge>}
        {!src.ready && !src.error && <Badge tone="warning">Caméra en préparation…</Badge>}
      </div>
      <Tooltip content={localCapture ? 'Arrêter le lot en cours avant de déconnecter.' : 'Déconnecter la source'}>
        <Button
          icon={Unplug}
          size="sm"
          disabled={localCapture}
          loading={disconnect.isPending}
          onClick={() => disconnect.mutate(undefined, { onError: (e) => toast.error(e) })}
        >
          Déconnecter
        </Button>
      </Tooltip>
    </div>
  );
}

function Indicator({ icon: Icon, label, value, tone }: { icon: typeof Camera; label: string; value: string; tone: string }) {
  return (
    <Tooltip content={label}>
      <span className={s.indicator} data-tone={tone}>
        <Icon size={14} aria-hidden />
        <span className="u-num">{value}</span>
        <span className="u-sr-only">{label}</span>
      </span>
    </Tooltip>
  );
}
