/** Découpages entraînement / validation / test : liste, création assistée avec aperçu, détail et ajustement manuel. */
import { useEffect, useMemo, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { Lock, Plus, Save, Trash2 } from 'lucide-react';
import {
  Badge,
  Button,
  Callout,
  ConfirmDialog,
  DataTable,
  EmptyState,
  ErrorState,
  Field,
  LoadingState,
  NumberInput,
  PageHeader,
  Panel,
  SegmentedControl,
  Select,
  TextInput,
  Toggle,
  useToast,
  type Column,
} from '@/components/ui';
import { defectLabel, lotTypeLabel } from '@/lib/domain';
import { formatDate, formatDateTime, formatNumber } from '@/lib/format';
import { useCreateSplit, useDebounced, useDeleteSplit, useOptions, usePatchAssignments, useSplit, useSplitPreview, useSplits } from './api';
import { SUBSET_LABEL, SUBSETS, SplitWarnings, SubsetBadge, SummaryTable } from './common';
import type { LotRow, Split, SplitParams, Subset } from './types';
import s from './entrainement.module.css';

// ------------------------------------------------------------------ liste
export function SplitList() {
  const navigate = useNavigate();
  const splits = useSplits();
  const columns: Column<Split>[] = [
    {
      key: 'name',
      header: 'Découpage',
      render: (sp) => (
        <div className={s.nameCell}>
          <span className="u-strong">
            {sp.name} <span className="u-muted">v{sp.version}</span>
          </span>
          <span className="u-xs u-muted">
            {sp.strategy.group_by === 'lot' ? 'par lot' : 'par session'} · graine {sp.strategy.seed ?? 0}
            {sp.strategy.holdout_farms?.length ? ` · élevages réservés : ${sp.strategy.holdout_farms.join(', ')}` : ''}
            {sp.strategy.holdout_dates?.length ? ` · journées réservées : ${sp.strategy.holdout_dates.join(', ')}` : ''}
            {sp.strategy.manual_changes ? ` · ${sp.strategy.manual_changes} ajustement(s) manuel(s)` : ''}
          </span>
        </div>
      ),
      sortValue: (sp) => `${sp.name} ${String(sp.version).padStart(4, '0')}`,
    },
    ...SUBSETS.map<Column<Split>>((sub) => ({
      key: sub,
      header: SUBSET_LABEL[sub],
      align: 'right',
      render: (sp) => (
        <span className="u-num">
          {formatNumber(sp.summary.subsets?.[sub]?.n_eggs)} <span className="u-xs u-muted">({formatNumber(sp.summary.subsets?.[sub]?.n_lots)} lots)</span>
        </span>
      ),
      sortValue: (sp) => sp.summary.subsets?.[sub]?.n_eggs ?? 0,
    })),
    {
      key: 'warn',
      header: 'Contrôles',
      render: (sp) =>
        !sp.summary.leakage_check?.ok ? (
          <Badge tone="danger">Fuite possible</Badge>
        ) : sp.summary.warnings?.length ? (
          <Badge tone="warning">{sp.summary.warnings.length} avert.</Badge>
        ) : (
          <Badge tone="success">OK</Badge>
        ),
    },
    {
      key: 'used',
      header: 'Utilisation',
      render: (sp) =>
        sp.locked ? (
          <Badge tone="neutral" icon={Lock}>
            {sp.used_by.length} modèle(s)
          </Badge>
        ) : (
          <span className="u-subtle">libre</span>
        ),
    },
    { key: 'date', header: 'Créé le', render: (sp) => formatDateTime(sp.created_at), sortValue: (sp) => sp.created_at ?? '' },
  ];
  return (
    <div className="u-stack-4">
      <div className={s.toolbar}>
        <span className="u-muted u-small">
          Un découpage est versionné : dès qu'un modèle l'utilise, il devient immuable et toute modification crée une nouvelle version.
        </span>
        <span className="u-spacer" />
        <Button variant="primary" icon={Plus} onClick={() => navigate('/entrainement/decoupages/nouveau')}>
          Nouveau découpage
        </Button>
      </div>
      <Panel flush>
        <DataTable
          columns={columns}
          rows={splits.data?.items}
          rowKey={(sp) => sp.id}
          loading={splits.isLoading}
          error={splits.error}
          onRetry={() => splits.refetch()}
          onRowClick={(sp) => navigate(`/entrainement/decoupages/${sp.id}`)}
          empty={
            <EmptyState
              title="Aucun découpage"
              description="Le découpage répartit les lots entre entraînement, validation et test — jamais les images d'un même lot."
              actions={
                <Button variant="primary" icon={Plus} onClick={() => navigate('/entrainement/decoupages/nouveau')}>
                  Créer le premier découpage
                </Button>
              }
            />
          }
        />
      </Panel>
    </div>
  );
}

// ------------------------------------------------------------------ création
const DEFAULT_PARAMS: SplitParams = {
  group_by: 'session',
  ratios: [0.7, 0.15, 0.15],
  holdout_farms: [],
  holdout_dates: [],
  seed: 0,
  reviewed_only: false,
  exclude_lots: [],
};

function toggleIn(list: string[], v: string): string[] {
  return list.includes(v) ? list.filter((x) => x !== v) : [...list, v];
}

export function SplitEditor() {
  const navigate = useNavigate();
  const toast = useToast();
  const options = useOptions();
  const create = useCreateSplit();
  const [name, setName] = useState('Découpage principal');
  const [p, setP] = useState<SplitParams>(DEFAULT_PARAMS);
  const ratioSum = p.ratios.reduce((a, b) => a + b, 0);
  const valid = ratioSum > 0 && p.ratios[0] > 0;
  const debounced = useDebounced(valid ? p : null, 300);
  const preview = useSplitPreview(debounced);
  const set = <K extends keyof SplitParams>(k: K, v: SplitParams[K]) => setP((old) => ({ ...old, [k]: v }));
  const setRatio = (i: number, v: number | null) => {
    const r = [...p.ratios] as [number, number, number];
    r[i] = Math.max(0, (v ?? 0) / 100);
    set('ratios', r);
  };

  const save = async () => {
    try {
      const sp = await create.mutateAsync({ ...p, name: name.trim() || 'Découpage' });
      toast.success(`Découpage « ${sp.name} » v${sp.version} enregistré`);
      navigate(`/entrainement/decoupages/${sp.id}`);
    } catch (e) {
      toast.error(e);
    }
  };

  if (options.isLoading) return <LoadingState />;
  if (options.error) return <ErrorState error={options.error} onRetry={() => options.refetch()} />;
  const opt = options.data!;

  return (
    <div className={s.formGrid}>
      <div className="u-stack-4">
        <Panel title="Répartition" description="L'unité d'affectation est le lot, ou tous les lots d'une même journée de prise de vue (session).">
          <div className="u-stack-4">
            <Field label="Regrouper par">
              <SegmentedControl
                value={p.group_by}
                onChange={(v) => set('group_by', v)}
                options={[
                  { value: 'session', label: 'Session (journée)', title: 'Tous les lots d’une même journée vont dans le même sous-ensemble' },
                  { value: 'lot', label: 'Lot' },
                ]}
              />
            </Field>
            <div className={s.fields4}>
              {SUBSETS.map((sub, i) => (
                <Field key={sub} label={SUBSET_LABEL[sub]}>
                  <NumberInput value={Math.round(p.ratios[i] * 1000) / 10} onChange={(v) => setRatio(i, v)} min={0} max={100} step={5} unit="%" decimals={1} />
                </Field>
              ))}
              <Field label="Graine" hint="même graine = même répartition">
                <NumberInput value={p.seed} onChange={(v) => set('seed', v ?? 0)} min={0} max={1_000_000} step={1} />
              </Field>
            </div>
            {!valid && <Callout tone="danger">Le ratio d'entraînement doit être strictement positif.</Callout>}
            {valid && Math.abs(ratioSum - 1) > 0.001 && (
              <p className="u-xs u-muted">Les ratios sont normalisés (somme actuelle {formatNumber(ratioSum * 100, 1)} %).</p>
            )}
            <Toggle checked={p.reviewed_only} onChange={(v) => set('reviewed_only', v)} label="Labels relus uniquement" description="Ignorer les œufs dont le label n'a pas été relu." />
          </div>
        </Panel>

        <Panel title="Réserver pour le test" description="Les lots de ces élevages ou journées vont entièrement dans le jeu de test (généralisation à des conditions jamais vues).">
          <div className="u-stack-4">
            <Field label="Élevages">
              <div className={s.chips}>
                {opt.split.farms.length === 0 && <span className="u-muted u-small">Aucun élevage renseigné.</span>}
                {opt.split.farms.map((f) => (
                  <button
                    type="button"
                    key={f.farm}
                    className={`${s.chip} ${p.holdout_farms.includes(f.farm) ? s.chipOn : ''}`}
                    onClick={() => set('holdout_farms', toggleIn(p.holdout_farms, f.farm))}
                    aria-pressed={p.holdout_farms.includes(f.farm)}
                  >
                    {f.farm || '(sans élevage)'} <span className="u-xs u-muted">{formatNumber(f.n_eggs)} œufs</span>
                  </button>
                ))}
              </div>
            </Field>
            <Field label="Journées">
              <div className={s.chips}>
                {opt.split.dates.map((d) => (
                  <button
                    type="button"
                    key={d.date}
                    className={`${s.chip} ${p.holdout_dates.includes(d.date) ? s.chipOn : ''}`}
                    onClick={() => set('holdout_dates', toggleIn(p.holdout_dates, d.date))}
                    aria-pressed={p.holdout_dates.includes(d.date)}
                  >
                    {formatDate(d.date)} <span className="u-xs u-muted">{formatNumber(d.n_eggs)} œufs</span>
                  </button>
                ))}
              </div>
            </Field>
          </div>
        </Panel>

        <Panel title="Lots" description="Affectation proposée. Décocher un lot pour l'exclure du découpage." flush>
          <LotTable
            lots={preview.data?.lots ?? []}
            loading={preview.isLoading}
            excluded={p.exclude_lots}
            onToggleExclude={(id) => set('exclude_lots', toggleIn(p.exclude_lots, id))}
            extraLots={opt.split.lots.filter((l) => p.exclude_lots.includes(l.id))}
          />
        </Panel>
      </div>

      <div className={s.formSide}>
        <Panel title="Aperçu" description="Rien n'est enregistré tant que vous n'avez pas cliqué sur « Enregistrer ».">
          {preview.error ? (
            <ErrorState error={preview.error} compact />
          ) : !preview.data ? (
            <LoadingState />
          ) : (
            <div className="u-stack-4">
              <SummaryTable summary={preview.data.summary} />
              <SplitWarnings summary={preview.data.summary} holdout={p.holdout_farms.length + p.holdout_dates.length > 0} />
            </div>
          )}
        </Panel>
        <Panel>
          <div className="u-stack-4">
            <Field label="Nom" hint="Un nom existant crée la version suivante.">
              <TextInput value={name} onChange={(e) => setName(e.target.value)} />
            </Field>
            <Button variant="primary" icon={Save} block onClick={save} loading={create.isPending} disabled={!valid || !preview.data}>
              Enregistrer le découpage
            </Button>
          </div>
        </Panel>
      </div>
    </div>
  );
}

function LotTable({
  lots,
  loading,
  excluded,
  onToggleExclude,
  extraLots,
}: {
  lots: LotRow[];
  loading?: boolean;
  excluded: string[];
  onToggleExclude: (id: string) => void;
  extraLots: LotRow[];
}) {
  const rows = useMemo(() => [...lots, ...extraLots.map((l) => ({ ...l, subset: null }))], [lots, extraLots]);
  const columns: Column<LotRow>[] = [
    {
      key: 'inc',
      header: '',
      width: 36,
      render: (l) => (
        <input type="checkbox" checked={!excluded.includes(l.id)} onChange={() => onToggleExclude(l.id)} aria-label={`Inclure le lot ${l.name}`} />
      ),
    },
    ...lotColumns(),
    { key: 'subset', header: 'Sous-ensemble', render: (l) => (excluded.includes(l.id) ? <Badge>exclu</Badge> : <SubsetBadge subset={l.subset} />), sortValue: (l) => l.subset ?? 'z' },
  ];
  return <DataTable columns={columns} rows={rows} rowKey={(l) => l.id} loading={loading && rows.length === 0} dense minWidth={720} empty="Aucun lot avec des œufs utilisables." />;
}

function lotColumns(): Column<LotRow>[] {
  return [
    {
      key: 'name',
      header: 'Lot',
      width: '32%',
      render: (l) => (
        <div className={s.nameCell}>
          <span className="u-truncate">{l.name}</span>
          <span className="u-xs u-muted">{lotTypeLabel(l.type)}</span>
        </div>
      ),
      sortValue: (l) => l.name,
    },
    { key: 'farm', header: 'Élevage', render: (l) => l.farm || '—', sortValue: (l) => l.farm },
    { key: 'date', header: 'Session', render: (l) => formatDate(l.session_date), sortValue: (l) => l.session_date },
    { key: 'eggs', header: 'Œufs', align: 'right', render: (l) => formatNumber(l.n_eggs), sortValue: (l) => l.n_eggs },
    { key: 'v', header: 'Viables', align: 'right', render: (l) => formatNumber(l.viable), sortValue: (l) => l.viable },
    {
      key: 'nv',
      header: 'Non viables',
      align: 'right',
      render: (l) => (
        <span title={Object.entries(l.defects).map(([k, v]) => `${defectLabel(k)} : ${v}`).join(', ')}>{formatNumber(l.non_viable)}</span>
      ),
      sortValue: (l) => l.non_viable,
    },
  ];
}

// ------------------------------------------------------------------ détail
export function SplitDetailPage() {
  const { splitId } = useParams();
  const id = Number(splitId);
  const navigate = useNavigate();
  const toast = useToast();
  const split = useSplit(Number.isFinite(id) ? id : null);
  const patch = usePatchAssignments();
  const del = useDeleteSplit();
  const [moveGroup, setMoveGroup] = useState(true);
  const [confirmDelete, setConfirmDelete] = useState(false);
  useEffect(() => setMoveGroup(true), [id]);

  if (split.isLoading) return <LoadingState />;
  if (split.error || !split.data) return <ErrorState error={split.error} onRetry={() => split.refetch()} />;
  const sp = split.data;
  const bySession = sp.strategy.group_by !== 'lot';

  const change = async (lotId: string, subset: Subset | null) => {
    try {
      const out = await patch.mutateAsync({ id: sp.id, changes: { [lotId]: subset }, move_group: bySession && moveGroup });
      if (out.new_version) {
        toast.info(`Découpage utilisé : nouvelle version v${out.version} créée`);
        navigate(`/entrainement/decoupages/${out.id}`);
      } else {
        toast.success('Affectation modifiée');
      }
      split.refetch();
    } catch (e) {
      toast.error(e);
    }
  };

  const columns: Column<LotRow>[] = [
    ...lotColumns(),
    {
      key: 'subset',
      header: 'Sous-ensemble',
      render: (l) => (
        <Select<string>
          size="sm"
          value={l.subset ?? ''}
          onChange={(v) => change(l.id, (v || null) as Subset | null)}
          options={[...SUBSETS.map((x) => ({ value: x, label: SUBSET_LABEL[x] })), { value: '', label: 'Retirer' }]}
          aria-label={`Sous-ensemble du lot ${l.name}`}
          disabled={patch.isPending}
        />
      ),
      sortValue: (l) => l.subset ?? 'z',
      width: 170,
    },
  ];

  return (
    <>
      <PageHeader
        breadcrumbs={[{ label: 'Entraînement', to: '/entrainement' }, { label: 'Découpages', to: '/entrainement/decoupages' }, { label: `${sp.name} v${sp.version}` }]}
        title={`${sp.name} — v${sp.version}`}
        subtitle={`${sp.strategy.group_by === 'lot' ? 'Regroupement par lot' : 'Regroupement par session'} · créé le ${formatDateTime(sp.created_at)}${sp.strategy.derived_from ? ` · dérivé du découpage n° ${sp.strategy.derived_from}` : ''}`}
        actions={
          <>
            <Button variant="primary" onClick={() => navigate(`/entrainement/nouveau?split=${sp.id}`)}>
              Entraîner sur ce découpage
            </Button>
            {!sp.locked && (
              <Button variant="danger" icon={Trash2} onClick={() => setConfirmDelete(true)}>
                Supprimer
              </Button>
            )}
          </>
        }
      />
      <div className={s.formGrid}>
        <div className="u-stack-4">
          {sp.locked && (
            <Callout tone="info" icon={Lock} title={`Utilisé par ${sp.used_by.length} modèle(s) : découpage figé`}>
              Toute modification d'affectation crée automatiquement une nouvelle version ; les modèles existants gardent leur découpage.
            </Callout>
          )}
          <Panel
            title="Lots affectés"
            description="Ajustement manuel de l'affectation d'un lot."
            actions={bySession ? <Toggle size="sm" checked={moveGroup} onChange={setMoveGroup} label="Déplacer toute la session" /> : undefined}
            flush
          >
            <DataTable columns={columns} rows={sp.lots} rowKey={(l) => l.id} dense minWidth={760} />
          </Panel>
          {sp.unassigned_lots.length > 0 && (
            <Panel title="Lots non affectés" description="Lots apparus (ou labellisés) depuis la création : ils ne sont pas utilisés tant qu'ils ne sont pas affectés." flush>
              <DataTable
                columns={[
                  ...lotColumns(),
                  {
                    key: 'add',
                    header: '',
                    render: (l) => (
                      <Select<string>
                        size="sm"
                        value=""
                        placeholder="Affecter à…"
                        onChange={(v) => v && change(l.id, v as Subset)}
                        options={SUBSETS.map((x) => ({ value: x, label: SUBSET_LABEL[x] }))}
                        aria-label={`Affecter le lot ${l.name}`}
                      />
                    ),
                    width: 170,
                  },
                ]}
                rows={sp.unassigned_lots}
                rowKey={(l) => l.id}
                dense
                minWidth={760}
              />
            </Panel>
          )}
        </div>
        <div className={s.formSide}>
          <Panel title="Effectifs actuels" description="Avec les labels actuels (ils peuvent avoir changé depuis l'enregistrement).">
            <div className="u-stack-4">
              <SummaryTable summary={sp.current_summary} />
              <SplitWarnings summary={sp.current_summary} holdout={!!(sp.strategy.holdout_farms?.length || sp.strategy.holdout_dates?.length)} />
            </div>
          </Panel>
          {sp.used_by.length > 0 && (
            <Panel title="Modèles utilisant ce découpage">
              <ul className={s.warnList}>
                {sp.used_by.map((m) => (
                  <li key={m}>
                    <a href={`#/entrainement/modeles/${encodeURIComponent(m)}`} className="u-mono u-small">
                      {m}
                    </a>
                  </li>
                ))}
              </ul>
            </Panel>
          )}
        </div>
      </div>
      <ConfirmDialog
        open={confirmDelete}
        title="Supprimer ce découpage ?"
        message={`« ${sp.name} » v${sp.version} sera supprimé définitivement.`}
        confirmLabel="Supprimer"
        variant="danger"
        onCancel={() => setConfirmDelete(false)}
        onConfirm={async () => {
          try {
            await del.mutateAsync(sp.id);
            toast.success('Découpage supprimé');
            navigate('/entrainement/decoupages');
          } catch (e) {
            toast.error(e);
          }
        }}
      />
    </>
  );
}
