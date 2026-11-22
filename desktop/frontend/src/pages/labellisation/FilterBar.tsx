import { useEffect, useMemo, useRef, useState } from 'react';
import { Check, ChevronDown, Search, X } from 'lucide-react';
import { Button, NumberInput, SegmentedControl, Select, TextInput, Toggle, Tooltip } from '@/components/ui';
import { cx } from '@/lib/cx';
import { DEFECTS, VERDICTS, lotTypeLabel } from '@/lib/domain';
import { formatNumber } from '@/lib/format';
import { activeFilterCount, EMPTY_FILTERS, useLots, useModels, type Filters, type ReviewStatus } from './api';
import { useLabelingState } from './state';
import s from './labellisation.module.css';

interface PickOption {
  value: string;
  label: string;
  hint?: string;
}

/** Sélection multiple compacte (bouton + menu à cases à cocher, recherche facultative). */
function MultiPick({
  label,
  options,
  value,
  onChange,
  searchable,
  allLabel = 'Tous',
}: {
  label: string;
  options: PickOption[];
  value: string[];
  onChange: (v: string[]) => void;
  searchable?: boolean;
  allLabel?: string;
}) {
  const [open, setOpen] = useState(false);
  const [q, setQ] = useState('');
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.stopPropagation();
        setOpen(false);
      }
    };
    document.addEventListener('mousedown', onDown);
    document.addEventListener('keydown', onKey, true);
    return () => {
      document.removeEventListener('mousedown', onDown);
      document.removeEventListener('keydown', onKey, true);
    };
  }, [open]);

  const shown = useMemo(() => {
    const t = q.trim().toLowerCase();
    return t ? options.filter((o) => `${o.label} ${o.hint ?? ''}`.toLowerCase().includes(t)) : options;
  }, [options, q]);

  const summary =
    value.length === 0
      ? allLabel
      : value.length === 1
        ? (options.find((o) => o.value === value[0])?.label ?? value[0])
        : `${value.length} sélectionnés`;

  const toggle = (v: string) => onChange(value.includes(v) ? value.filter((x) => x !== v) : [...value, v]);

  return (
    <div className={s.pick} ref={ref}>
      <button
        type="button"
        className={cx(s.pickButton, value.length > 0 && s.pickActive)}
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        aria-haspopup="listbox"
      >
        <span className={s.pickLabel}>{label}</span>
        <span className={s.pickValue}>{summary}</span>
        <ChevronDown size={13} aria-hidden />
      </button>
      {open && (
        <div className={s.pickMenu} role="listbox" aria-multiselectable>
          {searchable && (
            <div className={s.pickSearch}>
              <TextInput size="sm" autoFocus placeholder="Rechercher…" value={q} onChange={(e) => setQ(e.target.value)} leading={<Search size={13} />} />
            </div>
          )}
          <div className={s.pickList}>
            {shown.map((o) => {
              const on = value.includes(o.value);
              return (
                <button key={o.value} type="button" role="option" aria-selected={on} className={s.pickItem} onClick={() => toggle(o.value)}>
                  <span className={cx(s.pickCheck, on && s.pickChecked)}>{on && <Check size={11} strokeWidth={3} />}</span>
                  <span className={s.pickItemText}>{o.label}</span>
                  {o.hint && <span className="u-xs u-subtle">{o.hint}</span>}
                </button>
              );
            })}
            {shown.length === 0 && <div className="u-small u-muted" style={{ padding: 8 }}>Aucun résultat.</div>}
          </div>
          <div className={s.pickFoot}>
            <Button size="sm" variant="ghost" onClick={() => onChange([])} disabled={value.length === 0}>
              Effacer
            </Button>
            <Button size="sm" variant="secondary" onClick={() => setOpen(false)}>
              Fermer
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}

const VERDICT_OPTIONS: PickOption[] = [
  ...VERDICTS.map((v) => ({ value: v.key, label: v.label })),
  { value: 'aucun', label: 'Sans label' },
];
const DEFECT_OPTIONS: PickOption[] = [
  ...DEFECTS.map((d) => ({ value: d.key, label: d.label, hint: d.hotkey })),
  { value: 'aucun', label: 'Non viable sans type' },
];
const STATUS_OPTIONS: { value: ReviewStatus | 'tous'; label: string }[] = [
  { value: 'tous', label: 'Toutes' },
  { value: 'a_relire', label: 'À relire' },
  { value: 'relues', label: 'Relues' },
];

/** Panneau de filtres partagé par la relecture et la grille. */
export function FilterBar({ showQueueToggle = true }: { showQueueToggle?: boolean }) {
  const { filters, setFilters, patchFilters, activeQueue, setActiveQueue } = useLabelingState();
  const lots = useLots();
  const models = useModels();

  const lotOptions = useMemo<PickOption[]>(
    () =>
      (lots.data?.items ?? [])
        .filter((l) => l.track_count > 0)
        .map((l) => ({ value: l.id, label: l.name || l.id, hint: `${lotTypeLabel(l.type)} · ${formatNumber(l.track_count)}` })),
    [lots.data],
  );
  const modelOptions = useMemo(() => {
    const def = models.data?.default;
    return [
      { value: '', label: def ? `Par défaut (${def.name})` : 'Aucun modèle' },
      ...(models.data?.items ?? []).map((m) => ({ value: m.id, label: m.name })),
    ];
  }, [models.data]);
  const hasModel = (models.data?.items.length ?? 0) > 0;
  const n = activeFilterCount(filters);

  const set = (p: Partial<Filters>) => patchFilters(p);

  return (
    <div className={s.filters} role="search" aria-label="Filtres des pistes">
      <MultiPick label="Lot" options={lotOptions} value={filters.lots} onChange={(v) => set({ lots: v })} searchable />
      <MultiPick label="Label" options={VERDICT_OPTIONS} value={filters.verdicts} onChange={(v) => set({ verdicts: v })} />
      <MultiPick label="Défaut" options={DEFECT_OPTIONS} value={filters.defects} onChange={(v) => set({ defects: v })} />
      <SegmentedControl
        size="sm"
        aria-label="Statut de relecture"
        value={filters.status || 'tous'}
        onChange={(v) => set({ status: v === 'tous' ? '' : v })}
        options={STATUS_OPTIONS}
      />
      <Select
        size="sm"
        aria-label="Qualité de piste"
        value={filters.quality}
        onChange={(v) => set({ quality: v })}
        options={[
          { value: '', label: 'Toute qualité' },
          { value: 'ok', label: 'Qualité ok' },
          { value: 'douteuse', label: 'Pistes douteuses' },
        ]}
      />
      {hasModel && (
        <>
          <Select size="sm" aria-label="Modèle" value={filters.modelId} onChange={(v) => set({ modelId: v })} options={modelOptions} />
          <div className={s.scoreRange}>
            <Tooltip content="Plage de score de défaut du modèle (0 = viable, 1 = défaut)">
              <span className="u-small u-muted">Score</span>
            </Tooltip>
            <NumberInput
              size="sm"
              className={s.scoreInput}
              aria-label="Score minimal"
              value={filters.scoreMin}
              onChange={(v) => set({ scoreMin: v })}
              min={0}
              max={1}
              step={0.05}
              decimals={2}
              allowEmpty
              stepper={false}
              placeholder="0"
            />
            <span className="u-subtle">–</span>
            <NumberInput
              size="sm"
              className={s.scoreInput}
              aria-label="Score maximal"
              value={filters.scoreMax}
              onChange={(v) => set({ scoreMax: v })}
              min={0}
              max={1}
              step={0.05}
              decimals={2}
              allowEmpty
              stepper={false}
              placeholder="1"
            />
          </div>
        </>
      )}
      {n > 0 && (
        <Button size="sm" variant="ghost" icon={X} onClick={() => setFilters({ ...EMPTY_FILTERS, modelId: filters.modelId })}>
          Réinitialiser ({n})
        </Button>
      )}
      <span className={s.filtersSpacer} />
      {showQueueToggle && (
        <Tooltip content={hasModel ? 'Pistes où le modèle contredit le label ou hésite, non relues d’abord' : 'Disponible après l’entraînement d’un premier modèle'}>
          <span>
            <Toggle size="sm" checked={activeQueue} onChange={setActiveQueue} label="File d’apprentissage actif" disabled={!hasModel && !activeQueue} />
          </span>
        </Tooltip>
      )}
    </div>
  );
}
