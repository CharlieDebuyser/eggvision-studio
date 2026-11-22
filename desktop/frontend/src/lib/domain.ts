/**
 * Constantes métier partagées (défauts, verdicts, types de lot, types de tâche).
 * Les couleurs renvoient à des variables CSS définies dans styles/tokens.css,
 * donc elles suivent automatiquement le thème clair/sombre.
 */

// ---------------------------------------------------------------- Défauts
export const DEFECTS = [
  { key: 'souillure', label: 'Souillure', hotkey: '1' },
  { key: 'coulure', label: 'Coulure', hotkey: '2' },
  { key: 'felure', label: 'Fêlure', hotkey: '3' },
  { key: 'plume', label: 'Plume', hotkey: '4' },
  { key: 'sang', label: 'Sang', hotkey: '5' },
  { key: 'deformation', label: 'Déformation', hotkey: '6' },
] as const;

export type DefectKey = (typeof DEFECTS)[number]['key'];
export type DefectDef = (typeof DEFECTS)[number];

export const DEFECT_KEYS: readonly DefectKey[] = DEFECTS.map((d) => d.key);

export const DEFECT_BY_KEY: Record<DefectKey, DefectDef> = Object.fromEntries(DEFECTS.map((d) => [d.key, d])) as Record<
  DefectKey,
  DefectDef
>;

export const isDefectKey = (k: string): k is DefectKey => k in DEFECT_BY_KEY;

export const defectLabel = (k: string): string => (isDefectKey(k) ? DEFECT_BY_KEY[k].label : k);

/** Défaut associé à une touche ('1'..'6'), sinon undefined. */
export const defectByHotkey = (key: string): DefectDef | undefined => DEFECTS.find((d) => d.hotkey === key);

// ---------------------------------------------------------------- Verdicts
export type Tone = 'neutral' | 'accent' | 'success' | 'warning' | 'danger' | 'info';

export const VERDICTS = [
  { key: 'viable', label: 'Viable', tone: 'success', color: 'var(--verdict-viable)' },
  { key: 'non_viable', label: 'Non viable', tone: 'danger', color: 'var(--verdict-non-viable)' },
  { key: 'incertain', label: 'Incertain', tone: 'warning', color: 'var(--verdict-incertain)' },
  { key: 'exclu', label: 'Exclu', tone: 'neutral', color: 'var(--verdict-exclu)' },
] as const satisfies readonly { key: string; label: string; tone: Tone; color: string }[];

export type VerdictKey = (typeof VERDICTS)[number]['key'];
export type VerdictDef = (typeof VERDICTS)[number];

export const VERDICT_BY_KEY: Record<VerdictKey, VerdictDef> = Object.fromEntries(VERDICTS.map((v) => [v.key, v])) as Record<
  VerdictKey,
  VerdictDef
>;

export const isVerdictKey = (k: string): k is VerdictKey => k in VERDICT_BY_KEY;
export const verdictLabel = (k: string): string => (isVerdictKey(k) ? VERDICT_BY_KEY[k].label : k);

// ---------------------------------------------------------------- Types de lot
/** `viable` | `declasse` | `declasse:<défaut>` | `production` */
export type LotType = 'viable' | 'declasse' | `declasse:${DefectKey}` | 'production';

export const LOT_TYPE_BASES = [
  { key: 'viable', label: 'Viable', description: 'Œufs conformes (référence positive).' },
  { key: 'declasse', label: 'Déclassé', description: 'Œufs déclassés, défaut non précisé.' },
  { key: 'production', label: 'Production', description: 'Flux réel non trié (à labelliser).' },
] as const;

/** Toutes les valeurs possibles, dans l'ordre d'affichage conseillé. */
export const LOT_TYPES: { value: LotType; label: string }[] = [
  { value: 'viable', label: 'Viable' },
  { value: 'declasse', label: 'Déclassé (tous défauts)' },
  ...DEFECTS.map((d) => ({ value: `declasse:${d.key}` as LotType, label: `Déclassé : ${d.label.toLowerCase()}` })),
  { value: 'production', label: 'Production' },
];

export function parseLotType(v: string): { base: 'viable' | 'declasse' | 'production' | 'inconnu'; defect?: DefectKey } {
  if (v === 'viable' || v === 'production' || v === 'declasse') return { base: v };
  if (v.startsWith('declasse:')) {
    const d = v.slice('declasse:'.length);
    return isDefectKey(d) ? { base: 'declasse', defect: d } : { base: 'declasse' };
  }
  return { base: 'inconnu' };
}

export function lotTypeLabel(v: string): string {
  return LOT_TYPES.find((t) => t.value === v)?.label ?? v;
}

// ---------------------------------------------------------------- Tâches
export const JOB_KIND_LABELS: Record<string, string> = {
  demo: 'Projet de démonstration',
  sync: 'Synchronisation',
  capture: 'Prise de vue',
  detection: 'Détection',
  detector_training: 'Entraînement détecteur',
  training: 'Entraînement',
  evaluation: 'Évaluation',
  export: 'Export',
  bench: 'Banc de mesure',
  simulation: 'Simulation',
  video: 'Vidéo',
};

export const jobKindLabel = (k: string) => JOB_KIND_LABELS[k] ?? k;

export const JOB_STATUS_META = {
  queued: { label: 'En attente', tone: 'neutral' },
  running: { label: 'En cours', tone: 'info' },
  done: { label: 'Terminée', tone: 'success' },
  failed: { label: 'Échec', tone: 'danger' },
  cancelled: { label: 'Annulée', tone: 'warning' },
} as const satisfies Record<string, { label: string; tone: Tone }>;
