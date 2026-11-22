/**
 * Formatage français. Toutes les fonctions acceptent null/undefined et renvoient alors '—'.
 */

const DASH = '—';
const nf0 = new Intl.NumberFormat('fr-FR', { maximumFractionDigits: 0 });
const nfCache = new Map<string, Intl.NumberFormat>();

function nf(min: number, max: number): Intl.NumberFormat {
  const key = `${min}-${max}`;
  let f = nfCache.get(key);
  if (!f) {
    f = new Intl.NumberFormat('fr-FR', { minimumFractionDigits: min, maximumFractionDigits: max });
    nfCache.set(key, f);
  }
  return f;
}

const isNum = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);

/** 12345.6 → « 12 346 » ; formatNumber(0.1234, 3) → « 0,123 ». */
export function formatNumber(v: number | null | undefined, decimals = 0, minDecimals = 0): string {
  if (!isNum(v)) return DASH;
  return decimals === 0 ? nf0.format(v) : nf(Math.min(minDecimals, decimals), decimals).format(v);
}

/** Ratio 0..1 → « 87,5 % ». */
export function formatPercent(ratio: number | null | undefined, decimals = 1): string {
  if (!isNum(ratio)) return DASH;
  return `${nf(0, decimals).format(ratio * 100)} %`;
}

/** Secondes → « 450 ms », « 12 s », « 3 min 05 s », « 2 h 04 min ». */
export function formatDuration(seconds: number | null | undefined): string {
  if (!isNum(seconds) || seconds < 0) return DASH;
  if (seconds < 1) return `${Math.round(seconds * 1000)} ms`;
  if (seconds < 60) return `${nf(0, seconds < 10 ? 1 : 0).format(seconds)} s`;
  const s = Math.round(seconds);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  if (h > 0) return `${h} h ${String(m).padStart(2, '0')} min`;
  return `${m} min ${String(sec).padStart(2, '0')} s`;
}

/** Durée entre deux dates ISO (fin = maintenant si absente). */
export function durationBetween(start: string | null | undefined, end?: string | null): number | null {
  if (!start) return null;
  const a = Date.parse(start);
  const b = end ? Date.parse(end) : Date.now();
  if (Number.isNaN(a) || Number.isNaN(b)) return null;
  return Math.max(0, (b - a) / 1000);
}

/** Octets → « 1,2 Mo ». */
export function formatBytes(bytes: number | null | undefined, decimals = 1): string {
  if (!isNum(bytes)) return DASH;
  const units = ['o', 'Ko', 'Mo', 'Go', 'To'];
  let v = Math.abs(bytes);
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${bytes < 0 ? '-' : ''}${nf(0, i === 0 ? 0 : decimals).format(v)} ${units[i]}`;
}

function toDate(v: string | number | Date | null | undefined): Date | null {
  if (v === null || v === undefined || v === '') return null;
  const d = v instanceof Date ? v : new Date(v);
  return Number.isNaN(d.getTime()) ? null : d;
}

const dfDate = new Intl.DateTimeFormat('fr-FR', { day: '2-digit', month: '2-digit', year: 'numeric' });
const dfDateTime = new Intl.DateTimeFormat('fr-FR', {
  day: '2-digit',
  month: '2-digit',
  year: 'numeric',
  hour: '2-digit',
  minute: '2-digit',
});
const dfTime = new Intl.DateTimeFormat('fr-FR', { hour: '2-digit', minute: '2-digit', second: '2-digit' });

/** « 20/11/2026 » */
export const formatDate = (v: string | number | Date | null | undefined) => {
  const d = toDate(v);
  return d ? dfDate.format(d) : DASH;
};
/** « 20/11/2026 17:03 » */
export const formatDateTime = (v: string | number | Date | null | undefined) => {
  const d = toDate(v);
  return d ? dfDateTime.format(d) : DASH;
};
/** « 17:03:12 » */
export const formatTime = (v: string | number | Date | null | undefined) => {
  const d = toDate(v);
  return d ? dfTime.format(d) : DASH;
};

const rtf = new Intl.RelativeTimeFormat('fr-FR', { numeric: 'auto' });

/** « il y a 5 minutes », « hier »… */
export function formatRelative(v: string | number | Date | null | undefined, now: number = Date.now()): string {
  const d = toDate(v);
  if (!d) return DASH;
  const diff = (d.getTime() - now) / 1000;
  const abs = Math.abs(diff);
  if (abs < 45) return 'à l’instant';
  if (abs < 3600) return rtf.format(Math.round(diff / 60), 'minute');
  if (abs < 86400) return rtf.format(Math.round(diff / 3600), 'hour');
  if (abs < 86400 * 30) return rtf.format(Math.round(diff / 86400), 'day');
  return formatDate(d);
}

/** Accord simple : plural(3, 'image') → « 3 images ». */
export function plural(n: number, singular: string, pluralForm = `${singular}s`): string {
  return `${formatNumber(n)} ${Math.abs(n) >= 2 ? pluralForm : singular}`;
}
