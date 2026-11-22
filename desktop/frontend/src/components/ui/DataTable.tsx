import { useMemo, useState, type ReactNode } from 'react';
import { ArrowDown, ArrowUp, ArrowUpDown } from 'lucide-react';
import { cx } from '@/lib/cx';
import { LoadingState, EmptyState, ErrorState } from './Feedback';

export interface Column<T> {
  /** Identifiant unique de la colonne. */
  key: string;
  header: ReactNode;
  /** Rendu de la cellule (défaut : row[key]). */
  render?: (row: T, index: number) => ReactNode;
  /** Valeur de tri ; si fournie, la colonne est triable. */
  sortValue?: (row: T) => string | number | null | undefined;
  align?: 'left' | 'right' | 'center';
  /** Largeur CSS (« 120px », « 20% »). */
  width?: string | number;
  /** Texte tronqué avec ellipse. */
  truncate?: boolean;
  className?: string;
}

export type SortState = { key: string; dir: 'asc' | 'desc' } | null;

export interface DataTableProps<T> {
  columns: Column<T>[];
  rows: T[] | undefined;
  /** Clé React de chaque ligne. */
  rowKey: (row: T, index: number) => string | number;
  onRowClick?: (row: T) => void;
  /** Ligne mise en évidence. */
  selectedKey?: string | number | null;
  loading?: boolean;
  error?: unknown;
  onRetry?: () => void;
  /** Contenu affiché si rows est vide. */
  empty?: ReactNode;
  /** Tri initial (non contrôlé). */
  defaultSort?: SortState;
  /** En-tête collant (dans un conteneur scrollable). */
  stickyHeader?: boolean;
  dense?: boolean;
  /** Largeur minimale du tableau (px) avant défilement horizontal ; défaut = somme des largeurs fixes + 200 px par colonne libre. */
  minWidth?: number;
  className?: string;
}

const collator = new Intl.Collator('fr', { numeric: true, sensitivity: 'base' });

/**
 * Tableau simple : colonnes typées, tri client optionnel, ligne cliquable (Entrée/Espace au clavier).
 * Pour > ~500 lignes, utiliser @tanstack/react-virtual dans le module.
 */
export function DataTable<T>({
  columns,
  rows,
  rowKey,
  onRowClick,
  selectedKey,
  loading,
  error,
  onRetry,
  empty,
  defaultSort = null,
  stickyHeader,
  dense,
  minWidth,
  className,
}: DataTableProps<T>) {
  const computedMin =
    minWidth ?? columns.reduce((sum, c) => sum + (typeof c.width === 'number' ? c.width : c.width === undefined ? 200 : 0), 0);
  const [sort, setSort] = useState<SortState>(defaultSort);

  const sorted = useMemo(() => {
    if (!rows || !sort) return rows ?? [];
    const col = columns.find((c) => c.key === sort.key);
    if (!col?.sortValue) return rows;
    const get = col.sortValue;
    const mult = sort.dir === 'asc' ? 1 : -1;
    return [...rows].sort((a, b) => {
      const va = get(a);
      const vb = get(b);
      if (va == null && vb == null) return 0;
      if (va == null) return 1;
      if (vb == null) return -1;
      if (typeof va === 'number' && typeof vb === 'number') return (va - vb) * mult;
      return collator.compare(String(va), String(vb)) * mult;
    });
  }, [rows, sort, columns]);

  const toggleSort = (key: string) =>
    setSort((s) => (s?.key !== key ? { key, dir: 'asc' } : s.dir === 'asc' ? { key, dir: 'desc' } : null));

  const body = () => {
    if (error && !rows) return <ErrorState error={error} onRetry={onRetry} compact />;
    if (loading && !rows) return <LoadingState compact />;
    if (sorted.length === 0) return empty ?? <EmptyState title="Aucun élément" compact />;
    return null;
  };
  const placeholder = body();

  return (
    <div className={cx('table-wrap', className)}>
      <table
        className={cx('table', dense && 'table--dense', stickyHeader && 'table--sticky', onRowClick && 'table--clickable')}
        style={{ minWidth: computedMin }}
      >
        <colgroup>
          {columns.map((c) => (
            <col key={c.key} style={c.width !== undefined ? { width: c.width } : undefined} />
          ))}
        </colgroup>
        <thead>
          <tr>
            {columns.map((c) => {
              const active = sort?.key === c.key;
              const Icon = !active ? ArrowUpDown : sort.dir === 'asc' ? ArrowUp : ArrowDown;
              return (
                <th
                  key={c.key}
                  scope="col"
                  className={cx(c.align && `is-${c.align}`, c.className)}
                  aria-sort={active ? (sort.dir === 'asc' ? 'ascending' : 'descending') : undefined}
                >
                  {c.sortValue ? (
                    <button type="button" className={cx('table__sort', active && 'is-active')} onClick={() => toggleSort(c.key)}>
                      {c.header}
                      <Icon size={12} aria-hidden />
                    </button>
                  ) : (
                    c.header
                  )}
                </th>
              );
            })}
          </tr>
        </thead>
        {!placeholder && (
          <tbody>
            {sorted.map((row, i) => {
              const key = rowKey(row, i);
              return (
                <tr
                  key={key}
                  className={cx(selectedKey !== undefined && selectedKey === key && 'is-selected')}
                  onClick={onRowClick ? () => onRowClick(row) : undefined}
                  tabIndex={onRowClick ? 0 : undefined}
                  onKeyDown={
                    onRowClick
                      ? (e) => {
                          if (e.target !== e.currentTarget) return;
                          if (e.key === 'Enter' || e.key === ' ') {
                            e.preventDefault();
                            onRowClick(row);
                          }
                        }
                      : undefined
                  }
                >
                  {columns.map((c) => (
                    <td key={c.key} className={cx(c.align && `is-${c.align}`, c.truncate && 'is-truncate', c.className)}>
                      {c.render ? c.render(row, i) : String((row as Record<string, unknown>)[c.key] ?? '')}
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        )}
      </table>
      {placeholder && <div className="table__placeholder">{placeholder}</div>}
    </div>
  );
}
