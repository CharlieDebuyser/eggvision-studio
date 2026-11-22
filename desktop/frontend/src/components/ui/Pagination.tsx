import { ChevronLeft, ChevronRight } from 'lucide-react';
import { cx } from '@/lib/cx';
import { formatNumber } from '@/lib/format';

export interface PaginationProps {
  /** Page courante, base 0. */
  page: number;
  pageSize: number;
  total: number;
  onPageChange: (page: number) => void;
  className?: string;
}

function pageList(current: number, count: number): (number | '…')[] {
  if (count <= 7) return Array.from({ length: count }, (_, i) => i);
  const set = new Set([0, count - 1, current - 1, current, current + 1].filter((p) => p >= 0 && p < count));
  const sorted = [...set].sort((a, b) => a - b);
  const out: (number | '…')[] = [];
  sorted.forEach((p, i) => {
    if (i > 0 && p - sorted[i - 1] > 1) out.push('…');
    out.push(p);
  });
  return out;
}

/** Pagination (pages en base 0) avec résumé « 1–50 sur 1 234 ». */
export function Pagination({ page, pageSize, total, onPageChange, className }: PaginationProps) {
  const count = Math.max(1, Math.ceil(total / pageSize));
  const from = total === 0 ? 0 : page * pageSize + 1;
  const to = Math.min(total, (page + 1) * pageSize);
  return (
    <nav className={cx('pagination', className)} aria-label="Pagination">
      <span className="pagination__summary u-num">
        {formatNumber(from)}–{formatNumber(to)} sur {formatNumber(total)}
      </span>
      <div className="pagination__pages">
        <button type="button" className="btn btn--ghost btn--icon btn--sm" disabled={page <= 0} onClick={() => onPageChange(page - 1)} aria-label="Page précédente">
          <ChevronLeft size={14} aria-hidden />
        </button>
        {pageList(page, count).map((p, i) =>
          p === '…' ? (
            <span key={`e${i}`} className="pagination__ellipsis">
              …
            </span>
          ) : (
            <button
              key={p}
              type="button"
              className={cx('pagination__page', p === page && 'is-active')}
              aria-current={p === page ? 'page' : undefined}
              onClick={() => onPageChange(p)}
            >
              {p + 1}
            </button>
          ),
        )}
        <button type="button" className="btn btn--ghost btn--icon btn--sm" disabled={page >= count - 1} onClick={() => onPageChange(page + 1)} aria-label="Page suivante">
          <ChevronRight size={14} aria-hidden />
        </button>
      </div>
    </nav>
  );
}
