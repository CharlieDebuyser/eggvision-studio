import { useEffect, useLayoutEffect, useRef, useState, type CSSProperties } from 'react';
import { cx } from '@/lib/cx';

export interface LogViewProps {
  text: string;
  /** Défile automatiquement en bas quand du texte arrive (si l'utilisateur est déjà en bas). */
  follow?: boolean;
  emptyText?: string;
  height?: CSSProperties['height'];
  className?: string;
}

/** Affichage monospace d'un journal, avec suivi automatique de la fin. */
export function LogView({ text, follow = true, emptyText = 'Journal vide.', height = 360, className }: LogViewProps) {
  const ref = useRef<HTMLPreElement>(null);
  const [stick, setStick] = useState(true);

  useLayoutEffect(() => {
    const el = ref.current;
    if (el && follow && stick) el.scrollTop = el.scrollHeight;
  }, [text, follow, stick]);

  useEffect(() => {
    setStick(true);
  }, [follow]);

  return (
    <pre
      ref={ref}
      className={cx('log-view', !text && 'log-view--empty', className)}
      style={{ height }}
      tabIndex={0}
      aria-label="Journal"
      onScroll={(e) => {
        const el = e.currentTarget;
        setStick(el.scrollHeight - el.scrollTop - el.clientHeight < 24);
      }}
    >
      {text || emptyText}
    </pre>
  );
}
