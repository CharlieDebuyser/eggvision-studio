import {
  cloneElement,
  isValidElement,
  useCallback,
  useId,
  useLayoutEffect,
  useRef,
  useState,
  type ReactElement,
  type ReactNode,
} from 'react';
import { createPortal } from 'react-dom';

export interface TooltipProps {
  content: ReactNode;
  /** Un seul élément enfant capable de recevoir une ref (bouton, span…). */
  children: ReactElement;
  side?: 'top' | 'bottom' | 'left' | 'right';
  delay?: number;
  disabled?: boolean;
}

type AnyProps = Record<string, unknown>;

/**
 * Infobulle au survol et au focus clavier (rendue dans un portail, jamais rognée).
 * <Tooltip content="Supprimer"><button>…</button></Tooltip>
 */
export function Tooltip({ content, children, side = 'top', delay = 350, disabled }: TooltipProps) {
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState<{ x: number; y: number } | null>(null);
  const anchor = useRef<HTMLElement | null>(null);
  const bubble = useRef<HTMLDivElement | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout>>();
  const id = useId();

  const show = useCallback(() => {
    clearTimeout(timer.current);
    timer.current = setTimeout(() => setOpen(true), delay);
  }, [delay]);
  const hide = useCallback(() => {
    clearTimeout(timer.current);
    setOpen(false);
  }, []);

  useLayoutEffect(() => {
    if (!open || !anchor.current || !bubble.current) return;
    const a = anchor.current.getBoundingClientRect();
    const b = bubble.current.getBoundingClientRect();
    const gap = 6;
    let x = a.left + a.width / 2 - b.width / 2;
    let y = a.top - b.height - gap;
    if (side === 'bottom' || (side === 'top' && y < 4)) y = a.bottom + gap;
    if (side === 'left') {
      x = a.left - b.width - gap;
      y = a.top + a.height / 2 - b.height / 2;
    }
    if (side === 'right') {
      x = a.right + gap;
      y = a.top + a.height / 2 - b.height / 2;
    }
    x = Math.max(4, Math.min(x, window.innerWidth - b.width - 4));
    setPos({ x, y });
  }, [open, side, content]);

  if (!isValidElement(children) || disabled || content == null || content === '') return children;

  const childProps = children.props as AnyProps;
  const chain =
    (ours: () => void, theirs: unknown) =>
    (e: unknown) => {
      ours();
      if (typeof theirs === 'function') (theirs as (e: unknown) => void)(e);
    };

  const trigger = cloneElement(children as ReactElement<AnyProps>, {
    ref: (node: HTMLElement | null) => {
      anchor.current = node;
      const r = (children as unknown as { ref?: unknown }).ref;
      if (typeof r === 'function') r(node);
      else if (r && typeof r === 'object') (r as { current: unknown }).current = node;
    },
    onMouseEnter: chain(show, childProps.onMouseEnter),
    onMouseLeave: chain(hide, childProps.onMouseLeave),
    onFocus: chain(show, childProps.onFocus),
    onBlur: chain(hide, childProps.onBlur),
    onMouseDown: chain(hide, childProps.onMouseDown),
    'aria-describedby': open ? id : (childProps['aria-describedby'] as string | undefined),
  });

  return (
    <>
      {trigger}
      {open &&
        createPortal(
          <div
            ref={bubble}
            id={id}
            role="tooltip"
            className="tooltip"
            style={{ left: pos?.x ?? -9999, top: pos?.y ?? -9999, visibility: pos ? 'visible' : 'hidden' }}
          >
            {content}
          </div>,
          document.body,
        )}
    </>
  );
}
