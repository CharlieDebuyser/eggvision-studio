import { useEffect, useId, useRef, useState, type ReactNode } from 'react';
import { createPortal } from 'react-dom';
import { X } from 'lucide-react';
import { cx } from '@/lib/cx';
import { Button, type ButtonVariant } from './Button';

export interface ModalProps {
  open: boolean;
  onClose: () => void;
  title: ReactNode;
  description?: ReactNode;
  children?: ReactNode;
  /** Boutons du pied (alignés à droite). */
  footer?: ReactNode;
  size?: 'sm' | 'md' | 'lg' | 'xl';
  /** Fermer en cliquant sur le fond (défaut true). */
  dismissible?: boolean;
  /** Corps sans padding. */
  flush?: boolean;
  /** Sélecteur CSS de l'élément à focaliser à l'ouverture. */
  initialFocus?: string;
  className?: string;
}

const FOCUSABLE = 'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

/**
 * Fenêtre modale (portail, focus piégé, Échap pour fermer, focus restauré à la fermeture).
 */
export function Modal({ open, onClose, title, description, children, footer, size = 'md', dismissible = true, flush, initialFocus, className }: ModalProps) {
  const ref = useRef<HTMLDivElement>(null);
  const titleId = useId();
  const descId = useId();
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;

  useEffect(() => {
    if (!open) return;
    const previous = document.activeElement as HTMLElement | null;
    const node = ref.current;
    const target =
      (initialFocus && node?.querySelector<HTMLElement>(initialFocus)) ||
      node?.querySelector<HTMLElement>('[data-autofocus]') ||
      node?.querySelector<HTMLElement>('.modal__body ' + FOCUSABLE) ||
      node;
    target?.focus();

    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.stopPropagation();
        e.preventDefault();
        onCloseRef.current();
        return;
      }
      if (e.key === 'Tab' && node) {
        const items = Array.from(node.querySelectorAll<HTMLElement>(FOCUSABLE)).filter((el) => el.offsetParent !== null || el === document.activeElement);
        if (items.length === 0) return;
        const first = items[0];
        const last = items[items.length - 1];
        if (e.shiftKey && document.activeElement === first) {
          e.preventDefault();
          last.focus();
        } else if (!e.shiftKey && document.activeElement === last) {
          e.preventDefault();
          first.focus();
        }
      }
    };
    document.addEventListener('keydown', onKey, true);
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => {
      document.removeEventListener('keydown', onKey, true);
      document.body.style.overflow = prevOverflow;
      previous?.focus?.();
    };
  }, [open, initialFocus]);

  if (!open) return null;

  return createPortal(
    <div
      className="modal-overlay"
      onMouseDown={(e) => {
        if (dismissible && e.target === e.currentTarget) onClose();
      }}
    >
      <div
        ref={ref}
        className={cx('modal', `modal--${size}`, className)}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-describedby={description ? descId : undefined}
        tabIndex={-1}
      >
        <header className="modal__header">
          <div className="u-grow">
            <h2 id={titleId} className="modal__title">
              {title}
            </h2>
            {description && (
              <p id={descId} className="modal__desc">
                {description}
              </p>
            )}
          </div>
          <button type="button" className="btn btn--ghost btn--icon btn--sm" onClick={onClose} aria-label="Fermer">
            <X size={15} aria-hidden />
          </button>
        </header>
        <div className={cx('modal__body', flush && 'modal__body--flush')}>{children}</div>
        {footer && <footer className="modal__footer">{footer}</footer>}
      </div>
    </div>,
    document.body,
  );
}

export interface ConfirmDialogProps {
  open: boolean;
  title: ReactNode;
  /** Texte explicatif (conséquences). */
  message?: ReactNode;
  confirmLabel?: string;
  cancelLabel?: string;
  /** 'danger' pour une action destructive (défaut 'primary'). */
  variant?: Extract<ButtonVariant, 'primary' | 'danger' | 'accent'>;
  /** Peut renvoyer une promesse : le bouton affiche alors un spinner jusqu'à résolution. */
  onConfirm: () => void | Promise<unknown>;
  onCancel: () => void;
}

/**
 * Demande de confirmation. La fermeture est à la charge de l'appelant (passer open=false
 * dans onConfirm). Si onConfirm renvoie une promesse, le bouton reste en chargement jusqu'à sa fin.
 */
export function ConfirmDialog({
  open,
  title,
  message,
  confirmLabel = 'Confirmer',
  cancelLabel = 'Annuler',
  variant = 'primary',
  onConfirm,
  onCancel,
}: ConfirmDialogProps) {
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    if (!open) setBusy(false);
  }, [open]);

  const confirm = async () => {
    const r = onConfirm();
    if (r && typeof (r as Promise<unknown>).then === 'function') {
      setBusy(true);
      try {
        await r;
      } finally {
        setBusy(false);
      }
    }
  };

  return (
    <Modal
      open={open}
      onClose={busy ? () => undefined : onCancel}
      title={title}
      size="sm"
      footer={
        <>
          <Button variant="ghost" onClick={onCancel} disabled={busy}>
            {cancelLabel}
          </Button>
          <Button variant={variant} onClick={confirm} loading={busy} data-autofocus>
            {confirmLabel}
          </Button>
        </>
      }
    >
      {message && <div className="u-muted">{message}</div>}
    </Modal>
  );
}
