import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { createPortal } from 'react-dom';
import { AlertTriangle, CheckCircle2, Info, X, XCircle } from 'lucide-react';
import { cx } from '@/lib/cx';
import { errorMessage } from '@/api/client';

export type ToastTone = 'info' | 'success' | 'warning' | 'danger';

export interface ToastOptions {
  title: ReactNode;
  description?: ReactNode;
  tone?: ToastTone;
  /** ms ; 0 = reste affiché jusqu'à fermeture. Défaut 5000 (8000 pour danger). */
  duration?: number;
  /** Bouton d'action optionnel. */
  action?: { label: string; onClick: () => void };
}

interface ToastItem extends ToastOptions {
  id: number;
}

export interface ToastApi {
  /** Affiche un toast et renvoie son id. */
  show: (opts: ToastOptions) => number;
  success: (title: ReactNode, description?: ReactNode) => number;
  info: (title: ReactNode, description?: ReactNode) => number;
  warning: (title: ReactNode, description?: ReactNode) => number;
  /** Accepte un message ou directement une erreur (ApiError → detail). */
  error: (titleOrError: ReactNode | unknown, description?: ReactNode) => number;
  dismiss: (id: number) => void;
}

const ToastContext = createContext<ToastApi | null>(null);

const ICONS = { info: Info, success: CheckCircle2, warning: AlertTriangle, danger: XCircle };

function ToastView({ t, onClose }: { t: ToastItem; onClose: () => void }) {
  const tone = t.tone ?? 'info';
  const Icon = ICONS[tone];
  const duration = t.duration ?? (tone === 'danger' ? 8000 : 5000);
  const [paused, setPaused] = useState(false);
  const remaining = useRef(duration);
  const startedAt = useRef(Date.now());

  useEffect(() => {
    if (duration === 0 || paused) return;
    startedAt.current = Date.now();
    const timer = setTimeout(onClose, remaining.current);
    return () => {
      clearTimeout(timer);
      remaining.current -= Date.now() - startedAt.current;
    };
  }, [paused, duration, onClose]);

  return (
    <div
      className={cx('toast', `toast--${tone}`)}
      role={tone === 'danger' ? 'alert' : 'status'}
      onMouseEnter={() => setPaused(true)}
      onMouseLeave={() => setPaused(false)}
    >
      <Icon size={17} className="toast__icon" aria-hidden />
      <div className="toast__body">
        <div className="toast__title">{t.title}</div>
        {t.description && <div className="toast__desc">{t.description}</div>}
        {t.action && (
          <button
            type="button"
            className="toast__action"
            onClick={() => {
              t.action!.onClick();
              onClose();
            }}
          >
            {t.action.label}
          </button>
        )}
      </div>
      <button type="button" className="toast__close" onClick={onClose} aria-label="Fermer la notification">
        <X size={14} aria-hidden />
      </button>
    </div>
  );
}

export function ToastProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<ToastItem[]>([]);
  const nextId = useRef(1);

  const dismiss = useCallback((id: number) => setItems((l) => l.filter((t) => t.id !== id)), []);
  const show = useCallback((opts: ToastOptions) => {
    const id = nextId.current++;
    setItems((l) => [...l.slice(-4), { ...opts, id }]);
    return id;
  }, []);

  const apiValue = useMemo<ToastApi>(
    () => ({
      show,
      dismiss,
      success: (title, description) => show({ title, description, tone: 'success' }),
      info: (title, description) => show({ title, description, tone: 'info' }),
      warning: (title, description) => show({ title, description, tone: 'warning' }),
      error: (titleOrError, description) => {
        const isErr = titleOrError instanceof Error || (typeof titleOrError === 'object' && titleOrError !== null && 'detail' in titleOrError);
        return show({
          tone: 'danger',
          title: isErr ? 'Erreur' : (titleOrError as ReactNode),
          description: isErr ? errorMessage(titleOrError) : description,
        });
      },
    }),
    [show, dismiss],
  );

  return (
    <ToastContext.Provider value={apiValue}>
      {children}
      {createPortal(
        <div className="toast-region" aria-live="polite">
          {items.map((t) => (
            <ToastView key={t.id} t={t} onClose={() => dismiss(t.id)} />
          ))}
        </div>,
        document.body,
      )}
    </ToastContext.Provider>
  );
}

/**
 * const toast = useToast();
 * toast.success('Projet créé');
 * toast.error(err);                       // ApiError → message du backend
 * toast.show({ title: 'Export prêt', action: { label: 'Ouvrir', onClick } });
 */
export function useToast(): ToastApi {
  const ctx = useContext(ToastContext);
  if (!ctx) throw new Error('useToast doit être utilisé sous <ToastProvider>.');
  return ctx;
}
