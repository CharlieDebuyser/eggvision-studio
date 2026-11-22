/**
 * État partagé entre les écrans Relecture et Grille : filtres et file d'apprentissage actif.
 * Conservé pour la session (sessionStorage) afin de retrouver ses filtres en changeant d'onglet.
 */
import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react';
import { EMPTY_FILTERS, type Filters } from './api';

interface LabelingState {
  filters: Filters;
  setFilters: (f: Filters) => void;
  patchFilters: (p: Partial<Filters>) => void;
  activeQueue: boolean;
  setActiveQueue: (v: boolean) => void;
}

const Ctx = createContext<LabelingState | null>(null);
const STORAGE_KEY = 'eggvision.labellisation.state';

function load(): { filters: Filters; activeQueue: boolean } {
  try {
    const raw = sessionStorage.getItem(STORAGE_KEY);
    if (raw) {
      const data = JSON.parse(raw) as { filters?: Partial<Filters>; activeQueue?: boolean };
      return { filters: { ...EMPTY_FILTERS, ...(data.filters ?? {}) }, activeQueue: !!data.activeQueue };
    }
  } catch {
    /* stockage indisponible : valeurs par défaut */
  }
  return { filters: { ...EMPTY_FILTERS, status: 'a_relire' }, activeQueue: false };
}

export function LabelingStateProvider({ children }: { children: ReactNode }) {
  const [initial] = useState(load);
  const [filters, setFilters] = useState<Filters>(initial.filters);
  const [activeQueue, setActiveQueue] = useState(initial.activeQueue);

  useEffect(() => {
    try {
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify({ filters, activeQueue }));
    } catch {
      /* ignoré */
    }
  }, [filters, activeQueue]);

  const patchFilters = useCallback((p: Partial<Filters>) => setFilters((f) => ({ ...f, ...p })), []);
  const value = useMemo(
    () => ({ filters, setFilters, patchFilters, activeQueue, setActiveQueue }),
    [filters, patchFilters, activeQueue],
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useLabelingState(): LabelingState {
  const v = useContext(Ctx);
  if (!v) throw new Error('useLabelingState hors de LabelingStateProvider');
  return v;
}
