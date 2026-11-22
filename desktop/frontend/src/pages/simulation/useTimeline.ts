/**
 * Chargement progressif de la chronologie d'une simulation, par tranches de colonnes.
 * Les tranches sont conservées dans une Map (référence stable) : le dessin les lit sans re-rendu React.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { errorMessage } from '@/api/client';
import { fetchTimeline } from './api';
import type { Geometry, TimelineColumn } from './types';

export const CHUNK = 250;

export interface TimelineState {
  geometry: Geometry | null;
  threshold: number | null;
  total: number;
  error: string | null;
  /** Nombre de tranches chargées (déclenche un re-rendu quand il change). */
  loadedChunks: number;
  get: (k: number) => TimelineColumn | undefined;
  /** Demande le chargement des colonnes [lo, hi] (et de la tranche suivante, par anticipation). */
  ensure: (lo: number, hi: number) => void;
  isLoaded: (k: number) => boolean;
}

export function useTimeline(simId: string): TimelineState {
  const chunks = useRef(new Map<number, TimelineColumn[]>());
  const pending = useRef(new Set<number>());
  const [geometry, setGeometry] = useState<Geometry | null>(null);
  const [threshold, setThreshold] = useState<number | null>(null);
  const [total, setTotal] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [loadedChunks, setLoaded] = useState(0);
  const totalRef = useRef<number | null>(null);
  const alive = useRef(true);

  const load = useCallback(
    (idx: number) => {
      if (chunks.current.has(idx) || pending.current.has(idx)) return;
      if (totalRef.current !== null && idx * CHUNK >= totalRef.current) return;
      pending.current.add(idx);
      fetchTimeline(simId, idx * CHUNK, CHUNK)
        .then((data) => {
          if (!alive.current) return;
          chunks.current.set(idx, data.columns);
          if (totalRef.current === null) {
            totalRef.current = data.total;
            setTotal(data.total);
            setGeometry(data.geometry);
            setThreshold(data.threshold);
          }
          setLoaded(chunks.current.size);
        })
        .catch((e) => alive.current && setError(errorMessage(e)))
        .finally(() => pending.current.delete(idx));
    },
    [simId],
  );

  useEffect(() => {
    alive.current = true;
    chunks.current = new Map();
    pending.current = new Set();
    totalRef.current = null;
    setGeometry(null);
    setTotal(0);
    setError(null);
    setLoaded(0);
    load(0);
    return () => {
      alive.current = false;
    };
  }, [simId, load]);

  const get = useCallback((k: number) => chunks.current.get(Math.floor(k / CHUNK))?.[k % CHUNK], []);
  const isLoaded = useCallback((k: number) => chunks.current.has(Math.floor(k / CHUNK)), []);
  const ensure = useCallback(
    (lo: number, hi: number) => {
      const a = Math.max(0, Math.floor(lo / CHUNK));
      const b = Math.floor(Math.max(lo, hi) / CHUNK) + 1; // + tranche suivante (anticipation)
      for (let i = a; i <= b; i++) load(i);
    },
    [load],
  );

  return { geometry, threshold, total, error, loadedChunks, get, ensure, isLoaded };
}
