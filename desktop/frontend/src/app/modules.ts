import { lazy, type ComponentType, type LazyExoticComponent } from 'react';
import { BrainCircuit, Camera, Cpu, ScanSearch, Split, Tags, type LucideIcon } from 'lucide-react';

export interface ModuleDef {
  /** Segment d'URL (sans slash). */
  path: string;
  label: string;
  /** Rôle du module dans le parcours (affiché sur l'accueil projet). */
  description: string;
  icon: LucideIcon;
  /** Étape du parcours (1..6). */
  step: number;
  component: LazyExoticComponent<ComponentType>;
}

/**
 * Modules métier, dans l'ordre de la barre latérale et du parcours.
 * Chaque module est monté sur `/<path>/*` derrière la garde « projet ouvert ».
 */
export const MODULES: ModuleDef[] = [
  {
    path: 'capture',
    label: 'Prise de vue',
    description: 'Acquérir des lots d’images depuis le banc ou le Raspberry Pi.',
    icon: Camera,
    step: 1,
    component: lazy(() => import('@/pages/capture')),
  },
  {
    path: 'detection',
    label: 'Détection et suivi',
    description: 'Détecter les œufs et reconstituer leurs pistes sur le convoyeur.',
    icon: ScanSearch,
    step: 2,
    component: lazy(() => import('@/pages/detection')),
  },
  {
    path: 'labellisation',
    label: 'Labellisation',
    description: 'Relire les pistes et attribuer verdicts et défauts.',
    icon: Tags,
    step: 3,
    component: lazy(() => import('@/pages/labellisation')),
  },
  {
    path: 'entrainement',
    label: 'Entraînement',
    description: 'Entraîner et évaluer les modèles de classification.',
    icon: BrainCircuit,
    step: 4,
    component: lazy(() => import('@/pages/entrainement')),
  },
  {
    path: 'export',
    label: 'Export Raspberry Pi',
    description: 'Exporter, mesurer et déployer un modèle sur le Pi.',
    icon: Cpu,
    step: 5,
    component: lazy(() => import('@/pages/export')),
  },
  {
    path: 'simulation',
    label: 'Simulation',
    description: 'Rejouer un flux et simuler l’ouverture des trappes.',
    icon: Split,
    step: 6,
    component: lazy(() => import('@/pages/simulation')),
  },
];
