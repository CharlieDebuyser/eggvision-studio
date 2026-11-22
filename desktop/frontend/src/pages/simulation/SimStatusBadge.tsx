import { Badge } from '@/components/ui';
import type { SimStatus } from './types';

const META: Record<SimStatus, { label: string; tone: 'neutral' | 'info' | 'success' | 'danger' | 'warning' }> = {
  created: { label: 'En attente', tone: 'neutral' },
  running: { label: 'En cours', tone: 'info' },
  done: { label: 'Terminée', tone: 'success' },
  failed: { label: 'Échec', tone: 'danger' },
  cancelled: { label: 'Annulée', tone: 'warning' },
};

export function SimStatusBadge({ status }: { status: SimStatus }) {
  const m = META[status] ?? META.created;
  return (
    <Badge tone={m.tone} dot>
      {m.label}
    </Badge>
  );
}
