/** Explorateur d'erreurs : faux négatifs / faux positifs, vues avec scores, cartes de chaleur, lien vers la labellisation. */
import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Flame, PencilLine, RefreshCw } from 'lucide-react';
import { mediaUrl } from '@/api/client';
import {
  Badge,
  Button,
  DefectBadge,
  EmptyState,
  ErrorState,
  LoadingState,
  Pagination,
  Panel,
  SegmentedControl,
  Toggle,
  Tooltip,
  VerdictBadge,
  useToast,
} from '@/components/ui';
import { formatNumber } from '@/lib/format';
import { camUrl, useCam, useErrors, useRefreshReport } from './api';
import { SUBSET_LABEL } from './common';
import type { ErrorItem, ModelDetail, Subset } from './types';
import s from './entrainement.module.css';

const PAGE = 12;

export function ErrorExplorer({ model }: { model: ModelDetail }) {
  const toast = useToast();
  const refresh = useRefreshReport();
  const [kind, setKind] = useState<'fn' | 'fp'>('fn');
  const [subset, setSubset] = useState<Subset>('test');
  const [page, setPage] = useState(0);
  const [heat, setHeat] = useState(false);
  useEffect(() => setPage(0), [kind, subset]);
  const errors = useErrors(model.id, { kind, subset, threshold: null, limit: PAGE, offset: page * PAGE });
  const fnCount = useErrors(model.id, { kind: 'fn', subset, threshold: null, limit: 1, offset: 0 }).data?.total;
  const fpCount = useErrors(model.id, { kind: 'fp', subset, threshold: null, limit: 1, offset: 0 }).data?.total;

  return (
    <div className="u-stack-4">
      <div className={s.toolbar}>
        <SegmentedControl
          value={kind}
          onChange={setKind}
          options={[
            { value: 'fn', label: `Faux négatifs${fnCount != null ? ` (${fnCount})` : ''}`, title: 'Non viables laissés passer' },
            { value: 'fp', label: `Faux positifs${fpCount != null ? ` (${fpCount})` : ''}`, title: 'Viables éjectés à tort' },
          ]}
        />
        <SegmentedControl
          value={subset}
          onChange={setSubset}
          size="sm"
          options={(['test', 'val', 'train'] as Subset[]).map((v) => ({ value: v, label: SUBSET_LABEL[v] }))}
        />
        <Toggle checked={heat} onChange={setHeat} label={model.kind === 'anomaly' ? "Cartes d'anomalie" : 'Cartes Grad-CAM'} size="sm" />
        <span className="u-spacer" />
        <span className="u-small u-muted">Seuil {formatNumber(model.threshold, 3)}</span>
        <Button
          size="sm"
          icon={RefreshCw}
          loading={refresh.isPending}
          onClick={() => refresh.mutate(model.id, { onSuccess: () => toast.success('Rapport recalculé'), onError: (e) => toast.error(e) })}
        >
          Recalculer après corrections
        </Button>
      </div>
      <p className="u-small u-muted">
        {kind === 'fn'
          ? 'Œufs non viables dont le score est sous le seuil (laissés passer), du plus grave (score le plus bas) au moins grave.'
          : 'Œufs viables dont le score atteint le seuil (éjectés à tort), du score le plus haut au plus bas.'}{' '}
        Les labels affichés sont les labels actuels : un label corrigé disparaît de la liste.
      </p>
      {errors.error ? (
        <ErrorState error={errors.error} onRetry={() => errors.refetch()} />
      ) : !errors.data ? (
        <LoadingState />
      ) : errors.data.items.length === 0 ? (
        <EmptyState title={kind === 'fn' ? 'Aucun faux négatif' : 'Aucun faux positif'} description={`Sur le jeu de ${SUBSET_LABEL[subset].toLowerCase()} au seuil courant.`} />
      ) : (
        <Panel flush>
          <div className={s.errorList} style={{ padding: 'var(--sp-3)' }}>
            {errors.data.items.map((it) => (
              <ErrorCard key={it.track_id} item={it} model={model} heat={heat} />
            ))}
          </div>
          {errors.data.total > PAGE && (
            <div className={s.pagination}>
              <Pagination page={page} pageSize={PAGE} total={errors.data.total} onPageChange={setPage} />
            </div>
          )}
        </Panel>
      )}
    </div>
  );
}

function ErrorCard({ item, model, heat }: { item: ErrorItem; model: ModelDetail; heat: boolean }) {
  const navigate = useNavigate();
  const cam = useCam(model.id, item.track_id, heat);
  const scored = new Map(item.views.map((v) => [v.index, v.score]));
  const n = Math.max(item.n_views, ...item.views.map((v) => v.index + 1));
  const camByIndex = new Map((cam.data?.views ?? []).map((v) => [v.index, v]));
  const maxView = item.views.reduce((b, v) => (v.score > b.score ? v : b), item.views[0] ?? { index: -1, score: 0 });
  return (
    <div className={s.errorCard}>
      <div className={s.errorHead}>
        <span className={s.score}>score {formatNumber(item.score, 3)}</span>
        <VerdictBadge verdict={item.verdict} />
        {item.defect && <DefectBadge defect={item.defect} />}
        {item.reviewed ? <Badge tone="success">relu</Badge> : <Badge>pré-label</Badge>}
        <span className="u-small u-muted">
          piste {item.track_id} · {item.lot_name} · couloir {item.lane + 1}
        </span>
        <span className="u-spacer" />
        <Button size="sm" icon={PencilLine} onClick={() => navigate(`/labellisation/piste/${item.track_id}`)}>
          Corriger le label
        </Button>
      </div>
      {heat && cam.isLoading && <p className="u-xs u-muted">Calcul des cartes de chaleur…</p>}
      {heat && cam.error && <ErrorState error={cam.error} compact />}
      <div className={s.mosaic}>
        {Array.from({ length: n }, (_, i) => {
          const sc = scored.get(i);
          const c = camByIndex.get(i);
          const src = heat && c ? camUrl(c.url) : mediaUrl.crop(item.track_id, i);
          const hot = i === maxView.index;
          return (
            <figure key={i} className={`${s.view} ${hot ? s.viewHot : ''}`}>
              <img src={src} alt={`Vue ${i + 1}`} loading="lazy" />
              <figcaption className={s.viewCaption}>
                <span>vue {i + 1}</span>
                {sc != null ? (
                  <Tooltip content={hot ? 'Vue de score maximal (détermine le score de l’œuf)' : 'Score de défaut de la vue'}>
                    <span className={hot ? s.ko : undefined}>{formatNumber(sc, 2)}</span>
                  </Tooltip>
                ) : (
                  <span title="Vue non utilisée par le modèle">—</span>
                )}
              </figcaption>
            </figure>
          );
        })}
      </div>
      {heat && cam.data && (
        <p className="u-xs u-muted">
          <Flame size={12} aria-hidden /> {cam.data.method === 'grad_cam' ? 'Grad-CAM : zones qui augmentent le score de défaut' : 'Distance des patches à la banque des œufs viables'} — intensité
          normalisée sur l'œuf.
        </p>
      )}
    </div>
  );
}
