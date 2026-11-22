/**
 * Parcours des pistes d'un lot : statistiques de qualité, filtres (qualité, motif, couloir),
 * mosaïque des vues, informations de qualité et trajectoire des détections sur l'image.
 */
import { useEffect, useMemo, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { CheckCircle2, Footprints, ImageOff, TriangleAlert } from 'lucide-react';
import {
  Badge,
  DataTable,
  EmptyState,
  ErrorState,
  Field,
  KeyValue,
  LoadingState,
  Pagination,
  Panel,
  SegmentedControl,
  Select,
  Stat,
  type Column,
} from '@/components/ui';
import { formatNumber, formatPercent } from '@/lib/format';
import { useDetLots, useLotStats, useTrack, useTracks, type TrackFilter } from './api';
import { DetectionsOverlay, FrameStage, laneColor } from './FrameStage';
import type { TrackRow } from './types';
import s from './detection.module.css';

const PAGE = 30;
type QualityFilter = '' | 'ok' | 'douteuse';

export function Tracks() {
  const [params, setParams] = useSearchParams();
  const lots = useDetLots(0, 200);
  const treated = (lots.data?.items ?? []).filter((l) => l.track_count > 0 || l.detection_status === 'done');
  const lotId = params.get('lot') ?? treated[0]?.id ?? null;
  const lot = (lots.data?.items ?? []).find((l) => l.id === lotId) ?? null;

  const [quality, setQuality] = useState<QualityFilter>('');
  const [flag, setFlag] = useState('');
  const [lane, setLane] = useState('');
  const [page, setPage] = useState(0);
  const [selected, setSelected] = useState<number | null>(null);
  useEffect(() => {
    setPage(0);
    setSelected(null);
  }, [lotId, quality, flag, lane]);

  const filter: TrackFilter = { quality, flag: flag || undefined, lane: lane === '' ? null : Number(lane), page, size: PAGE };
  const tracks = useTracks(lotId, filter);
  const stats = useLotStats(lotId);
  const flagLabels = stats.data?.flag_labels ?? tracks.data?.flag_labels ?? {};

  if (lots.isLoading) return <LoadingState />;
  if (lots.error) return <ErrorState error={lots.error} onRetry={() => void lots.refetch()} />;
  if (!treated.length)
    return <EmptyState icon={Footprints} title="Aucune piste" description="Lancez le traitement d’un lot dans l’onglet « Traitement »." />;

  const st = stats.data;
  const columns: Column<TrackRow>[] = [
    { key: 'id', header: 'Piste', width: 80, render: (r) => <span className="u-mono">#{r.id}</span> },
    {
      key: 'lane',
      header: 'Couloir',
      width: 80,
      render: (r) => (
        <span className="u-row">
          <span className={s.laneSwatch} style={{ background: laneColor(r.lane) }} />
          {r.lane + 1}
        </span>
      ),
    },
    {
      key: 'frames',
      header: 'Images',
      width: 120,
      render: (r) => (
        <span className="u-num u-small">
          {r.start_frame}–{r.end_frame}
        </span>
      ),
    },
    { key: 'views', header: 'Vues', align: 'right', width: 60, render: (r) => formatNumber(r.n_views) },
    {
      key: 'quality',
      header: 'Qualité',
      render: (r) =>
        r.quality === 'ok' ? (
          <Badge tone="success" dot>
            ok
          </Badge>
        ) : (
          <span className={s.flagCell}>
            {r.flags.map((f) => (
              <Badge key={f} tone="warning" variant="soft">
                {f}
              </Badge>
            ))}
          </span>
        ),
    },
  ];

  return (
    <div className="u-stack-4">
      <div className={s.toolbar}>
        <Field label="Lot">
          <Select
            value={lotId ?? ''}
            onChange={(v) => setParams(v ? { lot: v } : {})}
            options={treated.map((l) => ({ value: l.id, label: `${l.name} — ${formatNumber(l.track_count)} pistes` }))}
          />
        </Field>
        <Field label="Qualité">
          <SegmentedControl<QualityFilter>
            value={quality}
            onChange={setQuality}
            options={[
              { value: '', label: 'Toutes' },
              { value: 'ok', label: 'Exploitables' },
              { value: 'douteuse', label: 'Douteuses' },
            ]}
          />
        </Field>
        <Field label="Motif">
          <Select
            value={flag}
            onChange={setFlag}
            placeholder="Tous les motifs"
            options={Object.entries(st?.flags ?? {}).map(([k, n]) => ({ value: k, label: `${flagLabels[k] ?? k} (${n})` }))}
          />
        </Field>
        <Field label="Couloir">
          <Select
            value={lane}
            onChange={setLane}
            placeholder="Tous"
            options={Object.keys(st?.lanes ?? {}).map((k) => ({ value: k, label: `Couloir ${Number(k) + 1}` }))}
          />
        </Field>
      </div>

      {st && (
        <div className={s.statRow}>
          <Stat label="Pistes" value={formatNumber(st.tracks)} hint={`${formatNumber(st.doubtful)} douteuses`} />
          <Stat
            label={`Exploitables (≥ ${st.min_views} vues)`}
            value={formatPercent(st.usable_rate, 1)}
            hint={`${formatNumber(st.usable)} pistes`}
            icon={CheckCircle2}
            tone="success"
          />
          <Stat
            label="Exploitables parmi les traversées complètes"
            value={formatPercent(st.usable_rate_complete, 1)}
            hint={`${formatNumber(st.complete_tracks)} pistes complètes — critère ≥ 95 %`}
          />
          <Stat
            label="Traitement"
            value={st.run ? `${formatNumber(st.run.ms_per_frame, 1)} ms/image` : '—'}
            hint={st.run?.speed_mm_s ? `vitesse estimée ${formatNumber(st.run.speed_mm_s, 0)} mm/s` : undefined}
          />
        </div>
      )}

      <div className={s.splitLayout}>
        <div className={s.mainCol}>
          <Panel flush title="Pistes" description={lot ? lot.name : undefined}>
            <DataTable
              columns={columns}
              rows={tracks.data?.items}
              rowKey={(r) => r.id}
              loading={tracks.isLoading}
              error={tracks.error}
              onRowClick={(r) => setSelected(r.id)}
              selectedKey={selected}
              dense
              empty={<EmptyState icon={Footprints} title="Aucune piste pour ces filtres" compact />}
            />
            {(tracks.data?.total ?? 0) > PAGE && (
              <div className={s.pagination}>
                <Pagination page={page} pageSize={PAGE} total={tracks.data?.total ?? 0} onPageChange={setPage} />
              </div>
            )}
          </Panel>
        </div>
        <div className={s.sideColWide}>
          {selected === null ? (
            <Panel>
              <EmptyState icon={Footprints} title="Choisissez une piste" description="Ses vues, sa qualité et sa trajectoire s’affichent ici." compact />
            </Panel>
          ) : (
            <TrackPanel trackId={selected} lotWidth={lot?.width ?? 0} lotHeight={lot?.height ?? 0} flagLabels={flagLabels} />
          )}
        </div>
      </div>
    </div>
  );
}

function TrackPanel({ trackId, lotWidth, lotHeight, flagLabels }: { trackId: number; lotWidth: number; lotHeight: number; flagLabels: Record<string, string> }) {
  const track = useTrack(trackId);
  const [view, setView] = useState(0);
  useEffect(() => setView(0), [trackId]);
  const t = track.data;
  const frameIndex = useMemo(() => {
    if (!t) return 0;
    return t.views[view]?.frame_index ?? t.detections[Math.floor(t.detections.length / 2)]?.frame_index ?? t.start_frame;
  }, [t, view]);

  if (track.isLoading) return <Panel><LoadingState /></Panel>;
  if (track.error || !t) return <Panel><ErrorState error={track.error} /></Panel>;

  const path = t.detections.map((d) => `${d.cx},${d.cy}`).join(' ');
  return (
    <Panel
      title={`Piste #${t.id}`}
      description={`Couloir ${t.lane + 1} · images ${t.start_frame} à ${t.end_frame}`}
      actions={
        t.quality === 'ok' ? (
          <Badge tone="success" icon={CheckCircle2}>Exploitable</Badge>
        ) : (
          <Badge tone="warning" icon={TriangleAlert}>Douteuse</Badge>
        )
      }
    >
      <div className="u-stack-4">
        {t.flags.length > 0 && (
          <ul className={s.plainList}>
            {t.flags.map((f) => (
              <li key={f}>
                <Badge tone="warning">{f}</Badge> <span className="u-small">{flagLabels[f] ?? t.flag_labels[f] ?? f}</span>
              </li>
            ))}
          </ul>
        )}
        {t.views.length ? (
          <div className={s.mosaic}>
            {t.views.map((v) => (
              <button
                key={v.index}
                type="button"
                className={v.index === view ? s.mosaicItemActive : s.mosaicItem}
                onClick={() => setView(v.index)}
                title={`Vue ${v.index + 1} — image ${v.frame_index} — netteté ${formatNumber(v.sharpness, 0)}`}
              >
                <img src={v.url} alt={`Vue ${v.index + 1}`} loading="lazy" />
              </button>
            ))}
          </div>
        ) : (
          <EmptyState icon={ImageOff} title="Aucune vue" description="Aucune détection complète et nette n’a pu être recadrée." compact />
        )}
        {lotWidth > 0 && (
          <FrameStage lotId={t.lot_id} frameIndex={frameIndex} width={lotWidth} height={lotHeight} thumb={900}>
            <polyline points={path} fill="none" stroke="var(--accent)" strokeWidth={2} vectorEffect="non-scaling-stroke" />
            {t.detections.map((d) => (
              <circle key={d.id} cx={d.cx} cy={d.cy} r={Math.max(2, d.minor / 40)} fill={d.partial ? 'var(--text-subtle)' : 'var(--accent)'} />
            ))}
            <DetectionsOverlay detections={t.detections.filter((d) => d.frame_index === frameIndex)} showCenters={false} />
          </FrameStage>
        )}
        <KeyValue
          items={[
            { label: 'Détections', value: `${formatNumber(t.n_detections)} (dont ${formatPercent(t.partial_ratio, 0)} tronquées)` },
            { label: 'Vues', value: formatNumber(t.n_views) },
            { label: 'Durée', value: `${formatNumber(t.end_ts - t.start_ts, 2)} s` },
            { label: 'Vitesse', value: `${formatNumber(t.speed_px_s, 0)} px/s` },
            { label: 'Taille moyenne', value: `${formatNumber(t.mean_major_px, 0)} × ${formatNumber(t.mean_minor_px, 0)} px` },
            { label: 'Netteté moyenne', value: formatNumber(t.mean_sharpness, 0) },
            { label: 'Traversée', value: `${t.entered ? 'entrée vue' : 'entrée manquée'} · ${t.exited ? 'sortie vue' : 'sortie manquée'}` },
            { label: 'Label', value: t.label ? `${t.label.verdict ?? '—'}${t.label.defect ? ` (${t.label.defect})` : ''} · ${t.label.reviewed ? 'relu' : t.label.source}` : '—' },
          ]}
        />
      </div>
    </Panel>
  );
}

export default Tracks;
