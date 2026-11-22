/**
 * Calibration de la scène (une fois par installation) : image de référence, couloirs
 * (quadrilatères à poignées, proposition automatique), sens de défilement, échelle (règle).
 */
import { useEffect, useState, type PointerEvent as ReactPointerEvent } from 'react';
import { ArrowDown, ArrowLeft, ArrowRight, ArrowUp, Eye, EyeOff, MousePointer2, Ruler, Save, Sparkles, Trash2, Undo2, Wand2 } from 'lucide-react';
import { errorMessage } from '@/api/client';
import {
  Badge,
  Button,
  Callout,
  EmptyState,
  ErrorState,
  Field,
  IconButton,
  LoadingState,
  NumberInput,
  Panel,
  SegmentedControl,
  useToast,
} from '@/components/ui';
import { formatNumber } from '@/lib/format';
import { proposeLanes, rulerScale, useDebounced, useDetLots, usePreview, useSaveScene, useScene } from './api';
import { DetectionsOverlay, DirectionArrow, FrameStage, LanesOverlay, LotFramePicker, laneColor, type ImagePoint } from './FrameStage';
import type { Direction, Lane, Scene } from './types';
import s from './detection.module.css';

type Tool = 'edit' | 'ruler';
type Drag = { kind: 'vertex'; lane: number; vertex: number } | { kind: 'lane'; lane: number; last: ImagePoint };

const DIRECTION_OPTIONS: { value: Direction; label: JSX.Element; title: string }[] = [
  { value: 'down', label: <ArrowDown size={14} aria-label="Vers le bas" />, title: 'Vers le bas' },
  { value: 'up', label: <ArrowUp size={14} aria-label="Vers le haut" />, title: 'Vers le haut' },
  { value: 'right', label: <ArrowRight size={14} aria-label="Vers la droite" />, title: 'Vers la droite' },
  { value: 'left', label: <ArrowLeft size={14} aria-label="Vers la gauche" />, title: 'Vers la gauche' },
];

export function Calibration() {
  const toast = useToast();
  const lots = useDetLots(0, 200);
  const lotItems = lots.data?.items ?? [];
  const sceneQ = useScene();
  const saved = sceneQ.data?.scene ?? null;

  const [lotId, setLotId] = useState<string | null>(null);
  const [frame, setFrame] = useState(0);
  // Lot initial : celui de l'image de référence enregistrée, sinon le premier lot avec des images.
  useEffect(() => {
    if (lotId || !lotItems.length || sceneQ.isLoading) return;
    const ref = saved?.reference;
    if (ref && lotItems.some((l) => l.id === ref.lot_id)) {
      setLotId(ref.lot_id);
      setFrame(ref.frame_index);
    } else {
      const first = lotItems.find((l) => l.frame_count > 0);
      if (first) {
        setLotId(first.id);
        setFrame(Math.floor(first.frame_count / 2));
      }
    }
  }, [lotItems, lotId, saved, sceneQ.isLoading]);

  const lot = lotItems.find((l) => l.id === lotId) ?? null;
  const lotScene = useScene(lotId ?? undefined);
  const [draft, setDraft] = useState<Scene | null>(null);
  const [dirty, setDirty] = useState(false);

  // Brouillon initial : calibration enregistrée si elle est à la résolution du lot, sinon calibration adaptée au lot.
  useEffect(() => {
    if (dirty || !lot || !lotScene.data) return;
    const base = saved && saved.image_width === lot.width && saved.image_height === lot.height ? saved : lotScene.data.lot_scene;
    if (base) setDraft(structuredClone(base));
  }, [lot, lotScene.data, saved, dirty]);

  const update = (fn: (d: Scene) => Scene) => {
    setDraft((d) => (d ? fn(structuredClone(d)) : d));
    setDirty(true);
  };

  const [tool, setTool] = useState<Tool>('edit');
  const [drag, setDrag] = useState<Drag | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const [ruler, setRuler] = useState<ImagePoint[]>([]);
  const [rulerMm, setRulerMm] = useState<number | null>(100);
  const [nLanes, setNLanes] = useState<number | null>(6);
  const [proposing, setProposing] = useState(false);
  const [showDets, setShowDets] = useState(true);

  const debouncedScene = useDebounced(draft, 400);
  const preview = usePreview(
    showDets && lotId && debouncedScene && !drag
      ? { lot_id: lotId, frame_index: frame, scene: debouncedScene }
      : null,
  );

  const save = useSaveScene();

  if (lots.isLoading || sceneQ.isLoading) return <LoadingState />;
  if (lots.error) return <ErrorState error={lots.error} onRetry={() => void lots.refetch()} />;
  if (!lotItems.some((l) => l.frame_count > 0))
    return (
      <EmptyState
        icon={Ruler}
        title="Aucune image disponible"
        description="La calibration se fait sur une image de référence : enregistrez d'abord un lot dans « Prise de vue »."
      />
    );

  const W = draft?.image_width ?? lot?.width ?? 1;
  const H = draft?.image_height ?? lot?.height ?? 1;
  const handleR = Math.max(5, W / 140);

  const onDown = (p: ImagePoint, e: ReactPointerEvent<SVGSVGElement>) => {
    if (tool === 'ruler') {
      setRuler((r) => (r.length >= 2 ? [p] : [...r, p]));
      return;
    }
    const target = e.target as SVGElement;
    const laneAttr = target.getAttribute('data-lane');
    if (laneAttr === null) {
      setSelected(null);
      return;
    }
    const laneIdx = Number(laneAttr);
    const vertex = Number(target.getAttribute('data-vertex') ?? -1);
    setSelected(laneIdx);
    (e.currentTarget as SVGSVGElement).setPointerCapture(e.pointerId);
    setDrag(vertex >= 0 ? { kind: 'vertex', lane: laneIdx, vertex } : { kind: 'lane', lane: laneIdx, last: p });
  };
  const onMove = (p: ImagePoint) => {
    if (!drag) return;
    const x = Math.min(W, Math.max(0, p[0]));
    const y = Math.min(H, Math.max(0, p[1]));
    update((d) => {
      const lane = d.lanes.find((l) => l.index === drag.lane);
      if (!lane) return d;
      if (drag.kind === 'vertex') {
        lane.polygon[drag.vertex] = [round1(x), round1(y)];
      } else {
        const dx = p[0] - drag.last[0];
        const dy = p[1] - drag.last[1];
        lane.polygon = lane.polygon.map(([px, py]) => [round1(px + dx), round1(py + dy)]);
      }
      return d;
    });
    if (drag.kind === 'lane') setDrag({ ...drag, last: p });
  };
  const onUp = () => setDrag(null);

  const propose = async (mode: 'regular' | 'detections') => {
    if (!lotId || !draft) return;
    setProposing(true);
    try {
      const r = await proposeLanes({
        lot_id: lotId,
        mode,
        n_lanes: nLanes ?? 6,
        direction: draft.direction,
        mm_per_px: draft.mm_per_px,
        frame_index: mode === 'detections' ? undefined : frame,
      });
      update((d) => ({ ...d, lanes: r.lanes }));
      setSelected(null);
      toast.success(
        'Couloirs proposés',
        mode === 'detections' ? `${formatNumber(r.n_points)} œufs détectés sur des images du lot ont servi à placer les limites.` : 'Découpage régulier.',
      );
    } catch (err) {
      toast.error(err);
    } finally {
      setProposing(false);
    }
  };

  const applyRuler = async () => {
    if (ruler.length !== 2 || !rulerMm) return;
    try {
      const r = await rulerScale(ruler[0], ruler[1], rulerMm);
      update((d) => ({ ...d, mm_per_px: r.mm_per_px }));
      toast.success('Échelle mise à jour', `${formatNumber(r.mm_per_px, 4)} mm par pixel (${formatNumber(r.distance_px, 1)} px pour ${formatNumber(rulerMm)} mm).`);
    } catch (err) {
      toast.error(err);
    }
  };

  const onSave = async () => {
    if (!draft || !lotId) return;
    try {
      await save.mutateAsync({ ...draft, reference: { lot_id: lotId, frame_index: frame } });
      setDirty(false);
      toast.success('Calibration enregistrée', 'Elle sera utilisée pour la détection, le suivi et l’aperçu de la prise de vue.');
    } catch (err) {
      toast.error(errorMessage(err));
    }
  };

  const rulerLen = ruler.length === 2 ? Math.hypot(ruler[1][0] - ruler[0][0], ruler[1][1] - ruler[0][1]) : null;

  return (
    <div className={s.splitLayout}>
      <div className={s.mainCol}>
        <Panel
          flush
          title="Image de référence"
          description={
            tool === 'ruler'
              ? 'Règle : cliquez les deux extrémités d’une longueur connue visible dans l’image.'
              : 'Faites glisser les poignées pour ajuster un couloir, ou l’intérieur d’un couloir pour le déplacer.'
          }
          actions={
            <div className="u-row">
              <SegmentedControl<Tool>
                size="sm"
                value={tool}
                onChange={(t) => {
                  setTool(t);
                  setRuler([]);
                }}
                aria-label="Outil"
                options={[
                  { value: 'edit', label: <span className="u-row"><MousePointer2 size={13} aria-hidden /> Couloirs</span> },
                  { value: 'ruler', label: <span className="u-row"><Ruler size={13} aria-hidden /> Règle</span> },
                ]}
              />
              <IconButton icon={showDets ? Eye : EyeOff} label={showDets ? 'Masquer les détections' : 'Afficher les détections'} size="sm" active={showDets} onClick={() => setShowDets((v) => !v)} />
            </div>
          }
        >
          <div className={s.stagePad}>
            {lotId && draft ? (
              <FrameStage
                lotId={lotId}
                frameIndex={frame}
                width={W}
                height={H}
                thumb={1456}
                onPointerDown={onDown}
                onPointerMove={onMove}
                onPointerUp={onUp}
                cursor={tool === 'ruler' ? 'crosshair' : drag ? 'grabbing' : undefined}
              >
                {showDets && preview.data && <DetectionsOverlay detections={preview.data.detections} showCenters />}
                <LanesOverlay lanes={draft.lanes} width={W} dim={tool === 'ruler'} />
                {tool === 'edit' && <LaneHandles lanes={draft.lanes} selected={selected} r={handleR} />}
                <DirectionArrow direction={draft.direction} width={W} height={H} />
                {tool === 'ruler' && <RulerOverlay points={ruler} r={handleR} />}
              </FrameStage>
            ) : (
              <LoadingState />
            )}
          </div>
          <div className={s.stageFooter}>
            <LotFramePicker
              lots={lotItems}
              lotId={lotId}
              onLot={(id) => {
                const l = lotItems.find((x) => x.id === id);
                setLotId(id);
                setFrame(l ? Math.floor(l.frame_count / 2) : 0);
              }}
              frame={frame}
              onFrame={setFrame}
              frameCount={lot?.frame_count ?? 1}
            />
          </div>
        </Panel>
      </div>

      <div className={s.sideCol}>
        <Panel
          title="Calibration"
          description={sceneQ.data?.configured ? 'Calibration enregistrée dans le projet.' : 'Aucune calibration enregistrée : valeurs par défaut.'}
          actions={sceneQ.data?.configured ? <Badge tone="success" dot>Enregistrée</Badge> : <Badge tone="warning" dot>Par défaut</Badge>}
          footer={
            <div className="u-row">
              <Button
                variant="ghost"
                icon={Undo2}
                disabled={!dirty}
                onClick={() => {
                  setDirty(false);
                  setRuler([]);
                }}
              >
                Annuler
              </Button>
              <span className="u-spacer" />
              <Button variant="primary" icon={Save} onClick={onSave} loading={save.isPending} disabled={!draft}>
                Enregistrer
              </Button>
            </div>
          }
        >
          {draft && (
            <div className="u-stack-4">
              <Field label="Sens de défilement" hint="Sens dans lequel les œufs traversent l’image.">
                <SegmentedControl<Direction>
                  value={draft.direction}
                  onChange={(d) => update((x) => ({ ...x, direction: d }))}
                  options={DIRECTION_OPTIONS}
                  aria-label="Sens de défilement"
                />
              </Field>

              <Field label="Échelle" hint="Utilisez l’outil « Règle » sur une règle visible dans l’image, ou saisissez la valeur.">
                <NumberInput
                  value={draft.mm_per_px}
                  onChange={(v) => v && update((x) => ({ ...x, mm_per_px: v }))}
                  min={0.001}
                  max={20}
                  step={0.0001}
                  decimals={4}
                  unit="mm/px"
                />
              </Field>

              {tool === 'ruler' && (
                <Callout tone="info" icon={Ruler} title="Règle">
                  <div className="u-stack-4">
                    <span>
                      {ruler.length < 2
                        ? `Cliquez le ${ruler.length === 0 ? 'premier' : 'second'} point sur l’image.`
                        : `Segment de ${formatNumber(rulerLen, 1)} px.`}
                    </span>
                    <Field label="Longueur réelle">
                      <NumberInput value={rulerMm} onChange={setRulerMm} min={1} max={5000} step={1} unit="mm" />
                    </Field>
                    <Button size="sm" variant="primary" disabled={ruler.length !== 2 || !rulerMm} onClick={applyRuler}>
                      Appliquer l’échelle
                    </Button>
                  </div>
                </Callout>
              )}

              <Field label="Couloirs" hint="Un quadrilatère par couloir, numérotés dans l’ordre des couloirs de la machine.">
                <div className="u-stack-4">
                  <div className="u-row">
                    <NumberInput value={nLanes} onChange={setNLanes} min={1} max={32} step={1} size="sm" aria-label="Nombre de couloirs" />
                    <span className="u-small u-muted">couloirs</span>
                  </div>
                  <div className="u-row">
                    <Button size="sm" icon={Sparkles} onClick={() => propose('detections')} loading={proposing}>
                      Déduire des œufs
                    </Button>
                    <Button size="sm" variant="ghost" icon={Wand2} onClick={() => propose('regular')} disabled={proposing}>
                      Découpage régulier
                    </Button>
                  </div>
                  <ul className={s.laneList}>
                    {draft.lanes.map((lane) => (
                      <li key={lane.index} className={selected === lane.index ? s.laneSelected : undefined}>
                        <span className={s.laneSwatch} style={{ background: laneColor(lane.index) }} />
                        <button type="button" className={s.linkButton} onClick={() => setSelected(lane.index)}>
                          Couloir {lane.index + 1}
                        </button>
                        <span className="u-subtle u-xs u-num">{lane.polygon.length} sommets</span>
                        <span className="u-spacer" />
                        <IconButton
                          icon={Trash2}
                          size="sm"
                          label={`Supprimer le couloir ${lane.index + 1}`}
                          onClick={() => update((d) => ({ ...d, lanes: renumber(d.lanes.filter((l) => l.index !== lane.index)) }))}
                        />
                      </li>
                    ))}
                  </ul>
                </div>
              </Field>

              {preview.data && showDets && (
                <p className="u-small u-muted">
                  Aperçu : {formatNumber(preview.data.count)} œufs détectés, dont{' '}
                  {formatNumber(preview.data.detections.filter((d) => d.lane < 0).length)} hors couloir.
                </p>
              )}
              {dirty && (
                <Callout tone="warning" title="Modifications non enregistrées">
                  La détection et le suivi utilisent la calibration enregistrée.
                </Callout>
              )}
            </div>
          )}
        </Panel>
      </div>
    </div>
  );
}

const round1 = (v: number) => Math.round(v * 10) / 10;
const renumber = (lanes: Lane[]) => lanes.map((l, i) => ({ ...l, index: i }));

function LaneHandles({ lanes, selected, r }: { lanes: Lane[]; selected: number | null; r: number }) {
  return (
    <g>
      {lanes.map((lane) => (
        <g key={lane.index}>
          <polygon
            data-lane={lane.index}
            points={lane.polygon.map((p) => p.join(',')).join(' ')}
            fill="transparent"
            stroke={selected === lane.index ? laneColor(lane.index) : 'transparent'}
            strokeWidth={3}
            vectorEffect="non-scaling-stroke"
            style={{ cursor: 'move' }}
          />
          {lane.polygon.map(([x, y], v) => (
            <circle
              key={v}
              data-lane={lane.index}
              data-vertex={v}
              cx={x}
              cy={y}
              r={selected === lane.index ? r * 1.3 : r}
              className={s.handle}
              stroke={laneColor(lane.index)}
              vectorEffect="non-scaling-stroke"
            />
          ))}
        </g>
      ))}
    </g>
  );
}

function RulerOverlay({ points, r }: { points: ImagePoint[]; r: number }) {
  return (
    <g pointerEvents="none" className={s.ruler}>
      {points.length === 2 && (
        <line x1={points[0][0]} y1={points[0][1]} x2={points[1][0]} y2={points[1][1]} strokeWidth={3} vectorEffect="non-scaling-stroke" />
      )}
      {points.map(([x, y], i) => (
        <circle key={i} cx={x} cy={y} r={r} strokeWidth={2} vectorEffect="non-scaling-stroke" />
      ))}
    </g>
  );
}

export default Calibration;
