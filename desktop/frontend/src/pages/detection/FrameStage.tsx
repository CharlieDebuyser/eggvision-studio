/**
 * Image d'un lot surmontée d'un calque SVG exprimé en pixels de l'image (viewBox = largeur × hauteur) :
 * couloirs, ellipses détectées, boîtes, outils d'édition. Les gestionnaires de pointeur reçoivent
 * directement les coordonnées dans l'image.
 */
import { useRef, type PointerEvent as ReactPointerEvent, type ReactNode } from 'react';
import { ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight } from 'lucide-react';
import { mediaUrl } from '@/api/client';
import { Field, IconButton, Select, Slider } from '@/components/ui';
import { cx } from '@/lib/cx';
import { formatNumber } from '@/lib/format';
import type { DetLot, Detection, Direction, Lane } from './types';
import s from './detection.module.css';

export type ImagePoint = [number, number];
type Handler = (p: ImagePoint, e: ReactPointerEvent<SVGSVGElement>) => void;

export const laneColor = (index: number) => `var(--chart-${(((index % 6) + 6) % 6) + 1})`;

export interface FrameStageProps {
  lotId: string;
  frameIndex: number;
  width: number;
  height: number;
  /** Largeur de l'image demandée au serveur (miniature). */
  thumb?: number;
  /** Image superposée (masque de segmentation, data URL). */
  overlayImage?: string | null;
  onPointerDown?: Handler;
  onPointerMove?: Handler;
  onPointerUp?: Handler;
  cursor?: string;
  className?: string;
  children?: ReactNode;
  label?: string;
}

export function FrameStage({
  lotId,
  frameIndex,
  width,
  height,
  thumb = 1200,
  overlayImage,
  onPointerDown,
  onPointerMove,
  onPointerUp,
  cursor,
  className,
  children,
  label,
}: FrameStageProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const toImage = (e: ReactPointerEvent<SVGSVGElement>): ImagePoint => {
    const r = svgRef.current!.getBoundingClientRect();
    return [((e.clientX - r.left) / r.width) * width, ((e.clientY - r.top) / r.height) * height];
  };
  return (
    <div className={cx(s.stage, className)} style={{ aspectRatio: `${width} / ${height}` }}>
      <img
        className={s.stageImg}
        src={mediaUrl.frame(lotId, frameIndex, Math.min(thumb, width))}
        alt={label ?? `Image ${frameIndex} du lot ${lotId}`}
        draggable={false}
      />
      {overlayImage && <img className={s.stageMask} src={overlayImage} alt="" aria-hidden draggable={false} />}
      <svg
        ref={svgRef}
        className={s.stageSvg}
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="none"
        style={{ cursor }}
        onPointerDown={onPointerDown && ((e) => onPointerDown(toImage(e), e))}
        onPointerMove={onPointerMove && ((e) => onPointerMove(toImage(e), e))}
        onPointerUp={onPointerUp && ((e) => onPointerUp(toImage(e), e))}
      >
        {children}
      </svg>
    </div>
  );
}

/** Couloirs (polygones colorés et numérotés). */
export function LanesOverlay({ lanes, width, dim }: { lanes: Lane[]; width: number; dim?: boolean }) {
  const fs = Math.max(12, width / 45);
  return (
    <g opacity={dim ? 0.55 : 1}>
      {lanes.map((lane) => {
        const pts = lane.polygon;
        const cx = pts.reduce((a, p) => a + p[0], 0) / pts.length;
        const cy = pts.reduce((a, p) => a + p[1], 0) / pts.length;
        return (
          <g key={lane.index}>
            <polygon
              points={pts.map((p) => p.join(',')).join(' ')}
              fill={laneColor(lane.index)}
              fillOpacity={0.07}
              stroke={laneColor(lane.index)}
              strokeWidth={1.5}
              strokeDasharray="6 4"
              vectorEffect="non-scaling-stroke"
            />
            <text x={cx} y={cy} fontSize={fs} textAnchor="middle" dominantBaseline="middle" className={s.laneLabel} fill={laneColor(lane.index)}>
              {lane.index + 1}
            </text>
          </g>
        );
      })}
    </g>
  );
}

/** Ellipses détectées : couleur du couloir, pointillés si l'œuf est tronqué, gris s'il est hors couloir. */
export function DetectionsOverlay({
  detections,
  highlight,
  showCenters = true,
}: {
  detections: Detection[];
  highlight?: (d: Detection) => boolean;
  showCenters?: boolean;
}) {
  return (
    <g>
      {detections.map((d, i) => {
        const color = d.lane >= 0 ? laneColor(d.lane) : 'var(--text-subtle)';
        const hot = highlight?.(d);
        return (
          <g key={d.id ?? i} opacity={highlight && !hot ? 0.35 : 1}>
            <ellipse
              cx={d.cx}
              cy={d.cy}
              rx={d.major / 2}
              ry={d.minor / 2}
              transform={`rotate(${d.angle} ${d.cx} ${d.cy})`}
              fill="none"
              stroke={color}
              strokeWidth={hot ? 3.5 : 2}
              strokeDasharray={d.partial ? '6 5' : undefined}
              vectorEffect="non-scaling-stroke"
            >
              <title>
                {`Couloir ${d.lane >= 0 ? d.lane + 1 : '—'}${d.partial ? ' · partiel' : ''} · ${formatNumber(d.major, 0)}×${formatNumber(d.minor, 0)} px · netteté ${formatNumber(d.sharpness, 0)}`}
              </title>
            </ellipse>
            {showCenters && <circle cx={d.cx} cy={d.cy} r={Math.max(2, d.minor / 30)} fill={color} />}
          </g>
        );
      })}
    </g>
  );
}

const ARROW_ANGLE: Record<Direction, number> = { down: 90, up: -90, right: 0, left: 180 };

/** Flèche du sens de défilement, au centre de l'image. */
export function DirectionArrow({ direction, width, height }: { direction: Direction; width: number; height: number }) {
  const len = Math.min(width, height) * 0.22;
  const cx = width / 2;
  const cy = height / 2;
  return (
    <g transform={`rotate(${ARROW_ANGLE[direction]} ${cx} ${cy})`} className={s.arrow} pointerEvents="none">
      <line x1={cx - len / 2} y1={cy} x2={cx + len / 2} y2={cy} strokeWidth={4} vectorEffect="non-scaling-stroke" />
      <polyline
        points={`${cx + len / 2 - len * 0.18},${cy - len * 0.14} ${cx + len / 2},${cy} ${cx + len / 2 - len * 0.18},${cy + len * 0.14}`}
        fill="none"
        strokeWidth={4}
        vectorEffect="non-scaling-stroke"
      />
    </g>
  );
}

/** Choix d'un lot (parmi ceux ayant des images) et d'une image dans ce lot. */
export function LotFramePicker({
  lots,
  lotId,
  onLot,
  frame,
  onFrame,
  frameCount,
  compact,
}: {
  lots: DetLot[];
  lotId: string | null;
  onLot: (id: string) => void;
  frame: number;
  onFrame: (i: number) => void;
  frameCount: number;
  compact?: boolean;
}) {
  const max = Math.max(0, frameCount - 1);
  const go = (i: number) => onFrame(Math.min(max, Math.max(0, i)));
  return (
    <div className={cx(s.picker, compact && s.pickerCompact)}>
      <Field label="Lot">
        <Select
          value={lotId ?? ''}
          onChange={(v) => v && onLot(v)}
          placeholder="Choisir un lot…"
          options={lots.filter((l) => l.frame_count > 0).map((l) => ({ value: l.id, label: `${l.name} — ${formatNumber(l.frame_count)} images` }))}
        />
      </Field>
      <Field label={`Image ${formatNumber(frame)} / ${formatNumber(max)}`}>
        <div className={s.frameNav}>
          <IconButton icon={ChevronsLeft} label="Reculer de 10 images" size="sm" variant="ghost" onClick={() => go(frame - 10)} disabled={!lotId} />
          <IconButton icon={ChevronLeft} label="Image précédente" size="sm" variant="ghost" onClick={() => go(frame - 1)} disabled={!lotId} />
          <Slider value={frame} onChange={go} min={0} max={max} step={1} hideValue aria-label="Rang de l'image" className={s.frameSlider} disabled={!lotId} />
          <IconButton icon={ChevronRight} label="Image suivante" size="sm" variant="ghost" onClick={() => go(frame + 1)} disabled={!lotId} />
          <IconButton icon={ChevronsRight} label="Avancer de 10 images" size="sm" variant="ghost" onClick={() => go(frame + 10)} disabled={!lotId} />
        </div>
      </Field>
    </div>
  );
}
