/**
 * Détecteur entraînable (option) : images d'annotation avec boîtes générées automatiquement par la
 * méthode classique, correction des boîtes à la main, entraînement, activation.
 */
import { useEffect, useState, type PointerEvent as ReactPointerEvent } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { BrainCircuit, CheckCircle2, ImagePlus, Play, Save, Square, Trash2, Undo2 } from 'lucide-react';
import { JobProgress } from '@/components/JobProgress';
import {
  Badge,
  Button,
  Callout,
  ConfirmDialog,
  EmptyState,
  ErrorState,
  Field,
  IconButton,
  LoadingState,
  NumberInput,
  Pagination,
  Panel,
  Select,
  TextInput,
  Toggle,
  useToast,
} from '@/components/ui';
import { useHotkeys } from '@/lib/hotkeys';
import { formatDateTime, formatNumber, formatPercent } from '@/lib/format';
import {
  detKeys,
  useActivateDetector,
  useAddAnnotations,
  useAnnotation,
  useAnnotations,
  useDeleteAnnotation,
  useDeleteDetector,
  useDetector,
  useDetLots,
  useSaveBoxes,
  useTrainDetector,
} from './api';
import { FrameStage, type ImagePoint } from './FrameStage';
import type { Box } from './types';
import s from './detection.module.css';

const PAGE = 12;
type Drag = { kind: 'new'; start: ImagePoint } | { kind: 'move'; index: number; last: ImagePoint } | { kind: 'resize'; index: number; corner: number };

export function Detector() {
  const toast = useToast();
  const qc = useQueryClient();
  const info = useDetector();
  const lots = useDetLots(0, 200);
  const [page, setPage] = useState(0);
  const anns = useAnnotations(page, PAGE);
  const [current, setCurrent] = useState<string | null>(null);
  const [addLot, setAddLot] = useState('');
  const [addN, setAddN] = useState<number | null>(10);
  const add = useAddAnnotations();

  const [epochs, setEpochs] = useState<number | null>(60);
  const [inputWidth, setInputWidth] = useState<number | null>(256);
  const [name, setName] = useState('');
  const [activateAfter, setActivateAfter] = useState(true);
  const [jobId, setJobId] = useState<string | null>(null);
  const train = useTrainDetector();
  const activate = useActivateDetector();
  const del = useDeleteDetector();

  useEffect(() => {
    if (!current && anns.data?.items.length) setCurrent(anns.data.items[0].id);
  }, [anns.data, current]);

  if (info.isLoading) return <LoadingState />;
  if (info.error) return <ErrorState error={info.error} onRetry={() => void info.refetch()} />;
  const d = info.data!;

  const onAdd = async () => {
    if (!addLot) return;
    try {
      const r = await add.mutateAsync({ lot_id: addLot, n: addN ?? 10 });
      toast.success('Images ajoutées', `${formatNumber(r.added)} image(s) avec boîtes automatiques à vérifier.`);
    } catch (err) {
      toast.error(err);
    }
  };

  const onTrain = async () => {
    try {
      const job = await train.mutateAsync({ epochs: epochs ?? 60, input_width: inputWidth ?? 256, name, activate: activateAfter });
      setJobId(job.id);
    } catch (err) {
      toast.error(err);
    }
  };

  return (
    <div className="u-stack-4">
      <Callout tone="info" icon={BrainCircuit} title="Option avancée">
        La méthode classique reste la méthode par défaut. Le détecteur entraînable sert si elle ne suffit pas (éclairage, rouleaux, œufs
        atypiques) : il apprend sur des boîtes générées automatiquement puis corrigées à la main. Méthode actuelle :{' '}
        <strong>{d.method === 'nn' ? `détecteur entraînable (${d.active})` : 'classique'}</strong>.
      </Callout>

      <div className={s.detectorLayout}>
        <Panel
          flush
          title="Images d’annotation"
          description={`${formatNumber(d.annotations.total)} images · ${formatNumber(d.annotations.n_boxes)} boîtes · ${formatNumber(d.annotations.n_corrected)} corrigées`}
        >
          <div className={s.annAdd}>
            <Select
              size="sm"
              value={addLot}
              onChange={setAddLot}
              placeholder="Lot source…"
              options={(lots.data?.items ?? []).filter((l) => l.frame_count > 0).map((l) => ({ value: l.id, label: l.name }))}
            />
            <NumberInput size="sm" value={addN} onChange={setAddN} min={1} max={200} step={1} aria-label="Nombre d’images" />
            <Button size="sm" icon={ImagePlus} onClick={onAdd} loading={add.isPending} disabled={!addLot}>
              Ajouter
            </Button>
          </div>
          {anns.isLoading ? (
            <LoadingState />
          ) : !anns.data?.items.length ? (
            <EmptyState icon={ImagePlus} title="Aucune image" description="Ajoutez des images d’un lot : les boîtes sont proposées par la méthode classique." compact />
          ) : (
            <ul className={s.annList}>
              {anns.data.items.map((a) => (
                <li key={a.id}>
                  <button type="button" className={a.id === current ? s.annItemActive : s.annItem} onClick={() => setCurrent(a.id)}>
                    <span className="u-mono u-xs u-truncate">{a.lot_id}</span>
                    <span className="u-small">
                      Image {a.frame_index} · {a.n_boxes} boîtes
                    </span>
                    {a.source === 'corrige' ? <Badge tone="success">corrigée</Badge> : <Badge tone="neutral">auto</Badge>}
                  </button>
                </li>
              ))}
            </ul>
          )}
          {(anns.data?.total ?? 0) > PAGE && (
            <div className={s.pagination}>
              <Pagination page={page} pageSize={PAGE} total={anns.data?.total ?? 0} onPageChange={setPage} />
            </div>
          )}
        </Panel>

        <div className={s.mainCol}>{current ? <BoxEditor key={current} annId={current} onDeleted={() => setCurrent(null)} /> : <Panel><EmptyState icon={Square} title="Choisissez une image" compact /></Panel>}</div>

        <div className="u-stack-4">
          <Panel
            title="Entraînement"
            footer={
              <Button variant="primary" icon={Play} onClick={onTrain} loading={train.isPending} disabled={d.annotations.total < 2 || !!jobId} block>
                Entraîner
              </Button>
            }
          >
            <div className="u-stack-4">
              <Field label="Époques">
                <NumberInput value={epochs} onChange={setEpochs} min={1} max={2000} step={10} size="sm" />
              </Field>
              <Field label="Largeur d’entrée du réseau" hint="Image sous-échantillonnée (multiple de 16).">
                <NumberInput value={inputWidth} onChange={setInputWidth} min={64} max={1024} step={16} unit="px" size="sm" />
              </Field>
              <Field label="Nom">
                <TextInput size="sm" value={name} onChange={(e) => setName(e.target.value)} placeholder="facultatif" />
              </Field>
              <Toggle size="sm" checked={activateAfter} onChange={setActivateAfter} label="Sélectionner le détecteur à la fin" />
              {d.annotations.total < 2 && <p className="u-small u-muted">Au moins deux images d’annotation sont nécessaires.</p>}
              {jobId && (
                <JobProgress
                  jobId={jobId}
                  compact
                  onDone={(job) => {
                    setJobId(null);
                    void qc.invalidateQueries({ queryKey: detKeys.detector });
                    void qc.invalidateQueries({ queryKey: detKeys.params });
                    if (job.status === 'done') toast.success('Détecteur entraîné');
                  }}
                />
              )}
            </div>
          </Panel>

          <Panel title="Détecteurs" flush>
            {!d.detectors.length ? (
              <EmptyState icon={BrainCircuit} title="Aucun détecteur entraîné" compact />
            ) : (
              <ul className={s.detList}>
                {d.detectors.map((m) => (
                  <li key={m.id}>
                    <div className="u-row">
                      <span className="u-strong u-truncate">{m.name}</span>
                      {d.active === m.id && (
                        <Badge tone={d.method === 'nn' ? 'success' : 'neutral'} icon={CheckCircle2}>
                          {d.method === 'nn' ? 'utilisé' : 'sélectionné'}
                        </Badge>
                      )}
                    </div>
                    <span className="u-xs u-subtle">
                      {formatDateTime(m.created_at)} · {m.n_images} images · {m.epochs} époques · {m.device}
                    </span>
                    <span className="u-small u-num">
                      Rappel {formatPercent(m.metrics.recall, 0)} · précision {formatPercent(m.metrics.precision, 0)} ({m.metrics.subset})
                    </span>
                    <div className="u-row">
                      {!(d.method === 'nn' && d.active === m.id) && (
                        <Button size="sm" onClick={() => activate.mutate({ detector_id: m.id, use: true }, { onError: (e) => toast.error(e) })}>
                          Utiliser
                        </Button>
                      )}
                      {d.method === 'nn' && d.active === m.id && (
                        <Button size="sm" variant="ghost" icon={Undo2} onClick={() => activate.mutate({ use: false }, { onError: (e) => toast.error(e) })}>
                          Revenir au classique
                        </Button>
                      )}
                      <span className="u-spacer" />
                      <IconButton
                        icon={Trash2}
                        size="sm"
                        label="Supprimer ce détecteur"
                        onClick={() => del.mutate(m.id, { onError: (e) => toast.error(e), onSuccess: () => toast.success('Détecteur supprimé') })}
                      />
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </Panel>
        </div>
      </div>
    </div>
  );
}

const round1 = (v: number) => Math.round(v * 10) / 10;

function normalize(a: ImagePoint, b: ImagePoint): Box {
  return { x: round1(Math.min(a[0], b[0])), y: round1(Math.min(a[1], b[1])), w: round1(Math.abs(b[0] - a[0])), h: round1(Math.abs(b[1] - a[1])) };
}

function BoxEditor({ annId, onDeleted }: { annId: string; onDeleted: () => void }) {
  const toast = useToast();
  const ann = useAnnotation(annId);
  const saveBoxes = useSaveBoxes();
  const delAnn = useDeleteAnnotation();
  const [boxes, setBoxes] = useState<Box[] | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const [drag, setDrag] = useState<Drag | null>(null);
  const [preview, setPreview] = useState<Box | null>(null);
  const [askDelete, setAskDelete] = useState(false);
  const dirty = !!ann.data && boxes !== null && JSON.stringify(boxes) !== JSON.stringify(ann.data.boxes);

  useEffect(() => {
    if (ann.data && boxes === null) setBoxes(ann.data.boxes);
  }, [ann.data, boxes]);

  const removeSelected = () => {
    if (selected === null) return;
    setBoxes((b) => (b ? b.filter((_, i) => i !== selected) : b));
    setSelected(null);
  };
  useHotkeys({ Delete: removeSelected, Backspace: removeSelected, Escape: () => setSelected(null) });

  if (ann.isLoading || !boxes) return <Panel><LoadingState /></Panel>;
  if (ann.error || !ann.data) return <Panel><ErrorState error={ann.error} /></Panel>;
  const a = ann.data;
  const hr = Math.max(5, a.width / 130);

  const onDown = (p: ImagePoint, e: ReactPointerEvent<SVGSVGElement>) => {
    const t = e.target as SVGElement;
    (e.currentTarget as SVGSVGElement).setPointerCapture(e.pointerId);
    const idx = t.getAttribute('data-box');
    const corner = t.getAttribute('data-corner');
    if (idx !== null && corner !== null) {
      setSelected(Number(idx));
      setDrag({ kind: 'resize', index: Number(idx), corner: Number(corner) });
    } else if (idx !== null) {
      setSelected(Number(idx));
      setDrag({ kind: 'move', index: Number(idx), last: p });
    } else {
      setSelected(null);
      setDrag({ kind: 'new', start: p });
    }
  };
  const onMove = (p: ImagePoint) => {
    if (!drag) return;
    const x = Math.min(a.width, Math.max(0, p[0]));
    const y = Math.min(a.height, Math.max(0, p[1]));
    if (drag.kind === 'new') {
      setPreview(normalize(drag.start, [x, y]));
    } else if (drag.kind === 'move') {
      const dx = p[0] - drag.last[0];
      const dy = p[1] - drag.last[1];
      setBoxes((bs) => bs!.map((b, i) => (i === drag.index ? { ...b, x: round1(b.x + dx), y: round1(b.y + dy) } : b)));
      setDrag({ ...drag, last: p });
    } else {
      setBoxes((bs) =>
        bs!.map((b, i) => {
          if (i !== drag.index) return b;
          const corners: ImagePoint[] = [[b.x, b.y], [b.x + b.w, b.y], [b.x + b.w, b.y + b.h], [b.x, b.y + b.h]];
          const opposite = corners[(drag.corner + 2) % 4];
          return normalize(opposite, [x, y]);
        }),
      );
    }
  };
  const onUp = () => {
    if (drag?.kind === 'new' && preview && preview.w > 8 && preview.h > 8) {
      setBoxes((bs) => [...bs!, preview]);
      setSelected(boxes.length);
    }
    setPreview(null);
    setDrag(null);
  };

  const onSave = async () => {
    try {
      const r = await saveBoxes.mutateAsync({ id: annId, boxes });
      setBoxes(r.boxes);
      toast.success('Boîtes enregistrées', `${formatNumber(r.boxes.length)} boîtes, image marquée « corrigée ».`);
    } catch (err) {
      toast.error(err);
    }
  };

  return (
    <Panel
      flush
      title={`Image ${a.frame_index}`}
      description="Glisser sur une zone vide : nouvelle boîte · glisser une boîte : la déplacer · coins : redimensionner · Suppr : supprimer la boîte choisie."
      actions={
        <div className="u-row">
          <IconButton icon={Trash2} label="Supprimer la boîte choisie" size="sm" disabled={selected === null} onClick={removeSelected} />
          <Button size="sm" variant="ghost" icon={Undo2} disabled={!dirty} onClick={() => setBoxes(a.boxes)}>
            Annuler
          </Button>
          <Button size="sm" variant="primary" icon={Save} disabled={!dirty} loading={saveBoxes.isPending} onClick={onSave}>
            Enregistrer
          </Button>
        </div>
      }
      footer={
        <div className="u-row">
          <span className="u-small u-muted">
            {a.lot_id} · {formatNumber(boxes.length)} boîtes · {a.source === 'corrige' ? 'corrigée' : 'boîtes automatiques'}
          </span>
          <span className="u-spacer" />
          <Button size="sm" variant="ghost" icon={Trash2} onClick={() => setAskDelete(true)}>
            Retirer l’image du jeu
          </Button>
        </div>
      }
    >
      <div className={s.stagePad}>
        <FrameStage lotId={a.lot_id} frameIndex={a.frame_index} width={a.width} height={a.height} thumb={1456} onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp} cursor={drag?.kind === 'move' ? 'grabbing' : 'crosshair'}>
          {boxes.map((b, i) => (
            <g key={i}>
              <rect
                data-box={i}
                x={b.x}
                y={b.y}
                width={b.w}
                height={b.h}
                className={i === selected ? s.boxSelected : s.box}
                vectorEffect="non-scaling-stroke"
                style={{ cursor: 'move' }}
              />
              {i === selected &&
                [[b.x, b.y], [b.x + b.w, b.y], [b.x + b.w, b.y + b.h], [b.x, b.y + b.h]].map(([cx, cy], c) => (
                  <circle key={c} data-box={i} data-corner={c} cx={cx} cy={cy} r={hr} className={s.handle} vectorEffect="non-scaling-stroke" style={{ cursor: 'nwse-resize' }} />
                ))}
            </g>
          ))}
          {preview && <rect x={preview.x} y={preview.y} width={preview.w} height={preview.h} className={s.boxSelected} vectorEffect="non-scaling-stroke" />}
        </FrameStage>
      </div>
      <ConfirmDialog
        open={askDelete}
        variant="danger"
        title="Retirer cette image du jeu d’annotation ?"
        message="Ses boîtes (et corrections) seront perdues. L’image du lot n’est pas supprimée."
        confirmLabel="Retirer"
        onCancel={() => setAskDelete(false)}
        onConfirm={async () => {
          try {
            await delAnn.mutateAsync(annId);
            setAskDelete(false);
            onDeleted();
          } catch (err) {
            toast.error(err);
          }
        }}
      />
    </Panel>
  );
}

export default Detector;
