/** Aides au réglage : indice de netteté (historique + maximum atteint), histogramme R/V/B/luminance, saturation. */
import { useEffect, useRef, useState } from 'react';
import { AlertTriangle, Focus, RotateCcw } from 'lucide-react';
import { Badge, Button, Callout, Panel, ProgressBar, Tooltip } from '@/components/ui';
import { formatNumber, formatPercent } from '@/lib/format';
import type { Analysis } from './types';
import s from './capture.module.css';

const HISTORY = 90; // ≈ 30 s à ~3 mesures/s

export function ImageQuality({ analysis, error, sourceKey }: { analysis: Analysis | undefined; error: unknown; sourceKey: string }) {
  const [history, setHistory] = useState<number[]>([]);
  const [peak, setPeak] = useState(0);
  const last = useRef<Analysis | undefined>(undefined);

  useEffect(() => {
    setHistory([]);
    setPeak(0);
  }, [sourceKey]);

  useEffect(() => {
    if (!analysis || analysis === last.current) return;
    last.current = analysis;
    setHistory((h) => [...h.slice(-(HISTORY - 1)), analysis.sharpness]);
    setPeak((p) => Math.max(p, analysis.sharpness));
  }, [analysis]);

  const current = analysis?.sharpness ?? 0;
  const ratio = peak > 0 ? current / peak : 0;
  const sat = analysis?.saturated;

  return (
    <Panel
      title="Qualité de l’image"
      description="Mesurée sur l’aperçu, ~3 fois par seconde."
      actions={
        <Tooltip content="Remettre à zéro le maximum de netteté (nouvelle mise au point)">
          <Button size="sm" variant="ghost" icon={RotateCcw} onClick={() => { setPeak(current); setHistory(current ? [current] : []); }}>
            Réinitialiser
          </Button>
        </Tooltip>
      }
    >
      {error && !analysis ? (
        <p className="u-muted u-small">Analyse indisponible : {String((error as { detail?: string }).detail ?? error)}</p>
      ) : (
        <div className="u-stack-4">
          <div className={s.focusHead}>
            <Focus size={18} aria-hidden className="u-muted" />
            <div className="u-col">
              <span className="u-xs u-muted">Netteté (variance du Laplacien)</span>
              <span className={s.focusValue}>{formatNumber(current, 0)}</span>
            </div>
            <div className={s.focusPeak}>
              <span className="u-xs u-muted">Maximum atteint</span>
              <span className="u-num u-strong">{formatNumber(peak, 0)}</span>
            </div>
          </div>
          <div>
            <ProgressBar value={ratio} tone={ratio >= 0.95 ? 'success' : ratio >= 0.8 ? 'warning' : 'danger'} size="md" label="Netteté par rapport au maximum" />
            <span className="u-xs u-muted">
              {formatPercent(ratio, 0)} du maximum — tourner la bague de mise au point jusqu’à atteindre le maximum.
            </span>
          </div>
          <Sparkline values={history} peak={peak} />
          <Histogram analysis={analysis} />
          {analysis?.saturation_warning ? (
            <Callout tone="danger" icon={AlertTriangle} title="Pixels saturés">
              {formatPercent(sat?.any ?? 0, 2)} des pixels sont saturés (R {formatPercent(sat?.r ?? 0, 2)}, V {formatPercent(sat?.g ?? 0, 2)}, B{' '}
              {formatPercent(sat?.b ?? 0, 2)}). Réduire l’exposition ou le gain.
            </Callout>
          ) : (
            <div className={s.satLine}>
              <Badge tone="success" dot>
                Pas de saturation
              </Badge>
              <span className="u-xs u-muted">
                saturés {formatPercent(sat?.any ?? 0, 2)} · très sombres {formatPercent(analysis?.dark_ratio ?? 0, 1)}
              </span>
            </div>
          )}
        </div>
      )}
    </Panel>
  );
}

function Sparkline({ values, peak }: { values: number[]; peak: number }) {
  const W = 300;
  const H = 48;
  if (values.length < 2) return <div className={s.sparkEmpty}>Historique de netteté…</div>;
  const max = Math.max(peak, ...values) || 1;
  const pts = values.map((v, i) => `${(i / (HISTORY - 1)) * W},${H - (v / max) * (H - 4) - 2}`).join(' ');
  return (
    <svg className={s.spark} viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" role="img" aria-label="Historique de la netteté">
      <line x1={0} x2={W} y1={2} y2={2} className={s.sparkPeak} />
      <polyline points={pts} className={s.sparkLine} />
    </svg>
  );
}

const CHANNELS = [
  { key: 'r', label: 'R', color: 'var(--danger)' },
  { key: 'g', label: 'V', color: 'var(--success)' },
  { key: 'b', label: 'B', color: 'var(--info)' },
  { key: 'luma', label: 'Lum.', color: 'var(--text-muted)' },
] as const;

function Histogram({ analysis }: { analysis: Analysis | undefined }) {
  const W = 256;
  const H = 70;
  const [shown, setShown] = useState<Record<string, boolean>>({ r: true, g: true, b: true, luma: false });
  if (!analysis) return <div className={s.sparkEmpty}>Histogramme…</div>;
  const h = analysis.histogram;
  // Échelle racine : les pics (fond uniforme) n'écrasent pas le reste.
  const max = Math.sqrt(Math.max(1e-9, ...CHANNELS.filter((c) => shown[c.key]).flatMap((c) => h[c.key])));
  const path = (arr: number[]) => {
    const step = W / arr.length;
    return `M0,${H} ` + arr.map((v, i) => `L${i * step},${H - (Math.sqrt(v) / max) * (H - 2)} L${(i + 1) * step},${H - (Math.sqrt(v) / max) * (H - 2)}`).join(' ') + ` L${W},${H} Z`;
  };
  return (
    <div>
      <div className={s.histHead}>
        <span className="u-xs u-muted">Histogramme</span>
        <span className={s.histToggles}>
          {CHANNELS.map((c) => (
            <button
              key={c.key}
              type="button"
              className={s.histToggle}
              data-on={shown[c.key] || undefined}
              style={{ ['--ch' as string]: c.color }}
              onClick={() => setShown((v) => ({ ...v, [c.key]: !v[c.key] }))}
              aria-pressed={shown[c.key]}
            >
              {c.label}
            </button>
          ))}
        </span>
      </div>
      <svg className={s.hist} viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" role="img" aria-label="Histogramme des niveaux">
        {CHANNELS.filter((c) => shown[c.key]).map((c) => (
          <path key={c.key} d={path(h[c.key])} fill={c.color} stroke={c.color} className={s.histPath} />
        ))}
        <line x1={W - 0.5} x2={W - 0.5} y1={0} y2={H} className={s.histSat} />
      </svg>
      <div className={s.histAxis}>
        <span>0</span>
        <span>255</span>
      </div>
    </div>
  );
}
