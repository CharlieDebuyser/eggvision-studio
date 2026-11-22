import {
  createContext,
  forwardRef,
  useContext,
  useEffect,
  useId,
  useState,
  type InputHTMLAttributes,
  type ReactNode,
  type SelectHTMLAttributes,
  type TextareaHTMLAttributes,
} from 'react';
import { ChevronDown, Minus, Plus } from 'lucide-react';
import { cx } from '@/lib/cx';
import { formatNumber } from '@/lib/format';

// ---------------------------------------------------------------- Field / Label
interface FieldCtx {
  id: string;
  describedBy?: string;
  invalid: boolean;
}
const FieldContext = createContext<FieldCtx | null>(null);

/** Id et aria-describedby du Field englobant (pour un contrôle maison). */
export const useFieldContext = () => useContext(FieldContext);

export interface FieldProps {
  label?: ReactNode;
  /** Aide sous le contrôle. */
  hint?: ReactNode;
  /** Message d'erreur (remplace l'aide, passe le contrôle en invalide). */
  error?: ReactNode;
  required?: boolean;
  /** Élément à droite du libellé (ex. valeur courante, lien). */
  labelAside?: ReactNode;
  /** Libellé à gauche, contrôle à droite. */
  inline?: boolean;
  children: ReactNode;
  className?: string;
  /** Id imposé du contrôle (sinon généré). */
  id?: string;
}

/**
 * Enveloppe un contrôle avec libellé, aide et erreur. Les contrôles de ce fichier
 * récupèrent automatiquement l'id et aria-describedby :
 *   <Field label="Nom du projet" hint="Lettres, chiffres, tirets"><TextInput value={…} onChange={…} /></Field>
 */
export function Field({ label, hint, error, required, labelAside, inline, children, className, id: forcedId }: FieldProps) {
  const autoId = useId();
  const id = forcedId ?? autoId;
  const hintId = `${id}-hint`;
  const msg = error ?? hint;
  return (
    <FieldContext.Provider value={{ id, describedBy: msg ? hintId : undefined, invalid: !!error }}>
      <div className={cx('field', inline && 'field--inline', !!error && 'is-invalid', className)}>
        {(label || labelAside) && (
          <div className="field__top">
            {label && (
              <Label htmlFor={id} required={required}>
                {label}
              </Label>
            )}
            {labelAside && <span className="field__aside">{labelAside}</span>}
          </div>
        )}
        <div className="field__control">{children}</div>
        {msg && (
          <div id={hintId} className={cx('field__hint', !!error && 'field__hint--error')} role={error ? 'alert' : undefined}>
            {msg}
          </div>
        )}
      </div>
    </FieldContext.Provider>
  );
}

export function Label({ htmlFor, required, children, className }: { htmlFor?: string; required?: boolean; children: ReactNode; className?: string }) {
  return (
    <label htmlFor={htmlFor} className={cx('label', className)}>
      {children}
      {required && (
        <span className="label__req" aria-hidden>
          {' '}*
        </span>
      )}
    </label>
  );
}

function useControlIds(id?: string, invalid?: boolean) {
  const ctx = useFieldContext();
  return {
    id: id ?? ctx?.id,
    'aria-describedby': ctx?.describedBy,
    'aria-invalid': invalid || ctx?.invalid || undefined,
  };
}

// ---------------------------------------------------------------- TextInput / TextArea
export interface TextInputProps extends Omit<InputHTMLAttributes<HTMLInputElement>, 'size'> {
  size?: 'sm' | 'md';
  invalid?: boolean;
  /** Élément à gauche (icône). */
  leading?: ReactNode;
  /** Élément à droite (unité, bouton). */
  trailing?: ReactNode;
}

export const TextInput = forwardRef<HTMLInputElement, TextInputProps>(function TextInput(
  { size = 'md', invalid, leading, trailing, className, id, type = 'text', ...rest },
  ref,
) {
  const ids = useControlIds(id, invalid);
  return (
    <div className={cx('input', `input--${size}`, className)} data-invalid={ids['aria-invalid'] || undefined}>
      {leading && <span className="input__adorn">{leading}</span>}
      <input ref={ref} type={type} className="input__el" {...ids} {...rest} />
      {trailing && <span className="input__adorn input__adorn--end">{trailing}</span>}
    </div>
  );
});

export const TextArea = forwardRef<HTMLTextAreaElement, TextareaHTMLAttributes<HTMLTextAreaElement> & { invalid?: boolean }>(function TextArea(
  { invalid, className, id, ...rest },
  ref,
) {
  const ids = useControlIds(id, invalid);
  return <textarea ref={ref} className={cx('textarea', className)} {...ids} {...rest} />;
});

// ---------------------------------------------------------------- NumberInput
export interface NumberInputProps {
  value: number | null;
  /** Appelé avec un nombre valide borné, ou null si le champ est vidé (si allowEmpty). */
  onChange: (value: number | null) => void;
  min?: number;
  max?: number;
  step?: number;
  /** Unité affichée à droite (« px », « ms », « % »). */
  unit?: string;
  /** Nombre de décimales affichées (défaut : déduit de step). */
  decimals?: number;
  /** Boutons −/+ (défaut true). */
  stepper?: boolean;
  allowEmpty?: boolean;
  disabled?: boolean;
  size?: 'sm' | 'md';
  id?: string;
  className?: string;
  placeholder?: string;
  'aria-label'?: string;
}

function decimalsOf(step: number): number {
  const s = String(step);
  const i = s.indexOf('.');
  return i < 0 ? 0 : s.length - i - 1;
}

const clamp = (v: number, min?: number, max?: number) => Math.min(max ?? Infinity, Math.max(min ?? -Infinity, v));

/**
 * Champ numérique (accepte la virgule). La valeur est validée et bornée à la perte de focus
 * ou sur Entrée ; flèches haut/bas = ±step (Maj = ×10).
 */
export function NumberInput({
  value,
  onChange,
  min,
  max,
  step = 1,
  unit,
  decimals,
  stepper = true,
  allowEmpty,
  disabled,
  size = 'md',
  id,
  className,
  placeholder,
  ...rest
}: NumberInputProps) {
  const ids = useControlIds(id);
  const dec = decimals ?? decimalsOf(step);
  const fmt = (v: number | null) => (v === null || Number.isNaN(v) ? '' : v.toFixed(dec).replace('.', ','));
  const [draft, setDraft] = useState(fmt(value));
  const [focused, setFocused] = useState(false);

  useEffect(() => {
    if (!focused) setDraft(fmt(value));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value, focused, dec]);

  const commit = (raw: string) => {
    const t = raw.trim().replace(/\s/g, '').replace(',', '.');
    if (t === '') {
      if (allowEmpty) onChange(null);
      else setDraft(fmt(value));
      return;
    }
    const n = Number(t);
    if (!Number.isFinite(n)) {
      setDraft(fmt(value));
      return;
    }
    const v = Number(clamp(n, min, max).toFixed(dec));
    setDraft(fmt(v));
    if (v !== value) onChange(v);
  };

  const bump = (dir: 1 | -1, mult = 1) => {
    const base = value ?? min ?? 0;
    const v = Number(clamp(base + dir * step * mult, min, max).toFixed(dec));
    setDraft(fmt(v));
    onChange(v);
  };

  return (
    <div className={cx('input', 'input--number', `input--${size}`, disabled && 'is-disabled', className)} data-invalid={ids['aria-invalid'] || undefined}>
      {stepper && (
        <button type="button" className="input__step" tabIndex={-1} aria-hidden disabled={disabled || (min !== undefined && (value ?? 0) <= min)} onClick={() => bump(-1)}>
          <Minus size={12} />
        </button>
      )}
      <input
        {...ids}
        aria-label={rest['aria-label']}
        className="input__el u-num"
        inputMode="decimal"
        role="spinbutton"
        aria-valuemin={min}
        aria-valuemax={max}
        aria-valuenow={value ?? undefined}
        value={draft}
        placeholder={placeholder}
        disabled={disabled}
        onFocus={() => setFocused(true)}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={(e) => {
          setFocused(false);
          commit(e.target.value);
        }}
        onKeyDown={(e) => {
          if (e.key === 'Enter') commit((e.target as HTMLInputElement).value);
          if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
            e.preventDefault();
            bump(e.key === 'ArrowUp' ? 1 : -1, e.shiftKey ? 10 : 1);
          }
        }}
      />
      {unit && <span className="input__unit">{unit}</span>}
      {stepper && (
        <button type="button" className="input__step" tabIndex={-1} aria-hidden disabled={disabled || (max !== undefined && (value ?? 0) >= max)} onClick={() => bump(1)}>
          <Plus size={12} />
        </button>
      )}
    </div>
  );
}

// ---------------------------------------------------------------- Select
export interface SelectOption<V extends string = string> {
  value: V;
  label: ReactNode;
  disabled?: boolean;
}

export interface SelectProps<V extends string> extends Omit<SelectHTMLAttributes<HTMLSelectElement>, 'value' | 'onChange' | 'size'> {
  value: V;
  onChange: (value: V) => void;
  options: SelectOption<V>[];
  /** Option vide en tête (valeur ''). */
  placeholder?: string;
  size?: 'sm' | 'md';
  invalid?: boolean;
}

/** Liste déroulante native stylée. */
export function Select<V extends string>({ value, onChange, options, placeholder, size = 'md', invalid, className, id, ...rest }: SelectProps<V>) {
  const ids = useControlIds(id, invalid);
  return (
    <div className={cx('select', `select--${size}`, className)} data-invalid={ids['aria-invalid'] || undefined}>
      <select {...ids} {...rest} value={value} onChange={(e) => onChange(e.target.value as V)}>
        {placeholder !== undefined && <option value="">{placeholder}</option>}
        {options.map((o) => (
          <option key={o.value} value={o.value} disabled={o.disabled}>
            {typeof o.label === 'string' ? o.label : o.value}
          </option>
        ))}
      </select>
      <ChevronDown size={14} className="select__chevron" aria-hidden />
    </div>
  );
}

// ---------------------------------------------------------------- Slider
export interface SliderProps {
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  /** Format de la valeur affichée (défaut : nombre FR + unité). */
  format?: (v: number) => ReactNode;
  unit?: string;
  /** Masquer la valeur affichée. */
  hideValue?: boolean;
  disabled?: boolean;
  id?: string;
  className?: string;
  'aria-label'?: string;
}

export function Slider({ value, onChange, min = 0, max = 100, step = 1, format, unit, hideValue, disabled, id, className, ...rest }: SliderProps) {
  const ids = useControlIds(id);
  const pct = max === min ? 0 : ((value - min) / (max - min)) * 100;
  const display = format ? format(value) : `${formatNumber(value, decimalsOf(step), decimalsOf(step))}${unit ? ` ${unit}` : ''}`;
  return (
    <div className={cx('slider', disabled && 'is-disabled', className)}>
      <input
        {...ids}
        aria-label={rest['aria-label']}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
        style={{ ['--slider-pct' as string]: `${pct}%` }}
      />
      {!hideValue && <output className="slider__value u-num">{display}</output>}
    </div>
  );
}

// ---------------------------------------------------------------- Toggle / Checkbox
export interface ToggleProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label?: ReactNode;
  /** Texte secondaire sous le libellé. */
  description?: ReactNode;
  disabled?: boolean;
  size?: 'sm' | 'md';
  id?: string;
  className?: string;
}

/** Interrupteur on/off (role="switch"). */
export function Toggle({ checked, onChange, label, description, disabled, size = 'md', id, className }: ToggleProps) {
  const ids = useControlIds(id);
  const autoId = useId();
  const btnId = ids.id ?? autoId;
  return (
    <div className={cx('toggle', `toggle--${size}`, disabled && 'is-disabled', className)}>
      <button
        id={btnId}
        type="button"
        role="switch"
        aria-checked={checked}
        aria-describedby={ids['aria-describedby']}
        disabled={disabled}
        className={cx('toggle__track', checked && 'is-on')}
        onClick={() => onChange(!checked)}
      >
        <span className="toggle__thumb" />
      </button>
      {(label || description) && (
        <label htmlFor={btnId} className="toggle__text">
          {label && <span className="toggle__label">{label}</span>}
          {description && <span className="toggle__desc">{description}</span>}
        </label>
      )}
    </div>
  );
}

export interface CheckboxProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label?: ReactNode;
  description?: ReactNode;
  indeterminate?: boolean;
  disabled?: boolean;
  id?: string;
  className?: string;
  'aria-label'?: string;
}

export function Checkbox({ checked, onChange, label, description, indeterminate, disabled, id, className, ...rest }: CheckboxProps) {
  const ids = useControlIds(id);
  return (
    <label className={cx('checkbox', disabled && 'is-disabled', className)}>
      <input
        {...ids}
        aria-label={rest['aria-label']}
        type="checkbox"
        checked={checked}
        disabled={disabled}
        ref={(el) => {
          if (el) el.indeterminate = !!indeterminate;
        }}
        onChange={(e) => onChange(e.target.checked)}
      />
      <span className="checkbox__box" aria-hidden />
      {(label || description) && (
        <span className="checkbox__text">
          {label && <span className="checkbox__label">{label}</span>}
          {description && <span className="checkbox__desc">{description}</span>}
        </span>
      )}
    </label>
  );
}

// ---------------------------------------------------------------- SegmentedControl
export interface SegmentedOption<V extends string> {
  value: V;
  label: ReactNode;
  disabled?: boolean;
  title?: string;
}

export interface SegmentedControlProps<V extends string> {
  value: V;
  onChange: (value: V) => void;
  options: SegmentedOption<V>[];
  size?: 'sm' | 'md';
  className?: string;
  'aria-label'?: string;
}

/** Choix exclusif entre quelques options (radiogroup). */
export function SegmentedControl<V extends string>({ value, onChange, options, size = 'md', className, ...rest }: SegmentedControlProps<V>) {
  return (
    <div className={cx('segmented', `segmented--${size}`, className)} role="radiogroup" aria-label={rest['aria-label']}>
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          role="radio"
          aria-checked={o.value === value}
          title={o.title}
          disabled={o.disabled}
          className={cx('segmented__item', o.value === value && 'is-active')}
          onClick={() => onChange(o.value)}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}
