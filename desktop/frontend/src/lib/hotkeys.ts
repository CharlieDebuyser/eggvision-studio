/**
 * Raccourcis clavier.
 *
 *   useHotkeys({
 *     '1': () => setDefect('souillure'),
 *     'ArrowRight': next,
 *     'mod+s': save,            // mod = Cmd (macOS) ou Ctrl
 *     'shift+Enter': validate,
 *     'Escape': close,
 *   }, { enabled: open });
 *
 * - Les frappes dans <input>, <textarea>, <select> ou un élément contenteditable sont ignorées
 *   (sauf `Escape` et les combinaisons avec `mod`, utiles pour fermer/enregistrer).
 * - Le gestionnaire reçoit l'événement ; preventDefault() est appelé automatiquement.
 * - Noms de touches : valeur de `KeyboardEvent.key` (insensible à la casse pour les lettres),
 *   'Space' pour la barre d'espace. Modificateurs : mod, ctrl, meta, alt, shift.
 */
import { useEffect, useRef } from 'react';

export type HotkeyHandler = (e: KeyboardEvent) => void;
export type HotkeyMap = Record<string, HotkeyHandler>;

export interface HotkeyOptions {
  enabled?: boolean;
  /** Ignorer aussi Escape/mod dans les champs (défaut false). */
  strictInputs?: boolean;
  /** Cible ; défaut window. */
  target?: Window | HTMLElement | null;
}

const isMac = typeof navigator !== 'undefined' && /Mac|iPhone|iPad/i.test(navigator.platform || navigator.userAgent);

export function isEditableTarget(t: EventTarget | null): boolean {
  if (!t || !(t instanceof HTMLElement)) return false;
  if (t.isContentEditable) return true;
  const tag = t.tagName;
  if (tag === 'TEXTAREA' || tag === 'SELECT') return true;
  if (tag === 'INPUT') {
    const type = (t as HTMLInputElement).type;
    return !['checkbox', 'radio', 'button', 'submit', 'reset', 'range', 'color'].includes(type);
  }
  return false;
}

interface ParsedCombo {
  key: string;
  ctrl: boolean;
  meta: boolean;
  alt: boolean;
  shift: boolean;
  mod: boolean;
}

function normKey(k: string): string {
  if (k === ' ' || k.toLowerCase() === 'space') return 'space';
  return k.toLowerCase();
}

export function parseCombo(combo: string): ParsedCombo {
  const parts = combo.split('+').map((p) => p.trim());
  // Gère « shift++ » (touche +)
  const key = parts[parts.length - 1] === '' ? '+' : parts[parts.length - 1];
  const mods = new Set(parts.slice(0, -1).map((p) => p.toLowerCase()));
  return {
    key: normKey(key),
    ctrl: mods.has('ctrl'),
    meta: mods.has('meta') || mods.has('cmd'),
    alt: mods.has('alt') || mods.has('option'),
    shift: mods.has('shift'),
    mod: mods.has('mod'),
  };
}

export function matchCombo(e: KeyboardEvent, c: ParsedCombo): boolean {
  const wantCtrl = c.ctrl || (c.mod && !isMac);
  const wantMeta = c.meta || (c.mod && isMac);
  if (e.ctrlKey !== wantCtrl || e.metaKey !== wantMeta || e.altKey !== c.alt) return false;
  // Chiffres : on se fie à la position physique (e.code) pour fonctionner en AZERTY,
  // où « 1 » s'obtient avec Maj (e.key vaut « & » sans Maj). Maj est alors toléré.
  if (/^[0-9]$/.test(c.key) && !c.shift) {
    return e.key === c.key || e.code === `Digit${c.key}` || e.code === `Numpad${c.key}`;
  }
  if (normKey(e.key) !== c.key) return false;
  // Shift : exigé s'il est demandé ; toléré sinon pour les touches imprimables non lettres (« ? », « + »).
  if (c.shift && !e.shiftKey) return false;
  if (!c.shift && e.shiftKey && /^[a-z0-9]$|^(arrow|enter|tab|space|escape|backspace|delete)/.test(c.key)) return false;
  return true;
}

export function useHotkeys(map: HotkeyMap, options: HotkeyOptions = {}): void {
  const { enabled = true, strictInputs = false, target } = options;
  const mapRef = useRef(map);
  mapRef.current = map;

  useEffect(() => {
    if (!enabled) return;
    const el: Window | HTMLElement = target ?? window;
    const handler = (ev: Event) => {
      const e = ev as KeyboardEvent;
      if (e.defaultPrevented || e.isComposing) return;
      const editable = isEditableTarget(e.target);
      for (const [combo, fn] of Object.entries(mapRef.current)) {
        const c = parseCombo(combo);
        if (!matchCombo(e, c)) continue;
        if (editable) {
          const passthrough = !strictInputs && (c.key === 'escape' || c.mod || c.ctrl || c.meta);
          if (!passthrough) continue;
        }
        e.preventDefault();
        fn(e);
        return;
      }
    };
    el.addEventListener('keydown', handler);
    return () => el.removeEventListener('keydown', handler);
  }, [enabled, strictInputs, target]);
}

/** Libellé affichable d'une combinaison : 'mod+s' → '⌘ S' (macOS) ou 'Ctrl S'. */
export function comboLabel(combo: string): string[] {
  const c = parseCombo(combo);
  const out: string[] = [];
  if (c.mod) out.push(isMac ? '⌘' : 'Ctrl');
  if (c.ctrl) out.push('Ctrl');
  if (c.meta) out.push(isMac ? '⌘' : 'Méta');
  if (c.alt) out.push(isMac ? '⌥' : 'Alt');
  if (c.shift) out.push('⇧');
  const names: Record<string, string> = {
    arrowleft: '←',
    arrowright: '→',
    arrowup: '↑',
    arrowdown: '↓',
    enter: '↵',
    escape: 'Échap',
    space: 'Espace',
    backspace: '⌫',
    delete: 'Suppr',
    tab: 'Tab',
  };
  out.push(names[c.key] ?? (c.key.length === 1 ? c.key.toUpperCase() : c.key));
  return out;
}
