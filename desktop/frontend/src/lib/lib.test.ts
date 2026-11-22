import { describe, expect, it } from 'vitest';
import { formatBytes, formatDuration, formatNumber, formatPercent, plural } from './format';
import { matchCombo, parseCombo } from './hotkeys';
import { DEFECTS, defectByHotkey, LOT_TYPES, parseLotType, verdictLabel } from './domain';

const nb = (s: string) => s.replace(/[  ]/g, ' ');

describe('format', () => {
  it('formate en français', () => {
    expect(nb(formatNumber(12345.6))).toBe('12 346');
    expect(formatNumber(0.1234, 2)).toBe('0,12');
    expect(nb(formatPercent(0.875))).toBe('87,5 %');
    expect(formatNumber(null)).toBe('—');
    expect(nb(formatBytes(1536))).toBe('1,5 Ko');
    expect(nb(formatDuration(0.25))).toBe('250 ms');
    expect(nb(formatDuration(185))).toBe('3 min 05 s');
    expect(nb(formatDuration(7440))).toBe('2 h 04 min');
    expect(nb(plural(3, 'image'))).toBe('3 images');
  });
});

describe('hotkeys', () => {
  const ev = (init: KeyboardEventInit) => new KeyboardEvent('keydown', init);
  it('reconnaît les chiffres en AZERTY (Maj + code physique)', () => {
    expect(matchCombo(ev({ key: '&', code: 'Digit1' }), parseCombo('1'))).toBe(true);
    expect(matchCombo(ev({ key: '1', code: 'Digit1', shiftKey: true }), parseCombo('1'))).toBe(true);
    expect(matchCombo(ev({ key: '2', code: 'Digit2' }), parseCombo('1'))).toBe(false);
  });
  it('gère les modificateurs', () => {
    expect(matchCombo(ev({ key: 'ArrowRight' }), parseCombo('ArrowRight'))).toBe(true);
    expect(matchCombo(ev({ key: 'ArrowRight', shiftKey: true }), parseCombo('ArrowRight'))).toBe(false);
    expect(matchCombo(ev({ key: 'Enter', shiftKey: true }), parseCombo('shift+Enter'))).toBe(true);
    expect(matchCombo(ev({ key: 's', ctrlKey: true }), parseCombo('ctrl+s'))).toBe(true);
  });
});

describe('domain', () => {
  it('expose défauts, verdicts et types de lot', () => {
    expect(DEFECTS.map((d) => d.hotkey)).toEqual(['1', '2', '3', '4', '5', '6']);
    expect(defectByHotkey('3')?.label).toBe('Fêlure');
    expect(verdictLabel('non_viable')).toBe('Non viable');
    expect(parseLotType('declasse:sang')).toEqual({ base: 'declasse', defect: 'sang' });
    expect(LOT_TYPES.map((t) => t.value)).toContain('declasse:deformation');
  });
});
