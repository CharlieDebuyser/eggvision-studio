import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, within } from '@testing-library/react';
import { Button } from './Button';
import { DataTable, type Column } from './DataTable';
import { NumberInput } from './Field';

describe('Button', () => {
  it('est désactivé et occupé pendant le chargement', () => {
    const onClick = vi.fn();
    render(<Button loading onClick={onClick}>Enregistrer</Button>);
    const btn = screen.getByRole('button', { name: 'Enregistrer' });
    expect(btn).toBeDisabled();
    expect(btn).toHaveAttribute('aria-busy', 'true');
    fireEvent.click(btn);
    expect(onClick).not.toHaveBeenCalled();
  });
});

describe('DataTable', () => {
  type Row = { id: number; name: string; n: number };
  const rows: Row[] = [
    { id: 1, name: 'Beta', n: 2 },
    { id: 2, name: 'alpha', n: 10 },
    { id: 3, name: 'Gamma', n: 1 },
  ];
  const cols: Column<Row>[] = [
    { key: 'name', header: 'Nom', sortValue: (r) => r.name },
    { key: 'n', header: 'N', sortValue: (r) => r.n, align: 'right' },
  ];
  const names = () => screen.getAllByRole('row').slice(1).map((r) => within(r).getAllByRole('cell')[0].textContent);

  it('trie au clic sur l’en-tête et rend les lignes cliquables', () => {
    const onRowClick = vi.fn();
    render(<DataTable columns={cols} rows={rows} rowKey={(r) => r.id} onRowClick={onRowClick} />);
    expect(names()).toEqual(['Beta', 'alpha', 'Gamma']);
    fireEvent.click(screen.getByRole('button', { name: /Nom/ }));
    expect(names()).toEqual(['alpha', 'Beta', 'Gamma']);
    fireEvent.click(screen.getByRole('button', { name: 'N' }));
    fireEvent.click(screen.getByRole('button', { name: 'N' }));
    expect(names()).toEqual(['alpha', 'Beta', 'Gamma']); // n décroissant : 10, 2, 1
    fireEvent.click(screen.getAllByRole('row')[2]);
    expect(onRowClick).toHaveBeenCalledWith(rows[0]);
  });

  it('affiche l’état vide', () => {
    render(<DataTable columns={cols} rows={[]} rowKey={(r) => r.id} />);
    expect(screen.getByText('Aucun élément')).toBeInTheDocument();
  });
});

describe('NumberInput', () => {
  it('accepte la virgule et borne la valeur', () => {
    const onChange = vi.fn();
    render(<NumberInput aria-label="Seuil" value={0.5} onChange={onChange} min={0} max={1} step={0.05} />);
    const input = screen.getByRole('spinbutton', { name: 'Seuil' });
    expect(input).toHaveValue('0,50');
    fireEvent.focus(input);
    fireEvent.change(input, { target: { value: '1,7' } });
    fireEvent.blur(input);
    expect(onChange).toHaveBeenLastCalledWith(1);
  });
});
