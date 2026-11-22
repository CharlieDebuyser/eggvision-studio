import { useEffect, useMemo, useState } from 'react';
import { ArrowUp, Egg, Folder, FolderOpen, FolderPlus, Home, FileText, CornerDownLeft } from 'lucide-react';
import { cx } from '@/lib/cx';
import { useFsHome, useFsList, useMkdir } from '@/api/hooks';
import { errorMessage } from '@/api/client';
import type { FsEntry } from '@/api/types';
import { Modal } from './Modal';
import { Button, IconButton } from './Button';
import { TextInput } from './Field';
import { Badge } from './Badge';
import { EmptyState, ErrorState, LoadingState } from './Feedback';

export interface FolderPickerProps {
  open: boolean;
  onClose: () => void;
  /** Appelé avec le chemin absolu choisi ; la modale se ferme ensuite. */
  onSelect: (path: string, entry?: FsEntry) => void;
  /**
   * 'folder'  : n'importe quel dossier (ex. dossier parent d'un nouveau projet) ;
   * 'project' : uniquement un dossier projet EggVision (marqué).
   */
  mode?: 'folder' | 'project';
  title?: string;
  description?: string;
  /** Dossier affiché à l'ouverture (défaut : dossier personnel). */
  initialPath?: string;
  confirmLabel?: string;
  /** Bouton « Nouveau dossier » (POST /api/fs/mkdir). Défaut : true en mode 'folder'. */
  allowCreate?: boolean;
}

/** Sélecteur de dossiers basé sur GET /api/fs/list (ne liste que les dossiers). */
export function FolderPicker({
  open,
  onClose,
  onSelect,
  mode = 'folder',
  title = mode === 'project' ? 'Ouvrir un projet' : 'Choisir un dossier',
  description,
  initialPath = '',
  confirmLabel,
  allowCreate = mode === 'folder',
}: FolderPickerProps) {
  const [path, setPath] = useState(initialPath);
  const [selected, setSelected] = useState<FsEntry | null>(null);
  const [typed, setTyped] = useState('');
  const home = useFsHome();
  const list = useFsList(path, open);
  const mkdir = useMkdir();
  const [newName, setNewName] = useState<string | null>(null);
  const [mkdirError, setMkdirError] = useState<string | null>(null);

  useEffect(() => {
    if (open) {
      setPath(initialPath);
      setSelected(null);
    }
  }, [open, initialPath]);

  const current = list.data?.path ?? path;
  useEffect(() => {
    setTyped(current);
  }, [current]);

  const entries = list.data?.entries;
  const sortedEntries = useMemo(
    () => (entries ? [...entries].sort((a, b) => Number(b.is_project) - Number(a.is_project) || a.name.localeCompare(b.name, 'fr')) : []),
    [entries],
  );

  const go = (p: string) => {
    setPath(p);
    setSelected(null);
    setNewName(null);
  };

  const createFolder = async () => {
    const name = (newName ?? '').trim();
    if (!name || /[\\/:*?"<>|]/.test(name)) {
      setMkdirError('Nom de dossier invalide.');
      return;
    }
    try {
      const { path: created } = await mkdir.mutateAsync({ parent_dir: current, name });
      setSelected({ name, path: created, is_dir: true, is_project: false });
      setNewName(null);
      setMkdirError(null);
    } catch (e) {
      setMkdirError(errorMessage(e));
    }
  };

  const crumbs = useMemo(() => {
    if (!current) return [];
    const sep = current.includes('\\') && !current.includes('/') ? '\\' : '/';
    const parts = current.split(sep).filter(Boolean);
    const out: { label: string; path: string }[] = [];
    let acc = '';
    parts.forEach((p, i) => {
      acc = i === 0 && !current.startsWith(sep) ? p : `${acc}${sep}${p}`;
      out.push({ label: p, path: i === 0 && /^[A-Za-z]:$/.test(p) ? `${p}${sep}` : acc });
    });
    return out;
  }, [current]);

  // Valeur retenue : l'entrée sélectionnée, sinon le dossier courant (mode 'folder' uniquement).
  const target: { path: string; entry?: FsEntry } | null = selected
    ? mode === 'project' && !selected.is_project
      ? null
      : { path: selected.path, entry: selected }
    : mode === 'folder' && current
      ? { path: current }
      : null;

  const confirm = () => {
    if (!target) return;
    onSelect(target.path, target.entry);
    onClose();
  };

  const activate = (e: FsEntry) => {
    if (mode === 'project' && e.is_project) {
      onSelect(e.path, e);
      onClose();
    } else {
      go(e.path);
    }
  };

  return (
    <Modal
      open={open}
      onClose={onClose}
      title={title}
      description={description ?? (mode === 'project' ? 'Les dossiers projet EggVision sont signalés par une icône d’œuf.' : undefined)}
      size="lg"
      flush
      footer={
        <>
          <div className="folder-picker__target u-grow">
            {target ? (
              <>
                <span className="u-subtle u-xs">Sélection</span>
                <span className="u-mono u-truncate" title={target.path}>
                  {target.path}
                </span>
              </>
            ) : (
              <span className="u-subtle u-small">{mode === 'project' ? 'Sélectionnez un dossier projet.' : 'Sélectionnez un dossier.'}</span>
            )}
          </div>
          <Button variant="ghost" onClick={onClose}>
            Annuler
          </Button>
          <Button variant="primary" onClick={confirm} disabled={!target}>
            {confirmLabel ?? (mode === 'project' ? 'Ouvrir le projet' : 'Choisir ce dossier')}
          </Button>
        </>
      }
    >
      <div className="folder-picker">
        <aside className="folder-picker__places" aria-label="Emplacements">
          <button type="button" className="folder-picker__place" onClick={() => go(home.data?.home ?? '')}>
            <Home size={14} aria-hidden /> Dossier personnel
          </button>
          {home.data?.documents && (
            <button type="button" className="folder-picker__place" onClick={() => go(home.data!.documents)}>
              <FileText size={14} aria-hidden /> Documents
            </button>
          )}
        </aside>
        <div className="folder-picker__main">
          <div className="folder-picker__bar">
            <IconButton size="sm" icon={ArrowUp} label="Dossier parent" disabled={!list.data?.parent} onClick={() => list.data?.parent && go(list.data.parent)} />
            <form
              className="u-grow"
              onSubmit={(e) => {
                e.preventDefault();
                if (typed.trim()) go(typed.trim());
              }}
            >
              <TextInput
                size="sm"
                value={typed}
                onChange={(e) => setTyped(e.target.value)}
                aria-label="Chemin du dossier"
                className="u-mono"
                trailing={
                  <button type="submit" className="input__inline-btn" aria-label="Aller à ce chemin">
                    <CornerDownLeft size={13} aria-hidden />
                  </button>
                }
              />
            </form>
            {allowCreate && (
              <IconButton
                size="sm"
                icon={FolderPlus}
                label="Nouveau dossier"
                disabled={!current}
                active={newName !== null}
                onClick={() => {
                  setNewName((v) => (v === null ? '' : null));
                  setMkdirError(null);
                }}
              />
            )}
          </div>
          {newName !== null && (
            <form
              className="folder-picker__new"
              onSubmit={(e) => {
                e.preventDefault();
                void createFolder();
              }}
            >
              <FolderPlus size={14} aria-hidden className="u-subtle" />
              <TextInput
                size="sm"
                autoFocus
                value={newName}
                placeholder="Nom du nouveau dossier"
                aria-label="Nom du nouveau dossier"
                invalid={!!mkdirError}
                className="u-grow"
                onChange={(e) => {
                  setNewName(e.target.value);
                  setMkdirError(null);
                }}
              />
              <Button size="sm" type="submit" variant="primary" loading={mkdir.isPending}>
                Créer
              </Button>
              <Button size="sm" variant="ghost" onClick={() => setNewName(null)}>
                Annuler
              </Button>
              {mkdirError && <span className="folder-picker__new-error">{mkdirError}</span>}
            </form>
          )}
          {crumbs.length > 0 && (
            <nav className="folder-picker__crumbs" aria-label="Chemin">
              {crumbs.map((c, i) => (
                <span key={c.path}>
                  <button type="button" onClick={() => go(c.path)} className={cx(i === crumbs.length - 1 && 'is-current')}>
                    {c.label}
                  </button>
                  {i < crumbs.length - 1 && <span className="u-subtle">/</span>}
                </span>
              ))}
            </nav>
          )}
          <div className="folder-picker__list" role="listbox" aria-label="Dossiers">
            {list.isError ? (
              <ErrorState error={list.error} onRetry={() => list.refetch()} compact />
            ) : list.isLoading ? (
              <LoadingState compact />
            ) : sortedEntries.length === 0 ? (
              <EmptyState icon={FolderOpen} title="Aucun sous-dossier" compact />
            ) : (
              sortedEntries.map((e) => {
                const isSel = selected?.path === e.path;
                const Icon = e.is_project ? Egg : Folder;
                return (
                  <button
                    key={e.path}
                    type="button"
                    role="option"
                    aria-selected={isSel}
                    className={cx('folder-picker__item', isSel && 'is-selected', e.is_project && 'is-project', mode === 'project' && !e.is_project && 'is-dim')}
                    onClick={() => setSelected(e)}
                    onDoubleClick={() => activate(e)}
                    onKeyDown={(ev) => {
                      if (ev.key === 'Enter') {
                        ev.preventDefault();
                        activate(e);
                      }
                    }}
                  >
                    <Icon size={15} aria-hidden className="folder-picker__icon" />
                    <span className="u-truncate u-grow">{e.name}</span>
                    {e.is_project && (
                      <Badge tone="accent" variant="soft">
                        Projet
                      </Badge>
                    )}
                  </button>
                );
              })
            )}
          </div>
          <div className="folder-picker__help u-xs u-subtle">Double-clic ou Entrée pour ouvrir un dossier.</div>
        </div>
      </div>
    </Modal>
  );
}
