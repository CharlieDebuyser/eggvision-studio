import { useEffect, useState, type FormEvent } from 'react';
import { Clock, FolderOpen, FolderPlus, FolderSearch, Sparkles, TriangleAlert } from 'lucide-react';
import { useCreateDemo, useCreateProject, useFsHome, useOpenProject, useRecentProjects } from '@/api/hooks';
import type { DemoSize, Job, RecentProject } from '@/api/types';
import { formatRelative } from '@/lib/format';
import {
  Badge,
  Button,
  EmptyState,
  ErrorState,
  Field,
  FolderPicker,
  LoadingState,
  Page,
  PageHeader,
  Panel,
  SegmentedControl,
  TextInput,
  useToast,
} from '@/components/ui';
import { JobProgress } from '@/components/JobProgress';
import s from './projet.module.css';

const DEMO_JOB_KEY = 'eggvision.demoJob';

/** Nom de dossier valide : non vide, sans séparateur ni caractère interdit. */
export function validateProjectName(name: string): string | null {
  const n = name.trim();
  if (!n) return 'Indiquez un nom.';
  if (/[\\/:*?"<>|]/.test(n)) return 'Caractères interdits : \\ / : * ? " < > |';
  if (n === '.' || n === '..') return 'Nom invalide.';
  if (n.length > 80) return '80 caractères maximum.';
  return null;
}

function PathField({ value, onPick, label, hint }: { value: string; onPick: () => void; label: string; hint?: string }) {
  return (
    <Field label={label} hint={hint}>
      <div className={s.pathRow}>
        <TextInput value={value} readOnly className="u-mono u-grow" placeholder="Aucun dossier choisi" />
        <Button icon={FolderSearch} onClick={onPick}>
          Parcourir…
        </Button>
      </div>
    </Field>
  );
}

function readDemoJob(): string | null {
  try {
    return sessionStorage.getItem(DEMO_JOB_KEY);
  } catch {
    return null;
  }
}
function writeDemoJob(id: string | null) {
  try {
    if (id) sessionStorage.setItem(DEMO_JOB_KEY, id);
    else sessionStorage.removeItem(DEMO_JOB_KEY);
  } catch {
    /* ignore */
  }
}

export function ProjectWelcome() {
  const toast = useToast();
  const home = useFsHome();
  const recent = useRecentProjects();
  const create = useCreateProject();
  const open = useOpenProject();
  const demo = useCreateDemo();

  const [name, setName] = useState('');
  const [nameTouched, setNameTouched] = useState(false);
  const [parent, setParent] = useState('');
  const [picker, setPicker] = useState<null | 'parent' | 'open' | 'demo'>(null);
  const [demoSize, setDemoSize] = useState<DemoSize>('mini');
  const [demoName, setDemoName] = useState('demo-eggvision');
  const [demoParent, setDemoParent] = useState('');
  const [demoJob, setDemoJob] = useState<string | null>(readDemoJob);

  // Dossier parent par défaut : Documents.
  useEffect(() => {
    if (home.data) {
      setParent((p) => p || home.data.documents || home.data.home);
      setDemoParent((p) => p || home.data.documents || home.data.home);
    }
  }, [home.data]);

  const nameError = validateProjectName(name);
  const demoNameError = validateProjectName(demoName);

  const openPath = async (path: string) => {
    try {
      const info = await open.mutateAsync({ path });
      toast.success('Projet ouvert', info.name);
    } catch (e) {
      toast.error(e);
    }
  };

  const onCreate = async (e: FormEvent) => {
    e.preventDefault();
    setNameTouched(true);
    if (nameError || !parent) return;
    try {
      const info = await create.mutateAsync({ parent_dir: parent, name: name.trim() });
      toast.success('Projet créé', info.path);
    } catch (err) {
      toast.error(err);
    }
  };

  const onDemo = async () => {
    if (demoNameError || !demoParent) return;
    try {
      const { job_id } = await demo.mutateAsync({ parent_dir: demoParent, name: demoName.trim(), size: demoSize });
      setDemoJob(job_id);
      writeDemoJob(job_id);
    } catch (err) {
      toast.error(err);
    }
  };

  const onDemoDone = (job: Job) => {
    writeDemoJob(null);
    if (job.status === 'done') {
      const path = typeof job.result?.path === 'string' ? job.result.path : null;
      if (path) void openPath(path);
      else toast.error('Projet de démonstration créé', 'Chemin du projet absent du résultat.');
      setDemoJob(null);
    } else if (job.status === 'failed') {
      toast.error('Échec de la création du projet de démonstration', job.error ?? undefined);
    }
  };

  return (
    <Page>
      <PageHeader
        eyebrow="EggVision Studio"
        title="Commencer"
        subtitle="Un projet regroupe les lots d’images, les pistes labellisées, les modèles entraînés et les simulations d’un même banc de tri."
      />

      <div className={s.welcomeGrid}>
        <div className="u-stack-4">
          <Panel title="Nouveau projet" description="Crée un dossier projet vide.">
            <form className="u-stack-4" onSubmit={onCreate} noValidate>
              <Field label="Nom du projet" required error={nameTouched ? nameError : undefined} hint="Deviendra le nom du dossier.">
                <TextInput value={name} onChange={(e) => setName(e.target.value)} onBlur={() => setNameTouched(true)} placeholder="ex. elevage-nord-2026" autoComplete="off" />
              </Field>
              <PathField label="Dossier parent" value={parent} onPick={() => setPicker('parent')} hint={name.trim() && parent ? `Sera créé dans ${parent}/${name.trim()}` : undefined} />
              <div className="u-row">
                <Button type="submit" variant="primary" icon={FolderPlus} loading={create.isPending} disabled={!parent}>
                  Créer le projet
                </Button>
              </div>
            </form>
          </Panel>

          <div className={s.lowerGrid}>
            <Panel title="Ouvrir un projet" description="Reprendre un projet existant.">
              <Button icon={FolderOpen} onClick={() => setPicker('open')} loading={open.isPending}>
                Parcourir les dossiers…
              </Button>
            </Panel>

            <Panel title="Projet de démonstration" description="Jeu synthétique pour découvrir l’outil.">
              {demoJob ? (
                <JobProgress jobId={demoJob} onDone={onDemoDone} hideTitle />
              ) : (
                <div className="u-stack-4">
                  <Field label="Taille">
                    <SegmentedControl<DemoSize>
                      value={demoSize}
                      onChange={setDemoSize}
                      size="sm"
                      options={[
                        { value: 'mini', label: 'Mini', title: 'Quelques lots, génération rapide' },
                        { value: 'standard', label: 'Standard', title: 'Jeu complet, plus long à générer' },
                      ]}
                    />
                  </Field>
                  <Field label="Nom" error={demoNameError ?? undefined}>
                    <TextInput size="sm" value={demoName} onChange={(e) => setDemoName(e.target.value)} />
                  </Field>
                  <div className={s.demoParent}>
                    <span className="u-xs u-subtle">Dans</span>
                    <span className="u-mono u-xs u-truncate u-grow" title={demoParent}>
                      {demoParent || '—'}
                    </span>
                    <Button size="sm" variant="ghost" onClick={() => setPicker('demo')}>
                      Changer
                    </Button>
                  </div>
                  <Button variant="accent" icon={Sparkles} onClick={onDemo} loading={demo.isPending} disabled={!!demoNameError || !demoParent}>
                    Créer un projet de démonstration
                  </Button>
                </div>
              )}
              {demoJob && (
                <div className={s.demoReset}>
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => {
                      setDemoJob(null);
                      writeDemoJob(null);
                    }}
                  >
                    Masquer
                  </Button>
                </div>
              )}
            </Panel>
          </div>
        </div>

        <Panel title="Projets récents" flush className={s.recent}>
          <RecentList data={recent.data} loading={recent.isLoading} error={recent.error} onRetry={() => recent.refetch()} onOpen={openPath} busyPath={open.isPending ? open.variables?.path : undefined} />
        </Panel>
      </div>

      <FolderPicker
        open={picker === 'parent' || picker === 'demo'}
        mode="folder"
        title="Choisir le dossier parent"
        description="Le dossier du projet sera créé à l’intérieur."
        initialPath={picker === 'demo' ? demoParent : parent}
        onClose={() => setPicker(null)}
        onSelect={(p) => (picker === 'demo' ? setDemoParent(p) : setParent(p))}
      />
      <FolderPicker open={picker === 'open'} mode="project" initialPath={home.data?.documents ?? ''} onClose={() => setPicker(null)} onSelect={(p) => void openPath(p)} />
    </Page>
  );
}

function RecentList({
  data,
  loading,
  error,
  onRetry,
  onOpen,
  busyPath,
}: {
  data?: RecentProject[];
  loading: boolean;
  error: unknown;
  onRetry: () => void;
  onOpen: (path: string) => void;
  busyPath?: string;
}) {
  if (loading) return <LoadingState compact />;
  if (error) return <ErrorState error={error} onRetry={onRetry} compact />;
  if (!data || data.length === 0) return <EmptyState icon={Clock} title="Aucun projet récent" description="Les projets ouverts apparaîtront ici." compact />;
  return (
    <ul className={s.recentList}>
      {data.map((r) => (
        <li key={r.path}>
          <button type="button" className={s.recentItem} disabled={!r.exists || !!busyPath} onClick={() => onOpen(r.path)} title={r.path}>
            <span className={s.recentTop}>
              <span className="u-strong u-truncate u-grow">{r.name}</span>
              {!r.exists ? (
                <Badge tone="danger" icon={TriangleAlert}>
                  Introuvable
                </Badge>
              ) : (
                <span className="u-xs u-subtle">{formatRelative(r.last_opened)}</span>
              )}
            </span>
            <span className="u-mono u-xs u-subtle u-truncate">{r.path}</span>
          </button>
        </li>
      ))}
    </ul>
  );
}
