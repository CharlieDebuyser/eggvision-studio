# Écrire une page de module — EggVision Studio

Socle : Vite + React 18 + TypeScript strict, HashRouter, React Query, CSS maison (jetons). Interface **en français**, 100 % hors-ligne (aucune ressource réseau externe).

## Lancer

```bash
npm install
npm run dev:mock                     # API simulée (mock/mockApi.ts), aucun backend requis
npm run dev                          # proxy /api → http://127.0.0.1:8765
# Backend sur un autre port + dev server sur un port libre :
VITE_API_TARGET=http://127.0.0.1:8790 npx vite --port 5182 --strictPort
# Backend réel (depuis la racine du dépôt) :
EGGVISION_DATA_DIR=/tmp/egv-front uv run eggvision-studio --no-window --port 8790
npm run build    # tsc --noEmit puis vite build → ../src/eggvision/web
npm test         # vitest
```

## Ajouter un module

Le module `<m>` ∈ `capture | detection | labellisation | entrainement | export | simulation` vit dans `src/pages/<m>/`. `index.tsx` **exporte par défaut** le composant de page (chargé en lazy). Il est monté sur `/<m>/*` derrière la garde « projet ouvert » (redirection vers `/` sinon) : inutile de revérifier le projet. Métadonnées (libellé, icône, étape) : `src/app/modules.ts` — ne pas modifier sans raison.

Sous-routes : déclarer des `<Routes>` **relatives** dans le module :

```tsx
import { Route, Routes } from 'react-router-dom';
import { Page, PageHeader, RouteTabs } from '@/components/ui';

export default function CapturePage() {
  return (
    <Page>
      <PageHeader eyebrow="Étape 1" title="Prise de vue" subtitle="…" actions={<Button variant="primary">…</Button>}>
        <RouteTabs items={[{ to: '/capture', label: 'Lots', end: true }, { to: '/capture/reglages', label: 'Réglages' }]} />
      </PageHeader>
      <Routes>
        <Route index element={<LotList />} />
        <Route path="lots/:lotId" element={<LotDetail />} />
        <Route path="reglages" element={<Settings />} />
      </Routes>
    </Page>
  );
}
```

Préférer des liens **absolus** (`/capture/lots/3`) : la résolution relative sous une route `/*` est ambiguë dans react-router v6.

## Appels API

```ts
import { api, ApiError, mediaUrl, errorMessage } from '@/api/client';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';

const lots = useQuery({ queryKey: ['lots'], queryFn: ({ signal }) => api.get<Lot[]>('/lots', undefined, { signal }) });
await api.post<JobRef>('/detection/run', { lot_id: 3 });          // chemin relatif à /api
api.get<Job[]>('/jobs', undefined, { query: { kind: 'training' } }); // null/undefined/'' ignorés
<img src={mediaUrl.frame(lotId, index, 320)} />  <img src={mediaUrl.crop(trackId, view)} />
```

- `api.get/post/put/patch/del<T>(path, body?, { signal, query, headers })` ; toute réponse non-2xx lève `ApiError { status, detail }` (`detail` = message français du backend ; `status 0` = serveur injoignable).
- Après une action qui change le projet : `queryClient.invalidateQueries({ queryKey: queryKeys.project })` (compteurs de l'accueil).
- Types du contrat : `src/api/types.ts`. Ajouter les types de son module dans `src/pages/<m>/types.ts` (ou `api.ts`), pas dans les fichiers partagés.
- Mock (`npm run dev:mock`) : créer `mock/<m>.ts` → `import { HttpError, type RegisterRoute } from './mockApi'; export default function register(route: RegisterRoute) { route('GET', '/api/lots/:id', ({ params, url, body }) => { … throw new HttpError(404, 'Lot introuvable.'); }); }`, puis l'importer dans `mock/mockApi.ts` et l'ajouter au tableau `MODULE_MOCKS`. Le handler renvoie l'objet JSON ; `{ __raw: string }` + `res.setHeader('Content-Type', …)` pour un contenu brut.

### Hooks fournis (`@/api/hooks`)

| Hook | Rôle |
|---|---|
| `useProject()` | `ProjectInfo` courant (`data.open`, `stats`) |
| `useRecentProjects()`, `useCreateProject()`, `useOpenProject()`, `useCloseProject()`, `useCreateDemo()` | gestion de projet |
| `useProjectConfig()`, `useUpdateProjectConfig()` | config.yaml (PUT = fusion partielle) |
| `useFsHome()`, `useFsList(path)`, `useMkdir()` | système de fichiers (dossiers uniquement) |
| `useJobs(filter?)` | liste ; polling 1,5 s si une tâche est active, sinon 5 s |
| `useJob(id)` | polling 1 s tant que `queued`/`running`, puis arrêt |
| `useCancelJob()` | `mutate(id)` |
| `useJobLog(id, { follow? })` | `{ text, loading, error, eof, reload }`, lecture incrémentale par offset |
| `useDiagnostics()`, `useDiagnosticsLog(lines)` | diagnostic |
| `isJobActive(status)`, `queryKeys` | utilitaires |

Lancer une tâche longue puis la suivre :

```tsx
const [jobId, setJobId] = useState<string | null>(null);
<JobProgress jobId={jobId} onDone={(job) => job.status === 'done' && qc.invalidateQueries({ queryKey: ['lots'] })} />
```

## Composants (`@/components/ui`)

| Composant | Props principales |
|---|---|
| `Page` | `wide?` — conteneur de page (marges, largeur max) |
| `PageHeader` | `title, subtitle?, actions?, eyebrow?, breadcrumbs?: {label,to?}[], children?` (onglets) ; met à jour le titre de fenêtre |
| `Button` | `variant: primary\|secondary\|ghost\|danger\|accent`, `size: sm\|md\|lg`, `loading?, icon?, iconRight?, block?` ; `type="button"` par défaut |
| `IconButton` | `icon, label` (obligatoire : aria-label + infobulle), `variant?, size?, active?, loading?` |
| `ButtonGroup` | boutons accolés |
| `Card` | `flush?, sunken?` — `Panel` : `title?, description?, actions?, footer?, flush?` — `KeyValue` : `items: {label, value, mono?}[]` |
| `Tabs` | `items: {key,label,count?,disabled?}[], value, onChange` — `RouteTabs` : `items: {to,label,count?,end?}[]` |
| `Modal` | `open, onClose, title, description?, footer?, size: sm\|md\|lg\|xl, flush?, dismissible?` (focus piégé, Échap) |
| `ConfirmDialog` | `open, title, message?, confirmLabel?, variant: primary\|danger, onConfirm (peut renvoyer une promesse), onCancel` |
| `Field` | `label?, hint?, error?, required?, inline?, labelAside?` — relie automatiquement id/aria au contrôle enfant ; `Label` |
| `TextInput` / `TextArea` | props natives + `size?, invalid?, leading?, trailing?` |
| `NumberInput` | `value: number\|null, onChange, min?, max?, step?, unit?, decimals?, stepper?, allowEmpty?` (virgule acceptée, bornage au blur/Entrée, ↑↓) |
| `Select<V>` | `value, onChange(v), options: {value,label,disabled?}[], placeholder?, size?` |
| `Slider` | `value, onChange, min, max, step, unit?, format?(v), hideValue?` |
| `Toggle` / `Checkbox` | `checked, onChange(bool), label?, description?` (+ `indeterminate?` pour Checkbox) |
| `SegmentedControl<V>` | `value, onChange, options: {value,label,title?,disabled?}[], size?` |
| `Badge` | `tone: neutral\|accent\|success\|warning\|danger\|info, variant: soft\|solid\|outline, dot?, icon?` |
| `StatusBadge` / `VerdictBadge` / `DefectBadge` | `status` (job) / `verdict` / `defect` |
| `ProgressBar` | `value: 0..1\|null` (null = indéterminé), `tone?, size: xs\|sm\|md, showValue?` |
| `Spinner`, `LoadingState`, `Skeleton` | chargement |
| `EmptyState` | `icon?, title, description?, actions?, compact?` |
| `ErrorState` | `error?` (ApiError → detail), `title?, onRetry?, compact?` — `Callout` : `tone, title?, icon?, children` |
| `Stat` | `label, value, hint?, icon?, onClick?` |
| `DataTable<T>` | `columns: {key, header, render?, sortValue?, align?, width?, truncate?}[], rows, rowKey, onRowClick?, selectedKey?, loading?, error?, empty?, defaultSort?, stickyHeader?, dense?, minWidth?` |
| `Pagination` | `page` (base 0), `pageSize, total, onPageChange` |
| `Tooltip` | `content, side?, children` (un seul élément) |
| `Kbd` | `<Kbd>1</Kbd>` ou `<Kbd keys={comboLabel('mod+s')} />` |
| `FolderPicker` | `open, onClose, onSelect(path, entry?), mode: folder\|project, title?, initialPath?, allowCreate?` |
| `useToast()` | `.success(t, d?)`, `.info`, `.warning`, `.error(errOuTitre, d?)`, `.show({title, description, tone, duration, action})` |

Hors `ui/` : `JobProgress` (`@/components/JobProgress` : `jobId, onDone?, hideTitle?, hideCancel?, compact?`), `LogView` (`@/components/LogView` : `text, follow?, height?`).

## Bibliothèque (`@/lib`)

- `format.ts` : `formatNumber(v, dec?)`, `formatPercent(ratio, dec?)`, `formatDuration(s)`, `durationBetween(isoA, isoB?)`, `formatBytes`, `formatDate`, `formatDateTime`, `formatTime`, `formatRelative`, `plural(n, mot)`. `null` → « — ».
- `hotkeys.ts` : `useHotkeys({ '1': fn, 'ArrowRight': next, 'mod+s': save, 'shift+Enter': ok }, { enabled })` — ignoré dans les champs de saisie (sauf Échap et combinaisons mod/ctrl) ; chiffres reconnus en AZERTY (`e.code`). `comboLabel(combo)`.
- `domain.ts` : `DEFECTS` (`key, label, hotkey` 1–6), `defectLabel`, `defectByHotkey`, `VERDICTS` (`key, label, tone, color` = var CSS), `verdictLabel`, `LOT_TYPES`, `parseLotType`, `lotTypeLabel`, `JOB_KIND_LABELS`, `JOB_STATUS_META`.
- `theme.ts` : `useTheme()`, `cssVar('--chart-1')` (couleurs pour recharts) ; `cx(...)` dans `cx.ts`.

## Styles

- Jetons dans `src/styles/tokens.css` (`--bg, --surface, --surface-2/3, --border(-strong), --text(-muted/-subtle), --accent(-strong/-soft), --primary, --success/-soft, --warning, --danger, --info, --verdict-*, --chart-1..6, --sp-1..12, --r-*, --fs-*, --fw-*, --font-mono`). **Jamais de couleur en dur** : le thème sombre redéfinit les jetons.
- Styles de module : **CSS modules** `src/pages/<m>/<m>.module.css` (`import s from './capture.module.css'`), voir `pages/projet/projet.module.css`. Ne pas ajouter de classes globales.
- Utilitaires globaux : `u-row, u-col, u-stack-4/6, u-grid-2/3/auto, u-grow, u-spacer, u-mono, u-num, u-muted, u-subtle, u-small, u-xs, u-strong, u-truncate, u-break, u-sr-only`.
- Accent ambre avec parcimonie (sélection, étape active, action principale ponctuelle). Boutons principaux = `primary` (sombre).
- Cible ≥ 1280×800, utilisable à 1100 px : prévoir `@media (max-width: 1240px)` pour passer en une colonne.
