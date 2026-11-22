# API — Labellisation (`/api/labeling`)

Code : `desktop/src/eggvision/labeling/` (logique) et `desktop/src/eggvision/api/labeling.py` (routes).
Tests : `desktop/tests/test_labeling.py`, `desktop/tests/test_labeling_perf.py` (100 000 pistes).

Le label d'un œuf est porté par sa **piste** (table `labels`, une ligne par piste). Le pré-label
automatique (`prelabel.apply_prelabels`) vient du type du lot ; toute écriture humaine pose
`source="humain"`, `reviewer` = `user.name` de la configuration du projet (« inconnu » si vide) et
ajoute une entrée à `label_history` (instantanés `before` / `after`).

Erreurs : `400` saisie invalide, `404` piste / vue / modèle introuvable, `409` aucun projet ouvert ;
`detail` est un message en français.

## Vocabulaire

| Valeur | Sens |
|---|---|
| verdict | `viable`, `non_viable`, `incertain`, `exclu` ; `aucun` dans les filtres = sans label |
| défaut | `souillure`(1) `coulure`(2) `felure`(3) `plume`(4) `sang`(5) `deformation`(6) ; `aucun` dans les filtres = non viable sans type |
| actions d'historique | `prelabel`, `label`, `confirmation`, `label_masse`, `reouverture`, `marquage_vue`, `annulation` |

**Règles de cohérence** : un défaut implique `non_viable` (un défaut avec un autre verdict ⇒ 400) ;
`viable` / `incertain` / `exclu` vident le défaut ; `non_viable` seul conserve le défaut courant si la
piste était déjà non viable.

## Lecture

| Route | Rôle |
|---|---|
| `GET /meta` | verdicts, défauts (+ touches), actions, `user` (relecteur), `min_examples`, `default_margin` |
| `GET /models` | modèles ayant des prédictions (`n_predictions`, `threshold`) + `default` (dernier `trained` avec prédictions) |
| `GET /lots?limit&offset` | lots (id, nom, type, élevage, journée, nb de pistes), du plus récent au plus ancien |
| `GET /tracks` | recherche paginée (ci-dessous) |
| `GET /tracks/{id}?model_id` | détail : lot, qualité, `prelabel`, `label`, `views[]` (`index`, `defect_visible`, `url`), `prediction` (score, `view_scores`, `predicted`, `priority`, `reason`), `synthetic_truth` (vérité du générateur — contrôle uniquement, **jamais un label**), 30 dernières entrées d'historique |
| `GET /active` | file d'apprentissage actif (ci-dessous) |
| `GET /history` | historique paginé |
| `GET /history/users` | auteurs distincts |
| `GET /dashboard?min_examples=50&model_id` | tableau de bord |

### `GET /tracks` — recherche

Filtres (combinables ; paramètres répétables ou séparés par des virgules) :
`lot`, `lot_type` (`declasse` couvre `declasse:<défaut>`), `verdict`, `defect`, `reviewed` (true/false),
`quality` (`ok`/`douteuse`), `lane`, `source` (`auto`/`humain`), `score_min`/`score_max` (score du
modèle, dans [0, 1]), `reason` (`file` | `contredit` | `hesite`), `track_id`.

Tri `sort` : `naturel` (lot, image de début, couloir — défaut), `recent` (dernière modification),
`priorite` (apprentissage actif ; 400 sans modèle).

Pagination : `limit` (≤ 500, défaut 100) ; `cursor` = `next_cursor` de la page précédente (pagination
par clé, temps constant) ; `anchor=<track_id>` commence à cette piste (incluse) dans le tri courant
(liens profonds) ; `offset` pour un saut direct (plus lent sur de gros volumes). `with_total=false`
évite le comptage (à ne demander qu'à la première page).

`model_id` : modèle dont on joint le score (défaut : modèle par défaut ; `aucun` pour ne rien joindre).
`margin` : demi-largeur de la zone d'hésitation autour du seuil (défaut 0,15).

Réponse :
```json
{"items": [{"id": 1, "lot_id": "…", "lot_type": "viable", "lane": 0, "start_frame": 12, "start_ts": 0.4,
            "n_views": 12, "quality": "ok", "flags": [], "prelabel": {"verdict": "viable", "defect": null},
            "verdict": "viable", "defect": null, "reviewed": true, "source": "humain", "reviewer": "Alice",
            "updated_at": "…", "note": "", "score": 0.12, "priority": 0.0, "reason": null}],
 "next_cursor": "…", "total": 198, "sort": "naturel", "model": {"id": "…", "name": "…", "threshold": 0.5}}
```
(`score`, `priority`, `reason` présents seulement quand un modèle est retenu.)

### `GET /active` — apprentissage actif

Mêmes filtres et pagination que `/tracks`, tri `priorite`, restreint aux pistes où le modèle
**contredit le label** (`reason="contredit"`) ou **hésite** (`reason="hesite"`). Sans modèle : `items`
vide, `model: null` et un `message`.

Score = probabilité de défaut, seuil `t` = `Model.threshold` :
- contradiction : label viable et score ≥ t → `c = (score − t)/(1 − t)` ; label non viable et score < t → `c = (t − score)/t` ;
- hésitation : `|score − t| < marge` → `h = 1 − |score − t| / marge` ;
- priorité = `1 + c` (contradiction) sinon `h` (hésitation) sinon 0, **+ 2 si la piste n'est pas relue** ;
  pistes exclues et pistes sans prédiction : −1 (jamais dans la file).

### `GET /history`

Paramètres : `track_id`, `user`, `action`, `lot`, `limit` (≤ 500), `before_id` (= `next_before_id`
de la page précédente), `with_total`. Entrées : `id, track_id, lot_id, lane, user, at, action,
action_label, before, after` (du plus récent au plus ancien).

### `GET /dashboard`

`totals` (`tracks, labeled, reviewed, unreviewed, progress, usable, usable_reviewed, doubtful,
doubtful_unreviewed, defect_untyped`), `verdicts[]` et `defects[]` (`total`, `reviewed`, y compris
« Sans label » / « Sans type »), `lots[]` (avancement, verdicts et défauts par lot), `sessions[]`
(par élevage × journée), `alerts[]` (`level`, `code`, `message`…), `active` (résumé de la file pour le
modèle par défaut). Codes d'alerte : `classe_sous_representee` (viable, non viable ou un type de
défaut < `min_examples` exemples), `defaut_sans_type`, `lots_non_relus`, `pistes_douteuses`,
`contradictions_modele`.

## Écriture

| Route | Corps | Effet |
|---|---|---|
| `PUT /tracks/{id}/label` | `{verdict?, defect?, reviewed=true, note?}` (champ absent = inchangé) | saisie + relu ; renvoie `{label, history_id, changed}` |
| `POST /tracks/{id}/confirm` | — | confirme le label affiché (pré-label compris) ; 400 si aucun label |
| `POST /tracks/{id}/reopen` | — | remet « à relire » (verdict conservé) |
| `POST /tracks/{id}/views/{index}/defect-visible` | `{visible?}` (absent = inverser) | marque la vue où le défaut est visible (historisé) |
| `POST /bulk` | `{track_ids[≤10 000], verdict?, defect?, reviewed=true, confirm=false}` | mode grille, une transaction ; `confirm=true` confirme le label existant (pistes sans label ignorées → `skipped`) ; renvoie `{updated, skipped, batch}` |
| `POST /undo` | — | annule la dernière action non annulée du relecteur courant (une saisie en masse d'un bloc) ; une piste modifiée depuis n'est pas restaurée (`conflicts`) ; renvoie `{undone, conflicts, track_ids, labels[], action, message}` |

Le nom du relecteur se change via `PUT /api/project/config` `{"user": {"name": "…"}}`.

## Volumétrie

Index complémentaires créés à la volée (`CREATE INDEX IF NOT EXISTS`, au premier appel) :
`tracks(lot_id, start_frame, lane)`, `labels(updated_at)`, `labels(verdict, defect, reviewed)`,
`label_history(user, at)`. Temps médians mesurés sur 100 000 pistes (`pytest -s
desktop/tests/test_labeling_perf.py`) : page de 100 en 3 à 45 ms selon filtres et tri, tableau de bord
≈ 135 ms, écriture d'un label ≈ 2 ms.
