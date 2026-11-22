# EggVision Studio

Application de bureau (Windows / macOS) pour le prototype de **tri d'œufs par vision** : prise de vue,
détection et suivi des œufs, labellisation, entraînement, export vers Raspberry Pi 5 et simulation des trappes.

Tout fonctionne **en local, sans connexion Internet** — entraînement compris. Aucune donnée ne quitte la machine.

| Étape | Module | Ce qu'on y fait |
|---|---|---|
| 1 | **Prise de vue** | Piloter la caméra (agent Pi, caméra USB ou source simulée), capturer des lots de statut connu, synchroniser vers le PC |
| 2 | **Détection et suivi** | Calibrer la scène, détecter les œufs, les suivre d'image en image → une *piste* par œuf, 12 vues recadrées |
| 3 | **Labellisation** | Relire les pistes au clavier (pré-label = type du lot), tableau de bord, apprentissage actif |
| 4 | **Entraînement** | Découpage par lot, modèle léger multi-vues, courbes en direct, rapport d'évaluation, choix du seuil, Grad-CAM |
| 5 | **Export Raspberry Pi** | ONNX FP32 + INT8, vérification de parité, paquet de déploiement, banc de mesure sur le Pi |
| 6 | **Simulation** | Rejouer le tri complet avec trappes sur le jeu de test : animation, latences, rapport, CSV, vidéo MP4 |

## Sommaire
- [Installation sur Windows et macOS](#installation-sur-windows-et-macos)
- [Installation de l'agent Raspberry Pi](#installation-de-lagent-raspberry-pi)
- [Tutoriel : du premier lot à la première simulation](#tutoriel--du-premier-lot-à-la-première-simulation)
- [Essayer sans matériel](#essayer-sans-matériel)
- [Organisation du dépôt](#organisation-du-dépôt)
- [Développement et tests](#développement-et-tests)
- [Construire les exécutables](#construire-les-exécutables)
- [Limites connues](#limites-connues)

---

## Installation sur Windows et macOS

### A. Exécutable (utilisateurs)
1. Récupérer l'archive de votre plateforme (voir [Construire les exécutables](#construire-les-exécutables)) :
   - **Windows 10/11 (x64)** : `EggVisionStudio-windows-amd64.zip` → décompresser, lancer `EggVisionStudio\EggVisionStudio.exe`.
   - **macOS 13+ (Apple Silicon ou Intel)** : `EggVisionStudio-macos-<arch>.dmg` → glisser « EggVision Studio » dans Applications.
     L'application n'étant pas signée, au premier lancement : clic droit → *Ouvrir*.
2. L'application ouvre sa propre fenêtre. Si la fenêtre dédiée n'est pas disponible (WebView2 absent sous Windows par
   exemple), elle s'ouvre dans le navigateur par défaut.
3. Autoriser l'accès au **réseau local** (et à la caméra si vous utilisez une caméra USB) quand le système le demande :
   c'est nécessaire pour dialoguer avec l'agent Pi.

L'exécutable embarque Python, PyTorch, ONNX Runtime, l'interface et les poids pré-entraînés : rien à télécharger ensuite.
L'accélération est choisie automatiquement : **CUDA** (Windows + GPU NVIDIA), **MPS** (Mac Apple Silicon), sinon CPU.
L'écran **Diagnostic** indique ce qui a été détecté.

### B. Lancement « développeur » (`uv run`)
Prérequis : [uv](https://docs.astral.sh/uv/) (`brew install uv`, ou `winget install astral-sh.uv`), Node.js ≥ 20, git.

```bash
git clone <dépôt> egg-triage && cd egg-triage
uv sync                                     # Python 3.11 + dépendances (PyTorch compris)
uv run python scripts/fetch_pretrained.py   # une fois, avec Internet : poids pré-entraînés → assets/pretrained/
cd desktop/frontend && npm ci && npm run build && cd ../..
uv run eggvision-studio                     # ouvre l'application
```

Options utiles : `--project <dossier>` (ouvrir un projet au démarrage), `--browser` (navigateur plutôt que fenêtre
dédiée), `--no-window --port 8765` (serveur seul).

> **Windows + GPU NVIDIA** : `uv sync` installe la version CPU de PyTorch. Pour CUDA :
> `uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124` puis relancer.

Les journaux et la liste des projets récents sont dans le dossier de données de l'application (affiché dans
*Diagnostic*). Un projet est un simple dossier : on peut le copier ou le zipper pour le partager.

---

## Installation de l'agent Raspberry Pi

Matériel : Raspberry Pi 5, caméra Raspberry Pi **Global Shutter** (IMX296, CSI), SSD (USB 3 ou NVMe),
Raspberry Pi OS Bookworm 64 bits.

```bash
# Sur le PC : copier le dépôt sur le Pi (au moins shared/, pi-agent/ et docs/)
rsync -a --exclude .venv --exclude node_modules egg-triage/ pi@raspberrypi.local:egg-triage/

# Sur le Pi
cd egg-triage
sudo ./pi-agent/install.sh --ssd-device /dev/nvme0n1p1 --with-bench   # apt, venv, SSD, service systemd
systemctl status eggvision-agent                                      # l'agent écoute sur le port 8770
```

L'agent s'annonce sur le réseau local (mDNS `_eggvision._tcp`) : l'application le trouve avec **Découvrir**, ou
saisissez son adresse (`http://<ip>:8770`).

**Élevage sans réseau** : configurez le Pi en point d'accès Wi-Fi (le PC s'y connecte directement, adresse du Pi
`10.42.0.1`). La procédure complète — installation détaillée, SSD, point d'accès avec `nmcli`, validation sur
matériel, dépannage — est dans **[docs/pi-agent.md](docs/pi-agent.md)**.

Pour essayer l'agent sans Pi ni caméra, sur n'importe quel ordinateur :
```bash
uv run eggvision-agent --camera simulated --storage /tmp/eggvision-agent
```

---

## Tutoriel : du premier lot à la première simulation

### 0. Créer un projet
Écran **Projet** → *Créer un projet* (nom + dossier). Renseignez votre nom de relecteur quand on vous le demande
(il signe l'historique des labels).

### 1. Prise de vue
1. **Prise de vue → Connexion** : choisissez *Agent Pi* (→ *Découvrir* ou adresse), *Caméra USB* ou *Simulée*.
   La barre d'état affiche la cadence réelle, la température et l'espace disque du Pi.
2. **Réglages** : exposition (500 µs par défaut), gain, cadence (30 images/s). Placez une carte grise et cliquez
   **Calibrer la balance des blancs**. Enregistrez un **préréglage** pour retrouver ces réglages.
3. **Mise au point** : réglez l'objectif en maximisant l'**indice de netteté** ; surveillez l'histogramme et l'alerte
   de pixels saturés. Ajustez les repères de couloirs pour vérifier le cadrage (6 œufs par colonne).
4. **Lots** : capturez séparément des lots de statut connu, par exemple :
   - un lot `viable` (uniquement des œufs bons) ;
   - un lot `declasse:<défaut>` par type de défaut (souillure, coulure, fêlure, plume, sang, déformation) ;
   - éventuellement un lot `production` (mélange réel, sans pré-label).

   Renseignez élevage et opérateur, puis **Démarrer** / **Arrêter**. Pendant la capture, les réglages sont verrouillés
   et l'écran affiche durée, images, cadence réelle, **images perdues** et espace disque.
5. À l'arrêt, le lot est **synchronisé** vers le PC (reprise en cas de coupure, vérification par checksum).

> Répétez sur **plusieurs journées et, si possible, plusieurs élevages** : le découpage entraînement / test se fait
> par lot ou par journée, il faut donc plusieurs lots de chaque type. Visez au moins 50 œufs par classe.

### 2. Détection et suivi
1. **Détection → Calibration** (une fois par installation) : choisissez une image de référence, faites proposer les
   couloirs automatiquement puis ajustez les 6 quadrilatères, indiquez le **sens de défilement** et l'**échelle** (outil règle : deux
   points sur une règle visible + longueur réelle en mm).
2. **Réglages** : ajustez si besoin les seuils avec l'aperçu en direct (les valeurs par défaut conviennent à des œufs
   clairs sur rouleaux sombres).
3. **Traitement** : sélectionnez les lots → *Lancer*. Chaque œuf devient une **piste** de 12 vues 224 × 224.
   La colonne « exploitables » doit dépasser 95 %. L'onglet **Pistes** montre les pistes douteuses et pourquoi.

### 3. Labellisation
Chaque piste reçoit le **pré-label** de son lot (un lot `declasse:felure` → « non viable / fêlure »). Relisez :

| Touche | Action |
|---|---|
| `V` | viable |
| `N` | non viable |
| `1`…`6` | non viable + défaut (1 souillure, 2 coulure, 3 fêlure, 4 plume, 5 sang, 6 déformation) |
| `I` | incertain |
| `X` | exclure (piste inexploitable) |
| `Entrée` / `Espace` | confirmer le label affiché |
| `←` `→` | naviguer — `Z` annuler — `R` remettre à relire |

Chaque touche enregistre et passe à la piste suivante. Le mode **Grille** valide d'un coup des dizaines de pistes
viables évidentes. Un clic sur une vue marque « défaut visible ici » (facultatif). Le **Tableau de bord** suit les
effectifs par classe et alerte sous 50 exemples.

### 4. Entraînement
1. **Entraînement → Découpage** : *Nouveau découpage* (par session ou par lot ; vous pouvez réserver un élevage ou une
   journée entière pour le test). Vérifiez l'aperçu : chaque sous-ensemble doit contenir des viables et des défauts.
2. **Nouvel entraînement** : préréglage **rapide** (premier essai, quelques minutes) ou **standard**. Les courbes
   s'affichent en direct ; l'arrêt anticipé s'occupe du reste. Un entraînement interrompu peut être repris.
3. **Rapport** : rappel sur les non viables (global et par défaut), taux de faux rejets, matrice de confusion, ROC/PR.
   Fixez un objectif — « rappel ≥ 95 % » ou « faux rejets ≤ 2 % » — et appliquez le **seuil proposé**.
4. **Explorateur d'erreurs** : faux négatifs et faux positifs avec cartes Grad-CAM ; *Corriger le label* renvoie à la
   labellisation. Une fois un modèle entraîné, la labellisation propose en priorité les pistes où il hésite ou
   contredit le label (**apprentissage actif**).

### 5. Export vers le Raspberry Pi
**Export → Exporter** : produit `model_fp32.onnx`, `model_int8.onnx`, vérifie la **parité PyTorch ↔ ONNX**
(écart ≤ 1e-4) et assemble le paquet zip (`inference_config.json`, modèles, `preprocess.py`, `infer.py`, œufs de
référence). **Tester sur le Pi** envoie le paquet à l'agent, qui mesure la latence par colonne (6 œufs × 12 vues) et
revérifie la parité sur place. Sur le Pi, le paquet s'utilise seul :

```bash
python infer.py --verify            # scores identiques à PyTorch ?
python infer.py --bench 50          # latence par colonne
python infer.py --column ./colonne  # sous-dossiers lane_0 … lane_5 → {"decisions": [0,1,0,0,0,0], "scores": [...]}
```

### 6. Simulation du tri
**Simulation → Nouvelle simulation** : choisissez le modèle exporté, la table virtuelle (cadence 0,5 s/colonne, pas,
distance caméra → trappes, temps d'ouverture), la proportion de défauts (5 % comme en production, 30 % en test de
charge), une graine pour rejouer à l'identique. Seules les pistes du **jeu de test** sont utilisées.
- **Lecteur** : animation vue de dessus, vitesse ×0,25 à ×10, pas à pas ; cliquez un œuf pour voir ses vues, son
  score et sa décision.
- **Résultats** : non viables éjectés / manqués, viables éjectés à tort, décisions en retard par rapport au budget de
  temps (distance ÷ vitesse), ventilation par défaut ; export **CSV** et **vidéo MP4**.
- **Comparaison** : plusieurs simulations (modèles ou seuils différents) côte à côte.

---

## Essayer sans matériel
- Écran **Projet** → **Créer un projet de démonstration** : génère un projet synthétique complet (lots d'images,
  pistes issues du vrai module de détection, pré-labels). « mini » ≈ 360 œufs en moins d'une minute.
- **Prise de vue → source Simulée** : capture de lots à partir d'images synthétiques.
- Le générateur (`shared/src/eggvision_shared/synth.py`) est déterministe et fournit la vérité terrain : c'est le mini
  jeu de données livré avec le projet et utilisé par tous les tests automatisés.

---

## Organisation du dépôt
```
shared/      Code commun PC + Pi : format des lots, prétraitement (source unique), moteur d'inférence, générateur
desktop/     Application : backend FastAPI (src/eggvision), frontend React (frontend/), tests, packaging
pi-agent/    Agent de capture Raspberry Pi 5 (FastAPI, Picamera2), script d'installation, service systemd
docs/        ARCHITECTURE.md, pi-agent.md, api/*.md (référence des API par module)
scripts/     fetch_pretrained.py, build_desktop.py, acceptance_capture.py
```
Un **projet** est un dossier autonome :
```
MonProjet/
  project.db  config.yaml
  lots/<lot_id>/{lot.json, metadata.jsonl, frames/000000.jpg …}
  crops/<track_id>/<vue>.jpg
  models/<model_id>/        simulations/<sim_id>/        logs/
```
Le **prétraitement** (`shared/.../preprocess.py`) est un seul fichier utilisé par l'entraînement, la simulation et le
Pi (il est copié tel quel dans le paquet de déploiement). Voir [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Développement et tests
```bash
uv run pytest                       # toute la suite (≈ 5 min), sans matériel
uv run pytest -m "not slow"         # sans le parcours complet de bout en bout
uv run ruff check .
cd desktop/frontend && npm test && npm run dev    # interface en mode développement (proxy vers le port 8765)
uv run python scripts/acceptance_capture.py --duration 600   # critère n° 1 : 10 min à 30 images/s + synchronisation
```
`desktop/tests/test_e2e.py` rejoue tout le parcours par l'API : images → détection → labels → entraînement →
export ONNX → simulation → CSV et vidéo.

## Construire les exécutables
PyInstaller ne compile pas d'une plateforme vers une autre : lancer la construction **sur chaque plateforme cible**.
```bash
uv sync --group build
uv run python scripts/fetch_pretrained.py
uv run python scripts/build_desktop.py      # → dist/EggVisionStudio-<plateforme>-<arch>.zip (Windows) ou .dmg (macOS)
```
Le flux GitHub Actions `.github/workflows/build.yml` fait la même chose pour Windows x64, macOS Apple Silicon et
macOS Intel (sur étiquette `v*` ou déclenchement manuel) et publie les archives.

## Limites connues
- **Matériel non testé** : le code Picamera2 de l'agent, `install.sh`, le point d'accès Wi-Fi et la caméra USB n'ont
  été vérifiés qu'en simulation. Procédure de validation sur un vrai Pi : [docs/pi-agent.md](docs/pi-agent.md) §6.
- **Images réelles** : les seuils de détection par défaut sont réglés sur le fond de rouleaux du générateur ; ils
  seront à ajuster dans *Détection → Réglages* sur les premières vraies images.
- **INT8** : MobileNetV3 se quantifie mal (l'export le signale par une alerte) ; pour un déploiement INT8, entraîner
  `efficientnet_lite0`. La variante par défaut du paquet reste FP32.
- **Latence sur le Pi** : à mesurer avec *Tester sur le Pi*. Si elle dépasse la cadence (0,5 s par colonne), réduire
  la taille d'entrée ou le nombre de vues, ou passer en INT8 ; la simulation (`latency_scale`) permet d'en évaluer
  l'effet avant d'avoir le matériel.
- **Windows / CUDA** : le code est écrit pour être portable et le flux CI prévoit les tests sous Windows, mais ce flux n'a pas encore été exécuté : le développement et les
  mesures ont été faits sur macOS (Apple Silicon).
