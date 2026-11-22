# Agent de capture Raspberry Pi 5

L'agent (`pi-agent/`, paquet `eggvision_agent`) est un service léger qui tourne sur le Raspberry Pi 5
de la table de tri. Il pilote la caméra **Raspberry Pi Global Shutter** (IMX296, CSI, couleur,
1456 × 1088, jusqu'à 60 images/s), enregistre les lots sur un **SSD** local, diffuse un aperçu et
expose une API HTTP (port **8770**) que l'application de bureau commande. Il s'annonce sur le réseau
local en mDNS (`_eggvision._tcp.local.`).

```
PC (EggVision Studio) ──HTTP──► agent (FastAPI/uvicorn, port 8770)
                                  ├─ caméra : Picamera2 (flux continu, cadence fixe, AE/AWB coupés)
                                  │    main 1456×1088 RGB ──► file bornée ──► N fils JPEG q95 ──► SSD
                                  │    lores 640×478 YUV  ──► aperçu (JPEG / MJPEG à la demande)
                                  └─ lots : lot.json + metadata.jsonl + frames/NNNNNN.jpg
```

Principes : pas d'encodage vidéo (chaque image en JPEG qualité 95) ; producteur / consommateur avec
file bornée — si la file est pleine l'image est abandonnée et **comptée perdue**, la capture ne
bloque jamais ; chaque image porte l'horodatage capteur (`SensorTimestamp`, ns), l'heure système et
`encoder_count: null` ; écriture au fil de l'eau (un lot interrompu reste lisible et passe au statut
`interrompu` au redémarrage) ; arrêt automatique sous un seuil d'espace disque.

## 1. Matériel et système

* Raspberry Pi 5 (4 ou 8 Go), alimentation officielle 27 W (5 V 5 A), refroidisseur actif.
* Caméra Raspberry Pi Global Shutter (nappe 22 broches → port CAM/DISP du Pi 5).
* SSD : NVMe via M.2 HAT+ (recommandé) ou SSD USB 3. Débit d'écriture requis à 30 images/s en pleine
  résolution : ≈ 13 Mo/s (≈ 430 Ko/image mesurés en simulé), soit ≈ 8 Go pour 10 minutes.
* **Raspberry Pi OS Bookworm 64 bits** (Lite suffit), installé avec Raspberry Pi Imager
  (nom d'hôte, ex. `eggpi-01`, utilisateur et SSH activés).

Vérifier la caméra :

```bash
rpicam-hello --list-cameras        # doit lister « imx296 [1456x1088] »
```

Si elle n'apparaît pas : nappe (sens, port), puis dans `/boot/firmware/config.txt` laisser
`camera_auto_detect=1` ou, à défaut, ajouter `dtoverlay=imx296` (`dtoverlay=imx296,cam0` sur le port CAM0), redémarrer.

## 2. Préparer le SSD

Repérer le disque (`lsblk`), par exemple `/dev/nvme0n1` (NVMe) ou `/dev/sda` (USB). **Attention :
efface le disque.**

```bash
sudo wipefs -a /dev/nvme0n1
sudo parted /dev/nvme0n1 --script mklabel gpt mkpart primary ext4 0% 100%
sudo mkfs.ext4 -L eggvision /dev/nvme0n1p1
```

Le script d'installation monte ensuite la partition sur `/mnt/eggvision-ssd` (entrée `/etc/fstab` par
UUID, options `noatime,nofail`) et y crée `/mnt/eggvision-ssd/eggvision/lots/`.
SSD USB : brancher sur un port USB 3 (bleu) ; avec une alimentation 5 A, le Pi 5 fournit 1,6 A aux
ports USB (sinon ajouter `usb_max_current_enable=1` dans `config.txt`).

## 3. Installation

Copier le dépôt (au moins `shared/`, `pi-agent/` et `docs/`) sur le Pi, par exemple :

```bash
rsync -a --exclude .venv --exclude node_modules egg-triage/ pi@eggpi-01.local:egg-triage/
ssh pi@eggpi-01.local
cd egg-triage
sudo ./pi-agent/install.sh --ssd-device /dev/nvme0n1p1 --with-bench
```

Le script (idempotent, relançable pour une mise à jour) :

1. installe les paquets apt (`python3-picamera2`, `python3-venv`, `avahi-daemon`…) ;
2. crée l'utilisateur système `eggvision` (groupes `video`, `render`) ;
3. monte le SSD (si `--ssd-device`) et crée le dossier de stockage ;
4. copie les sources dans `/opt/eggvision-agent/src`, crée le venv `/opt/eggvision-agent/venv`
   **avec `--system-site-packages`** (accès à Picamera2 / libcamera installés par apt) et y installe
   `eggvision-shared` et `eggvision-agent` depuis cette copie locale (`numpy<2` imposé : les modules
   apt de Bookworm sont compilés contre numpy 1.x) ;
5. écrit `/etc/default/eggvision-agent` (conservé s'il existe) ;
6. installe et active `eggvision-agent.service` (démarrage au boot, redémarrage automatique,
   arrêt propre : le lot en cours est clos, statut `interrompu`).

Options : `--storage DIR`, `--port N`, `--min-free-gb X`, `--wheelhouse DIR`, `--with-bench`
(ONNX Runtime pour le banc de mesure), `--no-apt`, `--no-start`.

**Installation sans Internet** : seul l'apt initial demande un accès réseau. Pour pip, préparer un
dossier de wheels sur une machine connectée (même version de Python que le Pi : 3.11, aarch64) :

```bash
pip download -d wheels --platform manylinux2014_aarch64 --python-version 3.11 --only-binary=:all: \
  "numpy>=1.26,<2" opencv-python-headless "pydantic>=2.6" "fastapi>=0.111" "uvicorn>=0.30" \
  "psutil>=5.9" "zeroconf>=0.132" "python-multipart>=0.0.9" onnxruntime
pip download -d wheels hatchling          # outil de construction des paquets du projet
```

puis `sudo ./pi-agent/install.sh --wheelhouse wheels ...` (pip n'interroge alors aucun index).

Vérifier :

```bash
systemctl status eggvision-agent
journalctl -u eggvision-agent -f
curl http://localhost:8770/api/status
```

### Configuration (`/etc/default/eggvision-agent`)

| Variable | Défaut | Rôle |
|---|---|---|
| `EGGVISION_AGENT_HOST` / `_PORT` | `0.0.0.0` / `8770` | écoute HTTP |
| `EGGVISION_AGENT_STORAGE` | `/mnt/eggvision-ssd/eggvision` | dossier des lots (`lots/<id>/`) et des réglages (`agent_settings.json`) |
| `EGGVISION_AGENT_CAMERA` | `auto` | `auto` (Picamera2 si une caméra est détectée, sinon simulée), `picamera2`, `simulated` |
| `EGGVISION_AGENT_MIN_FREE_GB` | `2` | seuil d'arrêt automatique (et de refus de démarrage) |
| `EGGVISION_AGENT_MDNS` | `1` | annonce mDNS |
| `EGGVISION_AGENT_QUEUE_SIZE` | `45` | capacité de la file (images ; ≈ 215 Mo en pleine résolution) |
| `EGGVISION_AGENT_WRITERS` | `3` | fils d'encodage JPEG |
| `EGGVISION_AGENT_SIM_SCALE` / `_SIM_FOLDER` / `_SIM_CYCLE` | `1` / — / `1` | caméra simulée |

Les mêmes réglages existent en ligne de commande : `eggvision-agent --help`
(`--host`, `--port`, `--storage`, `--camera`, `--no-mdns`, `--min-free-gb`, `--queue-size`,
`--writers`, `--sim-scale`, `--sim-folder`, `--sim-cycle`, `--log-level`).
Sur un PC de développement : `uv run eggvision-agent --camera simulated --storage /tmp/agent`.

## 4. Point d'accès Wi-Fi (élevage sans réseau)

Le Pi crée son propre réseau Wi-Fi ; le PC s'y connecte et joint l'agent à une adresse fixe.
Sous Bookworm, le réseau est géré par **NetworkManager** (`nmcli`). Avec `ipv4.method shared`,
NetworkManager attribue l'IP fixe au Pi et sert le **DHCP** (dnsmasq intégré) aux clients.

```bash
# pays Wi-Fi (obligatoire pour activer la radio)
sudo raspi-config nonint do_wifi_country FR
sudo rfkill unblock wifi

# point d'accès WPA2, IP fixe 10.42.0.1/24, DHCP pour les clients
sudo nmcli connection add type wifi ifname wlan0 con-name eggvision-ap autoconnect yes \
  ssid "EggVision-01" mode ap \
  802-11-wireless.band bg 802-11-wireless.channel 6 \
  wifi-sec.key-mgmt wpa-psk wifi-sec.proto rsn wifi-sec.pairwise ccmp wifi-sec.group ccmp \
  wifi-sec.psk "ChangerCeMotDePasse" \
  ipv4.method shared ipv4.addresses 10.42.0.1/24 \
  ipv6.method disabled
sudo nmcli connection modify eggvision-ap connection.autoconnect-priority 100
sudo nmcli connection up eggvision-ap
```

* Le PC se connecte au réseau `EggVision-01`, reçoit une adresse 10.42.0.x ; dans l'application :
  **Prise de vue → Agent Pi → Découvrir** (mDNS) ou saisir `10.42.0.1` (port 8770 implicite).
* 5 GHz (plus de débit pour la synchronisation, portée moindre) :
  `sudo nmcli connection modify eggvision-ap 802-11-wireless.band a 802-11-wireless.channel 36`.
* Revenir au Wi-Fi de l'exploitation : `sudo nmcli connection down eggvision-ap` (puis
  `nmcli connection up <réseau>`), ou supprimer la connexion : `sudo nmcli connection delete eggvision-ap`.
* Liaison filaire directe PC ↔ Pi (plus rapide pour synchroniser de gros lots) :
  `sudo nmcli connection add type ethernet ifname eth0 con-name eggvision-eth ipv4.method shared ipv4.addresses 10.43.0.1/24`
  puis agent sur `10.43.0.1`.
* Horloge : sans Internet, l'heure du Pi vient de son horloge RTC (pile RTC conseillée sur le
  connecteur BAT du Pi 5). L'application affiche l'écart d'horloge (`clock_offset_s`) ; les dates des
  lots (`created_at`) sont fournies par le PC et les intervalles entre images viennent de l'horloge
  capteur, insensibles à ce décalage.

## 5. API HTTP de l'agent

Toutes les réponses sont en JSON sauf mention ; erreurs `{"detail": "message en français"}`.
Codes : `400` saisie invalide, `404` lot / fichier introuvable, `409` conflit (capture en cours :
réglages verrouillés ; lot en cours de capture ; banc en cours), `413` paquet trop gros, `416` plage
invalide, `422` valeur hors limites, `501` ONNX Runtime absent, `503` caméra non prête, `507` espace
disque sous le seuil.

| Route | Rôle |
|---|---|
| `GET /api/status` | `{agent: "eggvision-agent", version, api_version: 1, hostname, device: {model, cpu_count, temperature_c}, time, uptime_s, storage: {path, total_bytes, used_bytes, free_bytes, min_free_bytes}, camera: {backend, model, sensor_width, sensor_height, output_width, output_height, ready, error, actual_fps, frames_seen, settings, awb_running}, capturing, capture, last_capture, bench_running, interrupted_at_start}` |
| `GET /api/camera/settings` | réglages (`exposure_us`, `analogue_gain`, `red_gain`, `blue_gain`, `fps`, `roi`) |
| `PUT /api/camera/settings` | réglages complets ou partiels ; `409` pendant une capture ; `422` (ex. exposition > période d'image) |
| `POST /api/camera/awb` | `{duration_s: 2}` : AWB activée ~2 s (carte grise), gains lus puis figés → `{red_gain, blue_gain, settings}` |
| `GET /api/preview.jpg?width&quality` | dernière image basse résolution (flux `lores`), en-tête `X-Frame-Seq` |
| `GET /api/preview.mjpeg?fps=10&width&quality` | flux MJPEG (`multipart/x-mixed-replace; boundary=frame`) |
| `GET /api/capture` | lot en cours : `{active, lot_id, …statistiques}` ou `{active: false, last}` |
| `POST /api/lots` | démarre un lot ; corps : `{lot_id?, name, type, farm, operator, comment, created_at?, camera?, min_free_bytes?, source?}` (manifeste complet fourni par le PC ; `camera` appliqué puis figé) → `{lot_id, manifest, capture}` |
| `POST /api/lots/stop`, `POST /api/lots/{id}/stop` | arrête (vide la file, clôt le lot `termine`) → `{lot_id, manifest, capture}` |
| `GET /api/lots?limit&offset` | `{total, items: [{lot_id, name, type, farm, operator, status, created_at, started_at, ended_at, width, height, stats, capturing}]}` |
| `GET /api/lots/{id}` | `{manifest, capture}` |
| `GET /api/lots/{id}/files` | manifeste de synchronisation `{lot_id, status, n_files, total_bytes, files: [{path, size, sha256}]}` (`409` pendant la capture) |
| `GET /api/lots/{id}/files/{chemin}` | téléchargement ; en-tête `Range: bytes=début-[fin]` → `206` + `Content-Range` (reprise) |
| `DELETE /api/lots/{id}` | supprime le lot du SSD (`409` s'il est en capture) |
| `POST /api/bench` | banc de mesure d'un paquet de déploiement (ci-dessous) |

Statistiques de capture (`capture`, `/api/capture`) : `frames_received`, `frames_written`,
`frames_queued`, `frames_dropped` (= file pleine `frames_dropped_queue` + trous de séquence capteur
`frames_dropped_upstream`), `drop_ratio`, `measured_fps`, `instant_fps`, `bytes_written`,
`queue_fill`, `queue_max_fill`, `disk_free_bytes`, `elapsed_s`, `duration_s`, `stop_reason`
(`user`, `disk_low`, `write_error`, `shutdown`).

**`POST /api/bench`** (multipart) : fichier `package` (zip du paquet de déploiement, à la racine ou
dans un sous-dossier), champs `variant` (`fp32` | `int8`, défaut `fp32`), `n_columns` (défaut 50),
`threads` (facultatif). L'agent dézippe dans un dossier temporaire, charge
`eggvision_shared.runtime.EggClassifier`, exécute `benchmark(n_columns)` et `verify_reference(dossier)`.
Réponse : le dictionnaire de `benchmark()` + `parity` (résultat de `verify_reference` ou `null`) +
`device: {model, cpu_count, temperature_c}` + `onnxruntime` (version). `409` pendant une capture
(et inversement, pas de capture pendant un banc), `501` si ONNX Runtime n'est pas installé
(`install.sh --with-bench`), `400` zip invalide, `422` variante absente / paramètres hors limites.

## 6. Validation sur le matériel

Le code Picamera2 (`camera/picamera2_backend.py`) suit l'API documentée de Picamera2 mais **n'a pas
pu être exécuté sur un vrai Pi pendant le développement**. À la mise en service :

1. `rpicam-hello --list-cameras` liste l'IMX296 ;
2. `sudo systemctl stop eggvision-agent` puis, en tant qu'`eggvision`, lancer au premier plan :
   `sudo -u eggvision /opt/eggvision-agent/venv/bin/eggvision-agent --camera picamera2 --log-level debug`
   — le journal doit afficher « Caméra configurée : main (1456, 1088), lores (640, 478), 30.0 images/s » ;
3. `curl localhost:8770/api/status` : `camera.backend = picamera2`, `actual_fps` ≈ 30,
   `camera.metadata.ExposureTime` ≈ 500, `AnalogueGain` ≈ 1 ;
4. aperçu : `http://<ip>:8770/api/preview.jpg` dans un navigateur (couleurs correctes : ordre BGR) ;
5. ROI : `PUT /api/camera/settings {"roi": [0, 200, 1456, 600]}` → `output_height` = 600 ;
   calibration : carte grise, `POST /api/camera/awb` → gains plausibles (rouge ≈ 1,5–3, bleu ≈ 1,2–2,5) ;
6. critère d'acceptation depuis le PC (Wi-Fi ou Ethernet) :
   `uv run python scripts/acceptance_capture.py --agent-url http://<ip>:8770 --duration 600`
   → pertes < 0,1 %, cadence 30,0, synchronisation vérifiée ;
7. surveiller `vcgencmd measure_temp` et `vcgencmd get_throttled` (doit rester `0x0`).

## 7. Dépannage

| Symptôme | Piste |
|---|---|
| `camera.backend = simulated` sur le Pi | Picamera2 non importable ou caméra non détectée : `rpicam-hello --list-cameras`, `python3 -c "import picamera2"`, journal du service. Forcer `EGGVISION_AGENT_CAMERA=picamera2` pour obtenir l'erreur explicite |
| `numpy.dtype size changed` / erreur d'import de `simplejpeg` | numpy 2 installé dans le venv : relancer `install.sh` (contrainte `numpy<2`) |
| « Device or resource busy » | un autre programme utilise la caméra (`rpicam-*`, autre instance) : l'arrêter |
| Images perdues (`frames_dropped_queue` > 0) | SSD trop lent ou processeur bridé : `vcgencmd get_throttled`, test d'écriture `dd if=/dev/zero of=/mnt/eggvision-ssd/test bs=1M count=2000 oflag=direct`, refroidissement ; augmenter `EGGVISION_AGENT_WRITERS` (4) ou `QUEUE_SIZE` |
| `frames_dropped_upstream` > 0 | trous dans les horodatages capteur : boucle d'acquisition en retard (charge processeur), alimentation insuffisante |
| Capture arrêtée seule, `stop_reason = disk_low` | espace sous `MIN_FREE_GB` : synchroniser puis supprimer des lots depuis l'application |
| Service ne démarre pas après reboot | SSD non monté (`RequiresMountsFor`) : `lsblk`, `sudo mount -a`, `journalctl -b -u eggvision-agent` |
| Agent introuvable par « Découvrir » | multicast filtré (pare-feu Windows en profil « Public », VPN) : saisir l'adresse IP ; vérifier `avahi-browse -rt _eggvision._tcp` sur le Pi |
| Synchronisation lente | Wi-Fi 2,4 GHz : préférer 5 GHz ou un câble Ethernet ; la synchronisation reprend où elle s'est arrêtée |
| Lot `interrompu` | coupure de courant ou arrêt du service pendant la capture : les images écrites restent valides et synchronisables |
