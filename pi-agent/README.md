# eggvision-agent — agent de capture Raspberry Pi 5

Service FastAPI (port 8770) qui pilote la caméra Raspberry Pi Global Shutter (IMX296), enregistre
les lots sur SSD (JPEG qualité 95, file bornée sans blocage, horodatage capteur), diffuse un aperçu
(JPEG / MJPEG), annonce sa présence en mDNS (`_eggvision._tcp.local.`) et sert les fichiers des
lots avec reprise (`Range`) pour la synchronisation vers EggVision Studio. Il propose aussi un banc
de mesure des modèles exportés (`POST /api/bench`).

Documentation complète (installation, SSD, point d'accès Wi-Fi, API, dépannage) : [`docs/pi-agent.md`](../docs/pi-agent.md).

## Démarrage rapide

```bash
# Sur le Pi (Raspberry Pi OS Bookworm 64 bits), depuis une copie du dépôt :
sudo ./pi-agent/install.sh --ssd-device /dev/nvme0n1p1 --with-bench

# En développement (PC), caméra simulée :
uv run eggvision-agent --camera simulated --storage /tmp/eggvision-agent --port 8770
```

## Structure

```
src/eggvision_agent/
  __main__.py        ligne de commande (eggvision-agent / python -m eggvision_agent)
  config.py          configuration (options et variables EGGVISION_AGENT_*)
  api.py             routes HTTP /api/...
  service.py         caméra, aperçu, réglages, lots, fichiers
  camera/            CameraBackend : picamera2_backend.py (Pi, importé seulement sur le Pi), simulated.py
  bench.py           banc de mesure ONNX (option « bench »)
  mdns.py            annonce zeroconf
  system.py          modèle du Pi, température, disque
  testing.py         agent réel en fil d'exécution pour les tests
install.sh           installation sur le Pi (apt, venv --system-site-packages, systemd)
systemd/             unité eggvision-agent.service
tests/               tests en mode simulé (uv run pytest pi-agent/tests)
```

Le moteur d'enregistrement (`CaptureRecorder`) est partagé avec l'application de bureau :
`shared/src/eggvision_shared/capture.py`.
