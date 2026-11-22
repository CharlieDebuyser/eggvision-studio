"""EggVision — agent de capture pour Raspberry Pi 5 (caméra Global Shutter IMX296).

Service léger (FastAPI + uvicorn, port 8770 par défaut) qui pilote la caméra,
enregistre les lots sur le SSD local, diffuse un aperçu et expose une API HTTP
commandée par l'application de bureau. Voir ``docs/pi-agent.md``.
"""

__version__ = "0.1.0"
API_VERSION = 1
DEFAULT_PORT = 8770
SERVICE_TYPE = "_eggvision._tcp.local."
