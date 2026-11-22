#!/usr/bin/env bash
# Installation de l'agent de capture EggVision sur Raspberry Pi 5
# (Raspberry Pi OS Bookworm 64 bits, caméra Raspberry Pi Global Shutter IMX296).
#
# À lancer en root depuis une copie locale du dépôt (dossiers shared/ et pi-agent/) :
#
#   sudo ./pi-agent/install.sh [options]
#
# Options :
#   --storage DIR        dossier des lots (défaut : /mnt/eggvision-ssd/eggvision)
#   --ssd-device DEV     partition du SSD à monter sur /mnt/eggvision-ssd (ex. /dev/sda1) ;
#                        ajoute une entrée /etc/fstab par UUID (nofail). La partition doit
#                        déjà être formatée (ext4 conseillé, voir docs/pi-agent.md).
#   --port N             port HTTP de l'agent (défaut : 8770)
#   --min-free-gb X      arrêt automatique de la capture sous X Go libres (défaut : 2)
#   --wheelhouse DIR     installer les paquets Python depuis ce dossier de wheels
#                        (installation sans Internet ; voir docs/pi-agent.md)
#   --with-bench         installer ONNX Runtime (banc de mesure POST /api/bench)
#   --no-apt             ne pas lancer apt (paquets système déjà installés)
#   --no-start           installer sans démarrer le service
#
# Le script est idempotent : le relancer met à jour l'agent (les lots ne sont pas touchés).

set -euo pipefail

STORAGE="/mnt/eggvision-ssd/eggvision"
SSD_DEVICE=""
SSD_MOUNT="/mnt/eggvision-ssd"
PORT="8770"
MIN_FREE_GB="2"
WHEELHOUSE=""
WITH_BENCH=0
DO_APT=1
DO_START=1
PREFIX="/opt/eggvision-agent"
SERVICE_USER="eggvision"

die() { echo "ERREUR : $*" >&2; exit 1; }
info() { echo "==> $*"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --storage) STORAGE="$2"; shift 2 ;;
    --ssd-device) SSD_DEVICE="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --min-free-gb) MIN_FREE_GB="$2"; shift 2 ;;
    --wheelhouse) WHEELHOUSE="$(realpath "$2")"; shift 2 ;;
    --with-bench) WITH_BENCH=1; shift ;;
    --no-apt) DO_APT=0; shift ;;
    --no-start) DO_START=0; shift ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    *) die "option inconnue : $1" ;;
  esac
done

[[ $EUID -eq 0 ]] || die "lancer avec sudo."
SRC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ -d "$SRC_ROOT/shared" && -d "$SRC_ROOT/pi-agent" ]] || die "copie du dépôt incomplète (shared/ et pi-agent/ attendus dans $SRC_ROOT)."
if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "ATTENTION : architecture $(uname -m) — ce script vise Raspberry Pi OS 64 bits (aarch64)."
fi
if grep -qs "Raspberry Pi" /proc/device-tree/model; then
  info "Carte : $(tr -d '\0' < /proc/device-tree/model)"
fi

# --- 1. Paquets système ----------------------------------------------------
if [[ $DO_APT -eq 1 ]]; then
  info "Paquets système (apt)"
  apt-get update
  # python3-picamera2 : Picamera2 + libcamera (non disponible via pip)
  apt-get install -y --no-install-recommends \
    python3-picamera2 python3-venv python3-pip python3-dev \
    libcap-dev rsync avahi-daemon
fi
python3 -c "import picamera2" 2>/dev/null || echo "ATTENTION : picamera2 introuvable (python3-picamera2) — l'agent démarrera en caméra simulée."

# --- 2. Utilisateur de service --------------------------------------------
if ! id "$SERVICE_USER" >/dev/null 2>&1; then
  info "Création de l'utilisateur système $SERVICE_USER"
  useradd --system --create-home --home-dir "/var/lib/$SERVICE_USER" --shell /usr/sbin/nologin "$SERVICE_USER"
fi
usermod -aG video,render "$SERVICE_USER" 2>/dev/null || usermod -aG video "$SERVICE_USER"

# --- 3. SSD et dossier de stockage -----------------------------------------
if [[ -n "$SSD_DEVICE" ]]; then
  [[ -b "$SSD_DEVICE" ]] || die "périphérique introuvable : $SSD_DEVICE"
  UUID="$(blkid -s UUID -o value "$SSD_DEVICE")" || true
  FSTYPE="$(blkid -s TYPE -o value "$SSD_DEVICE")" || true
  [[ -n "$UUID" && -n "$FSTYPE" ]] || die "$SSD_DEVICE n'est pas formaté (voir docs/pi-agent.md, section SSD)."
  mkdir -p "$SSD_MOUNT"
  if ! grep -qs "UUID=$UUID" /etc/fstab; then
    info "Montage du SSD ($FSTYPE, UUID=$UUID) sur $SSD_MOUNT via /etc/fstab"
    echo "UUID=$UUID  $SSD_MOUNT  $FSTYPE  defaults,noatime,nofail,x-systemd.device-timeout=10s  0  2" >> /etc/fstab
    systemctl daemon-reload
  fi
  mountpoint -q "$SSD_MOUNT" || mount "$SSD_MOUNT"
fi
mkdir -p "$STORAGE"
chown -R "$SERVICE_USER:$SERVICE_USER" "$STORAGE"
info "Stockage des lots : $STORAGE ($(df -h --output=avail "$STORAGE" | tail -1 | tr -d ' ') libres)"

# --- 4. Copie locale des sources et environnement Python ------------------------
info "Copie des sources dans $PREFIX/src"
mkdir -p "$PREFIX/src"
rsync -a --delete --exclude '.venv' --exclude '__pycache__' --exclude 'tests' \
  "$SRC_ROOT/shared" "$SRC_ROOT/pi-agent" "$PREFIX/src/"
mkdir -p "$PREFIX/src/docs"
[[ -f "$SRC_ROOT/docs/pi-agent.md" ]] && cp "$SRC_ROOT/docs/pi-agent.md" "$PREFIX/src/docs/"

if [[ ! -x "$PREFIX/venv/bin/python" ]]; then
  info "Environnement virtuel (avec --system-site-packages pour picamera2)"
  python3 -m venv --system-site-packages "$PREFIX/venv"
fi
PIP=("$PREFIX/venv/bin/pip" install --upgrade)
if [[ -n "$WHEELHOUSE" ]]; then
  PIP+=(--no-index --find-links "$WHEELHOUSE")
fi
# numpy < 2 : picamera2 / simplejpeg (apt) sont compilés contre le numpy 1.x de Bookworm.
CONSTRAINTS="$PREFIX/constraints.txt"
echo "numpy>=1.26,<2" > "$CONSTRAINTS"
EXTRA=""
[[ $WITH_BENCH -eq 1 ]] && EXTRA="[bench]"
info "Installation des paquets Python (eggvision-shared, eggvision-agent$EXTRA)"
"${PIP[@]}" -c "$CONSTRAINTS" "$PREFIX/src/shared" "$PREFIX/src/pi-agent$EXTRA"
"$PREFIX/venv/bin/python" -c "import eggvision_agent, eggvision_shared; print('agent', eggvision_agent.__version__)"

# --- 5. Configuration et service systemd ------------------------------------------
if [[ ! -f /etc/default/eggvision-agent ]]; then
  info "Configuration /etc/default/eggvision-agent"
  cat > /etc/default/eggvision-agent <<EOF
# Agent de capture EggVision — relancer après modification :
#   sudo systemctl restart eggvision-agent
EGGVISION_AGENT_HOST=0.0.0.0
EGGVISION_AGENT_PORT=$PORT
EGGVISION_AGENT_STORAGE=$STORAGE
EGGVISION_AGENT_CAMERA=auto
EGGVISION_AGENT_MIN_FREE_GB=$MIN_FREE_GB
EGGVISION_AGENT_MDNS=1
# EGGVISION_AGENT_QUEUE_SIZE=45
# EGGVISION_AGENT_WRITERS=3
EOF
else
  info "/etc/default/eggvision-agent existant conservé"
fi

info "Service systemd eggvision-agent.service"
UNIT=/etc/systemd/system/eggvision-agent.service
cp "$SRC_ROOT/pi-agent/systemd/eggvision-agent.service" "$UNIT"
if [[ -n "$SSD_DEVICE" ]]; then
  sed -i "s|^# RequiresMountsFor=@STORAGE@|RequiresMountsFor=$SSD_MOUNT|" "$UNIT"
fi
systemctl daemon-reload
systemctl enable eggvision-agent.service
if [[ $DO_START -eq 1 ]]; then
  systemctl restart eggvision-agent.service
  sleep 3
  systemctl --no-pager --lines=5 status eggvision-agent.service || true
  IP="$(hostname -I | awk '{print $1}')"
  info "Agent : http://${IP:-<ip-du-pi>}:$PORT/api/status"
fi
info "Installation terminée. Journal : journalctl -u eggvision-agent -f"
