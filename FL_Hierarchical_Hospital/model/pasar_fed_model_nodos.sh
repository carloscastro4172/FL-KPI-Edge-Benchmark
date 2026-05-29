#!/usr/bin/env bash
set -e

# Este script copia el fed_model.py local corregido
# a todos los nodos en ~/fl_cluster_hospital/model/fed_model.py

LOCAL_FILE="./fed_model.py"
REMOTE_PROJECT="~/fl_cluster_hospital"
REMOTE_FILE="~/fl_cluster_hospital/model/fed_model.py"

NODES=(
  "nodea@172.23.209.149"
  "nodeb@172.23.209.148"
  "nodoc@172.23.209.153"
  "noded@172.23.209.152"
  "nodee@172.23.209.144"
  "nodef@172.23.209.150"
)

if [ ! -f "$LOCAL_FILE" ]; then
  echo "[ERROR] No existe el archivo local: $LOCAL_FILE"
  echo "Ejecuta este script desde la carpeta model/"
  exit 1
fi

echo "============================================================"
echo "Copiando fed_model.py corregido a todos los nodos"
echo "Archivo local: $LOCAL_FILE"
echo "============================================================"

for NODE in "${NODES[@]}"; do
  echo
  echo "------------------------------------------------------------"
  echo "[INFO] Nodo: $NODE"
  echo "------------------------------------------------------------"

  echo "[INFO] Probando conexión SSH..."
  ssh "$NODE" "echo '[OK] Conectado a:' \$(hostname)"

  echo "[INFO] Creando backup remoto..."
  ssh "$NODE" "
    set -e
    mkdir -p ~/fl_cluster_hospital/model
    if [ -f ~/fl_cluster_hospital/model/fed_model.py ]; then
      cp ~/fl_cluster_hospital/model/fed_model.py ~/fl_cluster_hospital/model/fed_model.py.bak_\$(date +%Y%m%d_%H%M%S)
      echo '[OK] Backup creado'
    else
      echo '[WARN] No existía fed_model.py remoto, se copiará nuevo'
    fi
  "

  echo "[INFO] Copiando fed_model.py..."
  scp "$LOCAL_FILE" "$NODE:$REMOTE_FILE"

  echo "[INFO] Verificando sintaxis e imports en nodo..."
  ssh "$NODE" "
    set -e
    cd ~/fl_cluster_hospital

    python3 -m py_compile model/fed_model.py

    python3 - <<'PY'
from model.fed_model import MLP, ModelTrainer, aggregate
print('[OK] Imports correctos: MLP, ModelTrainer, aggregate')
PY

    echo '[OK] Archivo instalado en:' ~/fl_cluster_hospital/model/fed_model.py
    grep -nE 'read_csv|dropna|class MLP|class ModelTrainer|def aggregate' model/fed_model.py | head -30
  "

  echo "[OK] Nodo actualizado: $NODE"
done

echo
echo "============================================================"
echo "[OK] fed_model.py fue copiado a todos los nodos"
echo "============================================================"
