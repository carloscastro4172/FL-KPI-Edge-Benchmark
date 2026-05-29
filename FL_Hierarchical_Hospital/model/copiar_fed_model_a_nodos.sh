#!/usr/bin/env bash
set -e

LOCAL_FILE="fed_model.py"
REMOTE_PATH="~/fl_cluster_hospital/model/fed_model.py"

NODES=(
  "nodea@172.23.209.149"
  "nodeb@172.23.209.148"
  "nodoc@172.23.209.153"
  "noded@172.23.209.152"
  "nodee@172.23.209.144"
  "nodef@172.23.209.150"
)

if [ ! -f "$LOCAL_FILE" ]; then
  echo "[ERROR] No existe $LOCAL_FILE en esta carpeta."
  echo "Debes ejecutar este script desde:"
  echo "~/Documents/INVESTIGACIÓN/MODELOS/fl_cluster_hospital/model"
  exit 1
fi

echo "============================================================"
echo "Sobrescribiendo fed_model.py en todos los nodos"
echo "Archivo local: $LOCAL_FILE"
echo "Destino remoto: $REMOTE_PATH"
echo "============================================================"

for NODE in "${NODES[@]}"; do
  echo
  echo ">>> Copiando a $NODE ..."

  scp "$LOCAL_FILE" "$NODE:$REMOTE_PATH"

  echo ">>> Verificando en $NODE ..."
  ssh "$NODE" "
    cd ~/fl_cluster_hospital
    python3 -m py_compile model/fed_model.py
    echo '[OK] fed_model.py sobrescrito y compilado correctamente en:' \$(hostname)
    ls -lh model/fed_model.py
  "
done

echo
echo "============================================================"
echo "[OK] fed_model.py fue sobrescrito en todos los nodos"
echo "============================================================"
