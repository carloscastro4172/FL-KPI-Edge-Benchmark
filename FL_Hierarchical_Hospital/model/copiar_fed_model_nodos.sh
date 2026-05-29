#!/usr/bin/env bash
set -e

LOCAL_FILE="./fed_model.py"
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
  echo "[ERROR] No existe $LOCAL_FILE. Ejecuta este script desde la carpeta model/"
  exit 1
fi

for NODE in "${NODES[@]}"; do
  echo "Copiando fed_model.py a $NODE ..."
  scp "$LOCAL_FILE" "$NODE:$REMOTE_FILE"
  echo "[OK] Copiado a $NODE"
done

echo "[OK] fed_model.py copiado a todos los nodos"
