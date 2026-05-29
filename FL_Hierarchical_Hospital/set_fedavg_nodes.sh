#!/usr/bin/env bash

# Cambia AGGREGATION_METHOD = 'FedAvg' en todos los nodos

NODES=(
  "nodea@172.23.209.149"
  "nodeb@172.23.209.148"
  "nodoc@172.23.209.153"
  "noded@172.23.209.152"
  "nodee@172.23.209.144"
  "nodef@172.23.209.150"
)

REMOTE_CONFIG="~/fl_cluster_hospital/config.py"

echo "============================================================"
echo "Configurando AGGREGATION_METHOD = 'FedAvg' en todos los nodos"
echo "============================================================"

for NODE in "${NODES[@]}"; do
  echo
  echo "------------------------------------------------------------"
  echo "[INFO] Nodo: $NODE"
  echo "------------------------------------------------------------"

  ssh "$NODE" "
    set -e

    CONFIG_PATH=\$HOME/fl_cluster_hospital/config.py

    if [ ! -f \"\$CONFIG_PATH\" ]; then
      echo '[ERROR] No existe:' \"\$CONFIG_PATH\"
      exit 1
    fi

    cp \"\$CONFIG_PATH\" \"\$CONFIG_PATH.bak_fedavg_\$(date +%Y%m%d_%H%M%S)\"

    if grep -q '^AGGREGATION_METHOD' \"\$CONFIG_PATH\"; then
      sed -i \"s/^AGGREGATION_METHOD = .*/AGGREGATION_METHOD = 'FedAvg'/\" \"\$CONFIG_PATH\"
    else
      echo \"AGGREGATION_METHOD = 'FedAvg'\" >> \"\$CONFIG_PATH\"
    fi

    echo '[OK] Config actualizado en:' \$(hostname)
    grep -n '^AGGREGATION_METHOD' \"\$CONFIG_PATH\"
  "

  if [ $? -eq 0 ]; then
    echo "[OK] Nodo actualizado correctamente: $NODE"
  else
    echo "[ERROR] Falló el nodo: $NODE"
  fi
done

echo
echo "================================================------------"
echo "[OK] Proceso terminado"
echo "================================================------------"
