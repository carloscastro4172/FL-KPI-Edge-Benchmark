#!/bin/bash
# ============================================================
#  run_server.sh — Servidor central (nodea)
#  Uso: bash run_server.sh [FedAvg|FedProx|FedNova]
# ============================================================
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$BASE_DIR/venv/bin/activate"
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

METHOD=${1:-FedAvg}

# Cambiar método en config
sed -i "s/AGGREGATION_METHOD = .*/AGGREGATION_METHOD = '$METHOD'/" "$BASE_DIR/config.py"
ACTUAL=$(grep "AGGREGATION_METHOD" "$BASE_DIR/config.py" | grep -o "'[^']*'" | head -1 | tr -d "'")

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  FL Cluster — SERVIDOR CENTRAL${NC}"
echo -e "${GREEN}  Método: $ACTUAL | Rondas: $(grep '^ROUNDS' $BASE_DIR/config.py | grep -o '[0-9]*')${NC}"
echo -e "${GREEN}============================================${NC}"

[ ! -f "$VENV" ] && echo -e "${RED}ERROR: venv no encontrado. Corre prepare_environment.py primero.${NC}" && exit 1
source "$VENV"

MY_IP=$(hostname -I | awk '{print $1}')
echo -e "${YELLOW}  Mi IP: $MY_IP${NC}"
echo -e "${YELLOW}  Workers deben ejecutar: bash run_worker.sh $MY_IP${NC}"
echo ""

mkdir -p "$BASE_DIR/logs" "$BASE_DIR/models" "$BASE_DIR/received_files"
pkill -f "main.py" 2>/dev/null; sleep 2

python "$BASE_DIR/main.py" "${MY_IP}:8765" "null" 2>&1 | tee "$BASE_DIR/logs/server_${ACTUAL}_$(date +%Y%m%d_%H%M%S).log"

echo -e "${GREEN}Servidor finalizado. Recolectando logs...${NC}"
sleep 2
bash "$BASE_DIR/collect_results.sh"
