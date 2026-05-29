#!/bin/bash
# ============================================================
#  run_worker.sh — WORKER
#  Uso: bash run_worker.sh <numero_silo> <ip_servidor>
#
#  Ejemplos:
#    bash run_worker.sh 1 172.23.207.115   ← nodeb
#    bash run_worker.sh 2 172.23.207.115   ← nodoc
#    bash run_worker.sh 3 172.23.207.115   ← noded
#    bash run_worker.sh 4 172.23.207.115   ← nodee
#    bash run_worker.sh 5 172.23.207.115   ← nodef
# ============================================================

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$BASE_DIR/venv/bin/activate"
LOGS="$BASE_DIR/logs"
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

SILO_NUM=$1
SERVER_IP=$2

if [ -z "$SILO_NUM" ] || [ -z "$SERVER_IP" ]; then
    echo -e "${RED}Uso: bash run_worker.sh <numero_silo> <ip_servidor>${NC}"
    echo "  bash run_worker.sh 1 172.23.207.115"
    exit 1
fi

CSV_FILE="$BASE_DIR/data/silos/silo_${SILO_NUM}.csv"

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  FL Centralizado — WORKER silo_${SILO_NUM}${NC}"
echo -e "${GREEN}  Servidor: ws://${SERVER_IP}:8765${NC}"
echo -e "${GREEN}============================================${NC}"

if [ ! -f "$VENV" ]; then
    echo -e "${RED}ERROR: venv no encontrado${NC}"
    exit 1
fi
if [ ! -f "$CSV_FILE" ]; then
    echo -e "${RED}ERROR: $CSV_FILE no encontrado${NC}"
    exit 1
fi

source "$VENV"
mkdir -p "$LOGS"

pkill -f "worker_centralized.py" 2>/dev/null
sleep 1

# Verificar servidor
python3 -c "
import socket,sys
try:
    s=socket.socket(); s.settimeout(5); s.connect(('${SERVER_IP}',8765)); s.close()
    print('  Servidor alcanzable ✓')
except Exception as e:
    print(f'  ERROR: {e}'); sys.exit(1)
"
if [ $? -ne 0 ]; then
    echo -e "${RED}Asegúrate de que el servidor ya está corriendo${NC}"
    exit 1
fi

echo -e "${GREEN}Iniciando worker silo_${SILO_NUM}...${NC}"
python "$BASE_DIR/worker_centralized.py" \
    "silo_${SILO_NUM}" \
    "$CSV_FILE" \
    "ws://${SERVER_IP}:8765" 2>&1 | tee "$LOGS/run_worker_silo_${SILO_NUM}.log"

echo -e "${GREEN}✓ Worker silo_${SILO_NUM} finalizado${NC}"
