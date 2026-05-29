#!/bin/bash
# ============================================================
#  run_worker.sh — Nodo worker (nodeb-nodef)
#  Uso: bash run_worker.sh <silo_numero> <ip_servidor>
#  Ej:  bash run_worker.sh 1 172.23.209.149
# ============================================================
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$BASE_DIR/venv/bin/activate"
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

SILO_NUM=${1}
SERVER_IP=${2}

if [ -z "$SILO_NUM" ] || [ -z "$SERVER_IP" ]; then
    echo -e "${RED}Uso: bash run_worker.sh <numero_silo> <ip_servidor>${NC}"
    echo "  Ej: bash run_worker.sh 1 172.23.209.149"
    exit 1
fi

CSV_FILE="/home/carlos/Documents/INVESTIGACIÓN/JERARQUICO/MODEL/centralized_6silos_metrics/centralized_7silos/data/silos/silo_${SILO_NUM}.csv"
PREPROC="/home/carlos/Documents/INVESTIGACIÓN/JERARQUICO/MODEL/centralized_6silos_metrics/centralized_7silos/artifacts/preprocessor_global.joblib"

[ ! -f "$CSV_FILE" ] && echo -e "${RED}ERROR: $CSV_FILE no encontrado${NC}" && exit 1
[ ! -f "$PREPROC"  ] && echo -e "${RED}ERROR: $PREPROC no encontrado${NC}"  && exit 1
[ ! -f "$VENV" ]     && echo -e "${RED}ERROR: venv no encontrado${NC}"       && exit 1

# Actualizar rutas en config.py de esta RPi
sed -i "s|DATA_PATH.*=.*|DATA_PATH             = \"/home/carlos/Documents/INVESTIGACIÓN/JERARQUICO/MODEL/centralized_6silos_metrics/centralized_7silos/data/silos/silo_${SILO_NUM}.csv\"|" "$BASE_DIR/config.py"
sed -i "s|PREPROCESSOR_PATH.*=.*|PREPROCESSOR_PATH     = \"/home/carlos/Documents/INVESTIGACIÓN/JERARQUICO/MODEL/centralized_6silos_metrics/centralized_7silos/artifacts/preprocessor_global.joblib\"|" "$BASE_DIR/config.py"

METHOD=$(grep "AGGREGATION_METHOD = " "$BASE_DIR/config.py" | grep -o "'[^']*'" | head -1 | tr -d "'")
MY_IP=$(hostname -I | awk '{print $1}')

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  FL Cluster Hospital — WORKER silo_${SILO_NUM}${NC}"
echo -e "${GREEN}  Método: $METHOD | Servidor: $SERVER_IP${NC}"
echo -e "${GREEN}============================================${NC}"

source "$VENV"
mkdir -p "$BASE_DIR/logs" "$BASE_DIR/models" "$BASE_DIR/received_files"
pkill -f "main.py" 2>/dev/null; sleep 1

python3 -c "
import socket,sys
try:
    s=socket.socket(); s.settimeout(5); s.connect(('',8765)); s.close()
    print('  Servidor alcanzable ✓')
except Exception as e:
    print(f'  ERROR: {e}'); sys.exit(1)
" || (echo -e "${RED}Servidor no disponible.${NC}" && exit 1)

echo -e "${GREEN}Iniciando worker silo_${SILO_NUM}...${NC}"
python "$BASE_DIR/main.py" "${MY_IP}:8765" "${SERVER_IP}:8765" 2>&1 |     tee "$BASE_DIR/logs/worker_silo${SILO_NUM}_${METHOD}_$(date +%Y%m%d_%H%M%S).log"

echo -e "${GREEN}✓ Worker silo_${SILO_NUM} finalizado${NC}"
