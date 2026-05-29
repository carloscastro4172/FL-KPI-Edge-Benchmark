#!/bin/bash
# ============================================================
#  run_server.sh — SERVIDOR (nodea 172.23.207.115)
#  Al terminar el entrenamiento recoge los logs de todos
#  los workers automáticamente.
#
#  Uso: bash run_server.sh
# ============================================================

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$BASE_DIR/venv/bin/activate"
LOGS="$BASE_DIR/logs"
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  FL Centralizado — SERVIDOR (nodea)${NC}"
echo -e "${GREEN}  Esperando 5 workers...${NC}"
echo -e "${GREEN}============================================${NC}"

if [ ! -f "$VENV" ]; then
    echo -e "${RED}ERROR: venv no encontrado${NC}"
    exit 1
fi

source "$VENV"
mkdir -p "$LOGS"

pkill -f "server_centralized.py" 2>/dev/null
sleep 2

echo -e "${GREEN}Iniciando servidor...${NC}"
python "$BASE_DIR/server_centralized.py" 2>&1 | tee "$LOGS/run_server.log"

# ── Cuando el servidor termina, recoger logs automáticamente ──
EXIT_CODE=$?
echo ""
echo -e "${YELLOW}Servidor finalizado (código $EXIT_CODE)${NC}"
echo -e "${GREEN}Recolectando logs de todos los nodos...${NC}"
sleep 3
bash "$BASE_DIR/collect_on_server.sh"
