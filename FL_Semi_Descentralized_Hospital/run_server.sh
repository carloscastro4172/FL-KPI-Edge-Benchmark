#!/bin/bash
# Servidor coordinador semi-descentralizado (nodea)
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$BASE_DIR/venv/bin/activate"
LOGS="$BASE_DIR/logs"
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

METHOD=$(grep "AGGREGATION_METHOD" "$BASE_DIR/config.py" | grep -o "'[^']*'" | head -1 | tr -d "'")
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  FL Semi-Desc — SERVIDOR (nodea)${NC}"
echo -e "${GREEN}  Método: $METHOD | Esperando 5 agentes...${NC}"
echo -e "${GREEN}============================================${NC}"

[ ! -f "$VENV" ] && echo -e "${RED}ERROR: venv no encontrado${NC}" && exit 1
source "$VENV"
mkdir -p "$LOGS"

pkill -f "server.py" 2>/dev/null; sleep 2

MY_IP=$(hostname -I | awk '{print $1}')
echo -e "${YELLOW}  IP servidor: $MY_IP${NC}"
echo -e "${YELLOW}  Agentes deben apuntar a: ws://$MY_IP:8765${NC}"
echo ""

python "$BASE_DIR/server.py" 2>&1 | tee "$LOGS/run_server_${METHOD}.log"

echo -e "${GREEN}Servidor finalizado. Recolectando logs...${NC}"
sleep 3
bash "$BASE_DIR/collect_on_server.sh"
