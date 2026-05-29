#!/bin/bash
# ============================================================
#  run_agent.sh — AGENTE Semi-Descentralizado
#  Uso: bash run_agent.sh <numero_agente> <ip_servidor>
#
#  Ejemplos:
#    bash run_agent.sh 1 172.23.207.115   ← nodeb
#    bash run_agent.sh 2 172.23.207.115   ← nodoc
#    bash run_agent.sh 3 172.23.207.115   ← noded
#    bash run_agent.sh 4 172.23.207.115   ← nodee
#    bash run_agent.sh 5 172.23.207.115   ← nodef
# ============================================================

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$BASE_DIR/venv/bin/activate"
LOGS="$BASE_DIR/logs"
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

AGENT_NUM=$1
SERVER_IP=$2

if [ -z "$AGENT_NUM" ] || [ -z "$SERVER_IP" ]; then
    echo -e "${RED}Uso: bash run_agent.sh <numero_agente> <ip_servidor>${NC}"
    echo "  bash run_agent.sh 1 172.23.207.115"
    exit 1
fi

CSV_FILE="$BASE_DIR/data/silos/silo_${AGENT_NUM}.csv"
METHOD=$(grep "AGGREGATION_METHOD" "$BASE_DIR/config.py" | grep -o "'[^']*'" | head -1 | tr -d "'")

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  FL Semi-Desc — AGENTE agent_${AGENT_NUM}${NC}"
echo -e "${GREEN}  Método: $METHOD${NC}"
echo -e "${GREEN}  Servidor: ws://${SERVER_IP}:8765${NC}"
echo -e "${GREEN}============================================${NC}"

[ ! -f "$VENV" ]    && echo -e "${RED}ERROR: venv no encontrado${NC}" && exit 1
[ ! -f "$CSV_FILE" ] && echo -e "${RED}ERROR: $CSV_FILE no encontrado${NC}" && exit 1

source "$VENV"
mkdir -p "$LOGS"

pkill -f "agent_csv.py" 2>/dev/null; sleep 1

# Verificar servidor
python3 -c "
import socket,sys
try:
    s=socket.socket(); s.settimeout(5); s.connect(('${SERVER_IP}',8765)); s.close()
    print('  Servidor alcanzable ✓')
except Exception as e:
    print(f'  ERROR: {e}'); sys.exit(1)
"
[ $? -ne 0 ] && echo -e "${RED}Asegúrate de que el servidor ya está corriendo${NC}" && exit 1

echo -e "${GREEN}Iniciando agente agent_${AGENT_NUM}...${NC}"
python "$BASE_DIR/agent_csv.py" \
    "agent_${AGENT_NUM}" \
    "$CSV_FILE" \
    "ws://${SERVER_IP}:8765" 2>&1 | tee "$LOGS/run_agent_${AGENT_NUM}_${METHOD}.log"

echo -e "${GREEN}✓ Agente agent_${AGENT_NUM} finalizado${NC}"
