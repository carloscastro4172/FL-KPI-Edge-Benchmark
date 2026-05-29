#!/bin/bash
# ============================================================
#  collect_on_server.sh
#  Ejecutar en nodea (servidor) cuando termine el entrena-
#  miento. Jala los CSVs de cada worker y los organiza en:
#
#  logs/
#  └── experimento_YYYYMMDD_HHMMSS/
#      ├── nodea/    ← server + worker propio
#      ├── nodeb/
#      ├── nodoc/
#      ├── noded/
#      ├── nodee/
#      └── nodef/
#
#  Uso: bash collect_on_server.sh
#  (se llama automáticamente desde run_server.sh al terminar)
# ============================================================

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

TS=$(date +%Y%m%d_%H%M%S)
DEST="$BASE_DIR/logs/experimento_$TS"
mkdir -p "$DEST"

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  RECOLECTANDO LOGS DE TODOS LOS NODOS${NC}"
echo -e "${GREEN}  Destino: $DEST${NC}"
echo -e "${GREEN}============================================${NC}"

# ── Logs locales del servidor ─────────────────────────────────
echo -e "${YELLOW}[1/6] nodea — servidor (local)...${NC}"
mkdir -p "$DEST/nodea"
cp "$BASE_DIR/logs"/server_centralized_*.csv "$DEST/nodea/" 2>/dev/null
cp "$BASE_DIR/logs"/run_server.log            "$DEST/nodea/" 2>/dev/null
cp -r "$BASE_DIR/models"                      "$DEST/nodea/" 2>/dev/null
echo -e "${GREEN}  ✓ nodea copiado${NC}"

# ── Workers ───────────────────────────────────────────────────
declare -A WORKERS
WORKERS["nodeb"]="nodeb@172.23.207.114"
WORKERS["nodoc"]="nodoc@172.23.195.226"
WORKERS["noded"]="noded@172.23.195.199"
WORKERS["nodee"]="nodee@172.23.195.196"
WORKERS["nodef"]="nodef@172.23.195.194"

IDX=2
for NODE in nodeb nodoc noded nodee nodef; do
    TARGET="${WORKERS[$NODE]}"
    USER="${TARGET%%@*}"
    IP="${TARGET##*@}"
    echo -e "${YELLOW}[$IDX/6] $NODE ($IP)...${NC}"
    mkdir -p "$DEST/$NODE"
    scp -o ConnectTimeout=10 -o StrictHostKeyChecking=no \
        "${TARGET}:/home/${USER}/centralized_6silos/logs/worker_*.csv" \
        "$DEST/$NODE/" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}  ✓ $NODE OK${NC}"
    else
        echo -e "${RED}  ✗ $NODE falló (¿SSH keys configuradas? bash setup_ssh_keys.sh)${NC}"
    fi
    IDX=$((IDX+1))
done

# ── Consolidar todos los CSVs en una carpeta plana también ────
mkdir -p "$DEST/all_csv"
find "$DEST" -name "*.csv" ! -path "*/all_csv/*" -exec cp {} "$DEST/all_csv/" \;

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  ✓ Recolección completada${NC}"
echo -e "${GREEN}  Archivos en: $DEST${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "Archivos recolectados:"
ls "$DEST/all_csv/"
echo ""
echo "Analizar métricas:"
echo -e "${YELLOW}  source venv/bin/activate${NC}"
echo -e "${YELLOW}  python metrics_analysis.py --logs-dir $DEST/all_csv --output-dir $DEST/analysis${NC}"

# Ejecutar análisis automáticamente
source "$BASE_DIR/venv/bin/activate" 2>/dev/null
if python "$BASE_DIR/metrics_analysis.py" \
    --logs-dir "$DEST/all_csv" \
    --output-dir "$DEST/analysis" 2>/dev/null; then
    echo ""
    echo -e "${GREEN}  ✓ Análisis completado en: $DEST/analysis${NC}"
fi
