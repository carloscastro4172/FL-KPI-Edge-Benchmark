#!/bin/bash
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

METHOD=$(grep "AGGREGATION_METHOD" "$BASE_DIR/config.py" | grep -o "'[^']*'" | head -1 | tr -d "'")
TS=$(date +%Y%m%d_%H%M%S)
DEST="$BASE_DIR/resultados/${METHOD}_${TS}"
mkdir -p "$DEST"/{nodea,nodeb,nodoc,noded,nodee,nodef}

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  Recolectando logs — $METHOD${NC}"
echo -e "${GREEN}  Destino: $DEST${NC}"
echo -e "${GREEN}============================================${NC}"

echo -e "${YELLOW}[1/6] nodea (local)...${NC}"
cp "$BASE_DIR/logs"/*.csv  "$DEST/nodea/" 2>/dev/null
cp "$BASE_DIR/logs"/*.log  "$DEST/nodea/" 2>/dev/null
cp -r "$BASE_DIR/models"   "$DEST/nodea/" 2>/dev/null
echo -e "${GREEN}  ✓ nodea${NC}"

declare -A WORKERS=(
    ["nodeb"]="nodeb@172.23.209.148"
    ["nodoc"]="nodoc@172.23.209.153"
    ["noded"]="noded@172.23.209.152"
    ["nodee"]="nodee@172.23.209.144"
    ["nodef"]="nodef@172.23.209.150"
)

IDX=2
for NODE in nodeb nodoc noded nodee nodef; do
    TARGET="${WORKERS[$NODE]}"
    USER="${TARGET%%@*}"
    echo -e "${YELLOW}[$IDX/6] $NODE...${NC}"
    scp -o ConnectTimeout=10 -o StrictHostKeyChecking=no \
        "${TARGET}:/home/${USER}/fl_cluster_hospital/logs/*.csv" "$DEST/$NODE/" 2>/dev/null
    scp -o ConnectTimeout=10 -o StrictHostKeyChecking=no \
        "${TARGET}:/home/${USER}/fl_cluster_hospital/logs/*.log" "$DEST/$NODE/" 2>/dev/null
    echo -e "${GREEN}  ✓ $NODE${NC}"
    IDX=$((IDX+1))
done

mkdir -p "$DEST/all_csv"
find "$DEST" -name "*.csv" ! -path "*/all_csv/*" -exec cp {} "$DEST/all_csv/" \;

echo ""
echo -e "${GREEN}✓ Listo: $DEST/all_csv/${NC}"
ls "$DEST/all_csv/"
