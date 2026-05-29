#!/bin/bash
# ============================================================
#  collect_logs.sh — Recoger logs de las 6 RPis
#  Ejecutar desde tu PC cuando termine el entrenamiento
#  Uso: bash collect_logs.sh
# ============================================================

OUTPUT="/home/josepht/Documents/9no/Paper-FL-S12026/Final/logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT"

echo "============================================"
echo "  Recolectando logs de las 6 RPis"
echo "============================================"

echo "[1/6] nodea — Servidor (172.23.207.115)..."
scp nodea@172.23.207.115:/home/nodea/centralized_6silos/logs/*.csv "$OUTPUT/" 2>/dev/null
scp nodea@172.23.207.115:/home/nodea/centralized_6silos/models/*.pt "$OUTPUT/" 2>/dev/null

echo "[2/6] nodeb — Worker silo_1 (172.23.207.114)..."
scp nodeb@172.23.207.114:/home/nodeb/centralized_6silos/logs/*.csv "$OUTPUT/" 2>/dev/null

echo "[3/6] nodoc — Worker silo_2 (172.23.195.226)..."
scp nodoc@172.23.195.226:/home/nodoc/centralized_6silos/logs/*.csv "$OUTPUT/" 2>/dev/null

echo "[4/6] noded — Worker silo_3 (172.23.195.199)..."
scp noded@172.23.195.199:/home/noded/centralized_6silos/logs/*.csv "$OUTPUT/" 2>/dev/null

echo "[5/6] nodee — Worker silo_4 (172.23.195.196)..."
scp nodee@172.23.195.196:/home/nodee/centralized_6silos/logs/*.csv "$OUTPUT/" 2>/dev/null

echo "[6/6] nodef — Worker silo_5 (172.23.195.194)..."
scp nodef@172.23.195.194:/home/nodef/centralized_6silos/logs/*.csv "$OUTPUT/" 2>/dev/null

echo ""
echo "✓ Logs guardados en: $OUTPUT"
ls "$OUTPUT"
echo ""
echo "Analizar métricas:"
echo "  cd /home/josepht/Documents/9no/Paper-FL-S12026/Final"
echo "  python metrics_analysis.py --logs-dir $OUTPUT --output-dir analysis_centralizado"
