#!/bin/bash
echo "Deteniendo worker..."
pkill -f "worker_centralized.py"
rm -f "$(dirname "$0")"/.pid_worker_*
echo "✓ Worker detenido"
