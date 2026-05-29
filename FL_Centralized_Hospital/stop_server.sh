#!/bin/bash
echo "Deteniendo servidor..."
pkill -f "server_centralized.py"
rm -f "$(dirname "$0")/.pid_server"
echo "✓ Servidor detenido"
