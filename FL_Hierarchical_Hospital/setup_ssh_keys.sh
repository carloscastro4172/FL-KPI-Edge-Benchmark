#!/bin/bash
PASSWORD="1234567890"
WORKERS=(
    "nodeb@172.23.209.148"
    "nodoc@172.23.209.153"
    "noded@172.23.209.152"
    "nodee@172.23.209.144"
    "nodef@172.23.209.150"
)
echo "Instalando sshpass..."
sudo apt-get install -y sshpass -q
[ ! -f ~/.ssh/id_rsa ] && ssh-keygen -t rsa -b 2048 -f ~/.ssh/id_rsa -N "" -q
echo "Copiando clave pública a workers..."
for W in "${WORKERS[@]}"; do
    echo -n "  $W ... "
    sshpass -p "$PASSWORD" ssh-copy-id -o StrictHostKeyChecking=no "$W" 2>/dev/null
    echo "✓"
done
echo ""
echo "Probando conexiones..."
for W in "${WORKERS[@]}"; do
    echo -n "  $W: "
    ssh -o ConnectTimeout=5 "$W" "echo OK" 2>/dev/null || echo "FALLO"
done
