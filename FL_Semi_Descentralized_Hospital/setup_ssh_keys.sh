#!/bin/bash
PASSWORD="1234567890"
WORKERS=("nodeb@172.23.207.114" "nodoc@172.23.195.226" "noded@172.23.195.199" "nodee@172.23.195.196" "nodef@172.23.195.194")
[ ! -f ~/.ssh/id_rsa ] && ssh-keygen -t rsa -b 2048 -f ~/.ssh/id_rsa -N "" -q
echo "Copiando SSH keys..."
for W in "${WORKERS[@]}"; do
    sshpass -p "$PASSWORD" ssh-copy-id -o StrictHostKeyChecking=no "$W" 2>/dev/null
    echo "  ✓ $W"
done
echo "✓ SSH keys listas"
