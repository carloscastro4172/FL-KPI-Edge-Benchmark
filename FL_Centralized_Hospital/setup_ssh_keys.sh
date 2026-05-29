#!/bin/bash
# ============================================================
#  setup_ssh_keys.sh
#  Ejecutar UNA SOLA VEZ en nodea (servidor)
#  Genera clave SSH y la copia a todos los workers
#  para que el servidor pueda jalar logs sin contraseña
#
#  Uso: bash setup_ssh_keys.sh
# ============================================================

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

WORKERS=(
    "nodeb@172.23.207.114"
    "nodoc@172.23.195.226"
    "noded@172.23.195.199"
    "nodee@172.23.195.196"
    "nodef@172.23.195.194"
)

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  Configurando SSH keys en nodea${NC}"
echo -e "${GREEN}============================================${NC}"

# Generar clave si no existe
if [ ! -f ~/.ssh/id_rsa ]; then
    echo -e "${YELLOW}Generando clave SSH...${NC}"
    ssh-keygen -t rsa -b 2048 -f ~/.ssh/id_rsa -N ""
    echo -e "${GREEN}  ✓ Clave generada${NC}"
else
    echo -e "${GREEN}  ✓ Clave SSH ya existe${NC}"
fi

# Copiar clave a cada worker (pedirá contraseña una vez por worker)
echo ""
echo -e "${YELLOW}Copiando clave a workers (necesitarás la contraseña de cada uno):${NC}"
echo ""

for WORKER in "${WORKERS[@]}"; do
    echo -e "${YELLOW}  → $WORKER${NC}"
    ssh-copy-id -o StrictHostKeyChecking=no "$WORKER"
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}  ✓ $WORKER OK${NC}"
    else
        echo -e "${RED}  ✗ $WORKER falló — verifica IP y contraseña${NC}"
    fi
    echo ""
done

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  ✓ SSH keys configuradas${NC}"
echo -e "${GREEN}  El servidor puede ahora jalar logs${NC}"
echo -e "${GREEN}  automáticamente al terminar el entrena-${NC}"
echo -e "${GREEN}  miento.${NC}"
echo -e "${GREEN}============================================${NC}"
