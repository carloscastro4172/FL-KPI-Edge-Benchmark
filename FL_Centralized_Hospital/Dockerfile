FROM python:3.10-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY config.py .
COPY mlp.py .
COPY data_preparation.py .
COPY metrics_utils.py .
COPY server_centralized.py .
COPY worker_centralized.py .

# Crear directorios de salida
RUN mkdir -p logs models artifacts data/silos

# Punto de entrada configurado en docker-compose
CMD ["python", "--version"]
