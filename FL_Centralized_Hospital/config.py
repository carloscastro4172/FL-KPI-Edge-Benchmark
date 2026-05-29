"""
Configuración para Arquitectura Centralizada – 7 Silos (Docker)
Adaptada para captura extendida de métricas para paper académico.
"""


class Config:
    # ==================== SERVIDOR ====================
    SERVER_HOST = '0.0.0.0'
    SERVER_PORT = 8765
    MIN_AGENTS = 5               # 1 servidor + 5 workers
    STARTUP_DELAY = 3
    ROUND_START_DELAY = 1

    # ==================== AGENTES ====================
    MAX_RETRIES = 20
    RETRY_DELAY = 4
    LOCAL_EPOCHS = 3
    BATCH_SIZE = 16
    LEARNING_RATE = 0.01
    DATASET_SIZE = None          # None = usar todo el CSV del silo

    # WebSocket
    WS_TIMEOUT = 120
    WS_MAX_SIZE = 20 * 1024 * 1024   # 20 MB

    AGGREGATOR_PORT_BASE = 9000

    # ==================== MODELO ====================
    INPUT_SIZE = 10
    HIDDEN_SIZE = 20
    OUTPUT_SIZE = 2

    # ==================== EARLY STOPPING ====================
    EARLY_STOPPING_PATIENCE = 20
    MAX_ROUNDS = 100
    MIN_RECALL_IMPROVEMENT = 0.001

    # ==================== AGREGACIÓN ====================
    AGGREGATION_METHOD = 'FedProx'   # 'FedAvg' | 'FedProx'
    FEDPROX_MU = 0.01

    # ==================== RUTAS (Docker) ====================
    PREPROCESSOR_PATH = 'artifacts/preprocessor_global.joblib'
    LOGS_DIR = 'logs'
    MODELS_DIR = 'models'

    # ==================== MÉTRICAS ====================
    # KPIs de nodos
    COLLECT_CPU_RAM = True        # psutil CPU/RAM por ronda
    COLLECT_TEMPERATURE = True    # psutil sensors (None si no disponible en Docker)
    COLLECT_NET_IO = True         # psutil net_io_counters delta

    # KPIs de comunicaciones
    MEASURE_LATENCY = True        # receive_time por ronda
    MEASURE_THROUGHPUT = True     # bytes / time

    # Número fijo de saltos en arquitectura centralizada
    NETWORK_HOPS = 1              # siempre 1 (directo worker → server)
    NUM_SILOS = 5


class RaspberryPiConfig(Config):
    MAX_RETRIES = 25
    RETRY_DELAY = 5
    STARTUP_DELAY = 5
    ROUND_START_DELAY = 3
    WS_TIMEOUT = 180
    LOCAL_EPOCHS = 2
    BATCH_SIZE = 8


class FastConfig(Config):
    MAX_RETRIES = 10
    RETRY_DELAY = 1
    STARTUP_DELAY = 1
    ROUND_START_DELAY = 0.5
    WS_TIMEOUT = 60
    LOCAL_EPOCHS = 5
    BATCH_SIZE = 32
