"""
Configuración – Arquitectura Semi-Descentralizada | 6 Silos
1 servidor coordinador (nodea) + 5 agentes con rol dual (nodeb–nodef)
"""


class Config:
    # ==================== SERVIDOR COORDINADOR ====================
    SERVER_HOST = '0.0.0.0'
    SERVER_PORT = 8765
    MIN_AGENTS = 5               # 5 agentes (nodeb, nodoc, noded, nodee, nodef)
    STARTUP_DELAY = 3
    ROUND_START_DELAY = 1

    # ==================== AGENTES ====================
    MAX_RETRIES = 20
    RETRY_DELAY = 4
    LOCAL_EPOCHS = 3
    INTERNAL_ROUNDS = 5          # rondas internas con el mismo agregador
    BATCH_SIZE = 16
    LEARNING_RATE = 0.01
    DATASET_SIZE = None

    WS_TIMEOUT = 120
    WS_MAX_SIZE = 20 * 1024 * 1024

    # Puerto del agregador = AGGREGATOR_PORT_BASE + número del agente
    AGGREGATOR_PORT_BASE = 9000

    # Timeout: si el agregador no recibe todos los modelos en X seg, agrega con lo que hay
    ROUND_TIMEOUT = 60

    # ==================== MODELO ====================
    INPUT_SIZE = 10
    HIDDEN_SIZE = 20
    OUTPUT_SIZE = 2

    # ==================== EARLY STOPPING ====================
    EARLY_STOPPING_PATIENCE = 3
    MAX_ROUNDS = 50
    MIN_RECALL_IMPROVEMENT = 0.001

    # ==================== AGREGACIÓN ====================
    AGGREGATION_METHOD = 'FedAvg'    # 'FedAvg' | 'FedProx' | 'FedNova'
    FEDPROX_MU = 0.01
    FEDNOVA_RHO = 0.0

    # ==================== RUTAS ====================
    PREPROCESSOR_PATH = 'artifacts/preprocessor_global.joblib'
    LOGS_DIR = 'logs'
    MODELS_DIR = 'models'

    # ==================== MÉTRICAS ====================
    COLLECT_CPU_RAM = True
    COLLECT_TEMPERATURE = True
    COLLECT_NET_IO = True
    MEASURE_LATENCY = True
    MEASURE_THROUGHPUT = True

    # Semi-desc: worker → agregador = 1 salto P2P
    NETWORK_HOPS = 1
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
    LOCAL_EPOCHS = 5
    BATCH_SIZE = 32
