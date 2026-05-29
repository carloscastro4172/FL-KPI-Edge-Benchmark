"""
metrics_utils.py
Utilidades para captura de métricas del sistema (KPIs nodos + comunicaciones).
Centraliza psutil y cálculos derivados usados por worker y server.
"""
import os
import time
import psutil


# ---------------------------------------------------------------------------
# CPU / RAM
# ---------------------------------------------------------------------------

def get_cpu_percent(interval: float = 0.2) -> float:
    """Porcentaje de CPU del proceso actual (non-blocking con intervalo corto)."""
    try:
        return psutil.cpu_percent(interval=interval)
    except Exception:
        return -1.0


def get_ram_mb() -> float:
    """RAM RSS del proceso actual en MB."""
    try:
        proc = psutil.Process(os.getpid())
        return proc.memory_info().rss / (1024 * 1024)
    except Exception:
        return -1.0


def get_temperature_c() -> float:
    """
    Temperatura de la CPU (°C).

    Prioridad de sensores:
      1. 'cpu_thermal'  → Raspberry Pi (thermal zone del SoC)
      2. 'coretemp'     → Intel x86
      3. 'k10temp'      → AMD
      4. Primer sensor disponible (fallback genérico)

    Retorna -1.0 en Docker (sin sensores expuestos) o Windows.
    En Raspberry Pi físico funciona correctamente sin configuración extra.
    """
    PRIORITY = ['cpu_thermal', 'coretemp', 'k10temp', 'cpu-thermal',
                'soc_thermal', 'acpitz']
    try:
        temps = psutil.sensors_temperatures()
        if not temps:
            return -1.0
        # Buscar por prioridad
        for name in PRIORITY:
            if name in temps and temps[name]:
                return round(temps[name][0].current, 2)
        # Fallback: primer sensor que tenga entries
        for entries in temps.values():
            if entries:
                return round(entries[0].current, 2)
    except (AttributeError, Exception):
        pass
    return -1.0


def get_cpu_freq_mhz() -> float:
    """
    Frecuencia actual de la CPU en MHz.

    Útil en Raspberry Pi para detectar throttling térmico:
    si la RPi se calienta, el SoC baja frecuencia (p.ej. de 1500 MHz a 600 MHz).
    Retorna -1.0 si no está disponible (Docker sin privilegios, Windows parcial).
    """
    try:
        freq = psutil.cpu_freq()
        if freq and freq.current > 0:
            return round(freq.current, 1)
    except Exception:
        pass
    return -1.0


# ---------------------------------------------------------------------------
# Red
# ---------------------------------------------------------------------------

def get_net_snapshot() -> dict:
    """
    Snapshot de contadores de red del sistema.
    Usar dos snapshots y calcular delta para métricas de la ronda.
    """
    try:
        c = psutil.net_io_counters()
        return {
            'bytes_sent': c.bytes_sent,
            'bytes_recv': c.bytes_recv,
            'packets_sent': c.packets_sent,
            'packets_recv': c.packets_recv,
            'errin': c.errin,
            'errout': c.errout,
            'dropin': c.dropin,
            'dropout': c.dropout,
        }
    except Exception:
        return {k: 0 for k in
                ['bytes_sent', 'bytes_recv', 'packets_sent', 'packets_recv',
                 'errin', 'errout', 'dropin', 'dropout']}


def net_delta(before: dict, after: dict) -> dict:
    """Diferencia entre dos snapshots de red."""
    return {k: after[k] - before[k] for k in before}


def get_open_sockets() -> int:
    """Número de conexiones/sockets abiertos del proceso."""
    try:
        proc = psutil.Process(os.getpid())
        return len(proc.connections(kind='all'))
    except Exception:
        return -1


# ---------------------------------------------------------------------------
# Throughput / Bandwidth
# ---------------------------------------------------------------------------

def calc_bandwidth_kbps(bytes_transferred: int, elapsed_seconds: float) -> float:
    """Calcula bandwidth en kbps. Retorna 0.0 si elapsed_seconds <= 0."""
    if elapsed_seconds <= 0:
        return 0.0
    return round((bytes_transferred * 8) / (elapsed_seconds * 1000), 4)


def calc_throughput_kbps(bytes_transferred: int, elapsed_seconds: float) -> float:
    """Alias semántico – throughput en kbps."""
    return calc_bandwidth_kbps(bytes_transferred, elapsed_seconds)


# ---------------------------------------------------------------------------
# Métricas de modelo adicionales
# ---------------------------------------------------------------------------

def calc_specificity(all_labels, all_preds) -> float:
    """
    Especificidad = TN / (TN + FP).
    Requiere listas de etiquetas y predicciones binarias (0/1).
    """
    tn = sum(1 for l, p in zip(all_labels, all_preds) if l == 0 and p == 0)
    fp = sum(1 for l, p in zip(all_labels, all_preds) if l == 0 and p == 1)
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


# ---------------------------------------------------------------------------
# Comunicaciones
# ---------------------------------------------------------------------------

def calc_communication_cost(model_size_bytes: int, num_nodes: int) -> int:
    """
    Costo de comunicación por ronda = M * (N-1).
    M = tamaño del modelo en bytes, N = número de nodos.
    En centralizado: server envía a N workers + recibe N updates → 2 * M * N.
    """
    return model_size_bytes * (num_nodes - 1)


# ---------------------------------------------------------------------------
# KPIs de Energía
# ---------------------------------------------------------------------------

# Constantes del modelo energético (del paper de referencia)
ALPHA_CAPACITANCE = 2e-28    # Capacitancia efectiva del procesador (F)
C_CYCLES_PER_BIT  = 20       # Ciclos de CPU por bit de muestra
P_TRANSMISSION_W  = 0.5      # Potencia de transmisión fija (W)


def calc_ecomp(n_samples: int, n_features: int,
               cpu_freq_mhz: float,
               alpha: float = ALPHA_CAPACITANCE,
               c: int = C_CYCLES_PER_BIT) -> float:
    """
    Energía de cómputo por iteración local (Julios).
        Ecomp = 2 * alpha * c * D * f²
    D = n_samples * n_features * 32 bits (float32)
    f = cpu_freq_mhz * 1e6 Hz
    Retorna -1.0 si freq no disponible.
    """
    if cpu_freq_mhz <= 0:
        return -1.0
    D = n_samples * n_features * 32
    f = cpu_freq_mhz * 1e6
    return round(2 * alpha * c * D * (f ** 2), 9)


def calc_ecomm(tcomm_s: float, p: float = P_TRANSMISSION_W) -> float:
    """
    Energía de comunicación por ronda (Julios).
        Ecomm = p * Tcomm
    Retorna -1.0 si tcomm <= 0.
    """
    if tcomm_s <= 0:
        return -1.0
    return round(p * tcomm_s, 6)


def calc_etotal(ecomp: float, ecomm: float) -> float:
    """Energía total por ronda = Ecomp + Ecomm (Julios)."""
    if ecomp < 0 or ecomm < 0:
        return -1.0
    return round(ecomp + ecomm, 9)
