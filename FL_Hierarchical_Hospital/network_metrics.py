"""
Métricas de red y hardware.
Temperatura, RAM en MB, CPU freq, sockets, Ecomp/Ecomm/Etotal.
"""
import time
import psutil
from logging_config import get_logger
from config import ENERGY_ALPHA, ENERGY_C_CYCLES, ENERGY_P_TX

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────
# Energía
# ──────────────────────────────────────────────────────────────
def calc_ecomp(n_samples: int, n_features: int, cpu_freq_mhz: float) -> float:
    """Ecomp = 2 · α · c · D · f²  (julios)."""
    if cpu_freq_mhz <= 0:
        return 0.0
    f_hz = cpu_freq_mhz * 1e6
    D_bits = n_samples * n_features * 32
    return 2 * ENERGY_ALPHA * ENERGY_C_CYCLES * D_bits * (f_hz ** 2)

def calc_ecomm(transmission_time_s: float) -> float:
    """Ecomm = p · Tcomm  (julios)."""
    return ENERGY_P_TX * max(0.0, transmission_time_s)

def calc_etotal(ecomp: float, ecomm: float) -> float:
    return ecomp + ecomm


# ──────────────────────────────────────────────────────────────
# Hardware del nodo
# ──────────────────────────────────────────────────────────────
def collect_system_metrics() -> dict:
    """CPU%, RAM MB, temperatura, frecuencia CPU, sockets abiertos."""
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        vm = psutil.virtual_memory()
        ram_mb      = round(vm.used / (1024 * 1024), 2)
        ram_percent = round(vm.percent, 2)
        cpu_freq    = psutil.cpu_freq()
        cpu_freq_mhz = round(cpu_freq.current, 2) if cpu_freq else 0.0
        open_sockets = len(psutil.net_connections(kind="inet"))

        # Temperatura (RPi expone en thermal_zone0)
        temperature_c = -1.0
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                for key in ['cpu_thermal', 'cpu-thermal', 'thermal_zone0', 'coretemp', 'acpitz']:
                    if key in temps and temps[key]:
                        temperature_c = round(temps[key][0].current, 2)
                        break
                if temperature_c == -1.0:
                    for entries in temps.values():
                        if entries:
                            temperature_c = round(entries[0].current, 2)
                            break
        except Exception:
            pass

        return {
            "cpu_percent":   round(cpu_percent, 2),
            "ram_mb":        ram_mb,
            "ram_percent":   ram_percent,
            "temperature_c": temperature_c,
            "cpu_freq_mhz":  cpu_freq_mhz,
            "open_sockets":  open_sockets,
        }
    except Exception as e:
        logger.error(f"Error métricas sistema: {e}")
        return {
            "cpu_percent":   0.0,
            "ram_mb":        0.0,
            "ram_percent":   0.0,
            "temperature_c": -1.0,
            "cpu_freq_mhz":  0.0,
            "open_sockets":  0,
        }


# ──────────────────────────────────────────────────────────────
# Colector de red
# ──────────────────────────────────────────────────────────────
class NetworkMetricsCollector:
    def __init__(self, interface: str = None):
        self.interface = interface or self._get_default_interface()
        self.start_time = None
        self.start_stats = None
        self.bytes_model_tx = 0
        self.bytes_model_rx = 0

    @staticmethod
    def _get_default_interface() -> str:
        try:
            for iface, stats in psutil.net_if_stats().items():
                if iface != "lo" and stats.isup:
                    return iface
        except Exception:
            pass
        return "eth0"

    def _get_net_stats(self) -> dict:
        try:
            io = psutil.net_io_counters(pernic=True)
            if self.interface not in io:
                self.interface = next(iter(io))
            c = io[self.interface]
            return {
                "bytes_sent":   c.bytes_sent,
                "bytes_recv":   c.bytes_recv,
                "packets_sent": c.packets_sent,
                "packets_recv": c.packets_recv,
                "errin":        c.errin,
                "errout":       c.errout,
                "dropin":       c.dropin,
                "dropout":      c.dropout,
            }
        except Exception:
            return {k: 0 for k in
                    ["bytes_sent","bytes_recv","packets_sent","packets_recv",
                     "errin","errout","dropin","dropout"]}

    def start_monitoring(self):
        self.start_time  = time.time()
        self.start_stats = self._get_net_stats()

    def end_monitoring(self) -> dict:
        end_time  = time.time()
        end_stats = self._get_net_stats()
        elapsed   = max(end_time - (self.start_time or end_time), 0.001)

        bytes_tx    = end_stats["bytes_sent"]   - (self.start_stats or end_stats)["bytes_sent"]
        bytes_rx    = end_stats["bytes_recv"]   - (self.start_stats or end_stats)["bytes_recv"]
        pkts_sent   = end_stats["packets_sent"] - (self.start_stats or end_stats)["packets_sent"]
        pkts_recv   = end_stats["packets_recv"] - (self.start_stats or end_stats)["packets_recv"]
        errin       = end_stats["errin"]        - (self.start_stats or end_stats)["errin"]
        errout      = end_stats["errout"]       - (self.start_stats or end_stats)["errout"]
        dropin      = end_stats["dropin"]       - (self.start_stats or end_stats)["dropin"]
        dropout_    = end_stats["dropout"]      - (self.start_stats or end_stats)["dropout"]

        bw_tx = (bytes_tx * 8) / (elapsed * 1000)
        bw_rx = (bytes_rx * 8) / (elapsed * 1000)
        thru  = ((bytes_tx + bytes_rx) * 8) / (elapsed * 1000)

        return {
            "net_bytes_tx_system":     bytes_tx,
            "net_bytes_rx_system":     bytes_rx,
            "net_bytes_tx_model":      self.bytes_model_tx,
            "net_bytes_rx_model":      self.bytes_model_rx,
            "net_packets_sent":        pkts_sent,
            "net_packets_recv":        pkts_recv,
            "net_errors_in":           errin,
            "net_errors_out":          errout,
            "net_drops_in":            dropin,
            "net_drops_out":           dropout_,
            "net_bandwidth_tx_kbps":   round(bw_tx, 2),
            "net_bandwidth_rx_kbps":   round(bw_rx, 2),
            "net_throughput_kbps":     round(thru,  2),
            "net_transmission_time_s": round(elapsed, 3),
        }

    def set_model_bytes_transferred(self, bytes_tx: int = 0, bytes_rx: int = 0):
        self.bytes_model_tx = bytes_tx
        self.bytes_model_rx = bytes_rx

    @staticmethod
    def _empty_metrics() -> dict:
        return {k: 0 for k in [
            "net_bytes_tx_system","net_bytes_rx_system",
            "net_bytes_tx_model","net_bytes_rx_model",
            "net_packets_sent","net_packets_recv",
            "net_errors_in","net_errors_out",
            "net_drops_in","net_drops_out",
            "net_bandwidth_tx_kbps","net_bandwidth_rx_kbps",
            "net_throughput_kbps","net_transmission_time_s",
        ]}


class NetworkMetricsContext:
    def __init__(self, interface: str = None):
        self.collector = NetworkMetricsCollector(interface)

    def __enter__(self) -> NetworkMetricsCollector:
        self.collector.start_monitoring()
        return self.collector

    def __exit__(self, *args):
        return False
