import json
import csv
import os
import pandas as pd
import numpy as np
from config import NODES_JSON, METRICS_CSV, RECEIVED_MODEL_FILENAME, AGGREGATION_METHOD
from logging_config import get_logger

logger = get_logger(__name__)


def get_ipport(addr: str) -> tuple:
    host, port = addr.rsplit(":", 1)
    return host, int(port)


def save_nodes(ips: list, path: str = NODES_JSON) -> dict:
    nodes = {f"Nodo_{i+1}": addr for i, addr in enumerate(ips)}
    nodes["n_nodes"] = len(ips)
    with open(path, "w") as f:
        json.dump(nodes, f, indent=2)
    logger.info(f"[CENTRAL] Nodos guardados: {nodes}")
    return nodes


def load_nodes2dict(path: str = NODES_JSON) -> dict:
    with open(path) as f:
        data = json.load(f)
    return {addr: nid for nid, addr in data.items() if nid != "n_nodes"}


def load_nodes(path: str = NODES_JSON) -> dict:
    with open(path) as f:
        return json.load(f)


def append_metrics(
    metrics_list: list,
    round_n: int,
    K: int = 3,
    tol: float = 1e-4,
    path: str = METRICS_CSV,
    convergence_time_s: float = None,
    convergence_round: int = None,
    global_recall: float = None,
    extra_path: str = None,
    disable_loss_conv: bool = False,
) -> bool:
    """
    disable_loss_conv=True: ignora convergencia por loss,
    devuelve siempre False (el control queda en early stopping por recall).
    """
    """
    Guarda métricas al CSV y evalúa convergencia por loss.
    Incluye todos los KPIs comparables con centralizado/semi-desc.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    df_new = pd.DataFrame(metrics_list)
    df_new["round"] = round_n
    if "node"   not in df_new.columns: df_new["node"]   = "unknown"
    if "h_ronda" not in df_new.columns: df_new["h_ronda"] = None

    # ── Esquema principal ───────────────────────────────────────────────
    fieldnames = [
        # Identificación
        "round", "h_ronda", "node", "aggregation_method",

        # Modelo
        "accuracy", "precision", "recall", "f1_score",
        "specificity", "sensitivity",
        "trainning_time", "loss",

        # Hardware antes/después
        "cpu_percent_before", "cpu_percent_after",
        "ram_mb_before", "ram_mb_after",
        "temperature_c", "cpu_freq_mhz", "open_sockets",

        # Latencias
        "latency_model_download_s",
        "latency_model_upload_s",

        # Red nodo TX (subida modelo)
        "net_bytes_tx_model", "net_bytes_tx_system",
        "net_bandwidth_tx_kbps", "net_packets_sent",
        "net_errors_out", "net_drops_out",

        # Red nodo RX (descarga modelo)
        "net_bytes_rx_model", "net_bytes_rx_system",
        "net_bandwidth_rx_kbps", "net_packets_recv",
        "net_errors_in", "net_drops_in",

        "net_throughput_kbps", "net_transmission_time_s",

        # Red servidor central
        "central_net_bytes_rx_model", "central_net_bytes_rx_system",
        "central_net_bandwidth_rx_kbps", "central_net_throughput_kbps",
        "central_net_transmission_time_s",
        "central_net_bytes_tx_model", "central_net_bytes_tx_system",
        "central_net_bandwidth_tx_kbps", "central_latency_model_dist_s",

        # Energía
        "ecomp_j", "ecomm_j", "etotal_j", "cumulative_etotal_j",
        "n_train_samples", "fednova_tau", "nova_grad_available",

        # Agregación y convergencia
        "comm_overhead_bytes",
        "aggregation_time_s",
        "inter_silo_variance",
        "recall_variance",
        "global_recall",
        "best_recall_so_far",
        "straggler_delay_s",
        "worker_arrival_order",
        "rounds_no_improvement",
        "early_stopping_triggered",
        "total_etotal_round_j",
        "cumulative_ealpha_j",

        # Convergencia
        "converged", "converged_round", "convergence_time_s",
    ]

    # Columnas extra
    extra_path = extra_path or path.replace(".csv", "_extra.csv")
    extra_cols = [c for c in df_new.columns if c not in fieldnames]
    if extra_cols:
        ec_order = [c for c in ["round","node","h_ronda"] if c in df_new.columns] + extra_cols
        extra_df = df_new[ec_order].copy()
        write_hdr = not os.path.exists(extra_path)
        extra_df.to_csv(extra_path, mode="a", index=False, header=write_hdr)
        logger.info(f"[METRICS] Extra: {extra_cols} → {extra_path}")

    # ── Convergencia por loss ────────────────────────────────────────────
    existing = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    df_combined = pd.concat([existing, df_new], ignore_index=True, sort=False) \
                  if not existing.empty else df_new.copy()

    converged = False
    if disable_loss_conv:
        logger.info("[METRICS] Convergencia por loss deshabilitada — solo early stopping")
    else:
        try:
            df_combined["round"] = pd.to_numeric(df_combined["round"], errors="coerce")
            df_combined["loss"]  = pd.to_numeric(df_combined["loss"],  errors="coerce")
            df_combined = df_combined.dropna(subset=["round","loss"])
            df_combined["round"] = df_combined["round"].astype(int)
            df_rounds = df_combined.groupby("round")["loss"].mean().sort_index()
            if len(df_rounds) >= K:
                diffs = abs(pd.Series(df_rounds.tail(K).values).diff().dropna())
                converged = bool((diffs < tol).all())
        except Exception as e:
            logger.error(f"[CONVERGENCIA] {e}")

    df_new["converged"]        = converged
    df_new["converged_round"]  = convergence_round if (converged or convergence_round) else None
    df_new["convergence_time_s"] = (
        round(convergence_time_s, 3)
        if convergence_time_s is not None else None
    )

    df_main = df_new.reindex(columns=fieldnames)
    write_hdr = not os.path.exists(path)
    df_main.to_csv(path, mode="a", index=False, header=write_hdr)
    logger.info(f"[CENTRAL] Ronda {round_n} guardada → {path}")

    if converged:
        logger.info(f"[CONVERGENCIA] ¡Convergido en ronda {round_n}!")
    return converged


def hierarchy_convergence(ruta_csv: str, k: int = 3, tolerancia: float = 1e-4) -> bool:
    df = pd.read_csv(ruta_csv)
    df_ult = df.loc[df.groupby(["h_ronda","node"])["round"].idxmax()]
    df_prom = df_ult.groupby("h_ronda")["loss"].mean().sort_values()
    if len(df_prom) < k:
        return False
    var = np.max(df_prom.values[-k:]) - np.min(df_prom.values[-k:])
    return bool(var < tolerancia)


async def _save_file(data: bytes, save_path: str,
                     filename: str = RECEIVED_MODEL_FILENAME) -> str:
    os.makedirs(save_path, exist_ok=True)
    filepath = os.path.join(save_path, filename)
    with open(filepath, "wb") as f:
        f.write(data)
    logger.info(f"[FILE SAVED] {filepath}")
    return filepath
