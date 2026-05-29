"""
Lógica FL: servidor central (central_main) + nodo worker (client_main)
Métodos: FedAvg | FedProx | FedNova
KPIs: energía, straggler, varianza, early stopping
"""
import asyncio
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    ROUNDS, EPOCHS, LEARNING_RATE, IN_FEATURES,
    LISTENER_DURATION, NODES_JSON, METRICS_CSV, MODEL_PATH, DATA_PATH,
    SLEEP_INTERVAL, POST_ROUND_DELAY, NODES_LISTENER_DELAY, LABEL_COLUMN,
    AGGREGATION_METHOD, FEDPROX_MU, NUM_SILOS,
    EARLY_STOPPING_PATIENCE, MIN_RECALL_IMPROVEMENT,
    ENERGY_ALPHA, ENERGY_C_CYCLES, ENERGY_P_TX,
)
from connections.client import (send_identified, send_file_identified,
                                 send_file_to_nodes, send_message_to_nodes, send)
from connections.server import listener_nodes, listener_server
from model.fed_model import MLP, ModelTrainer, aggregate
from model.create_model import create_model
from network_metrics import (collect_system_metrics, calc_ecomp, calc_ecomm, calc_etotal,
                              NetworkMetricsContext)
from utils import save_nodes, append_metrics, get_ipport
from logging_config import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────
# SERVIDOR CENTRAL
# ──────────────────────────────────────────────────────────────
async def central_main(addr: str, ips: list, h_ronda: int = None):
    nodes        = save_nodes(ips, NODES_JSON)
    n            = nodes["n_nodes"]
    nodes_by_addr = {a: nid for nid, a in nodes.items() if nid != "n_nodes"}
    node_id_to_addr = {nid: a for a, nid in nodes_by_addr.items()}

    _, listen_port = get_ipport(addr)
    local_listen   = f"0.0.0.0:{listen_port}"

    # Estado early stopping
    best_recall           = 0.0
    rounds_no_improvement = 0
    convergence_round     = None
    convergence_time_s    = None
    experiment_start      = time.time()

    # Estado FedNova acumulado
    global_params_cache: dict = {}   # w0 actual
    nova_grads_cache: dict    = {}   # {node_id: grad}
    nova_samples_cache: dict  = {}   # {node_id: (n_samples, tau)}

    # Energía global acumulada
    cumul_ealpha = 0.0

    logger.info(f"[CENTRAL] Arquitectura=Cluster | Método={AGGREGATION_METHOD} | Rondas={ROUNDS}")

    for round_n in range(1, ROUNDS + 1):
        logger.info(f"\n[CENTRAL] ══ Ronda {round_n}/{ROUNDS} ══")
        round_start = time.time()

        node_dist_metrics: dict = {}
        if round_n > 1:
            await asyncio.sleep(POST_ROUND_DELAY)
            node_dist_metrics = await send_file_to_nodes(ips, MODEL_PATH,
                                                          delay=LISTENER_DURATION * 6)

        round_dir = f"received_files/round_{round_n}"
        os.makedirs(round_dir, exist_ok=True)
        results = await listener_nodes(
            local_listen, nodes=nodes_by_addr,
            delay=NODES_LISTENER_DELAY * 10,
            save_path=round_dir,
        )

        metrics_list: list  = []
        model_paths:  list  = []
        central_rx:   dict  = {}
        nova_grads_round: dict = {}
        sample_counts_round: dict = {}

        # Tiempo de llegada de cada nodo (para straggler)
        arrival_times: dict = {}

        for node_id, received in results.items():
            arrival_times[node_id] = time.time() - round_start
            for item in received:
                if isinstance(item, str) and os.path.isfile(item) and item.endswith(".pt"):
                    model_paths.append((node_id, item))
                elif isinstance(item, dict):
                    central_rx[node_id] = item
                elif isinstance(item, str):
                    try:
                        m = json.loads(item)
                        m["node"] = node_id
                        metrics_list.append(m)

                        # Extraer datos FedNova si están presentes
                        if m.get("nova_grad_available"):
                            n_s   = m.get("n_train_samples", 1)
                            tau_i = m.get("fednova_tau", 1)
                            nova_samples_cache[node_id] = (n_s, tau_i)
                            sample_counts_round[node_id] = (n_s, tau_i)
                    except json.JSONDecodeError:
                        pass

        # Cargar gradientes FedNova desde archivos si existen
        for node_id, _ in results.items():
            nova_file = os.path.join(round_dir, f"{node_id}_nova.pt")
            if os.path.exists(nova_file):
                try:
                    nova_grads_round[node_id] = torch.load(nova_file, map_location="cpu")
                    nova_grads_cache[node_id] = nova_grads_round[node_id]
                except Exception:
                    pass

        if not model_paths:
            logger.error("[CENTRAL] Sin modelos recibidos.")
            break

        # Straggler: max_arrival - min_arrival
        if len(arrival_times) >= 2:
            straggler_delay_s = round(
                max(arrival_times.values()) - min(arrival_times.values()), 3)
        else:
            straggler_delay_s = 0.0
        arrival_order = json.dumps(
            {nid: round(t, 3) for nid, t in
             sorted(arrival_times.items(), key=lambda x: x[1])})

        # Cargar pesos globales actuales para FedNova
        if os.path.exists(MODEL_PATH):
            try:
                architecture = MLP(in_features=IN_FEATURES)
                global_params_cache = architecture.load_state_dict(
                    torch.load(MODEL_PATH, map_location="cpu"), strict=False
                ) or {n: p.data for n, p in architecture.named_parameters()}
                global_params_cache = {n: p.data.clone()
                                       for n, p in architecture.named_parameters()}
            except Exception:
                global_params_cache = {}

        # ── Agregación ──────────────────────────────────────────────
        t_agg_start = time.time()
        mp_only     = [p for _, p in model_paths]
        avg_state   = aggregate(
            AGGREGATION_METHOD, mp_only,
            nova_grads     = nova_grads_round,
            sample_counts  = sample_counts_round,
            global_params  = global_params_cache,
        )
        aggregation_time_s = round(time.time() - t_agg_start, 3)
        os.makedirs(os.path.dirname(os.path.abspath(MODEL_PATH)), exist_ok=True)
        torch.save(avg_state, MODEL_PATH)

        convergence_elapsed = round(time.time() - experiment_start, 3)

        # ── Métricas globales ────────────────────────────────────────
        recalls     = []
        losses_list = []
        for m in metrics_list:
            try: recalls.append(float(m.get("recall", 0)))
            except Exception: pass
            try: losses_list.append(float(m.get("loss", 0)))
            except Exception: pass

        global_recall     = float(np.mean(recalls)) if recalls else 0.0
        recall_variance   = float(np.var(recalls))  if recalls else 0.0
        inter_silo_var    = float(np.var(losses_list)) if losses_list else 0.0

        # Energía total de los workers en esta ronda
        total_ecomp_round = sum(float(m.get("ecomp_j", 0)) for m in metrics_list)
        total_ecomm_round = sum(float(m.get("ecomm_j", 0)) for m in metrics_list)
        total_etotal_round = total_ecomp_round + total_ecomm_round
        cumul_ealpha += total_etotal_round

        logger.info(f"[CENTRAL] Ronda {round_n}: recall={global_recall:.4f} "
                    f"var={recall_variance:.4f} straggler={straggler_delay_s:.2f}s "
                    f"agg={aggregation_time_s:.3f}s")

        # ── Early stopping ──────────────────────────────────────────
        if global_recall > best_recall + MIN_RECALL_IMPROVEMENT:
            best_recall           = global_recall
            rounds_no_improvement = 0
            convergence_round     = round_n
            convergence_time_s    = convergence_elapsed
            logger.info(f"[CENTRAL] ✅ Mejora → recall={best_recall:.4f}")
        else:
            rounds_no_improvement += 1
            logger.info(f"[CENTRAL] Sin mejora: {rounds_no_improvement}/{EARLY_STOPPING_PATIENCE}")

        early_stop = rounds_no_improvement >= EARLY_STOPPING_PATIENCE
        if early_stop:
            logger.info(f"[CENTRAL] 🛑 EARLY STOPPING | mejor recall={best_recall:.4f} | ronda={convergence_round}")

        # Enriquecer métricas con datos del servidor central
        for m in metrics_list:
            nid = m.get("node")
            m["aggregation_method"]        = AGGREGATION_METHOD
            m["aggregation_time_s"]        = aggregation_time_s
            m["inter_silo_variance"]       = round(inter_silo_var, 6)
            m["recall_variance"]           = round(recall_variance, 6)
            m["global_recall"]             = round(global_recall, 4)
            m["straggler_delay_s"]         = straggler_delay_s
            m["worker_arrival_order"]      = arrival_order
            m["best_recall_so_far"]        = round(best_recall, 4)
            m["rounds_no_improvement"]     = rounds_no_improvement
            m["early_stopping_triggered"]  = early_stop
            m["cumulative_ealpha_j"]       = round(cumul_ealpha, 6)
            m["total_etotal_round_j"]      = round(total_etotal_round, 6)
            if h_ronda is not None:
                m["h_ronda"] = h_ronda

            # Red central RX
            if nid in central_rx:
                rx = central_rx[nid]
                m["central_net_bytes_rx_model"]      = rx.get("net_bytes_rx_model", 0)
                m["central_net_bytes_rx_system"]     = rx.get("net_bytes_rx_system", 0)
                m["central_net_bandwidth_rx_kbps"]   = rx.get("net_bandwidth_rx_kbps", 0)
                m["central_net_throughput_kbps"]     = rx.get("net_throughput_kbps", 0)
                m["central_net_transmission_time_s"] = rx.get("net_transmission_time_s", 0)
                m["comm_overhead_bytes"]             = int(
                    rx.get("net_bytes_rx_model", 0) * max(0, len(ips) - 1))

            # Red central TX
            if round_n > 1 and nid in node_id_to_addr:
                dist = node_dist_metrics.get(node_id_to_addr[nid], {})
                m["central_net_bytes_tx_model"]    = dist.get("net_bytes_tx_model", 0)
                m["central_net_bytes_tx_system"]   = dist.get("net_bytes_tx_system", 0)
                m["central_net_bandwidth_tx_kbps"] = dist.get("net_bandwidth_tx_kbps", 0)
                m["central_latency_model_dist_s"]  = dist.get("net_transmission_time_s", 0)
            else:
                m["central_net_bytes_tx_model"]    = 0
                m["central_net_bytes_tx_system"]   = 0
                m["central_net_bandwidth_tx_kbps"] = 0
                m["central_latency_model_dist_s"]  = 0

        append_metrics(
            metrics_list, round_n,
            path=METRICS_CSV, K=3,
            convergence_time_s=convergence_time_s if early_stop else None,
            convergence_round=convergence_round,
            global_recall=global_recall,
            disable_loss_conv=True,   # usar solo early stopping por recall
        )

        if round_n < ROUNDS:
            if early_stop:
                await send_message_to_nodes(ips, "CONVERGED", LISTENER_DURATION)
                break
            else:
                await send_message_to_nodes(ips, "NOT CONVERGED", LISTENER_DURATION)

    logger.info(f"\n[CENTRAL] Entrenamiento completado. Best recall={best_recall:.4f}")


# ──────────────────────────────────────────────────────────────
# NODO WORKER
# ──────────────────────────────────────────────────────────────
async def client_main(addr: str, server_addr: str, h_ronda: int = None):
    os.makedirs(os.path.dirname(os.path.abspath(MODEL_PATH)) or ".", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    _, listen_port = get_ipport(addr)
    local_listen   = f"0.0.0.0:{listen_port}"

    cumul_etotal = 0.0

    for round_n in range(1, ROUNDS + 1):
        logger.info(f"\n[NODE] ══ Ronda {round_n}/{ROUNDS} [{AGGREGATION_METHOD}] ══")

        net_rx: dict = {}
        net_tx: dict = {}
        latency_download = 0.0

        # ── Descargar modelo global (ronda > 1) ─────────────────────
        if round_n > 1:
            received, net_rx = await listener_server(
                local_listen, delay=LISTENER_DURATION * 6, file_path=MODEL_PATH)
            if received is None:
                logger.error("[NODE] Sin modelo. Abortando.")
                break
            latency_download = net_rx.get("net_transmission_time_s", 0.0)

        # ── Recolectar recursos ANTES del entrenamiento ─────────────
        hw_before = collect_system_metrics()

        # ── Entrenamiento local ──────────────────────────────────────
        architecture = MLP(in_features=IN_FEATURES)
        trainer      = ModelTrainer(model_path=MODEL_PATH,
                                    model_architecture=architecture)
        train_loader, test_loader = trainer.load_csv(DATA_PATH, label_col=LABEL_COLUMN)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(architecture.parameters(), lr=LEARNING_RATE)

        train_time, last_loss, fednova_tau, fednova_grad = trainer.fit(
            train_loader, criterion, optimizer, epochs=EPOCHS)

        metrics = trainer.evaluate(test_loader)
        metrics["trainning_time"] = train_time
        metrics["loss"]           = last_loss

        # ── Recursos DESPUÉS del entrenamiento ──────────────────────
        hw_after = collect_system_metrics()
        metrics["cpu_percent_before"]  = hw_before["cpu_percent"]
        metrics["cpu_percent_after"]   = hw_after["cpu_percent"]
        metrics["ram_mb_before"]       = hw_before["ram_mb"]
        metrics["ram_mb_after"]        = hw_after["ram_mb"]
        metrics["temperature_c"]       = hw_after["temperature_c"]
        metrics["cpu_freq_mhz"]        = hw_after["cpu_freq_mhz"]
        metrics["open_sockets"]        = hw_after["open_sockets"]

        # ── Energía ─────────────────────────────────────────────────
        n_train    = trainer.n_train or 1
        n_features = trainer.n_features or IN_FEATURES
        cpu_freq   = hw_after["cpu_freq_mhz"]
        ecomp  = calc_ecomp(n_train, n_features, cpu_freq)
        ecomm  = calc_ecomm(latency_download)
        etotal = calc_etotal(ecomp, ecomm)
        if etotal > 0:
            cumul_etotal += etotal

        metrics["ecomp_j"]            = ecomp
        metrics["ecomm_j"]            = ecomm
        metrics["etotal_j"]           = etotal
        metrics["cumulative_etotal_j"]= round(cumul_etotal, 6)
        metrics["n_train_samples"]    = n_train
        metrics["fednova_tau"]        = fednova_tau
        metrics["nova_grad_available"]= len(fednova_grad) > 0
        metrics["comm_overhead_bytes"]= int(
            net_rx.get("net_bytes_rx_model", 0) * max(0, NUM_SILOS - 1))

        # Red RX
        if net_rx:
            metrics["net_bytes_rx_model"]    = net_rx.get("net_bytes_rx_model", 0)
            metrics["net_bytes_rx_system"]   = net_rx.get("net_bytes_rx_system", 0)
            metrics["net_bandwidth_rx_kbps"] = net_rx.get("net_bandwidth_rx_kbps", 0)
            metrics["net_packets_recv"]      = net_rx.get("net_packets_recv", 0)
            metrics["net_errors_in"]         = net_rx.get("net_errors_in", 0)
            metrics["net_drops_in"]          = net_rx.get("net_drops_in", 0)

        metrics["latency_model_download_s"] = latency_download
        if h_ronda is not None:
            metrics["h_ronda"] = h_ronda

        trainer.save(MODEL_PATH)

        # ── Guardar gradiente FedNova si aplica ──────────────────────
        if AGGREGATION_METHOD == 'FedNova' and fednova_grad:
            nova_path = MODEL_PATH.replace(".pt", "_nova.pt")
            torch.save(fednova_grad, nova_path)

        # ── Enviar modelo al servidor ────────────────────────────────
        await asyncio.sleep(SLEEP_INTERVAL)
        success, net_tx = await send_file_identified(addr, server_addr, MODEL_PATH)

        metrics["latency_model_upload_s"]   = net_tx.get("net_transmission_time_s", 0.0)
        metrics["net_bytes_tx_model"]       = net_tx.get("net_bytes_tx_model", 0)
        metrics["net_bytes_tx_system"]      = net_tx.get("net_bytes_tx_system", 0)
        metrics["net_bandwidth_tx_kbps"]    = net_tx.get("net_bandwidth_tx_kbps", 0)
        metrics["net_packets_sent"]         = net_tx.get("net_packets_sent", 0)
        metrics["net_errors_out"]           = net_tx.get("net_errors_out", 0)
        metrics["net_drops_out"]            = net_tx.get("net_drops_out", 0)
        metrics["net_throughput_kbps"]      = net_tx.get("net_throughput_kbps", 0)
        metrics["net_transmission_time_s"]  = net_tx.get("net_transmission_time_s", 0)

        logger.info(f"[NODE] ⚡ Ecomp={ecomp:.3e}J Ecomm={ecomm:.4f}J "
                    f"Etotal={etotal:.4f}J Acum={cumul_etotal:.4f}J")

        await send_identified(addr, server_addr, json.dumps(metrics))

        if round_n < ROUNDS:
            received, _ = await listener_server(local_listen, LISTENER_DURATION * 100)
            if received == "CONVERGED":
                logger.info("[NODE] ✅ Servidor señaló CONVERGED. Parando.")
                break

    logger.info("\n[NODE] Entrenamiento federado completado.")


# ──────────────────────────────────────────────────────────────
# Punto de entrada
# ──────────────────────────────────────────────────────────────
def main(central: bool, addr: str, server_addr: str,
         childs: list = None, h_ronda: int = None):
    if central:
        asyncio.run(central_main(addr, childs, h_ronda))
    else:
        asyncio.run(client_main(addr, server_addr, h_ronda))
