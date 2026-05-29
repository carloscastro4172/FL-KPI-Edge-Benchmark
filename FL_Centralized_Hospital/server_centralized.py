"""
Servidor Centralizado – Arquitectura FL Centralizada
KPIs extendidos para paper académico: timing, straggler, convergencia,
overhead de comunicación, métricas globales del modelo.
"""
import asyncio
import websockets
import json
import torch
import pickle
import os
import csv
import numpy as np
import time
from typing import Dict, Optional
from datetime import datetime

from config import Config
from mlp import MLP
from metrics_utils import (
    get_cpu_percent, get_ram_mb, get_temperature_c, get_cpu_freq_mhz,
    get_net_snapshot, net_delta
)


# ---------------------------------------------------------------------------
# CSV headers
# ---------------------------------------------------------------------------
SERVER_CSV_HEADERS = [
    # Identificación
    'timestamp', 'round',
    # Arquitectura
    'aggregation_method', 'architecture', 'network_hops',
    'num_agents_total', 'num_workers_responded',
    # ── KPI: Timing ──────────────────────────────────────────────────────
    'aggregation_time_s',          # duración de FedAvg/FedProx
    'round_total_time_s',          # desde start_round hasta envío siguiente
    'total_elapsed_time_s',        # desde inicio del experimento
    # ── KPI: Straggler / Dinamismo ───────────────────────────────────────
    'first_worker_arrival_s',      # tiempo del primer worker en llegar (relativo a round start)
    'last_worker_arrival_s',       # tiempo del último worker (straggler)
    'straggler_delay_s',           # last - first (variabilidad de llegada)
    'worker_arrival_order',        # JSON: {agent_id: arrival_offset_s}
    'dropout_count',               # agentes desconectados en esta ronda
    # ── KPI: Comunicaciones ──────────────────────────────────────────────
    'model_size_bytes',            # tamaño del modelo serializado
    'total_upload_bytes',          # suma de todos los uploads de workers
    'total_download_bytes',        # suma de todos los downloads a workers
    'comm_overhead_bytes',         # M*(N-1) bidireccional
    'server_net_bytes_sent',       # delta psutil bytes enviados por servidor
    'server_net_bytes_recv',       # delta psutil bytes recibidos por servidor
    'server_net_packets_sent',     # delta psutil packets enviados
    'server_net_packets_recv',     # delta psutil packets recibidos
    # ── KPI: CPU / RAM / Temp / Freq del servidor ────────────────────────
    'server_cpu_before',
    'server_cpu_after',
    'server_ram_before_mb',
    'server_ram_after_mb',
    'server_temperature_c',        # temperatura del SoC (RPi: cpu_thermal)
    'server_cpu_freq_mhz',         # frecuencia actual (detecta throttling térmico)
    # ── KPI: Modelo Federado (global) ────────────────────────────────────
    'global_recall',
    'global_accuracy',
    'global_precision',
    'global_f1',
    'global_loss',
    'global_specificity',
    # Per-worker recall variance (indicador non-IID)
    'recall_variance',             # var de recalls locales → impacto non-IID
    'recall_std',
    'accuracy_variance',
    # ── KPI: Convergencia ────────────────────────────────────────────────
    'best_recall_so_far',
    'delta_recall',
    'rounds_without_improvement',
    # ── Early stopping ───────────────────────────────────────────────────
    'early_stopping_triggered',
    'early_stopping_reason',
    'patience_value',
    'convergence_round',           # ronda en que se alcanzó el mejor recall
    'convergence_time_s',          # tiempo total hasta convergencia
    'continue_training',
    # ── Overhead de FL ───────────────────────────────────────────────────
    'num_local_train_completed',
    'num_confirmations',
    'all_agents_confirmed',
    # ── KPI: Energía global (servidor agrega por ronda) ──────────────────
    'total_ecomp_j',           # suma Ecomp de todos los workers esta ronda
    'total_ecomm_j',           # suma Ecomm de todos los workers esta ronda
    'total_etotal_j',          # suma Etotal de todos los workers esta ronda
    'avg_etotal_per_worker_j', # promedio por worker
    'cumulative_ealpha_j',     # E_alpha: energía acumulada total hasta esta ronda
    'ealpha_at_convergence_j', # E_alpha en la ronda de convergencia (métrica del paper)
]


class CentralizedServer:
    def __init__(self, host='0.0.0.0', port=8765):
        self.host = host
        self.port = port
        self.agents: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.current_round = 0
        self.min_agents = Config.MIN_AGENTS

        self.model = None
        self.n_features = None

        self.received_models: Dict[str, dict] = {}
        self.received_metrics: Dict[str, dict] = {}   # métricas locales completas

        # Worker arrival timestamps (para straggler)
        self.round_start_time: float = 0.0
        self.worker_arrival_times: Dict[str, float] = {}

        # Early stopping
        self.recall_history = []
        self.best_recall = 0.0
        self.prev_recall = 0.0
        self.rounds_without_improvement = 0
        self.training_stopped = False
        self.best_model_params = None
        self.best_model_round = 0
        self.convergence_time: Optional[float] = None
        self.experiment_start_time: float = 0.0

        # Energía acumulada global (E_alpha)
        self.cumulative_ealpha: float = 0.0
        self.ealpha_at_convergence: float = 0.0

        # Tracking
        self.num_local_train_completed = 0
        self.num_confirmations = 0
        self.dropout_count = 0

        # Último modelo serializado (para métricas de tamaño)
        self.last_model_bytes: int = 0

        # CSV logging
        os.makedirs(Config.LOGS_DIR, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.server_log_file = os.path.join(Config.LOGS_DIR, f'server_centralized_{ts}.csv')
        self._init_csv()
        print(f"[SERVIDOR] Log: {self.server_log_file}")

    # -----------------------------------------------------------------------
    # CSV
    # -----------------------------------------------------------------------
    def _init_csv(self):
        with open(self.server_log_file, 'w', newline='') as f:
            csv.writer(f).writerow(SERVER_CSV_HEADERS)

    def _log_round(self, agg_time: float, round_time: float,
                   agg_params, net_before: dict, net_after: dict,
                   global_metrics: dict,
                   early_stopped: bool, es_reason: str,
                   cpu_before: float, cpu_after: float,
                   ram_before: float, ram_after: float,
                   temp_c: float = -1.0, cpu_freq: float = -1.0):

        elapsed = time.time() - self.experiment_start_time
        delta_recall = global_metrics['recall'] - self.prev_recall

        # Straggler
        arrivals = self.worker_arrival_times
        first_arr = min(arrivals.values()) if arrivals else 0.0
        last_arr = max(arrivals.values()) if arrivals else 0.0
        straggler_delay = last_arr - first_arr
        arrival_order_json = json.dumps(
            {aid: round(t, 4) for aid, t in sorted(arrivals.items(), key=lambda x: x[1])}
        )

        # Comunicaciones
        n = len(self.agents) or 1
        comm_overhead = self.last_model_bytes * (n - 1) * 2  # upload + download
        total_upload = sum(
            m.get('upload_bytes', 0) for m in self.received_metrics.values()
        )
        total_download = sum(
            m.get('download_bytes', 0) for m in self.received_metrics.values()
        )

        # Variance de métricas locales (non-IID indicator)
        recalls = [m.get('recall', 0.0) for m in self.received_metrics.values()]
        accs = [m.get('accuracy', 0.0) for m in self.received_metrics.values()]
        r_var = float(np.var(recalls)) if recalls else 0.0
        r_std = float(np.std(recalls)) if recalls else 0.0
        a_var = float(np.var(accs)) if accs else 0.0

        net_d = net_delta(net_before, net_after)

        # Energía agregada de todos los workers esta ronda
        total_ecomp  = sum(m.get('ecomp_j', 0.0)  for m in self.received_metrics.values() if m.get('ecomp_j', -1) >= 0)
        total_ecomm  = sum(m.get('ecomm_j', 0.0)  for m in self.received_metrics.values() if m.get('ecomm_j', -1) >= 0)
        total_etotal = sum(m.get('etotal_j', 0.0) for m in self.received_metrics.values() if m.get('etotal_j', -1) >= 0)
        n_workers = len(self.received_metrics) or 1
        avg_etotal = round(total_etotal / n_workers, 6) if total_etotal > 0 else 0.0

        if total_etotal > 0:
            self.cumulative_ealpha += total_etotal
        # Guardar E_alpha en el momento de convergencia
        if global_metrics['recall'] > self.best_recall and self.ealpha_at_convergence == 0.0:
            self.ealpha_at_convergence = round(self.cumulative_ealpha, 6)

        conv_round = self.best_model_round if early_stopped else ''
        conv_time = round(self.convergence_time, 4) if self.convergence_time else ''

        row = [
            datetime.now().isoformat(),
            self.current_round,
            Config.AGGREGATION_METHOD,
            'centralized',
            Config.NETWORK_HOPS,
            len(self.agents),
            len(self.received_models),
            # timing
            round(agg_time, 4),
            round(round_time, 4),
            round(elapsed, 4),
            # straggler
            round(first_arr, 4),
            round(last_arr, 4),
            round(straggler_delay, 4),
            arrival_order_json,
            self.dropout_count,
            # comunicaciones
            self.last_model_bytes,
            total_upload,
            total_download,
            comm_overhead,
            net_d['bytes_sent'],
            net_d['bytes_recv'],
            net_d['packets_sent'],
            net_d['packets_recv'],
            # cpu/ram servidor
            round(cpu_before, 2),
            round(cpu_after, 2),
            round(ram_before, 2),
            round(ram_after, 2),
            temp_c,
            cpu_freq,
            # métricas modelo
            round(global_metrics['recall'], 4),
            round(global_metrics['accuracy'], 4),
            round(global_metrics['precision'], 4),
            round(global_metrics['f1'], 4),
            round(global_metrics['loss'], 4),
            round(global_metrics['specificity'], 4),
            # varianza
            round(r_var, 6),
            round(r_std, 6),
            round(a_var, 6),
            # convergencia
            round(self.best_recall, 4),
            round(delta_recall, 4),
            self.rounds_without_improvement,
            # early stopping
            1 if early_stopped else 0,
            es_reason if early_stopped else '',
            Config.EARLY_STOPPING_PATIENCE,
            conv_round,
            conv_time,
            0 if early_stopped else 1,
            # fl overhead
            self.num_local_train_completed,
            self.num_confirmations,
            1 if self.num_confirmations >= len(self.agents) else 0,
            # energía global
            round(total_ecomp, 9),
            round(total_ecomm, 6),
            round(total_etotal, 6),
            avg_etotal,
            round(self.cumulative_ealpha, 6),
            self.ealpha_at_convergence,
        ]

        with open(self.server_log_file, 'a', newline='') as f:
            csv.writer(f).writerow(row)

        self.prev_recall = global_metrics['recall']
        self.num_local_train_completed = 0
        self.num_confirmations = 0
        self.dropout_count = 0

    # -----------------------------------------------------------------------
    # Modelo
    # -----------------------------------------------------------------------
    def initialize_model(self, n_features: int):
        if self.model is None:
            self.n_features = n_features
            self.model = MLP(in_features=n_features, seed=42, p_dropout=0.3)
            print(f"[SERVIDOR] Modelo inicializado con {n_features} features")

    def get_model_parameters(self):
        return {name: param.data.clone() for name, param in self.model.named_parameters()}

    def set_model_parameters(self, parameters):
        for name, param in self.model.named_parameters():
            param.data = parameters[name].clone()

    # -----------------------------------------------------------------------
    # Agregación
    # -----------------------------------------------------------------------
    def federated_averaging(self) -> dict:
        all_models = list(self.received_models.values())
        aggregated = {}
        for pname in all_models[0]:
            psum = torch.zeros_like(all_models[0][pname])
            for m in all_models:
                psum += m[pname]
            aggregated[pname] = psum / len(all_models)
        return aggregated

    # -----------------------------------------------------------------------
    # Métricas globales
    # -----------------------------------------------------------------------
    def compute_global_metrics(self) -> dict:
        keys = ['loss', 'accuracy', 'precision', 'recall', 'f1', 'specificity']
        agg = {k: 0.0 for k in keys}
        n = len(self.received_metrics)
        if n == 0:
            return agg
        for m in self.received_metrics.values():
            for k in keys:
                agg[k] += m.get(k, 0.0)
        return {k: v / n for k, v in agg.items()}

    # -----------------------------------------------------------------------
    # Early stopping
    # -----------------------------------------------------------------------
    def check_early_stopping(self, global_recall: float):
        self.recall_history.append({'round': self.current_round, 'recall': global_recall})

        if self.current_round >= Config.MAX_ROUNDS:
            return True, f"max_rounds_{Config.MAX_ROUNDS}"

        if global_recall > self.best_recall + Config.MIN_RECALL_IMPROVEMENT:
            self.best_recall = global_recall
            self.rounds_without_improvement = 0
            self.best_model_params = self.get_model_parameters()
            self.best_model_round = self.current_round
            if self.convergence_time is None:
                self.convergence_time = time.time() - self.experiment_start_time
            print(f"[SERVIDOR] ✅ Mejora recall → {global_recall:.4f}")
        else:
            self.rounds_without_improvement += 1
            print(f"[SERVIDOR] Sin mejora: {self.rounds_without_improvement}/{Config.EARLY_STOPPING_PATIENCE}")

        if self.rounds_without_improvement >= Config.EARLY_STOPPING_PATIENCE:
            return True, f"early_stopping_patience_{Config.EARLY_STOPPING_PATIENCE}"

        return False, None

    def save_best_model(self):
        if self.best_model_params is None:
            return None
        os.makedirs(Config.MODELS_DIR, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        fname = os.path.join(
            Config.MODELS_DIR,
            f"best_model_centralized_r{self.best_model_round}_"
            f"recall{self.best_recall:.4f}_{ts}.pt"
        )
        torch.save({
            'round': self.best_model_round,
            'recall': self.best_recall,
            'model_state_dict': self.best_model_params,
            'recall_history': self.recall_history,
            'convergence_time_s': self.convergence_time,
        }, fname)
        print(f"[SERVIDOR] 💾 Mejor modelo guardado: {fname}")
        return fname

    # -----------------------------------------------------------------------
    # Comunicación
    # -----------------------------------------------------------------------
    async def start_training_round(self):
        self.current_round += 1
        print(f"\n{'='*60}\n[SERVIDOR] === RONDA {self.current_round} ===\n{'='*60}")

        self.received_models = {}
        self.received_metrics = {}
        self.worker_arrival_times = {}
        self.num_local_train_completed = 0
        self.round_start_time = time.time()

        model_params = self.get_model_parameters()
        params_bytes = pickle.dumps(model_params)
        self.last_model_bytes = len(params_bytes)

        msg = json.dumps({
            'type': 'start_round',
            'round': self.current_round,
            'model_size': self.last_model_bytes
        })

        for agent_id, ws in list(self.agents.items()):
            try:
                await ws.send(msg)
                await ws.send(params_bytes)
                print(f"[SERVIDOR] Modelo enviado a {agent_id} ({self.last_model_bytes} bytes)")
            except Exception as e:
                print(f"[SERVIDOR] Error enviando a {agent_id}: {e}")
                self.dropout_count += 1

    async def register_agent(self, websocket, agent_id):
        self.agents[agent_id] = websocket
        print(f"[SERVIDOR] Agente {agent_id} registrado. Total: {len(self.agents)}")

    async def unregister_agent(self, agent_id):
        self.agents.pop(agent_id, None)
        print(f"[SERVIDOR] Agente {agent_id} desconectado. Total: {len(self.agents)}")

    async def handle_agent(self, websocket, path=None):
        agent_id = None
        try:
            async for message in websocket:
                data = json.loads(message)
                msg_type = data.get('type')

                if msg_type == 'register':
                    agent_id = data['agent_id']
                    n_features = data.get('n_features')
                    await self.register_agent(websocket, agent_id)
                    if self.model is None:
                        self.initialize_model(n_features)
                    await websocket.send(json.dumps({'type': 'registered', 'agent_id': agent_id}))

                    if len(self.agents) >= self.min_agents and self.current_round == 0:
                        self.experiment_start_time = time.time()
                        await asyncio.sleep(Config.STARTUP_DELAY)
                        await self.start_training_round()

                elif msg_type == 'model_update':
                    agent_id = data['agent_id']
                    # Registrar tiempo de llegada (straggler)
                    arrival_offset = time.time() - self.round_start_time
                    self.worker_arrival_times[agent_id] = arrival_offset
                    self.num_local_train_completed += 1
                    self.num_confirmations += 1

                    # Recibir parámetros del modelo
                    model_bytes = await websocket.recv()
                    model_params = pickle.loads(model_bytes)
                    self.received_models[agent_id] = model_params

                    # Recibir métricas extendidas del worker
                    worker_metrics = data.get('metrics', {})
                    # Compatibilidad retroactiva: si metrics no viene, usar local_recall
                    if not worker_metrics:
                        worker_metrics = {
                            'recall': data.get('local_recall', 0.0),
                            'accuracy': 0.0, 'precision': 0.0,
                            'f1': 0.0, 'loss': 0.0, 'specificity': 0.0,
                            'upload_bytes': len(model_bytes),
                            'download_bytes': data.get('download_bytes', 0),
                        }
                    self.received_metrics[agent_id] = worker_metrics

                    print(
                        f"[SERVIDOR] Recibido de {agent_id} "
                        f"(recall={worker_metrics.get('recall', 0):.4f}, "
                        f"arr={arrival_offset:.2f}s) "
                        f"[{len(self.received_models)}/{len(self.agents)}]"
                    )

                    if len(self.received_models) == len(self.agents):
                        await self._aggregate_and_decide()

        except websockets.exceptions.ConnectionClosed:
            self.dropout_count += 1
        except Exception as e:
            print(f"[SERVIDOR] Error: {e}")
            import traceback; traceback.print_exc()
        finally:
            if agent_id:
                await self.unregister_agent(agent_id)

    async def _aggregate_and_decide(self):
        """Agrega modelos, calcula métricas, decide continuar/detener."""
        print(f"\n[SERVIDOR] Todos los modelos recibidos. Agregando...")

        cpu_before = get_cpu_percent()
        ram_before = get_ram_mb()
        temp_c = get_temperature_c()
        cpu_freq = get_cpu_freq_mhz()
        net_before = get_net_snapshot()
        round_start = self.round_start_time

        # Agregación
        t_agg_start = time.time()
        aggregated_params = self.federated_averaging()
        agg_time = time.time() - t_agg_start
        self.set_model_parameters(aggregated_params)

        cpu_after = get_cpu_percent()
        ram_after = get_ram_mb()
        net_after = get_net_snapshot()
        round_total_time = time.time() - round_start

        # Métricas globales
        global_metrics = self.compute_global_metrics()
        print(f"[SERVIDOR] 📊 Global Recall={global_metrics['recall']:.4f}  "
              f"Acc={global_metrics['accuracy']:.4f}  F1={global_metrics['f1']:.4f}  "
              f"Agg={agg_time:.3f}s")

        # Early stopping
        should_stop, reason = self.check_early_stopping(global_metrics['recall'])

        # Log CSV
        self._log_round(
            agg_time=agg_time,
            round_time=round_total_time,
            agg_params=aggregated_params,
            net_before=net_before,
            net_after=net_after,
            global_metrics=global_metrics,
            early_stopped=should_stop,
            es_reason=reason or '',
            cpu_before=cpu_before,
            cpu_after=cpu_after,
            ram_before=ram_before,
            ram_after=ram_after,
            temp_c=temp_c,
            cpu_freq=cpu_freq,
        )

        if should_stop:
            self.training_stopped = True
            print(f"\n[SERVIDOR] 🛑 DETENIDO: {reason} | "
                  f"Total rondas: {self.current_round} | "
                  f"Mejor recall: {self.best_recall:.4f}")
            saved = self.save_best_model()
            stop_msg = json.dumps({
                'type': 'training_stopped',
                'reason': reason,
                'total_rounds': self.current_round,
                'best_recall': self.best_recall,
                'best_model_file': saved or ''
            })
            for ws in list(self.agents.values()):
                try:
                    await ws.send(stop_msg)
                except Exception:
                    pass
        else:
            await asyncio.sleep(Config.ROUND_START_DELAY)
            await self.start_training_round()

    async def start(self):
        print(f"[SERVIDOR] ws://{self.host}:{self.port} | "
              f"Esperando {self.min_agents} agentes...")
        async with websockets.serve(
            self.handle_agent, self.host, self.port,
            max_size=Config.WS_MAX_SIZE
        ):
            await asyncio.Future()


def main():
    server = CentralizedServer(host=Config.SERVER_HOST, port=Config.SERVER_PORT)
    asyncio.run(server.start())


if __name__ == '__main__':
    main()
