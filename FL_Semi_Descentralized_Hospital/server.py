"""
Servidor Coordinador Semi-Descentralizado
- Sorteo de agregador rotativo por lotería
- Early stopping global
- Timeout para rondas colgadas
- KPIs: topología dinámica, convergencia, CPU/RAM/Temp/Freq
"""
import asyncio
import websockets
import json
import random
import os
import csv
import time
from typing import Dict, Optional
from datetime import datetime

from config import Config
from metrics_utils import get_cpu_percent, get_ram_mb, get_temperature_c, get_cpu_freq_mhz


SERVER_CSV_HEADERS = [
    'timestamp', 'round',
    'architecture', 'selection_strategy',
    'aggregation_method',
    'selected_aggregator_id',
    'lottery_results',
    'aggregator_port',
    'num_agents_total', 'num_workers',
    'aggregator_changed',
    'topology_changes_total',
    'round_start_time', 'total_elapsed_time_s',
    'server_cpu_percent', 'server_ram_mb',
    'server_temperature_c', 'server_cpu_freq_mhz',
    'global_recall', 'best_recall_so_far', 'delta_recall',
    'rounds_without_improvement',
    'early_stopping_triggered', 'early_stopping_reason',
    'patience_value', 'convergence_round', 'convergence_time_s',
    'continue_training',
    'num_local_train_completed', 'num_confirmations', 'all_agents_confirmed',
    'dropout_count',
    # Energía global agregada
    'total_ecomp_j', 'total_ecomm_j', 'total_etotal_j',
    'avg_etotal_per_agent_j',
    'cumulative_ealpha_j', 'ealpha_at_convergence_j',
]


class FederatedServer:
    def __init__(self, host='0.0.0.0', port=8765):
        self.host = host
        self.port = port
        self.agents: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.agents_info: Dict[str, dict] = {}
        self.current_round = 0
        self.waiting_for_agents = set()
        self.min_agents = Config.MIN_AGENTS

        self.current_aggregator_id: Optional[str] = None
        self.prev_aggregator_id: Optional[str] = None
        self.topology_changes_total = 0

        self.recall_history = []
        self.best_recall = 0.0
        self.prev_recall = 0.0
        self.rounds_without_improvement = 0
        self.training_stopped = False
        self.best_model_round = 0
        self.best_aggregator_id: Optional[str] = None
        self.convergence_time: Optional[float] = None
        self.experiment_start_time: float = 0.0

        # Energía acumulada global
        self.cumulative_ealpha: float = 0.0
        self.ealpha_at_convergence: float = 0.0
        self.last_energy_report: dict = {}  # último reporte de energía del agregador

        self.num_local_train_completed = 0
        self.num_confirmations = 0
        self.dropout_count = 0

        # Anti-doble-procesamiento
        self._processing_round = -1

        os.makedirs(Config.LOGS_DIR, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(Config.LOGS_DIR, f'server_semidesc_{ts}.csv')
        self._init_csv()
        print(f"[SERVIDOR] Log: {self.log_file}")

    def _init_csv(self):
        with open(self.log_file, 'w', newline='') as f:
            csv.writer(f).writerow(SERVER_CSV_HEADERS)

    def _log_round(self, global_recall: float, lottery_results: dict,
                   early_stopped: bool, es_reason: str):
        elapsed = time.time() - self.experiment_start_time
        delta_recall = global_recall - self.prev_recall
        aggregator_changed = 1 if self.current_aggregator_id != self.prev_aggregator_id else 0

        try:
            agg_port = Config.AGGREGATOR_PORT_BASE + int(
                self.current_aggregator_id.split('_')[1]
            ) if self.current_aggregator_id else -1
        except Exception:
            agg_port = -1

        conv_round = self.best_model_round if early_stopped and self.convergence_time else ''
        conv_time = round(self.convergence_time, 4) if self.convergence_time else ''

        cpu = get_cpu_percent()
        ram = get_ram_mb()
        temp = get_temperature_c()
        freq = get_cpu_freq_mhz()

        # Energía
        e = self.last_energy_report
        total_ecomp  = e.get('total_ecomp_j', 0.0)
        total_ecomm  = e.get('total_ecomm_j', 0.0)
        total_etotal = e.get('total_etotal_j', 0.0)
        n_agents     = e.get('n_agents', 1) or 1
        avg_etotal   = round(total_etotal / n_agents, 6) if total_etotal > 0 else 0.0

        row = [
            datetime.now().isoformat(), self.current_round,
            'semi_decentralized', 'lottery',
            Config.AGGREGATION_METHOD,
            self.current_aggregator_id,
            json.dumps(lottery_results),
            agg_port,
            len(self.agents), len(self.agents) - 1,
            aggregator_changed, self.topology_changes_total,
            datetime.now().isoformat(), round(elapsed, 4),
            round(cpu, 2), round(ram, 2), temp, freq,
            round(global_recall, 4), round(self.best_recall, 4),
            round(delta_recall, 4), self.rounds_without_improvement,
            1 if early_stopped else 0,
            es_reason if early_stopped else '',
            Config.EARLY_STOPPING_PATIENCE, conv_round, conv_time,
            0 if early_stopped else 1,
            self.num_local_train_completed, self.num_confirmations,
            1 if self.num_confirmations >= len(self.agents) else 0,
            self.dropout_count,
            # Energía
            round(total_ecomp, 9), round(total_ecomm, 6), round(total_etotal, 6),
            avg_etotal,
            round(self.cumulative_ealpha, 6), self.ealpha_at_convergence,
        ]
        with open(self.log_file, 'a', newline='') as f:
            csv.writer(f).writerow(row)

        self.prev_recall = global_recall
        self.num_local_train_completed = 0
        self.num_confirmations = 0
        self.dropout_count = 0

    # -----------------------------------------------------------------------
    # Sorteo
    # -----------------------------------------------------------------------
    async def select_aggregator(self):
        print(f"\n[SERVIDOR] === RONDA {self.current_round + 1} – SORTEO ===")
        lottery = {aid: random.randint(1, 100) for aid in self.agents}
        winner = max(lottery, key=lottery.get)
        print(f"[SERVIDOR] 🎲 Resultados: {lottery}")
        print(f"[SERVIDOR] 🏆 Ganador: {winner} ({lottery[winner]})")

        self.prev_aggregator_id = self.current_aggregator_id
        self.current_aggregator_id = winner
        if winner != self.prev_aggregator_id and self.prev_aggregator_id is not None:
            self.topology_changes_total += 1
        return winner, lottery

    async def broadcast_aggregator_info(self, aggregator_id, lottery):
        other_agents = [aid for aid in self.agents if aid != aggregator_id]
        agg_info = self.agents_info[aggregator_id]
        agg_host = agg_info.get('hostname') or agg_info.get('ip', '127.0.0.1')
        if agg_host in ['::1', '::ffff:127.0.0.1', '127.0.0.1']:
            agg_host = agg_info.get('hostname', agg_host)

        try:
            agg_port = Config.AGGREGATOR_PORT_BASE + int(aggregator_id.split('_')[1])
        except Exception:
            agg_port = Config.AGGREGATOR_PORT_BASE + 1

        for agent_id, ws in self.agents.items():
            try:
                if agent_id == aggregator_id:
                    msg = json.dumps({
                        'type': 'aggregator_selected', 'role': 'aggregator',
                        'aggregator_id': aggregator_id,
                        'lottery_results': lottery,
                        'agents_list': other_agents,
                        'round': self.current_round
                    })
                else:
                    msg = json.dumps({
                        'type': 'aggregator_selected', 'role': 'worker',
                        'aggregator_id': aggregator_id,
                        'aggregator_info': {
                            'id': aggregator_id,
                            'host': agg_host,
                            'port': agg_port
                        },
                        'lottery_results': lottery,
                        'round': self.current_round
                    })
                await ws.send(msg)
            except Exception as e:
                print(f"[SERVIDOR] Error notificando a {agent_id}: {e}")

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
            self.best_model_round = self.current_round
            self.best_aggregator_id = self.current_aggregator_id
            if self.convergence_time is None:
                self.convergence_time = time.time() - self.experiment_start_time
            # E_alpha en convergencia
            if self.ealpha_at_convergence == 0.0 and self.cumulative_ealpha > 0:
                self.ealpha_at_convergence = round(self.cumulative_ealpha, 6)
            print(f"[SERVIDOR] ✅ Mejora → {global_recall:.4f}")
        else:
            self.rounds_without_improvement += 1
            print(f"[SERVIDOR] Sin mejora: {self.rounds_without_improvement}/{Config.EARLY_STOPPING_PATIENCE}")

        if self.rounds_without_improvement >= Config.EARLY_STOPPING_PATIENCE:
            return True, f"early_stopping_patience_{Config.EARLY_STOPPING_PATIENCE}"
        return False, None

    # -----------------------------------------------------------------------
    # Registro / baja de agentes
    # -----------------------------------------------------------------------
    async def register_agent(self, websocket, agent_id, hostname=None):
        self.agents[agent_id] = websocket
        ip = websocket.remote_address[0]
        self.agents_info[agent_id] = {
            'id': agent_id, 'connected': True,
            'ip': ip, 'hostname': hostname or agent_id,
            'last_seen': datetime.now().isoformat()
        }
        print(f"[SERVIDOR] {agent_id} registrado (ip={ip}, hostname={hostname}). "
              f"Total: {len(self.agents)}")

    async def unregister_agent(self, agent_id):
        self.agents.pop(agent_id, None)
        self.agents_info.pop(agent_id, None)
        self.dropout_count += 1
        print(f"[SERVIDOR] {agent_id} desconectado. Total: {len(self.agents)}")

    def agent_ready(self, agent_id) -> bool:
        self.waiting_for_agents.discard(agent_id)
        print(f"[SERVIDOR] {agent_id} listo. Faltan: {len(self.waiting_for_agents)}")
        return len(self.waiting_for_agents) == 0

    async def start_new_round(self):
        if len(self.agents) < self.min_agents:
            return
        winner, lottery = await self.select_aggregator()
        await self.broadcast_aggregator_info(winner, lottery)
        self.current_round += 1

    async def wait_for_all_agents(self):
        self.waiting_for_agents = set(self.agents.keys())
        confirm = json.dumps({'type': 'request_confirmation'})
        for ws in self.agents.values():
            try:
                await ws.send(confirm)
            except Exception:
                pass

    # -----------------------------------------------------------------------
    # Handler principal
    # -----------------------------------------------------------------------
    async def handle_agent(self, websocket, path=None):
        agent_id = None
        lottery_results = {}
        try:
            async for message in websocket:
                data = json.loads(message)
                msg_type = data.get('type')

                if msg_type == 'register':
                    agent_id = data['agent_id']
                    hostname = data.get('hostname', agent_id)
                    await self.register_agent(websocket, agent_id, hostname)
                    await websocket.send(json.dumps({
                        'type': 'registered', 'agent_id': agent_id
                    }))
                    if len(self.agents) >= self.min_agents and self.current_round == 0:
                        self.experiment_start_time = time.time()
                        await asyncio.sleep(Config.STARTUP_DELAY)
                        await self.start_new_round()

                elif msg_type == 'ready_for_next_round':
                    agent_id = data.get('agent_id')
                    self.num_confirmations += 1
                    if self.agent_ready(agent_id):
                        if not self.training_stopped:
                            await asyncio.sleep(Config.ROUND_START_DELAY)
                            await self.start_new_round()

                elif msg_type == 'training_complete':
                    self.num_local_train_completed += 1

                elif msg_type == 'aggregation_complete':
                    # Anti-doble
                    if self._processing_round == self.current_round:
                        continue
                    self._processing_round = self.current_round

                    agent_id = data.get('agent_id')
                    global_recall = data.get('global_recall', 0.0)
                    lottery_results = data.get('lottery_results', {})

                    # Energía reportada por el agregador
                    energy = data.get('energy', {})
                    if energy:
                        etotal = energy.get('total_etotal_j', 0.0)
                        self.cumulative_ealpha += etotal
                        self.last_energy_report = energy

                    print(f"[SERVIDOR] Agregador {agent_id} terminó. "
                          f"Recall={global_recall:.4f}")

                    should_stop, reason = self.check_early_stopping(global_recall)
                    self._log_round(global_recall, lottery_results, should_stop, reason or '')

                    if should_stop:
                        self.training_stopped = True
                        print(f"[SERVIDOR] 🛑 DETENIDO: {reason} | "
                              f"Rondas: {self.current_round} | "
                              f"Mejor recall: {self.best_recall:.4f} | "
                              f"Mejor agregador: {self.best_aggregator_id}")

                        if self.best_aggregator_id in self.agents:
                            try:
                                await self.agents[self.best_aggregator_id].send(json.dumps({
                                    'type': 'save_best_model',
                                    'best_round': self.best_model_round,
                                    'best_recall': self.best_recall
                                }))
                            except Exception:
                                pass

                        stop_msg = json.dumps({
                            'type': 'training_stopped',
                            'reason': reason,
                            'total_rounds': self.current_round,
                            'best_recall': self.best_recall,
                            'best_model_round': self.best_model_round,
                            'recall_history': self.recall_history
                        })
                        for ws in list(self.agents.values()):
                            try:
                                await ws.send(stop_msg)
                                await ws.close(1000, "Training finished")
                            except Exception:
                                pass
                        print("[SERVIDOR] ✓ Conexiones cerradas. Ctrl+C para salir.")
                    else:
                        await self.wait_for_all_agents()

        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            print(f"[SERVIDOR] Error: {e}")
            import traceback; traceback.print_exc()
        finally:
            if agent_id:
                await self.unregister_agent(agent_id)

    async def start(self):
        print(f"[SERVIDOR] Semi-desc | ws://{self.host}:{self.port} | "
              f"Esperando {self.min_agents} agentes...")
        async with websockets.serve(
            self.handle_agent, self.host, self.port,
            max_size=Config.WS_MAX_SIZE
        ):
            await asyncio.Future()


def main():
    server = FederatedServer(host=Config.SERVER_HOST, port=Config.SERVER_PORT)
    asyncio.run(server.start())


if __name__ == '__main__':
    main()
