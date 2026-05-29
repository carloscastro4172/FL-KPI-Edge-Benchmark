"""
Agente Semi-Descentralizado – Rol dual worker/agregador
Métodos: FedAvg | FedProx | FedNova
KPIs: CPU/RAM/Temp/Freq, latencia, throughput, energía (Ecomp/Ecomm/Etotal),
      straggler, varianza non-IID, topología dinámica
"""
import asyncio
import websockets
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import pickle
import sys
import os
import joblib
import socket
import time
import csv
from datetime import datetime
from sklearn.metrics import recall_score, precision_score, f1_score

from config import Config
from mlp import MLP
from metrics_utils import (
    get_cpu_percent, get_ram_mb, get_temperature_c, get_cpu_freq_mhz,
    get_net_snapshot, net_delta, get_open_sockets,
    calc_bandwidth_kbps, calc_specificity,
    calc_ecomp, calc_ecomm, calc_etotal,
    ALPHA_CAPACITANCE, C_CYCLES_PER_BIT, P_TRANSMISSION_W
)


# ---------------------------------------------------------------------------
# CSV headers
# ---------------------------------------------------------------------------
WORKER_CSV_HEADERS = [
    'timestamp', 'external_round', 'internal_round', 'client_id',
    'aggregator_id', 'aggregation_method',
    'local_samples', 'local_epochs', 'local_batch_size',
    # recursos
    'cpu_percent_before', 'cpu_percent_after',
    'ram_mb_before', 'ram_mb_after',
    'temperature_c', 'cpu_freq_mhz', 'open_sockets',
    # timing
    'local_train_time_s', 'local_send_time_s', 'local_receive_time_s', 'round_total_time_s',
    # red
    'local_upload_bytes', 'local_download_bytes', 'local_total_bytes',
    'bandwidth_upload_kbps', 'bandwidth_download_kbps',
    'net_bytes_sent', 'net_bytes_recv',
    'net_packets_sent', 'net_packets_recv',
    'net_errin', 'net_errout', 'net_dropin', 'net_dropout',
    # arquitectura
    'network_hops', 'communication_cost_bytes',
    # modelo
    'local_loss', 'local_accuracy', 'local_precision',
    'local_recall', 'local_f1', 'local_specificity', 'local_sensitivity',
    # FedNova
    'fednova_tau',
    # energía
    'ecomp_j', 'ecomm_j', 'etotal_j',
    'ecomp_params', 'ecomm_params',
    'cumulative_etotal_j',
    'status',
]

AGGREGATOR_CSV_HEADERS = [
    'timestamp', 'external_round', 'internal_round', 'aggregator_id',
    'aggregation_method',
    'num_workers_expected', 'num_workers_received', 'num_workers_failed',
    # recursos agregador
    'cpu_percent_before', 'cpu_percent_after',
    'ram_mb_before', 'ram_mb_after',
    'temperature_c', 'cpu_freq_mhz',
    # timing
    'aggregation_time_s', 'round_total_time_s',
    'first_worker_arrival_s', 'last_worker_arrival_s',
    'straggler_delay_s', 'worker_arrival_order',
    # red
    'net_bytes_sent', 'net_bytes_recv',
    'net_packets_sent', 'net_packets_recv',
    # modelo global
    'model_size_bytes',
    'global_loss', 'global_accuracy', 'global_precision',
    'global_recall', 'global_f1', 'global_specificity',
    # non-IID
    'recall_variance', 'recall_std', 'accuracy_variance',
    # comunicación
    'total_upload_bytes', 'comm_overhead_bytes', 'network_hops',
    # energía del nodo agregador
    'agg_ecomp_j', 'agg_ecomm_j', 'agg_etotal_j',
    # energía agregada de todos los workers
    'total_workers_ecomp_j', 'total_workers_ecomm_j', 'total_workers_etotal_j',
    'cumulative_agg_etotal_j',
]


class FederatedAgentCSV:
    def __init__(self, agent_id: str, csv_file: str,
                 server_uri: str = 'ws://localhost:8765'):
        self.agent_id = agent_id
        self.csv_file = csv_file
        self.server_uri = server_uri
        self.server_ws = None
        self.role = None

        # Modelo
        self.model = None
        self.optimizer = None
        self.criterion = nn.BCEWithLogitsLoss()
        self.train_data = None
        self.val_data = None
        self.n_features = None
        self.global_model_params = None

        # FedNova
        self.fednova_tau: int = 0
        self.fednova_grad: dict = {}

        # Energía acumulada
        self.cumulative_etotal_worker: float = 0.0
        self.cumulative_etotal_agg: float = 0.0

        # Agregador
        self.aggregator_server = None
        self.aggregator_info = None
        self.received_models: dict = {}
        self.received_recalls: dict = {}
        self.received_metrics: dict = {}
        self.received_nova_grads: dict = {}
        self.worker_arrival_times: dict = {}
        self.expected_agents: list = []
        self.aggregated_this_round = False
        self.final_aggregated_params = None
        self.models_sent_count = 0
        self.agg_round_start_time: float = 0.0

        self.internal_round_count = 0
        self.max_internal_rounds = Config.INTERNAL_ROUNDS
        self.current_aggregator_id = None

        try:
            self.aggregator_port = Config.AGGREGATOR_PORT_BASE + int(agent_id.split('_')[1])
        except Exception:
            self.aggregator_port = Config.AGGREGATOR_PORT_BASE + 1

        self.current_round = 0
        self.train_start_time: float = 0.0

        # Logs
        os.makedirs(Config.LOGS_DIR, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.worker_log    = os.path.join(Config.LOGS_DIR, f'worker_{agent_id}_{ts}.csv')
        self.agg_log       = os.path.join(Config.LOGS_DIR, f'aggregator_{agent_id}_{ts}.csv')
        self._init_csv_logs()

    # -----------------------------------------------------------------------
    # CSV
    # -----------------------------------------------------------------------
    def _init_csv_logs(self):
        with open(self.worker_log, 'w', newline='') as f:
            csv.writer(f).writerow(WORKER_CSV_HEADERS)
        with open(self.agg_log, 'w', newline='') as f:
            csv.writer(f).writerow(AGGREGATOR_CSV_HEADERS)

    def _log_worker(self, row: dict):
        with open(self.worker_log, 'a', newline='') as f:
            csv.writer(f).writerow([row.get(h, '') for h in WORKER_CSV_HEADERS])

    def _log_aggregator(self, row: dict):
        with open(self.agg_log, 'a', newline='') as f:
            csv.writer(f).writerow([row.get(h, '') for h in AGGREGATOR_CSV_HEADERS])

    # -----------------------------------------------------------------------
    # Datos
    # -----------------------------------------------------------------------
    def load_and_preprocess_data(self):
        print(f"\n{'='*60}\n[{self.agent_id}] CARGANDO DATOS\n{'='*60}")
        df = pd.read_csv(self.csv_file)

        preprocessor = joblib.load(Config.PREPROCESSOR_PATH)

        target_col = None
        for c in ['is_premature_ncd', 'Classification', 'target', 'label', 'y']:
            if c in df.columns:
                target_col = c; break
        if target_col is None:
            target_col = df.columns[-1]

        y = df[target_col].values
        cols_drop = [target_col] + [c for c in ['hospital_cliente', 'ncd_group']
                                     if c in df.columns]
        X = df.drop(cols_drop, axis=1)
        X_t = preprocessor.transform(X)
        self.n_features = X_t.shape[1]

        X_ten = torch.FloatTensor(np.array(X_t))
        y_ten = torch.FloatTensor(y).unsqueeze(1)
        dataset = TensorDataset(X_ten, y_ten)
        train_sz = int(0.8 * len(dataset))
        val_sz = len(dataset) - train_sz
        tr_ds, val_ds = torch.utils.data.random_split(
            dataset, [train_sz, val_sz],
            generator=torch.Generator().manual_seed(42)
        )
        self.train_data = DataLoader(tr_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
        self.val_data   = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False)

        self.model     = MLP(in_features=self.n_features, seed=42, p_dropout=0.3)
        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)
        print(f"[{self.agent_id}] Features={self.n_features} | "
              f"Train={train_sz} | Val={val_sz} | "
              f"Distrib={np.bincount(y.astype(int))}")

    # -----------------------------------------------------------------------
    # Entrenamiento — FedAvg | FedProx | FedNova
    # -----------------------------------------------------------------------
    def train_local(self, epochs=None) -> float:
        if epochs is None:
            epochs = Config.LOCAL_EPOCHS
        method = Config.AGGREGATION_METHOD
        self.model.train()

        # FedNova: guardar pesos iniciales
        w_init = None
        if method == 'FedNova':
            w_init = {n: p.data.clone() for n, p in self.model.named_parameters()}

        tau = 0
        t0 = time.time()
        for _ in range(epochs):
            for bx, by in self.train_data:
                self.optimizer.zero_grad()
                out  = self.model(bx)
                loss = self.criterion(out, by)

                if method == 'FedProx' and self.global_model_params is not None:
                    prox = sum(
                        ((p - self.global_model_params[n]) ** 2).sum()
                        for n, p in self.model.named_parameters()
                    )
                    loss += (Config.FEDPROX_MU / 2) * prox

                loss.backward()
                self.optimizer.step()
                tau += 1

        # FedNova: gradiente normalizado d_i = (w_init - w_local) / tau
        if method == 'FedNova' and w_init is not None and tau > 0:
            self.fednova_tau  = tau
            self.fednova_grad = {
                n: (w_init[n] - p.data.clone()) / tau
                for n, p in self.model.named_parameters()
            }
        else:
            self.fednova_tau  = tau
            self.fednova_grad = {}

        return time.time() - t0

    # -----------------------------------------------------------------------
    # Métricas
    # -----------------------------------------------------------------------
    def calculate_all_metrics(self) -> dict:
        if not self.val_data:
            return {k: 0.0 for k in
                    ['loss', 'accuracy', 'precision', 'recall', 'f1', 'specificity']}
        self.model.eval()
        total_loss, preds, labels = 0.0, [], []
        with torch.no_grad():
            for bx, by in self.val_data:
                out  = self.model(bx)
                total_loss += self.criterion(out, by).item()
                pred = (torch.sigmoid(out) > 0.5).float()
                preds.extend(pred.cpu().numpy().flatten())
                labels.extend(by.cpu().numpy().flatten())
        return {
            'loss':        total_loss / len(self.val_data),
            'accuracy':    sum(p == l for p, l in zip(preds, labels)) / len(labels),
            'precision':   precision_score(labels, preds, average='macro', zero_division=0),
            'recall':      recall_score(labels, preds, average='macro', zero_division=0),
            'f1':          f1_score(labels, preds, average='macro', zero_division=0),
            'specificity': calc_specificity(labels, preds),
        }

    def get_model_parameters(self):
        return {n: p.data.clone() for n, p in self.model.named_parameters()}

    def set_model_parameters(self, params):
        for n, p in self.model.named_parameters():
            p.data = params[n].clone()

    # -----------------------------------------------------------------------
    # Agregación — dispatcher FedAvg | FedProx | FedNova
    # -----------------------------------------------------------------------
    def _fedavg(self) -> dict:
        models = list(self.received_models.values())
        agg = {}
        for pname in models[0]:
            s = torch.zeros_like(models[0][pname])
            for m in models: s += m[pname]
            agg[pname] = s / len(models)
        return agg

    def _fedprox(self) -> dict:
        return self._fedavg()  # proximal term se aplica en el worker

    def _fednova(self) -> dict:
        current = self.get_model_parameters()
        total_samples = sum(
            m.get('local_samples', 1) for m in self.received_metrics.values()
        )
        correction = {pname: torch.zeros_like(p) for pname, p in current.items()}
        has_nova = any(aid in self.received_nova_grads for aid in self.received_models)

        if has_nova:
            for agent_id in self.received_models:
                m     = self.received_metrics.get(agent_id, {})
                p_i   = m.get('local_samples', 1) / max(total_samples, 1)
                tau_i = max(m.get('fednova_tau', 1), 1)
                nova_g = self.received_nova_grads.get(agent_id, {})
                for pname in correction:
                    if pname in nova_g:
                        correction[pname] += p_i * tau_i * nova_g[pname]
            return {pname: current[pname] - correction[pname] for pname in current}
        else:
            print(f"[{self.agent_id}] ⚠️ FedNova sin gradientes → fallback FedAvg")
            return self._fedavg()

    def aggregate(self) -> dict:
        method = Config.AGGREGATION_METHOD
        if method == 'FedNova':
            result = self._fednova()
        elif method == 'FedProx':
            result = self._fedprox()
        else:
            result = self._fedavg()
        self.final_aggregated_params = result
        return result

    def save_best_model(self, best_round, best_recall):
        os.makedirs(Config.MODELS_DIR, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        fname = os.path.join(
            Config.MODELS_DIR,
            f"best_model_semidesc_{self.agent_id}_{Config.AGGREGATION_METHOD}"
            f"_r{best_round}_recall{best_recall:.4f}_{ts}.pt"
        )
        params = (self.final_aggregated_params
                  if self.final_aggregated_params is not None
                  else self.get_model_parameters())
        torch.save({
            'agent_id': self.agent_id, 'round': best_round,
            'recall': best_recall, 'aggregation_method': Config.AGGREGATION_METHOD,
            'model_state_dict': params
        }, fname)
        print(f"[{self.agent_id}] 💾 Guardado: {fname}")
        return fname

    # -----------------------------------------------------------------------
    # Worker: enviar modelo al agregador
    # -----------------------------------------------------------------------
    async def send_model_to_aggregator(self, aggregator_info: dict,
                                       internal_round: int) -> bool:
        uri = f"ws://{aggregator_info['host']}:{aggregator_info['port']}"

        cpu_before = get_cpu_percent()
        ram_before = get_ram_mb()
        temp_c     = get_temperature_c()
        cpu_freq   = get_cpu_freq_mhz()
        sockets    = get_open_sockets()
        net_before = get_net_snapshot()

        metrics_local = self.calculate_all_metrics()

        # Energía de cómputo
        train_samples = len(self.train_data.dataset)
        ecomp = calc_ecomp(train_samples, self.n_features, cpu_freq)

        for attempt in range(Config.MAX_RETRIES):
            try:
                t_round = time.time()
                async with websockets.connect(
                    uri, max_size=Config.WS_MAX_SIZE,
                    open_timeout=Config.WS_TIMEOUT
                ) as ws:
                    params       = self.get_model_parameters()
                    params_bytes = pickle.dumps(params)
                    upload_bytes = len(params_bytes)

                    t_send = time.time()
                    # Enviar metadatos + métricas + tau FedNova
                    await ws.send(json.dumps({
                        'type': 'model_update',
                        'agent_id': self.agent_id,
                        'params_size': upload_bytes,
                        'internal_round': internal_round,
                        'local_recall': metrics_local['recall'],
                        'metrics': {
                            **metrics_local,
                            'local_samples': train_samples,
                            'fednova_tau': self.fednova_tau,
                            'ecomp_j': ecomp,
                        },
                    }))
                    # FedNova: enviar gradiente normalizado o placeholder
                    if Config.AGGREGATION_METHOD == 'FedNova' and self.fednova_grad:
                        await ws.send(pickle.dumps(self.fednova_grad))
                    else:
                        await ws.send(b'')

                    await ws.send(params_bytes)
                    send_time  = time.time() - t_send
                    upload_bw  = calc_bandwidth_kbps(upload_bytes, send_time)

                    # Recibir modelo agregado
                    t_recv = time.time()
                    response   = await ws.recv()
                    recv_time  = time.time() - t_recv

                    if isinstance(response, bytes) and len(response) > 0:
                        download_bytes = len(response)
                        agg_params     = pickle.loads(response)
                        self.set_model_parameters(agg_params)
                        self.global_model_params = {
                            n: p.clone() for n, p in agg_params.items()
                        }
                        download_bw = calc_bandwidth_kbps(download_bytes, recv_time)
                    else:
                        download_bytes, download_bw = 0, 0.0

                    # Energía de comunicación
                    ecomm  = calc_ecomm(send_time)
                    etotal = calc_etotal(ecomp, ecomm)
                    if etotal > 0:
                        self.cumulative_etotal_worker += etotal

                    train_time  = time.time() - self.train_start_time
                    round_total = time.time() - t_round

                    cpu_after  = get_cpu_percent()
                    ram_after  = get_ram_mb()
                    net_after  = get_net_snapshot()
                    nd         = net_delta(net_before, net_after)
                    comm_cost  = upload_bytes * (Config.NUM_SILOS - 1)

                    self._log_worker({
                        'timestamp':              datetime.now().isoformat(),
                        'external_round':         self.current_round,
                        'internal_round':         internal_round,
                        'client_id':              self.agent_id,
                        'aggregator_id':          self.current_aggregator_id,
                        'aggregation_method':     Config.AGGREGATION_METHOD,
                        'local_samples':          train_samples,
                        'local_epochs':           Config.LOCAL_EPOCHS,
                        'local_batch_size':       Config.BATCH_SIZE,
                        'cpu_percent_before':     round(cpu_before, 2),
                        'cpu_percent_after':      round(cpu_after, 2),
                        'ram_mb_before':          round(ram_before, 2),
                        'ram_mb_after':           round(ram_after, 2),
                        'temperature_c':          temp_c,
                        'cpu_freq_mhz':           cpu_freq,
                        'open_sockets':           sockets,
                        'local_train_time_s':     round(train_time, 4),
                        'local_send_time_s':      round(send_time, 4),
                        'local_receive_time_s':   round(recv_time, 4),
                        'round_total_time_s':     round(round_total, 4),
                        'local_upload_bytes':     upload_bytes,
                        'local_download_bytes':   download_bytes,
                        'local_total_bytes':      upload_bytes + download_bytes,
                        'bandwidth_upload_kbps':  round(upload_bw, 4),
                        'bandwidth_download_kbps':round(download_bw, 4),
                        'net_bytes_sent':         nd['bytes_sent'],
                        'net_bytes_recv':         nd['bytes_recv'],
                        'net_packets_sent':       nd['packets_sent'],
                        'net_packets_recv':       nd['packets_recv'],
                        'net_errin':              nd['errin'],
                        'net_errout':             nd['errout'],
                        'net_dropin':             nd['dropin'],
                        'net_dropout':            nd['dropout'],
                        'network_hops':           Config.NETWORK_HOPS,
                        'communication_cost_bytes': comm_cost,
                        'local_loss':             round(metrics_local['loss'], 4),
                        'local_accuracy':         round(metrics_local['accuracy'], 4),
                        'local_precision':        round(metrics_local['precision'], 4),
                        'local_recall':           round(metrics_local['recall'], 4),
                        'local_f1':               round(metrics_local['f1'], 4),
                        'local_specificity':      round(metrics_local['specificity'], 4),
                        'local_sensitivity':      round(metrics_local['recall'], 4),
                        'fednova_tau':            self.fednova_tau,
                        'ecomp_j':                ecomp,
                        'ecomm_j':                ecomm,
                        'etotal_j':               etotal,
                        'ecomp_params':           json.dumps({'alpha': ALPHA_CAPACITANCE,
                                                              'c': C_CYCLES_PER_BIT,
                                                              'D_bits': train_samples * self.n_features * 32,
                                                              'f_hz': cpu_freq * 1e6 if cpu_freq > 0 else -1}),
                        'ecomm_params':           json.dumps({'p_w': P_TRANSMISSION_W, 'tcomm_s': round(send_time, 4)}),
                        'cumulative_etotal_j':    round(self.cumulative_etotal_worker, 6),
                        'status':                 'OK',
                    })
                    print(f"[{self.agent_id}] ⚡ Ecomp={ecomp:.3e}J "
                          f"Ecomm={ecomm:.4f}J Etotal={etotal:.4f}J "
                          f"Acum={self.cumulative_etotal_worker:.4f}J")
                    return True

            except Exception as e:
                if attempt < Config.MAX_RETRIES - 1:
                    await asyncio.sleep(Config.RETRY_DELAY)
                else:
                    print(f"[{self.agent_id}] ❌ Error: {e}")
                    self._log_worker({
                        'timestamp': datetime.now().isoformat(),
                        'external_round': self.current_round,
                        'internal_round': internal_round,
                        'client_id': self.agent_id,
                        'aggregator_id': self.current_aggregator_id,
                        'aggregation_method': Config.AGGREGATION_METHOD,
                        'status': 'FAILED',
                    })
                    return False
        return False

    # -----------------------------------------------------------------------
    # Agregador: recibir modelos
    # -----------------------------------------------------------------------
    async def _timeout_watchdog(self):
        """Si pasan ROUND_TIMEOUT segundos sin todos los modelos, agrega con lo que hay."""
        await asyncio.sleep(Config.ROUND_TIMEOUT)
        pending = len(self.expected_agents) - len(self.received_models)
        if pending > 0 and not self.aggregated_this_round:
            print(f"[{self.agent_id}] ⏰ TIMEOUT: {len(self.received_models)}/"
                  f"{len(self.expected_agents)} modelos. Agregando sin los {pending} faltantes.")

    async def handle_worker_connection(self, websocket, path=None):
        worker_id = None
        try:
            meta_raw = await websocket.recv()
            data     = json.loads(meta_raw)

            if data.get('type') != 'model_update':
                return

            worker_id        = data['agent_id']
            internal_round   = data.get('internal_round', 1)
            worker_metrics   = data.get('metrics', {
                'recall': data.get('local_recall', 0.0),
                'accuracy': 0.0, 'precision': 0.0,
                'f1': 0.0, 'loss': 0.0, 'specificity': 0.0,
            })

            arrival_offset = time.time() - self.agg_round_start_time
            self.worker_arrival_times[worker_id] = arrival_offset

            # FedNova: recibir gradiente o placeholder
            nova_or_placeholder = await websocket.recv()
            if Config.AGGREGATION_METHOD == 'FedNova' and nova_or_placeholder:
                try:
                    self.received_nova_grads[worker_id] = pickle.loads(nova_or_placeholder)
                except Exception:
                    self.received_nova_grads[worker_id] = {}

            params_bytes = await websocket.recv()
            self.received_models[worker_id]  = pickle.loads(params_bytes)
            self.received_recalls[worker_id] = worker_metrics.get('recall', 0.0)
            self.received_metrics[worker_id] = worker_metrics

            print(f"[{self.agent_id}] ← {worker_id} "
                  f"(recall={worker_metrics.get('recall', 0):.4f}, "
                  f"arr={arrival_offset:.2f}s) "
                  f"[{len(self.received_models)}/{len(self.expected_agents)}]")

            # Esperar todos (con timeout implícito via asyncio.wait_for)
            deadline = self.agg_round_start_time + Config.ROUND_TIMEOUT
            while (len(self.received_models) < len(self.expected_agents)
                   and time.time() < deadline):
                await asyncio.sleep(0.3)

            if not self.aggregated_this_round:
                self.aggregated_this_round = True
                self.internal_round_count += 1

                cpu_before   = get_cpu_percent()
                ram_before   = get_ram_mb()
                temp_c       = get_temperature_c()
                cpu_freq     = get_cpu_freq_mhz()
                net_before_a = get_net_snapshot()

                t_agg     = time.time()
                agg_params = self.aggregate()
                agg_time   = time.time() - t_agg

                self.set_model_parameters(agg_params)
                agg_metrics = self.calculate_all_metrics()

                cpu_after   = get_cpu_percent()
                ram_after   = get_ram_mb()
                net_after_a = get_net_snapshot()
                nd          = net_delta(net_before_a, net_after_a)
                round_total = time.time() - self.agg_round_start_time

                arrivals   = self.worker_arrival_times
                first_arr  = min(arrivals.values()) if arrivals else 0.0
                last_arr   = max(arrivals.values()) if arrivals else 0.0
                arr_order  = json.dumps(
                    {aid: round(t, 4) for aid, t in
                     sorted(arrivals.items(), key=lambda x: x[1])}
                )

                recalls   = list(self.received_recalls.values())
                accs      = [m.get('accuracy', 0.0) for m in self.received_metrics.values()]
                r_var     = float(np.var(recalls)) if recalls else 0.0
                r_std     = float(np.std(recalls)) if recalls else 0.0
                a_var     = float(np.var(accs))    if accs    else 0.0

                model_size   = len(pickle.dumps(agg_params))
                total_upload = len(params_bytes) * len(self.received_models)
                comm_overhead = model_size * len(self.expected_agents)

                # Energía del nodo agregador (cómputo de agregación)
                agg_ecomp  = calc_ecomp(
                    sum(m.get('local_samples', 1) for m in self.received_metrics.values()),
                    self.n_features, cpu_freq
                )
                agg_ecomm  = calc_ecomm(agg_time)
                agg_etotal = calc_etotal(agg_ecomp, agg_ecomm)
                self.cumulative_etotal_agg += agg_etotal if agg_etotal > 0 else 0

                # Energía total de todos los workers
                total_w_ecomp  = sum(m.get('ecomp_j', 0.0) for m in self.received_metrics.values() if m.get('ecomp_j', -1) >= 0)
                total_w_ecomm  = sum(m.get('ecomm_j', 0.0) for m in self.received_metrics.values() if m.get('ecomm_j', -1) >= 0)
                # ecomm de workers viene en los metrics si fue enviado desde worker
                total_w_etotal = total_w_ecomp + total_w_ecomm

                self._log_aggregator({
                    'timestamp':              datetime.now().isoformat(),
                    'external_round':         self.current_round,
                    'internal_round':         self.internal_round_count,
                    'aggregator_id':          self.agent_id,
                    'aggregation_method':     Config.AGGREGATION_METHOD,
                    'num_workers_expected':   len(self.expected_agents),
                    'num_workers_received':   len(self.received_models),
                    'num_workers_failed':     len(self.expected_agents) - len(self.received_models),
                    'cpu_percent_before':     round(cpu_before, 2),
                    'cpu_percent_after':      round(cpu_after, 2),
                    'ram_mb_before':          round(ram_before, 2),
                    'ram_mb_after':           round(ram_after, 2),
                    'temperature_c':          temp_c,
                    'cpu_freq_mhz':           cpu_freq,
                    'aggregation_time_s':     round(agg_time, 4),
                    'round_total_time_s':     round(round_total, 4),
                    'first_worker_arrival_s': round(first_arr, 4),
                    'last_worker_arrival_s':  round(last_arr, 4),
                    'straggler_delay_s':      round(last_arr - first_arr, 4),
                    'worker_arrival_order':   arr_order,
                    'net_bytes_sent':         nd['bytes_sent'],
                    'net_bytes_recv':         nd['bytes_recv'],
                    'net_packets_sent':       nd['packets_sent'],
                    'net_packets_recv':       nd['packets_recv'],
                    'model_size_bytes':       model_size,
                    'global_loss':            round(agg_metrics['loss'], 4),
                    'global_accuracy':        round(agg_metrics['accuracy'], 4),
                    'global_precision':       round(agg_metrics['precision'], 4),
                    'global_recall':          round(agg_metrics['recall'], 4),
                    'global_f1':              round(agg_metrics['f1'], 4),
                    'global_specificity':     round(agg_metrics['specificity'], 4),
                    'recall_variance':        round(r_var, 6),
                    'recall_std':             round(r_std, 6),
                    'accuracy_variance':      round(a_var, 6),
                    'total_upload_bytes':     total_upload,
                    'comm_overhead_bytes':    comm_overhead,
                    'network_hops':           Config.NETWORK_HOPS,
                    'agg_ecomp_j':            agg_ecomp,
                    'agg_ecomm_j':            agg_ecomm,
                    'agg_etotal_j':           agg_etotal,
                    'total_workers_ecomp_j':  round(total_w_ecomp, 9),
                    'total_workers_ecomm_j':  round(total_w_ecomm, 6),
                    'total_workers_etotal_j': round(total_w_etotal, 6),
                    'cumulative_agg_etotal_j':round(self.cumulative_etotal_agg, 6),
                })

                print(f"[{self.agent_id}] 📊 [{Config.AGGREGATION_METHOD}] "
                      f"Ronda interna {self.internal_round_count} | "
                      f"Recall={agg_metrics['recall']:.4f} | Agg={agg_time:.3f}s")

            # Enviar modelo agregado
            out_bytes = pickle.dumps(self.final_aggregated_params)
            await websocket.send(out_bytes)
            self.models_sent_count += 1

            if self.models_sent_count >= len(self.expected_agents):
                if self.internal_round_count >= self.max_internal_rounds:
                    global_recall = (sum(self.received_recalls.values()) /
                                     len(self.received_recalls)) if self.received_recalls else 0.0
                    await self.notify_server_aggregation_complete(global_recall)
                else:
                    self.agg_round_start_time = time.time()
                    self.received_models      = {}
                    self.received_recalls     = {}
                    self.received_metrics     = {}
                    self.received_nova_grads  = {}
                    self.worker_arrival_times = {}
                    self.aggregated_this_round = False
                    self.models_sent_count    = 0

        except Exception as e:
            print(f"[{self.agent_id}] Error con worker {worker_id}: {e}")
            import traceback; traceback.print_exc()

    async def notify_server_aggregation_complete(self, global_recall: float):
        if self.server_ws:
            await self.server_ws.send(json.dumps({
                'type': 'aggregation_complete',
                'agent_id': self.agent_id,
                'internal_rounds_completed': self.internal_round_count,
                'global_recall': global_recall,
                'lottery_results': {},
                'energy': {
                    'total_ecomp_j':  0.0,
                    'total_ecomm_j':  0.0,
                    'total_etotal_j': round(self.cumulative_etotal_agg, 6),
                    'n_agents': len(self.expected_agents),
                }
            }))

    async def start_aggregator_server(self):
        if self.aggregator_server:
            self.aggregator_server.close()
            await self.aggregator_server.wait_closed()
        self.aggregator_server = await websockets.serve(
            self.handle_worker_connection, '0.0.0.0', self.aggregator_port,
            max_size=Config.WS_MAX_SIZE
        )
        self.agg_round_start_time = time.time()
        asyncio.create_task(self._timeout_watchdog())
        print(f"[{self.agent_id}] 🔌 Servidor agregador [{Config.AGGREGATION_METHOD}] "
              f"en 0.0.0.0:{self.aggregator_port}")

    # -----------------------------------------------------------------------
    # Mensajes del servidor coordinador
    # -----------------------------------------------------------------------
    async def connect_to_server(self):
        print(f"[{self.agent_id}] Conectando a {self.server_uri}...")
        for attempt in range(Config.MAX_RETRIES):
            try:
                async with websockets.connect(
                    self.server_uri,
                    max_size=Config.WS_MAX_SIZE,
                    open_timeout=Config.WS_TIMEOUT
                ) as ws:
                    self.server_ws = ws
                    await ws.send(json.dumps({
                        'type': 'register',
                        'agent_id': self.agent_id,
                        'hostname': socket.gethostname(),
                    }))
                    async for msg in ws:
                        data = json.loads(msg)
                        await self._handle_server_message(data)
                    break
            except (ConnectionRefusedError, OSError):
                if attempt < Config.MAX_RETRIES - 1:
                    await asyncio.sleep(Config.RETRY_DELAY)
                else:
                    print(f"[{self.agent_id}] ❌ No pudo conectar al servidor")

    async def _handle_server_message(self, data: dict):
        msg_type = data.get('type')

        if msg_type == 'registered':
            print(f"[{self.agent_id}] ✓ Registrado | método: {Config.AGGREGATION_METHOD}")

        elif msg_type == 'aggregator_selected':
            self.role         = data['role']
            self.current_round = data.get('round', 0) + 1
            print(f"\n{'='*60}")
            print(f"[{self.agent_id}] RONDA {self.current_round} – "
                  f"ROL: {self.role.upper()} | {Config.AGGREGATION_METHOD}")
            print(f"{'='*60}")

            if self.role == 'aggregator':
                self.expected_agents       = data.get('agents_list', [])
                self.current_aggregator_id = self.agent_id
                self.received_models       = {}
                self.received_recalls      = {}
                self.received_metrics      = {}
                self.received_nova_grads   = {}
                self.worker_arrival_times  = {}
                self.internal_round_count  = 0
                self.aggregated_this_round = False
                self.models_sent_count     = 0
                self.final_aggregated_params = None
                print(f"[{self.agent_id}] 🏆 AGREGADOR | Esperando: {self.expected_agents}")
                await self.start_aggregator_server()
            else:
                self.aggregator_info       = data['aggregator_info']
                self.current_aggregator_id = data['aggregator_id']
                self.internal_round_count  = 0
                print(f"[{self.agent_id}] WORKER | Agregador: {self.current_aggregator_id}")
                await self.server_ws.send(json.dumps({
                    'type': 'training_complete', 'agent_id': self.agent_id
                }))
                asyncio.create_task(self._worker_internal_loop())

        elif msg_type == 'request_confirmation':
            await asyncio.sleep(1)
            await self._ready_for_next_round()

        elif msg_type == 'save_best_model':
            self.save_best_model(data.get('best_round', 0), data.get('best_recall', 0.0))

        elif msg_type == 'training_stopped':
            print(
                f"\n[{self.agent_id}] 🛑 DETENIDO\n"
                f"  Método: {Config.AGGREGATION_METHOD}\n"
                f"  Razón: {data.get('reason')}\n"
                f"  Rondas: {data.get('total_rounds')}\n"
                f"  Mejor recall: {data.get('best_recall', 0.0):.4f}\n"
                f"[{self.agent_id}] ✓ Agente finalizado."
            )
            if self.aggregator_server:
                self.aggregator_server.close()
                await self.aggregator_server.wait_closed()
            if self.server_ws:
                await self.server_ws.close(1000, "Training finished")

    async def _worker_internal_loop(self):
        for i in range(1, self.max_internal_rounds + 1):
            print(f"\n[{self.agent_id}] Ronda interna {i}/{self.max_internal_rounds} "
                  f"[{Config.AGGREGATION_METHOD}]")
            self.train_start_time = time.time()
            self.train_local()
            ok = await self.send_model_to_aggregator(self.aggregator_info, i)
            if not ok:
                print(f"[{self.agent_id}] ⚠️ Error ronda interna {i}")
                break
            if i < self.max_internal_rounds:
                await asyncio.sleep(2)
        await self._ready_for_next_round()

    async def _ready_for_next_round(self):
        if self.role == 'aggregator' and self.aggregator_server:
            self.aggregator_server.close()
            await self.aggregator_server.wait_closed()
            self.aggregator_server = None
        if self.server_ws:
            await self.server_ws.send(json.dumps({
                'type': 'ready_for_next_round', 'agent_id': self.agent_id
            }))

    async def start(self):
        self.load_and_preprocess_data()
        await self.connect_to_server()


def main():
    if len(sys.argv) < 3:
        print("Uso: python agent_csv.py <agent_id> <csv_file> [server_uri]")
        print("Ejemplo: python agent_csv.py agent_1 data/silos/silo_1.csv ws://172.23.207.115:8765")
        sys.exit(1)

    agent_id   = sys.argv[1]
    csv_file   = sys.argv[2]
    server_uri = sys.argv[3] if len(sys.argv) > 3 else 'ws://localhost:8765'

    agent = FederatedAgentCSV(agent_id, csv_file, server_uri)
    try:
        asyncio.run(agent.start())
    except KeyboardInterrupt:
        print(f"\n[{agent_id}] Detenido por usuario")
    except Exception as e:
        print(f"[{agent_id}] Error: {e}")
        import traceback; traceback.print_exc()


if __name__ == '__main__':
    main()
