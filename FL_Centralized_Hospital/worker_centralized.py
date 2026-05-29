"""
Worker Centralizado – KPIs extendidos para paper académico.
Captura: CPU/RAM, temperatura, red (psutil), latencia, throughput,
accuracy/loss/F1/precision/recall/especificidad por ronda.
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
import time
import csv
from datetime import datetime
from sklearn.metrics import recall_score, precision_score, f1_score

from config import Config
from mlp import MLP
from data_preparation import load_preprocessor
from metrics_utils import (
    get_cpu_percent, get_ram_mb, get_temperature_c, get_cpu_freq_mhz,
    get_net_snapshot, net_delta, get_open_sockets,
    calc_bandwidth_kbps, calc_specificity,
    calc_ecomp, calc_ecomm, calc_etotal,
    ALPHA_CAPACITANCE, C_CYCLES_PER_BIT, P_TRANSMISSION_W
)


# ---------------------------------------------------------------------------
# CSV headers del worker
# ---------------------------------------------------------------------------
WORKER_CSV_HEADERS = [
    # Identificación
    'timestamp', 'round', 'client_id',
    # Dataset local
    'local_samples', 'local_epochs', 'local_batch_size',
    # ── KPI: Recursos computacionales ───────────────────────────────────
    'cpu_percent_before',
    'cpu_percent_after',
    'ram_mb_before',
    'ram_mb_after',
    'temperature_c',
    'cpu_freq_mhz',                # frecuencia actual (detecta throttling en RPi)
    'open_sockets',
    # ── KPI: Timing ──────────────────────────────────────────────────────
    'local_train_time_s',
    'local_send_time_s',
    'local_receive_time_s',      # latencia de descarga del modelo global
    'round_total_time_s',        # entrenamiento + envío + recepción
    # ── KPI: Red (tamaños de modelo) ─────────────────────────────────────
    'local_upload_bytes',
    'local_download_bytes',
    'local_total_bytes',
    'bandwidth_upload_kbps',
    'bandwidth_download_kbps',
    'throughput_upload_kbps',
    'throughput_download_kbps',
    # ── KPI: Red (psutil delta) ───────────────────────────────────────────
    'net_bytes_sent',
    'net_bytes_recv',
    'net_packets_sent',
    'net_packets_recv',
    'net_errin',
    'net_errout',
    'net_dropin',
    'net_dropout',
    # ── KPI: Arquitectura ────────────────────────────────────────────────
    'network_hops',              # siempre 1 en centralizado
    'communication_cost_bytes',  # M * (N-1)
    # ── KPI: Modelo ──────────────────────────────────────────────────────
    'local_loss',
    'local_accuracy',
    'local_precision',
    'local_recall',
    'local_f1',
    'local_specificity',
    'local_sensitivity',         # = recall (alias clínico)
    # ── KPI: Energía ─────────────────────────────────────────────────────────
    'ecomp_j',              # Energía cómputo local (J) = 2·alpha·c·D·f²
    'ecomm_j',              # Energía comunicación (J)  = p · Tcomm
    'etotal_j',             # Energía total ronda (J)   = Ecomp + Ecomm
    'ecomp_params',         # JSON: {alpha, c, D_bits, f_hz} — trazabilidad
    'ecomm_params',         # JSON: {p_w, tcomm_s}
    'cumulative_etotal_j',  # Energía acumulada desde ronda 1 (E_alpha parcial)
    # Status
    'status',
]


class CentralizedWorker:
    def __init__(self, agent_id: str, csv_file: str,
                 server_uri: str = 'ws://localhost:8765'):
        self.agent_id = agent_id
        self.csv_file = csv_file
        self.server_uri = server_uri

        self.model: MLP = None
        self.optimizer = None
        self.criterion = nn.BCEWithLogitsLoss()
        self.train_data: DataLoader = None
        self.val_data: DataLoader = None
        self.n_features: int = None

        self.global_model_params = None   # para FedProx
        self.current_round = 0
        self.cumulative_etotal = 0.0   # E_alpha: energía acumulada total
        self.round_start_time: float = 0.0

        # Log CSV
        os.makedirs(Config.LOGS_DIR, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(
            Config.LOGS_DIR, f'worker_{agent_id}_{ts}.csv'
        )
        self._init_csv()
        print(f"[{agent_id}] Log: {self.log_file}")

    # -----------------------------------------------------------------------
    # CSV
    # -----------------------------------------------------------------------
    def _init_csv(self):
        with open(self.log_file, 'w', newline='') as f:
            csv.writer(f).writerow(WORKER_CSV_HEADERS)

    def _log_round(self, metrics: dict):
        with open(self.log_file, 'a', newline='') as f:
            csv.writer(f).writerow([metrics.get(h, '') for h in WORKER_CSV_HEADERS])

    # -----------------------------------------------------------------------
    # Carga de datos
    # -----------------------------------------------------------------------
    def load_and_preprocess_data(self):
        print(f"\n{'='*60}\n[{self.agent_id}] CARGANDO DATOS\n{'='*60}")
        df = pd.read_csv(self.csv_file)
        print(f"[{self.agent_id}] Filas: {len(df)} | Columnas: {list(df.columns)}")

        preprocessor = load_preprocessor(Config.PREPROCESSOR_PATH)

        # Detectar target
        target_col = None
        for c in ['is_premature_ncd', 'Classification', 'target', 'label', 'y']:
            if c in df.columns:
                target_col = c
                break
        if target_col is None:
            last = df.columns[-1]
            target_col = last if last != 'hospital_cliente' else df.columns[-2]
        print(f"[{self.agent_id}] Target: {target_col}")

        y = df[target_col].values
        cols_to_drop = [target_col]
        for c in ['hospital_cliente', 'ncd_group']:
            if c in df.columns:
                cols_to_drop.append(c)
        X = df.drop(cols_to_drop, axis=1)

        X_transformed = preprocessor.transform(X)
        self.n_features = X_transformed.shape[1]
        print(f"[{self.agent_id}] Features: {self.n_features} | "
              f"Muestras: {len(y)} | Distrib: {np.bincount(y.astype(int))}")

        X_t = torch.FloatTensor(X_transformed)
        y_t = torch.FloatTensor(y).unsqueeze(1)
        dataset = TensorDataset(X_t, y_t)

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_ds, val_ds = torch.utils.data.random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        self.train_data = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
        self.val_data = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False)

        self.model = MLP(in_features=self.n_features, seed=42, p_dropout=0.3)
        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)
        print(f"[{self.agent_id}] Modelo MLP creado. Train={train_size} Val={val_size}")

    # -----------------------------------------------------------------------
    # Entrenamiento
    # -----------------------------------------------------------------------
    def train_local(self, epochs: int = None):
        if epochs is None:
            epochs = Config.LOCAL_EPOCHS
        method = Config.AGGREGATION_METHOD
        self.model.train()
        t0 = time.time()
        for epoch in range(epochs):
            for bx, by in self.train_data:
                self.optimizer.zero_grad()
                out = self.model(bx)
                loss = self.criterion(out, by)
                if method == 'FedProx' and self.global_model_params is not None:
                    prox = sum(
                        ((p - self.global_model_params[n]) ** 2).sum()
                        for n, p in self.model.named_parameters()
                    )
                    loss += (Config.FEDPROX_MU / 2) * prox
                loss.backward()
                self.optimizer.step()
        return time.time() - t0

    # -----------------------------------------------------------------------
    # Evaluación
    # -----------------------------------------------------------------------
    def calculate_all_metrics(self) -> dict:
        if self.val_data is None:
            return {k: 0.0 for k in
                    ['loss', 'accuracy', 'precision', 'recall', 'f1', 'specificity']}
        self.model.eval()
        total_loss = 0.0
        preds, labels = [], []
        with torch.no_grad():
            for bx, by in self.val_data:
                out = self.model(bx)
                total_loss += self.criterion(out, by).item()
                predicted = (torch.sigmoid(out) > 0.5).float()
                preds.extend(predicted.cpu().numpy().flatten())
                labels.extend(by.cpu().numpy().flatten())

        avg_loss = total_loss / len(self.val_data)
        accuracy = sum(p == l for p, l in zip(preds, labels)) / len(labels)
        precision = precision_score(labels, preds, average='macro', zero_division=0)
        recall = recall_score(labels, preds, average='macro', zero_division=0)
        f1 = f1_score(labels, preds, average='macro', zero_division=0)
        specificity = calc_specificity(labels, preds)

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'specificity': specificity,
        }

    # -----------------------------------------------------------------------
    # Parámetros del modelo
    # -----------------------------------------------------------------------
    def get_model_parameters(self):
        return {n: p.data.clone() for n, p in self.model.named_parameters()}

    def set_model_parameters(self, parameters):
        for n, p in self.model.named_parameters():
            p.data = parameters[n].clone()

    # -----------------------------------------------------------------------
    # Conexión al servidor
    # -----------------------------------------------------------------------
    async def connect_to_server(self):
        print(f"[{self.agent_id}] Conectando a {self.server_uri} ...")
        retry = 0
        while retry < Config.MAX_RETRIES:
            try:
                async with websockets.connect(
                    self.server_uri,
                    max_size=Config.WS_MAX_SIZE,
                    open_timeout=Config.WS_TIMEOUT
                ) as ws:
                    # Registrar
                    await ws.send(json.dumps({
                        'type': 'register',
                        'agent_id': self.agent_id,
                        'n_features': self.n_features
                    }))
                    # Escuchar mensajes
                    async for msg in ws:
                        data = json.loads(msg)
                        await self._handle_message(data, ws)
                    break
            except (ConnectionRefusedError, OSError) as e:
                retry += 1
                print(f"[{self.agent_id}] Reintento {retry}/{Config.MAX_RETRIES}: {e}")
                await asyncio.sleep(Config.RETRY_DELAY)
        else:
            print(f"[{self.agent_id}] ❌ No pudo conectar tras {Config.MAX_RETRIES} intentos")

    async def _handle_message(self, data: dict, ws):
        msg_type = data.get('type')

        if msg_type == 'registered':
            print(f"[{self.agent_id}] ✓ Registrado en servidor")

        elif msg_type == 'start_round':
            self.current_round = data['round']
            self.round_start_time = time.time()
            print(f"\n{'='*60}\n[{self.agent_id}] RONDA {self.current_round}\n{'='*60}")

            # ── Snapshot ANTES ──────────────────────────────────────────
            cpu_before = get_cpu_percent()
            ram_before = get_ram_mb()
            temp_c = get_temperature_c()
            cpu_freq = get_cpu_freq_mhz()
            sockets = get_open_sockets()
            net_before = get_net_snapshot()

            # ── Recibir modelo global (latencia de descarga) ─────────────
            t_recv_start = time.time()
            model_bytes = await ws.recv()
            recv_time = time.time() - t_recv_start
            download_bytes = len(model_bytes)
            download_bw = calc_bandwidth_kbps(download_bytes, recv_time)

            global_params = pickle.loads(model_bytes)
            self.set_model_parameters(global_params)
            self.global_model_params = {n: p.clone() for n, p in global_params.items()}
            print(f"[{self.agent_id}] ↓ Modelo recibido: {download_bytes} B  "
                  f"latencia={recv_time:.3f}s  bw={download_bw:.1f} kbps")

            # ── Entrenar localmente ──────────────────────────────────────
            train_time = self.train_local()
            print(f"[{self.agent_id}] ✓ Entrenamiento: {train_time:.3f}s")

            # ── Calcular métricas ────────────────────────────────────────
            metrics = self.calculate_all_metrics()
            cpu_after = get_cpu_percent()
            ram_after = get_ram_mb()

            # ── Enviar modelo al servidor ─────────────────────────────────
            local_params = self.get_model_parameters()
            params_bytes = pickle.dumps(local_params)
            upload_bytes = len(params_bytes)
            upload_bw = calc_bandwidth_kbps(upload_bytes, 0.001)  # pre-compute

            t_send_start = time.time()
            # Pre-calcular Ecomm estimado con latencia de descarga (proxy antes de envío)
            ecomm_pre = calc_ecomm(recv_time)   # Tcomm = recv_time (latencia medida)
            ecomp_pre = calc_ecomp(len(self.train_data.dataset), self.n_features, cpu_freq)
            # Enviar mensaje JSON con métricas extendidas (incluyendo energía pre-calculada)
            await ws.send(json.dumps({
                'type': 'model_update',
                'agent_id': self.agent_id,
                'local_recall': metrics['recall'],
                'download_bytes': download_bytes,
                'metrics': {
                    **metrics,
                    'upload_bytes': upload_bytes,
                    'download_bytes': download_bytes,
                    'ecomp_j': ecomp_pre,
                    'ecomm_j': ecomm_pre,
                    'etotal_j': calc_etotal(ecomp_pre, ecomm_pre),
                }
            }))
            await ws.send(params_bytes)
            send_time = time.time() - t_send_start
            upload_bw_actual = calc_bandwidth_kbps(upload_bytes, send_time)

            print(f"[{self.agent_id}] ↑ Modelo enviado: {upload_bytes} B  "
                  f"send={send_time:.3f}s  bw={upload_bw_actual:.1f} kbps")

            # ── Net delta ────────────────────────────────────────────────
            net_after = get_net_snapshot()
            nd = net_delta(net_before, net_after)

            round_total = time.time() - self.round_start_time
            train_samples = len(self.train_data.dataset)

            # ── Communication cost: M * (N-1) ────────────────────────────
            comm_cost = upload_bytes * (Config.NUM_SILOS - 1)

            # ── Energía ──────────────────────────────────────────────────
            ecomp = calc_ecomp(train_samples, self.n_features, cpu_freq)
            ecomm = calc_ecomm(send_time)
            etotal = calc_etotal(ecomp, ecomm)
            if etotal > 0:
                self.cumulative_etotal += etotal

            ecomp_params = json.dumps({
                'alpha': ALPHA_CAPACITANCE,
                'c_cycles_per_bit': C_CYCLES_PER_BIT,
                'D_bits': train_samples * self.n_features * 32,
                'f_hz': round(cpu_freq * 1e6, 0) if cpu_freq > 0 else -1,
            })
            ecomm_params = json.dumps({
                'p_w': P_TRANSMISSION_W,
                'tcomm_s': round(send_time, 4),
            })
            print(f"[{self.agent_id}] ⚡ Ecomp={ecomp:.3e}J  Ecomm={ecomm:.4f}J  Etotal={etotal:.4f}J  Acum={self.cumulative_etotal:.4f}J")

            # ── Guardar en CSV ───────────────────────────────────────────
            log_row = {
                'timestamp': datetime.now().isoformat(),
                'round': self.current_round,
                'client_id': self.agent_id,
                'local_samples': train_samples,
                'local_epochs': Config.LOCAL_EPOCHS,
                'local_batch_size': Config.BATCH_SIZE,
                # recursos
                'cpu_percent_before': round(cpu_before, 2),
                'cpu_percent_after': round(cpu_after, 2),
                'ram_mb_before': round(ram_before, 2),
                'ram_mb_after': round(ram_after, 2),
                'temperature_c': temp_c,
                'cpu_freq_mhz': cpu_freq,
                'open_sockets': sockets,
                # timing
                'local_train_time_s': round(train_time, 4),
                'local_send_time_s': round(send_time, 4),
                'local_receive_time_s': round(recv_time, 4),
                'round_total_time_s': round(round_total, 4),
                # red (modelo)
                'local_upload_bytes': upload_bytes,
                'local_download_bytes': download_bytes,
                'local_total_bytes': upload_bytes + download_bytes,
                'bandwidth_upload_kbps': round(upload_bw_actual, 4),
                'bandwidth_download_kbps': round(download_bw, 4),
                'throughput_upload_kbps': round(upload_bw_actual, 4),
                'throughput_download_kbps': round(download_bw, 4),
                # red (psutil)
                'net_bytes_sent': nd['bytes_sent'],
                'net_bytes_recv': nd['bytes_recv'],
                'net_packets_sent': nd['packets_sent'],
                'net_packets_recv': nd['packets_recv'],
                'net_errin': nd['errin'],
                'net_errout': nd['errout'],
                'net_dropin': nd['dropin'],
                'net_dropout': nd['dropout'],
                # arquitectura
                'network_hops': Config.NETWORK_HOPS,
                'communication_cost_bytes': comm_cost,
                # modelo
                'local_loss': round(metrics['loss'], 4),
                'local_accuracy': round(metrics['accuracy'], 4),
                'local_precision': round(metrics['precision'], 4),
                'local_recall': round(metrics['recall'], 4),
                'local_f1': round(metrics['f1'], 4),
                'local_specificity': round(metrics['specificity'], 4),
                'local_sensitivity': round(metrics['recall'], 4),
                # energía
                'ecomp_j': ecomp,
                'ecomm_j': ecomm,
                'etotal_j': etotal,
                'ecomp_params': ecomp_params,
                'ecomm_params': ecomm_params,
                'cumulative_etotal_j': round(self.cumulative_etotal, 6),
                'status': 'OK',
            }
            self._log_round(log_row)

        elif msg_type == 'training_stopped':
            print(
                f"\n[{self.agent_id}] 🛑 ENTRENAMIENTO DETENIDO\n"
                f"  Razón: {data.get('reason')}\n"
                f"  Rondas: {data.get('total_rounds')}\n"
                f"  Mejor recall: {data.get('best_recall', 0.0):.4f}"
            )

    # -----------------------------------------------------------------------
    # Entry point
    # -----------------------------------------------------------------------
    async def start(self):
        self.load_and_preprocess_data()
        await self.connect_to_server()


def main():
    if len(sys.argv) < 3:
        print("Uso: python worker_centralized.py <agent_id> <csv_file> [server_uri]")
        print("Ejemplo: python worker_centralized.py silo_1 data/silos/silo_1.csv ws://server:8765")
        sys.exit(1)

    agent_id = sys.argv[1]
    csv_file = sys.argv[2]
    server_uri = sys.argv[3] if len(sys.argv) > 3 else 'ws://localhost:8765'

    worker = CentralizedWorker(agent_id, csv_file, server_uri)
    asyncio.run(worker.start())


if __name__ == '__main__':
    main()
