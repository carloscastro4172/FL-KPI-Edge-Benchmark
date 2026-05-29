"""
ModelTrainer + FedAvg / FedProx / FedNova
Compatible con preprocessor_global.joblib y silos 1-5
"""
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import os
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from config import (TEST_SIZE, BATCH_SIZE, LABEL_COLUMN, DROP_COLUMNS,
                    RANDOM_STATE, EPOCHS, AGGREGATION_METHOD,
                    FEDPROX_MU, IN_FEATURES, PREPROCESSOR_PATH,
                    ENERGY_ALPHA, ENERGY_C_CYCLES, ENERGY_P_TX)
from network_metrics import calc_ecomp, calc_ecomm, calc_etotal, collect_system_metrics
from logging_config import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────
# MLP (igual que en proyectos centralizado/semi-desc)
# ──────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, in_features: int = IN_FEATURES, seed: int = 42, p_dropout: float = 0.3):
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(in_features, 120),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(84, 1),
        )

    def forward(self, x):
        return self.net(x)


# ──────────────────────────────────────────────────────────────
# ModelTrainer
# ──────────────────────────────────────────────────────────────
class ModelTrainer:
    def __init__(self, model_path: str, model_architecture: nn.Module, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = model_architecture.to(self.device)
        self.n_features = None
        self.n_train    = None

        # Guardar pesos iniciales para FedProx/FedNova
        self.global_params = None

        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            logger.info(f"[INIT] Pesos cargados: {model_path}")
        except Exception as e:
            logger.warning(f"[INIT] Sin pesos previos: {e}. Entrenando desde cero.")

    def load_csv(self, path: str, label_col: str = LABEL_COLUMN,
                 test_size: float = TEST_SIZE, batch_size: int = BATCH_SIZE):
        df = pd.read_csv(path)
        
        # Imputar valores vacíos y nulos
        df = df.replace(r'^\s*$', np.nan, regex=True)
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val if not pd.isna(median_val) else 0)
            else:
                mode_val = df[col].mode()
                df[col] = df[col].fillna(mode_val[0] if not mode_val.empty else "Desconocido")

        # Usar preprocessor_global.joblib si existe
        drop_cols = [label_col] + [c for c in DROP_COLUMNS if c in df.columns]
        X_raw = df.drop(columns=drop_cols)
        y = df[label_col].values.astype("float32")

        if os.path.exists(PREPROCESSOR_PATH):
            try:
                preproc = joblib.load(PREPROCESSOR_PATH)
                X = preproc.transform(X_raw).astype("float32")
                logger.info(f"[DATA] Preprocessor cargado. Features={X.shape[1]}")
            except Exception as e:
                logger.warning(f"[DATA] Preprocessor falló ({e}), usando StandardScaler local")
                from sklearn.preprocessing import StandardScaler
                X = StandardScaler().fit_transform(X_raw.values.astype("float32"))
        else:
            from sklearn.preprocessing import StandardScaler
            X = StandardScaler().fit_transform(X_raw.values.astype("float32"))

        self.n_features = X.shape[1]
        logger.info(f"[DATA] shape={X.shape}, target={np.bincount(y.astype(int))}")

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_STATE)
        self.n_train = len(X_tr)

        to_loader = lambda xd, yd: DataLoader(
            TensorDataset(torch.tensor(xd), torch.tensor(yd).unsqueeze(1)),
            batch_size=batch_size, shuffle=True)
        return to_loader(X_tr, y_tr), to_loader(X_te, y_te)

    def fit(self, train_loader: DataLoader, criterion: nn.Module,
            optimizer: torch.optim.Optimizer, epochs: int = EPOCHS) -> tuple:
        """
        Devuelve (train_time_s, last_loss, fednova_tau, fednova_grad).
        """
        method = AGGREGATION_METHOD
        logger.info(f"[TRAINING] {method} | {epochs} épocas...")

        # Guardar pesos iniciales para proximal/nova
        w0 = {n: p.data.clone() for n, p in self.model.named_parameters()}
        self.global_params = w0

        start = time.time()
        self.model.train()
        last_loss = 0.0
        tau = 0

        for epoch in range(epochs):
            running = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(xb), yb)

                if method == 'FedProx' and self.global_params:
                    prox = sum(
                        ((p - self.global_params[n].to(self.device)) ** 2).sum()
                        for n, p in self.model.named_parameters()
                    )
                    loss += (FEDPROX_MU / 2) * prox

                loss.backward()
                optimizer.step()
                running += loss.item() * xb.size(0)
                tau += 1

            last_loss = running / len(train_loader.dataset)
            logger.info(f" Época [{epoch+1}/{epochs}] Loss={last_loss:.4f}")

        train_time = time.time() - start
        logger.info(f"[TRAINING] Completado en {train_time:.2f}s | tau={tau}")

        # FedNova: gradiente normalizado d_i = (w0 - w_local) / tau
        fednova_grad = {}
        if method == 'FedNova' and tau > 0:
            fednova_grad = {
                n: (w0[n] - p.data.clone()) / tau
                for n, p in self.model.named_parameters()
            }

        return train_time, last_loss, tau, fednova_grad

    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader) -> dict:
        self.model.eval()
        preds_all, labels_all = [], []
        total_loss = 0.0
        criterion  = nn.BCEWithLogitsLoss()

        for xb, yb in test_loader:
            xb = xb.to(self.device)
            out  = self.model(xb)
            total_loss += criterion(out, yb.to(self.device)).item()
            pred = (torch.sigmoid(out) >= 0.5).long().squeeze(1)
            preds_all.extend(pred.cpu().numpy())
            labels_all.extend(yb.squeeze(1).long().numpy())

        accuracy  = sum(p == l for p, l in zip(preds_all, labels_all)) / len(labels_all)
        precision = precision_score(labels_all, preds_all, average='macro', zero_division=0)
        recall    = recall_score(labels_all, preds_all, average='macro', zero_division=0)
        f1        = f1_score(labels_all, preds_all, average='macro', zero_division=0)
        spec, sens = 0.0, 0.0
        try:
            tn, fp, fn, tp = confusion_matrix(labels_all, preds_all, labels=[0,1]).ravel()
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        except Exception:
            pass

        metrics = {
            "accuracy":    round(float(accuracy),  4),
            "precision":   round(float(precision), 4),
            "recall":      round(float(recall),    4),
            "f1_score":    round(float(f1),        4),
            "specificity": round(float(spec),      6),
            "sensitivity": round(float(sens),      6),
        }
        logger.info(f"[EVAL] recall={recall:.4f} f1={f1:.4f} acc={accuracy:.4f}")
        return metrics

    def get_model_parameters(self) -> dict:
        return {n: p.data.clone() for n, p in self.model.named_parameters()}

    def set_model_parameters(self, params: dict):
        for n, p in self.model.named_parameters():
            if n in params:
                p.data = params[n].clone()

    def save(self, output_path: str):
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        torch.save(self.model.state_dict(), output_path)
        logger.info(f"[SAVED] {output_path}")


# ──────────────────────────────────────────────────────────────
# Agregación: FedAvg | FedProx | FedNova
# ──────────────────────────────────────────────────────────────
def federated_average(model_paths: list) -> dict:
    """FedAvg clásico."""
    sds = [torch.load(p, map_location="cpu") for p in model_paths]
    avg = {}
    for key in sds[0]:
        avg[key] = torch.stack([sd[key].float() for sd in sds]).mean(dim=0)
    return avg


def fedprox_aggregate(model_paths: list) -> dict:
    """FedProx: misma agregación que FedAvg (proximal aplicado en cliente)."""
    return federated_average(model_paths)


def fednova_aggregate(model_paths: list, nova_grads: dict,
                      sample_counts: dict, global_params: dict) -> dict:
    """
    FedNova: w_global = w0 - sum(p_i * tau_i * d_i)
    nova_grads: {node_id: {param_name: tensor}}
    sample_counts: {node_id: (n_samples, tau)}
    global_params: w0 actual
    """
    if not nova_grads or not global_params:
        logger.warning("[FedNova] Sin gradientes nova, fallback a FedAvg")
        return federated_average(model_paths)

    total_samples = sum(v[0] for v in sample_counts.values())
    if total_samples == 0:
        return federated_average(model_paths)

    result = {k: v.clone() for k, v in global_params.items()}

    for node_id, grad in nova_grads.items():
        n_i, tau_i = sample_counts.get(node_id, (1, 1))
        p_i = n_i / total_samples
        for pname, g in grad.items():
            if pname in result:
                result[pname] -= p_i * tau_i * g

    return result


def aggregate(method: str, model_paths: list,
              nova_grads: dict = None,
              sample_counts: dict = None,
              global_params: dict = None) -> dict:
    """Dispatcher de agregación."""
    if method == 'FedProx':
        return fedprox_aggregate(model_paths)
    elif method == 'FedNova' and nova_grads:
        return fednova_aggregate(model_paths, nova_grads, sample_counts, global_params)
    else:
        return federated_average(model_paths)
