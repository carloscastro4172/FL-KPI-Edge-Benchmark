"""
Entrada principal — FL Cluster Jerárquico (flat para comparación).

Corrección:
- El modelo inicial ya no se crea a ciegas con IN_FEATURES de config.py.
- Primero intenta inferir automáticamente el número real de features desde
  data/preprocessor_global.joblib o PREPROCESSOR_PATH.
- Si existe un models/model.pt viejo con otra dimensión de entrada, lo elimina
  y crea uno nuevo compatible.

Error corregido:
RuntimeError: mat1 and mat2 shapes cannot be multiplied (16x105 and 104x120)
"""

import asyncio
import os
import sys

from connections.client import send, send_file_to_nodes, send_message_to_nodes, send_identified
from connections.server import listener_ips, listener_server
from config import (
    H_ROUNDS,
    LISTENER_DURATION,
    MODEL_PATH,
    IN_FEATURES,
    METRICS_CSV,
    NODES_JSON,
    AGGREGATION_METHOD,
)
from utils import get_ipport, hierarchy_convergence, load_nodes2dict
from federated import main as fed
from logging_config import get_logger
from model.create_model import create_model

logger = get_logger(__name__)


def _project_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _get_config_attr(name: str, default=None):
    try:
        import config as cfg
        return getattr(cfg, name, default)
    except Exception:
        return default


def _candidate_preprocessor_paths():
    """
    Devuelve rutas posibles del preprocessor_global.joblib.
    En tu cluster actualmente está en:
      ~/fl_cluster_hospital/data/preprocessor_global.joblib
    """
    base = _project_dir()

    candidates = []

    cfg_preproc = _get_config_attr("PREPROCESSOR_PATH", None)
    if cfg_preproc:
        candidates.append(cfg_preproc)

    candidates.extend([
        os.path.join(base, "data", "preprocessor_global.joblib"),
        os.path.join(base, "artifacts", "preprocessor_global.joblib"),
        os.path.join(os.getcwd(), "data", "preprocessor_global.joblib"),
        os.path.join(os.getcwd(), "artifacts", "preprocessor_global.joblib"),
    ])

    # Quitar duplicados conservando orden
    seen = set()
    clean = []
    for p in candidates:
        if not p:
            continue
        p = os.path.abspath(os.path.expanduser(p))
        if p not in seen:
            seen.add(p)
            clean.append(p)
    return clean


def infer_in_features(default: int = IN_FEATURES) -> int:
    """
    Intenta obtener el número real de features desde el preprocesador.
    Si no se puede, usa IN_FEATURES de config.py.
    """
    env_features = os.environ.get("FL_IN_FEATURES", "").strip()
    if env_features:
        try:
            val = int(env_features)
            logger.info(f"[MODEL] IN_FEATURES desde variable FL_IN_FEATURES={val}")
            return val
        except ValueError:
            logger.warning(f"[MODEL] FL_IN_FEATURES inválido: {env_features}")

    for preproc_path in _candidate_preprocessor_paths():
        if not os.path.exists(preproc_path):
            continue

        try:
            import joblib
            preproc = joblib.load(preproc_path)

            if hasattr(preproc, "get_feature_names_out"):
                n_features = len(preproc.get_feature_names_out())
                logger.info(
                    f"[MODEL] Features inferidas desde preprocessor: "
                    f"{n_features} | {preproc_path}"
                )
                return int(n_features)

            # Algunas versiones de sklearn guardan el ColumnTransformer dentro de un Pipeline.
            if hasattr(preproc, "named_steps"):
                for step_name, step_obj in preproc.named_steps.items():
                    if hasattr(step_obj, "get_feature_names_out"):
                        n_features = len(step_obj.get_feature_names_out())
                        logger.info(
                            f"[MODEL] Features inferidas desde pipeline step "
                            f"'{step_name}': {n_features} | {preproc_path}"
                        )
                        return int(n_features)

            logger.warning(
                f"[MODEL] Preprocessor encontrado pero sin get_feature_names_out(): "
                f"{preproc_path}"
            )

        except Exception as e:
            logger.warning(f"[MODEL] No se pudo leer preprocessor {preproc_path}: {e}")

    logger.warning(
        f"[MODEL] No se pudo inferir IN_FEATURES desde preprocessor. "
        f"Usando IN_FEATURES de config.py={default}"
    )
    return int(default)


def checkpoint_input_features(model_path: str):
    """
    Lee un checkpoint .pt y devuelve la dimensión de entrada de la primera capa Linear.
    Para el error actual, el checkpoint viejo devuelve 104.
    """
    if not os.path.exists(model_path):
        return None

    try:
        import torch
        obj = torch.load(model_path, map_location="cpu")

        # Posibles formatos:
        # 1) state_dict directo
        # 2) {"state_dict": ...}
        # 3) modelo completo
        if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
            state = obj["state_dict"]
        elif isinstance(obj, dict):
            state = obj
        elif hasattr(obj, "state_dict"):
            state = obj.state_dict()
        else:
            return None

        # Buscar pesos 2D de una capa lineal. En tu modelo suele ser net.0.weight.
        preferred_keys = [
            "net.0.weight",
            "model.net.0.weight",
            "module.net.0.weight",
        ]

        for key in preferred_keys:
            tensor = state.get(key)
            if tensor is not None and hasattr(tensor, "ndim") and tensor.ndim == 2:
                return int(tensor.shape[1])

        for key, tensor in state.items():
            if hasattr(tensor, "ndim") and tensor.ndim == 2 and "weight" in key:
                return int(tensor.shape[1])

    except Exception as e:
        logger.warning(f"[MODEL] No se pudo inspeccionar checkpoint {model_path}: {e}")

    return None


def ensure_initial_model():
    """
    Crea el modelo inicial con la dimensión correcta.
    Si hay un checkpoint viejo incompatible, lo elimina.
    """
    os.makedirs(os.path.dirname(os.path.abspath(MODEL_PATH)) or ".", exist_ok=True)

    real_in_features = infer_in_features(IN_FEATURES)
    old_in_features = checkpoint_input_features(MODEL_PATH)

    if old_in_features is not None and old_in_features != real_in_features:
        logger.warning(
            f"[MODEL] Checkpoint incompatible detectado: "
            f"{MODEL_PATH} espera {old_in_features}, pero los datos tienen {real_in_features}. "
            f"Eliminando checkpoint viejo."
        )
        try:
            os.remove(MODEL_PATH)
        except FileNotFoundError:
            pass

    logger.info(f"[MODEL] Creando modelo inicial con in_features={real_in_features}")
    create_model(in_features=real_in_features, path=MODEL_PATH)

    final_in_features = checkpoint_input_features(MODEL_PATH)
    if final_in_features is not None and final_in_features != real_in_features:
        raise RuntimeError(
            f"El modelo creado sigue incompatible: "
            f"checkpoint={final_in_features}, esperado={real_in_features}"
        )

    logger.info("[CENTRAL] Modelo inicial creado correctamente")


async def sharing(ip_father: str, ip: str, ips_children: list):
    ips = []
    _, listen_port = get_ipport(ip)
    local_listen = f"0.0.0.0:{listen_port}"

    if ip_father == ip:
        logger.info(f"[CENTRAL] Iniciando | Método={AGGREGATION_METHOD}")
        ensure_initial_model()

    if ip_father != ip:
        logger.info("[HIER] Enviando IP al padre...")
        await send(ip_father, ip)
        logger.info("[HIER] Escuchando IPs hijas...")
        ips = await listener_ips(local_listen, LISTENER_DURATION)
        await get_model(local_listen, LISTENER_DURATION * 10)
    else:
        logger.info("[HIER] Raíz. Esperando hijos...")
        ips = await listener_ips(local_listen, LISTENER_DURATION * 2)

    if ips:
        await send_model(ips)
    else:
        logger.info("[HIER] Nodo hoja sin hijos.")

    for i in ips:
        ips_children.append(i)


async def get_model(local_listen: str, delay: int = LISTENER_DURATION):
    received, _ = await listener_server(local_listen, delay, file_path=MODEL_PATH)
    if received is None:
        logger.error("[HIER] Sin modelo del padre.")


async def send_model(ips_children: list):
    logger.info(f"[HIER] Distribuyendo modelo a {len(ips_children)} hijo(s)...")
    await send_file_to_nodes(ips_children, MODEL_PATH, delay=LISTENER_DURATION * 6)
    try:
        os.remove(MODEL_PATH)
    except FileNotFoundError:
        pass


async def distribute_model(ips_children: list, ip_father: str, ip: str):
    _, listen_port = get_ipport(ip)
    local_listen = f"0.0.0.0:{listen_port}"
    if ips_children and ip_father == ip:
        await send_model(ips_children)
    elif ips_children:
        await get_model(local_listen, LISTENER_DURATION * 100)
        await send_model(ips_children)
    else:
        await get_model(local_listen, LISTENER_DURATION * 100)


async def convergence(ip_father: str, ip: str, ips_children: list, result: list):
    _, listen_port = get_ipport(ip)
    local_listen = f"0.0.0.0:{listen_port}"
    status = "CONVERGED" if (
        os.path.exists(METRICS_CSV) and hierarchy_convergence(METRICS_CSV)
    ) else "NOT CONVERGED"

    if ip_father == ip:
        nodes_by_addr = load_nodes2dict(NODES_JSON)
        from connections.server import listener_nodes
        conv = await listener_nodes(
            local_listen,
            nodes_by_addr,
            LISTENER_DURATION * 100,
            message_only=True,
        )
        vals = [msgs[0] for msgs in conv.values() if msgs]
        res = "CONVERGED" if all(v == "CONVERGED" for v in [*vals, status]) else "NOT CONVERGED"
        result.append(res)
        await send_message_to_nodes(ips_children, result[-1], LISTENER_DURATION * 10)

    elif ips_children:
        nodes_by_addr = load_nodes2dict(NODES_JSON)
        from connections.server import listener_nodes
        conv = await listener_nodes(
            local_listen,
            nodes_by_addr,
            LISTENER_DURATION * 100,
            message_only=True,
        )
        vals = [msgs[0] for msgs in conv.values() if msgs]
        res = "CONVERGED" if all(v == "CONVERGED" for v in [*vals, status]) else "NOT CONVERGED"
        result.append(res)
        await send_identified(ip, ip_father, res)
        res2, _ = await listener_server(local_listen, LISTENER_DURATION * 100)
        result.append(res2)
        await send_message_to_nodes(ips_children, result[-1], LISTENER_DURATION * 10)
    else:
        await send_identified(ip, ip_father, status)
        res, _ = await listener_server(local_listen, LISTENER_DURATION * 100)
        result.append(res)


def main():
    if len(sys.argv) < 3:
        print("Uso: python main.py <mi_ip:puerto> <ip_padre:puerto | null>")
        print("  Servidor: python main.py 172.23.207.115:8765 null")
        print("  Worker:   python main.py 172.23.207.114:8765 172.23.207.115:8765")
        sys.exit(1)

    ip = sys.argv[1]
    ip_father = sys.argv[2] if sys.argv[2].lower() != "null" else sys.argv[1]

    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    ips = []
    asyncio.run(sharing(ip_father, ip, ips))
    leaf = len(ips) == 0

    logger.info(
        f"\n[INFO] ip={ip} | ip_father={ip_father} | "
        f"leaf={leaf} | método={AGGREGATION_METHOD}"
    )

    for h_ronda in range(H_ROUNDS):
        logger.info(f"\n{'=' * 60}\n[HIER] Ronda jerárquica {h_ronda}\n{'=' * 60}")

        if leaf:
            fed(central=False, addr=ip, server_addr=ip_father, h_ronda=h_ronda)
        elif ip_father == ip:
            fed(central=True, addr=ip, server_addr=ip, childs=ips, h_ronda=h_ronda)
        else:
            fed(central=True, addr=ip, server_addr=ip_father, childs=ips, h_ronda=h_ronda)
            fed(central=False, addr=ip, server_addr=ip_father, h_ronda=h_ronda)

        if h_ronda < (H_ROUNDS - 1):
            result = []
            asyncio.run(distribute_model(ips, ip_father, ip))
            asyncio.run(convergence(ip_father, ip, ips, result))
            if result and result[-1] == "CONVERGED":
                break

    logger.info("[HIER] Proceso completado.")


if __name__ == "__main__":
    main()
