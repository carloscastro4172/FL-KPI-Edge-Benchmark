import torch
import os
from config import IN_FEATURES

def create_model(in_features=IN_FEATURES, path="models/model.pt"):
    """Crea modelo MLP inicial y lo guarda en path."""
    # Importación local para evitar circular
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from model.fed_model import MLP
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    model = MLP(in_features=in_features)
    torch.save(model.state_dict(), path)
    print(f"[create_model] Modelo guardado: {path} ({in_features} features)")
    return path

class MLP:
    pass  # Placeholder - use model.fed_model.MLP
