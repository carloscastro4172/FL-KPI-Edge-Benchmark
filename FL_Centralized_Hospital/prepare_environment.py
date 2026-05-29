"""
prepare_environment.py
======================
Paso PRE-DOCKER. Ejecutar UNA VEZ antes de `docker-compose up`.

Hace dos cosas:
  1. Entrena un ColumnTransformer (preprocessor_global.joblib) sobre TODOS los datos
     combinados de los 4 CSV originales.
  2. Divide los datos en 7 silos CSV (data/silos/silo_1.csv … silo_7.csv)
     usando splitting estratificado para mantener distribución de clases.

Uso:
    python prepare_environment.py [--iid] [--seed 42]

    --iid    : split aleatorio estratificado (default)
    --non-iid: split por hospital_cliente (si hay suficientes hospitales)
    --seed   : semilla aleatoria (default 42)
"""
import os
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
RAW_DIR = os.path.join('data', 'raw')
SILOS_DIR = os.path.join('data', 'silos')
ARTIFACTS_DIR = 'artifacts'
NUM_SILOS = 7
TARGET_COL = 'is_premature_ncd'
DROP_COLS = ['hospital_cliente', 'ncd_group']   # evitar leakage

CATEGORICAL_COLS = [
    'sexo', 'etnia', 'sabe_leer', 'est_civil', 'niv_inst',
    'prov_res', 'prov_fall', 'cant_fall',
    'area_res', 'area_fall', 'lugar_ocur',
    'mor_viol', 'lug_viol', 'autopsia', 'residente', 'mes_fall'
]
NUMERICAL_COLS = ['edad_anos', 'anio_fall', 'dia_fall']


def load_all_raw(raw_dir: str) -> pd.DataFrame:
    dfs = []
    for fname in sorted(os.listdir(raw_dir)):
        if fname.endswith('.csv'):
            path = os.path.join(raw_dir, fname)
            df = pd.read_csv(path)
            print(f"  Cargado {fname}: {len(df)} filas")
            dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    print(f"\n  Total combinado: {len(combined)} filas")
    print(f"  Distribución target: {combined[TARGET_COL].value_counts().to_dict()}")
    return combined


def build_preprocessor(df_features: pd.DataFrame) -> ColumnTransformer:
    """Construye y entrena el ColumnTransformer sobre el dataset completo."""
    # Detectar qué columnas realmente están presentes
    cat_cols = [c for c in CATEGORICAL_COLS if c in df_features.columns]
    num_cols = [c for c in NUMERICAL_COLS if c in df_features.columns]

    print(f"\n  Columnas categóricas ({len(cat_cols)}): {cat_cols}")
    print(f"  Columnas numéricas   ({len(num_cols)}): {num_cols}")

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
            ('num', StandardScaler(), num_cols),
        ],
        remainder='drop'
    )
    preprocessor.fit(df_features)

    # Verificar dimensiones
    n_features = preprocessor.transform(df_features.iloc[:1]).shape[1]
    print(f"  Features tras transformación: {n_features}")
    return preprocessor


def split_iid(df: pd.DataFrame, n_silos: int, seed: int) -> list:
    """
    Split estratificado IID: cada silo tiene distribución similar a global.
    Retorna lista de n_silos DataFrames.
    """
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    y = df[TARGET_COL].values
    silos = []

    remaining = df.copy()
    for i in range(n_silos - 1):
        n_remaining = n_silos - i
        size = 1.0 / n_remaining
        sss = StratifiedShuffleSplit(n_splits=1, test_size=size, random_state=seed + i)
        idx_keep, idx_silo = next(sss.split(remaining, remaining[TARGET_COL]))
        silos.append(remaining.iloc[idx_silo].reset_index(drop=True))
        remaining = remaining.iloc[idx_keep].reset_index(drop=True)
    silos.append(remaining.reset_index(drop=True))

    return silos


def split_non_iid(df: pd.DataFrame, n_silos: int, seed: int) -> list:
    """
    Split non-IID por hospital_cliente.
    Si hay < n_silos hospitales, mezcla con split IID.
    """
    if 'hospital_cliente' not in df.columns:
        print("  ⚠️  hospital_cliente no encontrado, usando IID split")
        return split_iid(df, n_silos, seed)

    hospitals = df['hospital_cliente'].unique()
    print(f"  Hospitales únicos: {len(hospitals)}")

    if len(hospitals) >= n_silos:
        # Asignar hospitales a silos
        np.random.seed(seed)
        np.random.shuffle(hospitals)
        silos = []
        groups = np.array_split(hospitals, n_silos)
        for g in groups:
            silo_df = df[df['hospital_cliente'].isin(g)].reset_index(drop=True)
            silos.append(silo_df)
        return silos
    else:
        print(f"  Solo {len(hospitals)} hospitales para {n_silos} silos → usando IID con sesgo")
        return split_iid(df, n_silos, seed)


def save_silos(silos: list, silos_dir: str):
    os.makedirs(silos_dir, exist_ok=True)
    for i, silo_df in enumerate(silos, 1):
        path = os.path.join(silos_dir, f'silo_{i}.csv')
        silo_df.to_csv(path, index=False)
        pos = int(silo_df[TARGET_COL].sum())
        neg = len(silo_df) - pos
        print(f"  silo_{i}.csv: {len(silo_df)} filas  |  positivos={pos}  negativos={neg}")


def main():
    parser = argparse.ArgumentParser(description='Prepara entorno para FL centralizado 7 silos')
    parser.add_argument('--mode', choices=['iid', 'non-iid'], default='iid',
                        help='Modo de split (default: iid)')
    parser.add_argument('--seed', type=int, default=42, help='Semilla aleatoria')
    args = parser.parse_args()

    print("=" * 70)
    print("PREPARE_ENVIRONMENT – Centralizado 7 Silos")
    print("=" * 70)

    # 1. Cargar todos los CSV raw
    print("\n[1/4] Cargando CSVs originales...")
    df_all = load_all_raw(RAW_DIR)

    # 2. Preparar features (drop target y columnas no usadas)
    print("\n[2/4] Construyendo preprocessor global...")
    df_features = df_all.drop(columns=[TARGET_COL] + [c for c in DROP_COLS if c in df_all.columns])
    preprocessor = build_preprocessor(df_features)

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    preproc_path = os.path.join(ARTIFACTS_DIR, 'preprocessor_global.joblib')
    joblib.dump(preprocessor, preproc_path)
    print(f"  ✓ Preprocessor guardado: {preproc_path}")

    # 3. Split en silos
    print(f"\n[3/4] Dividiendo en {NUM_SILOS} silos (modo: {args.mode})...")
    if args.mode == 'non-iid':
        silos = split_non_iid(df_all, NUM_SILOS, args.seed)
    else:
        silos = split_iid(df_all, NUM_SILOS, args.seed)

    # 4. Guardar
    print(f"\n[4/4] Guardando silos en {SILOS_DIR}/...")
    save_silos(silos, SILOS_DIR)

    print("\n" + "=" * 70)
    print("✓ Entorno listo. Ahora ejecuta:")
    print("    docker-compose up --build")
    print("=" * 70)


if __name__ == '__main__':
    main()
