"""
prepare_environment.py — Configuración del entorno en cada RPi
Ejecutar UNA VEZ en cada nodo: python3 prepare_environment.py
"""
import os
import sys
import subprocess

def run(cmd: str, check: bool = True):
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    if check and result.returncode != 0:
        print(f"  ERROR: código {result.returncode}")
    return result.returncode == 0

def main():
    print("\n" + "="*50)
    print("  FL Cluster — Preparación del entorno")
    print("="*50)

    base = os.path.dirname(os.path.abspath(__file__))
    venv = os.path.join(base, "venv")

    print("\n[1] Creando entorno virtual...")
    if not os.path.exists(venv):
        run(f"python3 -m venv {venv}")
    else:
        print("  venv ya existe, omitiendo.")

    pip = os.path.join(venv, "bin", "pip")

    print("\n[2] Actualizando pip...")
    run(f"{pip} install --upgrade pip -q")

    print("\n[3] Instalando PyTorch (CPU)...")
    run(f"{pip} install torch --index-url https://download.pytorch.org/whl/cpu -q")

    print("\n[4] Instalando dependencias...")
    run(f"{pip} install websockets pandas scikit-learn psutil joblib numpy -q")

    print("\n[5] Verificando...")
    activate = os.path.join(venv, "bin", "activate")
    result = subprocess.run(
        f"source {activate} && python3 -c 'import torch,websockets,pandas,sklearn,psutil,joblib; "
        f"print(\"✓ torch\",torch.__version__); print(\"✓ All OK\")'",
        shell=True, executable="/bin/bash", capture_output=True, text=True
    )
    print(result.stdout.strip())
    if result.returncode != 0:
        print("ERROR:", result.stderr.strip())
        sys.exit(1)

    print("\n[6] Creando directorios...")
    for d in ["logs", "models", "received_files", "data/silos"]:
        os.makedirs(os.path.join(base, d), exist_ok=True)
        print(f"  ✓ {d}/")

    print(f"\n✓ Entorno listo en: {venv}")
    print(f"\nPara activar: source {venv}/bin/activate")
    print(f"Para ejecutar servidor: bash run_server.sh FedAvg")
    print(f"Para ejecutar worker:   bash run_worker.sh <num_silo> <ip_servidor>")

if __name__ == "__main__":
    main()
