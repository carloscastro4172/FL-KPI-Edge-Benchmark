"""
metrics_analysis.py
===================
Post-procesa los CSV generados por server y workers.
Calcula métricas derivadas para el paper:
  - Jitter (std de latencia por worker)
  - Curvas de convergencia
  - Tabla resumen de KPIs de comunicación
  - Impacto straggler
  - Varianza inter-silo (indicador non-IID)

Uso:
    python metrics_analysis.py [--logs-dir logs] [--output-dir analysis]
"""
import os
import glob
import argparse
import json
import numpy as np
import pandas as pd


def load_server_logs(logs_dir: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(logs_dir, 'server_centralized_*.csv'))
    if not files:
        print("⚠️  No se encontraron logs del servidor")
        return pd.DataFrame()
    dfs = [pd.read_csv(f) for f in sorted(files)]
    return pd.concat(dfs, ignore_index=True)


def load_worker_logs(logs_dir: str) -> dict:
    """Retorna dict {silo_id: DataFrame}"""
    workers = {}
    for f in sorted(glob.glob(os.path.join(logs_dir, 'worker_*.csv'))):
        name = os.path.basename(f).replace('.csv', '')
        try:
            workers[name] = pd.read_csv(f)
        except Exception as e:
            print(f"⚠️  Error leyendo {f}: {e}")
    return workers


# ---------------------------------------------------------------------------
# KPIs de comunicación
# ---------------------------------------------------------------------------
def comm_kpis(workers: dict) -> pd.DataFrame:
    rows = []
    for name, df in workers.items():
        row = {
            'worker': name,
            'rounds': len(df),
            # Throughput promedio
            'avg_bandwidth_upload_kbps': df['bandwidth_upload_kbps'].mean(),
            'avg_bandwidth_download_kbps': df['bandwidth_download_kbps'].mean(),
            # Latencia (receive_time como proxy de latencia de descarga)
            'avg_latency_recv_s': df['local_receive_time_s'].mean(),
            'min_latency_recv_s': df['local_receive_time_s'].min(),
            'max_latency_recv_s': df['local_receive_time_s'].max(),
            # Jitter = std de la latencia
            'jitter_recv_s': df['local_receive_time_s'].std(),
            'avg_latency_send_s': df['local_send_time_s'].mean(),
            'jitter_send_s': df['local_send_time_s'].std(),
            # Bytes
            'total_upload_bytes': df['local_upload_bytes'].sum(),
            'total_download_bytes': df['local_download_bytes'].sum(),
            'avg_upload_bytes_per_round': df['local_upload_bytes'].mean(),
            'avg_download_bytes_per_round': df['local_download_bytes'].mean(),
            # Packets
            'total_net_packets_sent': df['net_packets_sent'].sum(),
            'total_net_packets_recv': df['net_packets_recv'].sum(),
            # Errores / drops
            'total_net_errin': df['net_errin'].sum(),
            'total_net_errout': df['net_errout'].sum(),
            'total_net_dropin': df['net_dropin'].sum(),
            'total_net_dropout': df['net_dropout'].sum(),
            # Communication cost (M*(N-1)) acumulado
            'total_comm_cost_bytes': df['communication_cost_bytes'].sum(),
            'avg_comm_cost_per_round': df['communication_cost_bytes'].mean(),
            # Network hops (siempre 1 en centralizado)
            'network_hops': df['network_hops'].iloc[0] if 'network_hops' in df else 1,
        }
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# KPIs de recursos
# ---------------------------------------------------------------------------
def resource_kpis(workers: dict) -> pd.DataFrame:
    rows = []
    for name, df in workers.items():
        row = {
            'worker': name,
            'avg_cpu_before': df['cpu_percent_before'].mean(),
            'avg_cpu_after': df['cpu_percent_after'].mean(),
            'avg_ram_before_mb': df['ram_mb_before'].mean(),
            'avg_ram_after_mb': df['ram_mb_after'].mean(),
            'avg_ram_delta_mb': (df['ram_mb_after'] - df['ram_mb_before']).mean(),
            'avg_temp_c': df['temperature_c'].replace(-1, np.nan).mean(),
            'avg_train_time_s': df['local_train_time_s'].mean(),
            'avg_open_sockets': df['open_sockets'].replace(-1, np.nan).mean(),
        }
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# KPIs del modelo
# ---------------------------------------------------------------------------
def model_kpis(workers: dict) -> pd.DataFrame:
    rows = []
    for name, df in workers.items():
        row = {
            'worker': name,
            'final_recall': df['local_recall'].iloc[-1],
            'final_accuracy': df['local_accuracy'].iloc[-1],
            'final_f1': df['local_f1'].iloc[-1],
            'final_precision': df['local_precision'].iloc[-1],
            'final_specificity': df['local_specificity'].iloc[-1],
            'best_recall': df['local_recall'].max(),
            'avg_recall': df['local_recall'].mean(),
            'avg_loss': df['local_loss'].mean(),
        }
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Convergencia y straggler (desde servidor)
# ---------------------------------------------------------------------------
def convergence_and_straggler(server_df: pd.DataFrame) -> dict:
    if server_df.empty:
        return {}

    conv_round = server_df.loc[server_df['best_recall_so_far'].idxmax(), 'round']
    conv_time = server_df['convergence_time_s'].dropna()

    result = {
        'total_rounds': server_df['round'].max(),
        'convergence_round': int(conv_round),
        'convergence_time_s': float(conv_time.iloc[-1]) if len(conv_time) else None,
        'best_global_recall': server_df['global_recall'].max(),
        'best_global_f1': server_df['global_f1'].max(),
        # Straggler
        'avg_straggler_delay_s': server_df['straggler_delay_s'].mean(),
        'max_straggler_delay_s': server_df['straggler_delay_s'].max(),
        # non-IID indicator
        'avg_recall_variance': server_df['recall_variance'].mean(),
        'avg_recall_std': server_df['recall_std'].mean(),
        # Comunicación total (servidor)
        'total_server_bytes_sent': server_df['server_net_bytes_sent'].sum(),
        'total_server_bytes_recv': server_df['server_net_bytes_recv'].sum(),
        'avg_aggregation_time_s': server_df['aggregation_time_s'].mean(),
        'avg_round_time_s': server_df['round_total_time_s'].mean(),
    }
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Análisis de métricas FL centralizado')
    parser.add_argument('--logs-dir', default='logs', help='Directorio con CSVs de logs')
    parser.add_argument('--output-dir', default='analysis', help='Directorio de salida')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("ANÁLISIS DE MÉTRICAS – FL Centralizado 7 Silos")
    print("=" * 70)

    server_df = load_server_logs(args.logs_dir)
    workers = load_worker_logs(args.logs_dir)

    if not workers:
        print("❌ No se encontraron logs de workers.")
        return

    # 1. Comunicación
    comm_df = comm_kpis(workers)
    comm_path = os.path.join(args.output_dir, 'kpi_comunicaciones.csv')
    comm_df.to_csv(comm_path, index=False)
    print(f"\n[1] KPIs de Comunicación → {comm_path}")
    print(comm_df[['worker', 'avg_latency_recv_s', 'jitter_recv_s',
                   'avg_bandwidth_upload_kbps', 'total_comm_cost_bytes',
                   'network_hops']].to_string(index=False))

    # 2. Recursos
    res_df = resource_kpis(workers)
    res_path = os.path.join(args.output_dir, 'kpi_recursos.csv')
    res_df.to_csv(res_path, index=False)
    print(f"\n[2] KPIs de Recursos → {res_path}")
    print(res_df[['worker', 'avg_cpu_after', 'avg_ram_after_mb',
                  'avg_train_time_s']].to_string(index=False))

    # 3. Modelo
    mod_df = model_kpis(workers)
    mod_path = os.path.join(args.output_dir, 'kpi_modelo.csv')
    mod_df.to_csv(mod_path, index=False)
    print(f"\n[3] KPIs del Modelo → {mod_path}")
    print(mod_df[['worker', 'best_recall', 'final_f1',
                  'final_specificity']].to_string(index=False))

    # 4. Convergencia y straggler
    conv = convergence_and_straggler(server_df)
    conv_path = os.path.join(args.output_dir, 'kpi_convergencia.json')
    with open(conv_path, 'w') as f:
        json.dump(conv, f, indent=2)
    print(f"\n[4] Convergencia / Straggler → {conv_path}")
    for k, v in conv.items():
        print(f"    {k}: {v}")

    # 5. Curva de convergencia global (CSV)
    if not server_df.empty:
        conv_curve = server_df[['round', 'global_recall', 'global_accuracy',
                                'global_f1', 'global_loss', 'recall_variance',
                                'straggler_delay_s', 'aggregation_time_s',
                                'round_total_time_s', 'total_elapsed_time_s']].copy()
        curve_path = os.path.join(args.output_dir, 'convergence_curve.csv')
        conv_curve.to_csv(curve_path, index=False)
        print(f"\n[5] Curva de convergencia → {curve_path}")

    # 6. Resumen global
    summary = {
        'architecture': 'centralized',
        'num_silos': len(workers),
        'aggregation_method': server_df['aggregation_method'].iloc[0]
            if not server_df.empty else 'unknown',
        'network_hops': 1,
        'comm_kpis': comm_df.mean(numeric_only=True).to_dict(),
        'resource_kpis': res_df.mean(numeric_only=True).to_dict(),
        'model_kpis': mod_df.mean(numeric_only=True).to_dict(),
        'convergence': conv,
    }
    summ_path = os.path.join(args.output_dir, 'summary.json')
    with open(summ_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[6] Resumen completo → {summ_path}")
    print("\n✓ Análisis completado")


if __name__ == '__main__':
    main()
