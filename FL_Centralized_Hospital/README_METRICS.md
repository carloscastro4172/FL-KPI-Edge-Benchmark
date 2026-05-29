# FL Centralizado – 7 Silos | Guía de Métricas para Paper

## Flujo de ejecución

```
1. python prepare_environment.py        # genera preprocessor + 7 CSVs
2. docker-compose up --build            # levanta server + 7 workers
3. (esperar a que termine el entrenamiento)
4. python metrics_analysis.py           # post-procesa logs → KPIs
```

---

## Paso 1 – Preparar entorno

```bash
# IID (distribución uniforme por defecto)
python prepare_environment.py

# Non-IID (agrupado por hospital)
python prepare_environment.py --mode non-iid
```

Genera:
- `artifacts/preprocessor_global.joblib`
- `data/silos/silo_1.csv` … `silo_7.csv`

---

## Paso 2 – Docker

```bash
docker-compose up --build
```

Logs en tiempo real: `logs/server_centralized_<ts>.csv` y `logs/worker_silo_X_<ts>.csv`

Para detener limpiamente: `Ctrl+C` (el servidor guarda el mejor modelo).

---

## Paso 3 – Análisis post-entrenamiento

```bash
python metrics_analysis.py --logs-dir logs --output-dir analysis
```

Genera en `analysis/`:
| Archivo | Contenido |
|---|---|
| `kpi_comunicaciones.csv` | Latencia, jitter, throughput, bandwidth, bytes, packet errors |
| `kpi_recursos.csv` | CPU, RAM, temperatura, tiempo de entrenamiento |
| `kpi_modelo.csv` | Accuracy, Loss, F1, Recall, Precision, Especificidad por silo |
| `kpi_convergencia.json` | Ronda de convergencia, tiempo, straggler delay, varianza non-IID |
| `convergence_curve.csv` | Curva global por ronda (para gráfica del paper) |
| `summary.json` | Resumen consolidado de todos los KPIs |

---

## KPIs capturados por módulo

### Worker (worker_centralized.py)

| KPI | Columna CSV | Sección paper |
|---|---|---|
| CPU % antes/después | `cpu_percent_before/after` | KPI Nodos |
| RAM MB antes/después | `ram_mb_before/after` | KPI Nodos |
| Temperatura (°C) | `temperature_c` | KPI Nodos |
| Sockets abiertos | `open_sockets` | KPI Nodos |
| Tiempo entrenamiento | `local_train_time_s` | KPI Nodos |
| Latencia descarga modelo | `local_receive_time_s` | KPI Comunicaciones |
| Latencia envío modelo | `local_send_time_s` | KPI Comunicaciones |
| Jitter (post-hoc: std latencia) | derivado | KPI Comunicaciones |
| Throughput upload (kbps) | `bandwidth_upload_kbps` | KPI Comunicaciones |
| Throughput download (kbps) | `bandwidth_download_kbps` | KPI Comunicaciones |
| Bytes upload | `local_upload_bytes` | KPI Comunicaciones |
| Bytes download | `local_download_bytes` | KPI Comunicaciones |
| Paquetes enviados/recibidos | `net_packets_sent/recv` | KPI Comunicaciones |
| Errores/drops de red | `net_errin/errout/dropin/dropout` | KPI Comunicaciones |
| Número de saltos | `network_hops` = 1 (constante) | KPI Comunicaciones |
| Costo comunicación M*(N-1) | `communication_cost_bytes` | KPI Comunicaciones |
| Accuracy | `local_accuracy` | KPI Modelo |
| Loss | `local_loss` | KPI Modelo |
| F1-score | `local_f1` | KPI Modelo |
| Precision | `local_precision` | KPI Modelo |
| Recall / Sensibilidad | `local_recall` / `local_sensitivity` | KPI Modelo |
| Especificidad | `local_specificity` | KPI Modelo |

### Server (server_centralized.py)

| KPI | Columna CSV | Sección paper |
|---|---|---|
| Tiempo de agregación | `aggregation_time_s` | KPI Nodos |
| Tiempo total por ronda | `round_total_time_s` | KPI Nodos |
| Tiempo total experimento | `total_elapsed_time_s` | KPI Nodos |
| CPU/RAM servidor | `server_cpu_before/after` | KPI Nodos |
| Net bytes servidor (psutil) | `server_net_bytes_sent/recv` | KPI Comunicaciones |
| Overhead por ronda M*(N-1) | `comm_overhead_bytes` | KPI Comunicaciones |
| Straggler delay | `straggler_delay_s` | Dinamismo |
| Orden de llegada workers | `worker_arrival_order` (JSON) | Dinamismo |
| Dropout count | `dropout_count` | Dinamismo |
| Recall global | `global_recall` | KPI Modelo |
| Accuracy global | `global_accuracy` | KPI Modelo |
| F1 global | `global_f1` | KPI Modelo |
| Especificidad global | `global_specificity` | KPI Modelo |
| Varianza recall entre silos | `recall_variance` | non-IID impact |
| Ronda de convergencia | `convergence_round` | KPI Modelo |
| Tiempo de convergencia | `convergence_time_s` | KPI Modelo |

---

## Métricas NO capturables en este setup (y por qué)

| KPI | Razón |
|---|---|
| Packet loss real | TCP en WebSocket retransmite automáticamente; se necesitaría tcpdump/eBPF |
| Packet retransmission | Idem – nivel kernel, no accesible desde Python |
| Temperatura en Docker | `psutil.sensors_temperatures()` retorna vacío en contenedores Linux sin privilegios especiales |
| Número de saltos reales | Siempre 1 en centralizado (worker → server directo); para multi-hop necesitas arquitectura descentralizada |
| Impacto caída de nodos | Requiere experimento dedicado (matar un worker a mitad) |
| Variación entre clusters | N/A en centralizado puro (sin jerarquía) |

---

## Nota sobre Jitter

El jitter **no se mide por paquete** sino como la **desviación estándar de la latencia de recepción del modelo** a través de las rondas:

```python
jitter = df['local_receive_time_s'].std()   # por worker
```

Esto es el jitter a nivel de flujo (flow-level jitter), aceptable para un paper de FL.

## Nota sobre Número de Saltos

En arquitectura centralizada, **siempre es 1** (conexión directa worker → servidor). Este valor se reporta como constante y es un diferenciador clave frente a arquitecturas P2P/descentralizadas donde los saltos son variables.

## Fórmula del overhead de comunicación por ronda

```
Overhead = M × (N − 1)   [bytes]
M = tamaño del modelo serializado
N = número de nodos (silos)

En centralizado bidireccional:
Overhead_total = M × N  (server → N workers) + M × N (N workers → server)
              = 2 × M × N
```
