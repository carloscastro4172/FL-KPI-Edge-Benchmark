# Federated Learning Hospital Architectures

This repository contains three Federated Learning (FL) architectures designed for hospital-based distributed machine learning experiments using siloed healthcare datasets.

The project compares:

1. **Centralized Federated Learning**
2. **Semi-Decentralized Federated Learning**
3. **Hierarchical Federated Learning**

All architectures use the same hospital dataset partitions (`silos`) and support multiple aggregation strategies such as:

- FedAvg
- FedProx
- FedNova

---

# Repository Structure

```text
FL/
├── FL_Centralized_Hospital/
├── FL_Semi_Descentralized_Hospital/
└── FL_Hierarchical_Hospital/
```

---

# Architectures Overview

## 1. Centralized FL

Traditional Federated Learning architecture with:

- One central server
- Multiple client silos
- Global aggregation on the server

### Characteristics

- Simple orchestration
- Centralized coordination
- Easier monitoring
- Lower communication complexity

---

## 2. Semi-Decentralized FL

Hybrid architecture where agents can:

- Train locally
- Act as aggregators
- Exchange models peer-to-peer

### Characteristics

- Reduced central dependency
- Internal aggregation rounds
- Improved resilience
- Better scalability

---

## 3. Hierarchical FL

Multi-level Federated Learning architecture.

### Characteristics

- Hierarchical aggregation
- Node clusters
- Distributed coordination
- Reduced global communication overhead

---

# Requirements

## Python

Recommended:

- Python 3.10+
- pip

---

# Installation


# Create Virtual Environment

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows:

```powershell
python -m venv venv
venv\Scripts\activate
```

---

# Install Dependencies

Each architecture has its own `requirements.txt`.

## Centralized

```bash
cd FL_Centralized_Hospital
pip install -r requirements.txt
```

## Semi-Decentralized

```bash
cd FL_Semi_Descentralized_Hospital
pip install -r requirements.txt
```

## Hierarchical

```bash
cd FL_Hierarchical_Hospital
pip install -r requirements.txt
```

---

# Shared Dataset Structure

The dataset is divided into hospital silos:

```text
data/
├── silos/
│   ├── silo_1.csv
│   ├── silo_2.csv
│   ├── silo_3.csv
│   ├── silo_4.csv
│   └── silo_5.csv
```

Additional files:

- `preprocessor_global.joblib`
- raw CSV datasets

---

# Supported Aggregation Methods

All architectures support:

| Method | Description |
|---|---|
| FedAvg | Standard federated averaging |
| FedProx | Handles data heterogeneity |
| FedNova | Normalized aggregation |

Configuration examples:

```python
AGGREGATION_METHOD = 'FedAvg'
```

or

```python
AGGREGATION_METHOD = 'FedProx'
```

or

```python
AGGREGATION_METHOD = 'FedNova'
```

---

# 1. Centralized Federated Learning

Folder:

```text
FL_Centralized_Hospital/
```

## Main Components

| File | Purpose |
|---|---|
| `metrics_analysis.py` | Metrics evaluation |
| `analize_results.py` | Result analysis |
| `collect_logs.sh` | Log collection |
| `test_save_model.py` | Model persistence test |

---

## Centralized Workflow

```text
Clients → Central Server → Global Aggregation → Updated Global Model
```

---

## Running the Centralized Architecture

## Step 1 — Enter the Folder

```bash
cd FL_Centralized_Hospital
```

---

## Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Step 3 — Verify Dataset Paths

Ensure the silos exist:

```text
data/silos/
```

---

## Step 4 — Run Training

Depending on your implementation entry point:

```bash
python metrics_analysis.py
```

or

```bash
python analize_results.py
```

---

## Centralized Configuration

Key hyperparameters:

| Parameter | Value |
|---|---|
| Epochs | 3 |
| Batch Size | 16 |
| Learning Rate | 0.01 |
| Rounds | 50 |

---

# 2. Semi-Decentralized Federated Learning

Folder:

```text
FL_Semi_Descentralized_Hospital/
```

---

## Semi-Decentralized Architecture

```text
Workers ↔ Aggregators ↔ Coordinator
```

Agents can behave as:

- Local trainers
- Aggregators
- Communication relays

---

## Main Files

| File | Purpose |
|---|---|
| `agent_csv.py` | Main agent execution |
| `config.py` | Architecture configuration |
| `run_agent.sh` | Agent launcher |
| `test_save_model.py` | Model save validation |

---

## Running the Semi-Decentralized System

## Step 1 — Enter Folder

```bash
cd FL_Semi_Descentralized_Hospital
```

---

## Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Step 3 — Configure Aggregation

Edit:

```python
AGGREGATION_METHOD = 'FedAvg'
```

Inside:

```text
config.py
```

---

## Step 4 — Launch Coordinator

Example:

```bash
python agent_csv.py
```

---

## Step 5 — Launch Agents

Open multiple terminals:

```bash
bash run_agent.sh
```

---

## Important Parameters

| Parameter | Description |
|---|---|
| `LOCAL_EPOCHS` | Local training epochs |
| `INTERNAL_ROUNDS` | Internal aggregation rounds |
| `ROUND_TIMEOUT` | Aggregation timeout |
| `MAX_ROUNDS` | Maximum FL rounds |

---

## Semi-Decentralized Features

- Peer-to-peer communication
- Internal aggregation
- Reduced server bottleneck
- Better fault tolerance

---

# 3. Hierarchical Federated Learning

Folder:

```text
FL_Hierarchical_Hospital/
```

---

## Hierarchical Workflow

```text
Workers → Intermediate Aggregators → Global Aggregator
```

---

## Main Files

| File | Purpose |
|---|---|
| `main.py` | Main execution entry |
| `federated.py` | Federated orchestration |
| `config.py` | Global configuration |
| `connections/` | Networking layer |
| `model/` | Neural network definitions |

---

## Running the Hierarchical Architecture

## Step 1 — Enter Folder

```bash
cd FL_Hierarchical_Hospital
```

---

## Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Step 3 — Configure Nodes

Review:

```text
nodes.json
```

Configure:

- node IPs
- ports
- hierarchy levels

---

## Step 4 — Configure Dataset Path

Inside `config.py`:

```python
DATA_PATH = "path/to/silo.csv"
```

---

## Step 5 — Configure Aggregation

```python
AGGREGATION_METHOD = 'FedAvg'
```

---

## Step 6 — Run Main Server

```bash
python main.py
```

---

## Hierarchical Features

- Cluster-based aggregation
- Communication optimization
- Reduced bandwidth usage
- Better scalability for large systems

---

# Metrics and Monitoring

The project includes support for:

- CPU usage
- RAM consumption
- Network I/O
- Latency
- Throughput
- Communication overhead
- Training metrics

Logs are stored in:

```text
logs/
```

---

# Model Configuration

The architectures use PyTorch neural networks.

Example parameters:

| Parameter | Value |
|---|---|
| Hidden Layer 1 | 120 |
| Hidden Layer 2 | 84 |
| Dropout | 0.3 |
| Learning Rate | 0.01 |

---

# Preprocessing

All architectures use a shared preprocessing artifact:

```text
preprocessor_global.joblib
```

This guarantees:

- Consistent feature engineering
- Unified normalization
- Reproducible transformations

---

# Experimental Setup

The project is designed for:

- Hospital silo simulation
- Distributed machine learning
- Edge/Fog computing research
- Federated Learning benchmarking

---

# Example Execution Flow

## Hierarchical Example

### Terminal 1

```bash
python main.py
```

### Terminal 2+

```bash
python federated.py
```

---

# Logs and Outputs

Generated outputs include:

- Global models
- Aggregated metrics
- Communication logs
- Round statistics

Output folders:

```text
models/
logs/
received_files/
```

---

# Troubleshooting

## Port Already in Use

Kill the process:

Linux/macOS:

```bash
lsof -i :8765
kill -9 <PID>
```

Windows:

```powershell
netstat -ano | findstr :8765
taskkill /PID <PID> /F
```

---

## Missing Dependencies

```bash
pip install -r requirements.txt
```

---

## WebSocket Errors

Verify:

- firewall rules
- ports
- IP addresses
- active listeners

---

# Research Context

This project is useful for:

- Federated Learning research
- Healthcare AI
- Distributed AI systems
- Multi-silo machine learning
- Edge AI experimentation

---

# Technologies Used

- Python
- PyTorch
- WebSockets
- NumPy
- Pandas
- Scikit-learn
- Joblib

---

# Future Improvements

Possible extensions:

- Differential Privacy
- Secure Aggregation
- Homomorphic Encryption
- Adaptive Aggregation
- Dynamic Node Discovery

---

# Authors

```latex
\author{
\IEEEauthorblockN{Carlos David Castro Rodriguez}
\IEEEauthorblockA{\textit{School of Mathematical and Computational Sciences} \\
\textit{Yachay Tech University}\\
Urcuquí, Ecuador\\
carlos.castro@yachaytech.edu.ec}

\and
\IEEEauthorblockN{Ariel Pincay}
\IEEEauthorblockA{\textit{School of Mathematical and Computational Sciences} \\
\textit{Yachay Tech University}\\
Urcuquí, Ecuador \\
ariel.pincay@yachaytech.edu.ec}
\linebreakand
\IEEEauthorblockN{Joseph Tipan}
\IEEEauthorblockA{\textit{School of Mathematical and Computational Sciences} \\
\textit{Yachay Tech University}\\
Urcuquí, Ecuador \\
joseph.tipan@yachaytech.edu.ec}
\and
\IEEEauthorblockN{Freddy Valenzuela}
\IEEEauthorblockA{\textit{School of Mathematical and Computational Sciences} \\
\textit{Yachay Tech University}\\
Urcuquí, Ecuador \\
freddy.valenzuela@yachaytech.edu.ec}
}
```

---

# License

Educational and research use.
