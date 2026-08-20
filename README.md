# FL-KPI-Edge-Benchmark

**KPI-driven benchmark of Federated Learning architectures for hospital silos and resource-constrained edge environments.**

This repository compares three Federated Learning topologies using the same siloed-data concept and multiple aggregation algorithms.

## Architectures

```text
1) Centralized FL
Clients ─────────────> Central Server ─────────────> Global Model

2) Semi-Decentralized FL
Workers <────────────> Aggregators <────────────> Coordinator

3) Hierarchical FL
Workers ─────────────> Intermediate Aggregators ─────────────> Global Aggregator
```

The repository contains:

```text
FL-KPI-Edge-Benchmark/
├── FL_Centralized_Hospital/
├── FL_Semi_Descentralized_Hospital/
└── FL_Hierarchical_Hospital/
```

## What is evaluated

The project is designed to study not only predictive performance, but also the systems cost of distributed learning.

Tracked or supported KPIs include:

- training metrics;
- CPU usage;
- RAM consumption;
- network I/O;
- latency;
- throughput;
- communication overhead;
- round-level statistics.

## Supported aggregation strategies

| Method | Purpose |
|---|---|
| **FedAvg** | Standard federated averaging |
| **FedProx** | Federated optimization with a proximal term for heterogeneous clients |
| **FedNova** | Normalized aggregation for heterogeneous local training |

## Technology stack

- Python
- PyTorch
- NumPy
- Pandas
- scikit-learn
- Joblib
- WebSockets

## Shared experiment design

The architectures are organized around hospital-style data silos. A shared preprocessing artifact is used to keep transformations consistent between participating nodes.

Example silo structure:

```text
data/
└── silos/
    ├── silo_1.csv
    ├── silo_2.csv
    ├── silo_3.csv
    ├── silo_4.csv
    └── silo_5.csv
```

## Centralized Federated Learning

A conventional Federated Learning topology with:

- one coordinating server;
- multiple training clients;
- server-side global aggregation;
- centralized orchestration and monitoring.

Typical workflow:

```text
Clients → Central Server → Global Aggregation → Updated Global Model
```

## Semi-Decentralized Federated Learning

A hybrid topology where participating agents can operate as:

- local trainers;
- aggregators;
- communication relays.

This architecture is designed to reduce dependency on a single central server while allowing internal aggregation and peer-to-peer communication.

Typical workflow:

```text
Workers ↔ Aggregators ↔ Coordinator
```

## Hierarchical Federated Learning

A multi-level topology with intermediate aggregation before updates reach the global level.

Typical workflow:

```text
Workers → Intermediate Aggregators → Global Aggregator
```

The implementation includes configuration for node IPs, ports and hierarchy levels.

## Installation

Python 3.10+ is recommended.

Each architecture contains its own dependencies.

```bash
cd FL_Centralized_Hospital
pip install -r requirements.txt
```

or:

```bash
cd FL_Semi_Descentralized_Hospital
pip install -r requirements.txt
```

or:

```bash
cd FL_Hierarchical_Hospital
pip install -r requirements.txt
```

## Configuration

Aggregation can be selected in the corresponding configuration, for example:

```python
AGGREGATION_METHOD = "FedAvg"
```

Available strategies include:

```text
FedAvg
FedProx
FedNova
```

## Research use cases

This repository is useful for experimentation in:

- Federated Learning;
- healthcare AI;
- distributed machine learning;
- edge/fog computing;
- non-IID data;
- communication-efficient learning;
- resource-constrained deployments.

## Authors

Research project developed at the **School of Mathematical and Computational Sciences, Yachay Tech University**, by:

- Carlos David Castro Rodriguez
- Ariel Pincay
- Joseph Tipan
- Freddy Valenzuela

## License

Educational and research use.
