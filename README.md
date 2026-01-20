# NOA Semantic Swarm Mapper

A distributed system for automatically mapping brownfield industrial tags to PA-DIM / IEC 61987 (IRDI) semantics using swarm intelligence and federated learning. Built following the NAMUR Open Architecture (NOA) principles.

## Features

- **OPC UA Discovery**: Async tag discovery with read-only enforcement
- **ML-Powered Mapping**: CharCNN + GNN fusion with temperature-scaled calibration
- **Swarm Consensus**: SWIM membership + MQTT gossip with confidence-weighted voting
- **Federated Learning**: FedProx with optional differential privacy
- **Dictionary Integration**: IEC CDD, eCl@ss, and curated seed providers
- **AAS Export**: BaSyx SDK integration with JSON/XML/AASX formats
- **Full Observability**: Prometheus metrics, structured logging, correlation IDs

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         NOA Semantic Swarm                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                      │
│  │ Agent 1  │    │ Agent 2  │    │ Agent 3  │   Semantic Agents    │
│  │ CharCNN  │    │ CharCNN  │    │ CharCNN  │   (discover, infer,  │
│  │ + GNN    │    │ + GNN    │    │ + GNN    │    vote, commit)     │
│  └────┬─────┘    └────┬─────┘    └────┬─────┘                      │
│       │               │               │                             │
│       └───────────────┼───────────────┘                             │
│                       │                                             │
│                       ▼                                             │
│              ┌────────────────┐                                     │
│              │  MQTT Broker   │  Gossip + Consensus                 │
│              │  (Mosquitto)   │                                     │
│              └────────┬───────┘                                     │
│                       │                                             │
│       ┌───────────────┼───────────────┐                             │
│       ▼               ▼               ▼                             │
│  ┌─────────┐    ┌──────────┐    ┌──────────┐                       │
│  │ OPC UA  │    │ Flower   │    │   AAS    │                       │
│  │ Server  │    │ Server   │    │ Registry │                       │
│  └─────────┘    └──────────┘    └──────────┘                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Requirements

- Python 3.11+
- Poetry 1.7+
- Docker & Docker Compose (for full stack)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/semantic-swarm-intelligence.git
cd semantic-swarm-intelligence

# Install dependencies
poetry install

# Install pre-commit hooks
poetry run pre-commit install
```

## Quickstart

### Option 1: Docker Compose (Recommended)

Start the full development stack with a single command:

```bash
make docker-up
```

This starts:
- **API Server**: http://localhost:8000 (FastAPI with OpenAPI docs at `/docs`)
- **Gradio Dashboard**: http://localhost:7860
- **Flower Server**: http://localhost:8080 (Federated learning)
- **MQTT Broker**: localhost:1883
- **PostgreSQL**: localhost:5432
- **3 Semantic Agents**: Pre-configured and connected

Stop the stack:
```bash
make docker-down
```

### Option 2: Local Development

```bash
# Start the API server
poetry run uvicorn noa_swarm.api.main:application --reload

# In another terminal, start a semantic agent
NOA_AGENT_ID=agent-001 \
NOA_OPCUA_ENDPOINT=opc.tcp://localhost:4840 \
MQTT_HOST=localhost \
poetry run python -m noa_swarm.swarm.agent
```

### ML Inference Modes

The agent supports two inference modes:

1. **Rule-based** (default): ISA-style prefix parsing + dictionary search
2. **ML Fusion**: CharCNN + GNN with calibrated confidence scores

To enable ML inference, provide model artifacts:

```bash
# Train a baseline model
make train

# Or specify paths via environment
NOA_ML_CHARCNN_CHECKPOINT=models/best_model.pt
NOA_ML_CALIBRATION_PATH=models/calibration.json
NOA_ML_USE_GNN=true
```

## Development

```bash
# Run linting
make lint

# Format code
make format

# Type checking
make typecheck

# Run all checks
make check

# Clean cache files
make clean
```

## Testing

```bash
# Run all tests (870 tests)
make test

# Run with coverage
make test-cov

# Unit tests only
make test-unit

# Integration tests only
make test-int
```

### ML Reproducibility

Verify deterministic training with fixed seeds:

```bash
# Quick benchmark (reduced samples)
make benchmark

# Full reproducibility test
make reproduce
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/metrics` | GET | Prometheus metrics |
| `/api/v1/discovery/tags` | GET | List discovered tags |
| `/api/v1/discovery/start` | POST | Start OPC UA discovery |
| `/api/v1/mapping/hypotheses` | GET | List mapping hypotheses |
| `/api/v1/mapping/consensus` | GET | Get consensus records |
| `/api/v1/aas/submodel` | GET | Export AAS submodel |
| `/api/v1/aas/export` | POST | Export to AASX package |
| `/api/v1/swarm/agents` | GET | List active agents |
| `/api/v1/swarm/status` | GET | Swarm status |
| `/api/v1/federated/status` | GET | FL round status |

Full OpenAPI documentation available at `/docs` when the API is running.

## Project Structure

```
semantic-swarm-intelligence/
├── src/noa_swarm/
│   ├── aas/                 # AAS export (BaSyx SDK)
│   ├── api/                 # FastAPI application
│   │   └── routes/          # REST API endpoints
│   ├── common/              # Config, logging, schemas, IRDI
│   ├── connectors/          # OPC UA, MQTT, filesystem
│   ├── dictionaries/        # IEC CDD, eCl@ss, seed providers
│   ├── federated/           # Flower client/server, FedProx, DP
│   ├── ml/
│   │   ├── datasets/        # TEP, C-MAPSS, synthetic
│   │   ├── models/          # CharCNN, GNN, fusion
│   │   ├── serving/         # Inference engine
│   │   └── training/        # Local training, calibration
│   ├── observability/       # Metrics, correlation IDs
│   ├── services/            # Domain services
│   ├── storage/             # Repository layer
│   ├── swarm/               # Agent, consensus, reputation
│   └── ui/                  # Gradio dashboard
├── docker/
│   ├── Dockerfile.agent     # Semantic agent image
│   ├── Dockerfile.server    # API/Flower server image
│   └── docker-compose.dev.yml
├── configs/
│   ├── dev.yaml             # Development settings
│   └── example_plant.yaml   # Example plant configuration
├── scripts/
│   ├── train_baseline.py    # ML training script
│   ├── run_benchmarks.sh    # Reproducibility tests
│   ├── download_datasets.py # Dataset downloader
│   └── export_aas.py        # CLI AAS export
├── tests/
│   ├── unit/                # Unit tests
│   └── integration/         # Integration tests
├── models/                  # Model artifacts (gitignored)
├── pyproject.toml
├── Makefile
└── README.md
```

## Configuration

Configuration via environment variables or YAML files:

```yaml
# configs/dev.yaml
noa:
  agent_id: "agent-001"
  opcua:
    endpoint: "opc.tcp://localhost:4840"
    max_nodes_per_browse: 1000
  mqtt:
    host: "localhost"
    port: 1883
  ml:
    checkpoint_path: "models/best_model.pt"
    use_gnn: true
  consensus:
    min_votes: 2
    hard_quorum_threshold: 0.8
```

## Success Criteria

| Metric | Target | Status |
|--------|--------|--------|
| OPC UA browse ≥1,000 nodes in <10 min | ✓ | Achieved |
| CharCNN Macro F1 ≥ 0.80 | ✓ | Achieved |
| 3-agent swarm converges on ≥90% of 500 tags in ≤30s | ✓ | Achieved |
| Flower completes 3 FL rounds with FedProx | ✓ | Achieved |
| AAS export loadable by BaSyx tooling | ✓ | Achieved |
| `make reproduce` generates benchmark table | ✓ | Achieved |

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`make check`)
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- NAMUR Open Architecture (NOA) working group
- IEC 61987 / PA-DIM standardization efforts
- eCl@ss and IEC CDD dictionary providers
- Eclipse BaSyx project
