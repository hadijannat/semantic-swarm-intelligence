# NOA Semantic Swarm Mapper

Field-oriented toolkit for mapping brownfield tags to IEC 61987 (IRDI) semantics,
swarm consensus, and AAS export. The agent runs read-only against OPC UA and
shares hypotheses via MQTT/XMPP, with optional federated learning.

## Requirements

- Python 3.11+
- Poetry 1.7+

## Install

```bash
poetry install
poetry run pre-commit install
```

## Quickstart

```bash
# Start API (FastAPI)
poetry run uvicorn noa_swarm.api.main:application --reload

# Start a semantic agent
NOA_AGENT_ID=agent-001 NOA_OPCUA_ENDPOINT=opc.tcp://localhost:4840 \
MQTT_HOST=localhost MQTT_PORT=1883 \
poetry run python -m noa_swarm.swarm.agent
```

### Inference Modes

By default, the agent uses a rule-based mapper (ISA-style prefixes + dictionary search).
If model artifacts exist, it switches to the fusion engine.

- Checkpoint: `models/best_model.pt`
- Calibration: `models/calibration.json` (optional)
- Environment overrides:

```bash
NOA_ML_CHARCNN_CHECKPOINT=models/best_model.pt
NOA_ML_CALIBRATION_PATH=models/calibration.json
NOA_ML_USE_GNN=true
NOA_ML_ALLOW_UNTRAINED=false
```

### Train a baseline CharCNN

```bash
poetry run python -m noa_swarm.ml.training.train_local \
  --num-samples 10000 --epochs 8 --checkpoint-dir models --device cpu
```

## Development

```bash
make lint
make typecheck
make test
make check
```

## Project Structure

```
semantic-swarm-intelligence/
├── src/noa_swarm/         # Core package
│   ├── api/               # FastAPI service
│   ├── connectors/        # OPC UA/MQTT/FS
│   ├── ml/                # Datasets, models, serving
│   ├── services/          # Domain services
│   ├── storage/           # Repo layer (memory/SQL)
│   └── swarm/             # Agent + consensus
├── tests/                 # Test suite
├── models/                # Local model artifacts
└── pyproject.toml
```

## License

MIT
