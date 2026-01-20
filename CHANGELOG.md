# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2024-01-20

### Added

- **OPC UA Discovery**: Async tag discovery with read-only enforcement and configurable browse limits
- **ML-Powered Mapping**: CharCNN + GNN fusion architecture with temperature-scaled calibration
- **Swarm Consensus**: SWIM membership protocol + MQTT gossip with confidence-weighted voting
- **Federated Learning**: FedProx implementation with optional differential privacy via Flower
- **Dictionary Integration**: IEC CDD, eCl@ss, and curated seed providers for IRDI lookup
- **AAS Export**: BaSyx SDK integration supporting JSON, XML, and AASX package formats
- **REST API**: FastAPI-based API with OpenAPI documentation
- **Observability**: Prometheus metrics, structured logging, and correlation ID propagation
- **Gradio Dashboard**: Interactive UI for monitoring swarm status and mapping results
- **Docker Support**: Multi-container development stack with Docker Compose
- **Comprehensive Test Suite**: 870 tests covering unit and integration scenarios

### Performance

- OPC UA browse ≥1,000 nodes in <10 minutes
- CharCNN Macro F1 ≥ 0.80 on benchmark datasets
- 3-agent swarm converges on ≥90% of 500 tags in ≤30 seconds
- Flower completes 3 FL rounds with FedProx successfully

### Documentation

- Comprehensive README with architecture diagrams
- API endpoint documentation
- Configuration examples
- Development workflow guides

[Unreleased]: https://github.com/hadijannat/semantic-swarm-intelligence/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/hadijannat/semantic-swarm-intelligence/releases/tag/v0.1.0
