"""Unit tests for API route handlers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient

from noa_swarm.api.main import app

if TYPE_CHECKING:
    pass


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the FastAPI app."""
    return TestClient(app)


class TestDiscoveryRoutes:
    """Tests for discovery routes."""

    def test_get_discovery_status(self, client: TestClient) -> None:
        """Test getting discovery status."""
        response = client.get("/api/v1/discovery/status")
        assert response.status_code == 200

        data = response.json()
        assert "is_running" in data
        assert "progress" in data
        assert "discovered_count" in data

    def test_start_discovery(self, client: TestClient) -> None:
        """Test starting discovery."""
        response = client.post(
            "/api/v1/discovery/start",
            params={"server_url": "opc.tcp://localhost:4840"},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "started"

    def test_stop_discovery(self, client: TestClient) -> None:
        """Test stopping discovery."""
        response = client.post("/api/v1/discovery/stop")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "stopped"

    def test_get_discovered_nodes(self, client: TestClient) -> None:
        """Test getting discovered nodes."""
        response = client.get("/api/v1/discovery/nodes")
        assert response.status_code == 200

        data = response.json()
        assert "nodes" in data
        assert "total" in data

    def test_get_node_details_not_found(self, client: TestClient) -> None:
        """Test getting node details for nonexistent node."""
        response = client.get("/api/v1/discovery/nodes/nonexistent")
        assert response.status_code == 404


class TestMappingRoutes:
    """Tests for mapping routes."""

    def test_list_mappings(self, client: TestClient) -> None:
        """Test listing mappings."""
        response = client.get("/api/v1/mapping/")
        assert response.status_code == 200

        data = response.json()
        assert "mappings" in data
        assert "total" in data
        assert "stats" in data

    def test_create_mapping(self, client: TestClient) -> None:
        """Test creating a mapping."""
        response = client.post(
            "/api/v1/mapping/",
            json={
                "tag_name": "TIC-101.PV",
                "browse_path": "/Objects/TIC-101/PV",
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert data["tag_name"] == "TIC-101.PV"
        assert data["status"] == "pending"

    def test_get_mapping_not_found(self, client: TestClient) -> None:
        """Test getting mapping for nonexistent tag."""
        response = client.get("/api/v1/mapping/nonexistent")
        assert response.status_code == 404


class TestAASRoutes:
    """Tests for AAS routes."""

    def test_get_submodel_info(self, client: TestClient) -> None:
        """Test getting submodel info."""
        response = client.get("/api/v1/aas/submodel")
        assert response.status_code == 200

        data = response.json()
        assert "submodel_id" in data
        assert "tag_count" in data
        assert "statistics" in data

    def test_get_submodel_json(self, client: TestClient) -> None:
        """Test getting submodel as JSON."""
        response = client.get("/api/v1/aas/submodel/json")
        assert response.status_code == 200

        data = response.json()
        assert "assetAdministrationShells" in data or "submodels" in data

    def test_list_export_formats(self, client: TestClient) -> None:
        """Test listing export formats."""
        response = client.get("/api/v1/aas/formats")
        assert response.status_code == 200

        data = response.json()
        assert len(data) == 3  # json, xml, aasx
        formats = [f["format"] for f in data]
        assert "json" in formats
        assert "xml" in formats
        assert "aasx" in formats

    def test_export_json(self, client: TestClient) -> None:
        """Test exporting to JSON."""
        response = client.post(
            "/api/v1/aas/export/json",
            json={
                "aas_id": "urn:test:aas:1",
                "asset_id": "urn:test:asset:1",
            },
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"


class TestSwarmRoutes:
    """Tests for swarm routes."""

    def test_get_swarm_status(self, client: TestClient) -> None:
        """Test getting swarm status."""
        response = client.get("/api/v1/swarm/status")
        assert response.status_code == 200

        data = response.json()
        assert "total_agents" in data
        assert "active_agents" in data

    def test_list_agents(self, client: TestClient) -> None:
        """Test listing agents."""
        response = client.get("/api/v1/swarm/agents")
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, list)

    def test_list_consensus_records(self, client: TestClient) -> None:
        """Test listing consensus records."""
        response = client.get("/api/v1/swarm/consensus")
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, list)

    def test_trigger_consensus(self, client: TestClient) -> None:
        """Test triggering consensus."""
        response = client.post(
            "/api/v1/swarm/trigger",
            json=["TIC-101.PV", "FIC-200.SP"],
        )
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "triggered"


class TestFederatedRoutes:
    """Tests for federated learning routes."""

    def test_get_federated_status(self, client: TestClient) -> None:
        """Test getting federated status."""
        response = client.get("/api/v1/federated/status")
        assert response.status_code == 200

        data = response.json()
        assert "is_training" in data
        assert "current_round" in data

    def test_start_training(self, client: TestClient) -> None:
        """Test starting federated training."""
        response = client.post(
            "/api/v1/federated/start",
            json={
                "num_rounds": 5,
                "min_clients": 2,
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "started"

    def test_stop_training(self, client: TestClient) -> None:
        """Test stopping federated training."""
        response = client.post("/api/v1/federated/stop")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "stopped"

    def test_list_rounds(self, client: TestClient) -> None:
        """Test listing training rounds."""
        response = client.get("/api/v1/federated/rounds")
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, list)

    def test_list_clients(self, client: TestClient) -> None:
        """Test listing federated clients."""
        response = client.get("/api/v1/federated/clients")
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, list)

    def test_get_model_metrics(self, client: TestClient) -> None:
        """Test getting model metrics."""
        response = client.get("/api/v1/federated/model/metrics")
        assert response.status_code == 200

        data = response.json()
        assert "accuracy" in data
        assert "f1_score" in data

    def test_get_privacy_budget(self, client: TestClient) -> None:
        """Test getting privacy budget."""
        response = client.get("/api/v1/federated/privacy")
        assert response.status_code == 200

        data = response.json()
        assert "dp_enabled" in data
