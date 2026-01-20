"""Unit tests for Gradio dashboard application."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestGradioAppCreation:
    """Tests for Gradio app creation."""

    def test_create_dashboard_returns_blocks(self) -> None:
        """Test that create_dashboard returns a Gradio Blocks app."""
        from noa_swarm.ui.gradio_app import create_dashboard

        with patch("noa_swarm.ui.gradio_app.httpx") as mock_httpx:
            mock_httpx.AsyncClient.return_value.__aenter__ = AsyncMock()
            mock_httpx.AsyncClient.return_value.__aexit__ = AsyncMock()

            app = create_dashboard()

            # Verify it's a Gradio Blocks instance
            import gradio as gr
            assert isinstance(app, gr.Blocks)

    def test_create_dashboard_has_title(self) -> None:
        """Test that dashboard has correct title."""
        from noa_swarm.ui.gradio_app import create_dashboard

        app = create_dashboard()
        assert app.title == "NOA Semantic Swarm Mapper"


class TestDashboardTab:
    """Tests for the dashboard overview tab."""

    def test_format_metrics_with_data(self) -> None:
        """Test formatting metrics with actual data."""
        from noa_swarm.ui.gradio_app import format_metrics

        metrics = {
            "total_tags": 1000,
            "mapped_tags": 750,
            "pending_tags": 200,
            "verified_tags": 500,
            "rejected_tags": 50,
            "mapping_rate": 0.75,
        }

        result = format_metrics(metrics)

        # Accept both formatted (1,000) and raw (1000) numbers
        assert "1,000" in result or "1000" in result
        assert "750" in result
        assert "75" in result or "0.75" in result

    def test_format_metrics_empty(self) -> None:
        """Test formatting empty metrics."""
        from noa_swarm.ui.gradio_app import format_metrics

        result = format_metrics({})

        assert "No data" in result or "0" in result


class TestTagBrowserTab:
    """Tests for the tag browser tab."""

    def test_format_tags_table_with_data(self) -> None:
        """Test formatting tags as table data."""
        from noa_swarm.ui.gradio_app import format_tags_table

        tags = [
            {
                "tag_name": "TIC-101.PV",
                "browse_path": "/Objects/TIC-101/PV",
                "irdi": "0173-1#02-AAA123#001",
                "status": "mapped",
                "confidence": 0.95,
            },
            {
                "tag_name": "FIC-200.SP",
                "browse_path": "/Objects/FIC-200/SP",
                "irdi": None,
                "status": "pending",
                "confidence": None,
            },
        ]

        result = format_tags_table(tags)

        # Should return list of lists for Gradio dataframe
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0][0] == "TIC-101.PV"
        assert result[1][0] == "FIC-200.SP"

    def test_format_tags_table_empty(self) -> None:
        """Test formatting empty tags list."""
        from noa_swarm.ui.gradio_app import format_tags_table

        result = format_tags_table([])

        assert result == []


class TestMappingReviewTab:
    """Tests for the mapping review tab."""

    def test_format_candidates_with_data(self) -> None:
        """Test formatting mapping candidates."""
        from noa_swarm.ui.gradio_app import format_candidates

        candidates = [
            {
                "irdi": "0173-1#02-AAA123#001",
                "preferred_name": "Temperature",
                "confidence": 0.95,
                "source": "charcnn",
            },
            {
                "irdi": "0173-1#02-BBB456#001",
                "preferred_name": "Pressure",
                "confidence": 0.72,
                "source": "gnn",
            },
        ]

        result = format_candidates(candidates)

        assert isinstance(result, list)
        assert len(result) == 2
        assert "Temperature" in str(result[0])
        assert "0.95" in str(result[0]) or "95" in str(result[0])

    def test_format_candidates_empty(self) -> None:
        """Test formatting empty candidates."""
        from noa_swarm.ui.gradio_app import format_candidates

        result = format_candidates([])

        assert result == []


class TestSwarmMonitorTab:
    """Tests for the swarm monitor tab."""

    def test_format_agents_with_data(self) -> None:
        """Test formatting agent information."""
        from noa_swarm.ui.gradio_app import format_agents

        agents = [
            {
                "agent_id": "agent-001",
                "status": "active",
                "reliability_score": 0.92,
                "tags_processed": 150,
            },
            {
                "agent_id": "agent-002",
                "status": "inactive",
                "reliability_score": 0.85,
                "tags_processed": 100,
            },
        ]

        result = format_agents(agents)

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0][0] == "agent-001"
        assert result[0][1] == "active"

    def test_format_consensus_with_data(self) -> None:
        """Test formatting consensus records."""
        from noa_swarm.ui.gradio_app import format_consensus

        records = [
            {
                "tag_name": "TIC-101.PV",
                "irdi": "0173-1#02-AAA123#001",
                "confidence": 0.95,
                "agreement_ratio": 0.90,
                "participating_agents": 5,
            },
        ]

        result = format_consensus(records)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0][0] == "TIC-101.PV"


class TestExportTab:
    """Tests for the export tab."""

    def test_get_export_formats(self) -> None:
        """Test getting available export formats."""
        from noa_swarm.ui.gradio_app import get_export_formats

        formats = get_export_formats()

        assert "json" in formats
        assert "xml" in formats
        assert "aasx" in formats

    def test_validate_export_config_valid(self) -> None:
        """Test validating valid export config."""
        from noa_swarm.ui.gradio_app import validate_export_config

        config = {
            "aas_id": "urn:test:aas:1",
            "asset_id": "urn:test:asset:1",
            "format": "json",
        }

        is_valid, message = validate_export_config(config)

        assert is_valid is True
        assert message == ""

    def test_validate_export_config_missing_aas_id(self) -> None:
        """Test validating config with missing AAS ID."""
        from noa_swarm.ui.gradio_app import validate_export_config

        config = {
            "asset_id": "urn:test:asset:1",
            "format": "json",
        }

        is_valid, message = validate_export_config(config)

        assert is_valid is False
        assert "aas_id" in message.lower()


class TestFederatedTab:
    """Tests for the federated learning tab."""

    def test_format_training_status_active(self) -> None:
        """Test formatting active training status."""
        from noa_swarm.ui.gradio_app import format_training_status

        status = {
            "is_training": True,
            "current_round": 5,
            "total_rounds": 10,
            "participating_clients": 3,
        }

        result = format_training_status(status)

        assert "Training" in result or "Active" in result
        assert "5" in result
        assert "10" in result

    def test_format_training_status_idle(self) -> None:
        """Test formatting idle training status."""
        from noa_swarm.ui.gradio_app import format_training_status

        status = {
            "is_training": False,
            "current_round": 0,
            "total_rounds": 0,
            "participating_clients": 0,
        }

        result = format_training_status(status)

        assert "Idle" in result or "Not" in result or "stopped" in result.lower()


class TestAPIClient:
    """Tests for the API client helper."""

    @pytest.mark.asyncio
    async def test_fetch_data_success(self) -> None:
        """Test successful API fetch."""
        from noa_swarm.ui.gradio_app import APIClient

        with patch("noa_swarm.ui.gradio_app.httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"data": "test"}

            mock_client.return_value.__aenter__ = AsyncMock(
                return_value=mock_client.return_value
            )
            mock_client.return_value.__aexit__ = AsyncMock()
            mock_client.return_value.get = AsyncMock(return_value=mock_response)

            client = APIClient("http://localhost:8000")
            result = await client.fetch("/api/v1/test")

            assert result == {"data": "test"}

    @pytest.mark.asyncio
    async def test_fetch_data_error(self) -> None:
        """Test API fetch with error."""
        from noa_swarm.ui.gradio_app import APIClient

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        mock_async_client = AsyncMock()
        mock_async_client.get.return_value = mock_response

        with patch("noa_swarm.ui.gradio_app.httpx.AsyncClient") as mock_client_class:
            mock_client_class.return_value.__aenter__.return_value = mock_async_client
            mock_client_class.return_value.__aexit__.return_value = None

            client = APIClient("http://localhost:8000")

            with pytest.raises(Exception):
                await client.fetch("/api/v1/test")
