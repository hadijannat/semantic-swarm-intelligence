"""Gradio dashboard for NOA Semantic Swarm Mapper.

This module provides a web-based dashboard for:
- Monitoring tag discovery and mapping progress
- Browsing and filtering tags
- Reviewing and approving mappings
- Monitoring swarm agent status and consensus
- Exporting AAS packages
- Managing federated learning sessions
"""

from __future__ import annotations

from typing import Any, cast

import httpx

# Compatibility shim for older gradio expectations when huggingface_hub removes HfFolder.
try:
    import huggingface_hub

    module = cast(Any, huggingface_hub)
    if not hasattr(module, "HfFolder"):
        class HfFolder:
            """Fallback HfFolder stub for gradio imports."""

            @staticmethod
            def get_token() -> str | None:  # pragma: no cover - compatibility shim
                return None

        module.HfFolder = HfFolder
except Exception:
    # If huggingface_hub is unavailable, gradio will handle import errors later.
    pass

import gradio as gr

from noa_swarm.common.logging import get_logger

logger = get_logger(__name__)

# Default API base URL
DEFAULT_API_URL = "http://localhost:8000"


class APIClient:
    """HTTP client for communicating with the API server."""

    def __init__(self, base_url: str = DEFAULT_API_URL) -> None:
        """Initialize the API client.

        Args:
            base_url: Base URL of the API server.
        """
        self.base_url = base_url.rstrip("/")

    async def fetch(self, endpoint: str) -> dict[str, Any] | list[dict[str, Any]]:
        """Fetch data from an API endpoint.

        Args:
            endpoint: API endpoint path.

        Returns:
            JSON response data.

        Raises:
            Exception: If the request fails.
        """
        url = f"{self.base_url}{endpoint}"
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            if response.status_code != 200:
                raise Exception(f"API error: {response.status_code} - {response.text}")
            return cast(dict[str, Any] | list[dict[str, Any]], response.json())

    async def post(
        self, endpoint: str, data: dict[str, Any] | None = None
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Post data to an API endpoint.

        Args:
            endpoint: API endpoint path.
            data: JSON data to post.

        Returns:
            JSON response data.

        Raises:
            Exception: If the request fails.
        """
        url = f"{self.base_url}{endpoint}"
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=data)
            if response.status_code not in (200, 201):
                raise Exception(f"API error: {response.status_code} - {response.text}")
            return cast(dict[str, Any] | list[dict[str, Any]], response.json())


# =============================================================================
# Formatting Functions
# =============================================================================


def format_metrics(metrics: dict[str, Any]) -> str:
    """Format metrics data as markdown.

    Args:
        metrics: Dictionary of metric values.

    Returns:
        Formatted markdown string.
    """
    if not metrics:
        return "## Metrics\n\nNo data available."

    total = metrics.get("total_tags", 0)
    mapped = metrics.get("mapped_tags", 0)
    pending = metrics.get("pending_tags", 0)
    verified = metrics.get("verified_tags", 0)
    rejected = metrics.get("rejected_tags", 0)
    rate = metrics.get("mapping_rate", 0)

    # Format as percentage
    rate_pct = f"{rate * 100:.1f}%" if rate else "0%"

    return f"""## Tag Mapping Metrics

| Metric | Value |
|--------|-------|
| Total Tags | {total:,} |
| Mapped Tags | {mapped:,} |
| Pending Tags | {pending:,} |
| Verified Tags | {verified:,} |
| Rejected Tags | {rejected:,} |
| **Mapping Rate** | **{rate_pct}** |
"""


def format_tags_table(tags: list[dict[str, Any]]) -> list[list[Any]]:
    """Format tags as table data for Gradio dataframe.

    Args:
        tags: List of tag dictionaries.

    Returns:
        List of lists for Gradio dataframe.
    """
    if not tags:
        return []

    result = []
    for tag in tags:
        row = [
            tag.get("tag_name", ""),
            tag.get("browse_path", ""),
            tag.get("irdi") or "-",
            tag.get("status", "unknown"),
            f"{tag.get('confidence', 0):.2f}" if tag.get("confidence") else "-",
        ]
        result.append(row)

    return result


def format_candidates(candidates: list[dict[str, Any]]) -> list[list[Any]]:
    """Format mapping candidates as table data.

    Args:
        candidates: List of candidate dictionaries.

    Returns:
        List of lists for Gradio dataframe.
    """
    if not candidates:
        return []

    result = []
    for candidate in candidates:
        row = [
            candidate.get("irdi", ""),
            candidate.get("preferred_name", ""),
            f"{candidate.get('confidence', 0):.2f}",
            candidate.get("source", ""),
        ]
        result.append(row)

    return result


def format_agents(agents: list[dict[str, Any]]) -> list[list[Any]]:
    """Format agent information as table data.

    Args:
        agents: List of agent dictionaries.

    Returns:
        List of lists for Gradio dataframe.
    """
    if not agents:
        return []

    result = []
    for agent in agents:
        row = [
            agent.get("agent_id", ""),
            agent.get("status", "unknown"),
            f"{agent.get('reliability_score', 0):.2f}",
            agent.get("tags_processed", 0),
        ]
        result.append(row)

    return result


def format_consensus(records: list[dict[str, Any]]) -> list[list[Any]]:
    """Format consensus records as table data.

    Args:
        records: List of consensus record dictionaries.

    Returns:
        List of lists for Gradio dataframe.
    """
    if not records:
        return []

    result = []
    for record in records:
        row = [
            record.get("tag_name", ""),
            record.get("irdi", ""),
            f"{record.get('confidence', 0):.2f}",
            f"{record.get('agreement_ratio', 0):.2f}",
            record.get("participating_agents", 0),
        ]
        result.append(row)

    return result


def format_training_status(status: dict[str, Any]) -> str:
    """Format training status as markdown.

    Args:
        status: Training status dictionary.

    Returns:
        Formatted markdown string.
    """
    is_training = status.get("is_training", False)
    current_round = status.get("current_round", 0)
    total_rounds = status.get("total_rounds", 0)
    clients = status.get("participating_clients", 0)

    if is_training:
        state = "🟢 **Active Training**"
        progress = f"Round {current_round} / {total_rounds}"
    else:
        state = "⚪ **Idle** (Not training)"
        progress = "No active training session"

    return f"""## Federated Learning Status

{state}

| Parameter | Value |
|-----------|-------|
| Progress | {progress} |
| Current Round | {current_round} |
| Total Rounds | {total_rounds} |
| Participating Clients | {clients} |
"""


def get_export_formats() -> list[str]:
    """Get available export formats.

    Returns:
        List of format names.
    """
    return ["json", "xml", "aasx"]


def validate_export_config(config: dict[str, Any]) -> tuple[bool, str]:
    """Validate export configuration.

    Args:
        config: Export configuration dictionary.

    Returns:
        Tuple of (is_valid, error_message).
    """
    required_fields = ["aas_id", "asset_id", "format"]

    for field in required_fields:
        if field not in config or not config[field]:
            return False, f"Missing required field: {field}"

    if config.get("format") not in get_export_formats():
        return False, f"Invalid format: {config.get('format')}"

    return True, ""


# =============================================================================
# Dashboard Creation
# =============================================================================


def create_dashboard(api_url: str = DEFAULT_API_URL) -> gr.Blocks:
    """Create the Gradio dashboard application.

    Args:
        api_url: Base URL for the API server.

    Returns:
        Gradio Blocks application.
    """
    client = APIClient(api_url)

    with gr.Blocks(
        title="NOA Semantic Swarm Mapper",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown("# NOA Semantic Swarm Mapper")
        gr.Markdown("Distributed semantic tag mapping using swarm intelligence")

        with gr.Tabs():
            # =================================================================
            # Dashboard Tab
            # =================================================================
            with gr.Tab("Dashboard"):
                gr.Markdown("## Overview")

                with gr.Row():
                    with gr.Column(scale=2):
                        metrics_display = gr.Markdown("Loading metrics...")

                    with gr.Column(scale=1):
                        discovery_status = gr.Markdown("Loading discovery status...")

                refresh_dashboard_btn = gr.Button("Refresh", variant="secondary")

                async def refresh_dashboard() -> tuple[str, str]:
                    try:
                        # Fetch metrics from API
                        mapping_payload = await client.fetch("/api/v1/mapping/")
                        discovery_payload = await client.fetch("/api/v1/discovery/status")

                        mapping_data = (
                            mapping_payload if isinstance(mapping_payload, dict) else {}
                        )
                        discovery_data = (
                            discovery_payload if isinstance(discovery_payload, dict) else {}
                        )

                        metrics = {
                            "total_tags": mapping_data.get("total", 0),
                            "mapped_tags": mapping_data.get("stats", {}).get("mapped", 0),
                            "pending_tags": mapping_data.get("stats", {}).get("pending", 0),
                            "verified_tags": mapping_data.get("stats", {}).get("verified", 0),
                            "rejected_tags": mapping_data.get("stats", {}).get("rejected", 0),
                            "mapping_rate": (
                                mapping_data.get("stats", {}).get("mapped", 0)
                                / max(mapping_data.get("total", 1), 1)
                            ),
                        }

                        discovery = f"""## Discovery Status

| Status | Value |
|--------|-------|
| Running | {discovery_data.get('is_running', False)} |
| Progress | {discovery_data.get('progress', 0):.1%} |
| Discovered | {discovery_data.get('discovered_count', 0)} |
"""
                        return format_metrics(metrics), discovery
                    except Exception as e:
                        logger.error("Failed to refresh dashboard", error=str(e))
                        return f"Error: {e}", f"Error: {e}"

                refresh_dashboard_btn.click(
                    fn=refresh_dashboard,
                    outputs=[metrics_display, discovery_status],
                )

            # =================================================================
            # Tag Browser Tab
            # =================================================================
            with gr.Tab("Tag Browser"):
                gr.Markdown("## Browse and Filter Tags")

                with gr.Row():
                    status_filter = gr.Dropdown(
                        choices=["all", "pending", "mapped", "verified", "rejected", "conflict"],
                        value="all",
                        label="Status Filter",
                    )
                    search_input = gr.Textbox(
                        label="Search",
                        placeholder="Search by tag name...",
                    )
                    search_btn = gr.Button("Search", variant="primary")

                tags_table = gr.Dataframe(
                    headers=["Tag Name", "Browse Path", "IRDI", "Status", "Confidence"],
                    datatype=["str", "str", "str", "str", "str"],
                    label="Tags",
                )

                async def search_tags(status: str, query: str) -> list[list[Any]]:
                    try:
                        endpoint = "/api/v1/mapping/"
                        if status != "all":
                            endpoint += f"?status={status}"

                        payload = await client.fetch(endpoint)
                        data = payload if isinstance(payload, dict) else {}
                        raw_tags = data.get("mappings", [])
                        tags = (
                            [item for item in raw_tags if isinstance(item, dict)]
                            if isinstance(raw_tags, list)
                            else []
                        )

                        # Filter by query if provided
                        if query:
                            tags = [
                                t for t in tags
                                if query.lower() in t.get("tag_name", "").lower()
                            ]

                        return format_tags_table(tags)
                    except Exception as e:
                        logger.error("Failed to search tags", error=str(e))
                        return []

                search_btn.click(
                    fn=search_tags,
                    inputs=[status_filter, search_input],
                    outputs=[tags_table],
                )

            # =================================================================
            # Mapping Review Tab
            # =================================================================
            with gr.Tab("Mapping Review"):
                gr.Markdown("## Review and Approve Mappings")

                with gr.Row():
                    tag_selector = gr.Textbox(
                        label="Tag Name",
                        placeholder="Enter tag name to review...",
                    )
                    load_candidates_btn = gr.Button("Load Candidates", variant="primary")

                with gr.Row():
                    with gr.Column():
                        candidates_table = gr.Dataframe(
                            headers=["IRDI", "Preferred Name", "Confidence", "Source"],
                            datatype=["str", "str", "str", "str"],
                            label="Candidates",
                        )

                    with gr.Column():
                        selected_irdi = gr.Textbox(label="Selected IRDI")
                        with gr.Row():
                            approve_btn = gr.Button("Approve", variant="primary")
                            reject_btn = gr.Button("Reject", variant="stop")
                        action_result = gr.Markdown("")

                async def load_candidates(tag_name: str) -> list[list[Any]]:
                    if not tag_name:
                        return []
                    try:
                        data = await client.fetch(f"/api/v1/mapping/{tag_name}/candidates")
                        candidates: list[dict[str, Any]]
                        if isinstance(data, list):
                            candidates = [item for item in data if isinstance(item, dict)]
                        elif isinstance(data, dict):
                            raw = data.get("candidates")
                            if isinstance(raw, list):
                                candidates = [item for item in raw if isinstance(item, dict)]
                            else:
                                candidates = []
                        else:
                            candidates = []
                        return format_candidates(candidates)
                    except Exception as e:
                        logger.error("Failed to load candidates", error=str(e))
                        return []

                async def approve_mapping(tag_name: str, irdi: str) -> str:
                    if not tag_name:
                        return "❌ Please enter a tag name"
                    _ = irdi
                    try:
                        await client.post(f"/api/v1/mapping/{tag_name}/approve")
                        return f"✅ Mapping approved for {tag_name}"
                    except Exception as e:
                        return f"❌ Error: {e}"

                async def reject_mapping(tag_name: str) -> str:
                    if not tag_name:
                        return "❌ Please enter a tag name"
                    try:
                        await client.post(f"/api/v1/mapping/{tag_name}/reject")
                        return f"✅ Mapping rejected for {tag_name}"
                    except Exception as e:
                        return f"❌ Error: {e}"

                load_candidates_btn.click(
                    fn=load_candidates,
                    inputs=[tag_selector],
                    outputs=[candidates_table],
                )

                approve_btn.click(
                    fn=approve_mapping,
                    inputs=[tag_selector, selected_irdi],
                    outputs=[action_result],
                )

                reject_btn.click(
                    fn=reject_mapping,
                    inputs=[tag_selector],
                    outputs=[action_result],
                )

            # =================================================================
            # Swarm Monitor Tab
            # =================================================================
            with gr.Tab("Swarm Monitor"):
                gr.Markdown("## Swarm Agent Status")

                with gr.Row():
                    with gr.Column():
                        swarm_status_display = gr.Markdown("Loading swarm status...")

                    with gr.Column():
                        agents_table = gr.Dataframe(
                            headers=["Agent ID", "Status", "Reliability", "Tags Processed"],
                            datatype=["str", "str", "str", "number"],
                            label="Agents",
                        )

                gr.Markdown("## Consensus Records")

                consensus_table = gr.Dataframe(
                    headers=["Tag Name", "IRDI", "Confidence", "Agreement", "Agents"],
                    datatype=["str", "str", "str", "str", "number"],
                    label="Recent Consensus",
                )

                refresh_swarm_btn = gr.Button("Refresh", variant="secondary")

                async def refresh_swarm() -> tuple[str, list[list[Any]], list[list[Any]]]:
                    try:
                        status_payload = await client.fetch("/api/v1/swarm/status")
                        agents_payload = await client.fetch("/api/v1/swarm/agents")
                        consensus_payload = await client.fetch("/api/v1/swarm/consensus")

                        status = status_payload if isinstance(status_payload, dict) else {}
                        agents = (
                            [item for item in agents_payload if isinstance(item, dict)]
                            if isinstance(agents_payload, list)
                            else []
                        )
                        consensus = (
                            [item for item in consensus_payload if isinstance(item, dict)]
                            if isinstance(consensus_payload, list)
                            else []
                        )

                        status_md = f"""## Swarm Status

| Metric | Value |
|--------|-------|
| Total Agents | {status.get('total_agents', 0)} |
| Active Agents | {status.get('active_agents', 0)} |
| Consensus In Progress | {status.get('consensus_in_progress', 0)} |
| Completed Today | {status.get('completed_today', 0)} |
"""
                        return status_md, format_agents(agents), format_consensus(consensus)
                    except Exception as e:
                        logger.error("Failed to refresh swarm", error=str(e))
                        return f"Error: {e}", [], []

                refresh_swarm_btn.click(
                    fn=refresh_swarm,
                    outputs=[swarm_status_display, agents_table, consensus_table],
                )

            # =================================================================
            # Export Tab
            # =================================================================
            with gr.Tab("AAS Export"):
                gr.Markdown("## Export Asset Administration Shell")

                with gr.Row():
                    with gr.Column():
                        aas_id_input = gr.Textbox(
                            label="AAS ID",
                            value="urn:noa:aas:tagmapping:default",
                        )
                        asset_id_input = gr.Textbox(
                            label="Asset ID",
                            value="urn:noa:asset:plant:default",
                        )
                        format_dropdown = gr.Dropdown(
                            choices=get_export_formats(),
                            value="json",
                            label="Export Format",
                        )
                        include_timestamps = gr.Checkbox(
                            label="Include Timestamps",
                            value=True,
                        )
                        include_stats = gr.Checkbox(
                            label="Include Statistics",
                            value=True,
                        )
                        export_btn = gr.Button("Export", variant="primary")

                    with gr.Column():
                        export_status = gr.Markdown("Ready to export.")
                        download_link = gr.Markdown("")

                async def do_export(
                    aas_id: str,
                    asset_id: str,
                    fmt: str,
                    timestamps: bool,
                    stats: bool,
                ) -> tuple[str, str]:
                    config = {
                        "aas_id": aas_id,
                        "asset_id": asset_id,
                        "format": fmt,
                    }

                    is_valid, error = validate_export_config(config)
                    if not is_valid:
                        return f"❌ Validation error: {error}", ""

                    try:
                        data = {
                            "aas_id": aas_id,
                            "asset_id": asset_id,
                            "include_timestamps": timestamps,
                            "include_statistics": stats,
                        }
                        await client.post(f"/api/v1/aas/export/{fmt}", data)
                        return (
                            f"✅ Export successful! Format: {fmt.upper()}",
                            f"[Download {fmt.upper()} file](/api/v1/aas/export/{fmt})",
                        )
                    except Exception as e:
                        return f"❌ Export failed: {e}", ""

                export_btn.click(
                    fn=do_export,
                    inputs=[
                        aas_id_input,
                        asset_id_input,
                        format_dropdown,
                        include_timestamps,
                        include_stats,
                    ],
                    outputs=[export_status, download_link],
                )

            # =================================================================
            # Federated Learning Tab
            # =================================================================
            with gr.Tab("Federated Learning"):
                gr.Markdown("## Federated Learning Management")

                with gr.Row():
                    with gr.Column():
                        fl_status_display = gr.Markdown("Loading status...")
                        refresh_fl_btn = gr.Button("Refresh Status", variant="secondary")

                    with gr.Column():
                        gr.Markdown("### Training Configuration")
                        num_rounds = gr.Slider(
                            minimum=1,
                            maximum=100,
                            value=10,
                            step=1,
                            label="Number of Rounds",
                        )
                        min_clients = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=2,
                            step=1,
                            label="Minimum Clients",
                        )
                        dp_enabled = gr.Checkbox(
                            label="Enable Differential Privacy",
                            value=False,
                        )
                        with gr.Row():
                            start_training_btn = gr.Button("Start Training", variant="primary")
                            stop_training_btn = gr.Button("Stop Training", variant="stop")

                training_result = gr.Markdown("")

                gr.Markdown("### Clients")
                clients_table = gr.Dataframe(
                    headers=["Client ID", "Status", "Rounds Participated", "Local Samples"],
                    datatype=["str", "str", "number", "number"],
                    label="Registered Clients",
                )

                async def refresh_fl_status() -> tuple[str, list[list[Any]]]:
                    try:
                        status_payload = await client.fetch("/api/v1/federated/status")
                        clients_payload = await client.fetch("/api/v1/federated/clients")

                        status = status_payload if isinstance(status_payload, dict) else {}
                        clients = (
                            [item for item in clients_payload if isinstance(item, dict)]
                            if isinstance(clients_payload, list)
                            else []
                        )

                        clients_data = [
                            [
                                c.get("client_id", ""),
                                c.get("status", "unknown"),
                                c.get("rounds_participated", 0),
                                c.get("local_samples", 0),
                            ]
                            for c in clients
                        ]

                        return format_training_status(status), clients_data
                    except Exception as e:
                        logger.error("Failed to refresh FL status", error=str(e))
                        return f"Error: {e}", []

                async def start_training_handler(
                    rounds: int,
                    clients_min: int,
                    dp: bool,
                ) -> str:
                    try:
                        config = {
                            "num_rounds": int(rounds),
                            "min_clients": int(clients_min),
                            "dp_enabled": dp,
                        }
                        payload = await client.post("/api/v1/federated/start", config)
                        result = payload if isinstance(payload, dict) else {}
                        return f"✅ {result.get('message', 'Training started')}"
                    except Exception as e:
                        return f"❌ Failed to start: {e}"

                async def stop_training_handler() -> str:
                    try:
                        payload = await client.post("/api/v1/federated/stop")
                        result = payload if isinstance(payload, dict) else {}
                        return f"✅ {result.get('message', 'Training stopped')}"
                    except Exception as e:
                        return f"❌ Failed to stop: {e}"

                refresh_fl_btn.click(
                    fn=refresh_fl_status,
                    outputs=[fl_status_display, clients_table],
                )

                start_training_btn.click(
                    fn=start_training_handler,
                    inputs=[num_rounds, min_clients, dp_enabled],
                    outputs=[training_result],
                )

                stop_training_btn.click(
                    fn=stop_training_handler,
                    outputs=[training_result],
                )

    return app


def main() -> None:
    """Run the Gradio dashboard."""
    import argparse

    parser = argparse.ArgumentParser(description="NOA Semantic Swarm Mapper Dashboard")
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help="API server URL",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to bind to",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public share link",
    )

    args = parser.parse_args()

    logger.info(
        "Starting Gradio dashboard",
        api_url=args.api_url,
        host=args.host,
        port=args.port,
    )

    app = create_dashboard(args.api_url)
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
