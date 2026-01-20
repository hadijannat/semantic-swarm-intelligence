"""End-to-end integration tests for the complete workflow (lightweight)."""

from __future__ import annotations

from pathlib import Path

import pytest

from noa_swarm.aas.basyx_export import AASExporter, ExportConfig
from noa_swarm.aas.submodels import TagMappingSubmodel
from noa_swarm.common.schemas import TagRecord, Vote
from noa_swarm.ml.models.charcnn import CharacterTokenizer, CharCNN, CharCNNConfig
from noa_swarm.ml.serving import InferenceConfig, RuleBasedInferenceEngine
from noa_swarm.swarm.consensus import ConsensusConfig, ConsensusEngine


class TestDiscoveryToInferencePipeline:
    """Tests for discovery to inference pipeline."""

    @pytest.fixture
    def discovered_tags(self) -> list[TagRecord]:
        return [
            TagRecord(
                node_id="ns=2;s=TIC-101.PV",
                browse_name="TIC-101.PV",
                source_server="opc.tcp://localhost:4840",
                description="Temperature indicator controller PV",
            ),
            TagRecord(
                node_id="ns=2;s=FIC-201.SP",
                browse_name="FIC-201.SP",
                source_server="opc.tcp://localhost:4840",
                description="Flow indicator controller SP",
            ),
        ]

    def test_charcnn_forward_pass(self, discovered_tags: list[TagRecord]) -> None:
        config = CharCNNConfig(
            max_seq_length=32, embedding_dim=32, conv1_channels=32, conv2_channels=32
        )
        model = CharCNN(config)
        tokenizer = CharacterTokenizer(max_length=config.max_seq_length)

        batch = tokenizer.encode_batch([tag.browse_name for tag in discovered_tags])
        outputs = model(batch)

        assert "property_class" in outputs
        assert "signal_role" in outputs
        assert outputs["property_class"].shape[0] == len(discovered_tags)
        assert outputs["signal_role"].shape[0] == len(discovered_tags)

    @pytest.mark.asyncio
    async def test_rule_based_inference_generates_hypotheses(
        self, discovered_tags: list[TagRecord]
    ) -> None:
        engine = RuleBasedInferenceEngine(InferenceConfig(agent_id="agent-001"))
        hypotheses = await engine.infer(discovered_tags)

        assert len(hypotheses) > 0
        assert all(h.candidates for h in hypotheses)


class TestInferenceToConsensusPipeline:
    """Tests for inference to consensus pipeline."""

    def test_hypotheses_to_consensus_record(self) -> None:
        tag = TagRecord(
            node_id="ns=2;s=TIC-101.PV",
            browse_name="TIC-101.PV",
            source_server="opc.tcp://localhost:4840",
        )

        votes = [
            Vote(
                agent_id="agent-001",
                candidate_irdi="0173-1#01-AAA001#001",
                confidence=0.92,
                reliability_score=0.9,
            ),
            Vote(
                agent_id="agent-002",
                candidate_irdi="0173-1#01-AAA001#001",
                confidence=0.88,
                reliability_score=0.85,
            ),
            Vote(
                agent_id="agent-003",
                candidate_irdi="0173-1#01-AAA001#001",
                confidence=0.90,
                reliability_score=0.8,
            ),
        ]

        engine = ConsensusEngine(ConsensusConfig(min_votes=2))
        record = engine.reach_consensus(tag.tag_id, votes, calibration_factors={})

        assert record.agreed_irdi == "0173-1#01-AAA001#001"
        assert record.quorum_type in {"hard", "soft"}


class TestConsensusToExportPipeline:
    """Tests for consensus to AAS export pipeline."""

    def test_submodel_export_to_json(self, tmp_path: Path) -> None:
        from noa_swarm.aas import DiscoveredTag, create_tag_mapping_aas

        submodel = TagMappingSubmodel(submodel_id="urn:noa:submodel:tagmapping:1")
        submodel.add_tag(
            DiscoveredTag(
                tag_name="TIC-101.PV",
                browse_path="Objects/Area/TIC-101/PV",
                irdi="0173-1#01-AAA001#001",
            )
        )

        aas, sm = create_tag_mapping_aas(
            submodel=submodel,
            aas_id="urn:noa:aas:1",
            asset_id="urn:noa:asset:1",
        )

        exporter = AASExporter(ExportConfig(pretty_print=True))
        json_output = exporter.export_json(aas=aas, submodel=sm)

        assert "submodels" in json_output or "assetAdministrationShells" in json_output
