"""Pydantic settings configuration with environment variable support.

All environment variables use the NOA_ prefix.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class MQTTSettings(BaseSettings):
    """MQTT broker connection settings."""

    model_config = SettingsConfigDict(env_prefix="NOA_MQTT_")

    host: str = Field(default="localhost", description="MQTT broker hostname")
    port: int = Field(default=1883, description="MQTT broker port")
    username: str | None = Field(default=None, description="MQTT username")
    password: SecretStr | None = Field(default=None, description="MQTT password")
    use_tls: bool = Field(default=False, description="Enable TLS for MQTT")
    ca_cert_path: Path | None = Field(default=None, description="CA certificate path")
    client_cert_path: Path | None = Field(default=None, description="Client certificate path")
    client_key_path: Path | None = Field(default=None, description="Client key path")
    keepalive: int = Field(default=60, description="MQTT keepalive interval in seconds")
    qos: Literal[0, 1, 2] = Field(default=1, description="Default MQTT QoS level")


class OPCUASettings(BaseSettings):
    """OPC UA connection settings."""

    model_config = SettingsConfigDict(env_prefix="NOA_OPCUA_")

    endpoints: list[str] = Field(
        default_factory=list,
        description="List of OPC UA server endpoints to connect to",
    )
    security_policy: Literal["None", "Basic256Sha256", "Aes128_Sha256_RsaOaep"] = Field(
        default="None",
        description="OPC UA security policy",
    )
    security_mode: Literal["None", "Sign", "SignAndEncrypt"] = Field(
        default="None",
        description="OPC UA security mode",
    )
    username: str | None = Field(default=None, description="OPC UA username")
    password: SecretStr | None = Field(default=None, description="OPC UA password")
    certificate_path: Path | None = Field(default=None, description="Client certificate path")
    private_key_path: Path | None = Field(default=None, description="Client private key path")
    application_uri: str = Field(
        default="urn:noa:swarm:client",
        description="OPC UA application URI",
    )
    session_timeout: int = Field(default=60000, description="Session timeout in milliseconds")
    subscription_interval: int = Field(
        default=1000,
        description="Subscription interval in milliseconds",
    )


class MLSettings(BaseSettings):
    """Machine learning model settings."""

    model_config = SettingsConfigDict(env_prefix="NOA_ML_")

    model_path: Path = Field(
        default=Path("models/"),
        description="Base path for ML model files",
    )
    embedding_dim: int = Field(default=768, description="Embedding dimension size")
    confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for predictions",
    )
    top_k_candidates: int = Field(
        default=5,
        ge=1,
        description="Number of top candidates to return",
    )
    batch_size: int = Field(default=32, description="Batch size for inference")
    use_gpu: bool = Field(default=True, description="Use GPU for inference if available")
    device: str = Field(default="auto", description="Device to use: auto, cpu, cuda, mps")


class SwarmSettings(BaseSettings):
    """Swarm intelligence and consensus settings."""

    model_config = SettingsConfigDict(env_prefix="NOA_SWARM_")

    agent_id: str = Field(
        default="agent-001",
        description="Unique identifier for this agent",
    )
    cluster_name: str = Field(default="noa-swarm", description="Swarm cluster name")
    gossip_port: int = Field(default=7946, description="SWIM gossip protocol port")
    gossip_interval: float = Field(
        default=1.0,
        description="Gossip interval in seconds",
    )
    quorum_size: int = Field(
        default=3,
        ge=1,
        description="Minimum number of votes for consensus",
    )
    hard_quorum_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for hard quorum",
    )
    soft_quorum_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for soft quorum",
    )
    consensus_timeout: float = Field(
        default=30.0,
        description="Consensus timeout in seconds",
    )
    reliability_decay: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Reliability score decay factor",
    )


class AASSettings(BaseSettings):
    """Asset Administration Shell settings."""

    model_config = SettingsConfigDict(env_prefix="NOA_AAS_")

    registry_url: str = Field(
        default="http://localhost:4000",
        description="AAS registry URL",
    )
    submodel_registry_url: str = Field(
        default="http://localhost:4001",
        description="Submodel registry URL",
    )
    repository_url: str = Field(
        default="http://localhost:4002",
        description="AAS repository URL",
    )
    enable_validation: bool = Field(
        default=True,
        description="Enable AAS model validation",
    )


class LoggingSettings(BaseSettings):
    """Logging configuration settings."""

    model_config = SettingsConfigDict(env_prefix="NOA_LOG_")

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Log level",
    )
    format: Literal["json", "text"] = Field(
        default="json",
        description="Log output format",
    )
    file_path: Path | None = Field(default=None, description="Log file path (optional)")
    rotation: str = Field(default="10 MB", description="Log rotation size")
    retention: str = Field(default="7 days", description="Log retention period")
    correlation_id_header: str = Field(
        default="X-Correlation-ID",
        description="HTTP header for correlation ID",
    )


class Settings(BaseSettings):
    """Main application settings combining all subsettings."""

    model_config = SettingsConfigDict(
        env_prefix="NOA_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application metadata
    app_name: str = Field(default="NOA Semantic Swarm Mapper")
    debug: bool = Field(default=False, description="Enable debug mode")
    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Deployment environment",
    )

    # Subsettings
    mqtt: MQTTSettings = Field(default_factory=MQTTSettings)
    opcua: OPCUASettings = Field(default_factory=OPCUASettings)
    ml: MLSettings = Field(default_factory=MLSettings)
    swarm: SwarmSettings = Field(default_factory=SwarmSettings)
    aas: AASSettings = Field(default_factory=AASSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)


def get_settings() -> Settings:
    """Get application settings instance.

    Returns:
        Settings: The application settings loaded from environment variables.
    """
    return Settings()
