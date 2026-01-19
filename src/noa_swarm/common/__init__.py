"""Common module for NOA Semantic Swarm Mapper.

This module provides the foundational types and utilities used throughout the system:

- **Configuration**: Pydantic settings with environment variable support
- **Logging**: Structured JSON logging with correlation IDs
- **IRDI**: International Registration Data Identifier parsing and normalization
- **Schemas**: Core data models for tags, predictions, votes, and consensus
- **Crypto**: mTLS helpers (stub for future implementation)

Example usage:
    >>> from noa_swarm.common import IRDI, TagRecord, Settings
    >>> irdi = IRDI.parse("0173-1#01-ABA234#001")
    >>> settings = Settings()
"""

from noa_swarm.common.config import (
    AASSettings,
    LoggingSettings,
    MLSettings,
    MQTTSettings,
    OPCUASettings,
    Settings,
    SwarmSettings,
    get_settings,
)
from noa_swarm.common.crypto import (
    CertificateInfo,
    CertificateManager,
    CertificateNotFoundError,
    CryptoError,
    InvalidCertificateError,
    create_mtls_context,
    generate_self_signed_cert,
    load_certificate_info,
    verify_certificate_chain,
)
from noa_swarm.common.ids import IRDI, IRDIError
from noa_swarm.common.logging import (
    LogContext,
    clear_correlation_id,
    clear_request_context,
    configure_logging,
    get_correlation_id,
    get_logger,
    get_request_context,
    log_with_context,
    set_correlation_id,
    set_request_context,
    update_request_context,
)
from noa_swarm.common.schemas import (
    AgentId,
    Candidate,
    ConsensusRecord,
    Hypothesis,
    QuorumType,
    TagId,
    TagRecord,
    Vote,
)

__all__ = [
    # Config
    "AASSettings",
    "LoggingSettings",
    "MLSettings",
    "MQTTSettings",
    "OPCUASettings",
    "Settings",
    "SwarmSettings",
    "get_settings",
    # Crypto
    "CertificateInfo",
    "CertificateManager",
    "CertificateNotFoundError",
    "CryptoError",
    "InvalidCertificateError",
    "create_mtls_context",
    "generate_self_signed_cert",
    "load_certificate_info",
    "verify_certificate_chain",
    # IDs
    "IRDI",
    "IRDIError",
    # Logging
    "LogContext",
    "clear_correlation_id",
    "clear_request_context",
    "configure_logging",
    "get_correlation_id",
    "get_logger",
    "get_request_context",
    "log_with_context",
    "set_correlation_id",
    "set_request_context",
    "update_request_context",
    # Schemas
    "AgentId",
    "Candidate",
    "ConsensusRecord",
    "Hypothesis",
    "QuorumType",
    "TagId",
    "TagRecord",
    "Vote",
]
