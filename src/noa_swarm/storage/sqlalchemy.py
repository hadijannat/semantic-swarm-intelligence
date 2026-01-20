"""SQLAlchemy-backed repositories for persistent storage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from sqlalchemy import JSON, DateTime, Float, Integer, String, Text, delete, select
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from noa_swarm.common.logging import get_logger
from noa_swarm.common.schemas import (
    Candidate,
    ConsensusRecord,
    MappingStatus,
    QuorumType,
    TagMappingRecord,
    TagRecord,
    Vote,
)
from noa_swarm.storage.base import ConsensusRepository, MappingRepository, TagRepository

logger = get_logger(__name__)


class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""


class TagModel(Base):
    __tablename__ = "tags"

    tag_id: Mapped[str] = mapped_column(String(512), primary_key=True)
    node_id: Mapped[str] = mapped_column(String(512), nullable=False)
    browse_name: Mapped[str] = mapped_column(String(255), nullable=False)
    display_name: Mapped[str | None] = mapped_column(String(255))
    data_type: Mapped[str | None] = mapped_column(String(128))
    description: Mapped[str | None] = mapped_column(Text())
    parent_path: Mapped[list[str]] = mapped_column(JSON, default=list)
    source_server: Mapped[str] = mapped_column(String(512), nullable=False)
    engineering_unit: Mapped[str | None] = mapped_column(String(128))
    access_level: Mapped[int | None] = mapped_column(Integer)
    discovered_at: Mapped[Any] = mapped_column(DateTime(timezone=True), nullable=False)


class MappingModel(Base):
    __tablename__ = "mappings"

    tag_name: Mapped[str] = mapped_column(String(255), primary_key=True)
    tag_id: Mapped[str] = mapped_column(String(512), nullable=False)
    browse_path: Mapped[str] = mapped_column(String(512), nullable=False)
    irdi: Mapped[str | None] = mapped_column(String(255))
    preferred_name: Mapped[str | None] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    confidence: Mapped[float | None] = mapped_column(Float)
    candidates: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    created_at: Mapped[Any] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[Any] = mapped_column(DateTime(timezone=True), nullable=False)


class ConsensusModel(Base):
    __tablename__ = "consensus"

    tag_id: Mapped[str] = mapped_column(String(512), primary_key=True)
    agreed_irdi: Mapped[str] = mapped_column(String(255), nullable=False)
    consensus_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    votes: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    quorum_type: Mapped[str] = mapped_column(String(32), nullable=False)
    human_validated: Mapped[bool] = mapped_column(Integer, default=0)
    audit_trail: Mapped[list[str]] = mapped_column(JSON, default=list)
    created_at: Mapped[Any] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[Any] = mapped_column(DateTime(timezone=True), nullable=False)
    validation_notes: Mapped[str | None] = mapped_column(Text())


@dataclass(frozen=True)
class DatabaseConfig:
    """Database configuration."""

    url: str


class Database:
    """Database engine and session management."""

    def __init__(self, config: DatabaseConfig) -> None:
        self._engine: AsyncEngine = create_async_engine(config.url, future=True)
        self._sessionmaker: async_sessionmaker[AsyncSession] = async_sessionmaker(
            self._engine, expire_on_commit=False
        )

    @property
    def sessionmaker(self) -> async_sessionmaker[AsyncSession]:
        return self._sessionmaker

    async def init_models(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database schema ensured")


class SqlAlchemyTagRepository(TagRepository):
    """Tag repository backed by SQLAlchemy."""

    def __init__(self, sessionmaker: async_sessionmaker[AsyncSession]) -> None:
        self._sessionmaker = sessionmaker

    async def upsert(self, tag: TagRecord) -> TagRecord:
        async with self._sessionmaker() as session:
            existing = await session.get(TagModel, tag.tag_id)
            if existing:
                existing.node_id = tag.node_id
                existing.browse_name = tag.browse_name
                existing.display_name = tag.display_name
                existing.data_type = tag.data_type
                existing.description = tag.description
                existing.parent_path = tag.parent_path
                existing.source_server = tag.source_server
                existing.engineering_unit = tag.engineering_unit
                existing.access_level = tag.access_level
                existing.discovered_at = tag.discovered_at
            else:
                session.add(
                    TagModel(
                        tag_id=tag.tag_id,
                        node_id=tag.node_id,
                        browse_name=tag.browse_name,
                        display_name=tag.display_name,
                        data_type=tag.data_type,
                        description=tag.description,
                        parent_path=tag.parent_path,
                        source_server=tag.source_server,
                        engineering_unit=tag.engineering_unit,
                        access_level=tag.access_level,
                        discovered_at=tag.discovered_at,
                    )
                )
            await session.commit()
        return tag

    async def list(
        self,
        *,
        offset: int = 0,
        limit: int = 1000,
        filter_pattern: str | None = None,
    ) -> list[TagRecord]:
        async with self._sessionmaker() as session:
            stmt = select(TagModel).offset(offset).limit(limit)
            result = await session.execute(stmt)
            rows = result.scalars().all()

        tags = [self._to_tag(record) for record in rows]
        if filter_pattern:
            import re

            try:
                pattern = re.compile(filter_pattern, re.IGNORECASE)
                tags = [
                    tag
                    for tag in tags
                    if pattern.search(tag.browse_name)
                    or pattern.search(tag.display_name or "")
                    or pattern.search(tag.full_path)
                ]
            except re.error:
                pass
        return tags

    async def get(self, tag_id: str) -> TagRecord | None:
        async with self._sessionmaker() as session:
            record = await session.get(TagModel, tag_id)
            return self._to_tag(record) if record else None

    async def clear(self) -> None:
        async with self._sessionmaker() as session:
            await session.execute(delete(TagModel))
            await session.commit()

    @staticmethod
    def _to_tag(record: TagModel) -> TagRecord:
        return TagRecord(
            node_id=record.node_id,
            browse_name=record.browse_name,
            display_name=record.display_name,
            data_type=record.data_type,
            description=record.description,
            parent_path=record.parent_path or [],
            source_server=record.source_server,
            engineering_unit=record.engineering_unit,
            access_level=record.access_level,
            discovered_at=record.discovered_at,
        )


class SqlAlchemyMappingRepository(MappingRepository):
    """Mapping repository backed by SQLAlchemy."""

    def __init__(self, sessionmaker: async_sessionmaker[AsyncSession]) -> None:
        self._sessionmaker = sessionmaker

    async def upsert(self, mapping: TagMappingRecord) -> TagMappingRecord:
        async with self._sessionmaker() as session:
            existing = await session.get(MappingModel, mapping.tag_name)
            payload = mapping.model_dump(mode="json")
            if existing:
                existing.tag_id = mapping.tag_id
                existing.browse_path = mapping.browse_path
                existing.irdi = mapping.irdi
                existing.preferred_name = mapping.preferred_name
                existing.status = mapping.status
                existing.confidence = mapping.confidence
                existing.candidates = payload["candidates"]
                existing.created_at = mapping.created_at
                existing.updated_at = mapping.updated_at
            else:
                session.add(
                    MappingModel(
                        tag_name=mapping.tag_name,
                        tag_id=mapping.tag_id,
                        browse_path=mapping.browse_path,
                        irdi=mapping.irdi,
                        preferred_name=mapping.preferred_name,
                        status=mapping.status,
                        confidence=mapping.confidence,
                        candidates=payload["candidates"],
                        created_at=mapping.created_at,
                        updated_at=mapping.updated_at,
                    )
                )
            await session.commit()
        return mapping

    async def list(
        self,
        *,
        status: str | None = None,
        offset: int = 0,
        limit: int = 1000,
    ) -> list[TagMappingRecord]:
        async with self._sessionmaker() as session:
            stmt = select(MappingModel).offset(offset).limit(limit)
            if status:
                stmt = stmt.where(MappingModel.status == status)
            result = await session.execute(stmt)
            rows = result.scalars().all()
        return [self._to_mapping(record) for record in rows]

    async def get(self, tag_name: str) -> TagMappingRecord | None:
        async with self._sessionmaker() as session:
            record = await session.get(MappingModel, tag_name)
            return self._to_mapping(record) if record else None

    async def delete(self, tag_name: str) -> bool:
        async with self._sessionmaker() as session:
            record = await session.get(MappingModel, tag_name)
            if record is None:
                return False
            await session.delete(record)
            await session.commit()
        return True

    async def clear(self) -> None:
        async with self._sessionmaker() as session:
            await session.execute(delete(MappingModel))
            await session.commit()

    @staticmethod
    def _to_mapping(record: MappingModel) -> TagMappingRecord:
        candidates = [
            Candidate.model_validate(candidate) for candidate in (record.candidates or [])
        ]
        return TagMappingRecord(
            tag_id=record.tag_id,
            tag_name=record.tag_name,
            browse_path=record.browse_path,
            irdi=record.irdi,
            preferred_name=record.preferred_name,
            status=cast(MappingStatus, record.status),
            confidence=record.confidence,
            candidates=candidates,
            created_at=record.created_at,
            updated_at=record.updated_at,
        )


class SqlAlchemyConsensusRepository(ConsensusRepository):
    """Consensus repository backed by SQLAlchemy."""

    def __init__(self, sessionmaker: async_sessionmaker[AsyncSession]) -> None:
        self._sessionmaker = sessionmaker

    async def record(self, consensus: ConsensusRecord) -> ConsensusRecord:
        async with self._sessionmaker() as session:
            existing = await session.get(ConsensusModel, consensus.tag_id)
            payload = consensus.model_dump(mode="json")
            if existing:
                existing.agreed_irdi = consensus.agreed_irdi
                existing.consensus_confidence = consensus.consensus_confidence
                existing.votes = payload["votes"]
                existing.quorum_type = consensus.quorum_type
                existing.human_validated = consensus.human_validated
                existing.audit_trail = payload["audit_trail"]
                existing.created_at = consensus.created_at
                existing.updated_at = consensus.updated_at
                existing.validation_notes = consensus.validation_notes
            else:
                session.add(
                    ConsensusModel(
                        tag_id=consensus.tag_id,
                        agreed_irdi=consensus.agreed_irdi,
                        consensus_confidence=consensus.consensus_confidence,
                        votes=payload["votes"],
                        quorum_type=consensus.quorum_type,
                        human_validated=consensus.human_validated,
                        audit_trail=payload["audit_trail"],
                        created_at=consensus.created_at,
                        updated_at=consensus.updated_at,
                        validation_notes=consensus.validation_notes,
                    )
                )
            await session.commit()
        return consensus

    async def list(
        self,
        *,
        tag_id: str | None = None,
        offset: int = 0,
        limit: int = 1000,
    ) -> list[ConsensusRecord]:
        async with self._sessionmaker() as session:
            stmt = select(ConsensusModel).offset(offset).limit(limit)
            if tag_id:
                stmt = stmt.where(ConsensusModel.tag_id == tag_id)
            result = await session.execute(stmt)
            rows = result.scalars().all()
        return [self._to_consensus(record) for record in rows]

    async def get(self, tag_id: str) -> ConsensusRecord | None:
        async with self._sessionmaker() as session:
            record = await session.get(ConsensusModel, tag_id)
            return self._to_consensus(record) if record else None

    async def clear(self) -> None:
        async with self._sessionmaker() as session:
            await session.execute(delete(ConsensusModel))
            await session.commit()

    @staticmethod
    def _to_consensus(record: ConsensusModel) -> ConsensusRecord:
        votes = [Vote.model_validate(vote) for vote in (record.votes or [])]
        return ConsensusRecord(
            tag_id=record.tag_id,
            agreed_irdi=record.agreed_irdi,
            consensus_confidence=record.consensus_confidence,
            votes=votes,
            quorum_type=cast(QuorumType, record.quorum_type),
            human_validated=bool(record.human_validated),
            audit_trail=record.audit_trail or [],
            created_at=record.created_at,
            updated_at=record.updated_at,
            validation_notes=record.validation_notes,
        )
