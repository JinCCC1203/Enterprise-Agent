#PostgreSQL / pgvector 持久化与语义检索
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import delete, select, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from .models import (
    Base,
    Memory,
    MemoryType,
)


class MemoryStore:
    """
    Long-term Memory PostgreSQL Store。
    职责：
    1. 管理 PostgreSQL 连接
    2. 初始化 vector extension / table
    3. 保存 Memory
    4. 查询 Memory
    5. 删除 Memory
    6. 基于 pgvector 做语义相似度检索
    """

    def __init__(
        self,
        database_url: str,
        *,
        echo: bool = False,
    ) -> None:

        self.engine: AsyncEngine = create_async_engine(
            database_url,
            echo=echo,
            pool_pre_ping=True,
        )

        self.session_factory = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    # Database Initialization
    async def initialize(self) -> None:
        """
        初始化数据库。
        1. 启用 pgvector
        2. 创建 Memory 表
        """

        async with self.engine.begin() as conn:

            # pgvector extension
            await conn.execute(
                text(
                    "CREATE EXTENSION IF NOT EXISTS vector"
                )
            )

            # Create tables
            await conn.run_sync(
                Base.metadata.create_all
            )

    # Create
    async def save(
        self,
        *,
        user_id: str,
        memory_type: MemoryType,
        content: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> Memory:

        self._validate_embedding(embedding)

        now = datetime.now(timezone.utc)

        memory = Memory(
            user_id=user_id,
            memory_type=memory_type,
            content=content,
            embedding=embedding,
            metadata_=metadata or {},
            created_at=now,
            updated_at=now,
        )

        async with self.session_factory() as session:

            session.add(memory)

            await session.commit()

            await session.refresh(memory)

            return memory

    # Get By ID
    async def get(
            self,
            *,
            memory_id: int,
            user_id: str,
    ) -> Memory | None:
        result = await self.session.execute(
            select(Memory).where(
                Memory.id == memory_id,
                Memory.user_id == user_id,
            )
        )

        return result.scalar_one_or_none()

    # List User Memories
    async def list_by_user(
        self,
        *,
        user_id: str,
        memory_type: MemoryType | None = None,
        limit: int = 100,
    ) -> list[Memory]:

        if limit <= 0:
            raise ValueError(
                "limit must be greater than 0"
            )

        async with self.session_factory() as session:

            stmt = (
                select(Memory)
                .where(
                    Memory.user_id == user_id
                )
                .order_by(
                    Memory.created_at.desc()
                )
                .limit(limit)
            )

            if memory_type is not None:
                stmt = stmt.where(
                    Memory.memory_type == memory_type
                )

            result = await session.execute(stmt)

            return list(result.scalars().all())

    # Semantic Search
    async def similarity_search(
        self,
        *,
        user_id: str,
        query_embedding: list[float],
        top_k: int = 5,
        memory_type: MemoryType | None = None,
    ) -> list[tuple[Memory, float]]:

        self._validate_embedding(query_embedding)

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0"
            )

        async with self.session_factory() as session:
            # cosine distance
            # pgvector:
            # 0 -> most similar
            # larger -> less similar

            distance = (
                Memory.embedding.cosine_distance(
                    query_embedding
                )
            )

            stmt = (
                select(
                    Memory,
                    distance.label("distance"),
                )
                .where(
                    Memory.user_id == user_id
                )
                .order_by(distance)
                .limit(top_k)
            )

            if memory_type is not None:
                stmt = stmt.where(
                    Memory.memory_type == memory_type
                )

            result = await session.execute(stmt)

            rows = result.all()

            return [
                (memory, float(distance))
                for memory, distance in rows
            ]

    # Delete
    async def delete(
            self,
            *,
            memory_id: int,
            user_id: str,
    ) -> bool:
        result = await self.session.execute(
            select(Memory).where(
                Memory.id == memory_id,
                Memory.user_id == user_id,
            )
        )

        memory = result.scalar_one_or_none()

        if memory is None:
            return False

        await self.session.delete(memory)
        await self.session.commit()

        return True

    # Delete User Memories
    async def delete_by_user(
        self,
        *,
        user_id: str,
    ) -> int:

        async with self.session_factory() as session:

            result = await session.execute(
                delete(Memory).where(
                    Memory.user_id == user_id
                )
            )

            await session.commit()

            return result.rowcount

    async def update(
            self,
            *,
            memory_id: int,
            user_id: str,
            content: str,
            embedding: list[float],
            metadata: dict[str, Any],
    ) -> Memory | None:
        result = await self.session.execute(
            select(Memory).where(
                Memory.id == memory_id,
                Memory.user_id == user_id,
            )
        )

        memory = result.scalar_one_or_none()

        if memory is None:
            return None

        memory.content = content
        memory.embedding = embedding
        memory.metadata_ = metadata

        await self.session.commit()
        await self.session.refresh(memory)

        return memory

    # Close
    async def close(self) -> None:
        """
        关闭数据库连接池。
        """
        await self.engine.dispose()

    # Validation
    @staticmethod
    def _validate_embedding(
        embedding: list[float],
    ) -> None:

        if len(embedding) != 768:
            raise ValueError(
                "Embedding dimension mismatch: "
                f"expected 768, got {len(embedding)}"
            )

        if not all(
            isinstance(value, (float, int))
            for value in embedding
        ):
            raise TypeError(
                "Embedding must contain only numbers"
            )