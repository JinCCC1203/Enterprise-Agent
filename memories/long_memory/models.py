# MemoryType + Memory ORM 模型

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pgvector.sqlalchemy import VECTOR
from sqlalchemy import DateTime, Enum as SAEnum, Index, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

MEMORY_EMBEDDING_DIM = 768    #嵌入模型的输出维度

# SQLAlchemy Base
class Base(DeclarativeBase):
    """
    SQLAlchemy的ORM基类
    """
    pass

# Memory Type
class MemoryType(str, Enum):
    """
    长期记忆类型
    """

    FACT = "fact"
    PREFERENCE = "preference"
    TASK = "task"
    SUMMARY = "summary"

# Memory ORM Model
class Memory(Base):
    """
    长期记忆数据库模型。
    一条 Memory 表示：
        某个 user 的一条长期记忆。

    例如：
        user_id:
            user_001
        memory_type:
            preference
        content:
            用户偏好使用中文回答
        embedding:
            content 对应的 768 维向量
    """
    #指定 PostgreSQL 中的表名
    __tablename__ = "long_term_memories"

    # Primary Key，它是一条 Memory 的唯一标识
    id: Mapped[int] = mapped_column(
        primary_key=True,
        autoincrement=True,
    )

    # User
    user_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        index=True,
    )

    # Memory Type
    memory_type: Mapped[MemoryType] = mapped_column(
        SAEnum(
            MemoryType,
            name="memory_type",
            native_enum=True,
        ),
        nullable=False,
        index=True,
    )

    # Memory Content
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    # Embedding
    embedding: Mapped[list[float]] = mapped_column(
        VECTOR(MEMORY_EMBEDDING_DIM),
        nullable=False,
    )

    # Metadata
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    # Indexes
    __table_args__ = (
        Index(
            "ix_long_term_memories_user_type",
            "user_id",
            "memory_type",
        ),

        # pgvector HNSW index
        # 使用 cosine distance。
        # HNSW + vector_cosine_ops。
        Index(
            "ix_long_term_memories_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={
                "embedding": "vector_cosine_ops",
            },
            postgresql_with={
                "m": 16,
                "ef_construction": 64,
            },
        ),
    )