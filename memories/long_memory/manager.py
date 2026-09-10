from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .embedder import MemoryEmbedder
from .extractor import MemoryCandidate, MemoryExtractor
from .models import Memory, MemoryType
from .store import MemoryStore



@dataclass(frozen=True, slots=True)
class MemorySearchResult:
    """
    Memory 语义检索结果。
    memory:
        PostgreSQL 中保存的 Memory。
    distance:
        pgvector 返回的 cosine distance。
        越小表示越相似。
    """

    memory: Memory
    distance: float


class MemoryOperation(str, Enum):
    """
    Memory 持久化操作类型。
    """
    ADD = "add"
    UPDATE = "update"
    IGNORE = "ignore"


@dataclass(frozen=True, slots=True)
class MemoryOperationResult:
    """
    一次长期记忆处理的结果。
    operation:
        ADD / UPDATE / IGNORE
    memory:
        最终对应的 Memory。
    reason:
        操作原因。
    """
    operation: MemoryOperation
    memory: Memory | None
    reason: str

# Memory Manager
class MemoryManager:
    """
    Long-term Memory Manager。
    负责协调：
        Conversation
              ↓
        MemoryExtractor
              ↓
        MemoryCandidate
              ↓
        Confidence Filter
              ↓
        MemoryEmbedder
              ↓
        Duplicate Detection
              ↓
        ADD / UPDATE / IGNORE
              ↓
        MemoryStore
              ↓
        PostgreSQL + pgvector

    查询流程：
        Query
          ↓
        MemoryEmbedder
          ↓
        query embedding
          ↓
        MemoryStore.similarity_search()
          ↓
        Similar Memories
    """

    def __init__(
        self,
        *,
        extractor: MemoryExtractor,
        embedder: MemoryEmbedder,
        store: MemoryStore,
        min_confidence: float = 0.7,
        similarity_threshold: float | None = 0.3,
        duplicate_threshold: float = 0.1,
    ) -> None:

        # Confidence
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError(
                "min_confidence must be between 0.0 and 1.0"
            )

        # Search similarity threshold
        if similarity_threshold is not None:
            if similarity_threshold < 0:
                raise ValueError(
                    "similarity_threshold must be >= 0"
                )

        # Duplicate threshold
        if duplicate_threshold < 0:
            raise ValueError(
                "duplicate_threshold must be >= 0"
            )

        self.extractor = extractor
        self.embedder = embedder
        self.store = store

        self.min_confidence = min_confidence
        self.similarity_threshold = similarity_threshold
        self.duplicate_threshold = duplicate_threshold

    # Initialization
    async def initialize(self) -> None:
        """
        初始化 Memory Store。
        """

        await self.store.initialize()

    # Remember
    async def remember(
        self,
        *,
        user_id: str,
        conversation: str,
    ) -> MemoryOperationResult:
        """
        从对话中提取长期记忆，并执行：
            ADD
            UPDATE
            IGNORE
        流程：
            conversation
                 ↓
            MemoryExtractor
                 ↓
            MemoryCandidate
                 ↓
            should_remember
                 ↓
            confidence filter
                 ↓
            embedding
                 ↓
            duplicate detection
                 ↓
            ADD / UPDATE / IGNORE
        """

        # Extract
        candidate = await self.extractor.extract(
            conversation=conversation,
        )

        # 是否值得记忆
        if not candidate.should_remember:
            return MemoryOperationResult(
                operation=MemoryOperation.IGNORE,
                memory=None,
                reason="candidate is not marked for remembering",
            )

        # Confidence Filter
        if candidate.confidence < self.min_confidence:
            return MemoryOperationResult(
                operation=MemoryOperation.IGNORE,
                memory=None,
                reason=(
                    f"candidate confidence "
                    f"{candidate.confidence:.2f} is below "
                    f"minimum threshold "
                    f"{self.min_confidence:.2f}"
                ),
            )

        # Memory Type
        # MemoryExtractor 已经保证：
        # should_remember=True 时 memory_type 不为空。

        if candidate.memory_type is None:
            raise ValueError(
                "memory_type cannot be None when "
                "should_remember=True"
            )

        # Generate Embedding
        embedding = self.embedder.embed_memory(
            candidate.content
        )

        # Find Similar Memory
        similar_memory = await self._find_similar_memory(
            user_id=user_id,
            embedding=embedding,
            memory_type=candidate.memory_type,
        )

        # ADD
        # 没有发现相似 Memory。
        if similar_memory is None:

            memory = await self.store.save(
                user_id=user_id,
                memory_type=candidate.memory_type,
                content=candidate.content,
                embedding=embedding,
                metadata=self._build_metadata(
                    candidate
                ),
            )

            return MemoryOperationResult(
                operation=MemoryOperation.ADD,
                memory=memory,
                reason=(
                    "no similar memory found; "
                    "new memory created"
                ),
            )

        # IGNORE
        # 内容完全一致，说明是重复提取。

        if (
            similar_memory.content.strip()
            == candidate.content.strip()
        ):
            return MemoryOperationResult(
                operation=MemoryOperation.IGNORE,
                memory=similar_memory,
                reason=(
                    "an identical memory already exists"
                ),
            )

        # UPDATE
        # 存在语义相似 Memory，但内容不同
        # 使用新的 Candidate 覆盖原 Memory
        updated_memory = await self.store.update(
            memory_id=similar_memory.id,
            content=candidate.content,
            embedding=embedding,
            metadata=self._build_metadata(
                candidate,
                previous_metadata=similar_memory.metadata_,
            ),
        )

        return MemoryOperationResult(
            operation=MemoryOperation.UPDATE,
            memory=updated_memory,
            reason=(
                f"similar memory found "
                f"(distance <= {self.duplicate_threshold}); "
                f"memory content was updated"
            ),
        )

    # Search
    async def search(
        self,
        *,
        user_id: str,
        query: str,
        top_k: int = 5,
        memory_type: MemoryType | None = None,
    ) -> list[MemorySearchResult]:
        """
        对长期记忆进行语义检索。
        """
        query_embedding = self.embedder.embed_query(
            query
        )

        results = await self.store.similarity_search(
            user_id=user_id,
            query_embedding=query_embedding,
            top_k=top_k,
            memory_type=memory_type,
        )

        filtered_results: list[MemorySearchResult] = []

        for memory, distance in results:

            # similarity_threshold
            # 只负责“搜索结果是否足够相关”

            if (
                self.similarity_threshold is not None
                and distance > self.similarity_threshold
            ):
                continue

            filtered_results.append(
                MemorySearchResult(
                    memory=memory,
                    distance=distance,
                )
            )

        return filtered_results

    # Get
    async def get(
        self,
        memory_id: int,
    ) -> Memory | None:
        """
        根据 Memory ID 获取记忆。
        """

        return await self.store.get(
            memory_id
        )

    # List
    async def list_user_memories(
        self,
        *,
        user_id: str,
        memory_type: MemoryType | None = None,
        limit: int = 100,
    ) -> list[Memory]:
        """
        获取用户长期记忆。
        """

        return await self.store.list_by_user(
            user_id=user_id,
            memory_type=memory_type,
            limit=limit,
        )

    # Delete
    async def delete(
        self,
        memory_id: int,
    ) -> bool:
        """
        删除指定 Memory。
        """

        return await self.store.delete(
            memory_id
        )

    async def delete_user_memories(
        self,
        *,
        user_id: str,
    ) -> int:
        """
        删除用户所有长期记忆。
        """

        return await self.store.delete_by_user(
            user_id=user_id
        )

    # Find Similar Memory
    async def _find_similar_memory(
        self,
        *,
        user_id: str,
        embedding: list[float],
        memory_type: MemoryType,
    ) -> Memory | None:
        """
        查找与 Candidate 最相似的已有 Memory。
        策略：
            top_k=1
                ↓
            distance <= duplicate_threshold
                ↓
            相似 Memory
        """

        results = await self.store.similarity_search(
            user_id=user_id,
            query_embedding=embedding,
            top_k=1,
            memory_type=memory_type,
        )

        if not results:
            return None

        memory, distance = results[0]

        if distance <= self.duplicate_threshold:
            return memory

        return None

    # Metadata
    @staticmethod
    def _build_metadata(
        candidate: MemoryCandidate,
        *,
        previous_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        构造持久化 Memory Metadata
        UPDATE 时保留已有 metadata，
        再覆盖当前 Candidate 的 metadata。
        """

        metadata = dict(
            previous_metadata or {}
        )

        metadata.update(
            candidate.metadata
        )

        metadata.update(
            {
                "confidence": candidate.confidence,
                "source": "memory_extractor",
            }
        )

        return metadata

    # Close
    async def close(self) -> None:
        """
        关闭 PostgreSQL 连接池。
        """

        await self.store.close()

