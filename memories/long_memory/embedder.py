#负责把候选记忆的 content 转成向量
from __future__ import annotations

from collections.abc import Sequence

from langchain_huggingface import HuggingFaceEmbeddings


class MemoryEmbedder:
    """
    Long-term Memory Embedding
    职责：
    1. 将 Memory content 转换为 embedding
    2. 将 query 转换为 query embedding
    3. 确保生成向量维度正确
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        expected_dim: int = 768,
    ) -> None:

        self.expected_dim = expected_dim

        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
        )

    # Embed Document / Memory
    def embed_memory(
        self,
        content: str,
    ) -> list[float]:
        """
        将一条 Memory Content 转换为 embedding。
        """

        if not content.strip():
            raise ValueError(
                "memory content cannot be empty"
            )

        embedding = self.embeddings.embed_query(
            content
        )

        self._validate_embedding(embedding)

        return embedding

    # Embed Query
    def embed_query(
        self,
        query: str,
    ) -> list[float]:
        """
        将用户查询转换为 query embedding。
        之后用于：
            MemoryStore.similarity_search()
        """

        if not query.strip():
            raise ValueError(
                "query cannot be empty"
            )

        embedding = self.embeddings.embed_query(
            query
        )

        self._validate_embedding(embedding)

        return embedding

    # Batch Memory Embedding
    def embed_memories(
        self,
        contents: Sequence[str],
    ) -> list[list[float]]:
        """
        批量生成 Memory embedding。
        """
        if not contents:
            return []

        if any(
            not content.strip()
            for content in contents
        ):
            raise ValueError(
                "memory content cannot be empty"
            )

        embeddings = (
            self.embeddings.embed_documents(
                list(contents)
            )
        )

        for embedding in embeddings:
            self._validate_embedding(embedding)

        return embeddings

    # Validation
    def _validate_embedding(
        self,
        embedding: list[float],
    ) -> None:

        if len(embedding) != self.expected_dim:
            raise ValueError(
                "Embedding dimension mismatch: "
                f"expected {self.expected_dim}, "
                f"got {len(embedding)}"
            )

        if not all(
            isinstance(value, (float, int))
            for value in embedding
        ):
            raise TypeError(
                "Embedding must contain only numbers"
            )
