from __future__ import annotations

from typing import Any

import httpx
from langchain.tools import tool


# ============================================================
# Enterprise-RAG Service
# ============================================================

RAG_SERVICE_URL = (
    "http://127.0.0.1:8000/search"
)

RAG_SERVICE_TIMEOUT = 300.0


# ============================================================
# RAG Tool Description
# ============================================================

RAG_SEARCH_DESCRIPTION = """
Search the enterprise internal knowledge base for grounded
information from the organization's documents.

Use this tool when the user's question requires information
stored in the enterprise knowledge base, especially information
about the following five organizations in the RAG-Multi-Corpus
dataset:

1. Aventro Motors
   - automotive products and specifications
   - service and technical documentation
   - safety protocols
   - company or product information

2. CloudWay
   - cloud and SaaS services
   - service agreements
   - API documentation
   - pricing
   - deployment and security documentation

3. Cendara University
   - academic policies
   - course information
   - student handbooks
   - administrative procedures
   - research and university documentation

4. Velvera Technologies
   - enterprise technology products
   - product specifications
   - technical documentation
   - integration guides
   - release and support documentation

5. ZX Bank
   - banking products and services
   - account and compliance policies
   - financial documentation
   - branch and service information
   - regulatory and operational documents

Use rag_search when the answer depends on organization-specific
or document-grounded information that should be retrieved from
the enterprise knowledge base rather than inferred from general
model knowledge.

Do NOT use this tool for:

- current service health or infrastructure status
  -> use get_service_health
- querying, creating, or updating tickets/incidents
  -> use get_ticket, create_ticket, or update_ticket
- public internet research, external documentation, news,
  or current external information
  -> use web_search
- general knowledge that does not require enterprise documents

The tool performs retrieval only. It does not generate the
final answer.

The returned documents should be treated as evidence from the
enterprise knowledge base. Use the retrieved evidence to answer
the user's question and do not invent information that is not
supported by the retrieved documents.

The retrieval strategy, retrieval depth, hybrid search,
parent-child expansion, reranking, validation, correction,
retry, and compression are handled internally by the
Enterprise-RAG service. The caller does not need to specify
k or retrieval parameters.
""".strip()


# ============================================================
# RAG Search Tool
# ============================================================

@tool(
    description=RAG_SEARCH_DESCRIPTION,
)
async def rag_search(
    query: str,
) -> str:
    """
    Search the Enterprise-RAG knowledge base and return
    grounded retrieval evidence.

    Args:
        query:
            The user's knowledge question or a concise retrieval
            query suitable for the enterprise knowledge base.

    Returns:
        Formatted retrieved evidence including document content
        and source metadata.

    Raises:
        ValueError:
            If the query is empty.

        RuntimeError:
            If the RAG service returns an invalid response.

        httpx.HTTPStatusError:
            If the RAG service returns a non-2xx response.

        httpx.RequestError:
            If the RAG service cannot be reached.
    """

    # ==========================================================
    # 1. Validate Query
    # ==========================================================

    if not isinstance(
        query,
        str,
    ):
        raise ValueError(
            "query must be a string."
        )

    query = query.strip()

    if not query:
        raise ValueError(
            "query must not be empty."
        )

    # ==========================================================
    # 2. Call Enterprise-RAG Service
    # ==========================================================

    try:

        async with httpx.AsyncClient(
            timeout=RAG_SERVICE_TIMEOUT,
        ) as client:

            response = await client.post(
                RAG_SERVICE_URL,
                json={
                    "query": query,
                },
            )

            response.raise_for_status()

    except httpx.HTTPStatusError as exc:

        raise RuntimeError(
            "Enterprise-RAG service returned "
            f"HTTP {exc.response.status_code}."
        ) from exc

    except httpx.RequestError as exc:

        raise RuntimeError(
            "Enterprise-RAG service is unreachable. "
            f"URL={RAG_SERVICE_URL!r}"
        ) from exc

    # ==========================================================
    # 3. Parse JSON
    # ==========================================================

    try:

        data: Any = response.json()

    except ValueError as exc:

        raise RuntimeError(
            "Enterprise-RAG service returned invalid JSON."
        ) from exc

    # ==========================================================
    # 4. Validate Response Shape
    # ==========================================================

    if not isinstance(
        data,
        dict,
    ):
        raise RuntimeError(
            "Enterprise-RAG response must be a JSON object."
        )

    documents = data.get(
        "documents",
        [],
    )

    if not isinstance(
        documents,
        list,
    ):
        raise RuntimeError(
            "Enterprise-RAG response field 'documents' "
            "must be a list."
        )

    # ==========================================================
    # 5. No Documents
    # ==========================================================

    if not documents:

        return (
            "No relevant enterprise knowledge was retrieved "
            "for the current query."
        )

    # ==========================================================
    # 6. Format Retrieved Evidence
    # ==========================================================

    context_parts: list[str] = []

    for index, document in enumerate(
        documents,
        start=1,
    ):

        if not isinstance(
            document,
            dict,
        ):
            continue

        page_content = document.get(
            "page_content",
            "",
        )

        metadata = document.get(
            "metadata",
            {},
        )

        if not isinstance(
            metadata,
            dict,
        ):
            metadata = {}

        if not isinstance(
            page_content,
            str,
        ):
            page_content = str(
                page_content
            )

        organization = metadata.get(
            "organization",
            "unknown",
        )

        source = metadata.get(
            "source",
            "unknown",
        )

        file_name = metadata.get(
            "file_name",
            "unknown",
        )

        file_type = metadata.get(
            "file_type",
            "unknown",
        )

        rerank_score = metadata.get(
            "rerank_score",
        )

        rerank_rank = metadata.get(
            "rerank_rank",
        )

        compressed = metadata.get(
            "compressed",
            False,
        )

        # ------------------------------------------------------
        # Build Evidence Block
        # ------------------------------------------------------

        metadata_lines = [
            f"Organization: {organization}",
            f"Source: {source}",
            f"File: {file_name}",
            f"File Type: {file_type}",
        ]

        if rerank_score is not None:

            metadata_lines.append(
                f"Rerank Score: {rerank_score}"
            )

        if rerank_rank is not None:

            metadata_lines.append(
                f"Rerank Rank: {rerank_rank}"
            )

        metadata_lines.append(
            f"Compressed: {compressed}"
        )

        context_parts.append(
            f"[Document {index}]\n"
            + "\n".join(
                metadata_lines
            )
            + "\n\n"
            + page_content.strip()
        )

    # ==========================================================
    # 7. Defensive Empty Result
    # ==========================================================

    if not context_parts:

        return (
            "No valid enterprise knowledge documents "
            "were returned."
        )

    # ==========================================================
    # 8. Return Evidence
    # ==========================================================

    return "\n\n".join(
        context_parts
    )