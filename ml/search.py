"""Semantic search over the configured Qdrant collection."""

import logging
import time
from dataclasses import dataclass

from qdrant_client.http.models import QueryRequest
from qdrant_client.models import ScoredPoint

from ml import indexer
from ml.config import DEBUG, DEFAULT_TOP_K, QDRANT_COLLECTION
from ml.embedder import encode_texts

logger = logging.getLogger("dora.ml.search")


@dataclass
class SearchResult:
    """A catalog hit and its Qdrant similarity score."""

    item_id: int | str
    name: str
    group: str
    pp_code: str
    pp_link: str
    score: float
    record_id: str = ""
    position: str = ""
    # Catalog facts stored in the point payload (BE-9). ``None`` means the
    # collection predates these payload keys; the API then falls back to the
    # catalog file on disk.
    code_role: str | None = None
    review_required: bool | None = None


def _to_results(hits: list[ScoredPoint]) -> list[SearchResult]:
    results: list[SearchResult] = []
    for hit in hits:
        payload = hit.payload or {}
        code_role = payload.get("code_role")
        review_required = (
            bool(payload["review_required"]) if "review_required" in payload else None
        )
        results.append(
            SearchResult(
                item_id=hit.id,
                name=payload.get("name", ""),
                group=payload.get("group", ""),
                pp_code=payload.get("pp_code", ""),
                pp_link=payload.get("pp_link", ""),
                record_id=payload.get("record_id", ""),
                position=payload.get("position", ""),
                score=float(hit.score),
                code_role=(code_role or None) if isinstance(code_role, str) else None,
                review_required=review_required,
            )
        )
    return results


def search(query: str, top_k: int = DEFAULT_TOP_K) -> list[SearchResult]:
    """Search one product name or description."""
    return search_many([query], top_k=top_k)[0]


def search_many(queries: list[str], top_k: int = DEFAULT_TOP_K) -> list[list[SearchResult]]:
    """Encode a group of queries together, then retrieve their nearest neighbors."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if not queries:
        return []

    started = time.perf_counter()
    vectors = encode_texts(queries)
    # Attribute lookup at call time (not an import-time ``from`` binding): the
    # client factory is lru_cached and tests replace it per fixture — a stale
    # binding here once leaked a stub client across test modules.
    client = indexer.get_qdrant_client()
    requests = [QueryRequest(query=vector, limit=top_k, with_payload=True) for vector in vectors]
    responses = client.query_batch_points(
        collection_name=QDRANT_COLLECTION,
        requests=requests,
    )
    if len(responses) != len(queries):
        raise RuntimeError("Qdrant вернул неверное число результатов для batch-запроса")
    result_groups = [_to_results(response.points) for response in responses]

    if DEBUG:
        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.debug(
            "Searched %d queries in %.1f ms; top_k=%d",
            len(queries),
            elapsed_ms,
            top_k,
        )
    return result_groups


def search_and_print(query: str, top_k: int = DEFAULT_TOP_K) -> None:
    """Search and print a small human-readable report (developer convenience)."""
    results = search(query, top_k)
    print(f"\n🔍 Query: '{query}'")
    print("-" * 60)
    for index, result in enumerate(results, 1):
        print(f"{index}. [score={result.score:.3f}] {result.name}")
        print(f"   Group: {result.group} | PP code: {result.pp_code}")


if __name__ == "__main__":
    for query in (
        "перфоратор bosch",
        "чем просверлить стену",
        "кран на кухню",
        "провода для электрики",
        "ноутбук для работы",
    ):
        search_and_print(query, top_k=3)
