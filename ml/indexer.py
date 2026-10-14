"""Build and atomically publish the PP-1875 vector index in Qdrant."""

import logging
import time
from functools import lru_cache
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    CreateAlias,
    CreateAliasOperation,
    DeleteAlias,
    DeleteAliasOperation,
    Distance,
    PointStruct,
    VectorParams,
)

from ml.catalog_loader import item_to_text, load_catalog
from ml.config import (
    DEBUG,
    INDEX_BATCH_SIZE,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    QDRANT_STORAGE_DIR,
    QDRANT_URL,
    QDRANT_VECTOR_DIM,
)
from ml.embedder import encode_texts

logger = logging.getLogger("dora.ml.indexer")


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    """Create and reuse the configured remote or local Qdrant client."""
    if QDRANT_URL:
        return QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=10,
        )
    return QdrantClient(path=str(QDRANT_STORAGE_DIR))


def _chunks(items: list, size: int):
    for offset in range(0, len(items), size):
        yield items[offset : offset + size]


def _aliases_for(client: QdrantClient, alias_name: str) -> list[tuple[str, str]]:
    response = client.get_aliases()
    return [
        (alias.alias_name, alias.collection_name)
        for alias in response.aliases
        if alias.alias_name == alias_name
    ]


def _publish_collection(client: QdrantClient, staging: str) -> str | None:
    """Point the stable collection alias at a complete staging collection.

    Alias changes are atomic in Qdrant, so searches can continue against the old
    index until the new one has been fully embedded and uploaded.
    """
    current_aliases = _aliases_for(client, QDRANT_COLLECTION)
    if current_aliases:
        old_collection = current_aliases[0][1]
        client.update_collection_aliases(
            change_aliases_operations=[
                DeleteAliasOperation(delete_alias=DeleteAlias(alias_name=QDRANT_COLLECTION)),
                CreateAliasOperation(
                    create_alias=CreateAlias(
                        collection_name=staging,
                        alias_name=QDRANT_COLLECTION,
                    )
                ),
            ]
        )
        return old_collection

    # Migrate a collection created by older versions of the indexer. Qdrant does
    # not allow an alias to share a name with a real collection.
    if client.collection_exists(QDRANT_COLLECTION):
        client.delete_collection(QDRANT_COLLECTION)
    client.update_collection_aliases(
        change_aliases_operations=[
            CreateAliasOperation(
                create_alias=CreateAlias(
                    collection_name=staging,
                    alias_name=QDRANT_COLLECTION,
                )
            )
        ]
    )
    return None


def index_catalog() -> int:
    """Embed the real catalog in bounded batches and atomically replace the index.

    Returns the number of indexed catalog rows. Existing search results remain
    available until the staging collection has been completely uploaded.
    """
    started = time.perf_counter()
    catalog = load_catalog()
    if not catalog:
        raise ValueError("Каталог для индексации пуст")

    client = get_qdrant_client()
    staging = f"{QDRANT_COLLECTION}__staging_{uuid4().hex[:10]}"
    staging_created = False
    published = False
    old_collection: str | None = None

    try:
        # Build the first batch before creating a collection, and verify the model
        # dimension rather than silently creating an incompatible vector schema.
        first_records = catalog[:INDEX_BATCH_SIZE]
        first_vectors = encode_texts([item_to_text(item) for item in first_records])
        vector_dim = len(first_vectors[0]) if first_vectors else 0
        if vector_dim == 0:
            raise ValueError("Эмбеддер вернул пустые векторы")
        if vector_dim != QDRANT_VECTOR_DIM:
            # The collection is rebuilt from scratch, so the real model dimension
            # wins; the config value is only a hint (bge-m3 = 1024, tfidf = 512).
            logger.warning(
                "Размерность модели (%d) отличается от QDRANT_VECTOR_DIM (%d); "
                "коллекция создаётся по размерности модели",
                vector_dim,
                QDRANT_VECTOR_DIM,
            )

        client.create_collection(
            collection_name=staging,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
        )
        staging_created = True

        def upsert_batch(records: list[dict], vectors: list[list[float]]) -> None:
            points = [
                PointStruct(
                    id=item["id"],
                    vector=vector,
                    payload={
                        "name": item["name"],
                        "group": item.get("group", ""),
                        "position": item.get("position", ""),
                        "record_id": item.get("record_id", ""),
                        "pp_code": item.get("pp_code", ""),
                        "pp_link": item.get("pp_link", ""),
                        "code_role": item.get("code_role", ""),
                        "excluded_codes": item.get("excluded_codes", []),
                    },
                )
                for item, vector in zip(records, vectors, strict=True)
            ]
            client.upsert(collection_name=staging, points=points, wait=True)

        # Keep at most one embedding batch in memory at a time.
        upsert_batch(first_records, first_vectors)
        del first_vectors, first_records
        for records in _chunks(catalog[INDEX_BATCH_SIZE:], INDEX_BATCH_SIZE):
            vectors = encode_texts([item_to_text(item) for item in records])
            upsert_batch(records, vectors)
            del vectors

        old_collection = _publish_collection(client, staging)
        published = True
        if old_collection and old_collection != staging:
            try:
                client.delete_collection(old_collection)
            except Exception:
                # Publishing already succeeded; stale collection cleanup is non-fatal.
                logger.exception(
                    "Published new index but could not remove old collection %s", old_collection
                )

        elapsed = time.perf_counter() - started
        logger.info(
            "Indexed %d products into Qdrant collection %s in %.1f seconds",
            len(catalog),
            QDRANT_COLLECTION,
            elapsed,
        )
        if DEBUG:
            logger.debug("Published staging collection %s", staging)
        return len(catalog)
    except Exception:
        logger.exception("Catalog indexing failed")
        if staging_created and not published:
            try:
                client.delete_collection(staging)
            except Exception:
                logger.exception("Could not remove failed staging collection %s", staging)
        raise


if __name__ == "__main__":
    index_catalog()
