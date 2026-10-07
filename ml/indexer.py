"""Indexes the product catalog into Qdrant."""

import time

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from ml.catalog_loader import item_to_text, load_catalog
from ml.config import (
    DEBUG,
    QDRANT_COLLECTION,
    QDRANT_STORAGE_DIR,
    QDRANT_VECTOR_DIM,
)
from ml.embedder import encode_texts


def get_qdrant_client() -> QdrantClient:
    """Creates a Qdrant client (local mode, no Docker)."""
    return QdrantClient(path=str(QDRANT_STORAGE_DIR))


def index_catalog() -> None:
    """Indexes the entire catalog into Qdrant."""
    if DEBUG:
        print("📦 Starting catalog indexing...")
        start = time.perf_counter()

    # 1. Load the catalog
    catalog = load_catalog()
    if DEBUG:
        print(f"   Loaded products: {len(catalog)}")

    # 2. Prepare texts
    texts = [item_to_text(item) for item in catalog]

    # 3. Encode all texts in a batch (faster than one-by-one)
    if DEBUG:
        print("   Encoding texts...")
    vectors = encode_texts(texts)

    # 4. Create client and collection
    client = get_qdrant_client()

    # Delete old collection if it exists
    if client.collection_exists(QDRANT_COLLECTION):
        client.delete_collection(QDRANT_COLLECTION)

    # Create new collection
    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(
            size=QDRANT_VECTOR_DIM,
            distance=Distance.COSINE,
        ),
    )

    # 5. Build points for Qdrant
    points = [
        PointStruct(
            id=item["id"],
            vector=vector,
            payload={
                "name": item["name"],
                "group": item.get("group", ""),
                "pp_code": item.get("pp_code", ""),
                "pp_link": item.get("pp_link", ""),
            },
        )
        for item, vector in zip(catalog, vectors)
    ]

    # 6. Upload to Qdrant
    client.upsert(collection_name=QDRANT_COLLECTION, points=points)

    if DEBUG:
        elapsed = time.perf_counter() - start
        print(f"✅ Indexing completed in {elapsed:.1f}s")
        print(f"   Uploaded {len(points)} products to collection '{QDRANT_COLLECTION}'")


if __name__ == "__main__":
    index_catalog()