import os, sys, argparse
import time

import polars as pl
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding

load_dotenv()

COLLECTION_NAME = os.getenv("COLLECTION_NAME")
QDRANT_URL = os.getenv("QDRANT_URL")

# Tune these
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "8"))
UPLOAD_BATCH = int(os.getenv("UPLOAD_BATCH", "1024"))

SHARDS = int(os.getenv("SHARDS", "2"))

TEXT_COL = "text_for_embedding"
PAYLOAD_COLS = ["title", "price", "categoryName", "stars", "reviews", "productURL", "isBestSeller", "boughtInLastMonth"]


def run_ingestion(parquet_path: str):
    # gRPC is usually faster for large payloads; uses 6334 by default
    client = QdrantClient(url=QDRANT_URL, prefer_grpc=True)

    # If your fastembed version supports threads:
    dense_model = TextEmbedding(model_name=os.getenv("DENSE_MODEL"))
    sparse_model = SparseTextEmbedding(model_name=os.getenv("SPARSE_MODEL"))

    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            shard_number=SHARDS,
            vectors_config={
                "dense": models.VectorParams(size=384, distance=models.Distance.COSINE),
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(index=models.SparseIndexParams(on_disk=True))
            },
            # bulk ingestion mode: defer HNSW
            hnsw_config=models.HnswConfigDiff(m=0),
            # optional (fastest, but watch RAM)
            optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0),
        )

    # Read only required columns
    df = pl.read_parquet(parquet_path, columns=[TEXT_COL] + PAYLOAD_COLS)
    total = df.height
    print(f"Ingesting {total} rows...")
    start_time = time.time()
    next_id = 0

    for batch in df.iter_slices(n_rows=EMBED_BATCH):
        docs = batch[TEXT_COL].to_list()

        dense_vecs = list(dense_model.embed(docs, batch_size=EMBED_BATCH))
        sparse_vecs = list(sparse_model.embed(docs, batch_size=EMBED_BATCH))

        payloads = batch.select(PAYLOAD_COLS).to_dicts()

        def gen_points():
            nonlocal next_id
            for j in range(len(docs)):
                sv = models.SparseVector(
                    indices=sparse_vecs[j].indices.tolist(),
                    values=sparse_vecs[j].values.tolist(),
                )
                pid = next_id
                next_id += 1
                yield models.PointStruct(
                    id=pid,
                    vector={"dense": dense_vecs[j].tolist(), "sparse": sv},
                    payload={
                        "title": payloads[j]["title"],
                        "price": payloads[j]["price"],
                        "category": payloads[j]["categoryName"],
                        "rating": payloads[j]["stars"],
                        "reviews": payloads[j]["reviews"],
                        "url": payloads[j]["productURL"],
                        "isBestSeller": payloads[j]["isBestSeller"],
                        "boughtInLastMonth": payloads[j]["boughtInLastMonth"],
                    },
                )

        # automatic batching + multi-process upload; don’t block on each batch
        client.upload_points(
            collection_name=COLLECTION_NAME,
            points=gen_points(),
            batch_size=UPLOAD_BATCH,
            wait=False,
        )

        sys.stdout.write(
            f"\r[Processing] Uploaded {min(next_id, total)}/{total}"
            f" | Average time per record: {(time.time() - start_time) / min(next_id, total)}")
        sys.stdout.flush()

    print("\n[SUCCESS] Ingestion Finished.")

    # Re-enable indexing for production settings
    client.update_collection(
        collection_name=COLLECTION_NAME,
        hnsw_config=models.HnswConfigDiff(m=16),
        # if you used indexing_threshold=0 above, restore something like 20000:
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=20000),
    )
    print("Collection switched to production indexing settings.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True)
    args = parser.parse_args()
    run_ingestion(args.input)
