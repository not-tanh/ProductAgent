import os
import argparse
import time
import math
from multiprocessing import get_context

import polars as pl
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding

load_dotenv()

# Configuration
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
QDRANT_URL = os.getenv("QDRANT_URL")

# Batch sizes
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "8"))
UPLOAD_BATCH = int(os.getenv("UPLOAD_BATCH", "1024"))

SHARDS = int(os.getenv("SHARDS", "2"))

# Schema
TEXT_COL = "text_for_embedding"
PAYLOAD_COLS = ["asin", "title", "price", "categoryName", "stars", "reviews", "productURL", "isBestSeller", "boughtInLastMonth"]


def init_collection():
    """
    Initializes the collection with indexing disabled for faster bulk ingestion.
    """
    client = QdrantClient(url=QDRANT_URL)

    if not client.collection_exists(COLLECTION_NAME):
        print(f"Creating collection {COLLECTION_NAME}...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            shard_number=SHARDS,
            vectors_config={
                "dense": models.VectorParams(size=384, distance=models.Distance.COSINE),
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(index=models.SparseIndexParams(on_disk=False))
            },
            # Bulk ingestion optimization: Disable HNSW indexing during upload
            hnsw_config=models.HnswConfigDiff(m=0),
            optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0),
        )
    else:
        print(f"Collection {COLLECTION_NAME} exists. Disabling indexing for upload...")
        client.update_collection(
            collection_name=COLLECTION_NAME,
            hnsw_config=models.HnswConfigDiff(m=0),
            optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0),
        )


def finalize_collection():
    """
    Re-enables indexing after ingestion is complete.
    """
    client = QdrantClient(url=QDRANT_URL)
    print("Re-enabling indexing and optimization...")
    client.update_collection(
        collection_name=COLLECTION_NAME,
        hnsw_config=models.HnswConfigDiff(m=16),
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=20000),
    )
    print("Collection switched to production settings.")


def worker_task(parquet_path: str, offset: int, length: int, worker_id: int):
    """
    Worker function to process a specific slice of the Parquet file.
    """
    # 1. Initialize resources LOCALLY within the process
    # Passing clients/models across processes causes pickling issues.
    client = QdrantClient(url=QDRANT_URL, prefer_grpc=True)
    dense_model = TextEmbedding(model_name=os.getenv("DENSE_MODEL"))
    sparse_model = SparseTextEmbedding(model_name=os.getenv("SPARSE_MODEL"))

    # 2. Read only the assigned slice
    # Polars scan/slice is lazy and efficient
    print(f"[Worker {worker_id}] Processing rows {offset} to {offset + length}...")

    # We use slice(offset, length) to get the specific partition
    df = pl.read_parquet(parquet_path, columns=[TEXT_COL] + PAYLOAD_COLS).slice(offset, length)

    local_start_time = time.time()
    current_global_id = offset  # Ensure IDs are unique across all workers

    processed_count = 0

    # 3. Iterate and Embed
    for batch in df.iter_slices(n_rows=EMBED_BATCH):
        docs = batch[TEXT_COL].to_list()

        dense_vecs = list(dense_model.embed(docs, batch_size=EMBED_BATCH))
        sparse_vecs = list(sparse_model.embed(docs, batch_size=EMBED_BATCH))

        payloads = batch.select(PAYLOAD_COLS).to_dicts()

        # Generator for Qdrant points
        def gen_points():
            nonlocal current_global_id
            for j in range(len(docs)):
                sv = models.SparseVector(
                    indices=sparse_vecs[j].indices.tolist(),
                    values=sparse_vecs[j].values.tolist(),
                )

                pt = models.PointStruct(
                    id=current_global_id,
                    vector={"dense": dense_vecs[j].tolist(), "sparse": sv},
                    payload={
                        "asin": payloads[j]["asin"],
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
                current_global_id += 1
                yield pt

        # Upload
        client.upload_points(
            collection_name=COLLECTION_NAME,
            points=gen_points(),
            batch_size=UPLOAD_BATCH,
            wait=False, # Don't wait for server confirmation to speed up
        )

        processed_count += len(docs)

        if processed_count % 512 == 0:
            elapsed = time.time() - local_start_time
            print(f"[Worker {worker_id}] Uploaded {processed_count}/{length} | {processed_count/elapsed:.1f} doc/s")

    print(f"[Worker {worker_id}] Finished {length} rows.")
    return processed_count


def run_parallel_ingestion(parquet_path: str, process_num: int):
    start_time = time.time()

    # 1. Setup Collection
    init_collection()

    # 2. Analyze Data Size
    total_rows = pl.scan_parquet(parquet_path).select(pl.len()).collect().item()
    print(f"Total rows to ingest: {total_rows}")
    print(f"Spawning {process_num} worker processes...")

    # 3. Calculate Partitions
    chunk_size = math.ceil(total_rows / process_num)
    tasks = []

    for i in range(process_num):
        offset = i * chunk_size

        length = min(chunk_size, total_rows - offset)

        if length > 0:
            tasks.append((parquet_path, offset, length, i + 1))

    # 4. Run Multiprocessing
    ctx = get_context('spawn')

    with ctx.Pool(processes=process_num) as pool:
        # starmap unpacks the tuple arguments for the function
        pool.starmap(worker_task, tasks)

    # 5. Finalize
    finalize_collection()

    total_time = time.time() - start_time
    print(f"\n[SUCCESS] Ingestion Finished in {total_time:.2f} seconds.")
    print(f"Global Average Speed: {total_rows/total_time:.1f} docs/sec")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-p", "--process_num", type=int, required=True)
    args = parser.parse_args()

    run_parallel_ingestion(args.input, args.process_num)
