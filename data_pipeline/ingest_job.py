import sys
import os
import argparse
import time

from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding
import polars as pl
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()
COLLECTION_NAME = os.getenv('COLLECTION_NAME')
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 8))

# TODO: optimize HNSW, quantization, etc.


def run_ingestion(parquet_path):
    client = QdrantClient(url=os.getenv('QDRANT_URL'))

    dense_model = TextEmbedding(model_name=os.getenv('DENSE_MODEL'))
    sparse_model = SparseTextEmbedding(model_name=os.getenv('SPARSE_MODEL'))

    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={"dense": models.VectorParams(size=384, distance=models.Distance.COSINE)},
            sparse_vectors_config={"sparse": models.SparseVectorParams(index=models.SparseIndexParams(on_disk=True))}
        )

    df = pl.read_parquet(parquet_path)
    print(f"Ingesting {df.height} rows...")
    start_time = time.time()

    for i in range(0, df.height, BATCH_SIZE):
        batch = df.slice(i, BATCH_SIZE)
        documents = batch["text_for_embedding"].to_list()

        dense_vectors = list(dense_model.embed(documents))
        sparse_vectors = list(sparse_model.embed(documents))

        points = []
        rows = batch.to_dicts()

        for idx, row in enumerate(rows):
            # Convert Sparse Vector from FastEmbed format to Qdrant format
            qdrant_sparse = models.SparseVector(
                indices=sparse_vectors[idx].indices.tolist(),
                values=sparse_vectors[idx].values.tolist()
            )

            payload = {
                "title": row["title"],
                "price": row["price"],
                "category": row["categoryName"],
                "rating": row["stars"],
                "reviews": row["reviews"],
                "url": row["productURL"],
                "isBestSeller": row["isBestSeller"],
                "boughtInLastMonth": row["boughtInLastMonth"]
            }

            points.append(models.PointStruct(
                id=i + idx,
                vector={
                    "dense": dense_vectors[idx].tolist(),
                    "sparse": qdrant_sparse
                },
                payload=payload
            ))

        client.upsert(COLLECTION_NAME, points)
        sys.stdout.write(f"\r[Processing] Uploaded {i + BATCH_SIZE}/{df.height}"
                         f" | Average time per record: {(time.time() - start_time) / (i + BATCH_SIZE)}")
        sys.stdout.flush()

    print("\n[SUCCESS] Ingestion Finished.")

    print("Ingestion Done. Database is ready.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", type=str, help="Path to parquet file (for ingest)", required=True)
    args = parser.parse_args()

    run_ingestion(args.input)
