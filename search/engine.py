import argparse
import sys
import os

from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()
COLLECTION_NAME = os.getenv('COLLECTION_NAME')
QDRANT_URL = os.getenv('QDRANT_URL')
DENSE_MODEL_NAME = os.getenv('DENSE_MODEL')
SPARSE_MODEL_NAME = os.getenv('SPARSE_MODEL')


class HybridSearchEngine:
    def __init__(self):
        self.client = QdrantClient(url=QDRANT_URL)

        print("[INFO] Loading Dense Model...")
        self.dense_model = TextEmbedding(model_name=DENSE_MODEL_NAME)

        print("[INFO] Loading Sparse Model (SPLADE)...")
        self.sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)

    def search(self,
               query_text: str,
               top_k: int = 5,
               min_price: float | None = None,
               max_price: float | None = None,
               min_rating: float | None = None,
               max_rating: float | None = None,
               min_reviews_num: int | None = None,
               max_reviews_num: int | None = None,
               is_bestseller: bool | None = None,
               min_bought_in_last_month: int | None = None,
               max_bought_in_last_month: int | None = None,
               ):
        # Prevent huge top_k number
        top_k = min(top_k, 30)

        query_dense = list(self.dense_model.embed([query_text]))[0]
        query_sparse_gen = list(self.sparse_model.embed([query_text]))[0]

        query_sparse = models.SparseVector(
            indices=query_sparse_gen.indices.tolist(),
            values=query_sparse_gen.values.tolist()
        )

        must_filters = [
            models.FieldCondition(
                key="price",
                range=models.Range(gte=min_price, lte=max_price)
            ),
            models.FieldCondition(
                key="rating",
                range=models.Range(gte=min_rating, lte=max_rating)
            ),
            models.FieldCondition(
                key="reviews",
                range=models.Range(gte=min_reviews_num, lte=max_reviews_num)
            ),
            models.FieldCondition(
                key="boughtInLastMonth",
                range=models.Range(gte=min_bought_in_last_month, lte=max_bought_in_last_month)
            ),
        ]

        # Boolean filters
        if is_bestseller is not None:
            must_filters.append(
                models.FieldCondition(
                    key="isBestSeller",
                    match=models.MatchValue(value=is_bestseller),
                )
            )

        filter_condition = models.Filter(must=must_filters)

        print(f"\n"
              f"[SEARCH] Query: '{query_text}' | Filter: Price {min_price}-{max_price} | "
              f"Top-k : {top_k}")

        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=models.Prefetch(
                query=query_sparse,
                using="sparse",
                filter=filter_condition,
                limit=top_k * 2,
            ),
            query=query_dense,
            using="dense",
            limit=top_k,
            with_payload=True
        )

        ret = []
        for hit in results.points:
            p = hit.payload
            p['search_score'] = hit.score
            ret.append(p)

        return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, help="Query text (for search)")

    args = parser.parse_args()

    engine = HybridSearchEngine()
    if not args.query:
        print("Please provide --query text")
    else:
        engine.search(query_text=args.query)
